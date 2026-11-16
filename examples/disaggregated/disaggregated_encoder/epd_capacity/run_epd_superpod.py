# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run single-node E+PD load tests with an existing Linux vLLM environment."""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def metrics(port):
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
        relevant = (
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:request_success_total",
            "vllm:mm_cache_queries_total",
            "vllm:mm_cache_hits_total",
            "vllm:kv_cache_usage_perc",
        )
        return [
            line
            for line in response.read().decode().splitlines()
            if line.startswith(relevant)
        ]


def wait_ready(workers, timeout):
    pending = dict(workers)
    deadline = time.monotonic() + timeout
    while pending and time.monotonic() < deadline:
        for port, process in list(pending.items()):
            if process.poll() is not None:
                raise RuntimeError(f"Service on port {port} exited. See its log.")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    pending.pop(port)
            except (URLError, TimeoutError):
                pass
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError(f"Services failed to become ready on ports {list(pending)}")


def run(args):
    proxy = args.repo / (
        "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
    )
    if not proxy.is_file() or not shutil.which("vllm"):
        raise RuntimeError("Provide a vLLM checkout and activate the vLLM environment.")
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if gpu_ids == [""]:
        gpu_ids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).splitlines()
    num_workers = args.encoders + args.pd
    if num_workers > len(set(gpu_ids)):
        raise ValueError(f"Need {num_workers} visible GPUs. Found {gpu_ids}.")
    args.output.mkdir(parents=True)
    cache = args.output / "ec-cache"
    cache.mkdir()
    (args.output / "run-config.json").write_text(
        json.dumps(
            {**vars(args), "gpu_ids": gpu_ids[:num_workers]}, default=str, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    ports = list(range(args.base_port + 1, args.base_port + num_workers + 1))
    processes = []

    def start(name, command, env=None):
        print(f"Starting {name}", flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        processes.append(child)
        return child

    try:
        workers = {}
        for index, port in enumerate(ports):
            encoder = index < args.encoders
            ec_config = {
                "ec_connector": "ECExampleConnector",
                "ec_role": "ec_producer" if encoder else "ec_consumer",
                "ec_connector_extra_config": {"shared_storage_path": str(cache)},
            }
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids[index]}
            if encoder:
                env["VLLM_MEDIA_LOADING_THREAD_COUNT"] = "1"
            if encoder and args.profile:
                profile_dir = args.output / "native-profile"
                profile_dir.mkdir(exist_ok=True)
                env["VLLM_EPD_PROFILE_DIR"] = str(profile_dir)
            if not encoder and args.profile_pd:
                profile_dir = args.output / f"native-profile-pd-{port}"
                profile_dir.mkdir()
                env["VLLM_EPD_PROFILE_DIR"] = str(profile_dir)
            workers[port] = start(
                ("encoder" if encoder else "pd") + f"-{port}",
                [
                    "vllm",
                    "serve",
                    args.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--enforce-eager",
                    "--disable-prefix-caching",
                    "--max-model-len",
                    "8192",
                    "--max-num-seqs",
                    str(args.encoder_max_seqs if encoder else 128),
                    "--enable-request-id-headers",
                    "--ec-transfer-config",
                    json.dumps(ec_config),
                    *(
                        ["--mm-encoder-only", "--max-num-batched-tokens", "114688"]
                        if encoder
                        else ["--gpu-memory-utilization", "0.80"]
                    ),
                    *(
                        [
                            "--kv-cache-memory-bytes",
                            str(args.pd_kv_bytes),
                            "--max-num-batched-tokens",
                            str(args.pd_chunk),
                            "--no-async-scheduling",
                        ]
                        if not encoder and args.pd_kv_bytes
                        else []
                    ),
                    *(["--renderer-num-workers", "1"] if encoder else []),
                ],
                env,
            )
        wait_ready(workers, args.startup_timeout)
        proxy_process = start(
            "proxy",
            [
                sys.executable,
                str(proxy),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.base_port),
                "--encode-servers-urls",
                ",".join(f"http://127.0.0.1:{port}" for port in ports[: args.encoders]),
                "--decode-servers-urls",
                ",".join(f"http://127.0.0.1:{port}" for port in ports[args.encoders :]),
                *(
                    [
                        "--workload-trace",
                        str(args.workload_trace.resolve()),
                        "--workload-sample-rate",
                        "1",
                    ]
                    if args.workload_trace
                    else []
                ),
            ],
        )
        wait_ready({args.base_port: proxy_process}, args.startup_timeout)
        waves = (
            json.loads(args.waves.read_text())
            if args.waves
            else [{"rate": rate} for rate in args.rates]
        )
        for index, wave in enumerate(waves):
            rate = wave["rate"]
            tag = wave.get("name", f"wave-{index:02d}-{rate:g}rps")
            seconds = wave.get("seconds", args.seconds)
            serial = wave.get("serial", False)
            prompts = wave.get("prompts", max(8, round(rate * seconds)))
            begin_ns = time.monotonic_ns()
            bench = start(
                tag,
                [
                    "vllm",
                    "bench",
                    "serve",
                    "--model",
                    args.model,
                    "--backend",
                    "openai-chat",
                    "--endpoint",
                    "/v1/chat/completions",
                    "--port",
                    str(args.base_port),
                    "--dataset-name",
                    "random-mm",
                    "--seed",
                    str(wave.get("seed", args.seed + index)),
                    "--num-prompts",
                    str(prompts),
                    "--random-input-len",
                    str(wave.get("input_len", 128)),
                    "--random-output-len",
                    str(wave.get("output_len", 128)),
                    "--random-mm-base-items-per-request",
                    "1",
                    "--random-mm-num-mm-items-range-ratio",
                    "0",
                    "--random-mm-limit-mm-per-prompt",
                    '{"image": 1, "video": 0}',
                    "--random-mm-bucket-config",
                    wave.get("buckets", "{(448, 448, 1): 1.0}"),
                    "--random-range-ratio",
                    str(wave.get("range_ratio", 0)),
                    "--ignore-eos",
                    "--request-rate",
                    "inf" if serial else str(rate),
                    *(["--max-concurrency", "1"] if serial else []),
                    "--disable-tqdm",
                    "--metric-percentiles",
                    "50,90,99",
                    "--percentile-metrics",
                    "ttft,tpot,e2el",
                    "--goodput",
                    "ttft:2000",
                    "tpot:50",
                    "--save-result",
                    "--save-detailed",
                    "--result-dir",
                    str(args.output),
                    "--result-filename",
                    f"{tag}.json",
                ],
            )
            deadline = time.monotonic() + seconds * 5 + 150
            with (args.output / f"{tag}-metrics.jsonl").open("w") as log:
                while bench.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{tag} exceeded its time limit")
                    for port, worker in workers.items():
                        if worker.poll() is not None:
                            raise RuntimeError(f"Worker {port} exited during {tag}")
                        try:
                            snapshot = metrics(port)
                        except (URLError, TimeoutError):
                            continue
                        log.write(
                            json.dumps(
                                {
                                    "timestamp": datetime.now(UTC).isoformat(),
                                    "monotonic_ns": time.monotonic_ns(),
                                    "port": port,
                                    "metrics": snapshot,
                                }
                            )
                            + "\n"
                        )
                    log.flush()
                    time.sleep(2)
            if bench.returncode != 0:
                raise RuntimeError(f"{tag} failed. See {tag}.log.")
            (args.output / f"{tag}-window.json").write_text(
                json.dumps(
                    {
                        **wave,
                        "begin_ns": begin_ns,
                        "end_ns": time.monotonic_ns(),
                        "offered_seconds": prompts / rate,
                    },
                ),
                encoding="utf-8",
            )
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 15
        for process in reversed(processes):
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        shutil.rmtree(cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoders", type=int, required=True)
    parser.add_argument("--pd", type=int, required=True)
    workload = parser.add_mutually_exclusive_group(required=True)
    workload.add_argument("--rates", type=float, nargs="+")
    workload.add_argument("--waves", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-pd", action="store_true")
    parser.add_argument("--workload-trace", type=Path)
    parser.add_argument("--encoder-max-seqs", type=int, default=128)
    parser.add_argument("--pd-kv-bytes", type=int)
    parser.add_argument("--pd-chunk", type=int, default=512)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--seed", type=int, default=117)
    parser.add_argument("--base-port", type=int, default=19530)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()
    if min(
        args.encoders,
        args.pd,
        args.seconds,
        args.startup_timeout,
        args.encoder_max_seqs,
    ) <= 0 or any(not math.isfinite(rate) or rate <= 0 for rate in (args.rates or [])):
        parser.error("GPU counts, seconds, and rates must be positive.")
    if not 1024 <= args.base_port <= 65535 - args.encoders - args.pd:
        parser.error("Choose a base port with room for all workers.")
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run(args)
