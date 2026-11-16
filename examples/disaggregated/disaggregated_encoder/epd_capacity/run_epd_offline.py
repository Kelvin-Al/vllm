# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a single fixed E+PD corpus to validate offline stage-rate planning."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import fmean

from summarize_epd_superpod import observe_metrics
from workload_trace import export

from vllm.benchmarks.epd_capture import read_json


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def lengths(path, count):
    result = read_json(path)
    if result["failed"] or result["completed"] != count:
        raise ValueError(f"Invalid corpus replay at {path}")
    return {name: result[name] for name in ("input_lens", "output_lens", "failed")}


def length_pairs(pd):
    return Counter(zip(pd["input_lens"], pd["output_lens"], strict=True))


def prepare(output, encoder_capture, serve):
    if encoder_capture is None:
        raise ValueError(
            "Offline validation requires the paired encoder calibration capture"
        )
    pd_capture = serve(
        "offline-pd-calibration",
        1,
        1,
        [
            {
                "name": "pd-training",
                "rate": 18,
                "seconds": 36,
                "buckets": "{(1120, 1120, 1): 1.0}",
                "input_len": 256,
                "output_len": 128,
                "range_ratio": 0.15,
            }
        ],
        profile_pd=True,
        encoder_max_seqs=1,
    )
    frozen = output / "offline-frozen.json"
    subprocess.run(
        [
            "vllm",
            "bench",
            "epd-calibrate",
            "--capture-dir",
            str(encoder_capture),
            "--min-visual-tokens",
            "1000",
            "--pd-capture-dir",
            str(pd_capture),
            "--output-json",
            str(frozen),
        ],
        check=True,
    )
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    print(f"Offline model frozen before trace collection ({digest})", flush=True)

    count = 480
    replay = {
        "buckets": (
            "{(1400, 1400, 1): 0.45, (1736, 1736, 1): 0.35, (2072, 2072, 1): 0.20}"
        ),
        "input_len": 128,
        "output_len": 48,
        "range_ratio": 0.25,
        "prompts": count,
        "seed": 8161,
    }
    trace_path = output / "sampled-requests.jsonl"
    trace = serve(
        "offline-heldout-trace",
        1,
        1,
        [{"name": "trace", "rate": 4, **replay}],
        encoder_max_seqs=1,
        workload_trace=trace_path,
    )
    observed = export(trace_path)
    encoder_samples = observed["encoder"]
    if len(encoder_samples) != count:
        raise ValueError(
            f"Expected {count} encoder shapes, captured {len(encoder_samples)}"
        )
    if {r["visual_tokens"] for r in encoder_samples} != {2500, 3844, 5476}:
        raise ValueError("The held-out shapes differ from the preassigned corpus")
    if length_pairs(observed["pd"]) != length_pairs(
        lengths(trace / "trace.json", count)
    ):
        raise ValueError("Sampled PD length pairs differ from client usage")
    workload = output / "offline-heldout.json"
    write_json(workload, {**observed, "replay": replay})
    plan_path = output / "offline-plan.json"
    subprocess.run(
        [
            "vllm",
            "bench",
            "epd-plan",
            "--num-gpus",
            "4",
            "--frozen-model",
            str(frozen),
            "--workload",
            str(workload),
            "--output-json",
            str(plan_path),
        ],
        check=True,
    )


def run_offline(output, encoder_capture, serve, previous=None):
    if previous is None:
        prepare(output, encoder_capture, serve)
    else:
        for filename in (
            "offline-frozen.json",
            "offline-heldout.json",
            "offline-plan.json",
        ):
            shutil.copyfile(previous / filename, output / filename)

    frozen = output / "offline-frozen.json"
    plan_path = output / "offline-plan.json"
    workload = output / "offline-heldout.json"
    plan = read_json(plan_path)
    data = read_json(workload)
    replay = data["replay"]
    pd_samples = data["pd"]
    count = replay["prompts"]
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    plan_digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    workload_digest = hashlib.sha256(workload.read_bytes()).hexdigest()
    estimate = plan["workload_estimate"]
    if (digest, workload_digest) != (
        estimate["model_sha256"],
        estimate["workload_sha256"],
    ):
        raise ValueError("The frozen inputs do not match the offline recommendation")
    print(
        f"Offline recommendation locked for cold-cache replay ({plan_digest})",
        flush=True,
    )

    selected = plan["encoder_gpus"]
    best = plan["predicted_capacity_rps"]
    rates = (max(25, ceil(1.55 * best)), max(31, ceil(1.9 * best)))
    validations = []
    for encoders in dict.fromkeys((selected, 1, 3, 2)):
        pd = 4 - encoders
        waves = []
        for rate in rates:
            name = f"wave-{rate}"
            capture = serve(
                f"offline-{encoders}e-{pd}pd-{rate}",
                encoders,
                pd,
                [{"name": name, "rate": rate, **replay}],
                encoder_max_seqs=1,
            )
            base = read_json(capture / "run-config.json")["base_port"]
            if length_pairs(lengths(capture / f"{name}.json", count)) != length_pairs(
                pd_samples
            ):
                raise ValueError(
                    f"The replay corpus changed for {encoders}E+{pd}PD at {rate} req/s"
                )
            window = read_json(capture / f"{name}-window.json")
            counters = observe_metrics(
                capture / f"{name}-metrics.jsonl", window["offered_seconds"]
            )
            if len(counters) != 4:
                raise ValueError(
                    f"Missing stage counters for {capture.name} at {rate} req/s"
                )
            encoder = [counters[p] for p in range(base + 1, base + encoders + 1)]
            decode = [counters[p] for p in range(base + encoders + 1, base + 5)]
            waves.append(
                {
                    "offered_rps": rate,
                    "capture": capture.name,
                    "encoder_rps": sum(x[0] for x in encoder),
                    "pd_rps": sum(x[0] for x in decode),
                    "encoder_waiting_per_worker": fmean(x[1] for x in encoder),
                    "pd_waiting_per_worker": fmean(x[1] for x in decode),
                }
            )
        validations.append(
            {
                "encoder_gpus": encoders,
                "pd_gpus": pd,
                "predicted_rps": min(
                    encoders * estimate["e_rate_upper_bound_rps"],
                    pd * estimate["pd_rate_rps"],
                ),
                "observed_pd_rps": fmean(w["pd_rps"] for w in waves),
                "waves": waves,
            }
        )
        assert digest == hashlib.sha256(frozen.read_bytes()).hexdigest()
        assert plan_digest == hashlib.sha256(plan_path.read_bytes()).hexdigest()
        write_json(
            output / "offline-validation.json",
            {
                "model_sha256": digest,
                "plan_sha256": plan_digest,
                "workload_sha256": plan["workload_estimate"]["workload_sha256"],
                "recommended_encoder_gpus": selected,
                "replay_cache_mode": "fresh_services_per_wave",
                "validations": validations,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder-capture", type=Path)
    source.add_argument("--frozen-run", type=Path)
    args = parser.parse_args()
    kv_bytes, chunk = 1 << 30, 512
    model = args.model or "Qwen/Qwen2.5-VL-3B-Instruct"
    if args.frozen_run:
        profile = read_json(args.frozen_run / "offline-frozen.json")
        model = profile["deployment"]["model"]
        kv_bytes = profile["deployment"]["kv_cache_memory_bytes"]
        chunk = profile["pd"]["chunk_tokens"]
        if args.model and args.model != model:
            parser.error("--model must match the frozen run")
    args.output.mkdir(parents=True)

    def serve(
        name,
        e,
        pd,
        waves,
        profile=False,
        profile_pd=False,
        encoder_max_seqs=128,
        workload_trace=None,
    ):
        manifest = args.output / f"{name}-waves.json"
        write_json(manifest, waves)
        capture = args.output / name
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_epd_superpod.py")),
                "--repo",
                str(args.repo),
                "--output",
                str(capture),
                "--model",
                model,
                "--encoders",
                str(e),
                "--pd",
                str(pd),
                "--waves",
                str(manifest),
                "--pd-kv-bytes",
                str(kv_bytes),
                "--pd-chunk",
                str(chunk),
                "--encoder-max-seqs",
                str(encoder_max_seqs),
                *(["--profile"] if profile else []),
                *(["--profile-pd"] if profile_pd else []),
                *(["--workload-trace", str(workload_trace)] if workload_trace else []),
            ],
            check=True,
        )
        return capture

    run_offline(args.output, args.encoder_capture, serve, args.frozen_run)


if __name__ == "__main__":
    main()
