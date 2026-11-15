# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce a native E+PD profile to InferRSI-style calibration anchors."""

import argparse
import json
import re
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import fmean

from vllm.benchmarks.epd_demand import fit_encoder


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path: Path):
    return [
        json.loads(line)
        for event_file in sorted(path.glob("epd-*.jsonl"))
        for line in event_file.read_text(encoding="utf-8").splitlines()
    ]


def window_events(events, window: dict, central: bool):
    start = window["begin_ns"]
    end = window["end_ns"]
    if central:
        first = min(
            event["start_ns"]
            for event in events
            if event["stage"] == "ingress" and start <= event["start_ns"] <= end
        )
        start = first + 0.2e9 * window["offered_seconds"]
        end = first + 0.8e9 * window["offered_seconds"]
        if end > window["end_ns"]:
            raise ValueError("Central window extends beyond captured benchmark")
    return [
        event
        for event in events
        if start <= event["end_ns"] <= end
        and (event["start_ns"] >= start or (central and event["stage"] == "request"))
    ]


def busy_ms(events, stages: set[str]) -> float:
    end = 0
    occupied = 0
    for event in sorted(events, key=lambda row: row["start_ns"]):
        if event["stage"] not in stages:
            continue
        start = max(end, event["start_ns"])
        end = max(end, event["end_ns"])
        occupied += max(0, event["end_ns"] - start)
    return occupied / 1e6


def serial_boundaries(events, requests) -> tuple[int, int]:
    """Measure control and response intervals between known boundaries."""
    control = response = 0
    stages = ("ingress", "render", "media", "processor", "worker", "output_ready")
    for request in requests:
        start, end = request["start_ns"], request["end_ns"]
        spans = {}
        for stage in stages:
            matches = [
                event
                for event in events
                if event["stage"] == stage
                and start <= event["start_ns"] <= event["end_ns"] <= end
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one serial {stage} event per request")
            spans[stage] = matches[0]
        h, render, media, processor, worker, ready = (spans[s] for s in stages)
        if not (
            start
            == h["start_ns"]
            <= h["end_ns"]
            <= render["start_ns"]
            <= media["start_ns"]
            <= media["end_ns"]
            <= processor["start_ns"]
            <= processor["end_ns"]
            <= render["end_ns"]
            <= worker["start_ns"]
            <= worker["end_ns"]
            <= ready["start_ns"]
            <= end
        ):
            raise ValueError("Serial EPD boundaries are out of order")
        control += (
            worker["start_ns"]
            - h["end_ns"]
            - (media["end_ns"] - media["start_ns"])
            - (processor["end_ns"] - processor["start_ns"])
            + ready["start_ns"]
            - worker["end_ns"]
        )
        response += end - ready["start_ns"]
    return control, response


def reduce_window(events, serial: bool):
    counts = Counter(event["stage"] for event in events)
    required = (
        "request",
        "tokens",
        "ingress",
        "processor",
        "media",
        "worker",
        "handoff",
        "scheduler",
        "response",
    )
    if missing := [stage for stage in required if not counts[stage]]:
        raise ValueError(f"Profile is missing mandatory events: {missing}")
    requests = [
        event
        for event in events
        if event["stage"] == "request" and event["status"] == 200
    ]
    workers = [event for event in events if event["stage"] == "worker"]
    if not requests or any(event["gpu_ms"] is None for event in workers):
        raise ValueError("Profile lacks successful completions or finished GPU events")
    tokens = {event["visual_tokens"] for event in events if event["stage"] == "tokens"}
    if len(tokens) != 1:
        raise ValueError(
            f"An anchor must have one measured visual-token count: {tokens}"
        )
    total = Counter()
    for event in events:
        total[event["stage"]] += event["end_ns"] - event["start_ns"]
    gpu_ms = sum(event["gpu_ms"] for event in workers)
    if serial:
        if counts["request"] != counts["tokens"] or counts["request"] != len(workers):
            raise ValueError("Serial pass must have one encode batch per image request")
        control = total["scheduler"]
        response = total["response"]
        if counts["render"] or counts["output_ready"]:
            control, response = serial_boundaries(events, requests)
        lanes = {
            "H": total["ingress"] / 1e6,
            "M": (total["media"] + total["processor"]) / 1e6,
            "C": control / 1e6,
            "X": (total["worker"] - total["handoff"]) / 1e6,
            "G": gpu_ms,
            "R": (total["handoff"] + response) / 1e6,
        }
        boundary_ms = total["request"] / 1e6 - sum(
            duration for stage, duration in lanes.items() if stage != "G"
        )
        if boundary_ms < -1e-6:
            raise ValueError("Serial spans overlap outside the worker handoff")
        boundary_ms = max(0.0, boundary_ms)
    else:
        # The synchronous filesystem handoff occupies the worker slot too.
        worker_threads = {(e["pid"], e["thread"]) for e in workers}
        shared_core = sum(
            e["end_ns"] - e["start_ns"]
            for e in events
            if e["stage"] == "scheduler" and (e["pid"], e["thread"]) in worker_threads
        )
        lanes = {
            "H": busy_ms(events, {"ingress", "response"}),
            "M": total["processor"] / 1e6,
            "media_decode": total["media"] / 1e6,
            "C": (total["scheduler"] - shared_core) / 1e6,
            "X": (total["worker"] + shared_core) / 1e6,
            "G": gpu_ms,
            "R": total["handoff"] / 1e6,
        }
        boundary_ms = 0.0
    return (
        tokens.pop(),
        {stage: value / len(requests) for stage, value in lanes.items()},
        {
            "completions": len(requests),
            "worker_batches": len(workers),
            "serial_unattributed_ms": boundary_ms / len(requests) if serial else None,
            "serial_wall_ms": total["request"] / (1e6 * len(requests))
            if serial
            else None,
        },
    )


def calibrate(root: Path, min_visual_tokens: int = 0):
    if min_visual_tokens < 0:
        raise ValueError("--min-visual-tokens must be nonnegative")
    events = read_events(root / "native-profile")
    anchors = []
    for serial_file in sorted(root.glob("serial-*-window.json")):
        saturated_file = serial_file.with_name(
            serial_file.name.replace("serial-", "saturated-", 1)
        )
        token_count, serial_ms, serial_quality = reduce_window(
            window_events(events, read_json(serial_file), central=False), serial=True
        )
        if token_count < min_visual_tokens:
            continue
        saturated_window = read_json(saturated_file)
        saturated_tokens, saturated_ms, saturated_quality = reduce_window(
            window_events(events, saturated_window, central=True), serial=False
        )
        saturated_quality["completion_rps"] = saturated_quality["completions"] / (
            0.6 * saturated_window["offered_seconds"]
        )
        saturated_quality["offered_rps"] = saturated_window["rate"]
        if token_count != saturated_tokens:
            raise ValueError("Serial and saturated token counts differ")
        anchors.append(
            {
                "visual_tokens": token_count,
                "serial_ms": serial_ms,
                "saturated_ms": saturated_ms,
                "diagnostics": {
                    "serial": serial_quality,
                    "saturated": saturated_quality,
                },
            }
        )
    if len(anchors) < 5:
        raise ValueError("Native calibration requires five or more token anchors")
    dataset = {
        "anchors": anchors,
        "parallelism": dict.fromkeys(anchors[0]["saturated_ms"], 1),
    }
    return {"encoder": fit_encoder(dataset), "measurements": dataset}


def calibrate_pd(root: Path):
    """Measure a single PD replica under the configured training replay."""
    config = read_json(root / "run-config.json")
    if config["pd"] != 1 or not config["profile_pd"] or not config["pd_chunk"]:
        raise ValueError("PD calibration needs one profiled replica and a chunk size")
    port = config["base_port"] + 1 + config["encoders"]
    profile = root / f"native-profile-pd-{port}"
    events = window_events(
        read_events(profile), read_json(root / "pd-training-window.json"), central=True
    )
    steps = [
        (event["end_ns"] - event["start_ns"]) / 1e9
        for event in events
        if event["stage"] == "engine_step"
    ]
    if not steps or min(steps) <= 0:
        raise ValueError("PD capture lacks positive engine step times")
    log = (root / f"pd-{port}.log").read_text(encoding="utf-8")
    capacities = re.findall(r"GPU KV cache size:\s+([\d,]+) tokens", log)
    if len(capacities) != 1:
        raise ValueError("Could not determine a single PD KV token capacity")
    return {
        "pd": {
            "kv_tokens": int(capacities[0].replace(",", "")),
            "chunk_tokens": config["pd_chunk"],
            "step_seconds": fmean(steps),
            "measured_steps": len(steps),
        },
        "deployment": {
            "model": config["model"],
            "kv_cache_memory_bytes": config["pd_kv_bytes"],
        },
    }


def kv_work(prompt_tokens: int, output_tokens: int, chunk_tokens: int) -> int:
    """Lifetime KV token-steps with a possibly short final prefill chunk."""
    full = ceil(prompt_tokens / chunk_tokens) - 1
    return (
        chunk_tokens * full * (full + 1) // 2
        + prompt_tokens * (1 + output_tokens)
        + output_tokens * (output_tokens + 1) // 2
    )


def pd_rate_bounds(bench: dict, model: dict) -> tuple[float, float]:
    """Estimate Nie et al.'s stable and overload limits from observed lengths."""
    requests = list(zip(bench["input_lens"], bench["output_lens"], strict=True))
    if bench["failed"] or not requests or any(min(row) <= 0 for row in requests):
        raise ValueError(
            "PD workload requires successful requests and measured token counts"
        )
    kv_tokens = model["kv_tokens"]
    if kv_tokens <= 0 or model["chunk_tokens"] <= 0 or model["step_seconds"] <= 0:
        raise ValueError("PD calibration needs positive KV, chunk, and step values")
    largest = max(prompt + output for prompt, output in requests)
    if largest >= kv_tokens:
        raise ValueError("A request exceeds the calibrated PD KV capacity")
    demands = [
        kv_work(prompt, output, model["chunk_tokens"]) for prompt, output in requests
    ]
    token_seconds = model["step_seconds"] * fmean(demands)
    return (kv_tokens - largest) / token_seconds, kv_tokens / token_seconds


def pd_rate(bench: dict, model: dict) -> float:
    return pd_rate_bounds(bench, model)[0]


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument(
        "--min-visual-tokens",
        type=int,
        default=0,
        help="Fit only anchors at this merged visual token count or higher.",
    )
    parser.add_argument("--pd-capture-dir", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)


def main(args: argparse.Namespace):
    result = calibrate(args.capture_dir, args.min_visual_tokens)
    if args.pd_capture_dir is not None:
        result.update(calibrate_pd(args.pd_capture_dir))
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
