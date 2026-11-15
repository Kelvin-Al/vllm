# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize benchmark JSON and completion counters from a SuperPOD E+PD run."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def observe_metrics(path, duration):
    by_port = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        completed = waiting = 0.0
        for sample in row["metrics"]:
            if sample.startswith("vllm:request_success_total{") and any(
                f'finished_reason="{reason}"' in sample
                for reason in ("stop", "length", "repetition")
            ):
                completed += float(sample.rsplit(maxsplit=1)[-1])
            if sample.startswith("vllm:num_requests_waiting{"):
                waiting += float(sample.rsplit(maxsplit=1)[-1])
        timestamp = datetime.fromisoformat(row["timestamp"]).timestamp()
        by_port[row["port"]].append((timestamp, completed, waiting))
    first_progress = min(
        (
            t
            for samples in by_port.values()
            for t, completed, _ in samples
            if completed > samples[0][1]
        ),
        default=None,
    )
    if first_progress is None:
        return {}
    start = first_progress + 0.2 * duration
    end = first_progress + 0.8 * duration
    estimates = {}
    for port, samples in by_port.items():
        middle = [row for row in samples if start <= row[0] <= end]
        if len(middle) >= 2:
            t0, completed0, _ = middle[0]
            t1, completed1, _ = middle[-1]
            estimates[port] = (
                (completed1 - completed0) / (t1 - t0),
                mean(row[2] for row in middle),
                max(row[2] for row in middle),
            )
    return estimates


def summarize(job_dir):
    for path in sorted(job_dir.glob("*/wave-*rps.json")):
        config = json.loads((path.parent / "run-config.json").read_text())
        bench = json.loads(path.read_text())
        offered_rps = float(path.stem.split("-", 2)[-1].removesuffix("rps"))
        duration = bench["num_prompts"] / offered_rps
        counters = observe_metrics(
            path.with_name(f"{path.stem}-metrics.jsonl"),
            duration,
        )
        pd_start = config["base_port"] + 1 + config["encoders"]
        e = [values for port, values in counters.items() if port < pd_start]
        pd = [values for port, values in counters.items() if port >= pd_start]
        yield {
            "job": job_dir.name,
            "split": f"{config['encoders']}:{config['pd']}",
            "offered_rps": offered_rps,
            "completed": bench["completed"],
            "failed": bench["failed"],
            "throughput_rps": round(bench["request_throughput"], 3),
            "goodput_rps": round(bench["request_goodput"], 3),
            "p90_ttft_ms": round(bench["p90_ttft_ms"], 2),
            "p90_tpot_ms": round(bench["p90_tpot_ms"], 2),
            "e_pool_rps_central": round(sum(row[0] for row in e), 3) if e else "",
            "pd_pool_rps_central": round(sum(row[0] for row in pd), 3) if pd else "",
            "e_max_waiting_per_worker": max(row[2] for row in e) if e else "",
            "pd_max_waiting_per_worker": max(row[2] for row in pd) if pd else "",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = [row for job_dir in args.job_dirs for row in summarize(job_dir)]
    if not rows:
        parser.error("No completed benchmark waves found.")
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
