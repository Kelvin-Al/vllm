# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import subprocess
from math import isclose

from vllm.benchmarks.epd_plan import add_cli_args, main


def test_epd_plan_keeps_incumbent_on_tie_and_checks_uncertainty(tmp_path):
    output = tmp_path / "plan.json"
    result = subprocess.run(
        [
            "vllm",
            "bench",
            "epd-plan",
            "--num-gpus",
            "8",
            "--encoder-rps",
            "25",
            "--pd-rps",
            "10",
            "--encoder-rps-range",
            "24",
            "26",
            "--current-encoder-gpus",
            "3",
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert (plan["encoder_gpus"], plan["pd_gpus"]) == (3, 5)
    assert plan["optimal_encoder_gpu_counts"] == [2, 3]
    assert plan["predicted_capacity_rps"] == 50
    assert plan["optimal_rate_ratio_interval"][1] == 2.5
    assert not plan["optimal_over_input_rate_ranges"]


def test_offline_plan_uses_frozen_rate_models_and_reports_neighboring_splits(tmp_path):
    frozen = tmp_path / "profile.json"
    trace = tmp_path / "workload.json"
    output = tmp_path / "plan.json"
    durations = {"H": [0], "M": [0], "C": [0], "X": [100], "G": [80], "R": [0]}
    profile = {
        "encoder": {
            "visual_token_range": [1600, 7744],
            "serial_ms": durations,
            "saturated_ms": durations,
            "parallelism": dict.fromkeys(durations, 1),
        },
        "pd": {"kv_tokens": 100, "step_seconds": 0.02, "chunk_tokens": 2},
    }
    workload = {
        "encoder": [{"visual_tokens": 2500}],
        "pd": {"input_lens": [10], "output_lens": [5], "failed": 0},
    }
    frozen.write_text(json.dumps(profile), encoding="utf-8")
    trace.write_text(json.dumps(workload), encoding="utf-8")
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    main(
        parser.parse_args(
            [
                "--num-gpus",
                "4",
                "--frozen-model",
                str(frozen),
                "--workload",
                str(trace),
                "--output-json",
                str(output),
            ]
        )
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["encoder_gpus"] == 3
    assert result["predicted_capacity_rps"] == 30
    assert (
        result["workload_estimate"]["workload_sha256"]
        == hashlib.sha256(trace.read_bytes()).hexdigest()
    )
    lower, upper = result["workload_estimate"]["pd_rate_bounds_rps"]
    assert isclose(lower, 85 / 1.9)
    assert isclose(upper, 100 / 1.9)
    assert [r["encoder_gpus"] for r in result["candidate_allocations"]] == [2, 3]
