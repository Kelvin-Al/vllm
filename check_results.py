"""Audit archived completion counters, corpora, and frozen predictions."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--reproducer", type=Path, required=True)
args = parser.parse_args()
root = args.results


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("summary", args.reproducer / "summarize_epd_superpod.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
groups, seeds, windows = {}, {}, {}
total = 0
for file in root.glob("*/wave-window.json"):
    label = file.parent.name
    window = read(file)
    windows[label] = window
    bench = read(file.with_name("wave.json"))
    config = read(file.with_name("run-config.json"))
    assert bench["completed"] == window["prompts"] and not bench["failed"]
    assert (config["model"], config["encoder_max_seqs"], config["pd_kv_bytes"], config["pd_chunk"]) == (
        "Qwen/Qwen2.5-VL-3B-Instruct", 1, 1073741824, 512
    )
    pairs = list(zip(bench["input_lens"], bench["output_lens"], strict=True))
    if window["seed"] in seeds:
        assert seeds[window["seed"]] == pairs
    seeds[window["seed"]] = pairs
    counters = sorted(summary.observe_metrics(file.with_name("wave-metrics.jsonl"), window["offered_seconds"]).items())
    assert len(counters) == 4 or label == "shift-shape-sample"
    e = config["encoders"]
    groups[label] = {"encoder": [v for _, v in counters[:e]], "pd": [v for _, v in counters[e:]]}
    total += bench["completed"]
assert len(groups) == 17 and total == 14016
assert set(seeds) == {6720401, 6720402, 6720411, 6720412, 6720420, 6720421, 6720422}
assert max(w["end_ns"] for name, w in windows.items() if name.startswith("calibrate")) < min(
    w["begin_ns"] for name, w in windows.items() if name.startswith("validate")
)
assert windows["shift-shape-sample"]["end_ns"] < min(
    w["begin_ns"] for name, w in windows.items() if name.startswith("shift-validate")
)
for stage, runs in read(root / "calibration.json").items():
    for run in runs:
        raw = groups[f"calibrate-{stage}-{run['offered_rps']}"]
        assert len(raw[stage]) == 1 and raw[stage][0][1] >= 1
        assert math.isclose(raw[stage][0][0], run[f"{stage}_rps"])
        assert all(v[1] < 1 for peer, rows in raw.items() if peer != stage for v in rows)
for prefix in ("", "shift-"):
    plan = root / f"{prefix}frozen-plan.json"
    report = read(root / f"{prefix}validation.json")
    assert digest(plan) == report["plan_sha256"] == (root / f"{prefix}frozen-plan.sha256").read_text().strip()
    assert len(report["runs"]) == 6
    for run in report["runs"]:
        e, pd = run["encoder_gpus"], run["pd_gpus"]
        raw = groups[f"{prefix}validate-{e}e-{pd}pd-{run['offered_rps']}"]
        for stage in ("encoder", "pd"):
            assert math.isclose(sum(v[0] for v in raw[stage]), run[f"{stage}_rps"])
shift = read(root / "shift-frozen-plan.json")
assert digest(args.reproducer / "frozen_predictor/frozen.json") == shift["frozen_model_sha256"]
assert digest(root / "shift-workload.json") == shift["workload_sha256"]
print(f"PASS: {total} successful requests, 17 runs, isolated calibration, disjoint seeds, frozen hashes, and raw counters")
