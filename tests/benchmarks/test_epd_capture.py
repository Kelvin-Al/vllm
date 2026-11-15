# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.benchmarks.epd_capture import (
    busy_ms,
    kv_work,
    pd_rate_bounds,
    reduce_window,
    window_events,
)
from vllm.benchmarks.epd_demand import fit_encoder, workload_rates


def test_encoder_profile_accounts_for_nested_gpu_and_synchronous_handoff():
    def event(stage, start, end, **fields):
        return {
            "stage": stage,
            "start_ns": start * 10**6,
            "end_ns": end * 10**6,
            "pid": 1,
            "thread": 1,
            **fields,
        }

    events = [
        event("request", 0, 18, status=200),
        event("ingress", 0, 1),
        event("media", 1, 3),
        event("processor", 3, 5),
        event("tokens", 5, 5, visual_tokens=64),
        event("scheduler", 6, 8),
        event("worker", 8, 15, gpu_ms=4),
        event("handoff", 13, 15),
        event("response", 16, 17),
    ]
    _, serial, diagnostics = reduce_window(events, serial=True)
    assert serial == {"H": 1, "M": 4, "C": 2, "X": 5, "G": 4, "R": 3}
    assert diagnostics["serial_unattributed_ms"] == 3
    _, saturated, _ = reduce_window(events, serial=False)
    assert saturated["X"] == 9
    assert saturated["C"] == 0
    events.extend([event("render", 1, 5), event("output_ready", 16, 16)])
    _, serial, diagnostics = reduce_window(events, serial=True)
    assert serial["C"] == 4
    assert serial["R"] == 4
    assert diagnostics["serial_unattributed_ms"] == 0


def test_central_window_counts_completions_of_requests_admitted_earlier():
    events = [
        {"stage": "ingress", "start_ns": 0, "end_ns": 1},
        {"stage": "request", "start_ns": 0, "end_ns": 6_000_000_000},
    ]
    window = {"begin_ns": 0, "end_ns": 10_000_000_000, "offered_seconds": 10}
    assert window_events(events, window, central=True) == events[1:]


def test_shared_frontend_lane_counts_overlapping_intervals_once():
    events = [
        {"stage": "ingress", "start_ns": 1_000_000, "end_ns": 3_000_000},
        {"stage": "ingress", "start_ns": 2_000_000, "end_ns": 4_000_000},
        {"stage": "response", "start_ns": 5_000_000, "end_ns": 6_000_000},
    ]
    assert busy_ms(events, {"ingress", "response"}) == 4


def test_mixed_workload_averages_resource_lanes_before_the_maximum():
    serial = dict.fromkeys(("H", "M", "C", "X", "G", "R"), [0.0])
    active = {**serial, "M": [17.0, -8.0], "X": [-7.0, 8.0]}
    model = {
        "visual_token_range": [1, 2],
        "serial_ms": serial,
        "saturated_ms": active,
        "parallelism": dict.fromkeys(active, 1),
    }
    result = workload_rates(
        model, [{"visual_tokens": 1}, {"visual_tokens": 2}], pd_rate=12
    )
    assert result["e_rate_upper_bound_rps"] == 200


def test_kv_work_accounts_for_a_short_final_prefill_chunk():
    assert kv_work(prompt_tokens=5, output_tokens=2, chunk_tokens=2) == 24


def test_pd_packing_slack_reduces_the_stability_bound():
    workloads = {"input_lens": [4], "output_lens": [2], "failed": 0}
    model = {"kv_tokens": 100, "step_seconds": 0.02, "chunk_tokens": 2}
    lower, upper = pd_rate_bounds(workloads, model)
    assert lower == 94 / 0.34
    assert upper == 100 / 0.34


def test_worker_cpu_variation_does_not_change_the_forward_curvature():
    anchors = []
    for tokens in range(1, 6):
        gpu = 1 + tokens + 2 * tokens**2
        # This CPU variation would affect X's curvature in an independent fit.
        worker = gpu + 3 + tokens + (tokens - 3) ** 2
        spans = {"H": 1, "M": 1, "C": 1, "X": worker, "G": gpu, "R": 1}
        anchors.append(
            {"visual_tokens": tokens, "serial_ms": spans, "saturated_ms": spans}
        )
    model = fit_encoder({"anchors": anchors, "parallelism": dict.fromkeys(spans, 1)})
    for regime in ("serial_ms", "saturated_ms"):
        x, g = model[regime]["X"], model[regime]["G"]
        assert x[2] == g[2]
        assert abs(x[0] - g[0] - 5) < 1e-12
        assert abs(x[1] - g[1] - 1) < 1e-12
    result = workload_rates(model, [{"visual_tokens": 3}], pd_rate=12)
    assert result["encoder_bottleneck_lane"] == "X"


def test_affine_resource_demand_stays_nonnegative_in_calibrated_range():
    anchors = []
    for tokens, media in enumerate((1, 1, 1, 1, 100), start=1):
        spans = {"H": 1, "M": media, "C": 1, "X": 2, "G": 1, "R": 1}
        anchors.append(
            {"visual_tokens": tokens, "serial_ms": spans, "saturated_ms": spans}
        )
    model = fit_encoder({"anchors": anchors, "parallelism": dict.fromkeys(spans, 1)})
    outcome = workload_rates(model, [{"visual_tokens": 1}], pd_rate=12)
    assert outcome["mean_normalized_lane_ms"]["M"] == 0
