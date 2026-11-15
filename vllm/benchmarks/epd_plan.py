# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan GPU allocation between encoder and combined prefill/decode."""

import argparse
import hashlib
import json
from fractions import Fraction
from math import ceil, floor, isfinite
from pathlib import Path


def _positive_rate(value: str) -> Fraction:
    try:
        numeric_rate = float(value)
        if isfinite(numeric_rate) and numeric_rate > 0:
            return Fraction(value)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a positive finite request rate")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Total GPU budget for the two pools. At least two GPUs are required.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--encoder-rps",
        type=_positive_rate,
        help="Sustainable requests/s per single-GPU encoder.",
    )
    source.add_argument(
        "--frozen-model", type=Path, help="Frozen E and PD calibration."
    )
    parser.add_argument("--workload", type=Path, help="Measured request shape trace.")
    parser.add_argument(
        "--pd-rps",
        type=_positive_rate,
        help="Sustainable requests/s per single-GPU combined prefill/decode worker.",
    )
    parser.add_argument(
        "--encoder-rps-range",
        type=_positive_rate,
        nargs=2,
        metavar="RPS",
        help="Optional lower and upper bounds on the encoder rate, in that order.",
    )
    parser.add_argument(
        "--pd-rps-range",
        type=_positive_rate,
        nargs=2,
        metavar="RPS",
        help="Optional lower and upper bounds on the PD rate, in that order.",
    )
    parser.add_argument(
        "--current-encoder-gpus",
        type=int,
        help="Keep this allocation if it is tied for maximum capacity.",
    )
    parser.add_argument(
        "--output-json", type=Path, help="Write the plan to a JSON file."
    )


def plan(args: argparse.Namespace) -> dict:
    num_gpus = args.num_gpus
    encoder_rps = args.encoder_rps
    pd_rps = args.pd_rps
    current = args.current_encoder_gpus
    if num_gpus < 2:
        raise ValueError("--num-gpus must be at least 2")
    if encoder_rps is None or pd_rps is None:
        raise ValueError("Supply both --encoder-rps and --pd-rps")
    if current is not None and not 1 <= current < num_gpus:
        raise ValueError("--current-encoder-gpus must be in [1, num_gpus - 1]")

    encoder_low, encoder_high = args.encoder_rps_range or (encoder_rps, encoder_rps)
    pd_low, pd_high = args.pd_rps_range or (pd_rps, pd_rps)
    if not encoder_low <= encoder_rps <= encoder_high:
        raise ValueError("--encoder-rps-range must contain --encoder-rps")
    if not pd_low <= pd_rps <= pd_high:
        raise ValueError("--pd-rps-range must contain --pd-rps")

    def capacity(encoder_gpus: int) -> Fraction:
        return min(encoder_gpus * encoder_rps, (num_gpus - encoder_gpus) * pd_rps)

    balance = num_gpus * pd_rps / (encoder_rps + pd_rps)
    candidates = [
        count for count in {floor(balance), ceil(balance)} if 0 < count < num_gpus
    ]
    best_capacity = max(map(capacity, candidates))
    optimal = sorted(count for count in candidates if capacity(count) == best_capacity)
    encoder_gpus = current if current in optimal else optimal[0]
    pd_gpus = num_gpus - encoder_gpus

    # Endpoints are inclusive. At the low end of 0 the ratio must still be positive.
    optimal_low = Fraction(pd_gpus - 1, encoder_gpus)
    optimal_high = Fraction(pd_gpus, encoder_gpus - 1) if encoder_gpus > 1 else None
    ratio_low, ratio_high = encoder_low / pd_high, encoder_high / pd_low
    nearby = range(max(1, encoder_gpus - 1), min(num_gpus, encoder_gpus + 2))
    return {
        "encoder_gpus": encoder_gpus,
        "pd_gpus": pd_gpus,
        "predicted_capacity_rps": float(best_capacity),
        "optimal_encoder_gpu_counts": optimal,
        "rate_ratio": float(encoder_rps / pd_rps),
        "optimal_rate_ratio_interval": [
            float(optimal_low),
            float(optimal_high) if optimal_high is not None else None,
        ],
        "input_rate_ratio_interval": [float(ratio_low), float(ratio_high)],
        "optimal_over_input_rate_ranges": (
            optimal_low <= ratio_low
            and (optimal_high is None or ratio_high <= optimal_high)
        ),
        "capacity_rps_at_lower_rates": float(
            min(encoder_gpus * encoder_low, pd_gpus * pd_low)
        ),
        "candidate_allocations": [
            {
                "encoder_gpus": count,
                "pd_gpus": num_gpus - count,
                "capacity_rps": float(capacity(count)),
                "bottleneck": (
                    "encoder"
                    if count * encoder_rps < (num_gpus - count) * pd_rps
                    else "pd"
                    if count * encoder_rps > (num_gpus - count) * pd_rps
                    else "balanced"
                ),
            }
            for count in nearby
        ],
    }


def main(args: argparse.Namespace) -> None:
    try:
        estimate = None
        if args.frozen_model is not None:
            if args.workload is None or any(
                value is not None
                for value in (args.pd_rps, args.encoder_rps_range, args.pd_rps_range)
            ):
                raise ValueError("Use --frozen-model with --workload and without rates")
            from vllm.benchmarks.epd_capture import pd_rate_bounds
            from vllm.benchmarks.epd_demand import workload_rates

            frozen = args.frozen_model.read_bytes()
            model = json.loads(frozen)
            workload = args.workload.read_bytes()
            trace = json.loads(workload)
            if len(trace["encoder"]) != len(trace["pd"]["input_lens"]):
                raise ValueError(
                    "E and PD workload traces must cover the same number of requests"
                )
            pd_low, pd_high = pd_rate_bounds(trace["pd"], model["pd"])
            estimate = workload_rates(model["encoder"], trace["encoder"], pd_low)
            estimate["model_sha256"] = hashlib.sha256(frozen).hexdigest()
            estimate["workload_sha256"] = hashlib.sha256(workload).hexdigest()
            estimate["encoder_visual_token_range"] = model["encoder"][
                "visual_token_range"
            ]
            estimate["pd_rate_bounds_rps"] = [pd_low, pd_high]
            estimate["basis"] = "encoder_load_upper_bound_and_pd_packing_model"
            args.encoder_rps = _positive_rate(str(estimate["e_rate_upper_bound_rps"]))
            args.pd_rps = _positive_rate(str(estimate["pd_rate_rps"]))
            args.pd_rps_range = (args.pd_rps, _positive_rate(str(pd_high)))
        elif args.workload is not None:
            raise ValueError("--workload requires --frozen-model")
        result = plan(args)
        if estimate is not None:
            result["workload_estimate"] = estimate
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.output_json is not None:
            args.output_json.write_text(rendered + "\n", encoding="utf-8")
    except (ValueError, OverflowError, OSError) as error:
        raise SystemExit(f"Invalid E+PD plan: {error}") from None
    print(rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_args(parser)
    main(parser.parse_args())
