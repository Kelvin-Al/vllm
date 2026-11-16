# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample request shapes and export a cohort for `vllm bench epd-plan`."""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)
_SSE_END = re.compile(rb"\r?\n\r?\n")


class TraceSampler:
    def __init__(self, path: Path, probability: float):
        if not 0 < probability <= 1:
            raise ValueError("Sample rate must be in (0, 1]")
        self.path = path
        self.probability = probability
        self.rng = random.Random()

    def write(self, event):
        data = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError:
            logger.warning("Could not append EPD workload sample", exc_info=True)
            return False
        return True

    def start(self, supported: bool):
        if self.rng.random() >= self.probability:
            return None
        sample = RequestSample(self)
        if not self.write({"sample": sample.id, "arrival": time.time()}):
            return None
        if not supported:
            sample.finish("unsupported")
            return None
        return sample


class RequestSample:
    def __init__(self, sampler):
        self.sampler = sampler
        self.id = uuid.uuid4().hex
        self.visual_tokens = None
        self.usage = None
        self.finished = False

    def record_image(self, item_meta, ec_params):
        features = item_meta.get(0, [])
        tokens = None
        if len(features) == 1:
            tokens = ec_params.get(features[0]["mm_hash"], {}).get("num_embeds")
        self.visual_tokens = tokens if type(tokens) is int and tokens > 0 else None

    def record_usage(self, body):
        usage = body.get("usage")
        if isinstance(usage, dict):
            lengths = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
            if all(type(value) is int and value > 0 for value in lengths):
                self.usage = lengths

    def finish(self, status):
        if self.finished:
            return
        self.finished = True
        row = {"sample": self.id, "status": status}
        if status == "completed":
            if self.visual_tokens is None:
                row["status"] = "missing_visual_tokens"
            elif self.usage is None:
                row["status"] = "missing_usage"
            else:
                row.update(
                    visual_tokens=self.visual_tokens,
                    input_tokens=self.usage[0],
                    output_tokens=self.usage[1],
                )
        self.sampler.write(row)


class StreamUsage:
    """Read vLLM SSE frames while preserving the client's requested stream."""

    def __init__(self, sample, injected: bool):
        self.sample = sample
        self.injected = injected
        self.buffer = b""
        self.terminal = None
        self.done = False
        self.error = False

    def _flush_terminal(self, fingerprint=None):
        if self.terminal is None:
            return
        frame, event = self.terminal
        self.terminal = None
        if fingerprint is not None:
            # vLLM stamps the fingerprint on the final usage frame when usage
            # is requested. Put it back on the finish frame for this client.
            event["system_fingerprint"] = fingerprint
            frame = b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n"
        yield frame

    def feed(self, chunk):
        self.buffer += chunk
        while end := _SSE_END.search(self.buffer):
            frame, self.buffer = self.buffer[: end.end()], self.buffer[end.end() :]
            lines = frame.splitlines()
            data = b"\n".join(
                line[5:].lstrip() for line in lines if line.startswith(b"data:")
            )
            if data == b"[DONE]":
                self.done = True
            elif data:
                try:
                    event = json.loads(data)
                except (ValueError, UnicodeError):
                    event = None
                if isinstance(event, dict):
                    self.error |= bool(event.get("error"))
                    if event.get("choices") and isinstance(event.get("usage"), dict):
                        # An upstream that forces continuous usage would also
                        # have sent the final usage to the original client.
                        self.injected = False
                    if event.get("choices") == []:
                        self.sample.record_usage(event)
                        if self.injected and event.get("usage"):
                            yield from self._flush_terminal(
                                event.get("system_fingerprint")
                            )
                            continue
                    if self.injected and any(
                        isinstance(choice, dict)
                        and choice.get("finish_reason") is not None
                        for choice in event.get("choices") or []
                    ):
                        yield from self._flush_terminal()
                        self.terminal = frame, event
                        continue
            yield from self._flush_terminal()
            yield frame

    def finish(self):
        yield from self._flush_terminal()
        if self.buffer:
            yield self.buffer


def export(path: Path, since: float | None = None):
    arrivals = {}
    outcomes = {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break  # A concurrent writer may still be finishing this line.
            row = json.loads(line)
            if "arrival" in row:
                if since is None or row["arrival"] >= since:
                    arrivals[row["sample"]] = row["arrival"]
            elif row["sample"] in arrivals:
                outcomes[row["sample"]] = row
    counts = Counter()
    encoder, inputs, outputs = [], [], []
    for sample in arrivals:
        row = outcomes.get(sample, {"status": "pending"})
        counts[row["status"]] += 1
        if row["status"] == "completed":
            encoder.append({"visual_tokens": row["visual_tokens"]})
            inputs.append(row["input_tokens"])
            outputs.append(row["output_tokens"])
    # The planner input contains only compatible completed samples. The cohort
    # status counts preserve coverage separately from its PD length arrays.
    return {
        "encoder": encoder,
        "pd": {"input_lens": inputs, "output_lens": outputs, "failed": 0},
        "sample_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float)
    parser.add_argument("--min-completed", type=int, default=100)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    if args.min_completed < 1 or (
        args.window_seconds is not None and not args.window_seconds > 0
    ):
        parser.error("Sample count and window length must be positive")
    plan_args = (args.frozen_model, args.num_gpus, args.plan_output)
    if any(value is not None for value in plan_args) and (
        not all(value is not None for value in plan_args) or args.num_gpus < 2
    ):
        parser.error(
            "Planning requires --frozen-model, --num-gpus >= 2 and --plan-output"
        )
    paths = [
        path.resolve()
        for path in (args.trace, args.output, args.frozen_model, args.plan_output)
        if path is not None
    ]
    if len(set(paths)) != len(paths):
        parser.error("Trace, model and results must use distinct paths")
    since = (
        time.time() - args.window_seconds if args.window_seconds is not None else None
    )
    workload = export(args.trace, since)
    if len(workload["encoder"]) < args.min_completed:
        parser.error(f"Too few completed samples: {workload['sample_counts']}")
    args.output.write_text(json.dumps(workload, indent=2) + "\n", encoding="utf-8")
    if args.frozen_model is not None:
        subprocess.run(
            [
                "vllm",
                "bench",
                "epd-plan",
                "--frozen-model",
                str(args.frozen_model),
                "--num-gpus",
                str(args.num_gpus),
                "--workload",
                str(args.output),
                "--output-json",
                str(args.plan_output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
