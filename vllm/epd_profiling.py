# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, single-node encoder profiling for E+PD capacity calibration."""

import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache, wraps
from typing import Any

_DIRECTORY = os.environ.get("VLLM_EPD_PROFILE_DIR")
_gpu_events: ContextVar[list | None] = ContextVar("epd_gpu_events", default=None)


def enabled() -> bool:
    return bool(_DIRECTORY)


@cache
def _file_descriptor(pid: int) -> int:
    assert _DIRECTORY is not None
    path = os.path.join(_DIRECTORY, f"epd-{pid}.jsonl")
    return os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)


def emit(stage: str, start_ns: int, end_ns: int, **fields: Any) -> None:
    if not _DIRECTORY:
        return
    event = {
        "stage": stage,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "pid": os.getpid(),
        "thread": threading.get_native_id(),
        **fields,
    }
    # One file per process. A complete record is written with one append.
    payload = (json.dumps(event, separators=(",", ":")) + "\n").encode()
    os.write(_file_descriptor(event["pid"]), payload)


@contextmanager
def span(stage: str):
    if not enabled():
        yield
        return
    start = time.monotonic_ns()
    try:
        yield
    finally:
        emit(stage, start, time.monotonic_ns())


def profile(stage: str):
    def decorate(function: Callable):
        if not enabled():
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            with span(stage):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@contextmanager
def gpu_forward(num_items: int):
    events = _gpu_events.get()
    if events is None:
        yield
        return

    import torch

    before = torch.cuda.Event(enable_timing=True)
    after = torch.cuda.Event(enable_timing=True)
    before.record()
    yield
    after.record()
    events.append((before, after, num_items))


def profile_encoder_batch(function: Callable):
    if not enabled():
        return function

    @wraps(function)
    def wrapped(self, scheduler_output, *args, **kwargs):
        # Record each physical batch once, even if it contains several requests.
        if not scheduler_output.scheduled_encoder_inputs:
            return function(self, scheduler_output, *args, **kwargs)
        events: list = []
        token = _gpu_events.set(events)
        start = time.monotonic_ns()
        try:
            result = function(self, scheduler_output, *args, **kwargs)
            end = time.monotonic_ns()
        finally:
            _gpu_events.reset(token)
        # ECExampleConnector copies embeddings to CPU before returning. A
        # connector that returns earlier needs a different collection boundary.
        complete = all(after.query() for _, after, _ in events)
        forwards = (
            [
                {"items": items, "gpu_ms": before.elapsed_time(after)}
                for before, after, items in events
            ]
            if complete and events
            else None
        )
        gpu_ms = sum(forward["gpu_ms"] for forward in forwards) if forwards else None
        emit("worker", start, end, gpu_ms=gpu_ms, gpu_forwards=forwards)
        return result

    return wrapped
