# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from examples.disaggregated.disaggregated_encoder import disagg_epd_proxy as proxy
from examples.disaggregated.disaggregated_encoder.epd_capacity import workload_trace


@pytest.mark.parametrize("mode", ["json", "stream", "stream_with_usage"])
def test_proxy_samples_shapes_without_changing_client_usage_mode(
    tmp_path, monkeypatch, mode
):
    path = tmp_path / "requests.jsonl"
    sampler = workload_trace.TraceSampler(path, 1)
    monkeypatch.setattr(proxy.app.state, "workload_sampler", sampler, raising=False)
    usage = {"prompt_tokens": 2610, "completion_tokens": 48}
    terminal = {"choices": [{"finish_reason": "stop"}]}

    async def backend(request):
        body = await request.json()
        if request.match_info["stage"] == "encode":
            image = body["messages"][0]["content"][0]
            return web.json_response(
                {
                    "ec_transfer_params": {
                        image["uuid"]: {
                            "modality": "image",
                            "item_indices": [0],
                            "num_embeds": 2500,
                            "metadata": {"image_grid_thw": [[1, 100, 100]]},
                        }
                    }
                }
            )
        if mode == "json":
            return web.json_response({"choices": [], "usage": usage})
        assert body["stream_options"]["include_usage"] is True
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: " + json.dumps(terminal).encode() + b"\n\n")
        event = json.dumps(
            {"choices": [], "usage": usage, "system_fingerprint": "fp_test"}
        ).encode()
        await response.write(b"data: " + event[:18])
        await response.write(event[18:] + b"\n\ndata: [DONE]\n\n")
        return response

    async def send():
        backend_app = web.Application()
        backend_app.router.add_post("/{stage}/v1/chat/completions", backend)
        async with TestServer(backend_app) as server:
            for field, value in {
                "e_urls": [str(server.make_url("/encode"))],
                "p_urls": [],
                "d_urls": [str(server.make_url("/decode"))],
                "d_ec_urls": [],
                "ec_consumer_dp_size": 1,
            }.items():
                monkeypatch.setattr(proxy.app.state, field, value, raising=False)
            body = {
                "model": "test",
                "stream": mode != "json",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,YWJj"},
                            },
                        ],
                    },
                ],
            }
            if mode == "stream_with_usage":
                body["stream_options"] = {"include_usage": True}
            await proxy.on_startup()
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=proxy.app),
                    base_url="http://proxy",
                ) as client:
                    return await client.post("/v1/chat/completions", json=body)
            finally:
                await proxy.on_shutdown()

    response = asyncio.run(send())
    assert response.status_code == 200
    if mode == "json":
        assert response.json()["usage"] == usage
    else:
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        expected = (
            [{**terminal, "system_fingerprint": "fp_test"}]
            if mode == "stream"
            else [
                terminal,
                {"choices": [], "usage": usage, "system_fingerprint": "fp_test"},
            ]
        )
        assert events == expected
        assert response.text.endswith("data: [DONE]\n\n")
    assert workload_trace.export(path) == {
        "encoder": [{"visual_tokens": 2500}],
        "pd": {"input_lens": [2610], "output_lens": [48], "failed": 0},
        "sample_counts": {"completed": 1},
    }


def test_export_counts_pending_cancelled_and_unsupported_samples(tmp_path):
    path = tmp_path / "trace.jsonl"
    sampler = workload_trace.TraceSampler(path, 1)
    sampler.start(True)
    sampler.start(True).finish("cancelled")
    sampler.start(False)
    sample = sampler.start(True)
    sample.visual_tokens = 100
    sample.finish("completed")
    assert workload_trace.export(path)["sample_counts"] == {
        "pending": 1,
        "cancelled": 1,
        "unsupported": 1,
        "missing_usage": 1,
    }
