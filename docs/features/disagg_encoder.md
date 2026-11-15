# Disaggregated Encoder

A **disaggregated encoder** runs the vision-encoder stage of a multimodal LLM in a process that is separate from the pre-fill / decoder stage. Deploying these two stages in independent vLLM instances brings three practical benefits:

1. **Independent, fine-grained scaling**  
2. **Lower time-to-first-token (TTFT)**  
3. **Cross-process reuse and caching of encoder outputs**

Design doc: <https://docs.google.com/document/d/1aed8KtC6XkXtdoV87pWT0a8OJlZ-CpnuLLzmR8l9BAE>

---

## 1  Motivation

### 1. Independent, fine-grained scaling

* Vision encoders are lightweight, while language models are orders of magnitude larger.  
* The language model can be parallelised without affecting the encoder fleet.  
* Encoder nodes can be added or removed independently.

### 2. Lower time-to-first-token (TTFT)

* Language-only requests bypass the vision encoder entirely.  
* Encoder output is injected only at required attention layers, shortening the pre-fill critical path.

### 3. Cross-process reuse and caching

* In-process encoders confine reuse to a single worker.  
* A remote, shared cache lets any worker retrieve existing embeddings, eliminating redundant computation.

---

## 2  Usage Example

### ExampleConnector

The following scripts demonstrate the workflow with **ExampleConnector**:

1 Encoder instance + 1 PD instance:
`examples/disaggregated/disaggregated_encoder/disagg_1e1pd_example.sh`

1 Encoder instance + 1 Prefill instance + 1 Decode instance:
`examples/disaggregated/disaggregated_encoder/disagg_1e1p1d_example.sh`

### ECMooncakeConnector

**ECMooncakeConnector** transfers encoder outputs using the Mooncake TransferEngine.
See the [ECMooncakeConnector Usage Guide](mooncake_ec_connector_usage.md) for
producer/consumer `--ec-transfer-config`, proxy `--ec-consumer-zmq-addrs`, and
transport options. The [Mooncake integration example](../../tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh)
is a complete 1 Encoder instance + 1 PD instance setup.

With vLLM and Mooncake installed, run the example from the repository root:

```bash
GPU_E=0 GPU_PD=1 MOONCAKE_EC_PROTOCOL=tcp \
    bash tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh
```

The script uses `Qwen/Qwen2.5-VL-3B-Instruct` by default, runs a single-GPU baseline,
and compares the disaggregated outputs against it. Set `MODEL` to use another model
or `MOONCAKE_EC_PROTOCOL=rdma` to use RDMA on supported hardware. See the script for
additional configuration options.

### Plan the encoder and PD GPU allocation

`vllm bench epd-plan` estimates a capacity-maximizing split for E+PD on a
fixed GPU budget. It assumes single-GPU instances, proportional capacity
within each pool, and no separate bottleneck in routing or cache transfer.
Use stage rates measured in end-user requests per second per instance,
for the same workload and serving configuration.

```bash
vllm bench epd-plan --num-gpus 8 --encoder-rps 50 --pd-rps 12 \
    --encoder-rps-range 45 55 --pd-rps-range 11 13
```

The command reports the selected GPU counts, predicted saturation capacity,
and the encoder-to-PD rate ratio interval in which the split stays optimal.
It also checks whether the supplied rate ranges fit within that interval
and reports capacity at their lower endpoints for the selected split.
Pass `--current-encoder-gpus` to keep the current split on a tie. The upper
ratio endpoint is `null` when the interval is unbounded. Use
`--output-json PATH` to save the plan for deployment tooling.

Calibrate on the deployed E+PD path. Use an open-loop replay with enough
capacity in the other pool to find each stage's throughput plateau. Match
the media mix, input and output lengths, connector, CPU budget, cache hit
rates, and routing policy to the intended deployment. Monitor queues and
completion counts at both stages to identify the bottleneck. For an
encoder that handles individual media items, normalize to the same
original request stream as PD.

Validate the selected split at the target offered load through the EPD
endpoint. Use [`vllm bench serve`](../cli/bench/serve.md) to measure TTFT,
TPOT, and goodput, and track failure and unfinished request counts. Leave
headroom below the predicted saturation rate. The command covers capacity
planning for E+PD. Fully disaggregated E/P/D needs separate P and D profiles.

### Fit a profile and plan for a different workload

Set `VLLM_EPD_PROFILE_DIR` on the encoder process to an existing directory
to collect per-process JSONL events. The native calibration path currently
targets a single-GPU CUDA encoder-only runner with `ECExampleConnector`,
non-streaming encoder requests, and one uncached image per request. Use one
media loading thread and one renderer worker for this profile. Capture the
production settings you intend to keep during replay.

Collect a serial and a sustained-backlog replay at each of five or more
merged visual token counts. The encoder records the image embedding count
after preprocessing and spatial merging. Keep enough PD capacity to isolate
encoder load.
Provide `serial-<anchor>-window.json` and `saturated-<anchor>-window.json`
in the capture directory, with `begin_ns` and `end_ns` on the same host
monotonic clock as the events, plus `rate` and `offered_seconds` for each
sustained replay. Place encoder events under `native-profile/`.

```bash
vllm bench epd-calibrate --capture-dir encoder-capture \
    --pd-capture-dir pd-capture --output-json frozen-model.json
vllm bench epd-plan --num-gpus 4 --frozen-model frozen-model.json \
    --workload heldout-workload.json --output-json blind-plan.json
```

Use `--min-visual-tokens` when calibrating a profile for a workload that
excludes small inputs. Choose anchors so that the resulting calibrated range
covers the target workload. The planner reports that range and rejects
workloads with requests outside it.

The encoder fit contains serial component curves, saturated resource demand
curves, and the unattributed interval from the serial audit.
In each pass, the GPU forward `G` is fitted as a quadratic. The worker cycle
`X` is reconstructed as `G` plus an affine fit to the observed `X - G`,
so both curves share the forward's quadratic coefficient. Affine component
fits have nonnegative predictions throughout the calibrated token range.
For PD, collect a sustained replay with one profiled PD replica using
`--no-async-scheduling` and a fixed `--max-num-batched-tokens` budget. Supply a
`run-config.json` with `encoders`, `pd`, `base_port`, `profile_pd`, `pd_chunk`,
`model`, and `pd_kv_bytes`. Name the replay window `pd-training-window.json`.
For PD HTTP port `<port>`, put PD events in `native-profile-pd-<port>/` and
the startup log in `pd-<port>.log`. The calibrator reads KV token capacity
from the startup log and averages engine step times in the replay window.
Omit `--pd-capture-dir` to fit just the encoder.

The workload file contains `encoder` rows with `visual_tokens` and a `pd`
object with joint `input_lens` and `output_lens` arrays and the `failed` count.
The planning result includes SHA-256 digests for both inputs, the PD packing
range, and predicted capacities and bottlenecks for the chosen allocation and
its neighbors. With a frozen profile, the range check varies the PD rate
within the packing range while holding the encoder estimate fixed.

Encoder estimates use the inverse of the highest mean normalized resource
demand as an upper bound. The planner uses the lower PD rate from the
[Nie et al. KV cache model](https://arxiv.org/html/2605.04595v1#S4), with the
workload's largest observed input plus output length as the packing slack.
The PD approximation requires representative batch time and a workload
whose backlog keeps KV occupancy high. Validate the E and PD rates with
separate bottleneck replays before using them to plan an allocation. Inspect
the serial audit residual before interpreting the elapsed-time fit.

### Audio inputs

The Python EPD proxy also rewrites audio in `/v1/chat/completions` into
metadata-only references for Qwen2-Audio, AudioFlamingo3, Ultravox,
Qwen2.5-Omni, and Qwen3-Omni. The encoder publishes `audio_num_tokens`
(the placeholder token count of each audio) together with any feature
lengths the model needs; the consumer uses them to reconstruct the audio
placeholders and receives the embeddings through its EC connector without
repeating audio preprocessing.

Qwen2.5-Omni and Qwen3-Omni also support videos with an embedded audio track
using `mm_processor_kwargs={"use_audio_in_video": true}` in the request. Start
both instances with `--trust-request-mm-kwargs` only for trusted clients.
The proxy keeps one
`video_embeds` content part per video, with an additional `audio_embeds` payload
for its audio track. Both features share the source UUID but retain their own
EC hashes and transfer handles. Audio lengths and video timing reconstruct the
interleaved placeholders and M-RoPE positions without decoding the media again.
If any item lacks metadata, the entire audio/video request falls back to raw
media rather than mixing raw and metadata-only inputs.

This does not add `/v1/audio/transcriptions` or realtime routes to the example
proxy.

---

## 3  Test Script

For optional cross-Encoder output reuse while retaining Mooncake P2P delivery,
see [Cross-encoder output reuse](cross_encoder_cache.md).

Please refer to the directories `tests/v1/ec_connector`

## 4  Development

Disaggregated encoding is implemented by running two parts:

* **Encoder instance** – a vLLM instance to performs vision encoding.  
* **Prefill/Decode (PD) instance(s)** – runs language pre-fill and decode.
    * PD can be in either a single normal instance with `disagg_encoder_example.sh` (E->PD) or in disaggregated instances with `disagg_epd_example.sh` (E->P->D)

A connector transfers encoder-cache (EC) embeddings from the encoder instance to the PD instance.  
All related code is under `vllm/distributed/ec_transfer`.

### Key abstractions

* **ECConnector** – interface for retrieving EC caches produced by the encoder.  
    * *Scheduler role* – checks cache existence and schedules loads.  
    * *Worker role* – loads the embeddings into memory.

Here is a figure illustrating disaggregate encoder flow:

![Disaggregated Encoder Flow](../assets/features/disagg_encoder/disagg_encoder_flow.png)

For the PD disaggregation part, the Prefill instance receives cache exactly the same as the disaggregated encoder flow above. Prefill instance executes 1 step (prefill -> 1 token output) and then transfers KV cache to the Decode instance for the remaining execution. The KV transfer part purely happens after the execution of the PD instance.

`docs/features/disagg_prefill.md` shows the brief idea about the disaggregated prefill (v0)

We create the example setup with the **NixlConnector** from `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` and referred to the `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` to facilitate the kv transfer between P and D;
