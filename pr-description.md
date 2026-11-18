## Overview

Add a runnable sizing example for a single-node E+PD deployment with one GPU
per worker. Given a GPU budget and measured encoder and combined prefill/decode
rates, report the recommended split, estimated capacity, and bottleneck.

Refs [#67204](https://github.com/vllm-project/vllm/issues/67204) and the
[EPD tracker](https://github.com/vllm-project/vllm/issues/52409). A search of
open PRs for EPD sizing, planning, and references to #67204 found related
proxy and connector changes. This example addresses replica sizing.
The issue currently has no comments and the duplicate search found no open PR
referencing #67204 or implementing EPD sizing.

## Claims

- The example evaluates each split using the smaller of the two estimated
  aggregate worker rates and reports the highest-capacity split.
- The documentation describes how to measure isolated stage saturation and
  check the estimate using independent requests.

## Validation

```bash
.venv/bin/python examples/disaggregated/disaggregated_encoder/size_epd.py \
  --num-gpus 8 --encoder-rps 57 --pd-rps 14.3
```

Expected result is `2E+6PD`, with an estimated `85.8 req/s` and PD as the
bottleneck. Local checks also covered a two-GPU deployment, a tied optimum, and
invalid rates. Ruff and formatting checks passed. A focused MkDocs build
rendered the feature guide and generated example and verified their link.

```bash
git clone --single-branch --branch epd-sizing-67204-artifacts \
  https://github.com/Kelvin-Al/vllm.git ../epd-sizing-artifacts
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements/lint.txt
.venv/bin/python ../epd-sizing-artifacts/check_sizing.py --repo .
.venv/bin/python -m pre_commit run --files \
  examples/disaggregated/disaggregated_encoder/size_epd.py \
  examples/disaggregated/disaggregated_encoder/README.md \
  docs/features/disagg_encoder.md
git diff --check
```

Four expected allocations and eight invalid inputs pass through the real CLI.
The audit and documentation commands are in the public
[reproduction guide](https://github.com/Kelvin-Al/vllm/tree/epd-sizing-67204-artifacts)
with [check output](https://github.com/Kelvin-Al/vllm/blob/epd-sizing-67204-artifacts/checks.log).
The whole-site Windows MkDocs attempt encountered a CUDA extension import in
unrelated CLI-reference generation. The recorded scoped build renders the changed
pages with the repository's example generator.

For the SuperPOD replay, use four H800 GPUs on one node and vLLM `v0.33.0`
at revision `4ba787ec9959e566d3b587a796dff07673324da5`. The workload uses
one synthetic image per request, Qwen2.5-VL-3B-Instruct, an encoder batch
cap of one, and a fresh EC cache and service processes for every pressure run.
Prefix caching was disabled. PD used a 1 GiB KV cache and a 512-token batch
budget with async scheduling disabled. Calibration and validation used disjoint
random seeds. SuperPOD job `783695` completed in 48 min 19 s on four H800s
with driver `580.178.04`. All 14,016 requests completed across 17 runs.
The model cache resolved `main` to snapshot
`66285546d2b821cf421d4f5eb2576359d3770cd3`.

```bash
mkdir epd-sizing-reproduction
tar -xzf ../epd-sizing-artifacts/epd-sizing-67204-reproducer.tar.gz -C epd-sizing-reproduction
cd epd-sizing-reproduction
sbatch epd-sizing-67204.sbatch  # SuperPOD, with the configured SIF and HF cache
# Alternative on Linux with four visible GPUs and the pinned vLLM virtualenv
source /path/to/vllm-0.33.0/.venv/bin/activate
/path/to/vllm-0.33.0/.venv/bin/python run_epd_sizing_experiment.py \
  --repo "$PWD" --sizer "$PWD/size_epd.py" \
  --output ./epd-sizing-results
```

The harness records launch arguments, load parameters and seeds, raw completion
counters, and client results. `calibration.json` contains the saturation
measurements. The harness writes and hashes `frozen-plan.json` before launching
the three splits at each held-out offered load. `validation.json` records the
result for each split and the frozen plan hash. Stage throughput uses completion
counter deltas within the central 20%–80% of the offered-load window, aligned to
the first observed completion. Client throughput includes the drain period.

[Reproducer archive](https://raw.githubusercontent.com/Kelvin-Al/vllm/epd-sizing-67204-artifacts/epd-sizing-67204-reproducer.tar.gz), SHA-256
`6dfde91a45150412ad6c991e16a8230ef06bca9d8a6f2c51661b9bb53584bf4d`.
[Raw results and logs](https://raw.githubusercontent.com/Kelvin-Al/vllm/epd-sizing-67204-artifacts/epd-sizing-783695-results.tar.gz), SHA-256
`b7bdc63bd80a5b4e4dd2166444328e6689a138356c5ef15244f16b3c1c8d4cd3`.

### SuperPOD independent calibration and held-out validation

| Stage | Offered req/s | Single-worker completions/s |
| --- | ---: | ---: |
| encoder | 20 | 11.33 |
| encoder | 25 | 11.18 |
| pd | 16 | 6.58 |
| pd | 20 | 6.92 |

The mean measured inputs were 11.25 req/s for E and 6.75 req/s for PD. The frozen recommendation was `2E+2PD` at 13.50 req/s, with `pd` as the estimated bottleneck.

Frozen plan SHA-256 `37288277d97d8a5db9ac8ee028a0cd2576d25a3c5256e10b3956b3507b4ba51e`.

| Offered req/s | Split | Estimated capacity | PD completions/s | Client req/s | Completed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 25 | 1E+3PD | 11.25 | 11.59 | 11.27 | 960 |
| 25 | 2E+2PD | 13.50 | 13.78 | 13.58 | 960 |
| 25 | 3E+1PD | 6.75 | 6.84 | 7.22 | 960 |
| 31 | 1E+3PD | 11.25 | 11.57 | 11.24 | 960 |
| 31 | 2E+2PD | 13.50 | 13.49 | 13.14 | 960 |
| 31 | 3E+1PD | 6.75 | 6.18 | 6.98 | 960 |

- At 25 offered req/s, the recommendation was fastest at 13.78 req/s, 18.9%
  ahead of the next split. Capacity estimate error was −2.0%.
- At 31 offered req/s, the recommendation was fastest at 13.49 req/s, 16.6%
  ahead of the next split. Capacity estimate error was +0.1%.


The same reproducer also runs a workload-shift check when its bundled
`frozen_predictor/` directory is present. The archived predictor and calibration
are research artifacts for this validation. The CLI example takes measured E and
PD service rates directly.

### Workload shift with the same frozen calibration

The original mix used three image sizes and a mean target of 48 output tokens. The shifted mix used only 1400×1400 images (2500 merged tokens) and a mean target of 128 output tokens. Model, hardware, cache policy and serving configuration were fixed. We sampled joint prompt and output lengths with seed 6720420 and locked both recommendations before replay with held-out seeds 6720421 and 6720422.

The frozen model estimated E at 17.04 req/s and PD at 4.57–5.06 req/s per worker. It selected `1E+3PD` at 13.70 req/s. Reusing the original rates selected `2E+2PD` at 13.50 req/s.

Frozen model SHA-256 `91e22357828838743e77796a0eae05bb0e9cc0ed54fb46eaf06e04e2a63025ec`. Shift plan SHA-256 `1930166cb592c7c9be7fedb7e4e3c6c6ade95f607701a44e5b46799e61d30c8e`.

| Offered req/s | Split | Frozen estimate | Observed E req/s | Observed PD req/s |
| ---: | --- | ---: | ---: | ---: |
| 20 | 1E+3PD | 13.70 | 16.44 | 13.82 |
| 20 | 2E+2PD | 9.14 | 20.27 | 9.10 |
| 20 | 3E+1PD | 4.57 | 20.20 | 4.46 |
| 25 | 1E+3PD | 13.70 | 16.35 | 13.08 |
| 25 | 2E+2PD | 9.14 | 23.95 | 8.85 |
| 25 | 3E+1PD | 4.57 | 23.95 | 4.33 |

- At 20 offered req/s, `1E+3PD` was fastest at 13.82 req/s, 51.9% ahead of the
  old-rate recommendation at 9.10 req/s. Frozen capacity error was −0.8%.
  The backlogged single E and PD workers completed 16.44 and 4.46 req/s,
  giving point-estimate errors of +3.6% and +2.5%, respectively.
- At 25 offered req/s, `1E+3PD` was fastest at 13.08 req/s, 47.9% ahead of the
  old-rate recommendation at 8.85 req/s. Frozen capacity error was +4.8%.
  The backlogged single E and PD workers completed 16.35 and 4.33 req/s,
  giving point-estimate errors of +4.2% and +5.5%, respectively.


## Details

The sizing rule is described in §3 of *Provision Before You Serve: Computing
the E:PD Ratio for Disaggregated Multimodal LLM Serving*, accepted at FAISys.
The workload-shift check studies whether one calibration can supply rates for
different request mixes within its encoder token range and under the same serving
configuration. The archived model recomputes rates from merged encoder tokens
and paired PD input/output lengths. The baseline keeps the two rates measured
for the original request mix. The held-out sweep tests both recommendations.
This experiment covers one in-range shift on one serving configuration. The PD
estimate still showed positive error on both shifted-load runs.
Each split has one run at each of two offered loads, using different seeds.
These measurements report variation across the tested conditions. They do not
establish repeated-run confidence intervals. The archived predictor and
calibration support the supplementary validation, while `size_epd.py` accepts
rates from either measurements or an external estimator. The archived sizing
script produces the same allocation arithmetic. The submitted version adds CLI
help and finite-capacity input validation.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill.
- [x] AI assistance was used during the creation of this PR.
- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

The submitting human should review every changed line and run the relevant tests
before completing the remaining checklist items.

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>**
