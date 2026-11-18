# E+PD sizing validation for vLLM #67204

Reproduction material for the proposed [E+PD sizing example](https://github.com/vllm-project/vllm/issues/67204).
The code contribution accepts measured per-worker E and combined PD service
rates. The supplementary workload-shift experiment uses an archived calibrated
model to supply updated rates for a new request distribution.

## Downloads and integrity

| Archive | SHA-256 |
| --- | --- |
| `epd-sizing-67204-reproducer.tar.gz` | `6dfde91a45150412ad6c991e16a8230ef06bca9d8a6f2c51661b9bb53584bf4d` |
| `epd-sizing-783695-results.tar.gz` | `b7bdc63bd80a5b4e4dd2166444328e6689a138356c5ef15244f16b3c1c8d4cd3` |

The archives preserve the code and outputs used by job 783695. `pr-description.md`
contains the current review draft. The PR branch adds help text and input
validation to the archived sizing CLI. The valid-rate allocation formula is the
same. Run `check_sizing.py` against the review checkout to verify its CLI.

## Experimental environment

- SuperPOD job 783695 on dgx-30, four H800 GPUs, driver 580.178.04.
- November 17, 2026, 06:33:32–07:21:51 Hong Kong time, exit code 0.
- vLLM 0.33.0, source tag commit `4ba787ec9959e566d3b587a796dff07673324da5`.
- Model `Qwen/Qwen2.5-VL-3B-Instruct`, cached snapshot
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- One GPU per worker, eager execution, encoder batch cap 1, one media-loading
  thread, one renderer worker, max model length 8192.
- PD KV memory 1 GiB, batch token budget 512, max sequences 128, async scheduling
  disabled. Prefix caching disabled. ECExampleConnector uses a fresh shared
  filesystem directory with freshly launched workers for every run.

The original workload mixes 1400×1400, 1736×1736 and 2072×2072 images with
probabilities 0.45, 0.35 and 0.20. Text input target is 128 tokens, output target
48, and random range ratio 0.25. Each request contains one synthetic image.
The shifted distribution uses only 1400×1400 images and changes the output
target to 128. The image processor confirms 2500 merged tokens for that size.

The measured original rates are 11.2545 E req/s and 6.7512 PD req/s. The frozen
recommendation is 2E+2PD at 13.5024 req/s. Both held-out offered loads select
2E+2PD as the fastest measured split. After the workload shift, the archived
model recommends 1E+3PD at 13.70 req/s. That split is fastest at both tested
offered loads, 51.9% and 47.9% above retaining the old-rate recommendation.
This tests one in-range shift with a fixed serving configuration.

## Reproduce the GPU experiment

Use Linux with four visible GPUs and the pinned vLLM version. The archived
runner launches `vllm serve`, so activate the corresponding virtual environment
first. The container used by the Slurm script already has vLLM installed.

```bash
sha256sum epd-sizing-67204-reproducer.tar.gz epd-sizing-783695-results.tar.gz
mkdir replay
tar -xzf epd-sizing-67204-reproducer.tar.gz -C replay
cd replay
source /path/to/vllm-0.33.0/.venv/bin/activate
/path/to/vllm-0.33.0/.venv/bin/python run_epd_sizing_experiment.py \
    --repo "$PWD" --sizer "$PWD/size_epd.py" --output "$PWD/replayed-results"
```

On SuperPOD, the provided script requests four GPUs on one node for up to two
hours. Set the image and Hugging Face cache locations to existing directories
accessible on the allocated node before submitting.

```bash
export EPD_SIF=/path/to/vllm-v0.33.0.sif
export EPD_HF_CACHE=/path/to/huggingface-cache
sbatch epd-sizing-67204.sbatch
```

The archived runners launch the container's Python interpreter. If replaying in
a virtual environment, the outer command and subprocesses use that environment.
The original run resolved the model's `main` revision from the recorded cache.
Use a cache whose `refs/main` points to the recorded snapshot to reproduce the
same weights and image processor.

`run_epd_sizing_experiment.py` writes four independent calibration captures,
then `frozen-plan.json` and its hash before six held-out captures. The shift
experiment verifies the preexisting frozen model hash, gathers 96 request-shape
samples, writes a second frozen plan and then runs six independent validation
captures. It performs no fitting. The sampling run observes joint input/output
lengths, so this mode needs a sample of completed requests.

## Audit existing outputs without GPUs

```bash
mkdir recorded
tar -xzf epd-sizing-783695-results.tar.gz -C recorded
uv venv --python 3.12 .venv
.venv/bin/python check_results.py \
    --results recorded/results-783695 --reproducer replay
.venv/bin/python check_sizing.py --repo /path/to/pr-checkout
```

`check_results.py` independently recomputes rates from archived raw counters,
verifies hashes, confirms calibration ended before held-out runs, checks replay
length equality for matching seeds, and checks that all 14,016 requests across
17 runs succeeded. Frozen-plan timestamps can additionally be checked in the
original tar archive. Re-extraction timestamps depend on the extraction tool.

The capacity metric is a completion-counter delta in the central 20%–80% of the
offered duration aligned to the first observed completion. Client throughput
includes the drain period and is reported separately. Each configuration has
one run at each of two loads with different seeds. These runs measure variation
across those conditions and do not establish repeated-run confidence intervals.

## Review checks

From the PR checkout, run the CLI checks and repository hooks.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements/lint.txt
.venv/bin/python /path/to/artifacts/check_sizing.py --repo .
.venv/bin/python -m pre_commit run --files \
    examples/disaggregated/disaggregated_encoder/size_epd.py \
    examples/disaggregated/disaggregated_encoder/README.md \
    docs/features/disagg_encoder.md
git diff --check
```

The scoped documentation check uses the repository's actual example generator
and renders the two affected pages with MkDocs. Its small temporary site emits
warnings for existing feature-guide links outside those two pages. The new
feature-to-example link is asserted directly.

```bash
uv pip install --python .venv/bin/python -r requirements/docs.txt
.venv/bin/python /path/to/artifacts/check_docs.py --repo .
```

The full Windows MkDocs build encountered a CUDA extension import while
generating unrelated CLI reference pages. The scoped render check is recorded
separately from a full site build. See `checks.log` for commands and results.
