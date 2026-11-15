# Offline E+PD capacity planning

This experiment recommends a split between single-GPU encoder replicas and
single-GPU combined prefill/decode replicas. It uses an offline calibration
and a target workload sampled from completed requests.

- **Encoder.** In a strict serial pass, collect ingress, media preprocessing,
  coordination, the worker cycle, GPU forward, and result handoff. Fit the GPU
  forward as a quadratic in merged visual tokens and fit the CPU-only part of
  the worker cycle as affine. Reconstruct the worker cycle from those two
  fits. In the saturated pass, fit the occupancy of each resource using its
  measured demand. Average each resource's normalized demand over the target
  workload, then invert the largest mean to estimate an encoder rate upper
  bound. Select a minimum merged-token count with `--min-visual-tokens` if the
  target workload excludes small inputs.
- **Prefill/decode.** Use the [Nie et al. KV cache model](https://arxiv.org/html/2605.04595v1#S4).
  Measure usable KV capacity and batch step time on the target PD
  configuration. Estimate lifetime KV token-steps from paired input and
  output lengths. The planner reports the packing range and uses its lower
  endpoint for the capacity calculation. These estimates depend on how well
  the calibration step time and the model's packing assumptions describe
  the target workload.
- **Allocation.** Under a fixed GPU budget, compare integer splits around
  the point that balances the two stage capacities. Each split's capacity
  is the smaller of its two pool rates. The report includes neighboring
  splits, bottlenecks, the calibrated E range, and input digests.

See [disaggregated encoder documentation](../../../../docs/features/disagg_encoder.md)
for the native calibration and planning commands. The example targets a Linux
CUDA node, a single uncached image per request, and the filesystem-backed
`ECExampleConnector`.

## Replay a four-GPU recommendation

With the vLLM environment active and a paired native encoder capture
available, run the experiment from the checkout root.

```bash
python examples/disaggregated/disaggregated_encoder/epd_capacity/run_epd_offline.py \
  --encoder-capture path/to/encoder-capture \
  --output epd-offline-run
```

The harness calibrates PD, freezes the model, samples a seeded held-out
corpus, and records a plan before pressure testing. It compares `1E+3PD`,
`2E+2PD`, and `3E+1PD` on the same corpus. Every pressure wave starts new
services with a fresh encoder cache directory so repeated requests cannot
reuse embeddings from the preceding wave. The output records observed
completion rates and queue depths for both stages.

To repeat just the pressure tests, pass `--frozen-run epd-offline-run`
with a new `--output`. The harness checks the profile and workload digests
against the saved plan before replaying it.
