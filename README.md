# plan7-gpu

`plan7-gpu` is an experimental hybrid GPU/CPU accelerator for Astra's
HMMER 3.4 protein-search pipeline.

The first implementation target is deliberately narrow:

1. reuse the HMMs and digital sequences already loaded by Astra;
2. evaluate HMMER-equivalent SSV/MSV stage-1 scores on a GPU;
3. conservatively select candidate model/sequence pairs;
4. run Astra's existing CPU HMMER pipeline on candidates; and
5. require exact agreement with ordinary Astra output.

The native backend must plug into Astra without replacing its input handling,
threshold logic, downstream HMMER pipeline, or result formatting.

## Status

The correctness oracle agrees with pristine HMMER on the initial 11,145
comparisons and covers all four SSV/fallback outcomes. Direct measurements in
Astra show that stage one is about 45% of wall time for the PFAM pilot but only
26% for HydDB, so GPU use must be workload-selective. The GPU target is the
public SSV-first behavior, not generic full MSV. No GPU performance claim
exists yet.

## Layout

- `baseline/`: CPU and end-to-end benchmark drivers
- `oracle/`: reference instrumentation and differential tests
- `cuda/`: CUDA implementation (created only after the oracle gate)
- `scripts/`: reproducible setup, discovery, and benchmark scripts
- `results/`: machine-readable measurements
- `refs/`: local upstream source/build/install trees (ignored by Git)

## Scientific policy

- HMMER 3.4 defines correctness.
- Performance benchmarks use real biological data.
- Generated/adversarial inputs may be used only for arithmetic correctness
  testing and are never reported as biological benchmarks.
- GPL implementations may be studied as prior art, but no GPL source is
  copied into this repository.
