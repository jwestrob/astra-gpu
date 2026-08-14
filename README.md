# plan7-gpu

`plan7-gpu` is an experimental hybrid GPU/CPU accelerator for the
HMMER 3.4 protein `hmmsearch` pipeline.

The first implementation target is deliberately narrow:

1. digitize a real protein sequence corpus once;
2. evaluate HMMER-equivalent SSV/MSV stage-1 scores on a GPU;
3. conservatively select candidate model/sequence pairs;
4. run the unmodified CPU HMMER pipeline on candidates; and
5. require exact agreement with reference HMMER output.

GPU Forward/Backward and Astra integration are out of scope until a
scientifically exact MSV accelerator demonstrates useful end-to-end speedup.

## Status

Reference-oracle and stage-timing phase. The independent scalar byte-MSV
recurrence agrees bit-for-bit with pristine HMMER on the initial 5,145 real
and upstream tutorial comparisons, covering all four SSV/fallback outcomes.
This is an initial result, not a completed correctness gate. No CUDA code or
GPU performance claim exists yet. See [`handoff.md`](handoff.md) for the
project brief and gates.

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
