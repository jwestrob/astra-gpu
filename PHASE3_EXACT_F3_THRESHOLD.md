# Phase 3 exact F3 threshold compiler

## Hypothesis

The host-side Forward/F3 decision can be replaced, for validated inputs, by a
single comparison against a profile-specific binary32 threshold without
approximating HMMER's statistics.

## Source predicate

HMMER 3.4 `src/p7_pipeline.c` stores `seq_score` as `float`, computes
`(fwdsc - filtersc) / eslCONST_LOG2`, and promotes that binary32 score to
`double` for:

```c
P = esl_exp_surv(seq_score, om->evparam[p7_FTAU],
                 om->evparam[p7_FLAMBDA]);
if (P > pli->F3) reject;
```

Easel `esl_exponential.c` returns `1.0` below `tau` and otherwise calls
`exp(-lambda * (score - tau))`. With finite `tau`, finite positive `lambda`,
and finite `F3` in `[0, 1]`, that exact host predicate is monotone over numeric
binary32 scores. No inverse exponential or algebraically reconstructed cutoff
is needed.

## Implementation

`cuda/f3_threshold.cc` searches the complete ordered numeric binary32 domain
from negative infinity through positive infinity. Each binary-search probe
calls the linked `esl_exp_surv()` used by the HMMER oracle. The result carries:

- the input `tau`, `lambda`, and `F3` bits;
- the smallest passing binary32 threshold;
- predecessor, threshold, and successor decisions;
- infinity decisions and the NaN caveat;
- a conservative unsupported reason for invalid parameters or a failed
  boundary certificate.

The device rule is exactly `bit_score >= threshold` for non-NaN binary32
scores. Both signed zeros and infinities are included. NaN must keep the
existing fallback guard: HMMER's `P > F3` expression treats a NaN probability
as not greater and therefore promotes it, while an ordered comparison with NaN
is false. Valid finite `fwdsc` and `filtersc` cannot produce NaN by subtraction,
but the guard remains explicit rather than relying on that fact.

The follow-up production slice uploads one threshold per profile and the
existing candidate filter scores, then forms the device bit score explicitly
as binary32 round-to-nearest subtraction, binary64 round-to-nearest division
by `eslCONST_LOG2`, and binary32 round-to-nearest conversion. Its pass/reject
bit is packed into the existing eight-byte kernel result, so result D2H traffic
does not grow.

The old host predicate still audits every otherwise eligible row. A matching
device decision is consumed; an unavailable or mismatching decision uses the
host answer. Separate additive counters record compiled/unsupported profiles,
host audits, device passes/rejects, fallbacks, and mismatches. These counters
do not change result, continuation, or provenance ABIs.

## Correctness evidence

The focused host oracle completed in 0.630 seconds:

```text
python -m unittest discover -s tests -p 'test_f3_threshold.py' -v
Ran 4 tests in 0.630s -- OK
```

It checks the exact host predicate at every compiled predecessor/threshold/
successor, adversarial binary32 exponent and mantissa patterns, an 8,193-value
window around each boundary, both zeros, infinities, canonical NaNs, and all
63,490 non-NaN binary16 values promoted exactly to binary32. Invalid and
degenerate parameters all select the conservative unsupported path.

The focused H200 old-vs-new oracle also passed:

```text
Slurm job 1182629
test_exact_f3_boundary_and_device_gather ... ok
Ran 1 test in 0.159s -- OK
```

It checks exact pass and reject decisions on opposite sides of the host F3
boundary, requires zero device/host mismatches, and verifies that a noncanonical
`F3 > 1` takes the host fallback.

## Cost and decision

A Python-exposed microbenchmark compiled 100,000 varying boundaries in
0.654 seconds (6.54 microseconds each, including dictionary construction).
At 27,481 profiles that is about 0.18 seconds before batching or removing the
test wrapper. The exact compiler and conservative CUDA consumer are retained.
This is a correctness slice, not yet the transfer optimization: candidate
filter scores are temporarily uploaded into a persistent workspace buffer and
the host audit remains enabled. Phase 3 residency can remove that upload and
the per-row host oracle after a representative zero-mismatch gate.
