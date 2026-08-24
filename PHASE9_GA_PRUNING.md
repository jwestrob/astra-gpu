# Phase 9: certified GA pruning

## Hypothesis

For `--cut_ga` searches, some targets may be provably unable to reach the
model-specific target cutoff after their isolated-domain Forward scores are
known. Such targets could skip the remaining isolated-domain Backward,
optimal-accuracy decoding, traceback, and null2 work without changing any
HMMER decision.

This phase is census-only until both the proof and real-workload opportunity
are strong. It does not change production search decisions.

## Source proof

The ordinary protein pipeline has two possible final target scores:

1. the whole-target Forward score, minus the null1 score and a nonnegative
   null2 bias; and
2. a reconstruction score formed from selected domain envelope Forward
   scores, plus a nonpositive length term, minus null1 and a nonnegative null2
   bias.

The larger score wins. Therefore whole-target Forward alone is **not** a safe
upper bound: reconstruction can override it.

Once every isolated-domain Forward score is available, these conservative
bounds are safe:

```text
whole_upper = (whole_forward + input_error - null1) / ln(2) + round_error

reconstruction_upper =
    (sum(max(domain_forward + input_error, 0)) - null1) / ln(2)
    + round_error

target_upper = max(whole_upper, reconstruction_upper)

domain_upper =
    (domain_forward + input_error - null1) / ln(2) + round_error
```

They deliberately omit the reconstruction length penalty and all null2
corrections because those terms cannot increase the final score. The input
and arithmetic allowances are the same `0.004` nats and `1e-5` bits already
authenticated by the private compact-domain continuation seam. A target is a
certified reject only when `target_upper < GA1`.

A domain-only rejection is counted, but is not yet an actionable production
optimization: exact target output may still need that domain's best-domain
metadata. The safe production unit considered here is the whole target row.

Relevant stock HMMER implementation is `p7_pipeline.c`'s
`pipeline_post_domains()`, including the whole score, reconstruction override,
and per-domain score expressions. Null1 is reproduced from
`p7_bg_SetLength()` and `p7_bg_NullOne()` for the authenticated target length.

## Implementation

`CandidateBatch.evaluate_ga_pruning()` is a private diagnostic over a sealed
sparse-journal-v3 batch. It:

- requires an exact `gathering`-cutoff pipeline and finite `GA1`/`GA2`;
- performs the normal live pipeline/profile preflight without mutation;
- scans only authenticated compact-domain rows with complete device results;
- reports certified target/domain counts and estimated region DP cells;
- optionally returns original target ordinals for output cross-checking; and
- allocates no native temporary workspace.

It does not suppress a row, change a counter, or alter TopHits.

## Correctness evidence

The focused host fixture contains two compact target rows. Both whole-target
Forward bounds are below GA, but one target has a high isolated-domain Forward
score. The evaluator certifies only the other target, counts its domain and
cells, preserves original ordinals, and leaves the canonical pipeline state
unchanged.

Result: `1/1 PASS` under the exact patched HMMER ABI.

`tests/audit_h200_phase9_ga_pruning.py` is the real-workload gate. It runs all
27,481 Pfam profiles against the immutable first 1,000 targets, verifies the
complete current output digest against the established CPU/GPU oracle, checks
that no certified target is reported, and records runtime plus host/device
memory.

## Benchmark result and decision

Pending the focused H200 census. Do not integrate pruning into production
unless the certified share of downstream compact-rescore cells is substantial
and the census remains exact.
