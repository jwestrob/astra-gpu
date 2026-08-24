# Phase 9: certified GA pruning

## Hypothesis

For `--cut_ga` searches, some targets may be provably unable to reach the
model-specific target cutoff after their isolated-domain Forward scores are
known. Such targets could skip the remaining isolated-domain Backward,
optimal-accuracy decoding, traceback, and null2 work without changing any
HMMER decision.

The first commit was census-only. The production experiment remains an
explicit private opt-in until exact-output, runtime, and memory gates pass.

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

H200 job `1182810` ran all 27,481 Pfam profiles against the immutable first
1,000 targets. The complete target/domain/accounting digest was exactly the
established CPU/GPU oracle
`a0d928902cd8f6c3342b05c71aedaf0190ac2e02478ff879988a41d250de85b1`,
and no certified target appeared as a reported hit.

Opportunity:

- 1,592 / 1,748 compact target rows certified: **91.08%**;
- 2,753 / 3,002 compact regions covered: **91.71%**;
- 24,618,449 / 28,625,148 region DP cells per downstream pass covered:
  **86.00%**; and
- 1,270 profiles had at least one certified target rejection.

The diagnostic scan took 0.437 s including 27,481 Python calls; its native
scan component was 0.00394 s and native temporary bytes were zero. Maximum
diagnostic CandidateBatch payload was 2,746,705 host bytes and zero owned
device bytes. The complete run observed 611,385,344 CUDA bytes in use and a
3,854,992 KiB process RSS high-water mark.

Evidence:
`build/phase9-ga-pruning-h200-20260824/attempt-01/result.json`, SHA-256
`709b529da95f678d01c7d5496dd9baeffdfdc49cefd7bad15ebed5dee9d1e8e2`.
The record's dirty flag refers only to two untracked fixture symlinks present
during launch; tracked source was exact commit `d4a00589...`, and the symlinks
were removed immediately afterward.

Decision: **proceed** with a production experiment that performs this exact
row decision immediately after isolated-domain Forward and skips downstream
Backward/OA/trace/null2 only for certified rows. Keep the current path as the
audit/rollback mode and require exact output plus runtime and memory evidence
before promotion.

## Production experiment

The `_ga_pruning=True` fused-generation option is deliberately restricted to
gathering cutoffs and direct sparse journal v3. After isolated-domain Forward,
the native rescore stage evaluates the same certified target bound as the
census. Certified rows retain their exact isolated Forward results but do not
enter isolated Backward, optimal-accuracy decoding, traceback, or null2.

The existing path remains byte-for-byte available when the option is false.
The first implementation also retains the existing matrix allocation so that
the experiment changes execution, not workspace admission; consequently its
initial memory high-water may remain flat. New counters report certified rows,
regions, skipped DP cells, and classification time.

For order-sensitive continuation, a certified row is encoded as the existing
exact terminal/no-region journal certificate: it contributes all four filter
promotion counters and no hit, while original target ordinals and residue
accounting remain unchanged. The native rescore census separately identifies
these GA certificates so they cannot be confused with Backward no-region
outcomes in performance analysis.

The focused production gate is
`tests/audit_h200_phase9_ga_pruning_production.py`. It compares the complete
target table, domain table, query identity, and pipeline accounting of the
ordinary and optimized paths for the first 64 Pfam profiles against 1,000
targets, including 11 certified target rows / 19 compact regions. A full H200
run is required before promotion.
