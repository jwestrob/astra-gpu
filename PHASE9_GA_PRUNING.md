# Phase 9: certified GA pruning

## Hypothesis

For `--cut_ga` searches, some targets may be provably unable to reach the
model-specific target cutoff after their isolated-domain Forward scores are
known. Such targets could skip the remaining isolated-domain Backward,
optimal-accuracy decoding, traceback, and null2 work without changing any
HMMER decision.

The first commit was census-only. The production implementation was kept as
an explicit private opt-in until its exact-output, runtime, and memory gates
passed.

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
targets, including 11 certified target rows / 19 compact regions. The full
H200 run below is the promotion gate.

## Production result and decision

Focused H200 job `1182812` passed the ordinary-versus-optimized semantic
oracle. Both paths produced canonical digest
`2af479c8b8343c98e8b68152194e72f1484a036849a678f3e27de970c5a15672`;
the optimized path certified 11 target rows / 19 regions and skipped 202,319
per-pass DP cells. Its generation time was 0.0925 seconds versus 0.1034
seconds for the ordinary path. Evidence is
`build/phase9-ga-production-h200-20260824/attempt-02/result.json`, SHA-256
`cf53d419449267ef48634e338c29e756153e7c31ecb8e66ee8c72c5876772279`.

Full H200 job `1182813` then searched all 27,481 Pfam profiles against all
300,186 targets. It reproduced the exact established output:

- SHA-256
  `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`;
- 39,010,327 bytes; and
- 383,235 lines including the header.

Measured request wall was **449.104112 seconds**, with 340.105077 seconds of
generation, 398.198633 seconds of continuation/output, 296.918163 seconds of
overlap, and 441.512164 seconds of pipeline wall. Against the previous best
retained packed-MSV run (454.006856 seconds), request wall improved by
4.902743 seconds (1.080%). Against the original 546.220705-second H200 run it
improved by 97.116593 seconds (17.780%). It is 1.5649x faster than Astra CPU64.

The optimization certified 117,545 target rows containing 203,880 regions.
Those rows moved from the `SIMPLE` continuation route into exact terminal
certificates, reducing compact attempts from 128,314 to 10,769 (91.61%) while
leaving all 552,390 `CPU_REQUIRED` rows unchanged. The sparse packet shrank
from 5,801,342,068 to 4,748,364,004 bytes (18.15%), and its maximum retained
CandidateBatch payload fell from 427,998,281 to 417,962,513 bytes (2.35%).

Maximum process RSS was 13,159,348 KiB, 86,508 KiB (0.65%) below the prior
best run. Peak sampled H200 memory was 3,392 MiB versus 3,390 MiB previously,
which is effectively flat and expected because this first production slice
deliberately retains the existing matrix allocation.

Decision: **retain** the exact gathering-cutoff GA-pruning path. It produces a
real end-to-end runtime win and a modest host-memory win without changing
HMMER output. The ordinary path remains available for non-GA modes and as the
explicit audit/rollback strategy.

Full evidence is under
`build/h200-phase9-ga-pruning-20260824/attempt-01-full/runs/h200-full`.
`worker.json` SHA-256 is
`e7930b72344676ca96d5d43fc97daa45fd8c6b2e066a8588efe732d199ab9320`;
`raw-validation.json` SHA-256 is
`4c71eab4bd7eac4359f434baf0fcdf688e82b00c62872c607b771e86ced53245`;
and the artifact-manifest SHA-256 is
`7157bfa28635e0532a5306eb2ac3eb68deb02908438cecac3f59d27f8bc86af8`.
