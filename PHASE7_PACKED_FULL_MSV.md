# Phase 7: packed full-MSV arithmetic

## Hypothesis

After exact device compaction, full MSV still executes 40,657,346 independent
four-lane byte recurrences on the sealed workload. Four candidates for the
same profile and target length can occupy the four bytes of each CUDA register
without changing the recurrence order within any candidate. Hopper's exact
packed unsigned-byte add, subtract, and maximum operations can therefore
increase arithmetic density without weakening HMMER semantics.

## Implementation

The compact full-MSV host plan groups candidates by exact profile and target
length in linear time. It uses the immutable target-length table and reusable
length-class buckets; no comparison sort is performed. Complete quartets are
placed first and the unchanged scalar kernel consumes all leftovers.

The packed kernel represents four independent DP cells in each `uint32_t` and
uses `__vaddus4`, `__vsubus4`, and `__vmaxu4`. Each byte retains its own target
residues, `tjb`, overflow state, numerator, and original candidate index.
Per-byte ERANGE detection prevents a saturated candidate from being
overwritten while the remaining candidates continue. Profiles, result rows,
postfilter ordering, journal layout, and downstream ABIs are unchanged.

`PLAN7_GPU_FULL_MSV_ARITHMETIC=scalar` forces the old scalar recurrence for
audit. The default compact path uses packed quartets when any exist and falls
back exactly for scalar leftovers and small workloads. Counters report packed
runs, groups, candidates, and scalar leftovers.

## Correctness evidence

The exact d486 private-ABI build produced sm75/PTX and sm90 code successfully.
Focused H200 job `1182763` compared forced scalar and packed paths over two
exact length groups, including a per-byte ERANGE group and a scalar leftover.
Complete postfilter records and offsets matched byte-for-byte. The packed path
executed three quartets (12 candidates) plus one scalar candidate.

## Performance gate

Full H200 job `1182771` reproduced the established TSV exactly: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines. All 83 chunks used packed arithmetic.
They executed 6,013,946 quartets containing 24,055,784 candidates and kept
16,601,562 scalar leftovers; together these exactly equal the 40,657,346
compacted full-MSV rows.

The request completed in 454.006856 seconds: 346.997314 seconds generation,
446.517924 seconds pipeline wall, 399.448593 seconds continuation/output, and
300.075803 seconds overlap. This is 1.781588 seconds (0.391%) faster than the
unpacked Phase 7 compaction run and 0.240634 seconds (0.053%) faster than Phase
6. The difference is too small to claim a material end-to-end speedup from one
run, but it is exact and introduces no measured regression, so the packed path
is retained while Phase 7 proceeds to the much larger Viterbi workload.

Evidence lives under
`build/h200-phase7-packed-msv-20260824/attempt-01-full/runs/h200-full`.
`worker.json` SHA-256 is
`53f958666c3eaeee3dcd9b19ace58e831b5dc235a2f0b57e8721a0295f5218e9`;
`raw-validation.json` SHA-256 is
`1615f9180674b669aae26ec1536bef9e197ba01cc522fe13f7fbd23df5b00c4d`.
