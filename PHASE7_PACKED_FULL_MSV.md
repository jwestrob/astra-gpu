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

The full sealed workload remains the deciding gate. It must reproduce the
established 39,010,327-byte TSV exactly and report the packed/scalar partition
for every chunk. Full-run timing and the retain/reject decision are pending.
