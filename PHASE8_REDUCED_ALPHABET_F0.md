# Phase 8 certified reduced-alphabet F0 evaluator

## Hypothesis

An inexpensive SSV recurrence over a reduced residue alphabet may certify
that many profile/target pairs cannot pass exact F1. Only certified rejects may
be discarded; every unresolved pair still executes the unchanged exact SSV.
This commit is an offline evaluator and does not alter production dispatch.

## Certificate

For model position `k` and residue class `C`, the evaluator defines

```text
coarse_cost(k, C) = min(exact_cost(k, residue) for residue in C)
```

Every sequence residue belongs to exactly one nonempty class. HMMER's packed
signed-byte score invariant is checked explicitly: every exact cost is at
least `-bias`. Signed saturating subtraction is monotone in its prior state and
antitone in its cost, so the coarse state is never below the exact state on
the same diagonal.

The `cost >= -bias` bound also closes the signed-byte wrap case. A path cannot
cross from the negative score range to a nonnegative byte without first
reaching the existing `255-bias` overflow guard; that outcome is retained for
exact SSV. Unsafe transition constants are likewise retained. Therefore only
a finite, ordinary coarse result below the unchanged exact F1 threshold is a
certificate of rejection.

The evaluator independently produces the exact F1 mask, reports any
`exact-pass/coarse-reject` pair as a false reject, and exposes per-profile and
aggregate pair/cell counts, table and temporary device bytes, and exact/coarse
timings. A singleton 29-class partition must reproduce the exact mask and is
the indexing/arithmetic oracle.

## Decision gate

Evaluate fixed 4-, 6-, and 8-class partitions, then a deterministic codebook
of up to 32 eight-class partitions with the tightest result selected per
profile. Record:

- zero false rejects;
- certified reject and remaining exact-SSV pair/cell fractions;
- F0 table, temporary HBM, and process RSS costs;
- exact-F1 and F0 build/upload/kernel timing;
- per-profile Pareto/selectivity distribution.

Production integration requires the measured exact work avoided to exceed the
F0 work added by a comfortable margin. A weak bound is a negative result even
when it is correct; it must not be contorted into the hot path.

## H200 result and decision

Job `1182800` evaluated all 27,481 Pfam profiles against the immutable first
1,000-target set (27,481,000 logical pairs and 1,182,809,892,909 logical SSV
cells) at clean revision `9a1106d`. The singleton 29-class oracle reproduced
the exact F1 mask: 879,857 candidates and zero false rejects.

The reduced bounds were exact but extremely loose:

| Partition | Certified pairs | Pair fraction | Certified cells | Cell fraction | F0 kernel | Exact packed F1 generation |
|---|---:|---:|---:|---:|---:|---:|
| 4 classes | 650 | 0.00237% | 1,114,098 | 0.000094% | 1.297 s | 0.649 s |
| 6 classes | 23,680 | 0.0862% | 122,207,040 | 0.0103% | 1.297 s | 0.705 s |
| 8 classes | 252,167 | 0.9176% | 1,670,500,269 | 0.1412% | 1.297 s | 0.678 s |

A deterministic 32-partition eight-class codebook, taking the tightest result
per profile, certified only 0.9575% of pairs and 0.1450% of cells. Its mean F0
kernel time was 1.297 seconds. Even granting a free profile-specific dispatch,
the bound leaves 99.855% of exact cell work in place. For the fixed eight-class
case, measured F0 plus proportional survivor work is approximately 2.91 times
the existing exact packed F1 time, before table build or upload.

The eight-class evaluator adds 38,510,549 bytes (36.7 MiB) of temporary device
storage. The diagnostic process reached 2,025,028 KiB maximum RSS; the exact
F1 batch held 259,069,973 persistent device bytes. No production memory is
saved because the exact profile representation remains necessary for nearly
all pairs.

Decision: **reject production F0 integration**. Retain only this offline
certificate evaluator and its negative evidence. Phase 8 is complete; do not
spend another optimization cycle packing or tuning a bound whose selectivity
misses break-even by orders of magnitude.

Evidence:

- result: `build/phase8-f0-evaluator-h200-20260824/attempt-02/result.json`,
  SHA-256 `071822c93880ad98a190e0fdd2593c14c514acb786c1968b41a5b1dd7afc960d`;
- stdout: SHA-256
  `76d43cbc4e0ceeaf39ed2301bb852eab12f4447993050baeb352a96fcfe90584`;
- stderr: empty, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- native module: SHA-256
  `c71dc25b88a7cb2bbf62f9bb415189c496db0f2e45ef623948e6883692b6976b`.

Job `1182799` completed the same CUDA evaluation but failed afterward while
serializing a text-valued profile name. Commit `9a1106d` fixed only that report
writer; it did not alter the evaluator or rerun inputs.
