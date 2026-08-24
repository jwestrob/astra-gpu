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
