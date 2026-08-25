# Phase 10 certified mandatory-seed experiment

## Hypothesis

For a fixed profile set, a single target-word scan may identify the small set
of profile/target pairs that can possibly pass exact SSV F1.  False positives
are harmless; a false negative is forbidden.  This phase is an offline
research evaluator and does not alter production dispatch.

## Exact SSV gain certificate

On one SSV diagonal, let `c(k,a)` be HMMER's signed byte cost and let
`g(k,a) = -c(k,a)`.  Before HMMER's existing overflow/J-state guards fire, the
CUDA and SSE recurrence is

```text
v(0) = -128
v(t) = max(-128, v(t-1) + g(t))
```

Consequently `v(t)+128` is the maximum integer gain of a contiguous suffix
ending at `t`, and the maximum SSV byte is `128` plus the maximum contiguous
diagonal gain.  For each `(profile,target-length)` cohort the evaluator replays
the exact unsigned SSV postprocessing for every possible pre-wrap gain
`0..127`, including raw overflow, adjusted overflow, J-state uncertainty, and
the existing compiled F1 cutoff.  The first gain that HMMER cannot definitely
reject is `R`.  Invalid constants, unsafe transition sums, nonmonotone
decisions, and `R=0` fail open to exact SSV.

Thus, for a supported cohort:

```text
HMMER requires downstream work  =>  some diagonal segment has gain >= R.
```

## Mandatory block theorem

Partition the model into `B` nonempty contiguous blocks.  Choose positive
integer quotas `alpha[b]` such that

```text
sum(alpha[b] - 1 for b in blocks) = R - 1.
```

Any diagonal segment is partitioned by its intersections with these blocks.
If every intersection had gain below its block quota, integer scores would
make its total at most `R-1`, contradicting a total of at least `R`.
Therefore at least one block intersection has gain at least its quota.

That intersection contains a first prefix that reaches the quota.  The
evaluator enumerates every such minimal threshold-crossing residue word from
every model start within each block.  Every F1-relevant target must contain at
least one enumerated word.  The word maps to the original profile and model
interval; scanning may emit extra profile/sequence/diagonal candidates but may
not omit a true one.

For a bounded global word index, the evaluator also uses a stronger practical
form that does not fix model-block boundaries.  Split the unknown passing
diagonal segment into consecutive chunks of at most `K` residues.  It has at
most

```text
Q = ceil(min(model_length, target_length) / K)
```

chunks, so one chunk must score at least `ceil(R/Q)`.  Indexing every model
word of length `1..K` that reaches this threshold is therefore also exact.
The census evaluates several `K` values; small thresholds are expected to
produce common words and are an explicit negative-result condition.

Only the 20 canonical protein residues are indexed initially.  A target with
any degenerate/noncanonical code is unresolved and runs exact SSV.  Enumeration
limits likewise fail open for the entire profile/length cohort.

## Evaluator gate

The experiment will measure block counts 2/4/8 and conservative enumeration
caps over real Pfam profiles and target-length cohorts.  It records:

- exact-pass/seed-miss count (must be zero);
- dictionary words and word/profile/offset associations;
- trie/index memory estimate and build wall;
- target-word occurrences and emitted profile/sequence candidates;
- fraction of the full profile/sequence and exact-SSV cell space retained;
- noncanonical targets and capped/unsupported profiles routed to exact SSV.

Production integration requires zero false negatives and a convincing
reduction after index-build, scan, candidate deduplication, and exact-SSV costs.
If the mandatory dictionary or emitted candidate set is too large, record the
negative result and leave the production path unchanged.
