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

The CUDA census includes the complete 29-code HMMER amino alphabet. The
dictionary-size probe deliberately uses only the 20 canonical residues, so it
is a lower bound; a production index would also need explicit degenerate/stop
handling or would fail those affected targets open to exact SSV. Enumeration
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

## H200 result and decision

Job `1183467` evaluated word lengths 1/2/4/8/16/32 across all 27,481 Pfam
profiles and the immutable first 1,000 targets (27,481,000 logical pairs and
1,182,809,892,909 logical SSV cells). Every arm retained all 879,857 exact F1
candidates with zero false rejects and zero unsupported pairs, proving the
compiled threshold and bounded-window certificate on the real workload.

The bound is too weak to use:

| Maximum word | Certified pairs | Pair fraction | Certified cells | Cell fraction | Seed kernel | Exact packed F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0% | 0 | 0% | 1.255 s | 0.756 s |
| 2 | 0 | 0% | 0 | 0% | 1.418 s | 0.652 s |
| 4 | 130 | 0.000473% | 122,321 | 0.000010% | 1.840 s | 0.652 s |
| 8 | 6,451 | 0.023474% | 12,552,237 | 0.001061% | 1.893 s | 0.680 s |
| 16 | 89,216 | 0.324646% | 307,230,432 | 0.025975% | 2.696 s | 0.689 s |
| 32 | 870,272 | 3.166813% | 5,839,203,942 | 0.493672% | 3.977 s | 0.680 s |

Even the best 32-residue condition leaves 99.506% of exact SSV cells and its
diagnostic recurrence costs 5.85 times the packed exact F1 kernel. The global
dictionary is worse: every one of 16 evenly sampled profiles hit the 100,000
association or 1,000,000 enumeration-node cap. Those incomplete samples
already contain 855,218 entries and 41,007,823 bytes of minimum word/metadata
payload. Linear extrapolation is a lower bound of 1.47 billion entries and
65.6 GiB for Pfam, before trie/hash overhead and before completing any capped
profile.

Decision: **reject production mandatory-seed/global-index integration**. Keep
the proof and evaluator as negative research evidence. Phase 10 is complete;
Phase 11 may use only the GPU strategies already validated and retained.

Evidence: `build/phase10-mandatory-seed-h200-20260825/attempt-01/result.json`,
SHA-256 `d3ee2d8b1db0f62c78c0af0ba268829d33c3b661d62e2e499b8ca98d8c1ce61d`;
job `1183467`, `COMPLETED 0:0`, empty stderr.
