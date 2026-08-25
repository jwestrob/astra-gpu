# Phase 11: deterministic internal GPU execution policy

## Objective

Choose among the exact GPU algorithms retained by Phases 2–9 without routing
work to the CPU, changing HMMER semantics, or imposing large-workload setup on
small searches.  Rejected experiments are never eligible.

## Version-1 policy

Each `SequenceBatch` owns one immutable policy selected at construction:

- `auto` applies the measured deterministic thresholds below;
- `simple` forces the reference scalar/query-major algorithms;
- `throughput` is a debug/benchmark control that requests every retained
  packed or compact algorithm where its representation is valid.

The policy is request-scoped.  Concurrent batches can use different policies;
the production path no longer needs process-global environment mutation to
force a coherent strategy.  The older per-stage environment controls remain
available for narrow legacy audits.

Automatic decisions are deliberately small and inspectable:

| Stage | `auto` decision | `simple` | `throughput` |
|---|---|---|---|
| F1 profile axis | pack length-compatible quartets when at least 32 profiles are selected | scalar | pack compatible quartets when at least four profiles are selected |
| F1 length metadata | compact at 256+ targets when distinct lengths are at most half the target count | expanded | compact whenever the device representation is valid |
| full-MSV launch | stable device compaction at 65,536+ retained candidates | legacy direct launch | stable device compaction within the uint32 index domain |
| full-MSV arithmetic | packed exact quartets after launch compaction | scalar | packed exact quartets after launch compaction |
| Forward | one four-lane candidate per warp | width 1 | width 1 |

Forward widths 2/4/8 are excluded because the exact full-workload benchmark
rejected automatic promotion.  Certified F0, mandatory seeds, packed Viterbi,
and every other rejected experiment are likewise absent.  GA pruning remains
an explicit semantic option because it is legal only under gathering cutoffs.

`SequenceBatch.execution_policy_statistics` reports the requested policy,
target and length-class shape, actual packed/compact/legacy route counts, and
the fixed Forward width.  Native workspace statistics expose the same
versioned counters for benchmark harnesses.

## Acceptance

1. `auto`, `simple`, and `throughput` must produce byte-identical complete
   postfilter records, offsets, and final HMMER target/domain output.
2. The six-shape matrix must cover tiny, one-profile/large-target,
   ten-profile/large-target, hundred-profile/medium-target,
   large-profile/small-target, and large-by-large workloads.
3. `auto` must retain scalar/query-major execution for small profile sets and
   avoid compact length metadata for small target sets.
4. Every forced strategy must be visible in exact counters; Forward must stay
   width 1.
5. Report cold and warm generation time plus persistent H200 allocation for
   every shape and policy.  The default full-workload output and established
   performance must remain unchanged.

H200 results and the final retain/reject decision will be appended after the
matrix completes.
