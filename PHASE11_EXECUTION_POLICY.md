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

## H200 policy matrix

Job `1183478` ran the six required shapes on an attested H200 from clean
benchmark revision `2ca7a3e4705077707656e0a39eb6cf281bcebf63`.  The loaded
native and pipeline module SHA-256 values were respectively
`4a64ff8dd5d491f0ea0282959ef8041d96e4b4efca82367b06a2500cf8ee623f`
and `5a7c2e5a28b799ccefe0ca4dd80fc543e94baa279beae5a2531b984193b7f51a`.
An untimed request initialized CUDA before measurement.  Times below are
cold-per-batch and median-of-three warm generation milliseconds; memory is
the exact persistent device allocation after the measured batch.

| Shape | `auto` cold / warm ms | `simple` cold / warm ms | `throughput` cold / warm ms | Persistent bytes (`auto` / `simple` / `throughput`) |
|---|---:|---:|---:|---:|
| 1 profile x 1 target | 0.123 / 0.049 | 0.108 / 0.051 | 0.140 / 0.052 | 3,662 / 3,662 / 3,663 |
| 1 x 4,096 | 3.357 / 3.110 | 3.938 / 3.145 | 3.488 / 3.088 | 2,999,626 / 2,999,618 / 3,013,962 |
| 10 x 4,096 | 7.017 / 5.969 | 6.319 / 5.980 | 7.223 / 6.113 | 21,225,230 / 21,225,222 / 21,370,830 |
| 100 x 512 | 11.720 / 10.451 | 10.928 / 10.468 | 10.580 / 10.260 | 38,492,247 / 37,989,623 / 38,621,015 |
| 512 x 16 | 8.407 / 7.574 | 10.193 / 7.545 | 8.100 / 7.477 | 14,065,757 / 10,723,645 / 14,085,589 |
| 512 x 4,096 | 274.265 / 240.914 | 308.351 / 267.299 | 278.225 / 240.068 | 379,930,221 / 371,639,229 / 379,930,221 |

For every shape, all three policies produced byte-identical complete
postfilter records and offsets.  The first and last profile also produced
byte-identical HMMER target and domain output, so the complete order-sensitive
input to continuation and representative final output both agree.  Exact
counters proved the intended dispatch: `auto` packed profiles only at 32+
profiles, compacted length metadata only at 256+ sufficiently repetitive
targets, compacted full-MSV work only above 65,536 retained candidates, and
used Forward width 1 everywhere.

On the large-by-large shape, `auto` was 9.871% faster than `simple` when warm
and used 8,290,992 more persistent bytes (2.231%).  The one- and ten-profile
shapes were effectively flat with only eight extra bytes under `auto`.
The 512-profile x 16-target shape was also flat in runtime (-0.382% relative
to `simple`) but used 3,342,112 extra bytes; this is a bounded 3.19 MiB cost
for the cached packed-profile view.  Forced `throughput` shows why the policy
is necessary: it activates representations even when their setup is not
justified, while `auto` preserves the simple path.

The result JSON is
`build/phase11-execution-policy-h200-20260825/attempt-03/result.json`, SHA-256
`337a225ec8e0faaf8252eebcf1d9d2c6df1514f762aa87ff8e748ed0d06a2ca4`.

## Full-workload confirmation

Job `1183483` then ran the unchanged production `auto` policy across all
27,481 Pfam profiles and 300,186 targets.  The output was byte-identical to
the CPU64 and Phase 9 oracle: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.  Request wall was 451.083043 seconds,
with 340.362131 seconds of generation, 400.407954 seconds of continuation,
297.237152 seconds of overlap, and 443.658796 seconds of pipeline wall.

Against the Phase 9 best, request wall was 1.978931 seconds (0.441%) slower.
This is an effectively flat single-run result and is not claimed as an
end-to-end speedup.  Maximum RSS rose 220,652 KiB (1.677%) to 13,380,000 KiB,
while peak sampled H200 memory fell 22 MiB to 3,370 MiB.  Thus the policy's
large-shape microbenchmark win does not materially change this full workload;
its production value is deterministic selection and small-workload protection.

The immutable run is under
`build/h200-phase11-policy-full-20260825/attempt-03-full/runs/h200-full`.
The worker and raw-validation SHA-256 values are respectively
`97a86813e92dc0404b2bd9efca9bfe77eb2b009166520c1012ccdb897a7ea0e7`
and `a0dcdb393aea7f6f598e0931ed24eeb549b54b7ee36a4839c8689c35ad5a0d3f`;
both report `PASS_FOR_EXECUTION`, and the artifact manifest verifies.

## Decision

Retain policy version 1.  It is deterministic, request-scoped, observable,
exact across the required workload matrix and full workload, materially
faster than the simple route for the isolated large-by-large shape, and does
not burden one/few-profile searches.  Its full-workload runtime is flat rather
than improved.  Rejected algorithms remain unreachable from every production
policy.
