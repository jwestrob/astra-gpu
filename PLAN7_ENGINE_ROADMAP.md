# Exact Plan7 GPU engine roadmap

This is the durable engineering record for the correctness-first Plan7 GPU
rewrite.  It records what was attempted, the exact source revision, the oracle
used, the evidence produced, and whether each optimization was retained.

## Governing baseline and oracle

- Upstream baseline: `5ffe5d43cf4177493b72f24b0fb96c00276c96ab`
  (`Add benchmark performance evidence`).
- Native implementation immediately below that evidence commit:
  `614161a4d24c05564b863a0d1b67f0b2f26aeaf1`.
- Existing full-workload GPU/CPU Astra output:
  `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`
  (39,010,327 bytes; 383,235 lines including the header).
- The acceptance target is exact existing HMMER/Astra semantics, not merely
  agreement with a performance baseline.  No optimization may introduce a
  false negative, alter an order-sensitive continuation, or weaken the final
  output oracle.

The original dense continuation and simple/query-major GPU implementations
remain available as reference/audit paths until their replacements have passed
the complete oracle suite and representative end-to-end workloads.

## Non-negotiable implementation rules

1. Preserve HMMER floating-point operation order where it is semantic.  No
   fast-math, reassociation, contraction, reduced precision, or approximate
   filter promotion.
2. Preserve original profile and sequence ordinals across any internal
   permutation and restore them before order-sensitive CPU continuation.
3. Make one architectural change per commit.  Each change carries tests,
   counters, a focused benchmark, and a written retain/reject decision.
4. Compare new behavior to the old path through an explicit oracle or dual
   execution mode before switching production behavior.
5. Keep small and one-off workloads on a valid low-overhead GPU strategy; do
   not optimize only the large-Pfam/large-target shape.
6. Hardware selection remains the caller's decision.  An eventual internal
   policy may choose only among validated GPU algorithms.

## Phase ledger

| Phase | Scope | State | Source commit | Evidence / decision |
|---|---|---|---|---|
| 0 | Per-profile/chunk work telemetry and real CPU-fallback taxonomy | complete | through `f9ef356` | jobs 1182345/1182355 PASS; work and matrix caps dominate CPU fallback |
| 1A | Host-side sparse-accounting journal v3 plus dense/sparse dual oracle | complete; production performance rejected | through `039c092` | exact full job 1182349 was 3.82% slower despite 41.35% fewer packet bytes |
| 1B | Produce sparse certificate before dense host materialization | complete; retained opt-in | through `87727cd` | jobs 1182389/1182391 exact; one-pass wall 536.168 s, 1.84% faster than dense, zero dense-v2 retention |
| 2 | Device-side stable F1 candidate compaction | complete; retained | through `959cdf6` | job 1182619 exact; 83 device compactions, zero host expansions/uploads; no standalone speedup claim |
| 3 | Device-resident Forward -> Backward/domain -> rescore chain | active; first slice retained | through `c24697a` | job 1182690 exact; resident Forward-to-Backward handoff saved 4.546 s versus Phase 2 |
| 4 | Forward/Backward/rescore candidates-per-warp variants | blocked on 3 | pending | pending |
| 5+ | Profile-axis SSV, cohorts, packed integer DP, certified F0/GA/index research | deferred | pending | pending |

## Active work: Phase 0

### Hypothesis

Accurate per-profile and per-chunk work, timing, transport, and fallback-reason
measurements will identify the dominant remaining semantic and computational
paths without changing search behavior.  Detailed collection must be opt-in
and have negligible overhead when disabled.

### Required record

Where the source can attribute them exactly, collect model length, target
residues/cells, survivor counts at each filter stage, postfilter/Forward/
Backward/compact rows, route counts, compact accepts/retries, journal bytes,
native-stage time, and CPU continuation time.  `CPU_REQUIRED` reasons must be
derived from actual branch conditions rather than inferred after aggregation.
The output must support per-profile records, cumulative Pareto summaries,
reason-by-row counts, reason-by-work estimates, and attributable CPU wall time.

### Acceptance

- Detailed collection disabled: no output change and negligible perturbation.
- Detailed collection enabled: counters reconcile exactly with existing batch
  totals and route partitions.
- Every reason is exhaustive, mutually interpretable, and tied to a real
  source transition; unknown cases fail closed into an explicit `other` bucket.
- Unit and CUDA-hidden integration tests pass before any GPU benchmark launch.

## Next work: Phase 1A

### Hypothesis

The dense continuation history can be reduced on the host to an exact
accounting certificate plus sparse rows that genuinely require downstream
continuation, without changing any meaningful `P7_PIPELINE` state or `TopHits`.

### Design constraints

- Introduce continuation-journal ABI v3; do not mutate v2 in place.
- Derive v3 initially from the existing dense host records.  GPU transport is
  unchanged in Phase 1A.
- Encode unambiguous prefix/delta accounting before every exceptional row and
  a tail certificate after the final exception.
- Never pre-account counters that the invoked HMMER continuation seam will
  increment for that same row.
- Enumerate and fingerprint all semantically mutable pipeline state, including
  sequence/residue/database accounting, filter promotions, threshold/reporting
  state, final target-length configuration, reuse state, and compact-tail state.
- Preserve first/last/consecutive exceptions, no-exception profiles, large
  omitted prefixes/tails, all filter-reject levels, simple/multidomain/
  CPU-required rows, compact accept, and compact retry.
- Provide a temporary dual-consumer mode that begins from equivalent cloned
  state and asserts dense-v2 versus sparse-v3 equivalence before returning.

### Acceptance

- Exact `TopHits` and final output.
- Exact semantic pipeline-state fingerprint and all meaningful counters.
- No double counting, missing tail accounting, ordinal drift, or changed final
  target/model/background configuration.
- Material CPU-continuation improvement demonstrated before Phase 1B begins;
  otherwise record the negative result and investigate rather than forcing the
  design forward.

### Current evidence

The host implementation includes journal-v3 planning, checked sparse
certificate execution, a canonical semantic pipeline/TopHits fingerprint, and
an independent dense-v2/sparse-v3 dual oracle. The integrated CUDA-hidden
regression suite passed at revision
`4649a91e4b3fe9ffea1519feca9ff577a79005d0`.

Focused H200 job 1182321 then generated authenticated fused CUDA batches and
exercised real Forward, simple-region, and compact-domain continuation. Dense
v2 and sparse v3 had identical semantic fingerprints and target/domain table
bytes in every profile. This closes the real-route correctness gate, but not
the phase: journal v2 remains the production default until the sparse switch
is implemented and its total wall time is measured honestly.

The explicit production switch is now implemented at revision `9a078c2`.
Focused H200 job 1182344 exercised the reusable seal-owned packet through real
`CandidateBatch.search()` calls and again matched dense v2 exactly for every
profile and all exercised Forward, simple, and compact routes. It remains
opt-in until the representative first-1000 and full-workload benchmark proves
the output and measures total wall time honestly.

Full job 1182349 proved exact output but rejected the present performance
hypothesis. The opt-in took 567.080 seconds versus 546.221 seconds for sealed
dense v2. It removed 4.089 GB (41.35%) of retained continuation payload, but
the dense-to-sparse plan and validation added 28.945 seconds and continuation
wall did not improve. Phase 1A therefore remains an audit/reference path, not
the production default. Phase 1B is justified only as direct sparse generation
that eliminates both the measured conversion overhead and dense replay
retention.

## Later phase order

After Phase 1A is proved: generate v3 before dense materialization; implement
stable device F1 compaction; consume device mappings directly; introduce a
persistent request/chunk workspace; compile exact host-oracle decision
boundaries; retain Forward state into Backward/domain and rescore; emit one
ordered sparse egress packet; only then explore multi-candidate subwarps and
the later certified/profile-axis research paths.

Every completed phase appends its exact commit, tests, workload, output hash,
measurements, and retain/reject decision to this file.

## Phase 2 result

Stable CUB compaction preserved profile-major and sequence order while
eliminating the mask expansion and candidate-mapping re-upload on all 83 full
workload chunks. Full H200 job `1182619` passed the first-1,000 gate and
reproduced the established output SHA-256 exactly (39,010,327 bytes; 383,235
lines). It ran in 538.822 seconds, 0.495% slower than the retained one-pass
sparse-v3 run and 1.354% faster than the original dense baseline. Phase 2 is
therefore retained as required infrastructure for Phase 3, without claiming a
standalone performance improvement.

## Phase 3 Forward-to-Backward residency result

The first Phase 3 slice retained gathered Forward special-state trajectories
on the selected device and passed them directly into Backward/domain while
preserving the existing host classification, provenance, and audit path. Full
H200 job `1182690` reproduced the established output SHA-256 exactly
(39,010,327 bytes; 383,235 lines). All 83 Forward and 83 Backward calls used the
resident route, with zero allocation fallbacks and zero legacy Forward-special
H2D bytes. The path materialized 6,022,020,720 resident bytes and eliminated
3,291,870,792 bytes of redundant H2D traffic.

The request ran in 534.276 seconds, including 464.689 seconds of generation,
526.763 seconds of pipeline wall, and 336.217 seconds of overlap. Compared with
Phase 2, request wall improved by 4.546 seconds (0.844%) and generation wall by
3.554 seconds (0.759%). Forward-special upload time inside Backward fell from
439.13 ms to 49.16 ms, and aggregate Backward wall fell from 26.880 seconds to
24.853 seconds. This slice is retained; Phase 3 remains active for persistent
workspaces and Backward-to-rescore residency.

## Phase 4 Forward subwarp result

Widths 1/2/4/8 were exact in focused H200 tests, but the microbenchmark-derived
automatic policy did not survive the full workload. Job `1182718` reproduced
the exact 39,010,327-byte output while selecting width 2 for all 83 chunks.
Forward kernels took 33.892 seconds versus 32.823 seconds for width 1, and
request wall took 539.052 seconds versus 535.213 seconds (+0.72%). The policy
is rejected and production is restored to width 1. Wider Forward and Backward
variants remain private diagnostics only; Phase 5 proceeds from the stable
width-1 production path.

## Phase 5 profile-axis packed SSV result

The standalone H200 experiment proved exact four-profile signed-byte SSV and
measured 1.45--1.49x kernel speedups for equal or length-sorted profiles, while
also showing that deliberately divergent quartets are harmful. Production
therefore packs only stable length-compatible quartets, retains original
profile ordinals, and sends small selections, unsuitable quartets, and
leftovers through the unchanged scalar kernel.

Focused production job `1182733` proved forced-scalar versus packed equality
for complete postfilter records and final HMMER output. Full job `1182734`
then reproduced the established 39,010,327-byte output exactly over all
8,249,411,466 logical comparisons. It completed in 455.026 seconds versus
535.213 seconds for retained Phase 3, a 14.982% request-wall improvement.
Generation fell 25.520%, from 463.471 to 345.195 seconds. The packed path
covered 27,160 of 27,481 profiles across all 83 chunks, with only 321 scalar
fallback profiles and 3,372 MiB peak sampled H200 memory.

Phase 5 is retained. Phase 6 proceeds from this path and targets exact
length-dependent execution metadata without changing logical sequence order.

## Phase 6 target-length metadata result

`SequenceBatch` now compiles stable target-length classes once and uploads one
class index per target. Fused F1 computes exact `tjb` values over unique
profile-scale by length-class pairs, then expands them on device into the
unchanged dense layout. Small or poorly compressible workloads retain the old
path, and original sequence ordinals never change.

Focused job `1182742` proved forced-expanded versus forced-compact equality.
Full job `1182743` reproduced the exact 39,010,327-byte output and completed in
454.247 seconds, 0.171% faster than Phase 5. Generation improved by 0.738%.
Across all 83 chunks, compact metadata reduced transition H2D from a
counterfactual 24,915,438 bytes to 121,595 bytes. The path is retained.

Phase 7 proceeds from this exact baseline and evaluates packed integer MSV and
Viterbi arithmetic. Production work must compact the rare full-MSV rows before
packing them; it must not add packed work across the complete postfilter stream.

## Phase 7 full-MSV compaction result

Job `1182754` reproduced the exact 39,010,327-byte output. Device compaction
ran for all 83 chunks and reduced 203,671,109 source rows to 40,657,346 actual
full-MSV executions, avoiding 163,013,763 no-op launches. It completed in
455.788 seconds, 0.339% slower than Phase 6. The compactor is retained as the
required exact work-list foundation for packed full-MSV arithmetic, without a
standalone performance claim. The prior 467,289 count referred only to final
MSV-range CPU fallbacks and must not be used as the full-MSV execution census.

## Phase 7 packed full-MSV result

Job `1182771` reproduced the exact 39,010,327-byte output. It packed
24,055,784 compacted full-MSV candidates into 6,013,946 profile-and-length
matched quartets and retained 16,601,562 scalar leftovers. Request wall was
454.007 seconds, 0.391% faster than the unpacked compaction run and effectively
tied with Phase 6. The packed implementation is retained as exact,
non-regressing infrastructure. Phase 7 now targets Viterbi, which executed
202,849,174 rows on the sealed workload and therefore offers substantially
more arithmetic work to densify.
