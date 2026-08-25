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
| 3 | Device-resident Forward -> Backward/domain -> rescore chain | partial; validated residency slices retained | through `171d544` | jobs 1182690/1182713 exact; redundant Forward and region replay removed, compiled F3 authoritative |
| 4 | Forward/Backward/rescore candidates-per-warp variants | complete; automatic promotion rejected | through `a8932dd` | exact, but full job 1182718 regressed 0.72%; production restored to width 1 |
| 5 | Profile-axis packed SSV | complete; retained | through `9730f39` | exact job 1182734: 455.026 s, 14.98% faster than retained Phase 3 |
| 6 | Length-cohort decision metadata | complete; retained | through `cfd756c` | exact job 1182743: 454.247 s; transition H2D reduced 24.92 MB -> 0.12 MB |
| 7 | Packed integer MSV/Viterbi | MSV retained; Viterbi rejected | through `f5ac24a` | packed MSV exact/flat at 454.007 s; packed Viterbi exact but slower and larger, so excluded from `main` |
| 8 | Certified reduced-alphabet F0 | complete; production rejected | through `913be5d` | exact job 1182800: best codebook certified only 0.145% of cells while F0 cost about 1.9x exact packed F1 |
| 9 | GA-specialized certified pruning | complete; retained for gathering cutoffs | through `c5eec74` | exact job 1182813: 449.104 s, 1.08% faster than prior best, 117,545 target rows certified |
| 10 | Mandatory-seed/global-profile index | complete; production rejected | `786d1a6` | exact job 1183467: zero false rejects, but best bound certified only 0.494% of cells and cost 5.85x exact F1 |
| 11 | Internal GPU execution policy | complete; retained | `2fc2a92` | exact six-shape H200 job 1183478: large-by-large auto policy was 9.871% faster than simple with 2.231% more persistent device memory |

## Phase 0 design and result

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

## Phase 1A design and result

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
non-regressing infrastructure.

## Phase 7 packed Viterbi result: rejected

Job `1182783` reproduced the exact 39,010,327-byte output, but it completed in
454.963 seconds versus 454.007 seconds for packed full MSV: 0.957 seconds
(0.211%) slower. Peak sampled H200 memory remained 3,390 MiB, while maximum
RSS increased by 94,836 KiB to 13,340,692 KiB. The packed path handled
191,687,950 candidates and left 11,983,159 scalar, but it also uploaded
814,684,436 bytes of execution-index metadata.

The experiment is rejected. Its implementation commit, `61c3545`, is kept on
an isolated branch for evidence and is deliberately not an ancestor of
`main`. Phase 8 starts from retained commit `f5ac24a`, before packed Viterbi.
The immutable run evidence is under
`build/h200-phase7-packed-viterbi-20260824/attempt-03-full/runs/h200-full`.

## Phase 9 certified GA-pruning result

The source proof established a conservative final-score upper bound only
after isolated-domain Forward scores are known. Full H200 job `1182813`
reproduced the exact 39,010,327-byte output and certified 117,545 target rows /
203,880 regions below their gathering cutoffs. Request wall fell from the
previous best 454.007 seconds to 449.104 seconds (1.080%); generation fell to
340.105 seconds and continuation/output to 398.199 seconds. Compact attempts
fell 91.61%, the sparse packet shrank 18.15%, maximum RSS fell 0.65%, and peak
H200 memory remained effectively flat. The path is retained for compatible
gathering-cutoff searches; the ordinary path remains the non-GA and audit
fallback.

## Phase 10 mandatory-seed/global-index result: rejected

The exact SSV source yields a formal integer certificate: any F1 survivor or
SSV-uncertain pair must contain a contiguous diagonal segment whose gain
reaches a compiled threshold. Partitioning that segment into bounded words
therefore guarantees one word with a profile/length-specific minimum gain.
The offline evaluator exhaustively retained all exact candidates for maximum
word lengths 1/2/4/8/16/32.

H200 job `1183467` evaluated all 27,481 Pfam profiles against the first 1,000
targets. Every arm had zero false rejects, but the best 32-residue arm
certified only 0.493672% of logical SSV cells while its scan took 3.977 seconds
versus 0.680 seconds for packed exact F1. All 16 sampled profile dictionaries
hit their enumeration caps; their incomplete payload extrapolates to at least
1.47 billion associations / 65.6 GiB across Pfam before index overhead.

Production integration is rejected. The exact proof and evaluator are retained
as evidence; no index build or scan cost is imposed on any workload.

## Phase 11 deterministic execution-policy result

Every `SequenceBatch` now owns an immutable `auto`, `simple`, or diagnostic
`throughput` GPU policy.  `auto` chooses only retained exact algorithms using
small, inspectable workload-shape thresholds; hardware selection remains the
caller's responsibility.  Rejected Forward widths, packed Viterbi, F0, and
mandatory-seed experiments cannot be selected.  Force-policy controls and the
unchanged scalar/query-major route remain available for audit and small work.

H200 job `1183478` ran the required tiny, one-profile/large-target,
ten-profile/large-target, hundred-profile/medium-target,
large-profile/small-target, and large-by-large matrix.  All three policies
produced identical complete postfilter records/offsets and identical HMMER
target/domain output on the checked boundary profiles.  Automatic route
counters matched every declared threshold and kept Forward at width 1.

Warm large-by-large generation was 240.914 ms under `auto` versus 267.299 ms
under `simple`, a 9.871% improvement, while persistent H200 allocation rose
8,290,992 bytes (2.231%).  One- and ten-profile cases were effectively flat
with only eight additional bytes.  The 512-profile x 16-target case was also
flat in runtime and paid a bounded 3.19 MiB packed-profile cache cost.  Phase
11 is retained.  Exact cold/warm timings and memory for all 18 arms are in
`PHASE11_EXECUTION_POLICY.md`.

Full H200 job `1183483` confirmed exact production behavior across the full
Pfam workload.  Request wall was 451.083 seconds, 0.441% slower than the Phase
9 best, while sampled H200 memory fell 22 MiB to 3,370 MiB.  This is a flat
end-to-end result rather than an additional speedup; Phase 9 remains the best
full-workload measurement.  The policy is retained because it chooses among
validated GPU algorithms without burdening the one/few-profile paths.

## Roadmap status

All numbered implementation and research phases have now produced an exact
oracle result and an explicit retain/reject decision.  This closes the numbered
experiment sequence, but not the primary device-resident architecture: Phase
3 remains deliberately marked partial in the ledger.  Production still
downloads full postfilter results before Forward selection, preserves host
Forward and Backward materializations for the sparse journal, and allocates
Backward/rescore workspaces per generation.  Those remaining host boundaries
must be removed or explicitly rejected by measurement before the overall
engine objective can be called complete.

The retained production line contains only strategies that preserved HMMER
semantics and survived their stated gates; rejected strategies remain
audit-only or isolated.  The next engineering work returns to the unfinished
Phase 3 boundary while retaining the Phase 11 policy as its workload-shape
dispatcher.

Every retained full-workload result will continue to report exact output
identity, request/generation/continuation/overlap timing, peak H200 memory,
maximum RSS, and material workspace or transfer changes. Flat runtime with a
real memory reduction is a positive result and will be recorded as such.
