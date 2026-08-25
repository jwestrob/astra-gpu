# Plan7 GPU roadmap completion audit

This audit applies the original correctness-first roadmap to the current
`main` line.  A phase is complete only when current source plus a matching
oracle or benchmark proves its stated requirement; the existence of a commit
or a plausible design is not enough.

## Overall verdict

The numbered Phase 0--11 sequence and the subsequent bottleneck-focused Phase
3 work are complete under the roadmap's measured stop condition. Every
retained change and every rejected tail experiment has an exact full-workload
result. The retained implementation remains job `1183504` at 448.140781
seconds with byte-identical HMMER/Astra output.

The aspirational single-copy, fully opaque device-egress design is not claimed
as literally complete: host-visible audit and sparse-egress representations
remain. The active objective nevertheless closes because full exact experiments
show that the scoped cap, threshold, and multidomain reductions all make the
critical path materially slower.

## Requirement audit

| Requirement | Verdict | Authoritative evidence |
|---|---|---|
| Preserve byte-identical HMMER/Astra output and exact continuation semantics | proved for every retained full-workload change | Full-output SHA-256 `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`; Phase 1A semantic fingerprints and dense/sparse H200 dual oracle; phase-specific records in `PERFORMANCE.md` |
| No fast math, reduced precision, or floating-point reassociation | proved for retained source | Build/source audit finds no `--use_fast_math` or `-ffast-math`; exact F3 and packed-integer notes record operation-order and boundary oracles |
| Preserve original logical ordinals and stable continuation order | proved for retained transformations | Journal-v3 typed ordinals and source-domain indexes; Phase 2 stable word/bit scatter; Phase 5 original mask positions; exact final output after every retained full run |
| Keep reference/audit paths | proved | Dense journal v2, sparse-v3 dual consumer, `simple` policy, F3 audit mode, and per-stage force controls remain present |
| Phase 0 work/fallback telemetry | complete | `PHASE0_ROUTE_TELEMETRY.md`; H200 jobs `1182345` and `1182355`; immutable per-profile report and exact fallback census |
| Phase 1A host-side sparse journal v3 and semantic fingerprint | complete; production form rejected | `PHASE1A_*`; authenticated H200 dual oracle; full job `1182349` exact but 3.82% slower, so dense-to-sparse replay remains audit-only |
| Phase 1B sparse generation before dense-v2 materialization | complete; retained opt-in | `PHASE1B_DIRECT_SPARSE_V3.md`; exact full job `1182391`; zero dense-v2 allocation/retention, one source scan per chunk, 536.168 s |
| Phase 2 device-side stable F1 compaction | complete; retained | `PHASE2_DEVICE_COMPACTION.md`; full job `1182619`; 83 device compactions, zero host expansions and candidate uploads |
| Phase 3 exact F3 decision compilation | complete; retained | `PHASE3_EXACT_F3_THRESHOLD.md`; exhaustive/adversarial boundary oracle and exact full job `1182713` |
| Phase 3 Forward-to-Backward and Backward-to-rescore residency | complete for the two implemented handoffs | `PHASE3_FORWARD_BACKWARD_RESIDENCY.md`, `PHASE3_BACKWARD_RESCORE_RESIDENCY.md`; full jobs `1182690` and `1182713`; zero legacy handoff H2D and zero fallbacks |
| Phase 3 fully opaque postfilter-to-final-egress transport and persistent chunk workspace | partially realized; closed by measured stop condition | Resident F2-to-Forward, Forward-to-Backward, and Backward-to-rescore handoffs are retained. Host-visible audit/egress materialization remains. Exact jobs `1183518`, `1183521`, `1183524`, and `1183528` reduced every dominant measured tail route but regressed request wall by 7.005%--8.494%, so no remaining scoped change is promoted. |
| Phase 4 candidates-per-warp experiment | complete; production promotion rejected | `PHASE4_FORWARD_SUBWARPS.md`; exact full job `1182718` regressed 0.72%, so production is width 1 |
| Phase 5 profile-axis packed SSV | complete; retained | `PHASE5_PROFILE_PACKED_SSV.md`; exact job `1182734`, 455.026 s and 14.982% faster than retained Phase 3 |
| Phase 6 length-cohort/compiled target metadata | complete for the evidence-supported representation | `PHASE6_LENGTH_CLASS_METADATA.md`; exact job `1182743`; transition H2D reduced from 24,915,438 to 121,595 bytes without physical sequence reordering |
| Phase 7 packed MSV/Viterbi | complete decision | Exact packed MSV retained at job `1182771`; packed Viterbi exact but slower/larger at job `1182783`, therefore excluded from `main` |
| Phase 8 certified reduced-alphabet F0 | complete; production rejected | `PHASE8_REDUCED_ALPHABET_F0.md`; H200 job `1182800`, zero false rejects but only 0.145% certified and about 1.9x exact-F1 cost |
| Phase 9 GA-specialized certified pruning | complete; retained | `PHASE9_GA_PRUNING.md`; exact full job `1182813`, 449.104 s, lower RSS and 18.15% smaller sparse packet |
| Phase 10 mandatory-seed/global-profile index | complete; production rejected | `PHASE10_MANDATORY_SEED.md`; H200 job `1183467`, zero false rejects but 0.494% certified, 5.85x cost, at least 65.6 GiB index payload |
| Phase 11 deterministic internal GPU policy and small-workload protection | complete; retained | `PHASE11_EXECUTION_POLICY.md`; clean H200 matrix job `1183478`, exact records/output across all policy modes and six shapes |

## Current regression evidence

- Clean H200 Phase 11 matrix result:
  `build/phase11-execution-policy-h200-20260825/attempt-03/result.json`,
  SHA-256
  `337a225ec8e0faaf8252eebcf1d9d2c6df1514f762aa87ff8e748ed0d06a2ca4`.
- Current CUDA-hidden suite: 356 tests passed, with 123 expected GPU skips,
  under private PyHMMER ABI
  `d4867ff865e9b8a7acdbbf9106e3d7e1223336d374cb0f46d7e352427b990689`.
- Retained full Pfam x metagenome job `1183504`: exact output SHA-256
  `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
  39,010,327 bytes, 383,235 lines; 448.140781 seconds request wall;
  333.993605 seconds generation; 400.083104 seconds continuation/output;
  293.693784 seconds overlap.
- Exact rejected full runs: external multidomain job `1183518` at 480.258523
  seconds, bounded Backward job `1183521` at 479.533852 seconds, bounded
  Backward/rescore job `1183524` at 482.477600 seconds, and zero-threshold
  upper-bound job `1183528` at 486.204557 seconds. All four reproduced the
  same output SHA, byte count, and line count.

## Goal disposition

There is no remaining gate under the current bottleneck-focused objective.
The retained computational handoffs passed the exact full oracle, and faithful
full experiments explicitly rejected the remaining scoped route/cap changes.
The production line therefore stops at `348f277` plus its documentation.

Future work should begin as a new objective only if it proposes a materially
different exact domain/rescore/multidomain algorithm. Merely increasing caps,
resetting them across waves, removing the threshold guard, or rerouting the
existing CPU semantics has already been measured and rejected. Consolidated
timing, route, memory, branch, and evidence details are in
`CPU_CONTINUATION_TAIL_RESULTS.md`.
