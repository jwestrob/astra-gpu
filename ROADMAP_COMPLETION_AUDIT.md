# Plan7 GPU roadmap completion audit

This audit applies the original correctness-first roadmap to the current
`main` line.  A phase is complete only when current source plus a matching
oracle or benchmark proves its stated requirement; the existence of a commit
or a plausible design is not enough.

## Overall verdict

The numbered Phase 0--11 experiment sequence is complete, and every experiment
has an explicit exact retain/reject decision.  The broader heterogeneous Plan7
engine objective is **not yet complete** because the original Phase 3
device-resident-chain requirement remains only partially implemented.

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
| Phase 3 fully device-resident postfilter-to-final-egress chain and persistent chunk workspace | **incomplete** | Current production still performs postfilter result D2H, Forward input/result transfers, host journal materialization, and per-generation Backward/rescore allocation.  The Phase 3 notes explicitly preserve these boundaries. |
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
- Phase 11 full Pfam x metagenome job `1183483`: exact output SHA-256
  `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
  39,010,327 bytes, 383,235 lines; 451.083043 seconds request wall; worker and
  raw-validation status `PASS_FOR_EXECUTION`; complete artifact manifest
  rehash passed.

## Remaining completion gate

The next architectural work returns to Phase 3.  Completion requires all of
the following while preserving the existing dense audit path:

1. keep postfilter/F2/Forward survivor state resident through Backward and
   rescore in production, copying only semantically necessary exception/final
   payloads;
2. replace per-generation Backward/rescore allocation with a bounded,
   exactly-accounted reusable workspace or record a measured rejection;
3. demonstrate that targeted D2H/H2D and host-materialization counters fall
   rather than move to another stage;
4. pass the exact focused H200 oracle, first-1,000 gate, six-shape workload
   matrix, and full-output SHA comparison.

Until those gates are proved or explicitly rejected by a faithful experiment,
the active implementation goal remains open.
