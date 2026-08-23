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
| 0 | Per-profile/chunk work telemetry and real CPU-fallback taxonomy | complete | through `f9ef356` | source-fact H200 job 1182345 PASS; immutable NFS report exact |
| 1A | Host-side sparse-accounting journal v3 plus dense/sparse dual oracle | production opt-in H200 oracle passed; representative benchmark pending | through `9a078c2` | jobs 1182321 and 1182344 PASS; see Phase 1A notes |
| 1B | Produce sparse certificate before dense host materialization | blocked on 1A | pending | pending |
| 2 | Device-side stable F1 candidate compaction | blocked on 1B | pending | pending |
| 3 | Device-resident Forward -> Backward/domain -> rescore chain | blocked on 2 | pending | pending |
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

## Later phase order

After Phase 1A is proved: generate v3 before dense materialization; implement
stable device F1 compaction; consume device mappings directly; introduce a
persistent request/chunk workspace; compile exact host-oracle decision
boundaries; retain Forward state into Backward/domain and rescore; emit one
ordered sparse egress packet; only then explore multi-candidate subwarps and
the later certified/profile-axis research paths.

Every completed phase appends its exact commit, tests, workload, output hash,
measurements, and retain/reject decision to this file.
