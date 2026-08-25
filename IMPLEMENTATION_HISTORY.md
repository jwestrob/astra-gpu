# Plan7 GPU implementation history

This is the short index to the optimization program on `main`. The detailed
phase plan, correctness rules, and retain/reject decisions live in
`PLAN7_ENGINE_ROADMAP.md`; exact performance measurements and local evidence
hashes live in `PERFORMANCE.md`.

## Canonical line

- Review baseline: `5ffe5d43cf4177493b72f24b0fb96c00276c96ab`.
- Current retained implementation: `2fc2a92` (deterministic Phase 11 GPU
  execution policy).  The best full-workload implementation remains the
  certified Phase 9 GA-pruning path incorporated in that tip.
- Exact full-output oracle: SHA-256
  `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
  39,010,327 bytes, 383,235 lines.

## Retained implementation sequence

| Phase | Retained source | Result |
|---|---|---|
| 0 | through `f9ef356` | Exact per-profile work/fallback telemetry and immutable reporting. |
| 1A | through `039c092` | Exact sparse-accounting oracle retained; its dense-to-sparse production experiment was slower. |
| 1B | through `87727cd` | Native one-pass sparse v3 retained; zero dense-v2 retention and exact output. |
| 2 | through `959cdf6` | Stable device F1 compaction retained; host expansion and candidate re-upload removed. |
| 3 | through `171d544` | Forward-to-Backward and Backward-to-rescore residency plus exact compiled F3 retained. |
| 4 | `a8932dd` | Wider subwarps remain diagnostic; production deliberately restored to width 1. |
| 5 | `9730f39` | Length-compatible profile-packed SSV retained; full request fell to 455.026 s. |
| 6 | `cfd756c` | Target-length metadata retained; transition H2D fell from 24.92 MB to 0.12 MB. |
| 7 | `f5ac24a` | Full-MSV compaction and exact packed full MSV retained; request 454.007 s. |
| 8 | `913be5d` | Certified F0 evaluator retained as evidence; production F0 rejected after exact H200 census. |
| 9 | `50f762c` | Certified gathering-cutoff GA pruning retained; exact request 449.104 s, 1.08% faster than the prior best. |
| 10 | `786d1a6` | Exact mandatory-seed evaluator retained as evidence; production index rejected after certifying only 0.494% of cells. |
| 11 | `2fc2a92` | Deterministic request-scoped GPU execution policy retained; exact six-shape H200 matrix preserves the simple path and accelerates large-by-large generation. |

## Rejected or non-production experiments

- Phase 1A dense-to-sparse production replay: exact, but 3.82% slower. The
  journal ABI and dual oracle remain useful; the measured execution strategy
  is not the default.
- Phase 4 automatic wider-Forward policy: exact, but 0.72% slower on the full
  workload. Production is width 1.
- Packed Viterbi at isolated commit `61c3545`: exact, but 0.211% slower than
  packed full MSV, with 94,836 KiB more RSS and 814,684,436 bytes of added
  execution-index H2D. It is not merged into `main`.
- Reduced-alphabet F0: zero false rejects, but the best 32-partition codebook
  certified only 0.145% of SSV cells while F0 itself cost about 1.9x exact
  packed F1. The offline evaluator remains; production integration is rejected.
- Mandatory-seed/global-profile indexing: zero false rejects, but its best
  32-residue certificate removed only 0.494% of exact SSV cells and cost 5.85x
  packed exact F1. Incomplete dictionary samples already imply at least 65.6
  GiB across Pfam, so no production index is built.

## Current status

All numbered roadmap phases have an implementation or evaluator, an exact
oracle result, and an explicit retain/reject decision.  Phase 11 is retained
after clean H200 job `1183478`: all policies produced identical records and
HMMER output across six workload shapes, while `auto` improved warm
large-by-large generation by 9.871% over `simple` for a 2.231% persistent-HBM
increase.  The production policy excludes every rejected experiment and
keeps the scalar/query-major route available.

Full H200 job `1183483` reproduced the exact oracle in 451.083 seconds, 0.441%
slower than the Phase 9 best, with sampled H200 memory 22 MiB lower.  The
policy is therefore retained for dispatch correctness and workload-shape
coverage, not as a new full-workload performance record.

This completes the numbered experiment sequence, not the full heterogeneous
engine objective.  The roadmap ledger still marks Phase 3 partial: dense
postfilter/Forward journal materialization and per-generation Backward/rescore
workspaces remain.  Subsequent work resumes at that boundary rather than
inventing a Phase 12 research algorithm.

For the complete commit-by-commit history, run:

```text
git log --reverse --oneline 5ffe5d43cf4177493b72f24b0fb96c00276c96ab..main
```
