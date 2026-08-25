# Plan7 GPU implementation history

This is the short index to the optimization program on `main`. The detailed
phase plan, correctness rules, and retain/reject decisions live in
`PLAN7_ENGINE_ROADMAP.md`; exact performance measurements and local evidence
hashes live in `PERFORMANCE.md`.

## Canonical line

- Review baseline: `5ffe5d43cf4177493b72f24b0fb96c00276c96ab`.
- Current retained implementation: `348f277` (resident exact F2-to-Forward
  handoff, on top of the deterministic Phase 11 policy and certified Phase 9
  GA pruning).
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
| 3 | through `348f277` | Forward-to-Backward, Backward-to-rescore, and F2-to-Forward residency plus exact compiled F2/F3 retained; exact tail extensions rejected by full measurement. |
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
- External multidomain continuation at isolated commit `d5731718`: exact and
  moved 132,654 rows off the general CPU route, but full request wall regressed
  7.167% to 480.259 seconds.
- Bounded Backward/domain waves at isolated commit `1685de3`: exact and
  eliminated all 366,939 cumulative work-cap fallbacks, but regressed 7.005%.
- Bounded Backward plus isolated-rescore waves at isolated commit `a3d3e3b`:
  exact and reduced 568,120 exposed rescore cap fallbacks to 9,998, but
  regressed 7.662%.
- Zero threshold guard at isolated commit `d94a3ab`: exact and removed the
  maximum 34,485 threshold-only CPU rows without retry-DP cost, but regressed
  8.494%. Production retains the calibrated guard.

## Current status

All numbered roadmap phases and the final continuation-tail experiments have
an exact oracle result and an explicit retain/reject decision. Full H200 job
`1183504` is the retained performance result: 448.140781 seconds with the
exact 39,010,327-byte output. Generation was 333.993605 seconds, CPU
continuation/output 400.083104 seconds, and overlap 293.693784 seconds.

The remaining measured continuation causes were then exercised directly on
the full workload. Multidomain rerouting, bounded Backward waves, bounded
rescore waves, and the zero-cost threshold upper bound all reduced their
target route counts but made request wall 7.005% to 8.494% slower. Those
implementations remain isolated and are not ancestors of `main`.

This satisfies the bottleneck-focused stop condition. The evidence does not
declare an absolute Plan7 hardware ceiling; it shows that the scoped cap,
threshold, and multidomain changes cannot materially reduce the present exact
CPU tail. A future performance program needs a fundamentally faster exact
domain/rescore/multidomain algorithm, not another route transfer. The complete
decision is in `CPU_CONTINUATION_TAIL_RESULTS.md`.

For the complete commit-by-commit history, run:

```text
git log --reverse --oneline 5ffe5d43cf4177493b72f24b0fb96c00276c96ab..main
```
