# Plan7 GPU implementation history

This is the short index to the optimization program on `main`. The detailed
phase plan, correctness rules, and retain/reject decisions live in
`PLAN7_ENGINE_ROADMAP.md`; exact performance measurements and local evidence
hashes live in `PERFORMANCE.md`.

## Canonical line

- Review baseline: `5ffe5d43cf4177493b72f24b0fb96c00276c96ab`.
- Current retained implementation before this record: `f5ac24a576803ab36e7f02688fb0a749c4cb5acd`.
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

## Rejected or non-production experiments

- Phase 1A dense-to-sparse production replay: exact, but 3.82% slower. The
  journal ABI and dual oracle remain useful; the measured execution strategy
  is not the default.
- Phase 4 automatic wider-Forward policy: exact, but 0.72% slower on the full
  workload. Production is width 1.
- Packed Viterbi at isolated commit `61c3545`: exact, but 0.211% slower than
  packed full MSV, with 94,836 KiB more RSS and 814,684,436 bytes of added
  execution-index H2D. It is not merged into `main`.

## Work now in progress

Phase 8 is an offline certified reduced-alphabet F0 evaluator. No unfinished
Phase 8 source is part of this `main` history record. It must prove an upper
bound with zero false negatives and demonstrate a runtime or memory benefit
before production integration. Phases 9-11 remain ordered research and policy
work as listed in `PLAN7_ENGINE_ROADMAP.md`.

For the complete commit-by-commit history, run:

```text
git log --reverse --oneline 5ffe5d43cf4177493b72f24b0fb96c00276c96ab..main
```
