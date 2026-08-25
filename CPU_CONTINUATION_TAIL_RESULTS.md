# CPU continuation tail: measured stop decision

## Starting point

The retained production path searched 300,186 targets against 27,481 Pfam
profiles in **448.140781 seconds** on one H200 (job `1183504`). It reproduced
the established HMMER/Astra output exactly:

- SHA-256: `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`
- bytes: 39,010,327
- lines: 383,235

Generation took 333.993605 seconds and CPU continuation/output took
400.083104 seconds, with 293.693784 seconds of overlap. The Phase-0 census
attributed the remaining 552,390 `CPU_REQUIRED` Backward/domain rows to three
dominant, overlapping source conditions:

- cumulative Backward work cap: 366,939 rows;
- multidomain posterior classification: 150,965 rows;
- threshold uncertainty: 52,797 rows.

Compact rescore additionally returned 231,725 regions to the CPU through the
matrix-cap family, including row-atomic siblings of capped regions. Numeric,
own-scale, host-environment, unsupported-mode, and catch-all failures were zero
in this workload.

## Full H200 experiments

Each experiment ran the first-1,000 exact gate followed by the complete
workload. Every full result had the same SHA-256, byte count, and line count as
the accepted path.

| Experiment | Job | Request wall | Change | Principal route effect | Peak H200 | Max RSS | Decision |
|---|---:|---:|---:|---|---:|---:|---|
| Accepted resident engine | 1183504 | **448.140781 s** | baseline | 552,390 CPU-required rows | 3,388 MiB | 13,195,416 KiB | retained |
| External multidomain continuation | 1183518 | 480.258523 s | +32.117742 s (+7.167%) | moved 132,654 rows off the general CPU route | 3,380 MiB | 13,080,804 KiB | rejected |
| Bounded Backward/domain waves | 1183521 | 479.533852 s | +31.393070 s (+7.005%) | work-cap fallbacks 366,939 -> 0; CPU-required 552,390 -> 340,435 | 3,376 MiB | 13,028,936 KiB | rejected |
| Bounded Backward + rescore waves | 1183524 | 482.477600 s | +34.336819 s (+7.662%) | work-cap 0; rescore cap fallbacks 568,120 -> 9,998 | 3,380 MiB | 12,744,140 KiB | rejected |
| Zero threshold guard upper bound | 1183528 | 486.204557 s | +38.063775 s (+8.494%) | removed the maximum 34,485 threshold-only CPU rows at zero retry-DP cost | 3,358 MiB | 13,137,688 KiB | rejected |

The bounded-wave results are especially diagnostic. They eliminated the
dominant cumulative caps without increasing peak H200 memory, but executing
the newly admitted exact Backward/domain and isolated-rescore work increased
generation and total request wall. The caps were therefore protecting the
critical path from work that is slower on the current GPU kernels; they were
not merely arbitrary lost throughput.

The zero-guard experiment is an upper bound for an exact threshold retry. It
resolved every threshold-only row without any retry computation and was still
38.064 seconds slower. A formal retry can resolve no more rows and must add
work, so it cannot materially improve this architecture.

The multidomain experiment likewise changed ownership rather than removing
semantics: exact multihit Forward, stochastic traces, clustering, overlap
accounting, and isolated rescoring still had to execute. Its lower route count
did not reduce the critical path.

## Decision

The accepted resident engine remains the production line. None of these tail
implementations or diagnostics is merged into `main`:

- external multidomain prototype: `d5731718` on its experiment branch;
- bounded Backward waves: `1685de3` on `agent/phase3-backward-waves`;
- bounded rescore waves: `a3d3e3b` on the same branch;
- zero-guard diagnostic: `d94a3ab` on `agent/phase3-threshold-exact`.

This closes the current CPU-continuation optimization objective. The evidence
does not claim that exact Plan7 can never be faster. It establishes that
raising/resetting the present caps, resolving the present guard, or moving the
present multidomain semantics across the seam does not improve end-to-end
runtime. A future material gain must make the exact domain/rescore/multidomain
algorithms themselves substantially faster, rather than simply routing more
rows to them.

## Evidence

- Multidomain: `build/h200-multidomain-full-20260825/attempt-01/result.json`,
  SHA-256 `431e8fadf987a18c89f55be11cad4ca02b8963dc9a3846fcfd3f60d45ec75c8d`.
- Backward waves: `build/h200-backward-waves-full-20260825/attempt-01/result.json`,
  SHA-256 `acae6a2c1c9b7c7c8e0b15569e8da60b1480444dad2b661dddfc1caa9eae24db`.
- Backward + rescore waves:
  `build/h200-backward-waves-full-20260825/attempt-02/result.json`, SHA-256
  `3737e1a23665e545250640aacd39535592854b3f931155711898ae5408e9f5e8`.
- Threshold upper bound:
  `build/h200-threshold-zero-full-20260825/attempt-01/result.json`, SHA-256
  `28f31426f599b7e778fe5829088651c36dff56a79f27c8685a1014853d726b0f`.
