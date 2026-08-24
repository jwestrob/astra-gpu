# Phase 7: packed integer arithmetic experiment

## Hypothesis

The exact postfilter MSV recurrence can advance four independent unsigned-byte
values with CUDA packed-video operations, and Viterbi can advance two
independent signed-int16 values. Hopper DPX add/max is potentially faster, but
is usable only if it preserves HMMER's saturating-add behavior.

## Experiment

`experiments/phase7_packed_integer.cu` compares scalar and packed synthetic
recurrences on an attested H200. It checks every unsigned-byte input pair for
saturating add/subtract and maximum, then checks all 65,536 signed-int16 left
operands against seven overflow/boundary right operands. The timed recurrences
process 1,048,576 logical values for 128 steps and compare complete result
bytes before timing.

The exact packed implementations use `__vaddus4`, `__vsubus4`, and
`__vmaxu4` for MSV, and `__vaddss2` plus `__vmaxs2` for Viterbi. A separate
`__viaddmax_s16x2` DPX arm measures the tempting fused form and deliberately
tests it against the saturating oracle.

## H200 result

Job `1182745` completed successfully:

| Recurrence | Scalar median | Packed median | Speedup |
|---|---:|---:|---:|
| unsigned-byte MSV-like | 0.079296 ms | 0.057344 ms | 1.3828x |
| signed-int16 Viterbi-like | 0.080928 ms | 0.044896 ms | 1.8026x |
| signed-int16 DPX add/max | 0.080928 ms | 0.018656 ms | 4.3379x |

The packed-video paths were byte-exact. The DPX arm was not: its non-saturating
packed add disagreed on 131,070 of 458,752 boundary primitive cases and on
584,208 of 1,048,576 recurrence outputs. It is therefore ineligible for the
general exact HMMER recurrence despite its throughput. Generated sm90 SASS
contains packed `VIADD`, `VIMNMX`, and fused `VIADDMNMX` instructions; resource
usage was 16 registers/thread for packed MSV and 15 for exact packed Viterbi.

Decision: retain this standalone evidence and proceed only with the exact
packed-video arithmetic. Prototype full MSV first because its byte recurrence
has the simpler state topology. Keep DPX out of production unless a future
source proof and guarded fallback establish that saturation cannot occur.

Evidence:

- result: `build/phase7-packed-integer-h200/attempt-01/result.json`
- result SHA-256: `a91941f5fe23b4021ed44c1cb122a2455cb2209ef8dffda24bcc52b1b158053e`
- Slurm job: `1182745`, `COMPLETED`, exit `0:0`
