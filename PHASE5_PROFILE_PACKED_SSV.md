# Phase 5: profile-axis packed SSV experiment

## Hypothesis

For sufficiently large profile sets, four independent signed-byte SSV
recurrences may be packed into one 32-bit word. A single target-residue load
then advances four profiles while preserving each profile's exact recurrence.

## Experimental implementation

`experiments/phase5_profile_packed_ssv.cu` is standalone and is not linked into
the production backend. It compares the current scalar recurrence with a
four-profile implementation using CUDA's packed signed-saturating subtract and
unsigned-byte maximum intrinsics. The packed table is organized as
quartet-by-model-position-by-residue. Original profile order is retained.

Unequal model lengths use an explicit per-byte active mask. Inactive table
bytes are deliberately filled with `0x80`, so a missing mask would produce an
oracle failure rather than a favorable padding accident. The executable also
checks every one of the 65,536 byte pairs for exact signed-saturating subtract
and unsigned maximum behavior before running profile cases.

Acceptance requires byte-identical result records for a partial quartet,
empty targets, boundary model lengths, equal-length profiles, sorted profiles,
and deliberately divergent quartets. Timing is kernel-only on an attested
H200. Generated sm90 SASS is inspected rather than assuming that CUDA packed
intrinsics map to a single Hopper instruction.

## Result

Focused H200 job `1182723` completed successfully on an NVIDIA H200 (compute
capability 9.0). All 65,536 primitive byte pairs passed. Every scalar and
packed result record was byte-identical, including the partial-quartet,
empty-target, boundary-length, and hostile-padding oracle.

Median kernel timings were:

| Case | Scalar | Packed four-profile | Speedup |
|---|---:|---:|---:|
| 1,024 equal-length profiles, M=96, L=96, 64 targets | 0.6282 ms | 0.4212 ms | 1.491x |
| 1,024 length-sorted profiles, M=32..383, L=256, 64 targets | 2.7744 ms | 1.9165 ms | 1.448x |
| 512 deliberately divergent quartets, M=32/64/128/384, L=512, 32 targets | 1.0193 ms | 1.7131 ms | 0.595x |

The sm90 packed kernel uses 40 registers/thread versus 32 for the scalar
kernel. Inspection of generated SASS shows a mixture of integer,
permutation/logic, `VIMNMX`, and `VIADDMNMX` instructions rather than a single
four-byte subtract/max instruction pair. The net result is nevertheless a
clear win when model lengths are compatible and a severe loss when they are
not.

Decision: retain the experiment and proceed to an optional, lazily built
length-compatible quartet execution view. Never pack arbitrarily divergent
profiles, and preserve the scalar path for one/few profiles and unsuitable
quartets. This microbenchmark is not an end-to-end performance claim.

Evidence:

- result: `build/phase5-h200-20260824/result.json`
- result SHA-256: `4abfef6f197c376f6a4363c4de0f12c412e0f477f5d9cb51fa576a3cd63eda76`
- Slurm job: `1182723`, `COMPLETED`, exit `0:0`
