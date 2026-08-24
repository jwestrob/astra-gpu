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

The production prototype then added a stable length-compatible quartet view.
It activates only for profile selections of at least 32 models, packs quartets
whose model lengths satisfy the audited compatibility bound, and sends every
unattested, unsuitable, or leftover model through the unchanged scalar kernel.
Packed results are written to the original profile-major mask positions, so no
user-visible or continuation ordering changes. The scalar implementation can
also be forced for audit with `PLAN7_GPU_SSV_PROFILE_POLICY=scalar`.

A focused H200 production oracle (job `1182733`) compared forced-scalar and
automatic-packed execution for 40 distinct amino-acid models and 65 targets.
All postfilter offsets, complete dense records, final HMMER results, and output
bytes were identical. All 40 profiles ran in 10 quartets with zero scalar
spill.

## Full-workload result

Full H200 job `1182734` searched all 27,481 Pfam models against all 300,186
targets and reproduced the established CPU64/HMMER output exactly:

- SHA-256: `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`
- size: 39,010,327 bytes
- lines: 383,235
- end-to-end request: 455.026448 seconds
- pipeline wall: 447.583375 seconds
- GPU generation: 345.194725 seconds
- CPU continuation/output: 401.684422 seconds
- generation/continuation overlap: 299.456843 seconds

All 83 chunks used the packed path. Of 27,481 profiles, 27,160 were packed in
6,790 quartets and only 321 used the scalar fallback. Packed-score tables
materialized 121,596,420 bytes cumulatively. Maximum sampled H200 memory was
3,372 MiB and maximum GPU utilization was 100%.

Against the retained combined Phase 3 width-one run, request wall fell from
535.212821 to 455.026448 seconds: 80.186372 seconds, or 14.982%. Generation
fell from 463.470921 to 345.194725 seconds: 118.276196 seconds, or 25.520%.
Against the original 546.220705-second dense GPU baseline, request wall is
16.695% lower. Against Astra CPU64 at 702.79 seconds, this run is 1.5445x
faster.

Decision: retain the length-compatible profile-packed SSV path in production.
Keep the scalar path for one/few profiles, unsuitable quartets, and audit. The
full exact run converts the earlier microbenchmark result into an end-to-end
performance result.

Evidence:

- result: `build/phase5-h200-20260824/result.json`
- result SHA-256: `4abfef6f197c376f6a4363c4de0f12c412e0f477f5d9cb51fa576a3cd63eda76`
- Slurm job: `1182723`, `COMPLETED`, exit `0:0`
- focused production oracle: job `1182733`, `COMPLETED`, exit `0:0`
- full run: `build/h200-phase5-profile-packed-ssv-20260824/attempt-01-full/runs/h200-full`
- full worker SHA-256: `92cb6510e11f0b248fe68c4ffd8c657ee2f903472ad224b834f9ad4e4ec814ab`
- full raw-validation SHA-256: `a4d7f394071acea2775c6bcef4239bb5d951f207a427cb3965ef1e4913cf2893`
- full Slurm job: `1182734`, `COMPLETED`, exit `0:0`
