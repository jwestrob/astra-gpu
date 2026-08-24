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

Pending the focused H200 run. Production remains on the existing SSV path.
