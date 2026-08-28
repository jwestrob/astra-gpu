# Post-342 generation campaign

The exact output oracle for every full result below is SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.

## Retained: CPU domain ownership plus raw-xE reuse

Full H200 job `1185334` moved domain work under the hidden sharded CPU lane and
retained packed-F1 raw xE so ordinary F1 survivors no longer replay SSV before
bias. All 203,671,109 candidate replays were eliminated. Request wall was
316.782413 seconds and generation was 307.990683 seconds, versus 342.819173
and 333.307039 seconds for the preceding retained engine.

## Retained: identity padding in packed SSV

The packed quartet reference kernel reconstructed a four-profile active mask
at every model position. The identity-padding specialization stores signed
score cost zero beyond each profile's model length. Signed saturating subtract
by zero preserves the terminal state bit-for-bit; repeating a terminal state
that has already participated in the running maximum cannot introduce a new
maximum. The old poisoned-padding kernel remains the default/reference path.

The specialization reduced register use from 57 to 54 threads on sm75 and
from 55 to 50 on sm90. Matched first-1,000 job `1185387` reproduced the exact
output in every arm and reduced median packed-F1 time from 0.823937 to
0.585124 seconds (-29.0%). Direct H200 oracle `1185453` then compared the old
and new paths over 40 profiles of lengths 87--400 and 65 targets of lengths
1--511. Postfilter CSR offsets, complete records, and final HMMER output were
byte-identical.

Full H200 job `1185415` reproduced the full oracle in 261.876203 seconds:
253.151836 seconds generation, 80.619169 seconds continuation/output,
79.714061 seconds overlap, and 254.108736 seconds pipeline wall. Packed-F1
native time fell from 164.483261 to 109.600812 seconds (-54.882448 seconds,
-33.4%). Against job `1185334`, request wall fell 54.906210 seconds (17.3%)
and generation fell 54.838847 seconds (17.8%). Against the former retained
342.819173-second line, request wall fell 80.942970 seconds (23.6%). Peak RSS
was 7,396,012 KiB, 59,392 KiB below job `1185334`; this run did not collect an
independent peak-HBM sample.

## Retained: cached Viterbi length transitions

The exact cache replaces per-candidate transcendental transition planning with
profile-by-observed-length-class lookup when the table is smaller than the
candidate stream. A 333-profile by full-target H200 oracle (`1185360`) was
exact and reduced measured candidate planning from 88.053 to 45.862 ms
(-47.9%). Full job `1185378` was exact at 315.392470 seconds request and
306.528780 seconds generation, improvements of 0.44% and 0.47% relative to
job `1185334`. Tiny workloads retain direct planning when the cache table would
exceed the candidate count.

The combined identity-padding and length-cache run, full H200 job `1185455`,
was again exact and completed in **258.817809 seconds**: 249.859082 seconds
generation, 81.052199 seconds continuation/output, 79.923879 seconds overlap,
and 251.045017 seconds pipeline wall. This improved the exact G8-only run by
3.058394 seconds (1.17%) and generation by 3.292753 seconds (1.30%). All 83
full-workload chunks used identity padding and the Viterbi cache. The cache
served all 203,671,109 candidates from 40,259,665 compiled entries, taking
0.527898 seconds to build its per-chunk tables and 2.300863 seconds for the
remaining candidate planning. Peak RSS was 7,356,372 KiB, 39,640 KiB below
the G8-only run. The first-1,000 prerequisite deliberately retained direct
planning because its cache table would have exceeded its 879,857 candidates.

Automatic promotion retains explicit `0`/reference controls. Identity padding
is selected whenever packed SSV is selected. Raw-xE is selected for supported
non-simple workloads with at least 65,536 logical pairs. Cached Viterbi
planning is considered only for non-simple streams with at least 65,536
candidates and is used only when its table is smaller than that stream. CPU
domain ownership is selected automatically at 65,536 or more targets; simple
policy and smaller target batches retain the prior GPU-domain route.

Focused H200 policy job `1185459` passed exact record, offset, and HMM-output
oracles for 1x1, 1x4,096, 10x4,096, 100x512, 512x16, and 512x4,096 workloads.
It also verified that the 65,536-target automatic CPU-domain boundary remains
disabled under the explicit simple policy.
