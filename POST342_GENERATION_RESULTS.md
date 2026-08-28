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

## Pending: cached Viterbi length transitions

The exact cache replaces per-candidate transcendental transition planning with
profile-by-observed-length-class lookup when the table is smaller than the
candidate stream. A 333-profile by full-target H200 oracle (`1185360`) was
exact and reduced measured candidate planning from 88.053 to 45.862 ms
(-47.9%). Full job `1185378` was exact at 315.392470 seconds request and
306.528780 seconds generation, improvements of 0.44% and 0.47% relative to
job `1185334`. Because the end-to-end difference is below 1%, this path is not
promoted without matched repeats. Tiny workloads retain direct planning when
the cache table would exceed the candidate count.
