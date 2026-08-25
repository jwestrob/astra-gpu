# Phase 3: resident postfilter/F2 to Forward handoff

## Hypothesis

The sealed generation path currently downloads the complete postfilter result
stream, calls the host Gumbel survival function while selecting F2 survivors,
reconstructs the Forward input rows, and uploads three four-byte arrays for
those survivors.  Postfilter already owns the same candidate mappings and
result records on the selected device.  An exact device-side F2 decision and
stable compaction should remove the host statistical calls and the redundant
Forward candidate upload without changing the continuation journal.

## Implementation

The linked host `esl_gumbel_surv` predicate is compiled into the smallest
binary32 bit score that passes F2.  The compiler searches the complete ordered
non-NaN binary32 domain and certifies the predecessor, threshold, successor,
and infinities; invalid parameters retain the old host route.

Postfilter classifies its resident result rows with HMMER's existing binary32
operation order, produces one ballot mask per 32 rows, performs an integer CUB
exclusive scan, and stably scatters original postfilter source ordinals.  The
host receives only those source ordinals for journal construction and audit.
Forward validates the host reconstruction against the resident view, gathers
profile ordinal, target ordinal, and filter score directly into its existing
workspace, and avoids their three host-to-device copies.  Existing postfilter
records, the dense audit journal, Forward output, provenance, and CPU
continuation remain unchanged.

No new persistent device capacity class is introduced: compaction runs after
postfilter completion and reuses inactive postfilter workspace buffers.  The
selected-source host vector and its exact downloaded bytes are reported.

## Correctness evidence

The exact d486 private-ABI sm75/sm90 build passes.  The focused host boundary
oracle checks adversarial binary32 values around every compiled boundary and
all 63,490 non-NaN binary16 values promoted to binary32 against the linked
Gumbel predicate.  Focused H200 job `1183491` compared the old and resident
paths and matched all three complete TopHits rows exactly.  It exercised 12
postfilter inputs, selected 11 F2 survivors in stable order, and reported one
resident Forward call with zero unsupported profiles.

## Benchmark decision

Full H200 job `1183504` searched 27,481 Pfam profiles against all 300,186
targets.  Its output is byte-identical to the CPU64 and prior GPU oracle:
SHA-256 `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.

The request completed in **448.140781 seconds**, 0.963331 seconds (0.2145%)
faster than the prior Phase 9 best.  Generation fell from 340.105077 to
333.993605 seconds (6.111472 seconds, 1.797%), while CPU continuation/output
remained the critical lane at 400.083104 seconds.  All 203,671,109 postfilter
rows were device-classified with zero unsupported profiles; 12,121,540 F2
survivors were transferred as 48,486,160 bytes of source ordinals, and
145,458,480 bytes of Forward candidate H2D were eliminated.  The added F2
compile/kernel/scan/download path totaled 76.684 ms across 83 chunks.

Memory was effectively flat: sampled H200 peak was 3,388 MiB (4 MiB below
Phase 9), while maximum RSS rose 36,068 KiB (0.274%) to 13,195,416 KiB.
Because the exact full request improved and the fallback path remains intact,
the optimization is retained.  Evidence lives under
`build/h200-phase3-f2-resident-full-20260825/attempt-01-full/runs/h200-full`;
`worker.json` SHA-256 is
`6053612b5e01d1b7f940e7650666839bf169ada3a640e13b48508ed8afbc913e`
and `raw-validation.json` SHA-256 is
`06b07e0f4007c68488a1d88e5bb582652066969dd26b6ba2af109d2cc8fc6c28`.
