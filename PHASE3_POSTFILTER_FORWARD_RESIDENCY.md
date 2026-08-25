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
Gumbel predicate.  A focused H200 old-versus-new fused-generation oracle is
the promotion gate; its result is pending in this implementation commit.

## Benchmark decision

Pending the exact H200 gate and representative/full timing.  Production must
retain the old host decision/upload path when compilation is unsupported, and
the change will be rejected if it changes any complete HMMER output or merely
moves the eliminated transfer into another buffer.
