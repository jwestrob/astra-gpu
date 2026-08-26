# Sequence-major exact bias experiment

Hypothesis: after profile-axis SSV, the retained profile-major bias list makes
neighboring CUDA lanes reread unrelated targets.  Grouping resident candidates
by dense target ordinal lets one warp broadcast each residue while its lanes
advance independent profile-specific two-state recurrences.

The diagnostic path builds a device histogram, exclusive scan, and scatter,
retains each original candidate ordinal, and writes the unchanged 12-byte bias
record back to that ordinal.  The bias recurrence and binary64 `log()` path use
the same explicitly rounded operations as the retained kernel.  The ordinary
entry point and production policy are unchanged.

On the local attested RTX 2080 Ti, 384 real Pfam profiles against the first
1,000 PLM targets produced 14,985 F1 candidates (14.985 per target).  All bias
records and CSR offsets were byte-identical.  Warm medians were approximately
4.36 ms retained versus 3.50 ms sequence-major including 0.018 ms grouping, a
1.24x stage speedup.  A dense 256-profile all-pass case improved from 44.75 ms
to 14.88 ms (3.01x).  The path loses for one/few very sparse profiles, so any
future policy must retain the existing candidate-major fallback.

Retain/reject decision: keep isolated pending a representative H200 exact and
stage-timing gate.  Do not enable production dispatch from this evidence alone.
