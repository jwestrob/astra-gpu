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

The focused H200 oracle reproduced every output byte. At the realistic
384-profile/1,000-target density (14.985 candidates per target), the retained
kernel took 0.617520 ms and sequence-major execution including grouping took
0.602896 ms: only 1.024x faster. A forced all-pass 384x1,000 case improved
2.311696 -> 1.643552 ms (1.407x), while the dense 384x16 small case regressed
0.411520 -> 0.837120 ms (0.492x). One-, 16-, and 64-profile realistic cases
were effectively flat.

Retain/reject decision: reject production integration. The exact experiment
remains isolated, but realistic H200 density does not clear the 5% stage gate
and the small dense case regresses substantially. Revisit only with a new
mapping that removes that measured loss.
