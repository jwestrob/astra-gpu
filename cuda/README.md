# CUDA

`ssv_cuda.cu` implements HMMER 3.4's public signed-byte SSV filter. The current
adapter accepts the optimized profile and digital sequences already held by
Astra; it does not load biological files or replace Astra's search pipeline.

`SequenceBatch` packs and uploads targets once, then reuses its CUDA allocations
across profiles. `cpu_candidates(profile, F1)` applies HMMER's exact F1 gate and
returns indexes that direct SSV cannot safely reject. Fallbacks, overflows,
empty targets, and invalid parameters are always retained. `filter_ssv`
remains available for score diagnostics. The corresponding `*_many` methods
compact HMMER's striped scores into GPU-native `[model][residue]` rows and
evaluate a profile batch in one two-dimensional CUDA launch. The single-profile
path keeps HMMER's striped layout. The batched host gate caches length-only terms
and uses an exact per-profile binary32 cutoff, falling back to the scalar Easel
calculation whenever a cutoff cannot be proven safe.
