# CUDA

`ssv_cuda.cu` implements HMMER 3.4's public signed-byte SSV filter. The current
adapter accepts the optimized profile and digital sequences already held by
Astra; it does not load biological files or replace Astra's search pipeline.

`SequenceBatch` packs and uploads targets once, then reuses its CUDA allocations
across profiles. `filter_ssv(profile, sequences)` remains the one-shot API.
