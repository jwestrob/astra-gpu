# CUDA

`ssv_cuda.cu` implements HMMER 3.4's public signed-byte SSV filter. The current
adapter accepts the optimized profile and digital sequences already held by
Astra; it does not load biological files or replace Astra's search pipeline.
