# Private PyHMMER pipeline seam

Decision: build and install one project-private PyHMMER wheel. Do not overwrite
the user-site library, load a second renamed HMMER DSO, or use `LD_PRELOAD`.
Every PyHMMER extension and `plan7_gpu._pipeline` must resolve the same physical
`pyhmmer.libs/liblibhmmer.so` in a fresh interpreter.

## Pinned source

- PyPI sdist: `pyhmmer-0.12.0.tar.gz`
- SHA-256: `27cdfd3cdf72abcc7a6670825fc0195e8ff01d4820efd0f99aec03d3972a922e`
- PyHMMER release commit: `cdf6a009cf2ce3d13b09341672c89bed4b723d74`
- bundled HMMER commit: `9acd8b6758a0ca5d21db6d167e0277484341929b`
- bundled Easel commit: `07ca83ba9ef0414dba9ce0a9331d465b5eb58f2b`

The patch is additive: it does not change a public struct layout. It factors the
existing comparison pipeline into shared post-MSV and post-F2 cores, exports
`p7_PipelineFromMSV` and `p7_PipelineFromFilterScores`, and installs matching
Cython declarations. The post-F2 entry is search-mode only.

`p7_VIT_EXTERNAL` supplies an exact Viterbi score. `p7_VIT_CPU` is the
fail-closed fallback for accelerator `eslERANGE` or `eslENORESULT`.
`p7_VIT_NONE` is valid only when the bias-corrected MSV P-value already bypasses
Viterbi (`P <= F2`). The seam rechecks F1/F2 and updates HMMER's survivor
counters itself. With bias filtering disabled, the supplied `filtersc` is
ignored and HMMER uses `nullsc`.

## Build and test

```bash
patched_pyhmmer/build-wheel.sh
patched_pyhmmer/test-wheel.sh \
  build/pyhmmer-patched/pyhmmer-0.12.0+plan7gpu.0-cp312-abi3-linux_x86_64.whl
```

The proven build is Linux x86-64, Python 3.12, GCC 11.4.0, CMake 3.24.0,
SSE4.1, Release mode. Set `PYHMMER_SDIST` to use a cached sdist.
`PLAN7_ALLOW_TOOLCHAIN_DRIFT=1` permits an exploratory build elsewhere; a
release artifact should instead be built in a pinned manylinux container and
repaired with `auditwheel`.

The wheel distribution version is `0.12.0+plan7gpu.0`, while
`pyhmmer.__version__` remains `0.12.0` because PyHMMER exposes the numeric CMake
project version. Existing exact-version guards therefore remain valid. The
private ABI fingerprint changes and intentionally forces `_pipeline` to be
rebuilt. Regenerate pressed manifests after switching wheels.

The test compares ordinary stock and patched PyHMMER byte-for-byte, then checks
both entry points, stage counters, bias and F2 rejects, Viterbi elision,
ERANGE/ENORESULT fallback, `--nobias`, required symbols, and that only one HMMER
and one Easel library are mapped.

For the installed `plan7_gpu._pipeline`, use a relative runpath
`$ORIGIN/../pyhmmer.libs`; never retain the current absolute development
runpath. Start Astra with the private environment from process startup.
