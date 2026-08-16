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
`p7_PipelineFromMSV`, `p7_PipelineFromFilterScores`, and
`p7_PipelineFromFilterAndForwardScores`, and installs matching Cython
declarations. The latter two entries are search-mode only.

`p7_VIT_EXTERNAL` supplies an exact Viterbi score. `p7_VIT_CPU` is the
fail-closed fallback for accelerator `eslERANGE` or `eslENORESULT`.
`p7_VIT_NONE` is valid only when the bias-corrected MSV P-value already bypasses
Viterbi (`P <= F2`). The seam rechecks F1/F2 and updates HMMER's survivor
counters itself. With bias filtering disabled, the supplied `filtersc` is
ignored and HMMER uses `nullsc`.

The Forward seam replays all four filter gates. F3 rejects require no matrix;
F3 survivors require exactly `6 * (L + 1)` binary32 E/N/J/B/C/SCALE cells. It
checks the special-state recurrences, reconstructs HMMER's `totscale` and
Forward score bit-for-bit, then imports the matrix for the unchanged Backward
and domain workflow. Invalid payloads and unattested host floating-point modes
fail before survivor counters or workspaces change.

The additive simple-region seam continues guarded F3 survivors without the
full-target CPU Backward/parser and domain-decoding passes. It accepts only a
finite posterior domain expectation and ordered, nonoverlapping simple-region
intervals (or an explicit no-regions route), rechecks the exact F1/F2/F3 and
bias-filter generation options, then uses HMMER's existing isolated-envelope
Forward/Backward, posterior decoding, null2, optimal-accuracy, display, and
shared hit-scoring path. Uncertain, clustered, nonfinite, or otherwise
unsupported rows must remain on the existing Forward continuation.

This patch supplies only the pinned HMMER boundary and its direct parity
probe. Product integration is intentionally still blocked: the final opaque
adapter must seal the complete external score/status tuple, route payload,
profile identity, target identity, thresholds, and background fingerprint as
one immutable generation. The HMMER seam additionally rejects stale live
profile/background length configuration, but cannot establish that hidden
cross-object provenance by itself.

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
all four entry points, stage counters, F1/F2/F3 rejects, Viterbi elision,
ERANGE/ENORESULT fallback, `--nobias`, malformed Forward payloads, required
symbols, a real Forward rescaling case, and that only one HMMER and one Easel
library are mapped.

For the installed `plan7_gpu._pipeline`, use a relative runpath
`$ORIGIN/../pyhmmer.libs`; never retain the current absolute development
runpath. Start Astra with the private environment from process startup.
