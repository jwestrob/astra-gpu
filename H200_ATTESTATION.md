# H200 correctness gate

The CUDA fatbinary already contained native `sm_90` cubins, but the exact
composition-bias runtime gate admitted only the original RTX 2080 Ti host and
device pair.  This branch keeps that `sm_75` pair unchanged and adds one new
target class: a full NVIDIA H200/H200 NVL device at compute capability 9.0.
H100, MIG-sized devices, other Hopper products, and every other architecture
remain rejected.

The runtime provenance record is available without launching a kernel:

```bash
PYTHONPATH=python python -c \
  'import json; from plan7_gpu import _native; print(json.dumps(_native.bias_environment_provenance(), indent=2))'
```

It binds the CUDA runtime and compiler build, driver API level, product name,
compute capability, SM count, memory size, UUID, PCI address, and host CPUID.
The first-1000 harness also joins the CUDA UUID to `nvidia-smi`, records the
package driver version, and rejects disagreement in product, capability, UUID,
or PCI identity.

## Build and source-review boundary

Build in this isolated worktree so the extension consumed by Astra cannot be
confused with an older `_native` shared object:

```bash
make cuda BUILD_DIR=build/h200-sm90 -j8
cuobjdump --list-elf python/plan7_gpu/_native.cpython-312-x86_64-linux-gnu.so
```

The listing must contain both `sm_75.cubin` and `sm_90.cubin`.  Astra candidate
`3238c76` is a different repository at
`../astra-profile-session-cache`; it should consume this extension by putting
this worktree's `python` directory first:

```bash
PYTHONPATH="$PWD/python:$PWD/../astra-profile-session-cache"
export PYTHONPATH
```

Do not submit the H200 audit until the source diff and build provenance have
been reviewed.

## Mandatory first-1000 semantic acceptance

From inside the released H200 allocation, run:

```bash
make h200-first1000-audit BUILD_DIR=build/h200-sm90 \
  H200_FIRST1000_FASTA=/absolute/path/PLM2_5.first1000.faa \
  H200_PFAM_BASE=/groups/banfield/users/jwestrob/.config/Astra/PFAM/PFAM \
  H200_PFAM_MANIFEST="$PWD/results/pressed/PFAM.manifest.json" \
  H200_FIRST1000_OUTPUT=/absolute/path/h200-pfam-first1000.json
```

This is a correctness run, not a timing result.  It requires exactly 1,000
FASTA records and compares, for all 27,481 PFAM models, CPU and GPU query
identity, target table bytes, domain table bytes, and Plan7 accounting bytes.
It emits `PASS` only when every model is exact and both aggregate SHA-256
digests match.  A whole-metagenome GPU run is prohibited until that record is
`PASS`.

The H200 node CPU signature is currently pinned to the cluster inventory
inference (GenuineIntel family 6, model 143, stepping 8, Sapphire Rapids).  The
first released allocation must inspect the provenance record before semantic
execution.  If the observed signature differs, stop and review it; do not
weaken the allow-list in the job script.
