# Codex Handoff: Modern GPU Acceleration of the HMMER3 Filter Pipeline

## Mission

Develop and rigorously evaluate a **modern hybrid GPU/CPU accelerator for HMMER3 protein `hmmsearch`**.

The intended architecture is:

```text
HMMER profile(s) + protein sequences
                │
                ▼
        GPU: MSV filter
                │
          surviving pairs
                ▼
        CPU: bias filter
                │
                ▼
        CPU: P7Viterbi
                │
                ▼
        CPU: Forward
                │
                ▼
 CPU: domain inference / Forward-Backward /
     null2 / reporting / normal HMMER output
```

If profiling later justifies it:

```text
GPU MSV → CPU or GPU bias → GPU P7Viterbi
                              │
                              ▼
                  CPU Forward + downstream
```

**Do not port Forward unless measured end-to-end results justify doing so.**

The initial project is an experimental standalone accelerator. **Do not modify Astra yet.** Astra/PyHMMER integration occurs only after the hybrid implementation demonstrates both exact scientific equivalence and useful end-to-end acceleration.

Working project name: `plan7-gpu` unless an existing directory/repository dictates otherwise.

---

## Known Prior Art: Read Before Implementing

This is **not** the first GPU or hybrid HMMER effort. Do not make novelty claims.

Study these before designing kernels:

### Lin Cheng, 2014 — `cudaHmmsearch`

*Thesis: Implementing and Accelerating HMMER3 Protein Sequence Search on CUDA-Enabled GPU.*

Repository:

```text
forestcheng/cudahmmsearch
```

This is the most direct predecessor. It implements a functioning HMMER3 search in which the MSV filter is executed on a GPU and results return to the CPU for subsequent HMMER stages.

Important historical observations:

* GPU MSV + CPU remainder was already demonstrated.
* Average reported advantage over single-core HMMER3 was ~2.5×.
* Advantage fell substantially against multicore HMMER.
* Sequence-length sorting substantially affected performance.
* Fixed-size sequence blocks limited occupancy.
* Forward acceleration was identified as future work.

**License warning:** the released code is GPLv3. Inspect it for understanding and historical comparison, but **do not copy GPL source into this project** unless Jacob explicitly authorizes adopting GPL licensing.

### Ahmed et al., 2012

*Hotspot Analysis Based Partial CUDA Acceleration of HMMER 3.0 on GPGPUs.*

Relevant because it explicitly investigated partial/hybrid acceleration based on pipeline hotspots and CPU↔GPU transfer cost.

### Jiang et al., 2016 — CUDAMPF

Paper:

```text
CUDAMPF: a multi-tiered parallel framework for accelerating
protein sequence search in HMMER on CUDA-enabled GPU
DOI: 10.1186/s12859-016-0946-4
```

Repository:

```text
Super-Hippo/CUDAMPF
```

Study its:

* multi-sequence mapping;
* byte-packed MSV arithmetic;
* 16-bit P7Viterbi arithmetic;
* shared/local memory strategies;
* coalesced sequence layouts;
* query-length-dependent kernels;
* NVRTC specialization.

The repository is MIT licensed, but it targets approximately CUDA 7 / Kepler-era hardware. Treat it as an algorithmic reference, not a codebase to renovate blindly.

### Jiang et al., 2017/2018 — CUDAMPF++

Study its multi-sequence MSV/SSV execution and resource-exhaustion strategy. It is primarily a filter-kernel accelerator rather than a modern full HMMER backend.

### Anderson & Wheeler, 2024 — HAVAC

Recent FPGA implementation of HMMER's SSV-style filtering. Relevant for modern accelerator architecture and data movement, but not our intended protein/CUDA implementation.

### Current collision risk: NERSC/JGI HMMER-GPU

NERSC currently lists an active **HMMER-GPU** NESAP project led by Kjiersten Fagnan at LBNL/JGI, explicitly aimed at porting and optimizing HMMER for GPUs.

No public implementation was located during initial reconnaissance.

**Before making any novelty claim, search again for public outputs from this project. If code or a technical report appears, stop and compare its scope with this plan before duplicating substantial engineering work.**

---

## Scientific Non-Negotiables

1. **HMMER3 defines correctness.**
2. We are accelerating execution, not changing Plan7.
3. Do not alter MSV, Viterbi, Forward, bias-filter, null2, threshold, domain, or E-value semantics to obtain speed.
4. GPU filter pass/fail decisions must agree exactly with the reference CPU pipeline.
5. When the downstream pipeline remains CPU HMMER, final reported targets, domains, scores, E-values, coordinates, and envelopes must match the CPU reference.
6. Optimizations that sacrifice sensitivity are unacceptable.
7. CPU HMMER/PyHMMER must always remain a functional fallback.
8. `--max` can initially fall back completely to the CPU because it bypasses the acceleration filters.
9. Benchmark with real biological data. **Do not generate dummy or synthetic benchmark datasets.** Upstream HMMER test fixtures may be used for unit-level correctness testing, but performance results must use real sequence/profile collections.

---

## Phase 0 — Environment and Collision Reconnaissance

Before modifying HMMER or writing CUDA:

### Hardware inventory

Record:

```bash
nvidia-smi
nvidia-smi topo -m
nvcc --version
lscpu
numactl --hardware
```

Also inspect Slurm:

```bash
sinfo
scontrol show partition
```

Determine:

* GPU model(s);
* compute capability;
* VRAM;
* CUDA driver/toolkit;
* PCIe versus NVLink topology;
* CPU model;
* sockets;
* physical cores;
* SMT configuration;
* NUMA topology;
* node RAM.

Write machine-readable results to:

```text
results/hardware.json
```

and a concise interpretation to:

```text
ENVIRONMENT.md
```

If Hopper (`sm_90`) or newer hardware is available, note that DPX optimization is possible **after** the reference MSV kernel is correct.

### Software references

Build clean reference versions locally in the project rather than trusting arbitrary system modules:

```text
HMMER 3.4
PyHMMER 0.12.1
```

Record compiler flags and exact revisions.

Run the complete HMMER test suite before using the build as an oracle.

### Prior-art reconnaissance

Produce:

```text
PRIOR_ART.md
```

It must summarize at minimum:

* cudaHmmsearch;
* Ahmed et al. partial HMMER3 acceleration;
* CUDAMPF;
* CUDAMPF++;
* HAVAC;
* current NERSC/JGI HMMER-GPU effort;
* any additional modern implementation discovered.

Explicitly distinguish:

```text
historical concept already demonstrated
        vs.
currently available maintained software
        vs.
our proposed implementation
```

Do not proceed on the assumption that this is conceptually unprecedented.

---

## Benchmark Corpus

Do not fabricate data.

### Historical Astra workload

First attempt to locate the East River FASTAs previously used for Astra benchmarking. Known candidate paths include:

```text
/groups/banfield/projects/environmental/EastRiver/hillslope/assembly.d/PLM_jun2017/PLM2_5_b1/PLM2_5_b1_jun17_idba_ud/PLM2_5_b1_jun17_scaffold_min1000.fa.genes.faa

/groups/banfield/projects/environmental/EastRiver/Vegtype/assembly.d/H2a2/H2a2_full_idba_ud/H2a2_full_scaffold_min1000.fa.genes.faa

/groups/banfield/projects/environmental/EastRiver/riparian/assemblies_for_missing_samples/ERMLT890.contigs.fa.genes.faa

/groups/banfield/projects/environmental/EastRiver/2020/assembly.d/EastRiver_08_16_2020_HR_Copper_idba_ud/EastRiver_08_16_2020_HR_Copper_scaffold_min1000.fa.genes.faa

/groups/banfield/projects/environmental/EastRiver/2020/assembly.d/EastRiver_08_14_2020_HR_Shumway_idba_ud/EastRiver_08_14_2020_HR_Shumway_scaffold_min1000.fa.genes.faa
```

Verify paths before use. Do not silently substitute nonexistent files.

### HMM collections

Locate installed Astra resources, preferably through the existing Astra configuration.

Primary benchmark databases:

```text
KOFAM
PFAM
```

Secondary/small-profile benchmarks:

```text
HydDB
MopB models if available
```

Use the exact database artifacts present on the cluster and record their identity.

### Workload classes

Use four real workload regimes:

**A. Mostly-negative large search**

```text
KOFAM × one or more large real metaproteomes
```

**B. Domain-rich search**

```text
PFAM × the same real protein corpus
```

**C. Small HMM collection**

```text
HydDB or MopB × large real protein corpus
```

**D. Survivor-enriched stress test**

Run reference HMMER first, extract real sequences that survive MSV/Viterbi or produce strong hits, and construct a real-sequence corpus enriched for downstream work.

This workload is specifically intended to expose whether GPU MSV merely moves the bottleneck into CPU Viterbi/Forward.

---

## Benchmark Method

Measure both **kernel/component performance** and **actual end-to-end wall time**.

CPU baselines:

```text
HMMER 3.4 hmmsearch
PyHMMER 0.12.1 hmmsearch
current Astra search
```

Thread counts:

```text
1
4
8
16
32
maximum physical cores available
```

Do not count SMT threads as separate physical-core scaling points unless explicitly testing SMT.

For each benchmark record:

```text
wall time
CPU time
peak RSS
input sequence count
input residue count
profile count
total profile states
MSV entrants/passes
bias passes
Viterbi passes
Forward passes
reported targets
reported domains
output bytes
```

For GPU runs additionally record:

```text
kernel time
H→D transfer time
D→H transfer time
VRAM peak
GPU utilization
memory bandwidth if available
GCUPS
```

Use Nsight Systems/Compute where useful.

Run compute-only and full end-to-end benchmarks separately so serialization and filesystem I/O do not get confused with Plan7 kernel performance.

Perform at least three measured replicates after basic warm-up when practical.

---

## Milestone 1 — Build an Exact CPU Oracle

This comes before CUDA.

Create a small diagnostic executable or patch linked against HMMER 3.4 internals.

For manageable HMM × sequence sets, emit per-comparison diagnostic information sufficient to reconstruct acceleration decisions:

```text
model_id
sequence_id

MSV raw/filter score
MSV bit score if distinct
MSV P-value
MSV pass/fail

bias-filter result

Viterbi score
Viterbi P-value
Viterbi pass/fail

Forward score
Forward P-value
Forward pass/fail
```

Also retain HMMER's aggregate pipeline counters.

Relevant HMMER code begins around:

```text
src/p7_pipeline.c
src/impl_*/msvfilter.c
src/impl_*/vitfilter.c
src/impl_*/fwdback.c
```

Inspect exact overflow/saturation and fallback behavior. Do not assume the common path is the only path.

The oracle must test:

* short and long HMMs;
* short and long real proteins;
* ambiguous residues;
* compositionally biased proteins;
* multi-domain proteins;
* MSV saturation/overflow behavior;
* altered `F1` thresholds;
* standard default pipeline behavior.

Select real proteins/HMMs across observed length quantiles rather than inventing examples.

### Gate 1

No GPU implementation work is considered valid until the CPU oracle is deterministic and trusted.

---

## Milestone 2 — Minimal CUDA MSV

Implement only:

```text
one HMMER3 optimized profile
×
a batch of real digital protein sequences
→
MSV filter scores / decisions
```

Do not integrate Astra.

Do not optimize aggressively initially.

Prefer an independent modern implementation based on the published algorithm and current HMMER source semantics. Do not copy GPL cudaHmmsearch code.

Input representations should originate from HMMER's current `P7_OPROFILE` / Easel digital-sequence state wherever feasible. Avoid inventing a permanently separate scientific representation.

### Gate 2

Required:

```text
100% MSV pass/fail agreement with HMMER
```

Desired:

```text
bit-identical integer/filter score before score conversion
```

If score equality is impossible because of a documented representation boundary, characterize the difference exhaustively and still require identical decisions.

Only begin performance tuning after this gate passes.

---

## Milestone 3 — Production-Scale GPU MSV

Extend to:

```text
many HMMs × many sequences
```

Investigate:

* sequence-length bucketing;
* multi-sequence warp execution;
* contiguous/coalesced digital-sequence layout;
* profile-length-dependent kernel strategies;
* packed 8-bit arithmetic matching HMMER;
* asynchronous H→D transfer;
* pinned host memory;
* CUDA streams;
* persistent sequence batches;
* work distribution by actual model/sequence length;
* profile caching;
* current warp primitives such as `_sync` shuffles;
* minimizing CPU/GPU synchronization.

Do **not** inherit pre-Volta warp-synchronous assumptions from old CUDA implementations.

If Hopper or newer hardware is available, create a separate experimental kernel using CUDA DPX instructions only after the portable reference CUDA kernel is correct. Measure whether DPX materially improves the HMMER recurrence rather than assuming it will.

### Gate 3

The batched implementation must preserve all Gate-2 results over the complete validation corpus.

Establish the workload-size break-even point at which GPU MSV becomes faster than current multicore CPU execution.

---

## Milestone 4 — Actual Hybrid HMMER

Create the narrowest possible integration seam in HMMER.

Desired architecture:

```text
GPU computes HMMER-equivalent MSV result
              ↓
HMMER pipeline resumes immediately after MSV
              ↓
existing CPU bias
existing CPU Viterbi
existing CPU Forward
existing CPU domain inference
existing CPU reporting
```

Do not fork/reimplement downstream HMMER logic.

Investigate whether `p7_pipeline.c` can be minimally refactored to accept an externally supplied MSV result rather than duplicating pipeline code.

The first hybrid executable may be standalone, e.g.:

```text
gpu-hmmsearch
```

It does **not** need PyHMMER integration yet.

### Gate 4 — End-to-End Scientific Equivalence

For every benchmark input, CPU HMMER and hybrid HMMER must agree on:

```text
reported targets
included targets
full-sequence scores
full-sequence E-values
domain count
domain scores
domain E-values
HMM coordinates
target coordinates
envelope coordinates
reported/included domain states
```

Because downstream inference remains CPU HMMER, discrepancies should be treated as bugs, not accepted as approximate agreement.

### Gate 4 — Performance

Compare against optimized modern multicore HMMER/PyHMMER, not merely one CPU core.

Interpretation:

```text
<1.5× end-to-end speedup:
    pause and profile before continuing

1.5–3×:
    worthwhile result; determine bottleneck

>3×:
    strong result; proceed aggressively
```

These are project heuristics, not publication thresholds.

---

## Milestone 5 — Reprofile Before Porting Anything Else

After GPU MSV succeeds, collect an actual stage breakdown.

Example only:

```text
Before:
MSV        60%
bias        8%
Viterbi    20%
Forward     8%
other       4%

After:
GPU MSV     5%
transfer     4%
bias        15%
Viterbi     45%
Forward     25%
other        6%
```

Do not assume Viterbi is the next target. Let measured workloads decide.

---

## Milestone 6 — GPU P7Viterbi, Conditional

Only proceed if P7Viterbi materially limits end-to-end performance.

Two architectures are acceptable for evaluation:

```text
GPU MSV
 ↓
CPU bias
 ↓
GPU P7Viterbi
 ↓
CPU Forward/domain pipeline
```

or, if CPU round-tripping itself becomes significant:

```text
GPU MSV
 ↓
GPU-compatible bias stage
 ↓
GPU P7Viterbi
 ↓
CPU Forward/domain pipeline
```

Again require exact HMMER filter decisions.

Study CUDAMPF's published P7Viterbi/Lazy-F strategy, but implement against current HMMER semantics and modern CUDA hardware.

---

## Forward Is a Separate Project

Do not implement GPU Forward merely because it is interesting.

Only consider it when:

1. GPU MSV is correct and useful.
2. GPU Viterbi is correct and useful if needed.
3. The CPU Forward/domain tail is now demonstrably the dominant runtime.
4. Further acceleration has a meaningful end-to-end payoff.

Until then, leaving Forward/Backward/domain inference in trusted HMMER code is an architectural advantage.

---

## Long-Term Integration Target

Only after the standalone implementation passes all scientific and performance gates:

```python
pyhmmer.hmmsearch(
    profiles,
    sequences,
    cpus=...,
    backend="cuda",
)
```

or an equivalent optional accelerator API.

Astra should eventually need little more than:

```bash
astra search ... --backend cuda
```

Desired invariant:

```text
same HMM files
same FASTA files
same thresholds
same HMMER scientific semantics
same result objects
different execution backend
```

No GPU-specific HMM format should be imposed on users.

---

## First Codex Run: Required Deliverables

Work autonomously through the following sequence:

1. Create the standalone project workspace.
2. Inventory CPU/GPU/Slurm/CUDA environment.
3. Build and test HMMER 3.4.
4. Install/build PyHMMER 0.12.1 in an isolated environment.
5. Locate real KOFAM/PFAM and benchmark FASTA data.
6. Produce `PRIOR_ART.md`.
7. Produce `BENCHMARK_SPEC.md`.
8. Implement baseline benchmark runners.
9. Produce baseline timing and stage-survival results.
10. Inspect HMMER's MSV/pipeline internals and design the oracle.
11. Implement the CPU MSV oracle.
12. Validate the oracle on real data.
13. If and only if the oracle is trustworthy, scaffold the minimal CUDA MSV implementation.

Expected project structure:

```text
plan7-gpu/
├── PRIOR_ART.md
├── BENCHMARK_SPEC.md
├── ENVIRONMENT.md
├── README.md
├── baseline/
├── oracle/
├── cuda/
├── scripts/
├── results/
│   ├── hardware.json
│   └── baseline/
└── refs/
```

Do not fabricate benchmark values.

Do not create synthetic biological benchmark data.

Do not modify Astra during this phase.

Do not copy GPL cudaHmmsearch implementation code.

Commit coherent milestones separately so regressions can be bisected.

At the end of the first run, report:

```text
1. hardware available
2. exact software versions
3. real benchmark datasets located
4. baseline performance
5. measured HMMER stage-survival fractions
6. measured stage timings if obtainable
7. oracle implementation status
8. exact technical obstacle to GPU MSV, if any
9. whether any previously unknown modern GPU-HMMER implementation was discovered
10. recommended next coding step
```

The first objective is **not** to show a GPU speedup.

The first objective is to establish a reference against which any future GPU speedup can be trusted.
