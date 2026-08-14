# Benchmark specification

## Purpose

Measure scientific equivalence and end-to-end value of a hybrid HMMER 3.4
accelerator. Kernel GCUPS is diagnostic; wall-clock speedup against optimized
multicore HMMER/PyHMMER on the same hardware is the decision metric.

## Immutable provenance

Every run records:

- host, allocation, CPU/GPU identity, affinity, NUMA placement, and software;
- exact command, working directory, environment controls, and exit status;
- HMM/FASTA canonical path, byte size, modification time, SHA-256, record
  count, residue count, profile count, and total profile states;
- HMMER/PyHMMER/Astra version and Git revision where applicable; and
- warm-up policy, replicate index, output path, and output byte count.

The initial corpus is frozen in `results/datasets.json`. Database upgrades are
new benchmark datasets, never silent replacements.

## Real workloads

All timed biological inputs are real. Deterministically extracted real-record
subsets may be used for scaling and break-even curves. Constructed inputs are
restricted to correctness/property tests and are never performance results.

### A. Mostly-negative large search

Current KOFAM against the PLM2_5 East River metaproteome, followed by the five
FASTA aggregate when practical.

### B. Domain-rich search

Current Pfam-A against the same corpus.

### C. Small profile collection

The three current HydDB models against complete East River FASTAs. Do not use
`--cut_ga`: the installed HydDB manifest claims cutoffs, but its HMM text has
no GA/TC/NC fields.

### D. Survivor-enriched tail stress

Run pristine HMMER first, retain real sequences that pass MSV/Viterbi or
produce strong hits, and freeze their original identifiers and source offsets.
This corpus measures the CPU downstream bottleneck after GPU filtering.

## Execution baselines

- pristine HMMER 3.4 `hmmsearch` built by `scripts/build_hmmer.sh`;
- PyHMMER 0.12.1 in `env/pyhmmer-0.12.1`;
- current Astra at its recorded revision; and
- hybrid builds, only after their correctness gate passes.

Primary CPU counts are 1, 4, 8, 16, 32, and the maximum allocated physical
cores. SMT is a separate experiment. HMMER CLI and PyHMMER/Astra scheduling
must be reported independently because they distribute multiple queries
differently.

Each reported point uses one warm-up and at least three measured replicates
when practical. Compute-only and end-to-end I/O runs are distinct. Reported
runs use uncontended allocations and explicit CPU/memory affinity.

## Measurements

For every run:

- wall, user, and system time; peak RSS; input/output bytes;
- sequences, residues, profiles, and total profile states;
- MSV, bias, Viterbi, and Forward promotions; targets and domains; and
- final command status and validation status.

For instrumented builds, record actual stage calls and elapsed time for
reconfiguration, null1, SSV, full-MSV fallback, bias, Viterbi, Forward,
Backward, and domain inference. HMMER's `n_past_*` counters are promotion
counters and must not be presented as call counts.

For GPU runs additionally record kernel, H2D, D2H, and synchronization time;
peak VRAM; utilization; achieved bandwidth; GCUPS with its exact denominator;
batch/profile/sequence layout; and candidate count.

## Gate 0: Amdahl bound

On at least one representative Astra workload, measure the actual SSV-first
stage fraction before CUDA. If fraction `f` is stage-1 time and acceleration is
`s`, report `1 / ((1-f) + f/s)` plus the infinite-acceleration ceiling
`1 / (1-f)`. This gate determines whether MSV-only work can matter.

## Scientific equivalence

The reference and hybrid must agree on target/report/include state, full
scores/E-values, domain count/state, domain scores/E-values, HMM and target
coordinates, and envelope coordinates. Also compare pipeline counters and
input accounting. Comparison is exact except for explicitly textual fields
such as timing lines.

The first hybrid is conservative: GPU stage-1 candidates are rerun through
untouched CPU `p7_Pipeline()`. False positives affect speed only; any false
negative is a release-blocking correctness failure.
