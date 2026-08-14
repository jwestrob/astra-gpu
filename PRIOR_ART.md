# Prior art

Reconnaissance refreshed 2026-08-14. This project makes no claim that GPU MSV
or hybrid HMMER execution is conceptually new. Its intended contribution is a
maintained, exact HMMER 3.4-compatible implementation with honest modern
multicore and end-to-end comparisons.

## Direct HMMER GPU work

### cudaHmmsearch (Cheng, 2014)

- [Thesis](https://spectrum.library.concordia.ca/id/eprint/978785/)
- [Source](https://github.com/forestcheng/cudahmmsearch), GPL-3.0
- HMMER 3.1b1-era CUDA MSV followed by the CPU downstream pipeline.
- Reported approximately 2.53x average speedup over one CPU core, but only
  approximately 1.33x over a six-core baseline; sequence sorting mattered.

This is the closest historical architectural precedent. It is read-only prior
art: no GPL source may be copied into `plan7-gpu`.

### Ahmed et al. (2012)

- [Paper](https://www.ijsce.org/wp-content/uploads/papers/v2i4/D0894072412.pdf)
- Partial CUDA study of jackhmmer Forward, Backward, and P7Viterbi hotspots,
  rather than the current protein SSV/MSV-first target.
- Its difference between kernel and combined speedup is an early warning that
  allocation and transfer costs can dominate.
- No public source or source license was found.

### CUDAMPF (Jiang et al., 2016)

- [Paper](https://doi.org/10.1186/s12859-016-0946-4)
- [Source](https://github.com/Super-Hippo/CUDAMPF), MIT
- HMMER 3.1b2-era standalone SSV/MSV/P7Viterbi filter framework using
  multi-sequence mappings, packed arithmetic, coalesced layouts,
  query-dependent kernels, and NVRTC specialization.
- Its reported outputs and timings are filter-oriented, not exact complete
  `hmmsearch` output or modern end-to-end application timing.

The license is compatible, but the Kepler/CUDA-7 implementation is an
algorithmic reference rather than a codebase to transplant.

### CUDAMPF++ (Jiang et al., 2017/2018)

- [Preprint](https://arxiv.org/abs/1707.09683)
- [TPDS paper](https://doi.org/10.1109/TPDS.2018.2830393)
- Multi-tier MSV/SSV filter kernels and proactive resource exhaustion.
- No public CUDAMPF++ source/license was located. CUDAMPF's MIT license must
  not be assumed to cover unpublished ++ code.

### HAVAC (Anderson and Wheeler, 2024)

- [Paper](https://doi.org/10.1186/s12859-024-05879-3)
- [Source](https://github.com/TravisWheelerLab/HAVAC), BSD-3-Clause
- FPGA/Alveo U50 nucleotide SSV-style filtering for nhmmer, without the
  downstream Viterbi/Forward pipeline.

HAVAC is relevant to modern streaming and data-movement architecture, but is
not a CUDA/protein `hmmsearch` collision.

## Current NERSC/JGI work

- [Official NERSC HMMER-GPU project](https://www.nersc.gov/what-we-do/support-for-scientists/nersc-science-acceleration-program/nesap-for-doudna)
- [JGI CPU benchmark repository](https://github.com/JGI-Bioinformatics/hmmer-benchmarking), no license found
- [Public `msv-filter` repository](https://github.com/mamelara/msv-filter), no license found

The NERSC project publicly states a GPU-port/optimization goal. JGI has also
released Perlmutter CPU benchmark scripts/data. As of this review, no public
CUDA kernel or complete hybrid pipeline was present.

`mamelara/msv-filter` is a February-May 2026 public C++ reimplementation of
generic floating/log-space `p7_GMSV`, with an upstream-HMMER parsing bridge and
small parity tests. Its README contains a JGI/NESAP development path; combined
with its authorship overlap with the JGI benchmark repository, this strongly
suggests related groundwork, but that relationship is an inference rather
than an official repository declaration. It currently has no CUDA, no packed
SSV/MSV stage, no pass/fail pipeline integration, and no source license.

This is a real public development trail and must be rechecked before novelty
or priority claims. We may reproduce the published JGI benchmark workload for
an apples-to-apples comparison, subject to confirming data terms. Their
unlicensed source is read-only and must not be copied.

## Adjacent modern work

- [ApHMM-GPU](https://github.com/CMU-SAFARI/ApHMM-GPU), GPL-3.0: CUDA
  Baum-Welch Forward/Backward training for Apollo polishing, not Plan7 search.
- [nail](https://github.com/TravisWheelerLab/nail), BSD-3-Clause: active
  CPU/Rust search using MMseqs2 candidates and sparse approximate
  Forward/Backward; intentionally not exact HMMER semantics.
- [rustyhmmer](https://github.com/necoli1822/rustyhmmer), BSD-3-Clause: new
  pure-Rust CPU `hmmsearch` implementation claiming HMMER 3.4 output parity;
  useful as a possible independent comparator, not a GPU accelerator.

## Scope distinction

| Category | Current finding |
|---|---|
| Historical concept | GPU SSV/MSV and GPU-filter/CPU-tail hybrids are established prior art. |
| Maintained exact HMMER 3.4 CUDA backend | None found publicly. |
| JGI collision | Active project plus public CPU/generic-MSV groundwork; no public CUDA/full pipeline yet. |
| `plan7-gpu` target | Exact packed stage-1 semantics, conservative CPU revalidation, persistent matrix scheduling, and modern end-to-end evidence. |
