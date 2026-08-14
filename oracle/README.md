# Byte-MSV oracle

`msv_oracle.c` evaluates pristine HMMER 3.4's public SSV-first
`p7_MSVFilter()` and an independent scalar implementation of the full
unsigned-byte MSV recurrence. The scalar path uses only the unstriped emission
cost array extracted from the optimized profile; it does not call or copy the
striped SIMD recurrence.

Build and run the small upstream fixture:

```sh
make test
```

Run a real profile/sequence differential and reduce its verbose JSONL:

```sh
build/oracle/msv-oracle --max-models 0 --max-seqs 100 --strict \
  results/datasets/oracle-hmm-quantiles.hmm \
  results/datasets/PLM2_5.first1000.faa \
  >results/oracle/hmm-quantiles-plm2_5-21x100.jsonl
python3 scripts/summarize_oracle.py \
  results/oracle/hmm-quantiles-plm2_5-21x100.jsonl \
  results/oracle/hmm-quantiles-plm2_5-21x100-summary.json
```

The generated HMM/FASTA extracts and verbose JSONL are intentionally ignored;
the frozen source inventories, extraction recipe, and compact summaries are
tracked.

The initial four runs total 11,145 comparisons with zero status or score-bit
mismatches. Together they cover `ssv_ok`, `ssv_erange`,
`ssv_fallback_msv_ok`, and `ssv_fallback_msv_erange`, model lengths 7 through
17,019, exact SIMD stripe boundaries at 15/16/17 and 31/32/33, real sequence
lengths 32 through 1,718, and 17 observed values of the length-dependent
`tjb_b` transition. Unit fixtures additionally exercise B/J/Z/X/U/O ambiguity
codes, nonresidues, empty targets, and the 100,000-residue pipeline limit.

CUDA remains gated on transition quantization boundaries, threshold-neighbor
sweeps, controlled overflow fixtures, actual pipeline tracing, and end-to-end
output/counter equivalence.

Build and smoke-test it from the repository root with `make test`. The output
is JSON Lines: metadata, one comparison record per nonempty model/sequence
pair, and a summary. HMMER's protein pipeline returns before null1/MSV for a
zero-length target, so those pairs are not passed to the filter API and do not
produce comparison records. The summary reports them separately as
`skipped_empty`.

`--max-models 0` and `--max-seqs 0` mean unlimited. `--F1` must be a finite
probability in `[0,1]`; `--strict` returns failure if the scalar and public
HMMER MSV status or score bits disagree. Like HMMER's protein comparison
pipeline, the oracle rejects targets longer than 100,000 residues.
