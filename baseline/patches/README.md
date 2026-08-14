# HMMER 3.4 stage timing patch

`hmmer-3.4-stage-timing.patch` adds an opt-in, hidden `hmmsearch`
`--stage-timing <path>` option. Normal search output remains on stdout; timing
rows go only to the explicit path. MPI plus timing is rejected. Serial and
threaded searches are supported, and threaded workers accumulate independently
before their counters are merged.

Build and smoke-test an isolated copy without modifying the pristine source:

```bash
scripts/build_timed_hmmer.sh
refs/install/hmmer-3.4-stage-timing/bin/hmmsearch \
  --cpu 4 --stage-timing run.tsv query.hmm proteins.faa
baseline/patches/validate_stage_timing.py run.tsv
```

The TSV is long-form schema version 1. Every query has 25 metrics: 11 timed
stages, seven MSV status counters, one `clock_errors` diagnostic, target
sequences/residues, and four filter survivor counters. The validator rejects
any clock-read error. `query_index` disambiguates duplicate HMM names.

Interpretation details:

- `pipeline_total` times each `p7_Pipeline()` call and contains the other
  pipeline stages; do not add it to them. It excludes sequence input and
  decoding, `p7_pli_NewSeq()`, target/background reconfiguration, work-queue
  scheduling, query setup, worker merging, thresholding, and result output.
- `msv_public` contains the SSV attempt and, only after `eslENORESULT`, the
  classic MSV fallback. `eslERANGE` is counted separately and is not a fallback
  trigger.
- `domain_workflow` is inclusive of HMMER's downstream domain-definition work,
  including its internal full Forward/Backward computations.
- Threaded `elapsed_ns` values are summed worker work time, not wall time. Use
  serial timings only for stage fractions or Amdahl-law projections.
- Timing uses `CLOCK_MONOTONIC_RAW` where available, otherwise
  `CLOCK_MONOTONIC`. With timing disabled, the hot path does not read a clock.
- The observer has measurable cost: each timed call performs two clock reads,
  nested timing perturbs `pipeline_total`, and no overhead is subtracted.
  Measure paired timing-on/off runs and treat very short stage times cautiously.

Set `PLAN7_SKIP_CHECK=1` to omit the upstream full test suite during a rebuild;
the timing-specific serial/threaded smoke test still runs. Set
`PLAN7_BUILD_JOBS` to control compilation parallelism.
