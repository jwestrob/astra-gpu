# Astra HMMER stage probe

This is an observer for the existing Astra executable, not another search
frontend. It uses Linux `LD_PRELOAD` to interpose the exported HMMER 3.4 ABI in
Astra's bundled `liblibhmmer.so`; Astra and its installed Python environment are
not modified.

Build it with:

```bash
baseline/astra_probe/build.sh
```

The build checks that the HMMER library used by Astra exports every symbol the
probe wraps. To observe a normal Astra invocation:

```bash
PLAN7_ASTRA_STAGE_PROBE=/absolute/path/stages.tsv \
LD_PRELOAD=$PWD/build/astra-stage-probe/libplan7_astra_stage_probe.so \
astra search ...
```

No report is written unless `PLAN7_ASTRA_STAGE_PROBE` names a nonempty path and
that process observes at least one `p7_Pipeline` call. This second guard allows
the preload environment to cover a generic benchmark parent without its
zero-call destructor overwriting the child Astra report. The report is emitted
at normal process exit and records process-wide counts, return-status classes,
and aggregate inclusive nanoseconds for the pipeline, SSV-first MSV path, bias
filter, Viterbi, Forward/Backward parser, and domain workflow entry points.

These timings include observer overhead: two `clock_gettime` calls and relaxed
atomic updates per observed call. Timings are nested (for example, MSV includes
SSV, and the pipeline includes all downstream stages), so stage nanoseconds
must not be summed and should not be presented as uninstrumented performance.
The probe is for tracing the Astra execution path and estimating stage balance;
external wall-time baselines remain authoritative.

Run the focused smoke check with:

```bash
baseline/astra_probe/smoke.sh
```

It runs the existing Astra command twice on 1,000 real PLM2_5 proteins and the
three installed HydDB profiles, uses two worker threads for the observed run,
requires exactly 3,000 pipeline/MSV/SSV observations, validates the TSV schema,
byte-compares Astra's scientific hit table with the uninstrumented run, and
checks that a preloaded process which never calls HMMER emits no report.

Summarize repeated serial controls and observations with:

```bash
baseline/astra_probe/summarize_probe.py \
  --control control-1.json control-1-hits.tsv \
  --control control-2.json control-2-hits.tsv \
  --observed observed-1.json observed-1-probe.tsv observed-1-hits.tsv \
  --observed observed-2.json observed-2-probe.tsv observed-2-hits.tsv \
  --fasta proteins.faa \
  --hmm profiles.hmm \
  --probe-binary build/astra-stage-probe/libplan7_astra_stage_probe.so \
  --output probe-summary.json
```

The summarizer accepts only successful `astra search --threads 1` records
marked as shared pilots, binds control/observed roles and probe environment
paths, and requires commands to differ only in output path. For
`--installed_hmms`, it also resolves the named database through Astra's config
and hashes the exact pressed `.h3*` set (or text HMM files) Astra loads. The
atomic, compact JSON also requires identical call/status counts and hit TSVs
and records medians, observer slowdown, stage times, fallback counts, and
end-to-end Amdahl projections.
Those projections use the median per-run MSV fraction of observed external wall
time; they are feasibility estimates, not performance claims.

`run_paired_pilot.sh` orchestrates that same procedure without implementing a
search frontend:

```bash
baseline/astra_probe/run_paired_pilot.sh \
  results/baseline/pilot-astra-probe-workload workload-id proteins.faa \
  database-source.hmm DATABASE --cut_ga
```

It calls `astra search` directly for warm-up, control, and observed runs, then
writes a compact sibling `-summary.json`. It refuses to overwrite an existing
raw-output directory. `ASTRA_PROBE_REPLICATES` and `ASTRA_PROBE_WARMUPS`
control the default one-plus-one pilot design.
