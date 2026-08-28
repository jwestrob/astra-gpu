# Exact within-profile continuation sharding

Hypothesis: the retained completion-driven scheduler still ends chunks behind
one pathological profile because a profile row is indivisible. Sparse-v3
exceptions are target-independent under HMMER's default deterministic
reseeding, so a heavy profile can be split at certificate/exception boundaries
and recombined with HMMER's database-partition merge.

Implementation:

- `ebaf573` adds a private exact sparse-v3 exception-range consumer. Only the
  last shard owns the terminal certificate and final target-length state.
- `0ea0e83` adds an explicit `sharded` scheduler policy. It splits only a
  single-profile task costing more than twice the balanced task target, copies
  the mutable optimized profile per concurrent shard, buffers partial results,
  and uses the final shard as the base for `TopHits.merge`.
- The retained `balanced` policy remains the default.

Correctness evidence: a host fixture and a real concurrent CUDA-generated
sparse batch produced byte-identical target/domain tables and identical HMMER
accounting to unsharded continuation. Six focused retained/sharded scheduler
tests passed.

Microbenchmark: 512 authenticated full-pipeline exceptions for one profile,
eight workers, three alternating repetitions. Unsharded median was 1.222742 s;
sharded median was 0.161588 s, a 7.567x speedup. Target and domain table hashes
were respectively `921ca92e709c2121366943d96533050f121dac52cf6582aa6fff3cc46267a3c3`
and `19624d6e7bef09a5a1c7b2fb3b272e23793867bab6dc16fafd1d5c2eefd087e0`.

Full H200 Pfam x PLM evaluation is queued as Slurm job `1185224`. Retain or
reject only after exact full-output and end-to-end timing evidence.
