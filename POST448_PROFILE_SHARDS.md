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

Full H200 Pfam x PLM evaluation, Slurm job `1185304`, passed the complete
oracle: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines. Request wall was 343.280213 s,
generation 333.695317 s, continuation/output 77.913357 s, measured overlap
76.569992 s, and pipeline wall 335.099062 s. This is 59.776855 s (14.83%)
faster than the retained 403.057068 s line. All 83 chunks used sharding,
creating 6,454 actual shard tasks. Peak RSS was 17,643,588 KiB, higher than the
retained line; combine with request-scoped worker reuse before choosing the
final memory policy.

The worker-pool combination gate, full H200 job `1185307`, also passed the
complete oracle. Request wall was 342.819173 s, generation 333.307039 s,
continuation/output 76.702733 s, measured overlap 75.605812 s, and pipeline
wall 334.455286 s. Peak RSS was 7,420,788 KiB. Relative to sharding alone,
runtime was flat (-0.461039 s, -0.134%) while RSS fell by 10,222,800 KiB
(57.94%). All 83 chunks used sharding and the request-scoped pool.

Decision: retain exact sharding and its request-scoped pool companion. The
combined result is the new full-workload reference: 342.819173 s with exact
output and roughly 7.08 GiB peak RSS.
