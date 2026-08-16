# Post-Forward census probe

`probe.c` interposes on pinned HMMER 3.4's post-Forward calls and records the
route, dimensions, decision margins, logical scratch footprint, and stage
timings for every F3 survivor. It is observational: the ordinary HMMER result
must retain the reference hash.

Build it with `build.sh`, then load the resulting shared object with
`LD_PRELOAD` for the pinned PFAM-versus-first-1000 search. The measured inputs
were:

- `Pfam-A.hmm`: SHA-256 `a78a42d6faf265b6bfca59e8f062d06fae6083ce2c6e335d7b381f20b82b7903`
- `PLM2_5.first1000.faa`: SHA-256 `b835fa20310971a507f5067f7136291cf2a4f5671e7ad64cf503c258cf24db2b`
- census TSV: SHA-256 `727b9b9422f35eedd6f3e29f7f5dcdb1071651bf107ef5fef681f4ca3a140f0d`
- reference output: SHA-256 `5b489418b89fb803fbb6815342d1115ce951dba792068187574097461834dbdb`

The census contained 4,170 F3 survivors: 2,879 simple-only rows and 1,291
rows requiring stochastic clustering. A conservative `2e-4` guard leaves
2,562 simple rows (61.4%) eligible. There were 7,257 regions, 1,425 clustered
regions, and 8,138 envelopes/domains. Total logical Backward/domain storage
was 63,462,108 bytes; the maximum per row was 67,980 bytes.

The generated TSV is deliberately not committed. Its measured location was
`build/post-forward-census/pfam-first1000.tsv`.
