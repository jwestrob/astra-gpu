# Baselines

Benchmark runners and parsers for pristine HMMER 3.4, PyHMMER 0.12.1, Astra,
and the hybrid executable live here. Raw measurements are written beneath
`results/baseline/` with full command and dataset provenance.

`run_hmmer_matrix.py` records pristine CLI scaling and writes both target and
domain tables so worker-count equivalence is checked on scientific rows, not
only pipeline counters. Application-level baselines use the existing Astra
implementation; this project does not maintain a parallel PyHMMER frontend.
