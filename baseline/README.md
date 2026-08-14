# Baselines

Reference-timing tools for pristine HMMER 3.4 and application measurements for
Astra and the hybrid executable live here. Raw measurements are written beneath
`results/baseline/` with full command and dataset provenance.

`run_hmmer_matrix.py` records pristine CLI scaling and writes both target and
domain tables so worker-count equivalence is checked on scientific rows, not
only pipeline counters. Application-level baselines use the existing Astra
implementation; this project does not maintain a parallel PyHMMER frontend.

`astra_probe/` observes stage calls inside that existing Astra process through
an opt-in preload shim; it neither modifies Astra nor implements a search
frontend. `patches/` contains the isolated pristine-HMMER timing reference.
Both observers expose their overhead and keep pilot timing separate from
scientific output equivalence.
