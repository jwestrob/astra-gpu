# Phase 7: sparse full-MSV launch compaction

## Hypothesis

The postfilter launches the exact full-MSV recurrence over every retained F1
candidate and returns immediately for rows whose SSV status is not
`eslENORESULT`. On the sealed full workload only 467,289 of 203,671,109
postfilter records require that recurrence. Launching one warp for every row
therefore spends most of the launch topology on no-op warps and also builds a
dense MSV DP-offset vector that is never consumed.

## Implementation

For candidate sets of at least 65,536 rows, CUB `DeviceSelect::If` now scans
the existing device SSV status records in bounded four-million-row chunks and
emits stable original candidate indexes only for `eslENORESULT` rows. The host
downloads those sparse indexes solely to construct exact DP byte offsets from
the already authenticated profile descriptors. The unchanged full-MSV kernel
then consumes the sparse index vector and writes results back at the original
row indexes.

The selection is stable, so diagnostic execution order remains original-row
order. Postfilter records, F1/F2 decisions, reason facts, journal layout, and
all downstream ABIs are unchanged. Small candidate sets retain the previous
direct launch. `PLAN7_GPU_FULL_MSV_POLICY=legacy` forces the reference path;
`compact` forces sparse selection for exact comparison.

The existing MSV-offset device allocation temporarily owns the bounded sparse
index and offset regions, and the existing DP allocation doubles as CUB
temporary storage before recurrence execution. No new persistent allocation
class is introduced. Counters expose source, selected, launched, avoided, and
sparse-index transfer totals.

## Acceptance

- Forced legacy and compact paths produce byte-identical postfilter offsets
  and complete records, including exact full-MSV success and range outcomes.
- Selected indexes are strictly increasing and remain inside their source
  chunk; counts and avoided launches reconcile exactly.
- Small workloads continue through the legacy path automatically.
- A representative/full H200 run retains the established final output and
  reports fewer full-MSV launch candidates than source candidates.

H200 evidence and the retain/reject decision will be appended after execution.
