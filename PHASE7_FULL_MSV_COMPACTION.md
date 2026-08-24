# Phase 7: sparse full-MSV launch compaction

## Hypothesis

The postfilter launches the exact full-MSV recurrence over every retained F1
candidate and returns immediately for rows whose SSV status is not
`eslENORESULT`. The earlier 467,289-row census was the final MSV-range CPU
fallback count, not the number of rows executing full MSV. The new exact
launch census shows that 40,657,346 of 203,671,109 postfilter records require
the recurrence. The remaining 163,013,763 rows still consumed no-op launch
topology and dense DP-offset planning.

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

## Full H200 result

Job `1182754` passed its exact first-1,000 prerequisite and the full sealed
workload oracle. The final TSV remained byte-identical to Astra CPU64:

- SHA-256: `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`
- bytes: 39,010,327
- lines: 383,235

All 83 full chunks used device compaction. They selected and launched
40,657,346 exact full-MSV rows from 203,671,109 source rows, avoiding
163,013,763 no-op candidate launches (80.038%). Sparse-index D2H was
162,629,384 bytes. No chunk used the legacy launch path.

Request wall was 455.788 seconds, including 346.818 seconds generation,
448.291 seconds pipeline wall, 400.770 seconds continuation/output, and
299.448 seconds overlap. Against Phase 6, request wall regressed 1.541 seconds
(0.339%) and generation regressed 4.170 seconds (1.217%). The structural path
is retained because packed full-MSV arithmetic requires the exact sparse work
list, but no standalone speedup is claimed.

Evidence lives at
`build/h200-phase7-msv-compaction-20260824/attempt-01-full/runs/h200-full`.
The worker JSON SHA-256 is
`856819b4f605c2a0ac91c2c1d34839ba3bd765fb90fffc615e7f5ab0cb8c3f8e`; raw
validation SHA-256 is
`2a6bf3e53c3a55e919de3a3122fa5130b5192bb2f8d9e98af180067d57b94935`.
