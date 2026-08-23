# Phase 0 route telemetry

Phase 0 adds an opt-in, version-1 diagnostic sidecar without changing any
postfilter, Forward, Backward/domain, compact-result, provenance, or
continuation-journal-v2 record. The ordinary entry points and Python return
shapes remain the production defaults; detailed reason buffers and Python
evidence exist only when the caller explicitly requests telemetry.

`CandidateBatch.generation_statistics` reports defensive per-profile and
whole-chunk counts, logical work cells, journal row/region attribution, the
exact journal allocation size, native counter reconciliation, and sparse
multi-label reason facts. Reason-attributed cells count only native work that
was actually admitted past each stage's preflight caps. Source-wide contract
fallbacks therefore retain row counts but carry zero downstream work cells.
Postfilter rows carry independent `full_msv_executed` and
`viterbi_executed` facts captured at the exact kernel predicates, so their
work is exactly zero, one, or two times `target_length * model_length`.
`msv_range_state` names the final candidate state and deliberately does not
claim which MSV implementation produced it. The summed execution counts and
cells are independently reconciled against the native reason census.

`CandidateBatch.search(..., return_telemetry=True)` returns
`(TopHits, continuation_statistics)` for a sealed fused batch. The search
record names the selected continuation path, the whole target count, the
retained postfilter-record count, the omitted F1 rejects, the terminal route
partition, journal routes, compact accept/retry counters, and wall time for
that one profile. Astra's bridge preserves query order and forwards the same
opt-in tuple in serial and threaded modes.
The continuation record also carries source-route counts, exact Forward and
journal configuration-bypass facts, exact compact-bypass facts, and the bound
profile/journal-row identity. Validation reconstructs every terminal route
from those source decisions, requires compact attempts to be a subset of the
SIMPLE journal rows, and binds the first attempt to the requested profile,
target range, and journal slice.

The facts are transition history, not mutually exclusive terminal labels. A
row can carry an early numerical/cap fact and a later row-atomic or host
validation fact. Facts known only while the CPU consumes the journal—such as
compact threshold and invalid-result retries—are intentionally recorded in
continuation telemetry rather than inferred during generation.

Disabled mode does not allocate, upload, execute, or download native reason
buffers. Backward/domain and rescore compile separate no-fact kernel variants;
postfilter and Forward call their unchanged native entry points. The opt-in
postfilter marker kernel and Forward reason vector exist only on reason-enabled
entry points. Backward facts use a private uint32 sidecar so every CPU route
has both a terminal fact and at least one source explanation; no result or
statistics structure was widened. The package-private sidecar
is trusted transport and is not added to the semantic journal integrity hash.
The defensive Python dictionaries and tuples used to expose telemetry are
ordinary object overhead and are intentionally outside `CandidateBatch`'s
`resident_bytes` contract, which measures exact retained backing-buffer
payload rather than Python-object RSS.

## Collection and export

`plan7_gpu.telemetry_report.TelemetryCollector` is an explicit-only,
thread-safe join between generation chunks and CPU continuation rows. The
caller supplies immutable global profile ordinals for each chunk; the
collector rejects duplicates, missing continuation rows, and any mismatch
between a continuation's local profile identity and its generation row.
Before work, `bind_expected_profiles(...)` seals the full intended ordinal
universe; complete snapshots and exports fail if any generation chunk or CPU
continuation row is absent.
Snapshots contain raw per-profile generation and continuation evidence, exact
reason-by-row and reason-by-logical-cell tables, per-profile continuation CPU
wall nanoseconds, and a stable cumulative CPU-wall Pareto ordered by
`(-wall_ns, global_ordinal)`.

`TelemetryCollector.export(path)` requires a new path and publishes the whole
report as one same-parent atomic directory rename. It writes canonical JSON,
profile/reason/summary/Pareto TSV files, and a SHA-256 manifest, fsyncs their
contents and directory metadata, and seals the resulting files `0444` and
directory `0555`.

The local Astra bridge does not attempt to synthesize generation telemetry.
`telemetry=True` or an explicit collector accepts only a precomputed sealed
fused `CandidateBatch` that already carries generation evidence; a
`SequenceBatch` or legacy batch fails synchronously before filtering/search
work. With a collector alone, search results retain the ordinary `TopHits`
shape. The separate production Astra integration enables native telemetry
only when a collector is supplied, passes exact original database ordinals for
each chunk, and otherwise preserves its existing generation and consumption
calls.
