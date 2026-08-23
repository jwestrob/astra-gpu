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

`CandidateBatch.search(..., return_telemetry=True)` returns
`(TopHits, continuation_statistics)` for a sealed fused batch. The search
record names the selected continuation path, the whole target count, the
retained postfilter-record count, the omitted F1 rejects, the terminal route
partition, journal routes, compact accept/retry counters, and wall time for
that one profile. Astra's bridge preserves query order and forwards the same
opt-in tuple in serial and threaded modes.

The facts are transition history, not mutually exclusive terminal labels. A
row can carry an early numerical/cap fact and a later row-atomic or host
validation fact. Facts known only while the CPU consumes the journal—such as
compact threshold and invalid-result retries—are intentionally recorded in
continuation telemetry rather than inferred during generation.

Disabled mode does not allocate, upload, execute, or download native reason
buffers. Backward/domain and rescore compile separate no-fact kernel variants;
postfilter calls its unchanged native entry point. The package-private sidecar
is trusted transport and is not added to the semantic journal integrity hash.
The defensive Python dictionaries and tuples used to expose telemetry are
ordinary object overhead and are intentionally outside `CandidateBatch`'s
`resident_bytes` contract, which measures exact retained backing-buffer
payload rather than Python-object RSS.
