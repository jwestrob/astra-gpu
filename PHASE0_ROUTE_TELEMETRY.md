# Phase 0 route telemetry

Phase 0 adds an opt-in, version-2 diagnostic sidecar without changing any
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
profile/journal-row identity. Both generation and continuation records carry
the same exact nonzero session, selection, and target-batch generation
identity copied from the validated sealed journal. Validation reconstructs every terminal route
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
between a continuation and its exact generation row. The join binds the sealed
batch identity, target and postfilter counts, and, on the journal path, the
exact cumulative row span and Backward CPU/no-region/SIMPLE route census.
Continuation evidence is committed only after the row's pipeline cleanup has
succeeded.
Before work, `bind_expected_profiles(...)` seals the full intended ordinal
universe; complete snapshots and exports fail if any generation chunk or CPU
continuation row is absent.
Snapshots contain raw per-profile generation and continuation evidence, exact
reason-by-row and reason-by-logical-cell tables, per-profile continuation CPU
wall nanoseconds, and a stable cumulative CPU-wall Pareto ordered by
`(-wall_ns, global_ordinal)`.

`TelemetryCollector.export(path)` requires a new path. It stages canonical
JSON, profile/reason/summary/Pareto TSV files, and a SHA-256 manifest, fsyncs
their contents and directory metadata, and seals the files `0444`. Filesystems
that support `renameat2(RENAME_NOREPLACE)` publish that stage with one atomic
same-parent rename. On filesystems such as NFSv3 that reject the flag, export
instead reserves the target with exclusive `mkdir` at mode `0700`, copies and
rehashes every staged member through one bound directory descriptor, fsyncs
the directory, then changes its mode to `0555` as the commit point. The
manifest is copied last; pathname identity is checked before and after commit.

A visible mode-`0700` target is deliberately incomplete: a crash or live
failure after reservation leaves it in place, blocks reuse of the name, and
requires audit plus explicit cleanup or a new attempt path. The portable
fallback assumes cooperating publishers do not rename an empty reservation in
the POSIX `mkdir`-to-`open` acquisition window; nonempty substitution is
rejected before mutation and later substitution cannot redirect descriptor-
bound writes or the commit chmod. Once the target reaches `0555`, failure of
its final fsync or the parent-directory fsync raises
`TelemetryReportCommittedError`; callers must audit that exact target and must
not retry publication blindly.

The local Astra bridge does not attempt to synthesize generation telemetry.
`telemetry=True` or an explicit collector accepts only a precomputed sealed
fused `CandidateBatch` that already carries generation evidence; a
`SequenceBatch` or legacy batch fails synchronously before filtering/search
work. With a collector alone, search results retain the ordinary `TopHits`
shape. The separate production Astra integration enables native telemetry
only when a collector is supplied, passes exact original database ordinals for
each chunk, and otherwise preserves its existing generation and consumption
calls.

## H200 evidence

The focused source-fact oracle passed on an attested H200 in Slurm job
`1182345` (exit 0) at source revision
`9a078c22f5ba386fdaeb726a688ef16179bb277c`. It exercised real postfilter,
Forward, Backward/domain, compact-rescore, and CPU-continuation work. The
schema-2 report covered all three expected profiles with no missing generation
or continuation rows and nonzero source-attributed rows, logical cells,
journal matches, compact attempts, compact accepts, and continuation wall
time.

The report was published through the real NFS fallback as one mode-`0555`
directory containing exactly six regular mode-`0444` files. Its SHA-256
manifest (`ca90c552241ed5c0619a2cc3bc2f639c33c58f0f0108303ac452c6e3662ee527`)
rehashes cleanly; the result JSON SHA-256 is
`64e22431e855abbd8e5c816324350e414a8945d68a0c7fb321fa0e7e551fe2c4`.

Full-corpus diagnostic job `1182355` then collected the complete reason census
with byte-identical output. Of 826,453 Backward/domain rows, 552,390 (66.84%)
routed to CPU. The dominant source facts were a conservative work cap on
366,939 rows, multidomain evidence on 150,965 rows, and threshold uncertainty
on 52,797 rows; these facts overlap. Numerical, host-environment, own-scale,
unsupported-mode, workspace-cap, and catchall failures were all zero.

Compact rescore routed 231,725 of 454,912 regions (50.94%) to CPU, entirely
because of the matrix cap; numeric and global-budget fallbacks were zero.
Postfilter CPU rows were only 0.229% of retained records, and Forward output-cap
rows were 0.215% of F2 survivors. The continuation wall was broadly spread
across profiles rather than dominated by a few models: the top 10 profiles
accounted for 5.11%, while 700 profiles were needed to reach 50%. The full
census is retained under `build/phase0-full-census-20260823/attempt-02`.
