# Phase 1B: direct sparse-v3 source

## Hypothesis

When sparse journal v3 is explicitly enabled, the native generation outputs
already contain every fact needed by the validated Phase 1A planner. Packing
those facts into a dense v2 allocation, validating that allocation, and then
copying only its exceptions into v3 is redundant.

## Implementation

The opt-in native path now returns a transient, segmented generation bundle
instead of allocating v2. The bundle carries immutable selection identity,
content fingerprints, Forward/Backward/rescore provenance, exact generation
options, and the live dense views required by the Phase 1A planner. The planner
emits an authenticated `NATIVE_DIRECT` v3 packet, validates it against those
facts, and immediately drops every dense planning view. Production continuation
and Phase 0 route statistics then read the v3 certificates and exceptions.

The default path and explicit dense dual-audit path still use journal v2.
Calling a dense replay/audit consumer on a direct-v3 seal fails closed.

Counters report dense-v2 validation/emit time and bytes, direct source
validation/staging time and bytes, eliminated v2 bytes, retained dense bytes,
v3 certificate/exception counts, and sparse-consumer preflight/core/statistics
time plus certificate/exception visits.

## Correctness evidence

- Both Cython extensions compile against the patched PyHMMER private ABI.
- Focused host tests preserve the v2 default, sparse-v3 dispatch, reusable
  sparse continuation, dense-vs-sparse state/output oracle, and resident-memory
  accounting.
- A visible-GPU first-1000 oracle and full H200 output comparison remain the
  required acceptance gates for the new native-direct source.

## Benchmark result

The visible-GPU gate passed on an attested H200, including real Forward,
simple-region, and compact-domain routes.  Full job `1182378` then reproduced
the CPU64 output exactly (SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, 383,235 lines).

Direct generation allocated and retained zero dense-v2 bytes and dropped all
transient staging ownership.  It eliminated a counterfactual 9,890,721,120
dense-v2 bytes while retaining the 5,801,342,068-byte sparse packet.  Request
wall was 545.983 seconds versus 546.221 seconds for sealed dense v2: a
0.043% difference, effectively a tie.  Generation was 478.207 seconds, CPU
continuation/output 399.914 seconds, and overlap 339.643 seconds.

The reason is explicit in the counters: direct staging cost 3.310 seconds,
but the retained two-pass sparse planner and validator still cost 20.445 and
8.559 seconds.  Removing dense-v2 construction recovered the Phase 1A
regression, but did not produce the required material speedup.

The retained follow-up fused decision planning into the existing generation
walks.  Each chunk now performs one packet-emission source scan and zero
separate decision scans.  Focused H200 job `1182389` matched dense pipeline
state and TopHits across real Forward, simple, compact, and forced-fallback
routes.  Full job `1182391` again produced the exact CPU64 output and measured:

- request wall: 536.168 seconds;
- generation: 466.502 seconds;
- CPU continuation/output: 399.591 seconds;
- overlap: 337.400 seconds;
- sparse planning: 13.384 seconds, down 34.5% from the two-pass direct form;
- 83 planner source scans and zero separate decision scans;
- zero dense-v2 allocation/retention and zero retained staging bytes.

This is 9.815 seconds (1.80%) faster than the first direct form and 10.052
seconds (1.84%) faster than the original 546.221-second dense baseline.  It is
1.311x faster than Astra CPU64, or 23.71% less request wall, but does not meet
the internal 5% Phase-1B stretch target.  Worker and raw-validation SHA-256
values are `ebc6972e6dc6d20600906882e7cca71c2d1fb9e432265838bd85b605daad2831`
and `cad2a5107ca10f786ba7fc269315bfa30f02a7b76670e260539ba162dc103ddf`.

Raw execution and validation records are rooted at
`build/phase1b-benchmark-harness/build/h200-phase1b-direct-v3-20260823/attempt-01-full`.
The immutable worker and raw-validation JSON SHA-256 values are respectively
`b34a78e71c2be8567be4cd4bd004e1eb896ccde0d7497866886d7da7e08ea338`
and `2483c29ec65928088279168c704ed146591847017280c5543a7bd5aee3ac0be0`.

## Decision

Retain the one-pass form behind explicit `sparse_journal_v3=True`; do not
switch the default from dense v2 yet.  Phase 1B is complete: exact sparse
generation is faster and materially smaller, but its gain is modest.  Proceed
to device-side stable F1 compaction rather than further tuning the host packet.
