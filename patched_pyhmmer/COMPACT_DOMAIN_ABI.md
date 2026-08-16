# Compact-domain continuation ABI

This is the CPU/HMMER side of the simple-envelope CUDA rescore handoff. The
outer adapter owns the immutable generation seal and content hashes. After it
authenticates those, it calls
`p7_PipelineFromFilterForwardAndCompactDomainsV2()` in original target order.
The V2 suffix is part of the ABI contract: this wheel intentionally exports
neither the unversioned compact entry point nor its unversioned fingerprint.
Consequently, an old adapter/new wheel or new adapter/old wheel sees the
compact seam as unavailable and uses the ordinary CPU path instead of calling
a shifted C signature.

## Record layout

`P7_PIPELINE_COMPACT_DOMAIN` is native-endian, pointer-free, and exactly 64
bytes. All coordinates are one-based, inclusive, and target-global.

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `uint32_t` | row index |
| 4 | `uint32_t` | profile index |
| 8 | `uint32_t` | sequence index |
| 12 | `uint32_t` | envelope begin |
| 16 | `uint32_t` | envelope end |
| 20 | `uint32_t` | alignment begin |
| 24 | `uint32_t` | alignment end |
| 28 | `uint32_t` | model begin |
| 32 | `uint32_t` | model end |
| 36 | `float` | envelope Forward score, nats |
| 40 | `float` | envelope Backward score, nats |
| 44 | `float` | optimal-accuracy score |
| 48 | `float` | summed null2 domain correction, nats |
| 52 | `float` | absolute Forward/Backward difference |
| 56 | `uint8_t` | status (`eslOK`) |
| 57 | `uint8_t` | action (`p7_COMPACT_DOMAIN_DEVICE_RESULT`) |
| 58 | `uint8_t` | has-own-scales (`FALSE`) |
| 59 | `uint8_t` | zero |
| 60 | `uint32_t` | zero |

Each domain has 29 positive finite binary32 null2 odds in HMMER digital-code
order. A `domain_count + 1` array of `uint64_t` offsets indexes a flat trace
array. When passing one row sliced from the global CUDA journal, the adapter
must rebase its offsets to that row's trace slice. The first offset is zero
and the last equals the supplied trace-step count.

`P7_PIPELINE_COMPACT_TRACE_STEP` is exactly 16 bytes:

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | `uint32_t` | target-global sequence position |
| 4 | `uint32_t` | model position |
| 8 | `float` | posterior probability |
| 12 | `uint8_t` | stock `p7T_*` state |
| 13 | 3 bytes | zero padding |

The trace is a complete stock unihit `S N* B (M/D/I)* E C* T` trace over the
envelope. Nonemitting positions and posteriors are zero. Emitting positions
cover every envelope residue exactly once. First/last match positions must
agree with the alignment and model bounds in the result record.

## Boundary behavior

The entry point also receives the sealed generation's four filter scores,
posterior expected-domain count, row/profile/sequence identity, final target
count, and the value of `p7_pipeline_CompactTailFingerprintV2()` captured for
that generation. `final_target_count` must be in `[1, 2^53]`; it supplies the
exact final dynamic target search space and bounds every possible integer
dynamic `domZ`. The live dynamic `Z` must be an integer in
`[1, final_target_count]`.

Before live state changes, HMMER validates every record, count, offset, trace
transition, coordinate, posterior, null2 degeneracy, and option/floating-point
precondition. Forward/Backward consistency, recomputed null2 correction, and
the OA score reconstructed from trace posteriors each use a `2e-3` absolute
tolerance. Accepted input is staged into stock traces, domains, and alignment
displays, then passed to the unchanged HMMER final tail.

Guard version 1 also prevents a compact/GPU rounding difference from changing
any final report or inclusion call. It encloses each accepted external score
by `0.004` nats plus a `1e-5`-bit arithmetic margin, reconstructs HMMER's exact
full-target, reconstruction, and per-domain score expressions (including the
ordered Kahan null2 sum), and checks target/domain report and inclusion
predicates before allocation or live-state mutation. Score thresholds include
model-specific GA/TC/NC values already loaded by `p7_pli_NewModel()`. E-value
thresholds use the current and final target `Z`; fixed `domZ` is checked
directly, while dynamic `domZ` binary-searches the exact rounded-double
predicate over every possible integer in `[1, final_target_count]`. The guard
version and both error constants are included in the V2 tail fingerprint.

The caller must already have performed `p7_pli_NewModel()` and exactly one
target's `p7_pli_NewSeq()` accounting (the sparse adapter performs that same
accounting directly), but must not pre-increment survivor counters. This seam
replays the score gates and owns the four `n_past_*` increments.
`sequence_index` is an adapter-authenticated resident-batch identity; it is
deliberately not compared with the pipeline target ordinal.

`eslEINVAL` means the adapter must discard the entire row payload and use its
authenticated ordinary Forward/CPU continuation. `eslEINACCURATE` is distinct:
the payload is valid, but a final call is threshold-adjacent, so the adapter
must rerun the whole ordinary `p7_Pipeline()` for the row and must not reuse the
GPU Forward score or any postfilter state. Both statuses are returned before
survivor counters, matrices, domain-definition state, or TopHits change.
Allocation errors retain normal HMMER semantics and are not part of that
transactional promise.

The adapter must authenticate its profile, target, background, result, null2,
and trace hashes before this call. The C seam validates live HMMER
configuration and record identities, but deliberately does not duplicate the
adapter's hashing implementation.
