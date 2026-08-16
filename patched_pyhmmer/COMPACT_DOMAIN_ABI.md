# Compact-domain continuation ABI

This is the CPU/HMMER side of the simple-envelope CUDA rescore handoff. The
outer adapter owns the immutable generation seal and content hashes. After it
authenticates those, it calls
`p7_PipelineFromFilterForwardAndCompactDomains()` in original target order.

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
posterior expected-domain count, row/profile/sequence identity, and the value
of `p7_pipeline_CompactTailFingerprint()` captured for that generation.

Before live state changes, HMMER validates every record, count, offset, trace
transition, coordinate, posterior, null2 degeneracy, and option/floating-point
precondition. Forward/Backward consistency, recomputed null2 correction, and
the OA score reconstructed from trace posteriors each use a `2e-3` absolute
tolerance. Accepted input is staged into stock traces, domains, and alignment
displays, then passed to the unchanged HMMER final tail.

The caller must already have performed `p7_pli_NewModel()` and exactly one
`p7_pli_NewSeq()` for the live target, but must not pre-increment survivor
counters. This seam replays the score gates and owns the four `n_past_*`
increments. `sequence_index` is an adapter-authenticated resident-batch
identity; it is deliberately not compared with the `p7_pli_NewSeq()` ordinal.

`eslEINVAL` means the adapter must discard the entire row payload and use the
ordinary CPU continuation. Validation failures do not change pipeline
counters, matrices, domain-definition state, or TopHits. Allocation errors
retain normal HMMER semantics and are not part of that transactional promise.

The adapter must authenticate its profile, target, background, result, null2,
and trace hashes before this call. The C seam validates live HMMER
configuration and record identities, but deliberately does not duplicate the
adapter's hashing implementation.
