# Phase 1A semantic fingerprint foundation

Status: retained as a private audit-only foundation; journal v3 is not part of
this change.

## Hypothesis

Dense and sparse continuation can be compared exactly only if the oracle
captures semantic HMMER state rather than raw C storage or PyHMMER pickle
state. Pointers, padding, allocation capacities, and stale workspace cells are
not semantic. Some fields serialized by upstream PyHMMER are undefined in the
short protein path.

## Implementation

`plan7_gpu._pipeline` now has private opt-in functions that construct a
versioned, typed, little-endian encoding and a SHA-256 digest for:

- initialized `P7_PIPELINE` options, thresholds, ownership modes, accounting,
  promotion counters, and short-search controls;
- exact domain-definition heuristics and successful reusable-state
  invariants;
- the active Easel RNG state, selecting only the state initialized by its RNG
  type;
- final background/null-filter state and optimized-profile length/mode
  transitions, using raw IEEE-754 bits;
- canonical `P7_TOPHITS` ordering and every initialized hit, domain,
  alignment-display, flag, threshold, and output-affecting field; and
- a non-mutating dual-state comparison result with digests, sizes, and the
  first differing byte offset.

The implementation asserts amino-acid, sequence-search, non-long-target mode,
clean error state, and successful reusable DP/domain state. It never hashes a
pointer or raw structure.

Source inspection identified additional undefined or conditionally undefined
storage that must not be read:

- `P7_PIPELINE.n_output`, `pos_output`, `strands`, and `block_length` in the
  short path;
- `P7_PIPELINE.W` before the first `NewModel` call;
- `P7_HIT.window_length`, `seqidx`, and `subseq_start` in the short path;
- `P7_DOMAIN.iorf` and `jorf` in ordinary protein domain construction;
- `ESL_RANDOMNESS.mt[]` for the fast LCG; and
- the filter HMM's numeric buffers before `p7_bg_SetFilter`, plus its unused
  `pi[M]` empty-sequence slot afterward.

These fields are excluded only under the asserted short protein contract.
`W` and the background filter state enter the encoding after a model makes
them defined and semantically observable.

## Correctness evidence

`tests/test_semantic_fingerprint.py` covers:

- byte stability across independently allocated equivalent states;
- sensitivity to raw signed-zero threshold bits and optimized-profile target
  configuration;
- empty databases and uninitialized empty `TopHits`;
- dynamic and fixed `Z`/`domZ` ownership;
- repeated pipeline reuse and `Pipeline.clear()`;
- TopHits content, order, copy, domains, and alignment displays;
- absence of known dormant/capacity fields and independence from a mutated
  `__getstate__()` projection; and
- exact-type and paired-input validation for future dual-consumer audit use.

The focused host-only fingerprint plus masked-continuation suites pass against
private ABI d486 (54 tests, 3 expected skips). The complete repository suite
also passes with CUDA hidden (271 tests, 116 expected skips). Independent
empty-result target/domain tables are byte-identical.

## Performance result

There is no production-path cost: all entry points are private and opt-in.
This change is an oracle prerequisite, not a continuation optimization.
