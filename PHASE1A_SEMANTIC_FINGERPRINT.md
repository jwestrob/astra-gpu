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
- final independently initialized background state and optimized-profile
  length/mode transitions, using raw IEEE-754 bits;
- canonical `P7_TOPHITS` ordering and every initialized hit, domain,
  alignment-display, flag, threshold, and output-affecting field; and
- a non-mutating dual-state comparison result with digests, sizes, and the
  first differing byte offset.

The implementation asserts amino-acid, sequence-search, non-long-target mode,
clean error state, and successful reusable DP/domain state. It never hashes a
pointer or raw structure.

The standalone pipeline-state encoding covers mutable execution state, not the
immutable scoring model. The dual-state oracle requires both optimized
profiles, computes the existing canonical pressed-profile fingerprint for each,
and rejects different model identities before comparing mutable state. When
`TopHits` operands retain optimized-profile queries, each query is bound to its
corresponding supplied profile by the same digest. HMM/profile queries receive
conservative alphabet and metadata consistency checks. This is the same
identity contract used by the continuation journal; matching
name/accession/model length alone is deliberately insufficient.

Source inspection identified additional undefined or conditionally undefined
storage that must not be read:

- `P7_PIPELINE.n_output`, `pos_output`, `strands`, and `block_length` in the
  short path;
- `P7_PIPELINE.W` before the first `NewModel` call;
- `P7_HIT.window_length`, `seqidx`, and `subseq_start` in the short path;
- `P7_DOMAIN.iorf` and `jorf` in ordinary protein domain construction;
- `ESL_RANDOMNESS.mt[]` for the fast LCG; and
- every filter-HMM numeric buffer until the audit executor records explicit
  provenance for the most recent `p7_pli_NewModel()` call.

`p7_bg_SetFilter()` is conditional on the bias filter, while PyHMMER permits
the live bias-filter option to change independently afterward. Neither
`nmodels` nor the current option proves that the filter HMM was initialized for
the most recent model. The encoding therefore retains only independently
initialized background alphabet/frequency/length/null2 state and emits a fixed
`excluded-unproven` marker without reading `fhmm`. Exact filter-HMM comparison
will require explicit `NewModel` provenance in the future. The reusable-state
assertion also covers the logical cursors reset inside the domain-definition
child ensemble and its two traces.

The short-path fields above are excluded only under the asserted short protein
contract. Filter-HMM storage is excluded independently until explicit
initialization provenance exists. `W` enters the encoding after a model makes
it defined and semantically observable.

## Correctness evidence

`tests/test_semantic_fingerprint.py` covers:

- byte stability across independently allocated equivalent states;
- sensitivity to raw signed-zero threshold bits and optimized-profile target
  configuration;
- empty databases and uninitialized empty `TopHits`;
- dynamic and fixed `Z`/`domZ` ownership;
- repeated bias-disabled searches with independently allocated, uninitialized
  filter-HMM storage;
- a bias-disabled search followed by enabling the live bias-filter option,
  proving that unproven filter-HMM storage is still never inspected;
- repeated pipeline reuse and `Pipeline.clear()`;
- every reusable segment-ensemble and trace cursor reset by HMMER;
- TopHits content, order, copy, domains, and alignment displays;
- rejection of different immutable optimized-profile identities even when
  their name, accession, model length, and mutable target configuration match;
- mandatory paired profiles and binding of optimized-profile `TopHits` queries
  to their corresponding supplied profile identities;
- absence of known dormant/capacity fields and independence from a mutated
  `__getstate__()` projection; and
- exact-type and paired-input validation for future dual-consumer audit use.

The focused host-only fingerprint plus masked-continuation suites pass against
private ABI d486 (60 tests, 3 expected skips). The complete repository suite
also passes with CUDA hidden (277 tests, 116 expected skips). Independent
empty-result target/domain tables are byte-identical.

## Performance result

There is no per-search or hot-path cost: all entry points are private and
opt-in. Extension initialization does load the private fingerprint helper.
This change is an oracle prerequisite, not a continuation optimization.
