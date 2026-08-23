# Phase 1A design: host-side sparse accounting journal v3

Status: source-audited design; implementation has not started.

This note fixes the semantic contract for the first sparse continuation
experiment.  Phase 1A changes only the host representation and consumer.  It
still starts from the complete version-2 host journal, and it keeps the dense
version-2 consumer as the reference implementation.

## Why a certificate is required

The current dense consumer walks every retained postfilter record.  For gaps
between retained records it accounts target ordinals and residue-prefix
deltas in constant time.  It nevertheless visits roughly 203.7 million records
in the sealed full workload although only roughly 826 thousand rows reach the
Forward/domain continuation layer.

Omitted rows are not semantically inert.  Depending on their last resolved
filter, they contribute exact deltas to `n_past_msv`, `n_past_bias`,
`n_past_vit`, and `n_past_fwd`.  Every target, retained or not, contributes to
the search's `nseqs`, `nres`, and dynamic `Z` accounting.  The sparse packet
therefore needs an accounting certificate in addition to exceptional rows.

## Dense consumer ownership, verified from source

At validated search entry the wrapper:

1. forces protein sequence-search mode and resets the call-local `nseqs`;
2. invokes `p7_pli_NewModel()` once, which increments `nmodels`, adds the
   model length to `nnodes`, configures the background filter model, loads
   model cutoffs where requested, and sets `W`;
3. accounts each target exactly once through checked ordinal and residue-prefix
   deltas;
4. configures background and optimized-profile target length before an
   exceptional row;
5. invokes exactly one continuation route for that row;
6. calls `p7_pipeline_Reuse()` after the row;
7. applies tail accounting and configures the background/profile to the last
   database target even when that target was rejected; and
8. sorts and thresholds `TopHits`, at which point dynamic `domZ` becomes the
   number of reported targets.

For a nonempty search, dynamic `Z` ends as the target count.  A zero-target
search is a special case: the current wrapper does not assign a new dynamic
`Z` and does not reconfigure target length.

`p7_pipeline_Reuse()` resets reusable matrix/domain work state only.  It does
not reset accounting, options, thresholds, `Z`, `domZ`, `W`, RNG state, or the
error buffer.  Most counters are cumulative across searches unless the caller
uses `Pipeline.clear()`; v3 must therefore apply checked deltas to a validated
prestate rather than overwrite absolute totals.

## Seam counter ownership

An exceptional target is accounted exactly once for `nseqs`/`nres` before its
seam.  Its promotion counters are owned by the seam and must not also appear in
the preceding certificate.

| Route outcome | Promotion deltas owned by the invoked HMMER path |
|---|---|
| raw F1 reject omitted by certificate | none |
| bias reject omitted by certificate | MSV |
| F2 reject omitted by certificate | MSV, bias |
| F3 reject omitted by certificate | MSV, bias, Viterbi |
| full `p7_Pipeline` exception | whatever stages the stock path reaches |
| filter-score seam exception | whatever stages that seam reaches |
| Forward-score F3 survivor | MSV, bias, Viterbi, Forward |
| simple-region success | all four |
| compact-domain success | all four |

The exported Forward-score seam has two early replay returns that do not own
the expected counters for arbitrary bias- or F2-reject input.  The current
sealed producer never supplies those rows to the seam: its Forward rows are
already F2 survivors.  Journal v3 must encode and validate this precondition;
it must not generalize the seam into an accounting-equivalent entry point.

Compact `eslEINACCURATE` and `eslEINVAL` are safe to retry because the patched
HMMER worker returns before semantic mutation.  A compact threshold retry uses
the whole stock pipeline and owns all of that row's promotions.  An invalid
compact payload retries the authenticated Forward route.  Neither retry row
may have its promotions pre-accounted.

## Proposed logical v3 representation

Version 3 receives new capsule names and structures.  Version 2 remains
unchanged and independently consumable.

Each profile owns:

- immutable profile/session/selection/target identities and generation
  options already authenticated by v2;
- an ordered exception interval in the global exception array;
- one certificate immediately before each exception; and
- one terminal certificate after its last exception.

A certificate covers a half-open target-ordinal span `[begin, end)` and stores:

- `begin` and `end` target ordinals;
- the corresponding residue-prefix values at both endpoints;
- the exact target-count and residue-count deltas (redundant by design and
  cross-checked against the endpoints);
- exact omitted-row deltas for MSV, bias, Viterbi, and Forward promotions; and
- a typed integrity contribution that binds the profile and segment order.

An exception stores its original target ordinal, its route, and only the
authenticated payload required by that route.  The exception target is outside
the preceding certificate span.  The consumer performs the equivalent of one
`NewSeq` accounting step for that target, configures its length, invokes the
route, and reuses the pipeline.  The tail certificate then covers the remaining
targets.  Empty spans are allowed where exceptions are consecutive.

Required exceptions are derived from semantics, not the observed full-run
count.  They include any row that must invoke a CPU/continuation seam, any
Forward survivor requiring domain work, every device-resolved result that must
enter the reporting tail, and any other row whose state cannot be represented
solely by certified accounting deltas.

Before mutation, v3 validation rejects:

- gaps, overlaps, duplicate application, decreasing ordinals, or a segment
  outside the target block;
- residue-prefix endpoints that differ from the sealed target prefix;
- counter overflow or impossible promotion nesting;
- an exception route whose source-stage preconditions are not met;
- profile, target, background, option, or generation identity drift; and
- a final certificate that does not end exactly at the target count.

## Semantic pre/post fingerprint

The fingerprint is a canonical typed encoding.  It is not a byte hash of C
structs, pointers, padding, Python pickle state, or `__getstate__()` output.

It includes every initialized and meaningful scalar configuration/state field:

- report/include modes and exact IEEE bits for target/domain score and E-value
  thresholds;
- model-cutoff mode and loaded target/domain report/include cutoffs;
- `Z`, `domZ`, and their ownership modes;
- F1/F2/F3, bias/null2/max controls, B1/B2/B3, mode, long-target flag, `W`,
  display/reseeding/alignment controls;
- all meaningful accounting and promotion counters;
- domain-definition heuristic configuration and successful reusable-state
  invariants;
- final background filter composition/length transition state;
- final optimized-profile target-length configuration; and
- RNG state whenever a supported path can consume randomness.

Successful dual execution additionally compares a canonical semantic TopHits
encoding: all output-affecting initialized hit, domain, alignment-display,
flag, ordering, threshold, and pipeline fields, using raw float bits where
appropriate.  It also compares the exact ordinary target/domain table bytes.

Several dormant short-mode fields are currently uninitialized by upstream
allocation (`n_output`, `pos_output`, `strands`, and `block_length`).  Protein
hits likewise leave `window_length`, `seqidx`, and `subseq_start` dormant while
upstream serialization includes them.  Raw struct hashing, generic TopHits
serialization, and PyHMMER state serialization are therefore invalid oracles.
The canonical encoding excludes such dormant fields only under an asserted
short protein-search contract, or a separate preparatory change must
initialize them before they can enter the fingerprint.

Pointers are never hashed.  The audit asserts the appropriate pointer/null and
reusable-state conditions instead.  The error buffer is checked for no new
error on success rather than treated as an opaque byte array.

## Dual-consumer audit mode

The audit entry point accepts two independently owned pipelines whose canonical
prestate fingerprints match.  It uses independent optimized-profile copies and
backgrounds, runs dense v2 on one and sparse v3 on the other, and compares:

- exact target/domain table bytes;
- canonical semantic TopHits encodings;
- canonical poststate fingerprints and checked counter deltas;
- final background/profile length configuration;
- compact attempt/accept/invalid/threshold-retry counts; and
- route/certificate reconciliation.

It must not attempt `copy.copy`, `deepcopy`, pickle, or raw state restoration on
`pyhmmer.plan7.Pipeline`; those operations are unsupported.  A convenience
test helper may construct two fresh equivalent pipelines from explicit options,
but the low-level dual validator treats equivalent caller-provided pipelines as
the trust boundary.

## Initial cert-only predicate

The first host compactor is deliberately narrower than every row that might
eventually be compressible.  It may omit a row only when the already validated
version-2 inputs prove one of these exact terminal states:

- a target absent from the postfilter stream, or an authenticated raw-F1 reject:
  no promotion counters;
- a finite postfilter row with `BIAS_DEFINITE_REJECT`: MSV only;
- a finite `BIAS_DEFINITE_PASS` row with no Forward record, after replaying the
  exact `esl_gumbel_surv` F2 predicates and proving a Viterbi/F2 reject: MSV and
  bias;
- an authenticated `FORWARD_DEFINITE_REJECT`: MSV, bias, and Viterbi; or
- an authenticated Forward pass whose domain journal is exactly
  `DOMAIN_NO_REGIONS`, with no own scales, uncertainty, multidomain state,
  regions, or compact payload: all four promotions.

The no-region seam temporarily assigns `ddef->L` and `ddef->nexpected`, but the
existing unconditional `p7_pipeline_Reuse()` restores all meaningful reusable
domain state to zero.  The certificate therefore reproduces its persistent
semantic effect by adding the four promotion deltas only.

Any malformed or unprovable F2 row remains a filter-score exception.  Every
row that can construct a hit, requires CPU work, carries a simple/compact
domain payload, or has an uncertain source-stage precondition remains an
exception.  Widening this set requires a separate source proof and oracle.

## Host-only testability contract

Journal-v2 minting is currently fused to GPU generation, so Phase 1A needs a
narrow private host hook.  The hook must exercise the production v3 planner,
ABI validator, and one-shot owner; it must not duplicate the stage classifier
in Python or substitute a dictionary for the binary journal.  It accepts an
already validated sealed dense batch and exposes only a defensive debug summary
of certificate spans/deltas and exception routes.  Production APIs and default
search selection remain unchanged.

The deterministic host matrix uses the existing masked-pipeline record helpers:

| Terminal state | Dense fixture | Expected promotion delta |
|---|---|---|
| before F1 | no postfilter row | `(0,0,0,0)` |
| raw F1 reject | `filtersc`/`vfsc` NaN, direct reject | `(0,0,0,0)` |
| bias reject | `F1=.02`, numerator 0, `filtersc=10`, direct reject | `(1,0,0,0)` |
| F2 reject | `F1=1`, `F2=0`, numerator 0, `filtersc=0`, `vfsc=-1e30` | `(1,1,0,0)` |
| F3 reject | `F1=F2=1`, `F3=0`, `vfsc=+inf`, `fwdsc=-1e30` | `(1,1,1,0)` |

Exception-shape tests use ordinary `BIAS_CPU_REQUIRED` records at the first and
last target, consecutive exceptions, a 2,048-target prefix, a large tail, and
no exceptions.  They also cover a final omitted target whose length differs
from the preceding exception and an empty final target.  Real CUDA validation
later uses the stable Thioesterase domain quartet: target 0 `NO_REGIONS`, 13
`SIMPLE`, 31 clustered/CPU-required, and 67 `SIMPLE`.

Private test seams are also required for a mutation-free compact `eslEINVAL`
retry and checked counter prestates near `UINT64_MAX`.  These exist only to
cover otherwise unreachable validator branches; they cannot be selected by
the public search path.

## Required tests before production selection

Fixtures cover reject-before-F1, F1, bias, F2, and F3 rejects; simple domain;
multidomain; CPU-required; compact accept and both retry classes; first, last,
and consecutive exceptions; no exceptions; large omitted prefix and tail;
empty target members; empty target database; fixed/dynamic `Z` and `domZ`;
pipeline reuse with and without `clear()`; threshold/cutoff modes; counter and
residue overflow; malformed spans; duplicate application; and final target
length/configuration.

Phase 1A is retained only if all semantic comparisons are exact and the sparse
consumer materially reduces CPU continuation time on representative workloads.
Otherwise the negative result and its measured cause are recorded here before
any Phase 1B transport change.
