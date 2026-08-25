# Phase 3: external multidomain continuation

## Hypothesis

The accepted full workload still spends about 400 seconds in CPU continuation.
Of 826,453 post-Forward rows, 552,390 use the general CPU route. Phase-0
telemetry attributes 150,965 of those rows to HMMER's multidomain posterior
classification. Rows whose posterior decisions are numerically certain should
not need to repeat full-target Backward and domain decoding on the CPU merely
because more than one domain may be present.

## Implementation

Backward/domain adds an `EXTERNAL_REGIONS` route for rows with finite exact
posterior state, at least one region, a multidomain classification, and no
threshold uncertainty. Region order and original profile/sequence ordinals are
unchanged. The high bit of each region start records whether HMMER's expected
domain count crossed `rt3`; the remaining 31 bits retain the exact one-based
coordinate. Capped, nonfinite, own-scale, and uncertain rows keep their prior
CPU route.

The ordered sparse-v3 packet carries these regions through a new authenticated
source-stage value while reusing the established simple-regions exception ABI.
The patched private HMMER seam validates the complete row before mutation.
Unflagged regions use the unchanged isolated-envelope path. Flagged regions use
HMMER's unchanged multihit Forward, stochastic trace ensemble, clustering,
overlap accounting, and isolated-envelope rescore. Final hit construction,
thresholding, ordering, and Astra output remain on the established exact CPU
tail.

The default simple/no-region/compact routes and the audit/reference path remain
available. The private PyHMMER ABI changes, so both native extensions and the
pressed-database manifest are rebuilt against the new wheel.

## Correctness evidence

The patched PyHMMER wheel and sm75/sm90 native extensions build successfully.
All three runtime private-ABI fingerprints equal
`08ef627fb0901946ed145dd65e4b3afe34e9740869fcb2e2807896e9bb1e17e7`.

Representative H200 job `1183515` searched PFAM profile ordinals 27,306–27,480
against all 300,186 targets. It exercised 5,012 external multidomain rows and
reduced the old Backward CPU route from 5,891 rows to 879. Every one of the 175
semantic TopHits fingerprints matched the accepted implementation, and the
framed target and domain table streams were byte-identical:

- targets SHA-256: `11cc3b5e9195c5f6a6d3d57d4d9f4e92fc23f52f33f8eff4a4e68414deef3d1c`;
- domains SHA-256: `9013228575d4e2afb3c16b4ca13e8b5b29cdae4d18f1942d98bcfbec3c7853dd`;
- semantic stream SHA-256: `1a4fc3d1f39a75c834545beb1b185a81feda928f8c7387bcf51f1854246d9618`.

The representative timing was not treated as a performance decision:
continuation measured 1.066 seconds in the accepted arm and 1.113 seconds in
the experimental arm inside one short allocation. Full-workload evidence is
required because the production critical path is chunked and overlapped.

## Full benchmark decision

Full H200 job `1183518` completed successfully and reproduced the established
CPU64/GPU output byte-for-byte: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.

The route behaved as designed but did not improve the critical path. It moved
132,654 rows to exact external-region continuation, reducing the final
`CPU_REQUIRED` census from 552,390 to 419,736 rows (24.015%). The remaining
18,311 multidomain rows were also threshold-uncertain and correctly stayed on
the conservative CPU route. Despite that route-count reduction, request wall
rose from the accepted 448.140781 seconds to **480.258523 seconds**: a
32.117742-second or **7.167% regression**.

| Measurement | Accepted path | External multidomain | Change |
|---|---:|---:|---:|
| Request wall | 448.140781 s | 480.258523 s | +32.117742 s (+7.167%) |
| Generation | 333.993605 s | 344.576864 s | +10.583259 s (+3.169%) |
| CPU continuation/output | 400.083104 s | 431.867034 s | +31.783930 s (+7.944%) |
| Overlap | 293.693784 s | 305.684247 s | +11.990463 s |
| Pipeline wall | 440.538394 s | 471.332009 s | +30.793615 s |
| Maximum RSS | 13,195,416 KiB | 13,080,804 KiB | -114,612 KiB (-0.869%) |
| Sampled H200 peak | 3,388 MiB | 3,380 MiB | -8 MiB (-0.236%) |

The exact multidomain work was shifted into the external HMMER seam rather than
removed: multihit Forward, stochastic traces, clustering, overlap accounting,
and isolated rescoring still execute on the CPU, while generation additionally
materializes and transports the external regions. On this workload that path
was more expensive than the old fallback. The small memory reduction is not
material enough to justify a 7.167% wall regression.

**Decision: rejected.** Commit `d5731718d1460e7d2b2c0cff9e5204b1fa2d8e1a`
preserves the prototype on its experiment branch, but the implementation is
not promoted to `main`. Evidence is under
`build/h200-multidomain-full-20260825/attempt-01`; `result.json` SHA-256 is
`431e8fadf987a18c89f55be11cad4ca02b8963dc9a3846fcfd3f60d45ec75c8d`.
