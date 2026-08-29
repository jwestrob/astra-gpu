# Post-258 P7: sealed bias-reject Viterbi skip

## Hypothesis

The reusable candidate-batch API must preserve Viterbi results for a later
consumer that changes bias-filter options.  A direct sparse-v3 Astra seal has
already authenticated fixed bias-filter options, so a finite exact bias reject
can skip Viterbi without changing any legal continuation.

The full Phase-0 census counted 56,021,749 finite bias rejects.  Their
2,694,191,895,808 Viterbi cells are 33.0% of the retained full-workload
Viterbi cells.

## Implementation

The fixed-options sealed path passes an internal flag through the resident
postfilter API.  The Viterbi kernel returns before DP for a finite
`DEFINITE_REJECT`; merge preserves the exact postfilter identity and terminal
bias-reject decision.  General/reusable postfilter APIs are unchanged.

The path is automatic only for direct sparse-v3 searches with more than 65,536
targets.  `PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP=0` disables it and `=1` forces it
for focused audit.  Small and nonsealed searches retain the general path.

## Correctness evidence

Focused H200 job 1186216 compared fixed and general paths on identical rows.
It reconciled 1,108 exact bias rejects, removed 87,246,368 Viterbi cells, and
produced identical postfilter records and final HMMER output.

Full H200 job 1186217 reproduced the established oracle exactly:

```text
SHA-256  3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6
bytes    39,010,327
lines    383,235
```

## Full benchmark result

Against retained job 1183504 / the 258.817809-second baseline:

```text
                         retained       P7          delta
request wall             258.817809 s   250.533515 s  -8.284294 s (-3.20%)
native generation        249.859082 s   241.456182 s  -8.402901 s (-3.36%)
continuation/output       81.052199 s     81.801867 s  +0.749668 s
measured overlap          79.923879 s     80.644104 s  +0.720226 s
pipeline wall             251.045017 s   242.671303 s  -8.373713 s (-3.34%)
peak RSS                7,420,788 KiB  7,395,568 KiB  -25,220 KiB
postfilter native          47.689324 s    36.932742 s -10.756582 s (-22.56%)
```

The full harness did not sample peak HBM.  Workspace capacity is unchanged by
this first implementation because it preserves the general offset layout; the
optimization removes arithmetic rather than retained allocation.

Evidence:

```text
build/post258-p7-sealed-bias-viterbi-h200/attempt-02/result.json
build/post258-p7-sealed-bias-viterbi-full-h200/attempt-01/summary.json
Slurm jobs 1186216 and 1186217
```

## Decision

Retain.  This is an exact fixed-options specialization with a material stage
and end-to-end gain.  The reusable API and small-workload path remain intact.
