# plan7-gpu

`plan7-gpu` is an experimental hybrid GPU/CPU accelerator for Astra's
HMMER 3.4 protein-search pipeline.

The first implementation target is deliberately narrow:

1. consume Astra's pressed HMM database and digital protein sequences;
2. evaluate HMMER-equivalent SSV/MSV stage-1 scores on a GPU;
3. conservatively select candidate model/sequence pairs;
4. run Astra's existing CPU HMMER pipeline on candidates; and
5. require exact agreement with ordinary Astra output.

The native backend must plug into Astra without replacing its input handling,
threshold logic, downstream HMMER pipeline, or result formatting.

## Status

The backend now implements exact GPU SSV/MSV, bias correction, Viterbi,
Forward, Backward/domain decoding, isolated-domain rescoring, stable device
compaction, sparse ordered continuation, and resident interstage handoffs. The
original query-major/audit paths remain available.

The retained full H200 run searched 27,481 Pfam profiles against 300,186
proteins (8.249 billion logical pairs) in **448.140781 seconds**, versus
702.79 seconds for Astra CPU64. Its 39,010,327-byte output was byte-identical
to the CPU oracle (SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`).

Full exact experiments subsequently attacked every dominant measured CPU-tail
reason. Multidomain rerouting, bounded Backward/rescore waves, and a zero-cost
threshold upper bound all reduced their target fallback counts but regressed
request wall by 7.005%--8.494%. They remain isolated experiments; the
448.140781-second path is the retained production winner.

The durable engineering record is in:

- `PLAN7_ENGINE_ROADMAP.md` -- original plan and phase decisions;
- `IMPLEMENTATION_HISTORY.md` -- retained and rejected source sequence;
- `PERFORMANCE.md` -- exact full-workload measurements and evidence hashes;
- `CPU_CONTINUATION_TAIL_RESULTS.md` -- final bottleneck experiments and stop
  decision;
- `ROADMAP_COMPLETION_AUDIT.md` -- requirement-by-requirement disposition.

## Layout

- `baseline/`: CPU and end-to-end benchmark drivers
- `oracle/`: reference instrumentation and differential tests
- `cuda/`: CUDA implementation (created only after the oracle gate)
- `scripts/`: reproducible setup, discovery, and benchmark scripts
- `results/`: machine-readable measurements
- `refs/`: local upstream source/build/install trees (ignored by Git)

## Scientific policy

- HMMER 3.4 defines correctness.
- Performance benchmarks use real biological data.
- Generated/adversarial inputs may be used only for arithmetic correctness
  testing and are never reported as biological benchmarks.
- GPL implementations may be studied as prior art, but no GPL source is
  copied into this repository.
