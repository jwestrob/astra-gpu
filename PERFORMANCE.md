# Astra GPU performance evidence

This file is the compact, source-controlled performance record for external review. The full immutable run evidence remains in the local workspace at the paths listed below; large biological outputs, runtimes, and datasets are intentionally not committed.

## Exact implementation under review

- Native `plan7_gpu`: commit `614161a4d24c05564b863a0d1b67f0b2f26aeaf1`, tree `cf2eab3a8b50ba989132110003d482b9e6b5924f`.
- Astra persistent profile cache and weighted ready queue: commit `3996a60e765315ad9286ffab14ae67141ab19762`, tree `bba6afeb56af6c0568e07f0310508b78353770db`, included under `integrations/astra/`.
- The byte-bounded queue requires the native `CandidateBatch.resident_bytes` API introduced at native commit `614161a`.
- Private patched-PyHMMER ABI: `d4867ff865e9b8a7acdbbf9106e3d7e1223336d374cb0f46d7e352427b990689`.

## Workload

- Target: complete PLM2_5 predicted-protein catalog from the East River metagenome assembly (scaffolds >=1 kb).
- FASTA SHA-256: `eadfb887b8a8a25921cd318d0e278b43f900b05e3dbe0fae7d30233d4932fd04`.
- 300,186 protein sequences; 69,453,045 amino-acid residues; 112,069,869 input bytes.
- Database: Pfam-A, 27,481 profile HMMs.
- Pfam-A text SHA-256: `a78a42d6faf265b6bfca59e8f062d06fae6083ce2c6e335d7b381f20b82b7903`.
- Search semantics: `--cut_ga`, normal sequence-search mode, no alignments in the native HMMER reporting command.
- Logical workload: 8,249,411,466 profile-by-sequence pairs.

## Headline measurements

All figures below are measured wall time from the sealed runs, not projections.

| System | Workers | Measured wall | Max RSS | Relative result |
|---|---:|---:|---:|---:|
| HMMER 3.4, standard CPU node | 48 | 22,458.00 s (6:14:18) | 1,452,668 KB | Astra CPU was 24.597489650x faster |
| Stock Astra/PyHMMER, standard CPU node | 48 | 913.02 s (15:13.02) | 2,432,444 KB | H200 request was 1.6715x faster |
| HMMER 3.4, standard CPU node | 64 | 19,683.00 s (5:28:03) | 1,818,828 KB | H200 request was 36.0349x faster |
| Stock Astra/PyHMMER, standard CPU node | 64 | 702.79 s (11:42.79) | 2,622,124 KB | baseline for the primary GPU comparison |
| Astra GPU, H200, warm persistent request | 64 host workers | 546.220704615 s | see telemetry below | **1.286641085x faster than CPU64; 22.278% less wall time** |
| Astra GPU, H200, one-pass sparse-v3 experiment | 64 host workers | 536.168411 s | 13,143,840 KiB | **1.31076x faster than CPU64; 23.71% less wall time** |
| Astra GPU, H200, Phase 2 device compaction | 64 host workers | 538.822140 s | 13,175,768 KiB | exact; 0.495% slower than one-pass sparse v3 |
| Astra GPU, H200, resident Forward-to-Backward handoff | 64 host workers | 534.276010 s | see run record | exact; 0.844% faster than Phase 2 |

Additional GPU timing layers:

- Search call: 538.746491935 s.
- Pipeline wall: 538.4393 s.
- Whole persistent worker process: 580.17 s, or 1.21135x versus the CPU64 application time.
- Slurm allocation from start through final sealing: 655 s, or 1.07296x versus the CPU64 application time. This includes provenance and validation work that is not part of the search request.
- The measured request was a cache hit in the same persistent GPU profile session: profile load and device build were both zero during the measured request. Target parsing and all per-request search/output work remained included.

These are one-run, cross-node measurements. They establish the observed result but do not provide a sampling distribution or confidence interval.

### Current optimization-stage result

The original 546.221-second row remains the sealed architectural baseline.
Phase 1A's host compaction was exact but 3.82% slower, because it built dense
v2 and then spent 28.945 seconds planning and validating sparse v3.  Phase 1B
eliminated dense-v2 allocation and fused the decision plan into generation.
Full H200 job `1182391` was byte-identical and completed in 536.168 seconds:
466.502 seconds generation, 399.591 seconds continuation/output, and 337.400
seconds overlap.  It retained a 5,801,342,068-byte sparse packet instead of a
counterfactual 9,890,721,120-byte dense journal, performed 83 source scans and
zero separate decision scans, and improved the original request by 10.052
seconds (1.84%).  The result is retained, while the next material performance
target is device residency across Forward, Backward/domain, and rescore.

Phase 2 full H200 job `1182619` was also byte-identical. It completed in
538.822 seconds: 468.243 seconds generation, 401.387 seconds
continuation/output, and 338.475 seconds overlap. All 83 chunks used stable
device F1 compaction; host expansions and candidate-mapping uploads were both
zero, and 83 uploads were avoided. The structural change is retained, but the
0.495% regression from the 536.168-second one-pass run supports no standalone
Phase 2 speedup claim. It remained 1.354% faster than the original dense run.

Phase 3 Forward-to-Backward residency job `1182690` was byte-identical and
completed in 534.276 seconds: 464.689 seconds generation, 526.763 seconds of
pipeline wall, and 336.217 seconds overlap. All 83 Forward and 83 Backward calls
used the resident handoff, with zero fallbacks and zero legacy Forward-special
H2D bytes. It materialized 6,022,020,720 resident bytes and eliminated
3,291,870,792 bytes of redundant H2D traffic. Backward Forward-special upload
time fell from 439.13 ms to 49.16 ms and aggregate Backward wall fell from
26.880 seconds to 24.853 seconds. Request wall improved by 4.546 seconds
(0.844%) versus Phase 2, so this first residency slice is retained.

The combined Phase 3 path (resident Forward-to-Backward and
Backward-to-rescore plus authoritative compiled F3 decisions) completed in
535.213 seconds in job `1182713`. Phase 4 then tested automatic packed Forward
subwarps on the same full workload in job `1182718`. Output remained exactly
identical, but the policy selected width 2 for all 83 chunks and regressed
Forward kernel time from 32.823 to 33.892 seconds (+3.26%) and request wall
from 535.213 to 539.052 seconds (+0.72%). Automatic promotion is rejected;
production remains fixed at one four-lane candidate per warp. Wider variants
remain diagnostic-only.

## GPU request-stage ledger

The measured 546.220704615 s request decomposed as follows:

- Target parse: approximately 0.7233 s.
- Request preflight: approximately 6.1330 s.
- Search: 538.746491935 s.
- Remaining request overhead: approximately 0.6179 s.
- Native selection/generation work: 476.7994 s.
- CPU continuation and output work: 401.7809 s.
- Measured overlap between those two lanes: 340.2795 s.
- Producer idle while the depth-1 ready queue was full: 59.7871 s across 8 blocking episodes.
- Consumer generation wait: 135.9900 s.
- 83 profile chunks; producer lookahead high-water mark 2; ready-queue high-water mark 1.

Because generation and continuation overlap, the stage values are not additive. With identical work and perfect scheduling, the measured-work lower bound is approximately 484.6 s. Queueing alone therefore cannot account for more than about 61.6 s of improvement; larger gains require reducing both native generation and CPU continuation work.

## Work and transport-relevant counters

- Postfilter-retained candidates: 203,671,109.
- Forward/domain continuation rows: 826,453 (0.4058% of retained candidates).
- Route partition: 552,390 `CPU_REQUIRED`, 274,063 `SIMPLE`, 0 `NO_REGION`; exact sum 826,453.
- Compact-device result regions: 223,187.
- Compact rescore attempts: 128,314 = 127,213 accepted + 1,101 exact-threshold/inaccurate retries.
- Host continuation-journal payload: 9,890,721,120 bytes total; largest chunk 469,665,656 bytes.
- Full-request cache hit reused profile session ID 1; profile load/build were 0; selection count and workspace lifecycle checks passed.
- All 83 native Forward workspaces ran and all queue/cache/final-zero invariants passed.

The current architecture repeatedly materializes and scans large intermediate candidate sets on the host. These counters motivate the next optimization target: keep the F1/postfilter/Forward/Backward/rescore chain device-resident and copy back one compact ordered egress packet for the exact CPU HMMER tail.

## H200 and telemetry

- Node: `node-224-2t-8gpu-1`, Slurm partition `gpu_h200`.
- Selected GPU UUID: `GPU-5e0f95a2-f8a3-9429-b566-672ebd017a32`.
- Compute capability 9.0; 132 SMs; 150,121,545,728 bytes device memory.
- NVIDIA driver package 570.86.15.
- 462 telemetry samples; the measured worker was observed on the selected GPU in 455 samples.
- Maximum sampled GPU utilization: 100%.
- Maximum sampled GPU memory use: 2,924 MiB.
- No competing compute process was observed on the selected GPU. The Slurm node itself was shared/MIXED, so this is not an exclusive-node measurement.

## Correctness result

- Stock Astra CPU48, stock Astra CPU64, and H200 Astra produced byte-identical TSV output.
- Output SHA-256: `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`.
- Output size: 39,010,327 bytes; 383,235 lines including the header.
- H200 first-1,000 semantic gate also passed exact CPU/GPU comparison across all 27,481 profiles with zero model mismatches and non-vacuous GPU work.
- Native HMMER versus Astra structural normalization matched all 383,234 domain rows, identities, states, multiplicities, and coordinates. Fourteen extreme E-values were printed as literal `0` by native HMMER `%g` formatting while Astra retained positive subnormal values; this was the sole strict decimal-interval comparison failure. Bitscore intervals all overlapped.

## Queue-depth experiment status

The depth-1/2/4, 1-GiB byte-bounded ready-queue experiment has been implemented and independently source-audited, but **it has not been run on H200**. There is no queue-depth performance claim yet.

- Frozen harness inventory: `f4e8ee02bc494e0aaa7d766c1ebad536a45a9c4062b551f1b2d4c0a29bad633e`.
- Frozen-record SHA-256: `0ebf41fe33625772b1369c00ecaec12f1718ac8ab49ebf369f370352c334d4c4`.
- Host/static validation: native integration 109 pass / 42 CUDA-hidden skips; Astra queue 70 pass / 4 CUDA-hidden skips; final harness 28/28 pass.
- No queue-sweep Slurm job or H200 timing attempt was launched.

## Sealed local evidence

These paths exist in the development workspace and are excluded from Git because they contain bulky run/runtime artifacts. Their hashes make this summary auditable locally.

- Final independent H200 audit: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06-independent-final-postrun-audit.json`, SHA-256 `8ff325fc1de2509881082a1c6a579154006171d9ba283bb07d810bad4aaeb5a7`.
- H200 worker record: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06/runs/h200-full/worker.json`, SHA-256 `6fc6046a91cca01506ff344d1be4bacd81410334a80df62ab82a12459a4ba8d9`.
- H200 raw validation: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06/runs/h200-full/raw-validation.json`, SHA-256 `56caf970587bd6a5c42637e3208fb4bd2e944cec230567637672e94dca5c3248`.
- Final H200 attempt seal: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06/FINAL_ATTEMPT_SEALED`, SHA-256 `13e27632fc77cba4ed981a3651a1bf009ddfeded5ba5cfea8c8aee407fa313d1`.
- Final H200 evidence manifest: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06/final-evidence.sha256`, SHA-256 `02f049611e5585ccbf82c3e4c3d692346f0358874a1cc2a008a6d53f29a0eb7c`; 78/78 entries verified.
- Exact CPU/GPU output comparison: `build/astra-h200-full-plm-pfam-slurm-20260817/attempt-06/comparison/full-output-comparison.json`, SHA-256 `c32745fdce9ce64c5204dac02d9aa7611b089206e3156e09f5e14807dfa248f0`.
- Phase 2 full run: `build/phase1b-benchmark-harness/build/h200-phase2-device-compaction-20260823/attempt-02-full/runs/h200-full`; worker SHA-256 `dcc3da2eafdcf076ab76a867472282cc290a529b8e7b10e64906e4a5709363f9`, raw-validation SHA-256 `7428eb2fd2c4cf152564f51b512b6f6ec0636b21cf1bed2945e98f7b8e022f2b`.
- Phase 3 Forward-to-Backward residency full run: `build/phase1b-benchmark-harness/build/h200-phase3-forward-residency-20260823/attempt-02-full/runs/h200-full`; worker SHA-256 `1e33ad05c1d38bcc01cbb24f61577c0375fb2420d964cf3a48a82694cec6c083`, raw-validation SHA-256 `688f20c47de5dc982af530457594a9a5ea61383be13c7cff37e2bb7d4e9794dd`.
- CPU48 raw manifest: `build/astra-full-plm-pfam-slurm-20260817/attempt-02-reviewed-retry/runs/cpu48/artifact.sha256`, manifest SHA-256 `9e6d4d96073c565a9b61fac5b804f98dd5f7ec57dd67af07c5a289b198dadd42`.
- CPU64 raw manifest: `build/astra-full-plm-pfam-slurm-20260817/attempt-02-reviewed-retry/runs/cpu64/artifact.sha256`, manifest SHA-256 `e3a2e4cd5674addfe347aa8c4604571a5bdc38f14b8e9fdade60f71cf5f35b46`.
- CPU semantic normalization report: `build/astra-full-plm-pfam-cpu-comparison-20260817/runs/validation-01/comparison.json`, SHA-256 `78bee4fd2d03c2658d0779345893f89e6dc3777f1a380123a85c25c9a9b2525`.

## Review cautions

- This is source and performance-evidence packaging, not a release artifact.
- `integrations/astra/` preserves the exact Astra source snapshot and history for review; it is not installed by the root build.
- The production path currently depends on a matching private-ABI PyHMMER build and exact native extension pair.
- The Makefile/runtime packaging still needs relocatable library paths and explicit third-party license/notice handling before binary distribution.
- Do not infer a queue-depth speedup from the host tests; the H200 sweep remains unmeasured.
