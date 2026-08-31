#!/usr/bin/env bash
#SBATCH --account=banfield
#SBATCH --qos=standard
#SBATCH --partition=gpu_h200
#SBATCH --nodelist=node-224-2t-8gpu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=astra-kofam-balance

# Direct full PLM2_5 x KOFAM E-only one-factor sweep. This intentionally
# leaves ready-queue depth and profile chunking at their retained defaults.
set -Eeuo pipefail

ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/hmmer_gpu
SUPPORT=$ROOT/build/kofam-h200-current-main-20260831
RUNTIME=$ROOT/build/astra-h200-full-plm-pfam-slurm-20260817/runtime-v2-h200-full
PYTHON=$RUNTIME/bin/python
INPUT=/groups/banfield/projects/environmental/EastRiver/hillslope/assembly.d/PLM_jun2017/PLM2_5_b1/PLM2_5_b1_jun17_idba_ud/PLM2_5_b1_jun17_scaffold_min1000.fa.genes.faa
DB=/groups/banfield/users/jwestrob/.config/Astra/KOFAM/KOFAM
REFERENCE=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/metacerberus_benchmark_20260830/astra-gpu-kofam/runs/matrix-1186921/outputs/cpu48/output/KOFAM_hits_df.tsv
MANIFEST=$SUPPORT/inputs/KOFAM.manifest.json
ARMS=(control shard15 forward150 forward100 workers48 workers32)

configure_arm() {
  local label=$1
  unset PLAN7_GPU_CONTINUATION_SHARD_TRIGGER \
    PLAN7_GPU_FORWARD_CPU_MAX_CELLS ASTRA_GPU_CONTINUATION_WORKERS
  case "$label" in
    control) ;;
    shard15) export PLAN7_GPU_CONTINUATION_SHARD_TRIGGER=1.5 ;;
    forward150) export PLAN7_GPU_FORWARD_CPU_MAX_CELLS=150000 ;;
    forward100) export PLAN7_GPU_FORWARD_CPU_MAX_CELLS=100000 ;;
    workers48) export ASTRA_GPU_CONTINUATION_WORKERS=48 ;;
    workers32) export ASTRA_GPU_CONTINUATION_WORKERS=32 ;;
    *) printf 'unknown arm: %s\n' "$label" >&2; return 2 ;;
  esac
}

run_arm() {
  local label=$1
  local arm=$2
  local stop=$arm/monitor.stop
  local cgroup_pid gpu_pid status gpu_id

  test ! -e "$arm"
  mkdir -p "$arm/provenance" "$arm/output-parent"
  configure_arm "$label"
  env | LC_ALL=C sort | grep -E \
    '^(PLAN7_GPU_|ASTRA_GPU_|LD_PRELOAD|LD_LIBRARY_PATH|PYTHONPATH)=' \
    >"$arm/provenance/effective-environment.txt"
  "$PYTHON" -B - "$label" >"$arm/provenance/resolved-balance-knobs.json" <<'PY'
import json
import os
import sys
from plan7_gpu._private_tuning import (
    auto_forward_cpu_max_cells,
    continuation_shard_trigger,
)

numerator, denominator = continuation_shard_trigger()
print(json.dumps({
    "arm": sys.argv[1],
    "continuation_shard_trigger": {
        "numerator": numerator,
        "denominator": denominator,
        "decimal": numerator / denominator,
    },
    "forward_cpu_max_cells": auto_forward_cpu_max_cells(),
    "continuation_workers": os.environ.get(
        "ASTRA_GPU_CONTINUATION_WORKERS", "auto-after-producer-reservation"
    ),
}, indent=2, sort_keys=True))
PY

  "$SUPPORT/verify_physical_affinity.py" \
    --expected 64 --mode select --mask-file "$arm/selected-cpus.txt" \
    --evidence "$arm/provenance/physical-affinity.json" \
    >"$arm/provenance/selected-cpu-list.txt"
  "$SUPPORT/monitor_cgroup_memory.sh" "$arm/cgroup-memory.tsv" "$stop" &
  cgroup_pid=$!
  gpu_id=${CUDA_VISIBLE_DEVICES%%,*}
  nvidia-smi --id="$gpu_id" \
    --query-gpu=timestamp,index,uuid,name,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits --loop=1 >"$arm/hbm-samples.csv" &
  gpu_pid=$!
  stop_monitors() {
    touch "$stop"
    wait "$cgroup_pid" 2>/dev/null || true
    kill "$gpu_pid" 2>/dev/null || true
    wait "$gpu_pid" 2>/dev/null || true
  }
  trap stop_monitors EXIT INT TERM

  set +e
  /usr/bin/time --verbose --output="$arm/time.txt" \
    "$PYTHON" -B "$SUPPORT/benchmark.py" \
      --input "$INPUT" --manifest "$MANIFEST" \
      --output-root "$arm/output-parent/output" \
      --summary "$arm/result.json" \
      --scheduler-telemetry "$arm/scheduler-telemetry.json" \
      >"$arm/stdout.log" 2>"$arm/stderr.log"
  status=$?
  set -e
  printf '%s\n' "$status" >"$arm/status.txt"
  stop_monitors
  trap - EXIT INT TERM
  test "$status" -eq 0
  cmp -s "$arm/output-parent/output/KOFAM_hits_df.tsv" "$REFERENCE"
}

if [[ ${1-} == --arm ]]; then
  test $# -eq 3
  run_arm "$2" "$3"
  exit 0
fi

test $# -eq 0
test "${SLURM_JOB_PARTITION-}" = gpu_h200
test "${SLURM_CPUS_PER_TASK-}" = 64
test "${SLURM_THREADS_PER_CORE-}" = 1
test "$(hostname -s)" = node-224-2t-8gpu-1
test -n "${CUDA_VISIBLE_DEVICES-}"
: "${KOFAM_SWEEP_CORE:?set to the clean, built SIMD-plus-balance core worktree}"
: "${KOFAM_SWEEP_ASTRA:?set to the clean Astra balance worktree}"
: "${KOFAM_SWEEP_CORE_COMMIT:?set to the exact expected core commit}"
: "${KOFAM_SWEEP_ASTRA_COMMIT:?set to the exact expected Astra commit}"
CORE=$(readlink -f "$KOFAM_SWEEP_CORE")
ASTRA=$(readlink -f "$KOFAM_SWEEP_ASTRA")
test "$(git -C "$CORE" rev-parse HEAD)" = "$KOFAM_SWEEP_CORE_COMMIT"
test "$(git -C "$ASTRA" rev-parse HEAD)" = "$KOFAM_SWEEP_ASTRA_COMMIT"
test -z "$(git -C "$CORE" status --porcelain=v1 --untracked-files=no)"
test -z "$(git -C "$ASTRA" status --porcelain=v1 --untracked-files=no)"
test -f "$CORE/python/plan7_gpu/_native.cpython-312-x86_64-linux-gnu.so"
test -f "$CORE/python/plan7_gpu/_pipeline.cpython-312-x86_64-linux-gnu.so"

RUN_ROOT=${KOFAM_SWEEP_OUTPUT_ROOT:-$ROOT/build/kofam-balance-sweep/runs/job-$SLURM_JOB_ID}
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/provenance"
export LC_ALL=C
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_PRELOAD=/home/jwestrob/dns_shim.so
export LD_LIBRARY_PATH=$RUNTIME/python-base/lib:$RUNTIME/lib/python3.12/site-packages/pyhmmer.libs
export PYTHONPATH=$CORE/python:$ASTRA
unset MALLOC_ARENA_MAX ASTRA_CPU_MALLOC_ARENA_MAX \
  ASTRA_GPU_PROFILE_SERIAL ASTRA_GPU_PROFILE_LEGACY_OVERLAP \
  ASTRA_GPU_READY_QUEUE_DEPTH ASTRA_GPU_READY_QUEUE_BYTES \
  PLAN7_GPU_FORWARD_OWNERSHIP PLAN7_GPU_FORWARD_CPU_MIN_CELLS \
  PLAN7_GPU_FORWARD_CPU_MIN_LENGTH
export PLAN7_GPU_GENERATION_LEDGER=1
export PLAN7_GPU_DOMAIN_OWNERSHIP=cpu
export PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP=1
export PLAN7_GPU_F1_RAW_XE=1
export PLAN7_GPU_SSV_IDENTITY_PADDING=1
export PLAN7_GPU_VIT_LENGTH_CACHE=1
export PLAN7_GPU_CONTINUATION_SCHEDULER=completion
export PLAN7_GPU_CONTINUATION_TASK_POLICY=sharded
export PLAN7_GPU_CONTINUATION_PROFILE=1
export ASTRA_GPU_CONTINUATION_POOL=1
export ASTRA_GPU_OVERLAP_TIMING=1

git -C "$CORE" show --no-patch --format='%H %T %P %s' \
  >"$RUN_ROOT/provenance/core-commit.txt"
git -C "$ASTRA" show --no-patch --format='%H %T %P %s' \
  >"$RUN_ROOT/provenance/astra-commit.txt"
# Verify and warm every immutable full-search input once; no prime, smoke, or
# reduced request runs before the measured arms.
sha256sum "$DB.h3m" "$DB.h3i" "$DB.h3f" "$DB.h3p" \
  "$INPUT" "$MANIFEST" "$REFERENCE" \
  >"$RUN_ROOT/provenance/input-sha256.txt"
sha256sum "$CORE/python/plan7_gpu/_native.cpython-312-x86_64-linux-gnu.so" \
  "$CORE/python/plan7_gpu/_pipeline.cpython-312-x86_64-linux-gnu.so" \
  "$CORE/python/plan7_gpu/astra_search.py" \
  "$CORE/python/plan7_gpu/_private_tuning.py" "$ASTRA/astra/search.py" \
  "$SUPPORT/benchmark.py" "${BASH_SOURCE[0]}" \
  >"$RUN_ROOT/provenance/executed-artifact-sha256.txt"

for label in "${ARMS[@]}"; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=64 --gpus-per-task=1 \
    --hint=nomultithread --cpu-bind=cores --gpu-bind=closest \
    "${BASH_SOURCE[0]}" --arm "$label" "$RUN_ROOT/$label"
done

"$PYTHON" -B - "$RUN_ROOT" "${ARMS[@]}" <<'PY'
import csv
import json
from pathlib import Path
import sys


root = Path(sys.argv[1])
labels = sys.argv[2:]
rows = []
for label in labels:
    arm = root / label
    result = json.loads((arm / "result.json").read_text())
    metrics = result["gpu_overlap_metrics"]
    scheduler = result["continuation_scheduler_summary"]["scalars"]
    time_values = {}
    for line in (arm / "time.txt").read_text().splitlines():
        if ": " in line:
            key, value = line.strip().split(": ", 1)
            time_values[key] = value
    hbm = []
    gpu_utilization = []
    for line in (arm / "hbm-samples.csv").read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 8:
            hbm.append(float(fields[4]))
            gpu_utilization.append(float(fields[5]))
    continuation_seconds = metrics["continuation_seconds"]
    average_active = (
        scheduler["task_wall_ns"] / 1e9 / continuation_seconds
        if continuation_seconds else 0.0
    )
    workers = metrics["continuation_worker_count"]
    rows.append({
        "arm": label,
        "status": result["status"],
        "output_sha256": result["output"]["sha256"],
        "output_bytes": result["output"]["bytes"],
        "output_rows": result["output"]["data_rows"],
        "raw_order_exact": result["output_oracle"]["raw_order_exact"],
        "request_seconds": result["request_wall_seconds"],
        "generation_seconds": metrics["generation_seconds"],
        "continuation_seconds": continuation_seconds,
        "overlap_seconds": metrics["overlap_seconds"],
        "pipeline_seconds": metrics["pipeline_wall_seconds"],
        "maximum_rss_kib": int(
            time_values["Maximum resident set size (kbytes)"]
        ),
        "search_peak_rss_kib": result["search_peak_rss_kib"],
        "peak_hbm_mib": max(hbm) if hbm else None,
        "mean_gpu_utilization_percent": (
            sum(gpu_utilization) / len(gpu_utilization)
            if gpu_utilization else None
        ),
        "continuation_workers": workers,
        "task_count": scheduler["task_count"],
        "shard_task_count": scheduler["shard_task_count"],
        "maximum_active_workers": scheduler["maximum_active_workers"],
        "average_active_workers": average_active,
        "average_task_occupancy_fraction": (
            average_active / workers if workers else 0.0
        ),
    })
if not all(row["status"] == "PASS" and row["raw_order_exact"] for row in rows):
    raise SystemExit("one or more full KOFAM arms failed exactness")
(root / "sweep-summary.json").write_text(
    json.dumps({"schema": "kofam-balance-one-factor-v1", "arms": rows},
               indent=2, sort_keys=True) + "\n"
)
with (root / "sweep-summary.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY
