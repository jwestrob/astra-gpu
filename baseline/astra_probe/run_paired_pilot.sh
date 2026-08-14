#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "usage: $0 OUTPUT_DIR DATASET_ID FASTA HMM_ARTIFACT INSTALLED_DB [ASTRA_SEARCH_OPTION ...]" >&2
  exit 2
fi

probe_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "$probe_dir/../.." && pwd)
output_dir=$1
dataset_id=$2
fasta=$3
hmm_artifact=$4
installed_db=$5
shift 5

replicates=${ASTRA_PROBE_REPLICATES:-1}
warmups=${ASTRA_PROBE_WARMUPS:-1}
astra_bin=${ASTRA_BIN:-/home/jwestrob/.pyenv/versions/3.12.0/bin/astra}
astra_config=${ASTRA_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/Astra/hmm_databases.json}
probe_so=${PROBE_SO:-$repo_root/build/astra-stage-probe/libplan7_astra_stage_probe.so}
summary=${ASTRA_PROBE_SUMMARY:-${output_dir}-summary.json}

cd "$repo_root"

for value in "$replicates" "$warmups"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "replicate and warmup counts must be nonnegative integers" >&2
    exit 2
  fi
done
if (( replicates < 1 )); then
  echo "ASTRA_PROBE_REPLICATES must be at least 1" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output directory: $output_dir" >&2
  exit 1
fi
if [[ -e "$summary" ]]; then
  echo "refusing to overwrite existing summary: $summary" >&2
  exit 1
fi
for input in "$fasta" "$hmm_artifact"; do
  if [[ ! -s "$input" ]]; then
    echo "required input is missing or empty: $input" >&2
    exit 1
  fi
done
if [[ ! -s "$astra_config" ]]; then
  echo "Astra database config is missing or empty: $astra_config" >&2
  exit 1
fi
if [[ ! -x "$astra_bin" ]]; then
  echo "Astra executable is unavailable: $astra_bin" >&2
  exit 1
fi

"$probe_dir/build.sh" "$probe_so" >/dev/null
mkdir -p -- "$output_dir"

astra_version=$(jq -r '.astra.version' "$repo_root/results/software.json")
astra_revision=$(jq -r '.astra.revision' "$repo_root/results/software.json")
pyhmmer_version=$(jq -r '.astra.pyhmmer_dependency' "$repo_root/results/software.json")
preload=$probe_so
if [[ -n "${LD_PRELOAD:-}" ]]; then
  preload="$probe_so:$LD_PRELOAD"
fi

metadata_json() {
  local role=$1
  local run_kind=$2
  local run_index=$3
  jq -cn \
    --arg engine astra \
    --arg astra_version "$astra_version" \
    --arg astra_revision "$astra_revision" \
    --arg pyhmmer_version "$pyhmmer_version" \
    --arg dataset_id "$dataset_id" \
    --arg role "$role" \
    --arg run_kind "$run_kind" \
    --argjson run_index "$run_index" \
    '{engine:$engine,astra_version:$astra_version,astra_revision:$astra_revision,
      pyhmmer_version:$pyhmmer_version,dataset_id:$dataset_id,role:$role,
      run_status:"pilot",threads:1,run_kind:$run_kind,run_index:$run_index}'
}

run_control() {
  local run_kind=$1
  local run_index=$2
  shift 2
  local stem=${dataset_id}-astra-control-cpu1-${run_kind}${run_index}
  "$repo_root/scripts/run_benchmark.py" \
    --output "$output_dir/$stem.json" \
    --stdout "$output_dir/$stem.out" \
    --stderr "$output_dir/$stem.err" \
    --label "$stem" \
    --hash-outputs \
    --set-env "PLAN7_ASTRA_STAGE_PROBE=" \
    --metadata-json "$(metadata_json astra_uninstrumented_control "$run_kind" "$run_index")" \
    -- "$astra_bin" search \
    --prot_in "$fasta" \
    --installed_hmms "$installed_db" \
    --outdir "$output_dir/$stem-astra-output" \
    --threads 1 \
    "$@"
}

run_observed() {
  local run_kind=$1
  local run_index=$2
  shift 2
  local stem=${dataset_id}-astra-probe-cpu1-${run_kind}${run_index}
  "$repo_root/scripts/run_benchmark.py" \
    --output "$output_dir/$stem.json" \
    --stdout "$output_dir/$stem.out" \
    --stderr "$output_dir/$stem.err" \
    --label "$stem" \
    --hash-outputs \
    --set-env "LD_PRELOAD=$preload" \
    --set-env "PLAN7_ASTRA_STAGE_PROBE=$output_dir/$stem.tsv" \
    --metadata-json "$(metadata_json astra_in_process_stage_probe "$run_kind" "$run_index")" \
    -- "$astra_bin" search \
    --prot_in "$fasta" \
    --installed_hmms "$installed_db" \
    --outdir "$output_dir/$stem-astra-output" \
    --threads 1 \
    "$@"
}

for (( index = 1; index <= warmups; index++ )); do
  run_control warmup "$index" "$@"
done
for (( index = 1; index <= replicates; index++ )); do
  run_control replicate "$index" "$@"
  run_observed replicate "$index" "$@"
done

summary_args=()
hits_name=${installed_db}_hits_df.tsv
for (( index = 1; index <= replicates; index++ )); do
  control_stem=${dataset_id}-astra-control-cpu1-replicate${index}
  observed_stem=${dataset_id}-astra-probe-cpu1-replicate${index}
  summary_args+=(
    --control
    "$output_dir/$control_stem.json"
    "$output_dir/$control_stem-astra-output/$hits_name"
    --observed
    "$output_dir/$observed_stem.json"
    "$output_dir/$observed_stem.tsv"
    "$output_dir/$observed_stem-astra-output/$hits_name"
  )
done

python3 "$probe_dir/summarize_probe.py" \
  "${summary_args[@]}" \
  --fasta "$fasta" \
  --hmm "$hmm_artifact" \
  --astra-config "$astra_config" \
  --probe-binary "$probe_so" \
  --output "$summary"

printf 'summary=%s\n' "$summary"
