#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "$probe_dir/../.." && pwd)
probe_so=${PROBE_SO:-"$repo_root/build/astra-stage-probe/libplan7_astra_stage_probe.so"}
fasta=${ASTRA_PROBE_FASTA:-"$repo_root/results/datasets/PLM2_5.first1000.faa"}
hmm=${ASTRA_PROBE_HMM:-"/groups/banfield/users/jwestrob/.config/Astra/HydDB/HydDB_all_MM2022.hmm"}
threads=${ASTRA_PROBE_THREADS:-2}

astra_entry=${ASTRA_BIN:-}
if [[ -z "$astra_entry" ]]; then
  if command -v pyenv >/dev/null 2>&1 && pyenv which astra >/dev/null 2>&1; then
    astra_entry=$(pyenv which astra)
  else
    astra_entry=$(command -v astra)
  fi
fi

for input in "$fasta" "$hmm"; do
  if [[ ! -s "$input" ]]; then
    echo "required smoke input is missing or empty: $input" >&2
    exit 1
  fi
done

"$probe_dir/build.sh" "$probe_so" >/dev/null

sequence_count=$(rg -c '^>' "$fasta")
profile_count=$(rg -c '^HMMER3/' "$hmm")
expected_calls=$((sequence_count * profile_count))

smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/plan7-astra-probe-smoke.XXXXXX")
cleanup() {
  rm -rf -- "$smoke_dir"
}
trap cleanup EXIT

control_dir="$smoke_dir/control"
observed_dir="$smoke_dir/observed"
probe_tsv="$smoke_dir/probe.tsv"
parent_only_tsv="$smoke_dir/parent-only.tsv"

# A benchmark parent inherits the preload environment but never loads HMMER.
# It must not overwrite the child Astra report with an all-zero destructor dump.
env LD_PRELOAD="$probe_so" PLAN7_ASTRA_STAGE_PROBE="$parent_only_tsv" /bin/true
if [[ -e "$parent_only_tsv" ]]; then
  echo "probe emitted a report without an observed p7_Pipeline call" >&2
  exit 1
fi

env -u PLAN7_ASTRA_STAGE_PROBE \
  "$astra_entry" search \
  --prot_in "$fasta" \
  --hmm_in "$hmm" \
  --outdir "$control_dir" \
  --threads "$threads" \
  >"$smoke_dir/control.stdout" \
  2>"$smoke_dir/control.stderr"

preload=$probe_so
if [[ -n "${LD_PRELOAD:-}" ]]; then
  preload="$probe_so:$LD_PRELOAD"
fi
env LD_PRELOAD="$preload" PLAN7_ASTRA_STAGE_PROBE="$probe_tsv" \
  "$astra_entry" search \
  --prot_in "$fasta" \
  --hmm_in "$hmm" \
  --outdir "$observed_dir" \
  --threads "$threads" \
  >"$smoke_dir/observed.stdout" \
  2>"$smoke_dir/observed.stderr"

control_hits="$control_dir/user_hmms_hits_df.tsv"
observed_hits="$observed_dir/user_hmms_hits_df.tsv"
cmp --silent "$control_hits" "$observed_hits"

validation=$(
  python3 "$probe_dir/validate_probe.py" \
    "$probe_tsv" \
    --expect-pipeline-calls "$expected_calls"
)
checksum=$(sha256sum "$control_hits" | cut -d ' ' -f 1)

printf 'scientific_output_byte_identical=true\n'
printf 'scientific_output_sha256=%s\n' "$checksum"
printf 'sequences=%s profiles=%s comparisons=%s threads=%s\n' \
  "$sequence_count" "$profile_count" "$expected_calls" "$threads"
printf 'probe_validation=%s\n' "$validation"
