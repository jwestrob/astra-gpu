#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)

pristine_source="$project_root/refs/src/hmmer-3.4"
timed_source="$project_root/refs/src/hmmer-3.4-stage-timing"
install_dir="$project_root/refs/install/hmmer-3.4-stage-timing"
patch_file="$project_root/baseline/patches/hmmer-3.4-stage-timing.patch"
validator="$project_root/baseline/patches/validate_stage_timing.py"
build_jobs=${PLAN7_BUILD_JOBS:-8}

if [[ -L "$pristine_source" ]]; then
    printf 'Refusing symlinked pristine HMMER source: %s\n' "$pristine_source" >&2
    exit 1
fi
if [[ ! -x "$pristine_source/configure" ]]; then
    printf 'Missing pristine HMMER 3.4 source: %s\n' "$pristine_source" >&2
    exit 1
fi
if [[ ! -f "$patch_file" ]]; then
    printf 'Missing timing patch: %s\n' "$patch_file" >&2
    exit 1
fi
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PLAN7_BUILD_JOBS must be a positive integer\n' >&2
    exit 1
fi

# Guard the only recursive removals with exact, non-configurable targets.
if [[ "$timed_source" != "$project_root/refs/src/hmmer-3.4-stage-timing" ||
      "$install_dir" != "$project_root/refs/install/hmmer-3.4-stage-timing" ]]; then
    printf 'Refusing unsafe build target\n' >&2
    exit 1
fi

mkdir -p "$project_root/refs/src" "$project_root/refs/install"
rm -rf -- "$timed_source" "$install_dir"
cp -a -- "$pristine_source" "$timed_source"

# The pristine tree may already be configured or built. Clean only the copy.
if [[ -f "$timed_source/Makefile" ]]; then
    make --directory="$timed_source" distclean
fi
patch --batch --forward --strip=1 --directory="$timed_source" --input="$patch_file"

(
    cd "$timed_source"
    ./configure --prefix="$install_dir"
)
make --directory="$timed_source" --jobs="$build_jobs"
if [[ ${PLAN7_SKIP_CHECK:-0} != 1 ]]; then
    make --directory="$timed_source" --jobs="$build_jobs" check
fi
make --directory="$timed_source" install
make --directory="$timed_source/easel" install

smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/plan7-stage-timing.XXXXXX")
trap 'rm -rf -- "$smoke_dir"' EXIT
hmmsearch="$install_dir/bin/hmmsearch"
query="$timed_source/tutorial/globins4.hmm"
targets="$timed_source/tutorial/globins45.fa"

"$hmmsearch" --cpu 0 --stage-timing "$smoke_dir/serial.tsv" "$query" "$targets" >/dev/null
"$hmmsearch" --cpu 2 --stage-timing "$smoke_dir/threaded.tsv" "$query" "$targets" >/dev/null
python3 "$validator" "$smoke_dir/serial.tsv" "$smoke_dir/threaded.tsv"

printf 'Built timed hmmsearch: %s\n' "$hmmsearch"
