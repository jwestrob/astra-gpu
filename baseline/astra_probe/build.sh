#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "$probe_dir/../.." && pwd)
output=${1:-"$repo_root/build/astra-stage-probe/libplan7_astra_stage_probe.so"}
compiler=${CC:-cc}

astra_entry=${ASTRA_BIN:-}
if [[ -z "$astra_entry" ]]; then
  if command -v pyenv >/dev/null 2>&1 && pyenv which astra >/dev/null 2>&1; then
    astra_entry=$(pyenv which astra)
  else
    astra_entry=$(command -v astra)
  fi
fi

astra_python=${ASTRA_PYTHON:-}
if [[ -z "$astra_python" && -f "$astra_entry" ]]; then
  first_line=$(head -n 1 -- "$astra_entry")
  if [[ "$first_line" == '#!'* ]]; then
    astra_python=${first_line#\#!}
    astra_python=${astra_python%% *}
  fi
fi
astra_python=${astra_python:-python3}

readarray -t astra_abi < <("$astra_python" - <<'PY'
from pathlib import Path
import pyhmmer

library = Path(pyhmmer.__file__).resolve().parent.parent / "pyhmmer.libs" / "liblibhmmer.so"
if not library.is_file():
    raise SystemExit(f"Astra PyHMMER library not found: {library}")
print(library)
print(pyhmmer.__version__)
PY
)
target_library=${astra_abi[0]}
pyhmmer_version=${astra_abi[1]}

required_symbols=(
  p7_Pipeline
  p7_bg_NullOne
  p7_MSVFilter
  p7_SSVFilter
  p7_bg_FilterScore
  p7_ViterbiFilter
  p7_ForwardParser
  p7_BackwardParser
  p7_domaindef_ByPosteriorHeuristics
  p7_Forward
  p7_Backward
  p7_DomainDecoding
)
defined_symbols=$(nm -D --defined-only "$target_library")
for symbol in "${required_symbols[@]}"; do
  if ! rg -q "[[:space:]]${symbol}$" <<<"$defined_symbols"; then
    echo "Astra HMMER ABI is missing required symbol: $symbol" >&2
    exit 1
  fi
done

mkdir -p -- "$(dirname -- "$output")"
"$compiler" \
  -std=c11 \
  -O2 \
  -fPIC \
  -shared \
  -Wall \
  -Wextra \
  -Werror \
  -o "$output" \
  "$probe_dir/astra_stage_probe.c" \
  -ldl \
  -pthread

printf 'probe=%s\n' "$output"
printf 'astra=%s\n' "$astra_entry"
printf 'pyhmmer=%s\n' "$pyhmmer_version"
printf 'target=%s\n' "$target_library"
