#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PATCHED_WHEEL" >&2
  exit 2
fi

here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
wheel=$(realpath -- "$1")
compiler=${CC:-gcc}
bootstrap_python=${PYTHON:-python3}
work=$(mktemp -d "${TMPDIR:-/tmp}/plan7-pyhmmer-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

$bootstrap_python -m venv "$work/patched"
patched_python=$work/patched/bin/python
$patched_python -m pip install --quiet \
  "pip==26.2.1" \
  "Cython==3.2.9" \
  "$wheel"

readarray -t build_paths < <($patched_python - <<'PY'
from pathlib import Path
import sysconfig
import pyhmmer

package = Path(pyhmmer.__file__).resolve().parent
print(package.parent / "pyhmmer.libs")
print(sysconfig.get_paths()["include"])
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)
library_dir=${build_paths[0]}
python_include=${build_paths[1]}
extension_suffix=${build_paths[2]}

$patched_python -m cython \
  -3 \
  -E HMMER_IMPL=SSE \
  -E TARGET_SYSTEM=Linux \
  -I "$work/patched/lib/python3.12/site-packages" \
  -I "$library_dir/cython/include" \
  --output-file "$work/seam_probe.c" \
  "$here/seam_probe.pyx"

$compiler \
  -shared \
  -fPIC \
  -O2 \
  -msse4.1 \
  -I "$python_include" \
  -I "$library_dir/include" \
  -I "$library_dir/include/libhmmer" \
  -I "$library_dir/include/libeasel" \
  -L "$library_dir" \
  -Wl,--no-as-needed \
  -Wl,-rpath,"$library_dir" \
  -llibhmmer \
  -llibeasel \
  -o "$work/seam_probe$extension_suffix" \
  "$work/seam_probe.c"

defined_symbols=$(nm -D --defined-only "$library_dir/liblibhmmer.so")
for symbol in \
  p7_PipelineFromMSV \
  p7_PipelineFromFilterScores \
  p7_PipelineFromFilterAndForwardScores \
  p7_PipelineFromFilterAndForwardSimpleRegions \
  p7_PipelineFromFilterForwardAndCompactDomains \
  p7_pipeline_CompactTailFingerprint; do
  if ! rg -q "[[:space:]]${symbol}$" <<<"$defined_symbols"; then
    echo "patched HMMER library is missing $symbol" >&2
    exit 1
  fi
done

$bootstrap_python -m venv "$work/stock"
stock_python=$work/stock/bin/python
$stock_python -m pip install --quiet "pip==26.2.1" "pyhmmer===0.12.0"

stock_library=$($stock_python - <<'PY'
from pathlib import Path
import pyhmmer

package = Path(pyhmmer.__file__).resolve().parent
print(package.parent / "pyhmmer.libs" / "liblibhmmer.so")
PY
)
if nm -D --defined-only "$stock_library" | \
    rg -q '[[:space:]]p7_PipelineFromFilter(AndForwardSimpleRegions|ForwardAndCompactDomains)$'; then
  echo "stock HMMER unexpectedly exports a private domain seam" >&2
  exit 1
fi

stock_signature=$($stock_python "$here/ordinary_signature.py")
patched_signature=$($patched_python "$here/ordinary_signature.py")
if [[ "$stock_signature" != "$patched_signature" ]]; then
  echo "ordinary PyHMMER output changed" >&2
  echo "stock:   $stock_signature" >&2
  echo "patched: $patched_signature" >&2
  exit 1
fi

PYTHONPATH=$work $patched_python "$here/test_seam.py"
printf 'stock ordinary Pipeline parity: 4 searches\n'
