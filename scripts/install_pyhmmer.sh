#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
pyhmmer_version=0.12.1
psutil_version=7.2.2
plan7_python=${PLAN7_PYTHON:-3.11.15}
environment_dir="$project_root/env/pyhmmer-${pyhmmer_version}"

if [[ ! -x "$environment_dir/bin/python" ]]; then
    uv venv --python "$plan7_python" "$environment_dir"
fi

uv pip install --python "$environment_dir/bin/python" \
    "pyhmmer==${pyhmmer_version}" "psutil==${psutil_version}"

"$environment_dir/bin/python" - <<'PY'
import json
import platform
import pyhmmer
import psutil

print(json.dumps({
    "python": platform.python_version(),
    "pyhmmer": pyhmmer.__version__,
    "psutil": psutil.__version__,
}, sort_keys=True))
PY
