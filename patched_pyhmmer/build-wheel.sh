#!/usr/bin/env bash
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "$here/.." && pwd)
output_dir=${1:-"$repo_root/build/pyhmmer-patched"}
bootstrap_python=${PYTHON:-python3}
compiler=${CC:-gcc}

version=0.12.0
local_version=0.12.0+plan7gpu.0
sdist_sha256=27cdfd3cdf72abcc7a6670825fc0195e8ff01d4820efd0f99aec03d3972a922e
sdist_url=https://files.pythonhosted.org/packages/b7/55/44372be1e0883df7c056bbfb979d329c033db6a1b5b304144fbaebdd55e7/pyhmmer-0.12.0.tar.gz

work=$(mktemp -d "${TMPDIR:-/tmp}/plan7-pyhmmer-build.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

if [[ -n "${PYHMMER_SDIST:-}" ]]; then
  sdist=$PYHMMER_SDIST
else
  sdist=$work/pyhmmer-$version.tar.gz
  curl --fail --location --retry 3 --output "$sdist" "$sdist_url"
fi
printf '%s  %s\n' "$sdist_sha256" "$sdist" | sha256sum --check --status

if [[ "${PLAN7_ALLOW_TOOLCHAIN_DRIFT:-0}" != 1 ]]; then
  [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
  [[ $($bootstrap_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') == 3.12 ]]
  [[ $($compiler -dumpfullversion -dumpversion) == 11.4.0 ]]
  [[ $(cmake --version | head -n 1) == "cmake version 3.24.0" ]]
fi

tar -xzf "$sdist" -C "$work"
source_root=$work/pyhmmer-$version
patch --batch --forward -d "$source_root" -p1 < "$here/pyhmmer-0.12.0-plan7gpu.patch"
patch --batch --forward -d "$source_root" -p1 < "$here/pyhmmer-0.12.0-simple-regions.patch"

$bootstrap_python -m venv "$work/build-venv"
build_python=$work/build-venv/bin/python
$build_python -m pip install --quiet \
  "pip==26.2.1" \
  "build==1.5.0" \
  "Cython==3.2.9" \
  "scikit-build-core==1.0.3" \
  "ninja==1.13.0"

export CC=$compiler
export CMAKE_GENERATOR=Ninja
export SOURCE_DATE_EPOCH=1769106080
export BUILD_SHARED_LIBS=true
export PYHMMER_INSTALL_LIBS=true

mkdir -p "$work/dist"
$build_python -m build \
  --wheel \
  --no-isolation \
  --config-setting=cmake.build-type=Release \
  --outdir "$work/dist" \
  "$source_root"

wheel=$work/dist/pyhmmer-${local_version}-cp312-abi3-linux_x86_64.whl
[[ -f "$wheel" ]]
mkdir -p "$output_dir"
cp -- "$wheel" "$output_dir/"
wheel=$output_dir/${wheel##*/}

printf 'wheel=%s\n' "$wheel"
printf 'wheel_sha256=%s\n' "$(sha256sum "$wheel" | cut -d' ' -f1)"
printf 'patch_sha256=%s\n' "$(sha256sum "$here/pyhmmer-0.12.0-plan7gpu.patch" | cut -d' ' -f1)"
printf 'simple_regions_patch_sha256=%s\n' "$(sha256sum "$here/pyhmmer-0.12.0-simple-regions.patch" | cut -d' ' -f1)"
