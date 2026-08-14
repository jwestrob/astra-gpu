#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)

hmmer_version=3.4
hmmer_archive="hmmer-${hmmer_version}.tar.gz"
# The upstream release host currently serves tarballs over HTTP only. The
# archive is authenticated below with the published SHA-256 digest.
hmmer_url="http://eddylab.org/software/hmmer/${hmmer_archive}"
hmmer_sha256="ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3"

download_dir="$project_root/refs/downloads"
source_dir="$project_root/refs/src/hmmer-${hmmer_version}"
install_dir="$project_root/refs/install/hmmer-${hmmer_version}"
archive_path="$download_dir/$hmmer_archive"
plan7_build_jobs=${PLAN7_BUILD_JOBS:-8}

mkdir -p "$download_dir" "$project_root/refs/src" "$install_dir"

if [[ ! -f "$archive_path" ]]; then
    curl --fail --location --retry 3 --output "$archive_path" "$hmmer_url"
fi

printf '%s  %s\n' "$hmmer_sha256" "$archive_path" | sha256sum --check --strict

if [[ ! -x "$source_dir/configure" ]]; then
    tar --extract --gzip --file "$archive_path" --directory "$project_root/refs/src"
fi

if [[ ! -f "$source_dir/Makefile" ]]; then
    (
        cd "$source_dir"
        ./configure --prefix="$install_dir"
    )
fi

make --directory="$source_dir" --jobs="$plan7_build_jobs"
if [[ ${PLAN7_SKIP_CHECK:-0} != 1 ]]; then
    make --directory="$source_dir" --jobs="$plan7_build_jobs" check
fi
make --directory="$source_dir" install
make --directory="$source_dir/easel" install

"$install_dir/bin/hmmsearch" -h | sed -n '1,4p'
