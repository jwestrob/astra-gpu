#!/usr/bin/env bash
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$here/../.." && pwd)
include_root=/home/jwestrob/.pyenv/versions/3.12.0/lib/python3.12/site-packages/pyhmmer.libs/include
target=/home/jwestrob/.pyenv/versions/3.12.0/lib/python3.12/site-packages/pyhmmer.libs/liblibhmmer.so
if [[ $# -gt 0 ]]; then
  output=$1
else
  output=$repo/build/post-forward-census/libplan7_post_forward_census.so
fi

defined_symbols=$(nm -D --defined-only "$target")
for symbol in \
  p7_BackwardParser p7_domaindef_ByPosteriorHeuristics p7_DomainDecoding \
  p7_Forward p7_Backward p7_StochasticTrace p7_spensemble_Cluster \
  p7_Decoding p7_Null2_ByExpectation p7_Null2_ByTrace p7_OptimalAccuracy \
  p7_OATrace p7_alidisplay_Create p7_tophits_SortBySortkey \
  p7_tophits_Threshold; do
  rg -q "[[:space:]]$symbol$" <<<"$defined_symbols" || {
    echo "missing HMMER symbol: $symbol" >&2
    exit 1
  }
done

mkdir -p -- "$(dirname -- "$output")"
cc -std=c11 -O2 -fPIC -shared -Wall -Wextra -Werror \
  -I"$include_root/libhmmer" -I"$include_root/libeasel" \
  -o "$output" "$here/probe.c" -ldl -pthread
printf 'probe=%s\ntarget=%s\n' "$output" "$target"
