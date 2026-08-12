#!/usr/bin/env bash
set -euo pipefail

binary=${1:-./build/schedforge-bench}
shift || true

perf stat \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  "$binary" "$@"
