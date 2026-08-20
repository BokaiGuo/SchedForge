#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
clang_bin=${CLANG_AARCH64:-$(command -v clang++-18 || command -v clang++)}
qemu_bin=${QEMU_AARCH64:-$(command -v qemu-aarch64 || true)}
source_file=${NEON_SOURCE:-"${root_dir}/results/generated_neon_matmul.cpp"}
smoke_file=${NEON_SMOKE:-"${root_dir}/tests/neon_runtime_smoke.cpp"}
output_file=${NEON_BINARY:-"${root_dir}/build/neon-runtime-smoke"}

if [[ -z "${qemu_bin}" ]]; then
  echo "qemu-aarch64 is required for the NEON runtime smoke" >&2
  exit 2
fi
if [[ ! -f "${source_file}" || ! -f "${smoke_file}" ]]; then
  echo "NEON source or smoke file is missing" >&2
  exit 2
fi
mkdir -p "$(dirname "${output_file}")"

binutils_root=${AARCH64_BINUTILS_ROOT:-}
if [[ -n "${binutils_root}" ]]; then
  export PATH="${binutils_root}/usr/bin:${PATH}"
  export LD_LIBRARY_PATH="${binutils_root}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  linker_prefix="-B${binutils_root}/usr/bin"
else
  linker_prefix=""
fi

"${clang_bin}" --target=aarch64-linux-gnu -march=armv8-a+simd \
  -ffreestanding -fno-builtin -nostdlib -static ${linker_prefix} \
  -Wl,-e,_start "${source_file}" "${smoke_file}" -o "${output_file}"
file "${output_file}"
"${qemu_bin}" "${output_file}"
echo "NEON AArch64 runtime smoke passed"
