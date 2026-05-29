#!/usr/bin/env bash
# Little-core Linux (glibc): triple_cam_mosaic_person_reader.elf only — NOT musl / NOT link.lds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_linux_fifo_reader"
TOOLCHAIN_FILE="${SCRIPT_DIR}/../ai_poc/cmake/Riscv64_glibc.cmake"

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Missing ${TOOLCHAIN_FILE}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
      -DBUILD_TRIPLE_CAM_MOSAIC_LINUX_GLIBC=ON \
      "${SCRIPT_DIR}"

make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
make install

echo "Output (glibc / Linux toolchain only):"
echo "  ${BUILD_DIR}/install/bin/triple_cam_mosaic_person_reader.elf"
