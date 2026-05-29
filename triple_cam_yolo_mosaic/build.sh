#!/usr/bin/env bash
# Standalone cross-build for K230 big-core (RISC-V musl).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TOOLCHAIN_BIN="${TOOLCHAIN_BIN:-/opt/toolchain/riscv64-linux-musleabi_for_x86_64-pc-linux-gnu/bin}"
TOOLCHAIN_FILE="${SCRIPT_DIR}/../ai_poc/cmake/Riscv64.cmake"

export PATH="${PATH}:${TOOLCHAIN_BIN}"

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Missing ${TOOLCHAIN_FILE}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
      "${SCRIPT_DIR}"

make -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
make install

echo "Output:"
echo "  ${BUILD_DIR}/install/bin/triple_cam_yolo_mosaic.elf"
echo "Run on device:"
echo "  ./triple_cam_yolo_mosaic.elf <kmodel> [obj_thresh] [nms_thresh] [max_mosaics] [frame_interval_ms] [debug_mode] [capture_period_ms]"
