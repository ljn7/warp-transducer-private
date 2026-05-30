#!/usr/bin/env bash
# install.sh - Build warp-transducer and install the PyTorch binding on Linux/macOS.
#
# Usage:
#   ./install.sh              # auto-detect CUDA, build everything, pip install
#   ./install.sh --no-gpu     # CPU-only build
#   NO_GPU=1 ./install.sh     # same as --no-gpu
#
# Requirements:
#   cmake >= 3.10
#   A C++ compiler (GCC >= 5 or Clang >= 8)
#   Python with PyTorch installed
#   Optional: CUDA toolkit (detected automatically)
#
# The script is intentionally short. For full control and Windows support, use:
#   python build.py --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BINDING_DIR="$SCRIPT_DIR/pytorch_binding"

NO_GPU="${NO_GPU:-0}"
if [[ "$1" == "--no-gpu" ]]; then
    NO_GPU=1
fi

echo ""
echo "============================================================"
echo " warp-transducer install"
echo " Platform: $(uname -s)"
echo " Build dir: $BUILD_DIR"
echo "============================================================"

# --- Require cmake ---
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install it with:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  macOS:         brew install cmake"
    exit 1
fi
echo " CMake: $(cmake --version | head -1)"

# --- Detect CUDA ---
WITH_GPU="OFF"
if [[ "$NO_GPU" -eq 0 ]]; then
    if command -v nvcc &>/dev/null; then
        WITH_GPU="ON"
        echo " CUDA: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
    elif [[ -d "${CUDA_HOME:-/usr/local/cuda}" ]]; then
        WITH_GPU="ON"
        echo " CUDA: found at ${CUDA_HOME:-/usr/local/cuda}"
    else
        echo " CUDA: not found (building CPU-only)"
    fi
else
    echo " GPU build: disabled by user"
fi

echo "============================================================"

# --- Step 1: Build C library ---
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU="$WITH_GPU"

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j"$NPROC"

cd "$SCRIPT_DIR"

# Verify the library was built
if [[ "$(uname -s)" == "Darwin" ]]; then
    LIB="$BUILD_DIR/libwarprnnt.dylib"
else
    LIB="$BUILD_DIR/libwarprnnt.so"
fi

if [[ ! -f "$LIB" ]]; then
    echo ""
    echo "ERROR: Build finished but $(basename "$LIB") was not found."
    echo "Check the cmake/make output above for errors."
    exit 1
fi
echo ""
echo "C library built: $LIB"

# --- Step 2: pip install the binding ---
export WARP_RNNT_PATH="$BUILD_DIR"
cd "$BINDING_DIR"
pip install .

echo ""
echo "============================================================"
echo " Installation complete."
echo ""
echo " Verify with:"
echo "   python -c \"from warprnnt_pytorch import RNNTLoss; print('OK')\""
echo "============================================================"
