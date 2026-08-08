#!/bin/bash
# vast_onstart.sh — environment prep for the maple verify job.
# Runs once when the instance boots. Builds/test/bench happen interactively over SSH.

exec > >(tee -a /workspace/onstart.log) 2>&1
echo "===== onstart begin $(date -u) ====="

# Locate CUDA (the template image provides it; just find nvcc).
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /usr/local/cuda*/bin /opt/cuda*/bin; do
        if [ -x "$d/nvcc" ]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi
nvcc --version || { echo "FATAL: nvcc not found"; exit 1; }
nvidia-smi || true

# Build deps.
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git cmake ninja-build build-essential wget python3-pip ca-certificates 2>&1 | tail -5

# Clone fork.
cd /workspace
if [ ! -d fork ]; then
    git clone --depth 1 --branch prism https://github.com/stamsam/llama.cpp.git fork
fi
cd fork
git checkout -q 9ee03ee 2>/dev/null || true

# Pull the 4 patches from the maple repo.
PURL=https://raw.githubusercontent.com/PascalAI2024/maple-preview-windows-cuda/main/patches
mkdir -p /workspace/patches
for n in 0001-tq2_0-enable-mmvq-for-batch1-moe 0002-tq2_0-simd-vecdot 0003-enable-tq2_0-test-coverage 0004-tq2_0-specialized-dequant 0005-tq2_0-vectorized-dequant; do
    wget -q "${PURL}/${n}.patch" -O "/workspace/patches/${n}.patch" || true
done
ls -la /workspace/patches/

# Apply patches. Be tolerant: warn but don't fail if one doesn't apply.
for p in /workspace/patches/*.patch; do
    [ -s "$p" ] || { echo "SKIP empty: $(basename $p)"; continue; }
    if git apply --reverse --check "$p" 2>/dev/null; then
        echo "patch already applied: $(basename $p)"
        continue
    fi
    if git apply --check "$p" 2>/dev/null; then
        if git apply "$p" 2>/dev/null; then
            echo "applied: $(basename $p)"
        else
            echo "WARN apply failed: $(basename $p)"
        fi
    else
        echo "WARN patch does not apply: $(basename $p)"
    fi
done

echo "===== onstart done $(date -u) ====="