#!/usr/bin/env bash
# Unix setup helper. Windows PowerShell/MSYS2 is the primary target.

set -euo pipefail

CUDA_ARG="${1:-auto}"

if [ ! -d "venv" ]; then
    echo "[setup] Creating virtual environment"
    python3 -m venv venv
else
    echo "[setup] venv already exists"
fi

source venv/bin/activate
pip install --quiet --upgrade pip

if [ "$CUDA_ARG" = "auto" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
        MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        if [ "$MAJOR" -ge 12 ]; then
            CUDA_ARG="cu121"
        else
            CUDA_ARG="cu118"
        fi
        echo "[setup] Detected CUDA ${CUDA_VERSION}; using ${CUDA_ARG}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARG="cu121"
        echo "[setup] nvidia-smi found; using ${CUDA_ARG}"
    else
        CUDA_ARG="cpu"
        echo "[setup] No CUDA detected; using CPU-only PyTorch"
    fi
fi

if [ "$CUDA_ARG" = "cpu" ]; then
    pip install torch torchvision
else
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_ARG}"
fi

grep -v '^torch' requirements.txt | grep -v '^#' | grep -v '^$' | pip install --quiet -r /dev/stdin

if command -v gcc >/dev/null 2>&1; then
    make
    echo "[setup] native kernel built"
else
    echo "[setup] WARNING: gcc not found. Run make after installing gcc."
fi

echo ""
echo "Setup complete."
echo "See README.md for the full pipeline commands."
