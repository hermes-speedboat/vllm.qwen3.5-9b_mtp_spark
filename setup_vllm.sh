#!/usr/bin/env bash
# setup_vllm.sh — vLLM + HuggingFace Setup on DGX Spark (GB10, Grace Blackwell ARM64)
# Run as normal user (e.g. vllm2). No root required.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL_DIR="${SCRIPT_DIR}/models"

echo "=== vLLM Setup for DGX Spark (Qwen3.5-9B) ==="
echo "Target: ${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 1) Install uv (if not present)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    # shellcheck disable=SC1091
    source "$HOME/.bashrc" 2>/dev/null || true
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2) Create Python venv with uv
# ---------------------------------------------------------------------------
echo "[2/5] Creating Python venv with uv ..."
uv venv "${VENV_DIR}" --python 3.12

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# 3) Install huggingface-hub
# ---------------------------------------------------------------------------
echo "[3/5] Installing huggingface-hub ..."
uv pip install huggingface-hub

# ---------------------------------------------------------------------------
# 4) Install vLLM (ARM64/CUDA via --torch-backend=auto)
#    vLLM >= 0.23.0 recommended for Qwen3.5 MTP (qwen3_next_mtp)
# ---------------------------------------------------------------------------
echo "[4/5] Installing vLLM (this may take a while) ..."
uv pip install vllm --torch-backend=auto

# ---------------------------------------------------------------------------
# 5) Create model directory
# ---------------------------------------------------------------------------
echo "[5/5] Creating model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete === "
echo ""
echo "  Venv:   source ${VENV_DIR}/bin/activate"
echo "  Models: ${MODEL_DIR}/"
echo ""
echo "Next: download the model:"
echo "  bash ${SCRIPT_DIR}/download_model.sh"
echo ""
echo "Then start the server:"
echo "  bash ${SCRIPT_DIR}/vllm-server.sh"