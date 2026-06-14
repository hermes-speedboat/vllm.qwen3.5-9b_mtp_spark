#!/usr/bin/env bash
# vllm-server.sh — vLLM Server for Qwen3.5-9B (AWQ 4-bit, MTP via qwen3_next_mtp)
# DGX Spark (GB10, 128 GB Unified Memory, Grace Blackwell ARM64)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_REPO="${MODEL_REPO:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"
MODEL_NAME="${MODEL_REPO##*/}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# ----------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ----------------------------------------------------------------------
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
LANGUAGE_ONLY="${LANGUAGE_ONLY:-false}"
SP_TOK_DEFAULT=2  # MTP default
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-${SP_TOK_DEFAULT}}

# ----------------------------------------------------------------------
# Check if model exists
# ----------------------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model not found: ${MODEL_PATH}"
  echo "Run download_model.sh first:"
  echo "  bash ${SCRIPT_DIR}/download_model.sh"
  exit 1
fi

# ----------------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------------
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo ""

# ----------------------------------------------------------------------
# Start server
# ----------------------------------------------------------------------
echo "=== Starting vLLM Server ==="
echo "  Model: ${MODEL_PATH}"
echo "  Port:  ${HOST}:${PORT}"
echo "  Context: ${MAX_MODEL_LEN}"
echo "  GPU Mem: ${GPU_MEM_UTIL}"
echo "  Quant: AWQ 4-bit"
echo "  MTP:  ${NUM_SPEC_TOKENS} draft tokens (method: qwen3_next_mtp)"
echo "  Vision: $([ "${LANGUAGE_ONLY}" = "true" ] && echo "OFF (text only)" || echo "ON")"
echo ""

ARGS=(
  "${MODEL_PATH}"
  "--port" "${PORT}"
  "--host" "${HOST}"
  "--max-model-len" "${MAX_MODEL_LEN}"
  "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
  "--trust-remote-code"
  "--quantization" "awq"
  "--dtype" "float16"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
  "--enforce-eager"
  "--served-model-name" "cyankiwi/Qwen3.5-9B-AWQ-4bit"
)

# Vision encoder (default: on)
if [ "${LANGUAGE_ONLY}" = "true" ]; then
  ARGS+=("--language-model-only")
fi

# MTP (Multi Token Prediction) — Qwen3.5 uses qwen3_next_mtp method
if [ "${NUM_SPEC_TOKENS}" -gt 0 ]; then
  ARGS+=("--speculative-config" '{"method":"qwen3_next_mtp","num_speculative_tokens":'"${NUM_SPEC_TOKENS}"'}')
fi

echo "> vllm serve ${ARGS[*]}"
vllm serve "${ARGS[@]}"

echo ""
echo "Server stopped."