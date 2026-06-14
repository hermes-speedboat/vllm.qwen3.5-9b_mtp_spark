#!/usr/bin/env bash
# test_vllm.sh — vLLM Server test (chat completion + model list)
# Usage: bash test_vllm.sh [model_name]
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8001}"
BASE_URL="http://${HOST}:${PORT}"
MODEL="${1:-cyankiwi/Qwen3.5-9B-AWQ-4bit}"

echo "=== vLLM Test ==="
echo "Server: ${BASE_URL}"
echo "Model:  ${MODEL}"
echo ""

# 1) Model list
echo "--- GET /v1/models ---"
curl -s "${BASE_URL}/v1/models" | jq . || echo "(FAIL)"

echo ""

# 2) Chat completion
echo "--- POST /v1/chat/completions ---"
curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [
      {"role": "system", "content": "Answer short and precise."},
      {"role": "user", "content": "What is 2+2? Answer in one word."}
    ],
    "max_tokens": 100,
    "temperature": 0.0
  }' | jq '.choices[0].message.content' 2>/dev/null || echo "(FAIL)"

echo ""
echo "=== Done ==="