# vLLM DGX Spark — Qwen3.5-9B (AWQ 4-bit, Dense, MTP)

Optimized vLLM inference setup for **DGX Spark (GB10, Grace Blackwell ARM64)** with 128 GB unified memory.

## Quickstart

```bash
cd /srv/vllm2
git clone https://github.com/hermes-speedboat/vllm.qwen3.5-9b_mtp_spark.git .
bash setup_vllm.sh        # uv + venv + vllm + huggingface-hub
bash download_model.sh     # ~5 GB model via snapshot_download()
bash vllm-server.sh        # start server on port 8001
```

## Performance

| Metric | Value |
|--------|-------|
| Generation throughput | ~90-140 tok/s (estimated) |
| MTP acceptance (1st draft) | ~75-85% |
| MTP acceptance (2nd draft) | ~55-70% |
| Mean acceptance length | ~2.2-2.5 |
| GPU KV cache usage | ~2-3 GB at 256k context |

## Architecture

- **Model:** Qwen3.5-9B (dense, hybrid Gated DeltaNet + Gated Attention, 32 layers)
- **Quantization:** AWQ 4-bit via compressed-tensors, ~8.5 GB disk, ~5 GB loaded
- **Format:** PyTorch safetensors (via `snapshot_download` from `huggingface_hub`)
- **Inference:** vLLM 0.23.0
- **Spec Decode:** MTP (Multi Token Prediction via `qwen3_next_mtp`), 2 draft tokens
- **Hardware:** DGX Spark (GB10), 128 GB unified memory, ~1.5 TB/s bandwidth
- **Active params/token:** 9B × 0.5 bytes = ~4.5 GB per forward pass

## Features

- **Reasoning:** Extracted to separate `reasoning` field via `--reasoning-parser qwen3` — clean content
- **Tool calling:** Native OpenAI `tool_calls` via `--tool-call-parser qwen3_coder`
- **Vision:** On by default (Qwen3.5 has native vision encoder). Set `LANGUAGE_ONLY=true` to disable and save ~2 GB
- **GPU memory:** `GPU_MEM_UTIL=0.3` (~36 GB reserved, ~7 GB actual usage) — leaves headroom alongside the 35B model and Ollama
- **MTP:** Speculative decoding with 2 draft tokens via `qwen3_next_mtp` method
- **No CUDA graphs:** `--enforce-eager` for instant first-request response
- **Model name:** Served as `cyankiwi/Qwen3.5-9B-AWQ-4bit` (not filesystem path)
- **Context:** 262144 tokens (256k)

## Files

| File | Purpose |
|------|---------|
| `setup_vllm.sh` | Install uv, venv, vllm, huggingface-hub |
| `download_model.sh` | Download model via snapshot_download() |
| `vllm-server.sh` | Start vLLM with all optimized flags |
| `vllm.service` | Systemd user service file |
| `test_vllm.sh` | API functional test |

## Systemd Autostart

**Prerequisites (headless/SSH-only):**

```bash
sudo apt install dbus dbus-user-session
sudo loginctl enable-linger $USER
echo 'export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"' >> ~/.bashrc
source ~/.bashrc
```

**Install:**

```bash
mkdir -p ~/.config/systemd/user/
cp vllm.service ~/.config/systemd/user/
chmod +x /srv/vllm2/vllm-server.sh
systemctl --user daemon-reload
systemctl --user enable --now vllm
```

**Troubleshooting:** If the service fails with `status=203/EXEC`, the script isn't executable (`chmod +x`). Fix, then `systemctl --user restart vllm`.

## Hermes Configuration

```bash
hermes config set model.base_url http://192.168.66.113:8001/v1
hermes config set model.provider custom
hermes config set model.default cyankiwi/Qwen3.5-9B-AWQ-4bit
```

## Known Issues

- **GGUF not supported:** The Qwen3.5 architecture is not supported by the transformers GGUF parser. Use PyTorch/safetensors format.
- **AWQ needs float16:** `--dtype float16` is required. `bfloat16` (the default with `--dtype auto`) may produce an error.
- **MTP method:** Qwen3.5 uses `qwen3_next_mtp` (not `mtp` as used by Qwen3.6-35B-A3B).
- **Port 8000:** Already in use by the 35B model. This repo uses port 8001.