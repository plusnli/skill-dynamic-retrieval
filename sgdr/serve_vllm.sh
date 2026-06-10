#!/bin/bash
# Start a vLLM OpenAI-compatible server in a separate tmux pane.
#
# Usage:
#   bash serve_vllm.sh                            # use defaults from env.sh
#   bash serve_vllm.sh Qwen/Qwen3-4B 1            # override model and TP
#   VLLM_PORT=8801 bash serve_vllm.sh             # override port
#   CUDA_VISIBLE_DEVICES=0,1 bash serve_vllm.sh   # choose GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Read default VLLM_MODEL_NAME / VLLM_TP / VLLM_PORT values from env.sh.
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.sh"
fi

MODEL=${1:-${VLLM_MODEL_NAME:-"Qwen/Qwen3-4B"}}
TP=${2:-${VLLM_TP:-1}}
PORT=${VLLM_PORT:-8800}
GPUS=${CUDA_VISIBLE_DEVICES:-0}

export CUDA_VISIBLE_DEVICES="$GPUS"

echo ""
echo "═══════════════════════════════════════"
echo "  vLLM Server"
echo "  Model : $MODEL"
echo "  TP    : $TP"
echo "  Port  : $PORT"
echo "  GPUs  : $GPUS"
echo "═══════════════════════════════════════"
echo ""
echo "Run this in the experiment shell:"
echo "  source env.sh vllm"
echo ""

MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-81920}
HF_OVERRIDES=${VLLM_HF_OVERRIDES:-'{"max_position_embeddings":81920,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":40960}}'}

vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --hf-overrides "$HF_OVERRIDES"
