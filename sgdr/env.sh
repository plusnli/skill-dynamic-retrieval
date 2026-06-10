#!/bin/bash
# ─────────────────────────────────────────────────────────────
# env.sh — shared configuration file; source it once before experiments.
#
# Usage (from skill-dynamic-retrieval/sgdr/):
#   source env.sh              # default litellm mode (commercial API)
#   source env.sh vllm         # switch to vLLM mode (open-source model)
# ─────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════
# 1. WebArena service endpoints (required in both modes)
#
#    WEBARENA_HOST precedence:
#      (1) Existing WEBARENA_HOST environment variable (highest priority)
#      (2) host.local.sh in this directory (gitignored and machine-specific)
# ═══════════════════════════════════════════════════════════════

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$WEBARENA_HOST" ] && [ -f "$_ENV_DIR/host.local.sh" ]; then
    source "$_ENV_DIR/host.local.sh"
fi

if [ -z "$WEBARENA_HOST" ]; then
    echo "[env.sh] ERROR: WEBARENA_HOST is not set." >&2
    echo "[env.sh] Copy and edit the local configuration:" >&2
    echo "[env.sh]   cp $_ENV_DIR/host.local.example.sh $_ENV_DIR/host.local.sh" >&2
    echo "[env.sh]   # then set WEBARENA_HOST in host.local.sh to your WebArena server host" >&2
    echo "[env.sh] Or export WEBARENA_HOST=<host> before sourcing env.sh." >&2
    return 1 2>/dev/null || exit 1
fi
export WEBARENA_HOST

export BASE_URL="http://${WEBARENA_HOST}"

export WA_SHOPPING="$BASE_URL:8082/"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"
export WA_FULL_RESET="$BASE_URL:7565/reset"

# ═══════════════════════════════════════════════════════════════
# 2. OpenAI API key (required in both modes)
#
#    - litellm mode: autoeval calls OpenAI directly
#    - vLLM mode: WebArena's built-in llm_fuzzy_match may still call OpenAI
# ═══════════════════════════════════════════════════════════════

if [ -z "$OPENAI_API_KEY" ] && [ -f "$_ENV_DIR/../.env" ]; then
    export OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' "$_ENV_DIR/../.env" | cut -d= -f2-)"
fi

# ═══════════════════════════════════════════════════════════════
# 3. vLLM server parameters (set in both modes)
#
#    serve_vllm.sh sources env.sh to read these values, so keep them
#    available regardless of the active client backend.
# ═══════════════════════════════════════════════════════════════

export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3-4B}"
export VLLM_TP="${VLLM_TP:-1}"
export VLLM_PORT="${VLLM_PORT:-8800}"

# ═══════════════════════════════════════════════════════════════
# 4. LLM backend selection
#
#    "litellm" (default) — commercial APIs such as OpenAI
#    "vllm"              — local vLLM server for open-source models
# ═══════════════════════════════════════════════════════════════

# Supports either `source env.sh vllm` or `export LLM_BACKEND=vllm && source env.sh`.
if [ -n "$1" ]; then
    LLM_BACKEND="$1"
fi
export LLM_BACKEND="${LLM_BACKEND:-litellm}"

if [ "$LLM_BACKEND" = "vllm" ]; then
    # ── vLLM mode ──────────────────────────────────────────────
    # agent, autoeval, and induction all use the local vLLM server.
    #
    # NOTE: LLM_MODEL_NAME overrides all CLI model arguments
    #     (--model, --model_name, --eval_model). If the agent and autoeval
    #     need different models, see the documentation.
    export LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LLM_API_KEY="EMPTY"
    export LLM_MODEL_NAME="${VLLM_MODEL_NAME}"

    # Clear litellm variables in this mode to avoid ambiguity.
    unset LITELLM_BASE_URL LITELLM_API_KEY
else
    # ── litellm mode (default) ─────────────────────────────────
    # agent / induction use LiteLLM -> OpenAI.
    # autoeval calls OpenAI directly using OPENAI_API_KEY above.
    export LITELLM_BASE_URL="https://api.openai.com/v1"
    export LITELLM_API_KEY="$OPENAI_API_KEY"

    # Clear vLLM activation variables in this mode to avoid ambiguity.
    unset LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME
fi

# ═══════════════════════════════════════════════════════════════
# 5. Task IDs by website (generated from test.json)
#    Note: excludes 48 cross-site tasks that require multiple websites,
#    matching the SkillWeaver evaluation setup. Wikipedia tasks are all
#    cross-site and are therefore excluded.
# ═══════════════════════════════════════════════════════════════

# Original version (including cross-site tasks):
# TASK_IDS_SHOPPING="21-26,47-51,96-96,117-118,124-126,141-150,158-167,188-192,225-235,238-242,260-264,269-286,298-302,313-313,319-338,351-355,358-362,368-368,376-376,384-388,431-440,465-469,506-521,528-532,571-575,585-589,653-657,671-675,689-693,792-798"
# TASK_IDS_ADMIN="0-6,11-15,41-43,62-65,77-79,94-95,107-116,119-123,127-131,157-157,183-187,193-204,208-217,243-247,288-292,344-348,374-375,423-423,453-464,470-474,486-505,538-551,676-680,694-713,768-782,790-790"
# TASK_IDS_REDDIT="27-31,66-69,399-410,580-584,595-652,681-688,714-735"
# TASK_IDS_GITLAB="44-46,102-106,132-136,156-156,168-182,205-207,258-259,293-297,303-312,314-318,339-343,349-350,357-357,389-398,411-422,441-452,475-485,522-527,533-537,552-570,576-579,590-594,658-670,736-736,742-756,783-789,791-791,799-811"
# TASK_IDS_MAP="7-10,16-20,32-40,52-61,70-76,80-93,97-101,137-140,151-155,218-224,236-237,248-257,287-287,356-356,363-367,369-373,377-383,757-767"
# TASK_IDS_WIKIPEDIA="265-268,424-430,737-741"

TASK_IDS_SHOPPING="21-26,47-51,96,117-118,124-126,141-150,158-167,188-192,225-235,238-242,260-264,269-286,298-302,313,319-338,351-355,358-362,368,376,384-388,431-440,465-469,506-521,528-532,571-575,585-589,653-657,689-693,792-798"

TASK_IDS_ADMIN="0-6,11-15,41-43,62-65,77-79,94-95,107-116,119-123,127-131,157,183-187,193-204,208-217,243-247,288-292,344-348,374-375,423,453-464,470-474,486-505,538-551,676-680,694-713,768-782,790"

TASK_IDS_REDDIT="27-31,66-69,399-410,580-584,595-652,714-735"

TASK_IDS_GITLAB="44-46,102-106,132-136,156,168-182,205-207,258-259,293-297,303-312,314-318,339-343,349-350,357,389-398,411-422,441-452,475-485,522-527,533-537,567-570,576-579,590-594,658-670,736,742-756,783-789,799-811"

TASK_IDS_MAP="7-10,16-20,32-40,52-61,70-76,80-93,98-101,137-140,151-155,218-224,236-237,248-257,287,356,363-367,369-373,377-383,757-758,761-767"

# ═══════════════════════════════════════════════════════════════
# 6. Playwright/Chromium temporary directory
#    Override TMPDIR before sourcing env.sh if needed.
# ═══════════════════════════════════════════════════════════════

export TMPDIR="${TMPDIR:-/tmp/skill-dynamic-retrieval-playwright}"
mkdir -p "$TMPDIR"

# ═══════════════════════════════════════════════════════════════
# 7. Configuration summary
# ═══════════════════════════════════════════════════════════════

echo "────────────────────────────────────"
echo "[env.sh] WEBARENA_HOST = ${WEBARENA_HOST}"
echo "[env.sh] TMPDIR = ${TMPDIR}"
echo "[env.sh] LLM_BACKEND = ${LLM_BACKEND}"
if [ "$LLM_BACKEND" = "vllm" ]; then
    echo "[env.sh] LLM_MODEL_NAME = ${LLM_MODEL_NAME}"
    echo "[env.sh] LLM_BASE_URL   = ${LLM_BASE_URL}"
    echo "[env.sh] Make sure the vLLM server is running: bash serve_vllm.sh"
else
    echo "[env.sh] LITELLM_BASE_URL = ${LITELLM_BASE_URL}"
    echo "[env.sh] CLI --model / --eval_model arguments are effective"
fi
echo "────────────────────────────────────"
