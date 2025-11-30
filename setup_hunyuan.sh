#!/usr/bin/env bash
# shellcheck disable=SC2312
#
# Bootstrap HunyuanVideo-1.5 for Image-to-Video (720p) inference.
#
# Usage:
#   chmod +x setup_hunyuan.sh
#   ./setup_hunyuan.sh
#
# Configuration (override via env vars before running):
#   HY_TARGET_DIR   - Folder where the repo will live (default: $PWD/HunyuanVideo-1.5)
#   HY_VENV_DIR     - Python venv location           (default: $HY_TARGET_DIR/.venv)
#   HY_MODEL_DIR    - Checkpoint directory           (default: $HY_TARGET_DIR/ckpts)
#   HY_MAIN_VARIANT - Transformer variant to fetch   (default: 720p_i2v)
#   HY_SIGLIP_TOKEN - Hugging Face token to access black-forest-labs/FLUX.1-Redux-dev
#   HY_QWEN_REPO    - LLM repo used for prompt rewrite (default: Qwen/Qwen2.5-VL-7B-Instruct)
#   HY_BYT5_REPO    - ByT5 repo for Glyph encoder      (default: google/byt5-small)
#   HY_GLYPH_MODEL  - Glyph modelscope ID              (default: AI-ModelScope/Glyph-SDXL-v2)
#   HY_INSTALL_SAGEATTN - Install SageAttention (default: true)
#   HY_SAGEATTN_DIR  - Path to install SageAttention   (default: $TARGET_DIR/external/SageAttention)
#   HY_SAGEATTN_REPO_URL - SageAttention repo URL      (default: https://github.com/cooper1637/SageAttention.git)
#   HY_SAGEATTN_ENV_OPTS - Extra env exports before build (e.g. "EXT_PARALLEL=4 MAX_JOBS=32")
#
# Requirements:
#   - Linux + bash
#   - git, python>=3.10, pip
#   - Sufficient disk space (~30 GB) and a valid Hugging Face token with access
#     to FLUX.1-Redux-dev exported via HY_SIGLIP_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN.

# Re-run with bash automatically when sourced via sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

log() {
  printf '[setup_hunyuan] %s\n' "$*" >&2
}

die() {
  printf '[setup_hunyuan][error] %s\n' "$*" >&2
  exit 1
}

usage() {
  sed -n '1,40p' "$0"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET_DIR=${HY_TARGET_DIR:-"$PWD/HunyuanVideo-1.5"}
VENV_DIR=${HY_VENV_DIR:-"$TARGET_DIR/.venv"}
MODEL_DIR=${HY_MODEL_DIR:-"$TARGET_DIR/ckpts"}
MAIN_VARIANT=${HY_MAIN_VARIANT:-"480p_i2v_distilled"}
QWEN_REPO=${HY_QWEN_REPO:-"Qwen/Qwen2.5-VL-7B-Instruct"}
BYT5_REPO=${HY_BYT5_REPO:-"google/byt5-small"}
GLYPH_MODEL=${HY_GLYPH_MODEL:-"AI-ModelScope/Glyph-SDXL-v2"}
SIGLIP_REPO="black-forest-labs/FLUX.1-Redux-dev"
SIGLIP_TOKEN=${HY_SIGLIP_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-""}}}
INSTALL_SAGEATTN=${HY_INSTALL_SAGEATTN:-"true"}
SAGEATTN_DIR=${HY_SAGEATTN_DIR:-"$TARGET_DIR/external/SageAttention"}
SAGEATTN_REPO_URL=${HY_SAGEATTN_REPO_URL:-"https://github.com/cooper1637/SageAttention.git"}
SAGEATTN_ENV_OPTS=${HY_SAGEATTN_ENV_OPTS:-""}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

log "Validating prerequisites"
for cmd in git python3; do
  need_cmd "$cmd"
done

python3 - <<'PY' || die "Python >= 3.10 is required"
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    raise SystemExit(1)
PY

mkdir -p "$(dirname "$TARGET_DIR")"
if [[ -d "$TARGET_DIR/.git" ]]; then
  log "Repository already present at $TARGET_DIR, pulling latest changes"
  git -C "$TARGET_DIR" pull --ff-only
else
  log "Cloning Tencent-Hunyuan/HunyuanVideo-1.5"
  git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git "$TARGET_DIR"
fi

log "Creating/updating Python virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$TARGET_DIR/requirements.txt"
pip install -i https://mirrors.tencent.com/pypi/simple/ --upgrade tencentcloud-sdk-python
pip install "huggingface_hub[cli]" modelscope
if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-}" =~ ^(1|true|TRUE)$ ]]; then
  pip install hf_transfer
fi
need_cmd modelscope

mkdir -p "$MODEL_DIR"

hf_snapshot() {
  local repo="$1"
  local dest="$2"
  shift 2 || true
  local patterns=("$@")
  python3 - "$repo" "$dest" "${patterns[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest, *patterns = sys.argv[1:]
kwargs = dict(
    repo_id=repo,
    local_dir=dest,
    local_dir_use_symlinks=False,
    resume_download=True,
)
if patterns:
    kwargs["allow_patterns"] = patterns

snapshot_download(**kwargs)
PY
}

download_variant_ckpts() {
  local repo="tencent/HunyuanVideo-1.5"
  local -a artifacts=(
    "config.json"
    "scheduler/scheduler_config.json"
    "vae/config.json"
    "vae/diffusion_pytorch_model.safetensors"
    "transformer/${MAIN_VARIANT}/config.json"
    "transformer/${MAIN_VARIANT}/diffusion_pytorch_model.safetensors"
    "upsampler/720p_sr_distilled/config.json"
    "upsampler/720p_sr_distilled/diffusion_pytorch_model.safetensors"
  )

  log "Downloading core HunyuanVideo checkpoints (${MAIN_VARIANT})"
  hf_snapshot "$repo" "$MODEL_DIR" "${artifacts[@]}"
}

download_text_encoders() {
  log "Downloading Glyph tokenizer weights via ModelScope ($GLYPH_MODEL)"
  modelscope download \
    --model "$GLYPH_MODEL" \
    --local_dir "$MODEL_DIR/text_encoder/Glyph-SDXL-v2"

  log "Downloading Qwen text encoder ($QWEN_REPO)"
  hf_snapshot "$QWEN_REPO" "$MODEL_DIR/text_encoder/llm"

  log "Downloading ByT5 encoder ($BYT5_REPO)"
  hf_snapshot "$BYT5_REPO" "$MODEL_DIR/text_encoder/byt5-small"
}

download_vision_encoder() {
  local token="${SIGLIP_TOKEN}"
  [[ -n "$token" ]] || die "HY_SIGLIP_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN must be set for $SIGLIP_REPO"

  log "Downloading Siglip vision encoder ($SIGLIP_REPO)"
  HUGGINGFACE_HUB_TOKEN="$token" hf_snapshot "$SIGLIP_REPO" "$MODEL_DIR/vision_encoder/siglip"
}

install_sageattention() {
  local flag
  flag=$(printf '%s' "$INSTALL_SAGEATTN" | tr '[:upper:]' '[:lower:]')
  case "$flag" in
    ""|true|1|yes|on) ;;
    *) log "Skipping SageAttention installation (HY_INSTALL_SAGEATTN=$INSTALL_SAGEATTN)"; return ;;
  esac

  log "Installing SageAttention into $SAGEATTN_DIR"
  mkdir -p "$(dirname "$SAGEATTN_DIR")"
  if [[ -d "$SAGEATTN_DIR/.git" ]]; then
    git -C "$SAGEATTN_DIR" pull --ff-only
  else
    git clone "$SAGEATTN_REPO_URL" "$SAGEATTN_DIR"
  fi

  (
    cd "$SAGEATTN_DIR"
    if [[ -n "$SAGEATTN_ENV_OPTS" ]]; then
      log "Applying SageAttention build env options: $SAGEATTN_ENV_OPTS"
      eval "$SAGEATTN_ENV_OPTS"
    fi
    python3 setup.py install
  )
}

download_text_encoders
download_variant_ckpts
download_vision_encoder
install_sageattention

log "Setup complete!"
cat <<EOF

Next steps:
  1. source "$VENV_DIR/bin/activate"
  2. export MODEL_PATH="$MODEL_DIR"
  3. Use generate.py for i2v 720p inference, e.g.:
      PROMPT='A beautiful scenery of mountains during sunrise, with vibrant colors and a serene atmosphere.'

      IMAGE_PATH="/workspace/HunyuanVideo-1.5/input/4.png"

      python generate.py \
        --prompt "$PROMPT" \
        --image_path $IMAGE_PATH \
        --resolution 480p \
        --model_path ckpts \
        --num_inference_steps 20 \
        --cfg_distilled true \
        --seed 76345345 \
        --sr false 

Optional: install FlashAttention / SageAttention following the upstream README.
EOF
