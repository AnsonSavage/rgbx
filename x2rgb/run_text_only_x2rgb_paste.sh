#!/usr/bin/env bash

# Simple wrapper where you edit (paste) paths directly into this file.
# Edit the USER CONFIG section below, then run this script to invoke the
# `text_only_x2rgb.py` script with those values.

set -euo pipefail

# ----------------------- USER CONFIG -----------------------
# Paste your paths and options here. Leave empty strings for unused AOVs.
# ALBEDO="/home/ansonsav/groups/grp_cs_650_rgb_x/rgbx/x2rgb/example/kitchen-albedo-512.png"
# NORMAL="/home/ansonsav/groups/grp_cs_650_rgb_x/rgbx/x2rgb/example/kitchen-normal-512.png"
# ROUGHNESS="/home/ansonsav/groups/grp_cs_650_rgb_x/rgbx/x2rgb/example/kitchen-roughness-512.png"
# METALLIC="/home/ansonsav/groups/grp_cs_650_rgb_x/rgbx/x2rgb/example/kitchen-metallic-512.png"
# IRRADIANCE=""
ALBEDO="/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03/42ecdcd1-a78f-49e5-8985-3098948e82b1/aovs/42ecdcd1-a78f-49e5-8985-3098948e82b1_cam_26_scatter_47_objsel_68_aovs/albedo0052.png"
NORMAL="/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03/42ecdcd1-a78f-49e5-8985-3098948e82b1/aovs/42ecdcd1-a78f-49e5-8985-3098948e82b1_cam_26_scatter_47_objsel_68_aovs/normal0052.png"
ROUGHNESS="/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03/42ecdcd1-a78f-49e5-8985-3098948e82b1/aovs/42ecdcd1-a78f-49e5-8985-3098948e82b1_cam_26_scatter_47_objsel_68_aovs/roughness0052.png"
METALLIC="/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03/42ecdcd1-a78f-49e5-8985-3098948e82b1/aovs/42ecdcd1-a78f-49e5-8985-3098948e82b1_cam_26_scatter_47_objsel_68_aovs/metallic0052.png"
IRRADIANCE=""

# Required: prompt (or use PROMPTS_FILE for batch)
# If you want a single prompt, set PROMPT. For batch use PROMPTS_FILE.
PROMPT=""
# Optional: path to a text file with one prompt per line to run in batch
PROMPTS_FILE="./prompts.txt"
# Directory where outputs will be written (always used)
OUTPUT_DIR="./outputs_from_dataset"
# Optional run label that will be included in the filename when provided.
# Example: RUN_NAME="experimentA" -> outputs/out_experimentA_20251209_123000.png
RUN_NAME="no_fine_tune"

# Optional runtime options
DEVICE="cuda"            # e.g. cuda or cpu
SEED=42
INFERENCE_STEP=100
UNET_CHECKPOINT="" #"/home/ansonsav/groups/grp_cs_650_rgb_x/rgbx/x2rgb/x2rgb-finetuned_discrete_with_velocity/checkpoint-350/"       # optional path to UNet checkpoint directory
CACHE_DIR="./model_cache"

# If you want to use a specific python, set the env var PYTHON before running,
# or edit the line below.
PYTHON=${PYTHON:-/grphome/grp_cs_650_rgb_x/conda_env/bin/python3}
# --------------------- END USER CONFIG ---------------------

# outputs are always written into $OUTPUT_DIR (created below)

if [[ -n "$PROMPTS_FILE" ]]; then
  if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "Error: PROMPTS_FILE set but file does not exist: $PROMPTS_FILE"
    exit 2
  fi
fi
# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARGS=()
[[ -n "$ALBEDO" ]] && ARGS+=(--albedo "$ALBEDO")
[[ -n "$NORMAL" ]] && ARGS+=(--normal "$NORMAL")
[[ -n "$ROUGHNESS" ]] && ARGS+=(--roughness "$ROUGHNESS")
[[ -n "$METALLIC" ]] && ARGS+=(--metallic "$METALLIC")
[[ -n "$IRRADIANCE" ]] && ARGS+=(--irradiance "$IRRADIANCE")
if [[ -n "$PROMPTS_FILE" ]]; then
  ARGS+=(--prompts_file "$PROMPTS_FILE")
fi
ARGS+=(--output_dir "$OUTPUT_DIR")
if [[ -n "$PROMPT" ]]; then
  ARGS+=(--prompt "$PROMPT")
fi
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")
[[ -n "$INFERENCE_STEP" ]] && ARGS+=(--inference_step "$INFERENCE_STEP")
[[ -n "$UNET_CHECKPOINT" ]] && ARGS+=(--unet_checkpoint "$UNET_CHECKPOINT")
[[ -n "$CACHE_DIR" ]] && ARGS+=(--cache_dir "$CACHE_DIR")

echo "About to run: $PYTHON text_only_x2rgb.py ${ARGS[*]}"
echo "If this looks good, running now..."

export HF_HUB_OFFLINE=1
exec "$PYTHON" text_only_x2rgb.py "${ARGS[@]}"
