#!/usr/bin/env bash

# Simple wrapper where you edit (paste) paths directly into this file.
# Edit the USER CONFIG section below, then run this script to invoke the
# `text_only_x2rgb.py` script with those values.

set -euo pipefail

# ----------------------- USER CONFIG -----------------------
# Paste your paths and options here. Leave empty strings for unused AOVs.
ALBEDO="/path/to/albedo.exr"
NORMAL="/path/to/normal.exr"
ROUGHNESS="/path/to/roughness.exr"
METALLIC="/path/to/metal.exr"
IRRADIANCE="/path/to/irr.exr"

# Required: prompt and output
PROMPT="A sunny studio render"
OUTPUT="./out.png"

# Optional runtime options
DEVICE="cuda"            # e.g. cuda or cpu
SEED=42
INFERENCE_STEP=100
UNET_CHECKPOINT=""       # optional path to UNet checkpoint directory
CACHE_DIR="./model_cache"

# If you want to use a specific python, set the env var PYTHON before running,
# or edit the line below.
PYTHON=${PYTHON:-python}
# --------------------- END USER CONFIG ---------------------

if [[ -z "$PROMPT" || -z "$OUTPUT" ]]; then
  echo "Error: set PROMPT and OUTPUT in the USER CONFIG section of this file."
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARGS=()
[[ -n "$ALBEDO" ]] && ARGS+=(--albedo "$ALBEDO")
[[ -n "$NORMAL" ]] && ARGS+=(--normal "$NORMAL")
[[ -n "$ROUGHNESS" ]] && ARGS+=(--roughness "$ROUGHNESS")
[[ -n "$METALLIC" ]] && ARGS+=(--metallic "$METALLIC")
[[ -n "$IRRADIANCE" ]] && ARGS+=(--irradiance "$IRRADIANCE")
ARGS+=(--prompt "$PROMPT")
ARGS+=(--output_path "$OUTPUT")
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")
[[ -n "$INFERENCE_STEP" ]] && ARGS+=(--inference_step "$INFERENCE_STEP")
[[ -n "$UNET_CHECKPOINT" ]] && ARGS+=(--unet_checkpoint "$UNET_CHECKPOINT")
[[ -n "$CACHE_DIR" ]] && ARGS+=(--cache_dir "$CACHE_DIR")

echo "About to run: $PYTHON text_only_x2rgb.py ${ARGS[*]}"
echo "If this looks good, running now..."

exec "$PYTHON" text_only_x2rgb.py "${ARGS[@]}"
