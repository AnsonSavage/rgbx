#!/bin/bash

# Example script to run training for x2rgb

# Set the dataset path
DATASET_PATH="/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/aov_test_05"
MODEL_NAME="zheng95z/x-to-rgb"
OUTPUT_DIR="x2rgb-finetuned"
export HF_HUB_OFFLINE=1

# Run training
python train_x2rgb.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_path=$DATASET_PATH \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=32 \
  --num_train_epochs=10 \
  --learning_rate=1e-5 \
  # --gradient_accumulation_steps=1 \
  # --device="cpu"