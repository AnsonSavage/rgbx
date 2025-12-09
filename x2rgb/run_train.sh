#!/bin/bash

#SBATCH --time=5:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1     # <--- UPDATE THIS TO MATCH THE NUMBER OF GPUS YOU WANT TO USE (e.g. --gpus=2)
#SBATCH --mem-per-cpu=32768M   # memory per CPU core
#SBATCH --qos=cs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ansonsav@byu.edu

# Example script to run training for x2rgb

# Set the dataset path
DATASET_PATH="/grphome/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/aov_test_05"
MODEL_NAME="zheng95z/x-to-rgb"
OUTPUT_DIR="x2rgb-finetuned_prompt_dropout"
export HF_HUB_OFFLINE=1

# Run training
cd /grphome/grp_cs_650_rgb_x/rgbx/x2rgb/
/grphome/grp_cs_650_rgb_x/conda_env/bin/python3 train_x2rgb.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_path=$DATASET_PATH \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=32 \
  --num_train_epochs=10 \
  --learning_rate=1e-6 \
  --prob_prompt_dropout=0.1 \
  # --gradient_accumulation_steps=1 \
  # --device="cpu"