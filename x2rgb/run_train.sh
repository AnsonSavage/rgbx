#!/bin/bash

#SBATCH --time=18:00:00   # walltime
#SBATCH --ntasks=16        # number of tasks (processes)
#SBATCH --nodes=1         # number of nodes
#SBATCH --gpus=8          # <-- update this to match the number of GPUs you want (e.g. --gpus=2)
#SBATCH --mem-per-cpu=32768M   # memory per CPU core
#SBATCH --qos=standby
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ansonsav@byu.edu

# Example script to run training for x2rgb using Hugging Face Accelerate

# Dataset selection: choose which dataset type to use

DATASET_TYPE="mixed"  # Options: "hdri", "discrete", "interiorverse", "mixed"
MODEL_NAME="zheng95z/x-to-rgb"
OUTPUT_DIR="./checkpoints/x2rgb-finetuned_${DATASET_TYPE}_mixed_with_velocity_random_material"
export HF_HUB_OFFLINE=1
export PYTHONPATH=/home/ansonsav/.local/lib/python3.10/site-packages:$PYTHONPATH

# Move to training directory
cd /grphome/grp_cs_650_rgb_x/rgbx/x2rgb/

# Use accelerate to launch multi-GPU training. accelerate will detect available GPUs.
# If you want to force a specific number of processes, add: --num_processes <N>

# Recommended: configure accelerate once interactively on the machine with `accelerate config`
# Then run via accelerate launch so it handles distributed setup automatically.

# GPU Selection
# We rely on SLURM to set CUDA_VISIBLE_DEVICES.
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
else
    # Fallback: count commas in CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

/grphome/grp_cs_650_rgb_x/conda_env/bin/accelerate launch --num_processes=$NUM_GPUS train_x2rgb.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --dataset_type="$DATASET_TYPE" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=32 \
  --num_train_epochs=40 \
  --learning_rate=1e-6 \
  --prob_prompt_dropout=0.1