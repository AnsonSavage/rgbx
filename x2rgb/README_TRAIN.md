# Training x2rgb

This directory contains the training script `train_x2rgb.py` for fine-tuning the x2rgb model.

## Usage

You can run the training script using `accelerate launch`.

```bash
accelerate launch train_x2rgb.py \
  --pretrained_model_name_or_path="zheng95z/x-to-rgb" \
  --dataset_path="/path/to/your/dataset" \
  --output_dir="output_directory" \
  --train_batch_size=4 \
  --num_train_epochs=100
```

## Arguments

- `--pretrained_model_name_or_path`: The path or Hugging Face ID of the pretrained model.
- `--dataset_path`: The root directory of the dataset (containing `aovs` folder and csv file).
- `--output_dir`: Directory to save the model checkpoints.
- `--train_batch_size`: Batch size per device.
- `--learning_rate`: Learning rate.
- `--mixed_precision`: "fp16", "bf16", or "no".

## Dataset

The script expects a dataset compatible with `LightingFineTuneDataset` defined in `../../dataloader.py`.
It should have:
- A CSV file with text descriptions.
- An `aovs` folder with subfolders for each scene/camera containing `albedo.png`, `roughness.png`, `metallic.png`.
- Target images in subfolders.

## Notes

- The script freezes the VAE and Text Encoder and only trains the UNet.
- It uses the specific scaling factors for AOVs defined in `pipeline_x2rgb.py`.
