import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
kay,
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

# Import the pipeline to load it easily
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline

# Import the dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from dataloader import LightingFineTuneDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple training script for x2rgb.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="x2rgb-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU."
        ),
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # 1. Load Pipeline and Models
    # We load the full pipeline to get all components, then extract what we need for training
    pipeline = StableDiffusionAOVDropoutPipeline.from_pretrained(args.pretrained_model_name_or_path)
    
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set UNet to train
    unet.train()

    # 2. Setup Dataset and Dataloader
    # The dataset returns: ((albedo, roughness, metallic, normal), prompt), target_image
    dataset = LightingFineTuneDataset(args.dataset_path, aov_types=['albedo', 'roughness', 'metallic', 'normal'])
    
    def collate_fn(examples):
        # examples is a list of tuples: [(((albedo, rough, metal, normal), prompt), target), ...]
        
        target_images = []
        prompts = []
        aov_images = {
            "albedo": [],
            "roughness": [],
            "metallic": [],
            "normal": []
        }

        for (aovs, prompt), target in examples:
            target_images.append(target)
            prompts.append(prompt)
            
            # aovs is a tuple (albedo, roughness, metallic, normal) corresponding to dataset.aov_types
            aov_images["albedo"].append(aovs[0])
            aov_images["roughness"].append(aovs[1])
            aov_images["metallic"].append(aovs[2])
            aov_images["normal"].append(aovs[3])

        target_images = torch.stack(target_images)
        target_images = target_images.to(memory_format=torch.contiguous_format).float()

        # Stack AOVs
        for k in aov_images:
            aov_images[k] = torch.stack(aov_images[k]).to(memory_format=torch.contiguous_format).float()

        return {
            "target_images": target_images,
            "prompts": prompts,
            "aov_images": aov_images
        }

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=1,
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
    )

    # 4. Prepare with Accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    # Move vae and text_encoder to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    # Scaling factors from pipeline_x2rgb.py
    SCALING_FACTORS = {
        "albedo": 0.17301377137652138,
        "normal": 0.17483895473058078,
        "roughness": 0.1680724853626448,
        "metallic": 0.13135013390855135,
    }

    # 5. Training Loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # A. Encode Target Image (Ground Truth)
                # Convert images to [-1, 1]
                latents = vae.encode(batch["target_images"].to(dtype=torch.float32) * 2.0 - 1.0).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # B. Sample Noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
                timesteps = timesteps.long()

                # C. Add Noise (Forward Diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # D. Encode Prompts
                inputs = tokenizer(
                    batch["prompts"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(inputs.input_ids.to(accelerator.device))[0]

                # E. Prepare AOV Conditioning
                # We need to encode each AOV and concatenate them
                aov_latents_list = []
                
                ordered_aov_keys = ["albedo", "roughness", "metallic"]
                
                for key in ordered_aov_keys:
                    aov_img = batch["aov_images"][key].to(accelerator.device, dtype=torch.float32) # TODO: ensure the order that aovs are passed matches what is expected in the VAE
                    # Normalize to [-1, 1]
                    aov_img = aov_img * 2.0 - 1.0
                    
                    # Encode
                    aov_latent = vae.encode(aov_img).latent_dist.mode()
                    
                    # Scale
                    aov_latent = aov_latent * SCALING_FACTORS[key]
                    
                    aov_latents_list.append(aov_latent)
                
                # Concatenate all AOV latents
                conditioning_latents = torch.cat(aov_latents_list, dim=1)
                
                # Concatenate noisy latents with conditioning latents
                # UNet input: [noisy_latents, albedo_latents, roughness_latents, metallic_latents]
                unet_input = torch.cat([noisy_latents, conditioning_latents], dim=1)

                # F. Predict Noise
                model_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # G. Loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 500 == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        pipeline.unet = unet
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
