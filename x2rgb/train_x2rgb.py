import argparse
import math
import os
import sys
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

# Import the pipeline to load it easily
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline

# Import the dataset
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from dataloader import LightingFineTuneDataset

from typing import Optional

logger = get_logger(__name__)

def create_dataset(dataset_type: str, dataset_path: Optional[str] = None):
    """Factory function to create the appropriate dataset based on type.
    
    Args:
        dataset_type: Either "hdri" or "discrete"
        dataset_path: Optional override path. If not provided, uses default paths based on dataset_type.
    
    Returns:
        Dataset instance of the appropriate type.
    """
    # Define default paths
    DEFAULT_PATHS = {
        "hdri": "/grphome/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/aov_test_05",
        "discrete": "/grphome/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03",
    }
    
    if dataset_type not in DEFAULT_PATHS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'hdri' or 'discrete'.")
    
    # Use provided path or fall back to default
    path = dataset_path if dataset_path is not None else DEFAULT_PATHS[dataset_type]
    
    if dataset_type == "hdri":
        return LightingFineTuneDataset(path)
    elif dataset_type == "discrete":
        from dataloader import LightingFineTuneDatasetDiscrete
        return LightingFineTuneDatasetDiscrete(path)

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
        "--dataset_type",
        type=str,
        default="hdri",
        choices=["hdri", "discrete"],
        help="Type of dataset to use: 'hdri' for aov_test_05 or 'discrete' for product_content_lock_test_03.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset root directory. If not provided, will be inferred from dataset_type.",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) to train on.",
    )
    parser.add_argument(
        "--prob_prompt_dropout",
        type=float,
        default=0.3,
        help="Probability of dropping the prompt to enable Classifier-Free Guidance.",
    )
    parser.add_argument(
        "--prob_intrinsic_dropout",
        type=float,
        default=0.3,
        help="Probability of dropping the intrinsic AOVs to enable Classifier-Free Guidance.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    
    args = parser.parse_args()
    return args

def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint-* directory in output_dir (diffusers format)."""
    assert os.path.isdir(output_dir), f"{output_dir} is not a valid directory."

    candidates = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not candidates:
        return None

    def key_fn(name):
        try:
            return int(name.split("checkpoint-")[-1])
        except Exception:
            return 0

    candidates.sort(key=key_fn, reverse=True)
    return os.path.join(output_dir, candidates[0])

def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Extract args into local variables for easier navigation and usage
    arg_pretrained_model_name_or_path = args.pretrained_model_name_or_path
    arg_dataset_type = args.dataset_type
    arg_dataset_path = args.dataset_path  # Can be None, factory function will use defaults
    arg_output_dir = args.output_dir
    arg_seed = args.seed
    arg_train_batch_size = args.train_batch_size
    arg_num_train_epochs = args.num_train_epochs
    arg_max_train_steps = args.max_train_steps
    arg_learning_rate = args.learning_rate
    arg_lr_scheduler_name = args.lr_scheduler
    arg_lr_warmup_steps = args.lr_warmup_steps
    arg_gradient_accumulation_steps = args.gradient_accumulation_steps
    arg_prob_prompt_dropout = args.prob_prompt_dropout
    arg_prob_intrinsic_dropout = args.prob_intrinsic_dropout

    # Ensure output dir exists
    if accelerator.is_main_process:
        os.makedirs(arg_output_dir, exist_ok=True)

    if arg_seed is not None:
        set_seed(arg_seed)

    # 1. Load Pipeline and Models
    # We load the full pipeline to get all components, then extract what we need for training
    pipeline = StableDiffusionAOVDropoutPipeline.from_pretrained(arg_pretrained_model_name_or_path, cache_dir = "./model_cache")
    
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)


    # If there are prior training runs, load the UNet weights from the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(arg_output_dir)
    if latest_checkpoint is not None:
        unet_dir = os.path.join(latest_checkpoint, "unet")
        logger.info(f"Found existing checkpoint at {latest_checkpoint}. Loading UNet from {unet_dir}.")
        unet = unet.__class__.from_pretrained(unet_dir)
        logger.info("UNet weights loaded from checkpoint.")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set UNet to train
    unet.train()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=torch.float32) # In SDXL, they keep the vae in float32 for better stability
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # 2. Setup Dataset and Dataloader
    # The dataset returns: ((albedo, normal, roughness, metallic), prompt), target_image
    dataset = create_dataset(arg_dataset_type, arg_dataset_path)
    def collate_fn(dataset_samples):
        # dataset_samples is a list of tuples: [(((albedo, normal, roughness, metal), prompt), target), ...]
        
        target_images = []
        prompts = []
        aov_images = {
            "albedo": [],
            "normal": [],
            "roughness": [],
            "metallic": []
        }

        for (aovs, prompt), target in dataset_samples:
            target_images.append(target)
            prompts.append(prompt)
            
            # aovs is a tuple (albedo, normal, roughness, metallic) corresponding to dataset.aov_types
            aov_images["albedo"].append(aovs[0])
            aov_images["normal"].append(aovs[1])
            aov_images["roughness"].append(aovs[2])
            aov_images["metallic"].append(aovs[3])

        target_images = torch.stack(target_images).float().contiguous()
    
        # Set value to be a stacked tensor
        for key in aov_images:
            aov_images[key] = torch.stack(aov_images[key]).to(memory_format=torch.contiguous_format).float()

        return {
            "target_images": target_images,
            "prompts": prompts,
            "aov_images": aov_images
        }

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=arg_train_batch_size,
        num_workers=1,
    )

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=arg_learning_rate,
    )

    # Scaling factors from pipeline_x2rgb.py
    SCALING_FACTORS = {
        "albedo": 0.17301377137652138,
        "normal": 0.17483895473058078,
        "roughness": 0.1680724853626448,
        "metallic": 0.13135013390855135,
    }

    # See https://github.com/huggingface/diffusers/commit/a1cfb0acccbfcbc70b83ca379594fe854a8df24b
    num_warmup_steps_for_scheduler = arg_lr_warmup_steps * accelerator.num_processes
    if arg_max_train_steps is None:
        # We divide by num_processes because the dataloader will be split across GPUs
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / arg_gradient_accumulation_steps)
        num_training_steps_for_scheduler = arg_num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
    else:
        num_training_steps_for_scheduler = arg_max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        arg_lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / arg_gradient_accumulation_steps)
    if arg_max_train_steps is None:
        arg_max_train_steps = arg_num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != arg_max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    arg_num_train_epochs = math.ceil(arg_max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(range(arg_max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # 5. Training Loop
    for epoch in range(arg_num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # A. Encode Target Image (Ground Truth)
                target_images = batch["target_images"]
                
                with torch.no_grad():
                    # Scale images to [-1, 1] using preprocess
                    target_images_processed = pipeline.image_processor.preprocess(target_images)
                    target_images_processed = target_images_processed.to(vae.device, dtype=vae.dtype)
                    target_image_latents = vae.encode(target_images_processed.to(dtype=weight_dtype)).latent_dist.sample()
                    target_image_latents = target_image_latents * vae.config.scaling_factor

                # B. Sample Noise
                noise = torch.randn_like(target_image_latents)
                batch_size = target_image_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=target_image_latents.device) # TODO: Where does config.num_train_timesteps come from?
                timesteps = timesteps.long()

                # C. Add Noise (Forward Diffusion)
                noisy_latents = noise_scheduler.add_noise(target_image_latents, noise, timesteps)

                # D. Encode Prompts
                prompts = batch["prompts"]
                if arg_prob_prompt_dropout > 0:
                    prompts = [
                        "" if torch.rand(1).item() < arg_prob_prompt_dropout else p 
                        for p in prompts
                    ]

                prompt_inputs = tokenizer(
                    prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                )
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(prompt_inputs.input_ids.to(accelerator.device))[0] # Text embeddings

                # E. Prepare AOV Conditioning
                aov_latents_list = []
                ordered_aov_keys = ["albedo", "normal", "roughness", "metallic"]
                
                with torch.no_grad():
                    for key in ordered_aov_keys:
                        aov_img = batch["aov_images"][key]

                        if arg_prob_intrinsic_dropout > 0 and torch.rand(1).item() < arg_prob_intrinsic_dropout:
                            # Drop this intrinsic AOV channel
                            aov_latent = torch.zeros_like(noise)
                        else:
                            if key != "normal": 
                                # Use preprocess for normalization
                                aov_img = pipeline.image_processor.preprocess(aov_img)
                            aov_img = aov_img.to(vae.device, dtype=vae.dtype)

                            aov_latent = vae.encode(aov_img.to(dtype=weight_dtype)).latent_dist.mode() 
                            aov_latent = aov_latent * SCALING_FACTORS[key]

                        assert aov_latent.shape == noise.shape, f"AOV latent shape {aov_latent.shape} does not match noise shape {noise.shape}"
                        aov_latents_list.append(aov_latent)
                
                empty_irradiance_channel = torch.zeros((batch_size, 3, *aov_latents_list[0].shape[2:]), device=accelerator.device)
                aov_latents_list.append(empty_irradiance_channel) # Append empty irradiance channel
                conditioning_latents = torch.cat(aov_latents_list, dim=1) # Size [batch_size, 4 x num_aovs, H/8, W/8]
                unet_input = torch.cat([noisy_latents, conditioning_latents], dim=1)

                # F. Predict Noise
                model_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                train_with_velocity = True
                if train_with_velocity:
                    target = noise_scheduler.get_velocity(target_image_latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float())
                
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % 50 == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(arg_output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrap_model = accelerator.unwrap_model(unet)
                        unwrap_model.save_pretrained(save_path)
                        logger.info(f"Saved UNet checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= arg_max_train_steps:
                break
    
    accelerator.end_training()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
