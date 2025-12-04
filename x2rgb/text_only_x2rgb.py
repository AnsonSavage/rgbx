import os

# Ensure EXR files can be read
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

# --- Assumed local imports ---
# These files (load_image.py, pipeline_x2rgb.py) 
# must be in the same directory.
from load_image import load_exr_image, load_ldr_image
from pipeline_x2rgb import StableDiffusionAOVDropoutPipeline
# -------------------------------


def get_default_device():
    """
    Determine the default device based on availability.
    Prefers CUDA if available, falls back to CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_aov_image(filepath, aov_type, device):
    """
    Loads and preprocesses a single AOV image based on its type and file extension.
    This logic is extracted directly from your original callback.
    """
    if filepath is None or not os.path.exists(filepath):
        if filepath:
            print(f"Warning: File not found, skipping: {filepath}")
        return None

    print(f"Loading {aov_type}: {filepath}")
    if filepath.endswith(".exr"):
        if aov_type == 'normal':
            return load_exr_image(filepath, normalize=True).to(device)
        if aov_type == 'irradiance':
            return load_exr_image(filepath, tonemaping=True, clamp=True).to(device)
        # albedo, roughness, metallic
        return load_exr_image(filepath, clamp=True).to(device)
    
    elif filepath.endswith((".png", ".jpg", ".jpeg")):
        if aov_type == 'normal':
            return load_ldr_image(filepath, normalize=True).to(device)
        if aov_type == 'albedo':
            return load_ldr_image(filepath, from_srgb=True).to(device)
        if aov_type == 'irradiance':
            return load_ldr_image(filepath, from_srgb=True, clamp=True).to(device)
        # roughness, metallic
        return load_ldr_image(filepath, clamp=True).to(device)
    
    else:
        print(f"Warning: Unsupported file type, skipping: {filepath}")
        return None

def main(args):
    """
    Main function to load the model, process inputs, and run inference.
    """
    # Determine device
    device = args.device if args.device else get_default_device()
    print(f"Using device: {device}")
    
    # 1. Load pipeline
    print("Loading pipeline...")
    cache_dir = os.path.join(args.cache_dir)
    pipe = StableDiffusionAOVDropoutPipeline.from_pretrained(
        "zheng95z/x-to-rgb",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    ).to(device)

    if args.unet_checkpoint:
        unet = pipe.unet
        assert os.path.isdir(args.unet_checkpoint), f"Provided UNet checkpoint path is not a directory: {args.unet_checkpoint}"
        print(f"Loading UNet weights from checkpoint: {args.unet_checkpoint}")
        unet = unet.__class__.from_pretrained(args.unet_checkpoint, torch_dtype=torch.float16).to(device)
        pipe.unet = unet

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    # 2. Load all AOV images
    print("Loading AOV images...")
    albedo_image = load_aov_image(args.albedo, 'albedo', device)
    normal_image = load_aov_image(args.normal, 'normal', device)
    roughness_image = load_aov_image(args.roughness, 'roughness', device)
    metallic_image = load_aov_image(args.metallic, 'metallic', device)
    irradiance_image = load_aov_image(args.irradiance, 'irradiance', device)

    # 3. Determine height/width
    height, width = 768, 768 # Default
    images = [albedo_image, normal_image, roughness_image, metallic_image, irradiance_image]
    
    found_res = False
    for img in images:
        if img is not None:
            height = img.shape[1]
            width = img.shape[2]
            print(f"Detected resolution: {width}x{height}")
            found_res = True
            break
    
    if not found_res:
        print("Error: At least one input image (albedo, normal, etc.) must be provided.")
        return

    # 4. Setup generator
    if args.seed == -1:
        # Generate a random seed if -1 is specified
        args.seed = torch.Generator(device=device).seed()
        
    print(f"Using seed: {args.seed}")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # 5. Run inference
    print("Running inference...")
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
    
    generated_image = pipe(
        prompt=args.prompt,
        albedo=albedo_image,
        normal=normal_image,
        roughness=roughness_image,
        metallic=metallic_image,
        irradiance=irradiance_image,
        num_inference_steps=args.inference_step,
        height=height,
        width=width,
        generator=generator,
        required_aovs=required_aovs,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        guidance_rescale=0.7,
        output_type="np",
    ).images[0] # Get the first (and only) image

    # 6. Save output
    print(f"Saving output to {args.output_path}...")
    
    # Convert from float [0.0, 1.0] to uint8 [0, 255]
    image_np = (np.clip(generated_image, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Save using PIL
    pil_image = Image.fromarray(image_np)
    pil_image.save(args.output_path)
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run X-to-RGB inference from the command line")
    
    # --- File I/O Arguments ---
    parser.add_argument("--albedo", type=str, default=None, help="Path to albedo image (.exr, .png, .jpg)")
    parser.add_argument("--normal", type=str, default=None, help="Path to normal image (.exr, .png, .jpg)")
    parser.add_argument("--roughness", type=str, default=None, help="Path to roughness image (.exr, .png, .jpg)")
    parser.add_argument("--metallic", type=str, default=None, help="Path to metallic image (.exr, .png, .jpg)")
    parser.add_argument("--irradiance", type=str, default=None, help="Path to irradiance image (.exr, .png, .jpg)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image (e.g., output.png)")
    
    # --- Model Parameter Arguments ---
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use -1 for a random seed.")
    parser.add_argument("--inference_step", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Text guidance scale")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="Image guidance scale")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to UNet checkpoint to load (optional)")
    
    # --- System Arguments ---
    parser.add_argument("--cache_dir", type=str, default="./model_cache", help="Directory to cache the downloaded model")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference (e.g., 'cuda', 'cpu'). If not provided, defaults to CUDA if available, otherwise CPU.")

    args = parser.parse_args()
    main(args)