import os

# Ensure EXR files can be read
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
import json
from datetime import datetime
import re

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


def load_aov_image(filepath: str, aov_type: str, device):
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
            return load_exr_image(filepath, tonemapping=True, clamp=True).to(device)
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

    # 5. Run inference (supports single prompt or batch from args._prompts)
    print("Running inference...")
    required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]

    prompts = getattr(args, '_prompts', [args.prompt])

    # Helper to sanitize prompt into file-safe label
    def _sanitize_prompt(p):
        s = re.sub(r'[^A-Za-z0-9._-]', '_', p)
        return s[:120]

    def _shape_of(img):
        if img is None:
            return None
        try:
            if hasattr(img, 'shape'):
                return [int(x) for x in img.shape]
        except Exception:
            pass
        try:
            return list(img.shape)
        except Exception:
            return None

    # Ensure output directory when needed
    out_dir = args.output_dir or "./outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Run the pipeline once with the list of prompts (handles single-item lists too)
    result = pipe(
        prompt=prompts,
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
    )

    images = result.images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build output paths: if a single prompt and --output_path given, use it; otherwise generate names
    out_paths = []
    if len(prompts) == 1 and args.output_path:
        out_paths = [args.output_path]
    else:
        for i, prompt_text in enumerate(prompts):
            label = _sanitize_prompt(prompt_text)
            fname = f"out_{label}_{timestamp}_{i}.png"
            out_paths.append(os.path.join(out_dir, fname))

    # Save images and metadata
    for i, (prompt_text, img_np, out_path) in enumerate(zip(prompts, images, out_paths)):
        print(f"Saving output for prompt #{i} to {out_path}")
        img_u8 = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(out_path)

        metadata = {
            "created_at": datetime.now().isoformat() + "Z",
            "prompt": prompt_text,
            "seed": int(args.seed),
            "device": device,
            "model_id": "zheng95z/x-to-rgb",
            "unet_checkpoint": args.unet_checkpoint,
            "inference_step": args.inference_step,
            "guidance_scale": args.guidance_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "cache_dir": args.cache_dir,
            "height": int(height),
            "width": int(width),
            "aovs": {
                "albedo": {"path": args.albedo, "shape": _shape_of(albedo_image)},
                "normal": {"path": args.normal, "shape": _shape_of(normal_image)},
                "roughness": {"path": args.roughness, "shape": _shape_of(roughness_image)},
                "metallic": {"path": args.metallic, "shape": _shape_of(metallic_image)},
                "irradiance": {"path": args.irradiance, "shape": _shape_of(irradiance_image)},
            },
            "output_image": out_path,
        }

        try:
            # add subdirectory and filename
            subdir_name = 'metadata'
            parent_dir = os.path.dirname(out_path)
            metadata_dir = os.path.join(parent_dir, subdir_name)
            os.makedirs(metadata_dir, exist_ok=True)
            base, _ = os.path.splitext(out_path)
            json_path = base + ".json"
            final_path = os.path.join(metadata_dir, os.path.basename(json_path))
            with open(final_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Wrote metadata to {final_path}")
        except Exception as e:
            print(f"Warning: failed to write metadata JSON: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run X-to-RGB inference from the command line")
    
    # --- File I/O Arguments ---
    parser.add_argument("--albedo", type=str, default=None, help="Path to albedo image (.exr, .png, .jpg)")
    parser.add_argument("--normal", type=str, default=None, help="Path to normal image (.exr, .png, .jpg)")
    parser.add_argument("--roughness", type=str, default=None, help="Path to roughness image (.exr, .png, .jpg)")
    parser.add_argument("--metallic", type=str, default=None, help="Path to metallic image (.exr, .png, .jpg)")
    parser.add_argument("--irradiance", type=str, default=None, help="Path to irradiance image (.exr, .png, .jpg)")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to a text file with one prompt per line (optional batch mode)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save outputs (default: ./outputs)")
    
    # --- Model Parameter Arguments ---
    parser.add_argument("--prompt", type=str, required=False, default=None, help="Text prompt for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Use -1 for a random seed.")
    parser.add_argument("--inference_step", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Text guidance scale")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5, help="Image guidance scale")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to UNet checkpoint to load (optional)")
    
    # --- System Arguments ---
    parser.add_argument("--cache_dir", type=str, default="./model_cache", help="Directory to cache the downloaded model")
    parser.add_argument("--device", type=str, default=None, help="Device to use for inference (e.g., 'cuda', 'cpu'). If not provided, defaults to CUDA if available, otherwise CPU.")

    args = parser.parse_args()
    # If a prompts file is provided, it takes precedence over single --prompt
    if args.prompts_file and args.prompt:
        # Allow both but prefer file; warn user
        print("Note: both --prompt and --prompts_file provided — using --prompts_file")

    # Read prompts list
    prompts = None
    if args.prompts_file:
        if not os.path.exists(args.prompts_file):
            raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")
        with open(args.prompts_file, 'r') as pf:
            lines = [l.strip() for l in pf.readlines()]
            # filter out empty lines
            prompts = [l for l in lines if l]
        if len(prompts) == 0:
            raise ValueError(f"No prompts found in {args.prompts_file}")
    else:
        # fallback to single prompt
        if not args.prompt:
            raise ValueError("Either --prompt or --prompts_file must be provided")
        prompts = [args.prompt]

    # attach prompts to args for main
    args._prompts = prompts

    main(args)