"""
Utility functions for loading and preprocessing AOV (Arbitrary Output Variable) images.
"""
import os
import torch
from load_image import load_exr_image, load_ldr_image


def load_aov_image(filepath, aov_type: str, device='cuda'):
    """
    Loads and preprocesses a single AOV image based on its type and file extension.
    
    Args:
        filepath: Path to the image file, or a file object with a 'name' attribute, or None
        aov_type: Type of AOV - one of 'albedo', 'normal', 'roughness', 'metallic', 'irradiance'
        device: PyTorch device to load the image onto (default: 'cuda')
    
    Returns:
        Preprocessed image tensor, or None if filepath is None or file doesn't exist
    """
    # Handle None case
    if filepath is None:
        return None
    
    # Handle file object (from Gradio) vs string path
    if hasattr(filepath, 'name'):
        filepath = filepath.name
    
    # Check if file exists
    if not os.path.exists(filepath):
        return None
    
    # Load based on file extension and AOV type
    if filepath.endswith(".exr"):
        if aov_type == 'normal':
            return load_exr_image(filepath, normalize=True).to(device)
        elif aov_type == 'irradiance':
            return load_exr_image(filepath, tonemapping=True, clamp=True).to(device)
        else:  # albedo, roughness, metallic
            return load_exr_image(filepath, clamp=True).to(device)
    
    elif filepath.endswith((".png", ".jpg", ".jpeg")):
        if aov_type == 'normal':
            return load_ldr_image(filepath, normalize=True).to(device)
        elif aov_type == 'albedo':
            return load_ldr_image(filepath, from_srgb=True).to(device)
        elif aov_type == 'irradiance':
            return load_ldr_image(filepath, from_srgb=True, clamp=True).to(device)
        else:  # roughness, metallic
            return load_ldr_image(filepath, clamp=True).to(device)
    
    else:
        return None


def load_all_aovs(albedo=None, normal=None, roughness=None, metallic=None, irradiance=None, device='cuda'):
    """
    Load all AOV images at once.
    
    Args:
        albedo: Path or file object for albedo image
        normal: Path or file object for normal image
        roughness: Path or file object for roughness image
        metallic: Path or file object for metallic image
        irradiance: Path or file object for irradiance image
        device: PyTorch device to load images onto (default: 'cuda')
    
    Returns:
        Dictionary with keys matching AOV names, values are loaded tensors or None
    """
    return {
        'albedo': load_aov_image(albedo, 'albedo', device),
        'normal': load_aov_image(normal, 'normal', device),
        'roughness': load_aov_image(roughness, 'roughness', device),
        'metallic': load_aov_image(metallic, 'metallic', device),
        'irradiance': load_aov_image(irradiance, 'irradiance', device),
    }
