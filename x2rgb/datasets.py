from torch.utils.data import Dataset, ConcatDataset
import glob
import os
import pandas as pd
import random
import re
from PIL import Image
import torchvision.transforms as T
from abc import ABC, abstractmethod

from aov_utils import load_aov_image
import math
import torch
import torch.nn.functional as F

class BaseAOVDataset(Dataset, ABC):
    def __init__(self, base_path: str, aov_types=['albedo', 'normal', 'roughness', 'metallic']):
        self.base_path = base_path
        self.aov_types = aov_types
        self.target_file_paths = self._get_paths_to_target_images()
        self.csv_file_path = self._get_csv_file_path()

    def __len__(self) -> int:
        return len(self.target_file_paths)

    def __getitem__(self, idx: int):
        target_image_path = self.target_file_paths[idx]
        aov_images = []
        for aov_type in self.aov_types:
            aov_path = self._get_path_to_aov(target_image_path, aov_type)
            if aov_path is not None:
                assert os.path.exists(aov_path), f"AOV path does not exist: {aov_path}"
                aov_image = load_aov_image(aov_path, aov_type, device='cpu')
            else:
                aov_image = None # TODO: is this okay, or should we do a zero tensor here?
            aov_images.append(aov_image)
        
        try:
            prompt = self._get_prompt_for_target_image(target_image_path)
        except ValueError as e:
            print(e)
            prompt = "" # Just drop out the prompt if it's not present
            
        target_image = self._load_target_image(target_image_path)
        return (tuple(aov_images), prompt), target_image

    def _load_target_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(), # Converts to [0, 1]
        ])
        return transform(image)
    
    def _get_csv_file_path(self) -> str:
        csv_file_path = glob.glob(os.path.join(self.base_path, "*.csv"))
        assert len(csv_file_path) == 1, f"There should be exactly one CSV file in the base path, but found {len(csv_file_path)}: {csv_file_path}"
        return csv_file_path[0]

    @abstractmethod
    def _get_path_to_aov(self, target_image_path: str, aov_type: str) -> str | None:
        pass

    @abstractmethod
    def _get_prompt_for_target_image(self, image_path: str) -> str:
        pass

    @abstractmethod
    def _get_paths_to_target_images(self) -> list:
        pass


class LightingFineTuneDataset(BaseAOVDataset):
    def __init__(self, base_path: str, aov_types=['albedo', 'normal', 'roughness', 'metallic']):
        super().__init__(base_path, aov_types)

        # Read the csv file into memory 
        self.text_descriptions = pd.read_csv(self.csv_file_path)
        # Set photo_path as index for efficient lookup
        self.text_descriptions.set_index('photo_path', inplace=True)
        print(f"Loaded {len(self.text_descriptions)} text descriptions from {self.csv_file_path}")

    def _get_paths_to_target_images(self) -> list:
        return glob.glob(os.path.join(self.base_path, "*", "*.png"))

    def _get_prompt_for_target_image(self, image_path: str) -> str:
        relative_path = os.path.relpath(image_path, self.base_path)
        row = None
        try:
            row = self.text_descriptions.loc[relative_path]
        except KeyError:
            raise ValueError(f"No text description found for image path: {relative_path}")
        return row['lighting_description_01'] if random.random() < 0.5 else row['lighting_description_02']
    
    def _get_path_to_aov(self, target_image_path: str, aov_type: str) -> str:
        base_name = os.path.basename(target_image_path)
        base_name_parts = base_name.split('_')
        camera_seed = base_name_parts[-2]
        scene_id = base_name_parts[-1].split('.')[0] # Remove file extension
        path_to_aov = glob.glob(os.path.join(self.base_path, 'aovs', f"{scene_id}_camera_{camera_seed}_aovs", aov_type + "*.png"))
        assert len(path_to_aov) == 1, f"Expected exactly one {aov_type} AOV for image {target_image_path}, found {len(path_to_aov)}."
        return path_to_aov[0]


class LightingFineTuneDatasetDiscrete(BaseAOVDataset):
    def __init__(self, base_path: str, aov_types=['albedo', 'normal', 'roughness', 'metallic']):
        super().__init__(base_path, aov_types)

        # Read the csv file into memory and build mapping
        df = pd.read_csv(self.csv_file_path)
        self.light_to_prompt = {}
        for index, row in df.iterrows():
            path = row['photo_path']
            prompt = row['lighting_description_01']
            match = re.search(r'light_(\d+)', path)
            if match:
                light_id = int(match.group(1))
                self.light_to_prompt[light_id] = prompt
        
        print(f"Loaded {len(self.light_to_prompt)} light descriptions from {self.csv_file_path}")

    def _get_paths_to_target_images(self) -> list:
        return glob.glob(os.path.join(self.base_path, "*", "*.png"))

    def _get_prompt_for_target_image(self, image_path: str) -> str:
        base_name = os.path.basename(image_path)
        match = re.search(r'light_(\d+)', base_name)
        if match:
            light_id = int(match.group(1))
            if light_id in self.light_to_prompt:
                return self.light_to_prompt[light_id]
        
        raise ValueError(f"No text description found for image path: {image_path}")

    def _get_path_to_aov(self, target_image_path: str, aov_type: str) -> str:
        dir_name = os.path.dirname(target_image_path)
        base_name = os.path.basename(target_image_path)
        
        # Remove extension
        filename_no_ext = os.path.splitext(base_name)[0]
        
        # Regex to remove light_XXX_
        aov_folder_name = re.sub(r'_light_\d+', '', filename_no_ext)
        aov_folder_name += "_aovs"
        
        aov_dir = os.path.join(dir_name, 'aovs', aov_folder_name)
        
        path_to_aov = glob.glob(os.path.join(aov_dir, aov_type + "*.png"))
        
        if len(path_to_aov) != 1:
             raise FileNotFoundError(f"Expected exactly one {aov_type} AOV in {aov_dir}, found {len(path_to_aov)}.")

        return path_to_aov[0]


class InteriorverseFineTuneDataset(BaseAOVDataset):
    def __init__(self, base_path: str=os.path.expanduser('~/groups/grp_cs_650_rgb_x/nobackup/autodelete/interiorverse_dataset')):
        super().__init__(base_path)
        
        self.text_descriptions = pd.read_csv(self.csv_file_path)
        self.text_descriptions.set_index('photo_path', inplace=True)

    def _get_paths_to_target_images(self) -> list:
        return glob.glob(os.path.join(self.base_path, 'dataset_85_part_*', '*', '*_im_denoised.png'))

    def _get_path_to_aov(self, target_image_path: str, aov_type: str) -> str:
        if aov_type in ('metallic', 'roughness'):
            return None  # These AOVs are not available in the Interiorverse dataset
        str_id = self._get_str_id_from_file_path(target_image_path)
        target_directory_name = os.path.dirname(target_image_path)
        aov_path = os.path.join(target_directory_name, f"{str_id}_{aov_type}.exr")
        return aov_path

    def _get_str_id_from_file_path(self, file_path: str) -> str:
        base_name = os.path.basename(file_path)
        str_id = base_name.split('_')[0]
        return str_id

    def _get_prompt_for_target_image(self, image_path: str) -> str:
        # Regex match the id from the image path
        base_name = os.path.basename(image_path)
        str_id = self._get_str_id_from_file_path(image_path)
        base_name_with_description = base_name.replace(str_id, '000') # only the first denoised image was given a description
        path_with_description = os.path.join(os.path.dirname(image_path), base_name_with_description)
        # In this case, path_with_description should index directly to the prompt
        try:
            row = self.text_descriptions.loc[path_with_description]
        except KeyError:
            raise ValueError(f"No text description found for image path: {path_with_description}")
        return row['lighting_description_01']


class MixedAOVDataset(Dataset):
    """A dataset that combines multiple AOV datasets and enforces a common image size.

    Responsibilities:
    - Concatenate multiple dataset instances into a single indexable dataset.
    - Resize (without stretching) and center-crop all AOVs and target images to
      a fixed `standard_size` supplied at construction time. Missing AOVs are
      substituted with zero tensors of the expected shape.
    """
    def __init__(self, datasets: list, standard_size=(512, 512)):
        if not datasets:
            raise ValueError("Must provide at least one dataset to MixedAOVDataset")

        self.datasets = datasets
        self.combined = ConcatDataset(datasets)
        self.standard_size = tuple(standard_size)

    def __len__(self) -> int:
        return len(self.combined)

    def _resize_and_crop(self, img: torch.Tensor) -> torch.Tensor:
        """Resize a tensor image (C,H,W) preserving aspect ratio, then center-crop.

        The image is upsampled/downsampled so that both dimensions are at least
        the target, then a center crop to the exact `standard_size` is performed.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Expected torch.Tensor for resizing")

        if img.ndim != 3:
            raise ValueError("Expected image tensor with shape (C,H,W)")

        c, h, w = img.shape
        target_h, target_w = self.standard_size

        # Compute scale so smaller side meets the target (no stretching)
        scale = max(float(target_h) / float(h), float(target_w) / float(w))
        new_h = max(int(math.ceil(h * scale)), 1)
        new_w = max(int(math.ceil(w * scale)), 1)

        img_b = img.unsqueeze(0)  # 1,C,H,W
        img_resized = F.interpolate(img_b, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Center crop to target size
        _, _, rh, rw = img_resized.shape
        top = (rh - target_h) // 2
        left = (rw - target_w) // 2
        img_cropped = img_resized[:, :, top: top + target_h, left: left + target_w]
        return img_cropped.squeeze(0)

    def __getitem__(self, idx: int):
        # Underlying datasets follow the ((aovs_tuple), prompt), target pattern
        (aovs, prompt), target = self.combined[idx]

        resized_aovs = []
        for aov in aovs:
            if aov is None:
                # Substitute missing AOV with zeros of the standard size
                placeholder = torch.zeros((3, self.standard_size[0], self.standard_size[1]), dtype=torch.float32)
                resized_aovs.append(placeholder)
            else:
                # Ensure float tensor on CPU
                aov_tensor = aov.detach().cpu().float()
                # If single-channel, expand to 3 channels
                if aov_tensor.shape[0] == 1:
                    aov_tensor = aov_tensor.repeat(3, 1, 1)
                resized = self._resize_and_crop(aov_tensor)
                resized_aovs.append(resized)

        # Resize target image as well
        target_tensor = target.detach().cpu().float()
        if target_tensor.ndim == 2:
            target_tensor = target_tensor.unsqueeze(0).repeat(3, 1, 1)
        if target_tensor.shape[0] == 1:
            target_tensor = target_tensor.repeat(3, 1, 1)

        target_resized = self._resize_and_crop(target_tensor)

        return (tuple(resized_aovs), prompt), target_resized


if __name__ == "__main__":
    # dataset = LightingFineTuneDataset("/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/custom_aov_dataset/aov_test_04")
    dataset = LightingFineTuneDatasetDiscrete("/home/ansonsav/groups/grp_cs_650_rgb_x/nobackup/autodelete/blender_aov_dataset/product_content_lock_test_03")
    items_to_print = 10
    for i in range(items_to_print):
        (aov_images, prompt), target_image = dataset[i]
        print(f"Item {i}:")
        print(f"  Prompt: {prompt}")
        for aov_img, aov_type in zip(aov_images, dataset.aov_types):
            print(f"  {aov_type} image shape: {aov_img.shape}")
        print(f"  Target image shape: {target_image.shape}")
