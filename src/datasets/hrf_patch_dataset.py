import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

class HRFPatchDataset(Dataset):
    def __init__(self, images_dir, masks_dir, fov_masks_dir, patch_size=512, overlap=128, transform=None, min_vessel_ratio=0.01):
        """
        HRF Dataset with patch-based approach
        
        Args:
            images_dir: Directory containing HRF images
            masks_dir: Directory containing vessel ground truth (manual1)
            fov_masks_dir: Directory containing FOV masks
            patch_size: Size of patches to extract (default: 512)
            overlap: Overlap between patches (default: 128)
            transform: Albumentations transforms
            min_vessel_ratio: Minimum ratio of vessel pixels in patch to include it
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fov_masks_dir = fov_masks_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.min_vessel_ratio = min_vessel_ratio
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.JPG', '.png', '.tif']:
            self.image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
        self.image_files = sorted(self.image_files)
        
        # Extract all valid patches
        self.patches_info = self._extract_patches()
        
        print(f"Found {len(self.image_files)} HRF images")
        print(f"Generated {len(self.patches_info)} valid patches")
    
    def _get_mask_filename(self, image_filename):
        """Convert image filename to mask filename"""
        base_name = os.path.splitext(image_filename)[0]
        return base_name + '.tif'
    
    def _get_fov_filename(self, image_filename):
        """Convert image filename to FOV mask filename"""
        base_name = os.path.splitext(image_filename)[0]
        return base_name + '_mask.tif'
    
    def _extract_patches(self):
        """Extract all valid patches from all images"""
        patches_info = []
        stride = self.patch_size - self.overlap
        
        for img_file in self.image_files:
            # Load vessel mask to check patch validity
            mask_file = self._get_mask_filename(img_file)
            mask_path = os.path.join(self.masks_dir, mask_file)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {img_file}")
                continue
                
            vessel_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if vessel_mask is None:
                continue
                
            h, w = vessel_mask.shape
            
            # Extract patches with stride
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    # Check if patch contains enough vessel pixels
                    patch_mask = vessel_mask[y:y+self.patch_size, x:x+self.patch_size]
                    vessel_ratio = np.sum(patch_mask > 0) / (self.patch_size * self.patch_size)
                    
                    if vessel_ratio >= self.min_vessel_ratio:
                        patches_info.append({
                            'image_file': img_file,
                            'x': x, 'y': y,
                            'vessel_ratio': vessel_ratio
                        })
        
        return patches_info
    
    def __len__(self):
        return len(self.patches_info)
    
    def __getitem__(self, idx):
        patch_info = self.patches_info[idx]
        img_file = patch_info['image_file']
        x, y = patch_info['x'], patch_info['y']
        
        # Get corresponding filenames
        mask_file = self._get_mask_filename(img_file)
        fov_file = self._get_fov_filename(img_file)
        
        # Load full images
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)
        fov_path = os.path.join(self.fov_masks_dir, fov_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read vessel mask
        vessel_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if vessel_mask is None:
            raise ValueError(f"Could not read vessel mask at {mask_path}")
        
        # Read FOV mask
        fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
        if fov_mask is None:
            raise ValueError(f"Could not read FOV mask at {fov_path}")
        
        # Extract patches
        image_patch = image[y:y+self.patch_size, x:x+self.patch_size]
        vessel_patch = vessel_mask[y:y+self.patch_size, x:x+self.patch_size]
        fov_patch = fov_mask[y:y+self.patch_size, x:x+self.patch_size]
        
        # Normalize masks
        vessel_patch = vessel_patch.astype(np.float32) / 255.0
        fov_patch = fov_patch.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            # Combine masks for joint augmentation
            combined_mask = np.stack([vessel_patch, fov_patch], axis=2)
            augmented = self.transform(image=image_patch, mask=combined_mask)
            image_patch = augmented['image']
            combined_mask = augmented['mask']
            
            # Split back to separate masks
            vessel_patch = combined_mask[:, :, 0] if len(combined_mask.shape) == 3 else combined_mask
            fov_patch = combined_mask[:, :, 1] if len(combined_mask.shape) == 3 else fov_patch
        
        # Convert to tensors and add channel dimension
        if isinstance(vessel_patch, np.ndarray):
            vessel_patch = torch.from_numpy(vessel_patch)
        if len(vessel_patch.shape) == 2:
            vessel_patch = vessel_patch.unsqueeze(0)
        
        if isinstance(fov_patch, np.ndarray):
            fov_patch = torch.from_numpy(fov_patch)
        if len(fov_patch.shape) == 2:
            fov_patch = fov_patch.unsqueeze(0)
        
        return image_patch, vessel_patch, fov_patch

def get_patch_train_transform():
    """Training transforms for patches (no resize needed!)"""
    return A.Compose([
        # Geometric transformations (patches are already 512x512)
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=0.0625,
            scale=(0.9, 1.1),
            rotate=(-20, 20),
            p=0.5
        ),
        
        # Light deformations
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=80 * 0.05, p=0.3),
            A.GridDistortion(distort_limit=0.2, p=0.3),
            A.OpticalDistortion(distort_limit=0.2, p=0.3),
        ], p=0.3),
        
        # Color and intensity augmentations
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
        ], p=0.5),
        
        # Noise (fixed parameter names)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
        ], p=0.3),
        
        A.CoarseDropout(
            max_holes=6,
            max_height=16,
            max_width=16,
            fill_value=0,
            p=0.2
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_patch_valid_transform():
    """Validation transforms for patches"""
    return A.Compose([
        # Only normalization for validation
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]) 