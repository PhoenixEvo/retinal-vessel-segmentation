import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

class DRIVEPatchDataset(Dataset):
    def __init__(self, images_dir, masks_dir, fov_masks_dir=None, patch_size=512, overlap=128, 
                 transform=None, min_vessel_ratio=0.01, split='train', train_ratio=0.8):
        """
        DRIVE Dataset with patch-based approach
        
        Args:
            images_dir: Directory containing DRIVE images
            masks_dir: Directory containing vessel ground truth (1st_manual)
            fov_masks_dir: Directory containing FOV masks (optional)
            patch_size: Size of patches to extract (default: 512)
            overlap: Overlap between patches (default: 128)
            transform: Albumentations transforms
            min_vessel_ratio: Minimum ratio of vessel pixels in patch to include it
            split: 'train' or 'val' for train/validation split
            train_ratio: Ratio of images for training (default: 0.8)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fov_masks_dir = fov_masks_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.min_vessel_ratio = min_vessel_ratio
        self.split = split
        self.train_ratio = train_ratio
        
        # Get all image files
        self.image_files = []
        for ext in ['.tif']:
            self.image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
        self.image_files = sorted(self.image_files)
        
        # Split images into train/val
        num_train = int(len(self.image_files) * train_ratio)
        if split == 'train':
            self.image_files = self.image_files[:num_train]
        else:  # val
            self.image_files = self.image_files[num_train:]
        
        # Extract all valid patches
        self.patches_info = self._extract_patches()
        
        print(f"DRIVE {split} dataset: {len(self.image_files)} images -> {len(self.patches_info)} patches (min_vessel_ratio={min_vessel_ratio})")
    
    def _get_mask_filename(self, image_filename):
        """Convert image filename to mask filename"""
        # Example: 21_training.tif -> 21_manual1.gif
        base_name = image_filename.replace('_training.tif', '')
        return base_name + '_manual1.gif'
    
    def _get_fov_filename(self, image_filename):
        """Convert image filename to FOV mask filename"""
        # Example: 21_training.tif -> 21_training_mask.gif
        base_name = image_filename.replace('.tif', '')
        return base_name + '_mask.gif'
    
    def _load_drive_mask(self, mask_path):
        """Load DRIVE mask using PIL (supports .gif)"""
        try:
            mask_pil = Image.open(mask_path).convert('L')
            mask = np.array(mask_pil)
            return mask.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None
    
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
                
            vessel_mask = self._load_drive_mask(mask_path)
            if vessel_mask is None:
                continue
                
            h, w = vessel_mask.shape
            
            # Load FOV mask if available
            fov_mask = None
            if self.fov_masks_dir:
                fov_file = self._get_fov_filename(img_file)
                fov_path = os.path.join(self.fov_masks_dir, fov_file)
                if os.path.exists(fov_path):
                    fov_mask = self._load_drive_mask(fov_path)
            
            # Extract patches with stride
            for y in range(0, h - self.patch_size + 1, stride):
                for x in range(0, w - self.patch_size + 1, stride):
                    # Check if patch is within FOV (if available)
                    if fov_mask is not None:
                        fov_patch = fov_mask[y:y+self.patch_size, x:x+self.patch_size]
                        fov_ratio = np.sum(fov_patch > 0.5) / (self.patch_size * self.patch_size)
                        if fov_ratio < 0.5:  # Skip patches mostly outside FOV
                            continue
                    
                    # Check if patch contains enough vessel pixels
                    patch_mask = vessel_mask[y:y+self.patch_size, x:x+self.patch_size]
                    
                    # Only count vessel pixels within FOV if available
                    if fov_mask is not None:
                        fov_patch = fov_mask[y:y+self.patch_size, x:x+self.patch_size]
                        valid_area = np.sum(fov_patch > 0.5)
                        if valid_area > 0:
                            vessel_ratio = np.sum((patch_mask > 0) & (fov_patch > 0.5)) / valid_area
                        else:
                            continue
                    else:
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
        
        # Load full images
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read vessel mask
        vessel_mask = self._load_drive_mask(mask_path)
        if vessel_mask is None:
            raise ValueError(f"Could not read vessel mask at {mask_path}")
        
        # Read FOV mask if available
        fov_mask = None
        if self.fov_masks_dir:
            fov_file = self._get_fov_filename(img_file)
            fov_path = os.path.join(self.fov_masks_dir, fov_file)
            if os.path.exists(fov_path):
                fov_mask = self._load_drive_mask(fov_path)
        
        # Extract patches
        image_patch = image[y:y+self.patch_size, x:x+self.patch_size]
        vessel_patch = vessel_mask[y:y+self.patch_size, x:x+self.patch_size]
        
        if fov_mask is not None:
            fov_patch = fov_mask[y:y+self.patch_size, x:x+self.patch_size]
        else:
            # Create dummy FOV mask (all ones) if not available
            fov_patch = np.ones_like(vessel_patch)
        
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
        
        # Apply FOV mask to vessel mask during training
        if isinstance(fov_patch, np.ndarray):
            fov_patch_tensor = torch.from_numpy(fov_patch)
        else:
            fov_patch_tensor = fov_patch
        
        if len(fov_patch_tensor.shape) == 2:
            fov_patch_tensor = fov_patch_tensor.unsqueeze(0)
        
        # Mask out vessel labels outside FOV
        vessel_patch = vessel_patch * (fov_patch_tensor > 0.5).float()
        
        return image_patch, vessel_patch 