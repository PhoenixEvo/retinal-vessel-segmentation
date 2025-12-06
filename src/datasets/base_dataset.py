import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class BaseRetinalDataset(Dataset):
    """
    Base class for retinal vessel datasets
    """
    def __init__(self, images_dir, masks_dir, fov_masks_dir=None, transform=None):
        """
        Base Retinal Vessel Dataset
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing vessel ground truth masks
            fov_masks_dir: Directory containing field of view (FOV) masks (optional)
            transform: Albumentations transforms
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.fov_masks_dir = fov_masks_dir
        self.transform = transform
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(('.gif', '.png', '.tif'))])
        
        # FOV masks (optional)
        self.fov_masks = None
        if fov_masks_dir and os.path.exists(fov_masks_dir):
            self.fov_masks = sorted([os.path.join(fov_masks_dir, f) for f in os.listdir(fov_masks_dir) if f.endswith(('.gif', '.png', '.tif'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read vessel mask (ground truth)
        if mask_path.endswith('.gif'):
            # Use PIL to read GIF files
            mask = Image.open(mask_path)
            mask = np.array(mask)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask at {mask_path}")
        
        # Normalize vessel mask to 0 and 1
        vessel_mask = mask.astype(np.float32) / 255.0
        
        # Read FOV mask if available
        fov_mask = None
        if self.fov_masks and idx < len(self.fov_masks):
            fov_path = self.fov_masks[idx]
            if fov_path.endswith('.gif'):
                fov_mask = Image.open(fov_path)
                fov_mask = np.array(fov_mask)
            else:
                fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
            
            if fov_mask is not None:
                fov_mask = fov_mask.astype(np.float32) / 255.0
        
        if self.transform:
            if fov_mask is not None:
                # Apply transform to both vessel and FOV masks
                combined_mask = np.stack([vessel_mask, fov_mask], axis=2)
                augmented = self.transform(image=image, mask=combined_mask)
                image = augmented['image']
                combined_mask = augmented['mask']
                
                # Split back to separate masks
                vessel_mask = combined_mask[:, :, 0] if len(combined_mask.shape) == 3 else combined_mask
                fov_mask = combined_mask[:, :, 1] if len(combined_mask.shape) == 3 else fov_mask
            else:
                # Apply transform to vessel mask only
                augmented = self.transform(image=image, mask=vessel_mask)
                image = augmented['image']
                vessel_mask = augmented['mask']
            
        # Add channel dimension for vessel mask if needed
        if isinstance(vessel_mask, np.ndarray):
            vessel_mask = torch.from_numpy(vessel_mask)
        if len(vessel_mask.shape) == 2:
            vessel_mask = vessel_mask.unsqueeze(0)
        
        # Add channel dimension for FOV mask if needed
        if fov_mask is not None:
            if isinstance(fov_mask, np.ndarray):
                fov_mask = torch.from_numpy(fov_mask)
            if len(fov_mask.shape) == 2:
                fov_mask = fov_mask.unsqueeze(0)
            
            return image, vessel_mask, fov_mask
        else:
            return image, vessel_mask

def get_train_transform():
    """Standard training transforms"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            translate_percent=0.0625,
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.5
            ),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(
                distort_limit=1,
                p=0.5
            ),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=0,
            p=0.3
        ),
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_valid_transform():
    """Standard validation transforms"""
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]) 