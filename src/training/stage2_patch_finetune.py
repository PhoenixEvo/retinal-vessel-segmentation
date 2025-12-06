import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.hrf_patch_dataset import HRFPatchDataset, get_patch_train_transform, get_patch_valid_transform
from src.models.loss_functions import DiceBCELoss
from src.models.unet_plus_plus import dice_coefficient

def progressive_unfreeze(model, epoch, total_epochs):
    """Progressive unfreezing strategy"""
    unfreeze_stages = {
        0: [],  # All frozen
        total_epochs // 3: ['encoder.layer4', 'encoder.bn2'],  # Unfreeze last layers
        2 * total_epochs // 3: ['encoder.layer3', 'encoder.layer4', 'encoder.bn2', 'encoder.bn1'],  # More layers
    }
    
    # Check if we need to unfreeze
    for stage_epoch, layers_to_unfreeze in unfreeze_stages.items():
        if epoch == stage_epoch and layers_to_unfreeze:
            print(f"\nEpoch {epoch}: Unfreezing layers: {layers_to_unfreeze}")
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"Unfroze {unfrozen_count} parameters")
            return True
    return False

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    """Train for one epoch with AMP"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch_data in enumerate(progress_bar):
        images, vessel_masks, fov_masks = batch_data
        images = images.to(device, non_blocking=True)
        vessel_masks = vessel_masks.to(device, non_blocking=True)
        fov_masks = fov_masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            # Apply FOV mask
            outputs = outputs * fov_masks
            vessel_masks = vessel_masks * fov_masks
            
            # Calculate loss
            loss = criterion(outputs, vessel_masks)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, vessel_masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{dice.item():.4f}',
            'Scale': f'{scaler.get_scale():.0f}' if scaler else 'N/A'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def validate_epoch(model, dataloader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch_data in enumerate(progress_bar):
            images, vessel_masks, fov_masks = batch_data
            images = images.to(device, non_blocking=True)
            vessel_masks = vessel_masks.to(device, non_blocking=True)
            fov_masks = fov_masks.to(device, non_blocking=True)
            
            # Forward pass with autocast for consistency
            with autocast():
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                
                # Apply FOV mask
                outputs = outputs * fov_masks
                vessel_masks = vessel_masks * fov_masks
                
                # Calculate loss and metrics
                loss = criterion(outputs, vessel_masks)
                dice = dice_coefficient(outputs, vessel_masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice

def setup_hrf_patch_dataset(hrf_dir="HRF", patch_size=512, overlap=128):
    """Setup HRF patch dataset"""
    images_dir = os.path.join(hrf_dir, "images")
    masks_dir = os.path.join(hrf_dir, "manual1")
    fov_dir = os.path.join(hrf_dir, "mask")
    
    # Check if directories exist
    for dir_path, name in [(images_dir, "images"), (masks_dir, "manual1"), (fov_dir, "mask")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Get all image files for splitting
    image_files = []
    for ext in ['.jpg', '.JPG', '.png', '.tif']:
        image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} HRF images")
    
    # Split into train and validation at IMAGE level (not patch level)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Create train dataset with train images only
    train_dataset = HRFPatchDataset(
        images_dir, masks_dir, fov_dir,
        patch_size=patch_size, overlap=overlap,
        transform=get_patch_train_transform()
    )
    
    # Filter train patches to only include train images
    train_patches = [p for p in train_dataset.patches_info if p['image_file'] in train_files]
    train_dataset.patches_info = train_patches
    
    # Create validation dataset with val images only
    val_dataset = HRFPatchDataset(
        images_dir, masks_dir, fov_dir,
        patch_size=patch_size, overlap=overlap,
        transform=get_patch_valid_transform()
    )
    
    # Filter val patches to only include val images
    val_patches = [p for p in val_dataset.patches_info if p['image_file'] in val_files]
    val_dataset.patches_info = val_patches
    
    print(f"Train images: {len(train_files)} -> {len(train_dataset)} patches")
    print(f"Val images: {len(val_files)} -> {len(val_dataset)} patches")
    
    return train_dataset, val_dataset

def main():
    print("🔥 Patch-based Fine-tuning U-Net++ on HRF Dataset")
    print("Features: High-Resolution Patches + AMP + Progressive Unfreezing")
    print("=" * 70)
    
    # Check CUDA and AMP availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        amp_enabled = True
    else:
        print("CUDA not available, AMP disabled")
        amp_enabled = False
    
    # Model configuration
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    
    # Load pretrained weights
    pretrained_path = "models/stage1_base_model.pth"
    if os.path.exists(pretrained_path):
        print(f"\n📂 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with Dice score: {checkpoint.get('best_dice', 'unknown')}")
    else:
        print(f"Error: Pretrained model {pretrained_path} not found!")
        return
    
    model = model.to(device)
    
    # Initial freeze (will be progressively unfrozen)
    print("\n🔒 Initial freezing of encoder...")
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            frozen_count += 1
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Training configuration  
    patch_size = 512
    overlap = 128
    batch_size = 8 if amp_enabled else 4  # Can use larger batch with patches
    learning_rate = 5e-5
    num_epochs = 30
    
    criterion = DiceBCELoss()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Advanced scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if amp_enabled else None
    
    # Setup dataset
    try:
        train_dataset, val_dataset = setup_hrf_patch_dataset(
            hrf_dir="data/HRF", patch_size=patch_size, overlap=overlap
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Can use more workers with patches
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    print(f"\nTraining Configuration:")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Overlap: {overlap}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Optimizer: AdamW with weight decay")
    print(f"Scheduler: CosineAnnealingWarmRestarts")
    
    # Training loop
    best_dice = 0
    train_losses, train_dices = [], []
    val_losses, val_dices = [], []
    
    print(f"\n🏃‍♂Starting patch-based training...")
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Progressive unfreezing
        if progressive_unfreeze(model, epoch, num_epochs):
            # Update optimizer with newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=learning_rate,
                weight_decay=1e-4
            )
            # Update scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=8, T_mult=2, eta_min=1e-7
            )
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch + 1
        )
        
        # Validate
        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            model_name = "models/stage2_patch_finetuned.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_dice': best_dice,
                'pretrained_from': pretrained_path,
                'config': {
                    'patch_size': patch_size,
                    'overlap': overlap,
                    'amp_enabled': amp_enabled,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                }
            }, model_name)
            print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Early stopping if Dice > 0.8
        if val_dice > 0.8:
            print(f"Target achieved! Dice score: {val_dice:.4f} > 0.8")
            break
    
    print(f"\n🏁 Patch-based training completed!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Model saved as: models/stage2_patch_finetuned.pth")

if __name__ == "__main__":
    main() 