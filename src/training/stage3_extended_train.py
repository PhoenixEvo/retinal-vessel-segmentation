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
        with autocast('cuda'):
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
            with autocast('cuda'):
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
    
    # Get all image files for splitting
    image_files = []
    for ext in ['.jpg', '.JPG', '.png', '.tif']:
        image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
    image_files = sorted(image_files)
    
    # Use same split as previous training
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = HRFPatchDataset(
        images_dir, masks_dir, fov_dir,
        patch_size=patch_size, overlap=overlap,
        transform=get_patch_train_transform()
    )
    
    val_dataset = HRFPatchDataset(
        images_dir, masks_dir, fov_dir,
        patch_size=patch_size, overlap=overlap,
        transform=get_patch_valid_transform()
    )
    
    # Filter patches 
    train_patches = [p for p in train_dataset.patches_info if p['image_file'] in train_files]
    train_dataset.patches_info = train_patches
    
    val_patches = [p for p in val_dataset.patches_info if p['image_file'] in val_files]
    val_dataset.patches_info = val_patches
    
    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")
    
    return train_dataset, val_dataset

def main():
    print("Resuming Patch-based Training for Higher Dice Score")
    print("Target: Dice > 0.85 (85%)")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        amp_enabled = True
    else:
        amp_enabled = False
    
    # Load checkpoint
    checkpoint_path = "best_model_hrf_patch_finetuned.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    current_dice = checkpoint.get('best_dice', 0)
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    print(f"Current best Dice: {current_dice:.6f}")
    print(f"🚀 Resuming from epoch {start_epoch}")
    
    # Create model
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Configuration
    config = checkpoint.get('config', {})
    patch_size = config.get('patch_size', 512)
    overlap = config.get('overlap', 128)
    batch_size = config.get('batch_size', 8)
    learning_rate = config.get('learning_rate', 5e-5)
    
    # Extended training
    additional_epochs = 25
    total_epochs = start_epoch + additional_epochs
    
    print(f"\nTraining Configuration:")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Additional epochs: {additional_epochs}")
    print(f"New target: Dice > 0.85")
    
    # Setup training components
    criterion = DiceBCELoss()
    
    # Check current parameter status
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nCurrent trainable parameters: {trainable_params:,}/{total_params:,}")
    
    # Setup optimizer (recreate to ensure all trainable params are included)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        except:
            print("Could not load optimizer state, using fresh optimizer")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-7
    )
    
    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        except:
            print("Could not load scheduler state, using fresh scheduler")
    
    # Scaler
    scaler = GradScaler('cuda') if amp_enabled else None
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded")
        except:
            print("Could not load scaler state, using fresh scaler")
    
    # Setup dataset
    train_dataset, val_dataset = setup_hrf_patch_dataset(
        patch_size=patch_size, overlap=overlap
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
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
    
    # Training loop
    best_dice = current_dice
    
    print(f"\n🏃‍♂Resuming training for {additional_epochs} more epochs...")
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"{'='*50}")
        
        # Progressive unfreezing (adjusted for resumed training)
        if progressive_unfreeze(model, epoch - start_epoch, additional_epochs):
            # Update optimizer with newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=learning_rate,
                weight_decay=1e-4
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
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            improvement = val_dice - current_dice
            print(f"New record! Dice: {best_dice:.6f} (+{improvement:.6f})")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_dice': best_dice,
                'pretrained_from': checkpoint.get('pretrained_from', 'unknown'),
                'config': config,
                'resumed_from': checkpoint_path,
                'resumed_at_epoch': start_epoch,
            }, "best_model_hrf_patch_extended.pth")
        
        # Early stopping at higher threshold
        if val_dice > 0.95:
            print(f"Extended target achieved! Dice: {val_dice:.6f} > 0.95")
            break
    
    print(f"\n🏁 Extended training completed!")
    print(f"Final Best Dice Score: {best_dice:.6f}")
    print(f"Improvement: +{best_dice - current_dice:.6f}")
    print(f"Model saved as: best_model_hrf_patch_extended.pth")

if __name__ == "__main__":
    main() 