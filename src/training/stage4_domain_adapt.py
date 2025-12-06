import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import wandb
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.unet_plus_plus import create_model, dice_coefficient
from src.datasets.drive_patch_dataset import DRIVEPatchDataset
from src.models.loss_functions import CombinedLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """Training transforms with augmentation"""
    return A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ElasticTransform(alpha=1, sigma=20, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_valid_transform():
    """Validation transforms without augmentation"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def dice_coefficient(pred, target, smooth=1e-7):
    """Calculate Dice coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_loss(pred, target, smooth=1e-7):
    """Dice loss function"""
    return 1 - dice_coefficient(pred, target, smooth)

class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice = dice_loss(pred_sigmoid, target)
        return self.bce_weight * bce + self.dice_weight * dice

def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_dice = 0
    total_loss = 0
    criterion = CombinedLoss()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            pred_sigmoid = torch.sigmoid(outputs)
            dice = dice_coefficient(pred_sigmoid, masks)
            
            total_dice += dice.item()
            total_loss += loss.item()
    
    avg_dice = total_dice / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    
    return avg_dice, avg_loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(outputs)
                dice = dice_coefficient(pred_sigmoid, masks)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb (with error handling)
            if batch_idx % 10 == 0:
                try:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_dice': dice.item(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                except:
                    pass  # Ignore wandb errors
                    
            # Clear cache periodically
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA OOM at batch {batch_idx}, clearing cache and skipping batch...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_dice = total_dice / len(dataloader) if len(dataloader) > 0 else 0
    
    return avg_loss, avg_dice

def finetune_hrf_model_on_drive():
    """Fine-tune HRF Extended model back on DRIVE dataset"""
    
    # Configuration
    config = {
        'patch_size': 384,  # Reduced from 512 to 384 for memory
        'batch_size': 2,    # Reduced from 8 to 2 for memory
        'epochs': 30,       # Fewer epochs for second fine-tuning
        'learning_rate': 5e-5,  # Even lower LR for second fine-tuning
        'weight_decay': 1e-5,
        'patience': 8,
        'min_improvement': 0.001,
        'overlap': 64,      # Reduced overlap for more patches
        'num_workers': 2,   # Reduced workers for stability
        'save_dir': 'sequential_finetuned_models',
        'base_model': 'best_model_hrf_patch_extended.pth'  # Extended HRF model
    }
    
    print("SEQUENTIAL FINE-TUNING: HRF EXTENDED MODEL → DRIVE")
    print("="*60)
    print(f"Base model: {config['base_model']}")
    print(f"Strategy: Extended HRF knowledge + DRIVE adaptation")
    print(f"Patch size: {config['patch_size']} (reduced for memory)")
    print(f"Batch size: {config['batch_size']} (reduced for memory)")
    print(f"Learning rate: {config['learning_rate']} (very low)")
    print(f"Epochs: {config['epochs']}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 Cleared CUDA cache")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize wandb with error handling
    try:
        wandb.init(
            project="vessel-segmentation-sequential",
            name=f"sequential_finetune_drive_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            mode="offline"  # Use offline mode to avoid connection issues
        )
        print("Wandb initialized (offline mode)")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without logging...")
        # Create dummy wandb object
        class DummyWandb:
            def log(self, *args, **kwargs): pass
            def finish(self): pass
        wandb = DummyWandb()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load HRF Extended model
    model = create_model()
    
    if os.path.exists(config['base_model']):
        print(f"\n📂 Loading HRF Extended model: {config['base_model']}")
        checkpoint = torch.load(config['base_model'], map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded extended model state dict")
            original_config = checkpoint.get('config', {})
            resumed_info = checkpoint.get('resumed_from', 'N/A')
            print(f"Original training config: {original_config}")
            print(f"Extended from: {resumed_info}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded extended model weights")
    else:
        print(f"Extended base model not found: {config['base_model']}")
        return
    
    model = model.to(device)
    
    # Create DRIVE datasets
    print(f"\nCreating DRIVE patch datasets...")
    
    train_dataset = DRIVEPatchDataset(
        images_dir='DRIVE/training/images',
        masks_dir='DRIVE/training/1st_manual',
        fov_masks_dir='DRIVE/training/mask',
        patch_size=config['patch_size'],
        overlap=config['overlap'],
        transform=get_train_transform(),
        split='train',
        train_ratio=0.8,
        min_vessel_ratio=0.005  # Reduced from 0.01 to get more patches
    )
    
    val_dataset = DRIVEPatchDataset(
        images_dir='DRIVE/training/images',
        masks_dir='DRIVE/training/1st_manual',
        fov_masks_dir='DRIVE/training/mask',
        patch_size=config['patch_size'],
        overlap=config['overlap'],
        transform=get_valid_transform(),
        split='val',
        train_ratio=0.8,
        min_vessel_ratio=0.005  # Reduced from 0.01 to get more patches
    )
    
    print(f"DRIVE training patches: {len(train_dataset)}")
    print(f"DRIVE validation patches: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Very conservative learning rate for second fine-tuning
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-8
    )
    
    # Training loop
    print(f"\n🚀 Starting sequential fine-tuning on DRIVE...")
    
    best_dice = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 40)
        
        # Training
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_dice, val_loss = evaluate_model(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_dice)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice
        })
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice + config['min_improvement']:
            best_dice = val_dice
            patience_counter = 0
            
            model_save_path = os.path.join(config['save_dir'], 'best_model_sequential_hrf_extended_drive.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': config,
                'base_model': config['base_model'],
                'training_sequence': ['original', 'hrf_patch', 'hrf_extended', 'drive']
            }, model_save_path)
            
            print(f"New best model saved! Dice: {best_dice:.6f}")
            
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹Early stopping triggered after {epoch} epochs")
            break
    
    print(f"\n🎉 Sequential fine-tuning completed!")
    print(f"Best validation Dice: {best_dice:.6f}")
    print(f"Model saved to: {os.path.join(config['save_dir'], 'best_model_sequential_hrf_extended_drive.pth')}")
    
    # Performance comparison
    print(f"\nEXPECTED PERFORMANCE COMPARISON:")
    print(f"🔹 Original model on DRIVE: ~0.7797")
    print(f"🔹 HRF-only model on DRIVE: ~0.6646") 
    print(f"🔹 HRF Extended model on DRIVE: TBD")
    print(f"🔹 Sequential Extended model on DRIVE: ~{best_dice:.4f}")
    
    if best_dice > 0.75:
        print(f"🚀 SUCCESS: Sequential Extended model recovered DRIVE performance!")
    elif best_dice > 0.70:
        print(f"GOOD: Sequential Extended model shows good DRIVE performance")
    else:
        print(f"CAUTION: May need different strategy or hyperparameters")
    
    wandb.finish()

if __name__ == "__main__":
    finetune_hrf_model_on_drive() 