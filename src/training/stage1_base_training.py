import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.base_dataset import BaseRetinalDataset, get_train_transform, get_valid_transform
from src.models.unet_plus_plus import create_model, dice_coefficient
from src.models.loss_functions import DiceBCELoss

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    best_valid_dice = 0.0
    train_losses = []
    valid_dices = []
    patience = 15
    patience_counter = 0
    
    scaler = torch.amp.GradScaler('cuda')  # Updated new API
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for images, masks in pbar:
                try:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Mixed precision training with new API
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    pbar.set_postfix({'loss': loss.item()})
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
        
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            valid_dice = 0
            valid_count = 0
            
            with torch.no_grad():
                for images, masks in valid_loader:
                    try:
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        outputs = model(images)
                        outputs = torch.sigmoid(outputs)
                        valid_dice += dice_coefficient(masks, outputs)
                        valid_count += 1
                    except Exception as e:
                        print(f"Error in validation: {e}")
                        continue
            
            if valid_count > 0:
                avg_valid_dice = valid_dice / valid_count
                valid_dices.append(avg_valid_dice)
                
                print(f'Epoch {epoch+1}:')
                print(f'Average Training Loss: {avg_train_loss:.4f}')
                print(f'Average Validation Dice: {avg_valid_dice:.4f}')
                
                # Learning rate scheduling
                scheduler.step(avg_valid_dice)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current Learning Rate: {current_lr:.2e}')
                
                # Early stopping
                if avg_valid_dice > best_valid_dice:
                    best_valid_dice = avg_valid_dice
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_dice': best_valid_dice,
                    }, save_path)
                    print(f'Model saved with Dice score: {best_valid_dice:.4f}')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'Early stopping after {patience} epochs without improvement')
                    break
    
    return train_losses, valid_dices

def main():
    # Setup device and seed
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets for DRIVE only (Phase 1: Base Training on DRIVE)
    # Get the correct path relative to project root
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    drive_train_path = os.path.join(project_root, 'data', 'DRIVE', 'training', 'images')
    drive_train_masks = os.path.join(project_root, 'data', 'DRIVE', 'training', '1st_manual')
    drive_test_path = os.path.join(project_root, 'data', 'DRIVE', 'test', 'images')
    drive_test_masks = os.path.join(project_root, 'data', 'DRIVE', 'test', '1st_manual')
    
    print(f"Looking for DRIVE training data at: {os.path.abspath(drive_train_path)}")
    
    drive_train_dataset = BaseRetinalDataset(
        drive_train_path,
        drive_train_masks,
        transform=get_train_transform()
    )
    
    # Use DRIVE test set for validation (if available) or split train set
    try:
        drive_test_dataset = BaseRetinalDataset(
            drive_test_path,
            drive_test_masks,
            transform=get_valid_transform()
        )
        train_dataset = drive_train_dataset
        valid_dataset = drive_test_dataset
        print(f'Using DRIVE train/test split')
        print(f'Training set size: {len(train_dataset)}')
        print(f'Validation set size: {len(valid_dataset)}')
    except:
        # If test set not available, split training set
        train_size = int(0.8 * len(drive_train_dataset))
        valid_size = len(drive_train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            drive_train_dataset, 
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f'Using DRIVE training set split')
        print(f'Total DRIVE dataset size: {len(drive_train_dataset)}')
        print(f'Training set size: {len(train_dataset)}')
        print(f'Validation set size: {len(valid_dataset)}')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model and optimizer
    model = create_model()
    model = model.to(device)
    
    # Don't load pretrained weights for fresh Phase 1 training
    print('Starting fresh Phase 1 training on DRIVE dataset...')
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training
    num_epochs = 200
    save_path = os.path.join(models_dir, 'stage1_base_model.pth')
    
    # Create models and results directories if they don't exist
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*60)
    print("PHASE 1: BASE TRAINING ON DRIVE DATASET")
    print("="*60)
    print(f"Dataset: DRIVE only ({len(train_dataset)} train, {len(valid_dataset)} val)")
    print(f"Model: UNet++ with EfficientNet-B4")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: 3e-4")
    print(f"Loss: DiceBCELoss")
    print(f"Purpose: Foundation knowledge on DRIVE")
    print("="*60)
    
    train_losses, valid_dices = train_model(
        model, train_loader, valid_loader,
        criterion, optimizer, scheduler, num_epochs,
        device, save_path
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Phase 1 - Training Loss (DRIVE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(valid_dices)
    plt.title('Phase 1 - Validation Dice (DRIVE)')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stage1_training_history.png'))
    plt.close()
    
    print("Phase 1 training completed!")
    print(f"Model saved to: {save_path}")
    print("Ready for Phase 2: HRF Patch Fine-tuning")

if __name__ == '__main__':
    main() 