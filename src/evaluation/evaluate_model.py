import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.base_dataset import BaseRetinalDataset, get_valid_transform
from src.models.unet_plus_plus import create_model, dice_coefficient

def evaluate():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = create_model()
    model = model.to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_dice = checkpoint['best_dice']
    print(f'Loading model with best validation Dice score: {best_dice:.4f}')
    
    model.eval()
    
    # Test dataset paths
    test_images_dir = 'DRIVE/test/images'
    test_masks_dir = 'DRIVE/test/mask'
    
    # Create test dataset
    test_dataset = BaseRetinalDataset(
        test_images_dir,
        test_masks_dir,
        transform=get_valid_transform()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create directory for predictions
    os.makedirs('predictions', exist_ok=True)
    
    # Evaluate on test set
    predictions = []
    
    try:
        with torch.no_grad():
            for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
                try:
                    images = images.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    
                    # Create prediction
                    pred = (outputs.cpu().numpy() > 0.5).astype(np.uint8)
                    pred = pred[0, 0] * 255  # Convert to 0-255 range
                    
                    # Apply FOV mask
                    fov_mask = masks.numpy()[0, 0]  # Get mask from tensor
                    fov_mask = (fov_mask > 0).astype(np.uint8)
                    pred = pred * fov_mask
                    
                    # Save prediction
                    cv2.imwrite(f'predictions/{idx+1:02d}_prediction.png', pred)
                    predictions.append(pred)
                    
                except Exception as e:
                    print(f"Error processing image {idx+1}: {str(e)}")
                    continue
                
        print(f'\nPredictions have been saved to the predictions directory')
        print(f'Number of predictions generated: {len(predictions)}')
        print(f'Best validation Dice score during training: {best_dice:.4f}')
        print(f'Please submit these predictions to the DRIVE challenge website for evaluation')
        
    except Exception as e:
        print(f"An error occurred during evaluation: {str(e)}")

if __name__ == '__main__':
    evaluate() 