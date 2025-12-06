import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.base_dataset import BaseRetinalDataset, get_valid_transform
from src.models.unet_plus_plus import create_model

def predict():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = create_model()
    model = model.to(device)
    checkpoint = torch.load('models/stage4_final_model.pth', map_location=device)
    
    # Process state dict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key in state_dict:
        if key.startswith('base_model.'):
            new_key = key.replace('base_model.', '')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
            
    model.load_state_dict(new_state_dict)
    best_dice = checkpoint.get('best_dice', 0.0)
    print(f'Loading model with best validation Dice score: {best_dice:.4f}')
    
    model.eval()
    
    # Test dataset paths
    test_images_dir = 'data/DRIVE/test/images'
    test_masks_dir = 'data/DRIVE/test/mask'  # FOV masks
    
    # Create output directory
    os.makedirs('results/predictions/drive_test', exist_ok=True)
    
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
    
    try:
        # Create predictions
        with torch.no_grad():
            for idx, (image, mask) in enumerate(tqdm(test_loader, desc='Creating predictions')):
                try:
                    image = image.to(device)
                    
                    # Predict
                    output = model(image)
                    output = torch.sigmoid(output)
                    
                    # Convert to numpy and apply threshold
                    pred = (output.cpu().numpy() > 0.5).astype(np.uint8)
                    pred = pred[0, 0]  # Extract image from batch
                    
                    # Resize to original DRIVE size (584x565)
                    pred = cv2.resize(pred, (565, 584), interpolation=cv2.INTER_LINEAR)
                    pred = (pred > 0.5).astype(np.uint8) * 255  # Convert to binary and scale to 0-255
                    
                    # Apply FOV mask
                    fov_mask = mask.numpy()[0, 0]
                    fov_mask = cv2.resize(fov_mask, (565, 584), interpolation=cv2.INTER_NEAREST)
                    fov_mask = (fov_mask > 0).astype(np.uint8)
                    pred = pred * fov_mask
                    
                    # Save prediction
                    output_path = f'results/predictions/drive_test/{idx+1:02d}_prediction.png'
                    cv2.imwrite(output_path, pred)
                    print(f'Saved prediction {idx+1:02d} to {output_path}')
                    
                except Exception as e:
                    print(f"Error processing image {idx+1}: {str(e)}")
                    continue
        
        print('\nAll predictions have been saved to the results/predictions/drive_test directory')
        print('Please submit these predictions to the DRIVE challenge website for evaluation')
        
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    predict() 