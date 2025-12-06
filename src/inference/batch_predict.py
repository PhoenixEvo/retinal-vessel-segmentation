import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from model import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def patch_based_inference(model, image, patch_size=384, overlap=64, device='cuda'):
    """
    Patch-based inference for high-resolution images
    """
    model.eval()
    
    transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    h, w = image.shape[:2]
    stride = patch_size - overlap
    
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    with torch.no_grad():
        # Regular patches
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
                
                output = model(patch_tensor)
                pred_patch = torch.sigmoid(output).cpu().numpy().squeeze()
                
                prediction[y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[y:y+patch_size, x:x+patch_size] += 1
        
        # Handle remaining edges
        if w % stride != 0:
            for y in range(0, h - patch_size + 1, stride):
                x = w - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
                output = model(patch_tensor)
                pred_patch = torch.sigmoid(output).cpu().numpy().squeeze()
                prediction[y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[y:y+patch_size, x:x+patch_size] += 1
        
        if h % stride != 0:
            for x in range(0, w - patch_size + 1, stride):
                y = h - patch_size
                patch = image[y:y+patch_size, x:x+patch_size]
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
                output = model(patch_tensor)
                pred_patch = torch.sigmoid(output).cpu().numpy().squeeze()
                prediction[y:y+patch_size, x:x+patch_size] += pred_patch
                count_map[y:y+patch_size, x:x+patch_size] += 1
        
        if h % stride != 0 and w % stride != 0:
            y, x = h - patch_size, w - patch_size
            patch = image[y:y+patch_size, x:x+patch_size]
            transformed = transform(image=patch)
            patch_tensor = transformed['image'].unsqueeze(0).to(device)
            output = model(patch_tensor)
            pred_patch = torch.sigmoid(output).cpu().numpy().squeeze()
            prediction[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    prediction = prediction / np.maximum(count_map, 1)
    return prediction

def visualize_prediction(image, prediction, gt_mask=None, save_path=None, title="Prediction"):
    """Visualize original image, prediction, and ground truth"""
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    # Ground truth if available
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
    
    plt.show()

def find_corresponding_mask(image_file):
    """Find corresponding mask file for HRF image"""
    base_name = os.path.splitext(image_file)[0]
    
    # Possible mask patterns
    possible_masks = [
        f"{base_name}.tif",
        f"{base_name}.png", 
        f"{base_name}.jpg",
        f"{base_name}.gif"
    ]
    
    masks_dir = "HRF/manual1"
    for mask_name in possible_masks:
        mask_path = os.path.join(masks_dir, mask_name)
        if os.path.exists(mask_path):
            return mask_name
    
    return None

def predict_hrf():
    """
    Generate predictions for HRF dataset using Sequential Extended Model
    """
    print("HRF DATASET PREDICTION - SEQUENTIAL EXTENDED MODEL")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'📱 Using device: {device}')
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'🧹 Cleared CUDA cache')
    
    # Load Sequential Extended Model
    model_path = 'sequential_finetuned_models/best_model_sequential_hrf_extended_drive.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please ensure the sequential fine-tuning has been completed!")
        return
    
    print(f'📂 Loading Sequential Extended Model: {model_path}')
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Model info
        training_sequence = checkpoint.get('training_sequence', ['unknown'])
        best_dice = checkpoint.get('best_dice', 0.0)
        config = checkpoint.get('config', {})
        
        print(f'Model loaded successfully!')
        print(f'Training sequence: {" → ".join(training_sequence)}')
        print(f'Best validation Dice: {best_dice:.6f}')
        print(f'Training config: {config}')
    else:
        model.load_state_dict(checkpoint)
        print(f'Model weights loaded!')
    
    model.to(device)
    model.eval()
    
    # Dataset paths
    images_dir = 'HRF/images'
    masks_dir = 'HRF/manual1'
    
    if not os.path.exists(images_dir):
        print(f"HRF images directory not found: {images_dir}")
        return
    
    # Create output directory
    output_dir = 'hrf_predictions_sequential'
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f'📁 Output directory: {output_dir}')
    
    # Get HRF images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = sorted(image_files)
    print(f'Found {len(image_files)} HRF images')
    
    # Prediction parameters
    patch_size = 384  # Same as training
    overlap = 64      # Same as training
    threshold = 0.5   # Binary threshold
    
    print(f'Prediction settings:')
    print(f'   - Patch size: {patch_size}x{patch_size}')
    print(f'   - Overlap: {overlap}')
    print(f'   - Threshold: {threshold}')
    
    print(f'\nStarting prediction process...')
    print("-" * 70)
    
    success_count = 0
    
    # Process each HRF image
    for idx, image_file in enumerate(tqdm(image_files, desc="Predicting")):
        try:
            # Load image
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_file}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = image.shape[:2]
            
            # Load ground truth mask if available
            mask_file = find_corresponding_mask(image_file)
            gt_mask = None
            if mask_file:
                mask_path = os.path.join(masks_dir, mask_file)
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = gt_mask.astype(np.float32) / 255.0
            
            # Patch-based prediction
            prediction = patch_based_inference(
                model, image, 
                patch_size=patch_size, 
                overlap=overlap, 
                device=device
            )
            
            # Apply threshold
            pred_binary = (prediction > threshold).astype(np.uint8)
            pred_visual = pred_binary * 255
            
            # Save prediction
            base_name = os.path.splitext(image_file)[0]
            output_filename = f'{base_name}_prediction.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, pred_visual)
            
            # Create visualization
            vis_filename = f'{base_name}_visualization.png'
            vis_path = os.path.join(vis_dir, vis_filename)
            visualize_prediction(
                image, prediction, gt_mask, 
                save_path=vis_path, 
                title=f"HRF Prediction - {image_file}"
            )
            
            print(f'{idx+1:2d}. {image_file} → {output_filename} ({h_orig}x{w_orig})')
            success_count += 1
            
        except Exception as e:
            print(f'Error processing {image_file}: {str(e)}')
            continue
    
    print("\n" + "="*70)
    print(f'🎉 PREDICTION COMPLETED!')
    print(f'Successfully processed: {success_count}/{len(image_files)} images')
    print(f'📁 Predictions saved in: {output_dir}/')
    print(f'🖼Visualizations saved in: {vis_dir}/')
    
    # List output files
    pred_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_prediction.png')])
    print(f'\n📄 Generated prediction files ({len(pred_files)} files):')
    for i, filename in enumerate(pred_files[:10], 1):  # Show first 10
        print(f'   {i:2d}. {filename}')
    if len(pred_files) > 10:
        print(f'   ... and {len(pred_files) - 10} more files')
    
    # Show some sample categories
    print(f'\nHRF Dataset Categories:')
    healthy_count = len([f for f in image_files if '_h.' in f])
    glaucoma_count = len([f for f in image_files if '_g.' in f])
    dr_count = len([f for f in image_files if '_dr.' in f.lower()])
    
    print(f'   Healthy: {healthy_count} images')
    print(f'   Glaucoma: {glaucoma_count} images')
    print(f'   Diabetic Retinopathy: {dr_count} images')
    
    print(f'\nModel Performance:')
    if 'best_dice' in locals():
        print(f'   Training sequence: Original → HRF → Extended → DRIVE')
        print(f'   Best validation Dice: {best_dice:.6f}')
    
    print(f'\n✨ HRF predictions completed successfully!')

if __name__ == '__main__':
    predict_hrf() 