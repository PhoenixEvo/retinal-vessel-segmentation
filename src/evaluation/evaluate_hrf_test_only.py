import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unet_plus_plus import create_model, dice_coefficient

def patch_based_inference(model, image, patch_size=512, overlap=128, device='cuda'):
    """Patch-based inference matching the training methodology"""
    h, w = image.shape[:2]
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    step = patch_size - overlap
    
    # Generate patches
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Handle edges
    if w % step != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    if h % step != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    count_map[count_map == 0] = 1
    prediction = prediction / count_map
    return prediction

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dice = checkpoint.get('best_dice', 0.0)
        training_info = checkpoint.get('training_sequence', [])
    else:
        model.load_state_dict(checkpoint)
        best_dice = 0.0
        training_info = []
    
    model.to(device)
    model.eval()
    return model, device, best_dice, training_info

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    dice = dice_coefficient(torch.tensor(y_true_flat, dtype=torch.float32), 
                           torch.tensor(y_pred_flat, dtype=torch.float32)).item()
    
    intersection = np.logical_and(y_true_flat, y_pred_flat).sum()
    union = np.logical_or(y_true_flat, y_pred_flat).sum()
    iou = intersection / union if union > 0 else 0
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_coefficient': dice,
        'iou': iou,
        'specificity': specificity,
        'confusion_matrix': cm
    }

def get_hrf_train_test_split():
    """Get the same train/test split used during training"""
    images_dir = 'data/HRF/images'
    
    image_files = []
    for ext in ['.jpg', '.JPG', '.png', '.tif']:
        image_files.extend([f for f in os.listdir(images_dir) if f.endswith(ext)])
    image_files = sorted(image_files)
    
    # Use SAME split as training (important: same random_state=42)
    train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    return train_files, test_files

def evaluate_hrf_test_only():
    """Evaluate model ONLY on HRF test set (20% held out during training)"""
    print("HRF TEST SET EVALUATION (PATCH-BASED)")
    print("="*50)
    
    # Check for available models
    model_paths = [
        'models/stage4_final_model.pth',
        'models/stage3_extended_model.pth',
        'models/stage2_patch_finetuned.pth',
        'sequential_finetuned_models/best_model_sequential_hrf_extended_drive.pth',
        'best_model.pth'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("[ERROR] No trained model found!")
        return
    
    print(f"[INFO] Using model: {model_path}")
    
    # Load model
    model, device, best_dice, training_info = load_model(model_path)
    print(f"[SUCCESS] Model loaded successfully!")
    if training_info:
        print(f"Training sequence: {' -> '.join(training_info)}")
    if best_dice > 0:
        print(f"Best training Dice: {best_dice:.6f}")
    
    # HRF dataset paths
    images_dir = 'data/HRF/images'
    masks_dir = 'data/HRF/manual1'
    
    if not os.path.exists(images_dir):
        print(f"[ERROR] HRF images directory not found: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"[ERROR] HRF masks directory not found: {masks_dir}")
        return
    
    # Get train/test split (SAME as training)
    train_files, test_files = get_hrf_train_test_split()
    
    print(f"\n[DATASET] Dataset split (same as training):")
    print(f"   - Total images: {len(train_files) + len(test_files)}")
    print(f"   - Train images: {len(train_files)} (80%)")
    print(f"   - Test images: {len(test_files)} (20%)")
    print(f"\n[TEST] Testing ONLY on test set:")
    for i, f in enumerate(test_files, 1):
        print(f"   {i:2d}. {f}")
    
    # Parameters
    patch_size = 512
    overlap = 128
    threshold = 0.5
    
    print(f"\n[SETTINGS] Patch-based Inference Settings:")
    print(f"   - Patch size: {patch_size}×{patch_size}")
    print(f"   - Overlap: {overlap} pixels")
    print(f"   - Threshold: {threshold}")
    
    # Results storage
    all_metrics = []
    category_metrics = {'healthy': [], 'glaucoma': [], 'dr': []}
    all_confusion_matrices = []
    
    print(f"\n[RUNNING] Starting TEST SET evaluation...")
    print("-" * 50)
    
    for idx, image_file in enumerate(tqdm(test_files, desc="Test Evaluation")):
        try:
            # Load image
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load ground truth mask
            mask_file = image_file.replace('.jpg', '.tif').replace('.png', '.tif')
            mask_path = os.path.join(masks_dir, mask_file)
            
            if not os.path.exists(mask_path):
                base_name = os.path.splitext(image_file)[0]
                possible_masks = [f for f in os.listdir(masks_dir) if base_name in f]
                if possible_masks:
                    mask_path = os.path.join(masks_dir, possible_masks[0])
                else:
                    print(f"[ERROR] No mask found for {image_file}")
                    continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[ERROR] Failed to load mask for {image_file}")
                continue
            
            # Patch-based prediction
            prediction = patch_based_inference(model, image, patch_size, overlap, device)
            
            # Binarize
            mask_binary = (mask > 127).astype(np.uint8)
            pred_binary = (prediction > threshold).astype(np.uint8)
            
            # Calculate metrics
            metrics = calculate_metrics(mask_binary, pred_binary)
            metrics['image_name'] = image_file
            metrics['image_size'] = f"{image.shape[1]}×{image.shape[0]}"
            
            # Determine category
            if '_h.' in image_file or '_h_' in image_file:
                category = 'healthy'
            elif '_g.' in image_file or '_g_' in image_file:
                category = 'glaucoma'
            elif '_dr.' in image_file or '_dr_' in image_file:
                category = 'dr'
            else:
                category = 'unknown'
            
            metrics['category'] = category
            all_metrics.append(metrics)
            all_confusion_matrices.append(metrics['confusion_matrix'])
            
            if category in category_metrics:
                category_metrics[category].append(metrics)
            
            print(f"   [OK] {image_file}: Dice={metrics['dice_coefficient']:.4f}, Acc={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Error processing {image_file}: {str(e)}")
            continue
    
    # Calculate overall statistics
    print(f"\n[RESULTS] HRF TEST SET RESULTS")
    print("="*50)
    
    if not all_metrics:
        print("[ERROR] No successful evaluations!")
        return
    
    # Overall metrics
    overall_metrics = {}
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_coefficient', 'iou', 'specificity']:
        values = [m[metric_name] for m in all_metrics]
        overall_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"[OVERALL] OVERALL TEST PERFORMANCE (n={len(all_metrics)} test images):")
    print(f"   Accuracy:         {overall_metrics['accuracy']['mean']:.4f} ± {overall_metrics['accuracy']['std']:.4f}")
    print(f"   Precision:        {overall_metrics['precision']['mean']:.4f} ± {overall_metrics['precision']['std']:.4f}")
    print(f"   Recall:           {overall_metrics['recall']['mean']:.4f} ± {overall_metrics['recall']['std']:.4f}")
    print(f"   Specificity:      {overall_metrics['specificity']['mean']:.4f} ± {overall_metrics['specificity']['std']:.4f}")
    print(f"   F1-Score:         {overall_metrics['f1_score']['mean']:.4f} ± {overall_metrics['f1_score']['std']:.4f}")
    print(f"   Dice Coefficient: {overall_metrics['dice_coefficient']['mean']:.4f} ± {overall_metrics['dice_coefficient']['std']:.4f}")
    print(f"   IoU (Jaccard):    {overall_metrics['iou']['mean']:.4f} ± {overall_metrics['iou']['std']:.4f}")
    
    # Category-wise metrics
    print(f"\n[CATEGORY] CATEGORY-WISE TEST PERFORMANCE:")
    for category, metrics_list in category_metrics.items():
        if metrics_list:
            category_means = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_coefficient', 'iou', 'specificity']:
                values = [m[metric_name] for m in metrics_list]
                category_means[metric_name] = np.mean(values)
            
            category_label = {'healthy': '[HEALTHY]', 'glaucoma': '[GLAUCOMA]', 'dr': '[DR]'}.get(category, '[UNKNOWN]')
            print(f"   {category_label} {category.upper()} (n={len(metrics_list)}):")
            print(f"      Accuracy: {category_means['accuracy']:.4f}")
            print(f"      Dice:     {category_means['dice_coefficient']:.4f}")
            print(f"      F1-Score: {category_means['f1_score']:.4f}")
    
    # Pixel-level metrics
    if all_confusion_matrices:
        overall_cm = sum(all_confusion_matrices)
        tn, fp, fn, tp = overall_cm.ravel()
        pixel_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        pixel_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        pixel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\n[PIXEL] PIXEL-LEVEL TEST ANALYSIS:")
        print(f"   True Positives:  {tp:,}")
        print(f"   True Negatives:  {tn:,}")
        print(f"   False Positives: {fp:,}")
        print(f"   False Negatives: {fn:,}")
        print(f"   Sensitivity (Recall): {pixel_sensitivity:.4f}")
        print(f"   Specificity:         {pixel_specificity:.4f}")
        print(f"   Precision:           {pixel_precision:.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_csv = results_df.drop('confusion_matrix', axis=1)
    results_csv.to_csv('hrf_test_evaluation_results.csv', index=False)
    
    # Save summary
    with open('hrf_test_evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write("HRF TEST SET EVALUATION SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test images evaluated: {len(all_metrics)}/{len(test_files)}\n\n")
        
        f.write("TEST SET FILES:\n")
        for i, file in enumerate(test_files, 1):
            f.write(f"{i:2d}. {file}\n")
        f.write("\n")
        
        f.write("OVERALL TEST METRICS:\n")
        for metric_name in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'dice_coefficient', 'iou']:
            mean_val = overall_metrics[metric_name]['mean']
            std_val = overall_metrics[metric_name]['std']
            f.write(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}\n")
    
    print(f"\n[SAVED] Results saved:")
    print(f"   - hrf_test_evaluation_results.csv")
    print(f"   - hrf_test_evaluation_summary.txt")
    print(f"\n[SUCCESS] HRF TEST SET evaluation completed!")
    print(f"[NOTE] This evaluation used ONLY the {len(test_files)} test images held out during training.")
    
    return overall_metrics, category_metrics

if __name__ == "__main__":
    evaluate_hrf_test_only() 