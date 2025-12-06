import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unet_plus_plus import create_model, dice_coefficient

def patch_based_inference(model, image, patch_size=512, overlap=128, device='cuda'):
    """
    Patch-based inference matching the training methodology
    """
    h, w = image.shape[:2]
    
    # Create output canvas
    prediction = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate step size
    step = patch_size - overlap
    
    # Generate patches
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Preprocess patch
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # Add to prediction map
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Handle edge cases - remaining strips
    # Right edge
    if w % step != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Bottom edge
    if h % step != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, step):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_normalized = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)
            patch_tensor = patch_tensor.to(device)
            
            with torch.no_grad():
                output = model(patch_tensor)
                patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            prediction[y:y+patch_size, x:x+patch_size] += patch_pred
            count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Bottom-right corner
    if w % step != 0 and h % step != 0:
        x, y = w - patch_size, h - patch_size
        patch = image[y:y+patch_size, x:x+patch_size]
        patch_normalized = patch.astype(np.float32) / 255.0
        patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).unsqueeze(0)
        patch_tensor = patch_tensor.to(device)
        
        with torch.no_grad():
            output = model(patch_tensor)
            patch_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        prediction[y:y+patch_size, x:x+patch_size] += patch_pred
        count_map[y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
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
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    # Note: might need to handle edge cases better here
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    # Calculate Dice coefficient
    dice = dice_coefficient(torch.tensor(y_true_flat, dtype=torch.float32), 
                           torch.tensor(y_pred_flat, dtype=torch.float32)).item()
    
    # Calculate IoU (Jaccard Index)
    intersection = np.logical_and(y_true_flat, y_pred_flat).sum()
    union = np.logical_or(y_true_flat, y_pred_flat).sum()
    iou = intersection / union if union > 0 else 0
    
    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'dice_coefficient': dice,
        'iou': iou,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cms, categories, save_path='hrf_confusion_matrices.png'):
    """Plot confusion matrices for different categories"""
    
    # Calculate overall confusion matrix
    overall_cm = sum(cms)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('HRF Dataset - Confusion Matrices', fontsize=16, fontweight='bold')
    
    # Overall confusion matrix
    ax = axes[0, 0]
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-vessel', 'Vessel'], 
                yticklabels=['Non-vessel', 'Vessel'])
    ax.set_title('Overall Performance')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Category-specific confusion matrices
    category_names = ['Healthy', 'Glaucoma', 'Diabetic Retinopathy']
    category_keys = ['healthy', 'glaucoma', 'dr']
    
    for i, (cat_key, cat_name) in enumerate(zip(category_keys, category_names)):
        if cat_key in categories and len(categories[cat_key]) > 0:
            # Sum confusion matrices for this category
            cat_cm = sum([m['confusion_matrix'] for m in categories[cat_key]])
            
            ax = axes[0, 1] if i == 0 else axes[1, i-1]
            sns.heatmap(cat_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Non-vessel', 'Vessel'], 
                        yticklabels=['Non-vessel', 'Vessel'])
            ax.set_title(f'{cat_name} (n={len(categories[cat_key])})')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    # If we don't have all categories, fill the remaining subplot
    if len([k for k in category_keys if k in categories and len(categories[k]) > 0]) < 3:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_cm

def evaluate_hrf_dataset():
    """Evaluate model on HRF dataset using patch-based inference"""
    print("COMPREHENSIVE HRF DATASET EVALUATION (PATCH-BASED)")
    print("="*70)
    
    # Check for available models
    model_paths = [
        'sequential_finetuned_models/best_model_sequential_hrf_extended_drive.pth',
        'models/stage4_final_model.pth',
        'models/stage3_extended_model.pth',
        'best_model.pth'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No trained model found!")
        print("Expected model paths:")
        for path in model_paths:
            print(f"   - {path}")
        return
    
    print(f"📂 Using model: {model_path}")
    
    # Load model
    model, device, best_dice, training_info = load_model(model_path)
    print(f"Model loaded successfully!")
    if training_info:
        print(f"Training sequence: {' → '.join(training_info)}")
    if best_dice > 0:
        print(f"Best training Dice: {best_dice:.6f}")
    
    # HRF dataset paths
    images_dir = 'data/HRF/images'
    masks_dir = 'data/HRF/manual1'
    
    if not os.path.exists(images_dir):
        print(f"HRF images directory not found: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"HRF masks directory not found: {masks_dir}")
        return
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])
    print(f"Found {len(image_files)} HRF images")
    
    # Patch-based parameters (matching training)
    patch_size = 512
    overlap = 128
    threshold = 0.5
    
    print(f"Patch-based Inference Settings:")
    print(f"   - Patch size: {patch_size}×{patch_size}")
    print(f"   - Overlap: {overlap} pixels")
    print(f"   - Threshold: {threshold}")
    
    # Results storage
    all_metrics = []
    category_metrics = {'healthy': [], 'glaucoma': [], 'dr': []}
    all_confusion_matrices = []
    
    print(f"\nStarting patch-based evaluation...")
    print("-" * 70)
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Patch-based Evaluation")):
        try:
            # Load image
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load ground truth mask
            mask_file = image_file.replace('.jpg', '.tif').replace('.png', '.tif')
            mask_path = os.path.join(masks_dir, mask_file)
            
            if not os.path.exists(mask_path):
                # Try different extensions
                base_name = os.path.splitext(image_file)[0]
                possible_masks = [f for f in os.listdir(masks_dir) if base_name in f]
                if possible_masks:
                    mask_path = os.path.join(masks_dir, possible_masks[0])
                else:
                    print(f"No mask found for {image_file}")
                    continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to load mask for {image_file}")
                continue
            
            # Patch-based prediction (no resizing!)
            print(f"   Processing {image_file} ({image.shape[1]}×{image.shape[0]}) with patches...")
            prediction = patch_based_inference(
                model, image, 
                patch_size=patch_size, 
                overlap=overlap, 
                device=device
            )
            
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
            
            print(f"   {image_file}: Dice={metrics['dice_coefficient']:.4f}, Acc={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Calculate overall statistics
    print(f"\nCOMPREHENSIVE EVALUATION RESULTS")
    print("="*70)
    
    if not all_metrics:
        print("No successful evaluations!")
        return
    
    # Overall metrics
    overall_metrics = {}
    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_coefficient', 'iou']:
        values = [m[metric_name] for m in all_metrics]
        overall_metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"OVERALL PERFORMANCE (n={len(all_metrics)} images):")
    print(f"   Accuracy:        {overall_metrics['accuracy']['mean']:.4f} ± {overall_metrics['accuracy']['std']:.4f}")
    print(f"   Precision:       {overall_metrics['precision']['mean']:.4f} ± {overall_metrics['precision']['std']:.4f}")
    print(f"   Recall:          {overall_metrics['recall']['mean']:.4f} ± {overall_metrics['recall']['std']:.4f}")
    print(f"   F1-Score:        {overall_metrics['f1_score']['mean']:.4f} ± {overall_metrics['f1_score']['std']:.4f}")
    print(f"   Dice Coefficient: {overall_metrics['dice_coefficient']['mean']:.4f} ± {overall_metrics['dice_coefficient']['std']:.4f}")
    print(f"   IoU (Jaccard):   {overall_metrics['iou']['mean']:.4f} ± {overall_metrics['iou']['std']:.4f}")
    
    # Category-wise metrics
    print(f"\nCATEGORY-WISE PERFORMANCE:")
    for category, metrics_list in category_metrics.items():
        if metrics_list:
            category_means = {}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_coefficient', 'iou']:
                values = [m[metric_name] for m in metrics_list]
                category_means[metric_name] = np.mean(values)
            
            category_icon = {'healthy': '', 'glaucoma': '', 'dr': ''}.get(category, '⚪')
            print(f"   {category_icon} {category.upper()} (n={len(metrics_list)}):")
            print(f"      Accuracy: {category_means['accuracy']:.4f}")
            print(f"      Dice:     {category_means['dice_coefficient']:.4f}")
            print(f"      F1-Score: {category_means['f1_score']:.4f}")
            print(f"      Precision: {category_means['precision']:.4f}")
            print(f"      Recall:   {category_means['recall']:.4f}")
    
    # Plot confusion matrices
    print(f"\nCreating confusion matrices visualization...")
    overall_cm = plot_confusion_matrix(all_confusion_matrices, category_metrics)
    
    # Calculate pixel-level metrics from overall confusion matrix
    tn, fp, fn, tp = overall_cm.ravel()
    pixel_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    pixel_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    pixel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nPIXEL-LEVEL ANALYSIS:")
    print(f"   True Positives:  {tp:,}")
    print(f"   True Negatives:  {tn:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   False Negatives: {fn:,}")
    print(f"   Sensitivity (Recall): {pixel_sensitivity:.4f}")
    print(f"   Specificity:         {pixel_specificity:.4f}")
    print(f"   Precision:           {pixel_precision:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(all_metrics)
    # Remove confusion matrix from CSV (too large)
    results_csv = results_df.drop('confusion_matrix', axis=1)
    results_csv.to_csv('hrf_evaluation_results.csv', index=False)
    print(f"\nDetailed results saved to: hrf_evaluation_results.csv")
    
    # Create comprehensive summary report
    with open('hrf_evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write("HRF DATASET EVALUATION SUMMARY (PATCH-BASED)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        if training_info:
            f.write(f"Training sequence: {' → '.join(training_info)}\n")
        f.write(f"Evaluation method: Patch-based inference (no resizing)\n")
        f.write(f"Patch size: {patch_size}×{patch_size}, Overlap: {overlap}\n")
        f.write(f"Number of images evaluated: {len(all_metrics)}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"Accuracy:        {overall_metrics['accuracy']['mean']:.4f} ± {overall_metrics['accuracy']['std']:.4f}\n")
        f.write(f"Precision:       {overall_metrics['precision']['mean']:.4f} ± {overall_metrics['precision']['std']:.4f}\n")
        f.write(f"Recall:          {overall_metrics['recall']['mean']:.4f} ± {overall_metrics['recall']['std']:.4f}\n")
        f.write(f"F1-Score:        {overall_metrics['f1_score']['mean']:.4f} ± {overall_metrics['f1_score']['std']:.4f}\n")
        f.write(f"Dice Coefficient: {overall_metrics['dice_coefficient']['mean']:.4f} ± {overall_metrics['dice_coefficient']['std']:.4f}\n")
        f.write(f"IoU (Jaccard):   {overall_metrics['iou']['mean']:.4f} ± {overall_metrics['iou']['std']:.4f}\n\n")
        
        f.write("PIXEL-LEVEL ANALYSIS:\n")
        f.write(f"True Positives:   {tp:,}\n")
        f.write(f"True Negatives:   {tn:,}\n")
        f.write(f"False Positives:  {fp:,}\n")
        f.write(f"False Negatives:  {fn:,}\n")
        f.write(f"Sensitivity:      {pixel_sensitivity:.4f}\n")
        f.write(f"Specificity:      {pixel_specificity:.4f}\n\n")
        
        f.write("CATEGORY-WISE PERFORMANCE:\n")
        for category, metrics_list in category_metrics.items():
            if metrics_list:
                f.write(f"\n{category.upper()} (n={len(metrics_list)}):\n")
                for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'dice_coefficient', 'iou']:
                    values = [m[metric_name] for m in metrics_list]
                    f.write(f"  {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
    
    print(f"Summary report saved to: hrf_evaluation_summary.txt")
    print(f"Confusion matrices saved to: hrf_confusion_matrices.png")
    
    print(f"\nHRF patch-based evaluation completed successfully!")
    return overall_metrics, category_metrics

if __name__ == '__main__':
    evaluate_hrf_dataset() 