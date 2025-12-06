"""
Training Package - 4-Stage Sequential Training Pipeline

Chứa các training scripts cho sequential 4-stage methodology:

Stage 1: Base Training (Combined DRIVE + HRF)
- Script: stage1_base_training.py  
- Output: stage1_base_model.pth
- Purpose: Foundation knowledge từ cả hai datasets

Stage 2: Patch Fine-tuning (HRF High-Resolution)  
- Script: stage2_patch_finetune.py
- Output: stage2_patch_finetuned.pth
- Purpose: Học high-resolution details từ HRF patches

Stage 3: Extended Training (Enhanced Convergence)
- Script: stage3_extended_train.py  
- Output: stage3_extended_model.pth
- Purpose: Enhanced convergence và stability

Stage 4: Domain Adaptation (HRF → DRIVE)
- Script: stage4_domain_adapt.py
- Output: stage4_final_model.pth  
- Purpose: Transfer knowledge về DRIVE domain
"""

# Training stages information
TRAINING_STAGES = {
    1: {
        "name": "Base Training",
        "script": "stage1_base_training.py",
        "output": "stage1_base_model.pth", 
        "purpose": "Foundation knowledge từ DRIVE + HRF combined"
    },
    2: {
        "name": "Patch Fine-tuning",
        "script": "stage2_patch_finetune.py",
        "output": "stage2_patch_finetuned.pth",
        "purpose": "High-resolution expertise từ HRF patches"
    },
    3: {
        "name": "Extended Training", 
        "script": "stage3_extended_train.py",
        "output": "stage3_extended_model.pth",
        "purpose": "Enhanced convergence và performance"
    },
    4: {
        "name": "Domain Adaptation",
        "script": "stage4_domain_adapt.py", 
        "output": "stage4_final_model.pth",
        "purpose": "Domain adaptation HRF → DRIVE"
    }
}  
