# Advanced Retinal Vessel Segmentation using Sequential Deep Learning

## Project Information

**Đề tài:** Phân tích mạch máu võng mạc nâng cao có sử dụng Deep Learning để phân đoạn ảnh có độ phân giải cao

**University:** Ho Chi Minh City University of Technology and Education  
**Faculty:** Faculty of International Education  
**Academic Year:** 2024-2025

### 👥 Research Team
- **Student:** Nguyễn Nhật Phát - ID: 23110053
- **Supervisor:** Assoc. Prof. Dr. Hoàng Văn Dũng

## Project Structure (Organized)

```
retinal_vessel_segmentation/
├── 📁 src/                              # Source Code
│   ├── 📁 models/
│   │   ├── unet_plus_plus.py           # U-Net++ architecture
│   │   └── loss_functions.py           # Loss functions
│   ├── 📁 datasets/
│   │   ├── base_dataset.py             # Base dataset class
│   │   ├── drive_dataset.py            # DRIVE dataset handler
│   │   ├── hrf_patch_dataset.py        # HRF patch dataset
│   │   └── drive_patch_dataset.py      # DRIVE patch dataset
│   ├── 📁 training/                    # Training Scripts
│   │   ├── stage1_base_training.py     # Stage 1: Combined training
│   │   ├── stage2_patch_finetune.py    # Stage 2: Patch fine-tuning
│   │   ├── stage3_extended_train.py    # Stage 3: Extended training
│   │   └── stage4_domain_adapt.py      # Stage 4: Domain adaptation
│   ├── 📁 evaluation/
│   │   └── evaluate_model.py           # Model evaluation
│   └── 📁 inference/
│       ├── predict.py                  # Single prediction
│       └── batch_predict.py            # Batch prediction
├── 📁 models/                          # Trained Models
│   ├── stage1_base_model.pth           # Stage 1 output
│   ├── stage2_patch_finetuned.pth      # Stage 2 output
│   ├── stage3_extended_model.pth       # Stage 3 output
│   └── stage4_final_model.pth          # Final best model
├── 📁 data/                            # Datasets
│   ├── DRIVE/                          # DRIVE dataset
│   └── HRF/                            # HRF dataset
├── 📁 results/                         # Results & Outputs
│   ├── predictions/
│   │   └── drive_test/                 # DRIVE test predictions
│   ├── evaluations/                    # Evaluation results
│   └── visualizations/                 # Result visualizations
├── 📁 webapp/                          # Web Application
│   ├── app.py                          # Flask application
│   ├── templates/                      # HTML templates
│   └── static/                         # Static files
├── 📁 configs/                         # Configuration Files
│   └── training_config.yaml            # Training configurations
├── 📁 docs/                            # Documentation
├── requirements.txt                     # Dependencies
└── README_ORGANIZED.md                  # This file
```

## Sequential Training Methodology

### **4-Stage Progressive Learning Pipeline:**

#### **Stage 1: Base Training** 
- **Script:** `src/training/stage1_base_training.py`
- **Input:** DRIVE + HRF combined dataset
- **Output:** `models/stage1_base_model.pth`
- **Purpose:** Foundation knowledge from both datasets

#### **Stage 2: Patch-based Fine-tuning**
- **Script:** `src/training/stage2_patch_finetune.py`  
- **Input:** Stage 1 model + HRF patches (512×512)
- **Output:** `models/stage2_patch_finetuned.pth`
- **Purpose:** High-resolution expertise with progressive unfreezing

#### **Stage 3: Extended Training**
- **Script:** `src/training/stage3_extended_train.py`
- **Input:** Stage 2 model  
- **Output:** `models/stage3_extended_model.pth`
- **Purpose:** Enhanced convergence targeting Dice > 0.85

#### **Stage 4: Domain Adaptation**
- **Script:** `src/training/stage4_domain_adapt.py`
- **Input:** Stage 3 model + DRIVE patches
- **Output:** `models/stage4_final_model.pth` ⭐ **Final Model**
- **Purpose:** HRF → DRIVE knowledge transfer

## Quick Start

### **1. Environment Setup**
```bash
pip install -r requirements.txt
```

### **2. Enhanced Package Structure**

Project được tổ chức với **professional Python package structure** sử dụng `__init__.py` files:

```python
# Main package info
import src
print(src.PROJECT_INFO['title'])    # Project title
print(src.__version__)              # Version 1.0.0
print(src.__author__)               # Nguyễn Nhật Phát

# Models - Direct imports (thanks to __init__.py)
from src.models import create_model, DiceBCELoss, dice_coefficient

# Datasets - Clean imports  
from src.datasets import BaseRetinalDataset, HRFPatchDataset, get_train_transform

# Training stages info
from src.training import TRAINING_STAGES
for stage_num, info in TRAINING_STAGES.items():
    print(f"Stage {stage_num}: {info['name']} -> {info['output']}")
```

**Benefits of Enhanced `__init__.py`:**
- **Cleaner imports:** `from src.models import create_model`
- **Package documentation:** Each package có docstring đầy đủ  
- **Metadata access:** Version, author, project info
- **Professional structure:** Tuân thủ Python best practices
- **Easy maintenance:** Central control của exports

### **3. Data Preparation**
```bash
# Place datasets in data/ folder
data/
├── DRIVE/
│   ├── training/images/
│   ├── training/1st_manual/
│   └── test/
└── HRF/
    ├── images/
    └── manual1/
```

### **4. Sequential Training**
```bash
# Stage 1: Base Training
python src/training/stage1_base_training.py

# Stage 2: Patch Fine-tuning  
python src/training/stage2_patch_finetune.py

# Stage 3: Extended Training
python src/training/stage3_extended_train.py

# Stage 4: Domain Adaptation
python src/training/stage4_domain_adapt.py
```

### **5. Web Application**
```bash
cd webapp/
python app.py
# Access: http://localhost:5000
```

### **6. Evaluation**
```bash
python src/evaluation/evaluate_model.py
```

## Key Features

### **Architecture**
- **U-Net++** with EfficientNet-B4 backbone
- **Nested skip connections** for multi-scale feature fusion
- **Mixed precision training** for GPU acceleration

### **Advanced Training**
- **Sequential 4-stage methodology**
- **Patch-based learning** for high-resolution images  
- **Progressive unfreezing** strategy
- **FOV-aware loss functions**

### **Technical Innovations**
- **Multi-dataset training:** DRIVE + HRF
- **Domain adaptation:** HRF → DRIVE transfer
- **Memory optimization:** Patch extraction + CUDA management
- **Professional deployment:** Flask web application

## Performance Results

### **Model Progression:**
```
Stage 1 → Stage 2 → Stage 3 → Stage 4
  Base   →  Patch  → Extended → Final
(244MB) → (110MB) → (244MB)  → (244MB)
```

### **Evaluation Metrics:**
- **Primary:** Dice Coefficient, FOV-aware Dice
- **Validation Range:** 0.35-0.38 (from training logs)
- **Best Recorded:** 0.3800
- **Output:** 20 DRIVE test predictions

## 🛠Configuration

All training parameters are centralized in `configs/training_config.yaml`:

```yaml
# Model Configuration
model:
  architecture: "UnetPlusPlus"
  encoder_name: "efficientnet-b4"

# Stage-specific hyperparameters
stages:
  stage1_base_training:
    epochs: 200
    batch_size: 4
    learning_rate: 3e-4
  # ... other stages
```

## 📚 Usage Examples

### **Single Image Prediction:**
```python
from src.inference.predict import predict_single_image
result = predict_single_image("path/to/image.jpg", "models/stage4_final_model.pth")
```

### **Batch Prediction:**
```python
from src.inference.batch_predict import predict_batch
results = predict_batch("path/to/images/", "models/stage4_final_model.pth")
```

## 🎨 Web Application Features

- **Multi-format support:** JPEG, PNG, TIFF
- **Real-time processing** with GPU acceleration
- **Visual comparison** of input and segmented output
- **Responsive design** for desktop and mobile

## 📖 Documentation

- **Methodology:** See `configs/training_config.yaml` for detailed parameters
- **Code Documentation:** All functions include comprehensive docstrings
- **Training Logs:** Available in `wandb/` directory

## Scientific Contribution

This project demonstrates:

1. **Sequential Training Methodology** for medical image segmentation
2. **Multi-dataset learning** combining different image characteristics
3. **Patch-based fine-tuning** for high-resolution medical images
4. **Domain adaptation** techniques in deep learning
5. **Production-ready deployment** of research models

## 📄 Dependencies

See `requirements.txt` for complete list:
- PyTorch ≥ 2.1.0
- segmentation-models-pytorch ≥ 0.3.3
- albumentations ≥ 1.3.1
- Flask (for web app)

## Research Achievements

- **4-stage sequential training** methodology implemented
- **Progressive learning** from low to high resolution
- **Domain adaptation** between different datasets
- **Production deployment** with web interface
- **Comprehensive evaluation** on DRIVE benchmark

---

**Contact:** Nguyễn Nhật Phát - 23110053@student.hcmute.edu.vn 