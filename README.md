# Retinal Vessel Segmentation (Sequential Deep Learning)

High-resolution retinal vessel segmentation using U-Net++ (EfficientNet-B4 encoder) and a four-stage sequential training pipeline, plus a Flask web app for real-time inference.

<img width="2235" height="654" alt="02_h_visualization" src="https://github.com/user-attachments/assets/56ca42a1-7aa4-4da2-9694-2f11026921f6" />

## Highlights
- Architecture: U-Net++ with attention, mixed precision, Dice+BCE loss.
- Data: DRIVE and HRF, patch-based training to save memory.
- Four stages: base -> patch fine-tune -> extended -> domain adaptation.
- Inference: scripts for DRIVE and HRF patch-based; Flask web demo.
- Centralized configuration: `configs/training_config.yaml`.

## Project structure
```
retinal_vessel_segmentation/
├── src/
│   ├── models/                # U-Net++ architecture, losses, metrics
│   ├── datasets/              # DRIVE, HRF, and patch datasets
│   ├── training/              # stage1..4 sequential training scripts
│   ├── evaluation/            # Model evaluation
│   └── inference/             # Single/batch inference
├── configs/                   # Training configs
├── webapp/                    # Flask demo app
├── data/                      # Place datasets here (not committed)
├── models/                    # Checkpoints (.pth) (not committed)
├── results/                   # Predictions/evaluations (not committed)
├── docs/                      # Additional docs
└── requirements.txt
```

## Environment setup
1) Create a virtual environment (Python 3.10+ recommended).  
2) Install dependencies:
```
pip install -r requirements.txt
```

## Data preparation
Organize datasets as follows:
```
data/
└── DRIVE/
    ├── training/images/
    ├── training/1st_manual/
    └── test/
        ├── images/
        ├── 1st_manual/        # used in stage1
        └── mask/              # used in inference/eval
└── HRF/
    ├── images/
    └── manual1/
```
Adjust paths in scripts if your layout differs.

## Sequential training (4 stages)
```
python src/training/stage1_base_training.py
python src/training/stage2_patch_finetune.py
python src/training/stage3_extended_train.py
python src/training/stage4_domain_adapt.py
```
Checkpoints are saved to `models/` by default. Hyperparameters live in `configs/training_config.yaml`.

## Evaluation
- DRIVE: `python src/evaluation/evaluate_model.py` (expects `best_model.pth` and DRIVE data under the specified paths; align to `data/DRIVE` if needed).
- Additional scripts: `evaluate_hrf_metrics.py`, `evaluate_hrf_test_only.py` for HRF.

## Inference
- DRIVE test: `python src/inference/predict.py`  
  - Loads `models/stage4_final_model.pth`, reads images from `data/DRIVE/test/images`, masks from `data/DRIVE/test/mask`, saves to `results/predictions/drive_test/`.
- HRF patch-based: `python src/inference/batch_predict.py`  
  - Default model `sequential_finetuned_models/best_model_sequential_hrf_extended_drive.pth` (update path if different), reads `HRF/images`, saves to `hrf_predictions_sequential/`.

## Web application
```
cd webapp
python app.py
# Visit http://localhost:5000
```

## Reported results
- Validation Dice roughly 0.35-0.38; best recorded 0.3800 (training logs).
- Generated 20 DRIVE test predictions and HRF evaluation outputs with confusion matrices.
- See also: `docs/FINAL_PROJECT_SUMMARY.md`, `docs/README_ORGANIZED.md`, `hrf_*evaluation*.txt/csv`, `hrf_confusion_matrices.png`.

## Versioning notes
- Do not commit data, checkpoints (.pth), `results/`, `wandb/`. Patterns are already in `.gitignore`.
- For checkpoint tracking, use Git LFS or attach them as release assets.


## Contact
Nguyễn Nhật Phát - 23110053@student.hcmute.edu.vn
