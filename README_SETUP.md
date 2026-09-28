# 🧠 BRATS 2020 Brain Tumor Segmentation - Setup Guide

## 📋 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download BRATS 2020 Dataset
Download from: [Kaggle BRATS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

Extract to: `data/BRATS_2020_training_data/`

Directory structure should look like:
```
data/
└── BRATS_2020_training_data/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_T1.nii.gz
    │   ├── BraTS20_Training_001_T1ce.nii.gz
    │   ├── BraTS20_Training_001_T2.nii.gz
    │   ├── BraTS20_Training_001_FLAIR.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    ├── BraTS20_Training_002/
    │   └── ...
    └── ...
```

## 🚀 Training

### Run Training Script
```bash
python train.py
```

**Note:** Update `BRATSDataset` class in `train.py` with your actual data loading logic.

### Training Configuration
Edit `config.py` to adjust:
- `TRAIN_CONFIG`: Epochs, batch size, learning rate
- `MODEL_CONFIG`: Network architecture
- `DATA_CONFIG`: Input dimensions, train/val split
- `AUG_CONFIG`: Data augmentation settings

### Monitor Training
```bash
tensorboard --logdir logs/
```

## 📊 Evaluation

### Evaluate on Validation Set
```bash
python evaluate.py checkpoints/best_model_epoch50.pth data/BRATS_2020_training_data/
```

Computes:
- **Dice Score**: Similarity between predicted and ground truth
- **Hausdorff95**: Maximum distance between segmentations
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate

Reported for:
- **WT** (Whole Tumor): Classes 1, 2, 3
- **TC** (Tumor Core): Classes 1, 3
- **ET** (Enhancing Tumor): Class 3

## 🔮 Inference

### Single Patient Prediction
```bash
python inference.py checkpoints/best_model_epoch50.pth data/test/patient_001/
```

### Batch Inference (Multiple Patients)
```bash
python inference.py checkpoints/best_model_epoch50.pth data/test/ --batch
```

Outputs saved to `output/` directory as `.nii.gz` files.

## 📁 Project Structure

```
brain/
├── MONAITHON-2K25/             # Original Kaggle notebook and assets
│   └── MONAITHON-2K25/
│       ├── monaithon-2k25-brats.ipynb
│       ├── README.md
│       ├── LICENSE
│       └── assets/             # Visualizations
├── config.py                   # Configuration settings
├── utils.py                    # Utility functions (I/O, metrics)
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── inference.py                # Inference script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
├── README.md                   # Main project documentation
└── data/                       # Data directory (not in repo)
    ├── BRATS_2020_training_data/
    ├── test/
    └── ...
```

## 🛠️ Configuration Details

### `config.py`

**TRAIN_CONFIG:**
- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size per GPU (default: 2)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `weight_decay`: L2 regularization (default: 1e-5)

**MODEL_CONFIG:**
- `in_channels`: 4 (T1, T1ce, T2, FLAIR)
- `out_channels`: 4 (Background, Necrotic, Edema, Enhancing)
- 3D UNet architecture with residual connections

**DATA_CONFIG:**
- `roi_x/y/z`: Input patch size (128×128×128)
- `train_val_split`: 80/20 train/validation split

**AUG_CONFIG:**
- Random rotations, scaling, flips, Gaussian noise

## 📚 Key Classes & Functions

### `BRATSDataset` (train.py)
Load NIFTI files, apply transforms, return tensor batches.

### `compute_metrics()` (evaluate.py)
Calculate Dice, Hausdorff95, Sensitivity, Specificity.

### `predict_patient()` (inference.py)
Run inference on a single patient, save `.nii.gz` prediction.

### `load_nifti()`, `save_nifti()` (utils.py)
I/O utilities for NIfTI medical images.

## 🎯 Next Steps

1. ✅ Complete `BRATSDataset` implementation with your data paths
2. ✅ Train model: `python train.py`
3. ✅ Evaluate: `python evaluate.py`
4. ✅ Predict: `python inference.py`
5. ✅ Deploy with Streamlit/Gradio (optional)

## 📖 References

- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials)
- [BRATS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [3D Medical Image Segmentation](https://arxiv.org/abs/1505.04597)

## 📝 License

See `MONAITHON-2K25/MONAITHON-2K25/LICENSE`
