# Brain2: Medical Image Binary Classification Pipeline

## Project Overview
This repository provides a complete, production-ready deep learning pipeline for binary classification of medical images: distinguishing MRI scans from Breast Histopathology images. The solution is built with PyTorch and includes all steps from data preparation to model training, evaluation, and inference.

## Features
- **Custom Shallow CNN** (5 conv layers) for binary classification
- **Comprehensive training loop** with early stopping and checkpointing
- **Robust evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Visualizations**: ROC curve, confusion matrix, training curves, sample predictions
- **Inference utility**: `testing.py` for single image prediction
- **Reproducible results**: random seed, organized code, clear configs

## Results
All results and visualizations are saved in the `results/` folder:
- `roc_curve.png`: ROC curve plot
- `confusion_matrix.png`: Confusion matrix (raw and normalized)
- `training_history.png`: Training loss, accuracy, F1, AUC curves
- `sample_predictions.png`: Example test predictions
- `test_results.json`: Test set metrics
- `training_history.json`: Training metrics per epoch
- `evaluation_summary.json`: Final summary report

### Model Performance
- **Test Accuracy**: 100%
- **Test Precision/Recall/F1/AUC**: 1.0000
- **Test Loss**: 0.0001
- **Total Test Samples**: 2,653
- **Best Validation Accuracy**: 100%

## Data Setup
**Datasets are NOT included in this repository.**

To run the pipeline, you must manually place the datasets in the following folders:
- `MRI/` — Place all MRI images (flattened from tumor subfolders)
- `BreastHisto/` — Place all Breast Histopathology images (sampled, 5000 per class)

After placing the data, run the notebook to process, split, and train. The code will automatically create `processed_data/` and organize splits.

## Usage
### 1. Training & Evaluation
Open and run `brain_classification.ipynb` in Jupyter/VS Code. This will:
- Prepare and process the datasets
- Train the model (custom CNN or pretrained)
- Save checkpoints and metrics
- Generate all results in `results/`

### 2. Inference
Use `testing.py` for single image prediction:
```bash
python testing.py
```
- Enter the path to an image file when prompted
- The script will output the predicted class (MRI or BreastHisto) and confidence

## File Structure
- `brain_classification.ipynb` — Main notebook (full pipeline)
- `testing.py` — Standalone script for inference
- `results/` — All evaluation outputs and visualizations
- `models/` — Saved model checkpoints
- `processed_data/` — Processed and split data (auto-generated)
- `MRI/`, `BreastHisto/` — Raw datasets (not included)

## Requirements
- Python 3.8+
- PyTorch, torchvision, scikit-learn, matplotlib, seaborn, pandas, PIL

Install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow tqdm
```

## Notes
- **Data folders are ignored in `.gitignore`** to prevent large uploads.
- **You must manually place the datasets** as described above.
- All results and model checkpoints are auto-saved for reproducibility.

## License
This project is for educational and research use. Please cite appropriately if used in publications.
