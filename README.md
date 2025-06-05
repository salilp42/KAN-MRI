# KAN-MRI: Deep Learning Models for MRI-based Parkinson's Disease Classification

This repository provides the code implementation for various 2D and 3D deep learning models designed for Parkinson's Disease (PD) classification using Magnetic Resonance Imaging (MRI) data. It includes implementations for Convolutional Neural Networks (CNNs), Graph Neural Networks (GCNs), Kolmogorov-Arnold Network inspired Convolutional models (ConvKANs), and Vision Transformers (ViTs).

## Repository Structure

```
KAN-MRI/
├── .gitignore           # Git ignore file
├── README.md            # This README file
├── requirements.txt     # Python package dependencies
├── scripts/             # Analysis and utility scripts
│   ├── run_analysis.py  # Main script to run training, evaluation, and comparison
│   ├── test_vit_models.py # Test script for ViT models
│   └── ...              # Other helper scripts (e.g., for metrics, summaries)
└── src/                 # Source code
    ├── datasets/        # Data loading and handling (e.g., mri_dataset.py)
    ├── evaluation/      # Evaluation metrics and statistical tests
    │   ├── metrics.py
    │   └── statistical_tests.py
    ├── graph/           # Graph construction code (e.g., graph_builder.py)
    ├── models/          # Model implementations
    │   ├── __init__.py  # Models module initialization
    │   ├── base.py      # Base model class with common functionality
    │   ├── convkan.py   # ConvKAN2D, ConvKAN3D
    │   ├── gnn.py       # GCN2D, GCN3D
    │   ├── resnet.py    # ResNet2D (pretrained), ResNet3D (from scratch)
    │   ├── vgg.py       # VGG16_2D (pretrained)
    │   └── vit.py       # ViT2D, ViT3D (Vision Transformers)
    ├── preprocessing/   # Data preprocessing functions (if any)
    └── utils/           # Utility functions (if any)
```

## Implemented Models

*   **ConvKAN (Convolutional Kolmogorov-Arnold Network):**
    *   `ConvKAN2D`: 2D version using B-spline based convolutional layers.
    *   `ConvKAN3D`: 3D version using B-spline based convolutional layers.
    *   *Key Parameter:* `num_knots` in `BSpline2D`/`BSpline3D` (default: 6).

*   **CNN (Convolutional Neural Network):**
    *   `ResNet2D`: Based on pretrained ResNet18, adapted for MRI input.
    *   `VGG16_2D`: Based on pretrained VGG16, adapted for MRI input.
    *   `ResNet3D`: Simple 3D ResNet implementation (trained from scratch).

*   **GCN (Graph Convolutional Network):**
    *   `GCN2D`: Operates on graphs constructed from 2D slices using SLIC superpixels.
    *   `GCN3D`: Operates on graphs constructed from 3D volumes using SLIC supervoxels.
    *   *Graph Construction:* Uses k-NN (default k=6) based on superpixel/supervoxel features (mean intensity, relative size, centroid).

*   **ViT (Vision Transformer):** *(New)*
    *   `ViT2D`: 2D Vision Transformer using ViT-Tiny architecture (5.7M parameters) with ImageNet pretraining. Optimized for small medical datasets with patch size 16x16.
    *   `ViT3D`: Hybrid 3D Vision Transformer that processes 3D volumes by applying 2D ViT to each slice and using cross-slice attention to capture 3D context. Avoids computational complexity of full 3D transformers while maintaining attention benefits.
    *   *Key Features:* Pretrained weights, medical image normalization, adaptive input sizing, multiple aggregation methods for 3D.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd KAN-MRI
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate 
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Testing ViT Models

Before running the full analysis, you can test the ViT models to ensure they work correctly:

```bash
python scripts/test_vit_models.py
```

This script will:
- Test ViT2D and ViT3D model instantiation
- Verify forward passes with sample data
- Compare different model configurations
- Display model statistics (parameters, size, etc.)

## Running the Analysis

The primary script for running the analyses (training models, evaluating performance, comparing models) is `scripts/run_analysis.py`.

**Basic Usage:**

```bash
python scripts/run_analysis.py
```

**Options:**

*   `--debug`: Run in debug mode, potentially using smaller datasets or fewer epochs for faster testing.
    ```bash
    python scripts/run_analysis.py --debug
    ```
*   *(Add other relevant arguments here if `run_analysis.py` uses them, e.g., `--config`, `--data_dir`, `--output_dir`)*

The script will typically perform the following steps (depending on its internal logic):

1.  Load and preprocess data (potentially defined via configuration).
2.  Initialize the models specified for the analysis (including the new ViT models).
3.  Run different analysis types (e.g., Isolated training/testing, Combined training/testing, Hold-out testing).
4.  Train models using appropriate settings (optimizer, learning rate, loss function with potential smoothing/weighting, early stopping).
5.  Evaluate models using various metrics (Accuracy, AUC, F1, Specificity, Sensitivity, etc.) and calculate confidence intervals.
6.  Perform statistical comparisons between models (e.g., using McNemar's test with FDR correction).
7.  Calculate efficiency metrics (training time per epoch, inference time per subject).
8.  Save results, potentially including performance metrics tables, plots (like ROC curves, confusion matrices), and trained model weights. Check the script or generated output directories for details.

## Model Architecture Details

### ViT2D Architecture
- **Backbone**: ViT-Tiny (vit_tiny_patch16_224) from timm library
- **Parameters**: ~5.7M (lightweight for medical datasets)
- **Input**: 224x224 images, automatically resized if needed
- **Features**: ImageNet pretraining, medical image normalization, dropout regularization

### ViT3D Architecture
- **Approach**: Hybrid 2D ViT + 3D attention
- **Slice Processing**: 2D ViT applied to each slice independently
- **3D Context**: Cross-slice attention mechanism with positional encoding
- **Aggregation**: Configurable (attention-weighted, mean, or max pooling)
- **Efficiency**: Avoids full 3D transformer complexity while capturing volumetric relationships

## Dependencies

Key dependencies include:
- PyTorch (with torchvision)
- timm >= 0.9.0 (for Vision Transformers)
- torch_geometric (for GNN models)
- scikit-learn, pandas, numpy
- nibabel (for MRI data handling)
- SimpleITK (for 3D image processing)

See `requirements.txt` for the complete list.

