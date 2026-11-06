# Brain Tumor Segmentation Using U-Net

**Binary semantic segmentation of brain tumors in MRI images using PyTorch.**

This project implements a U-Net architecture from scratch to predict pixel-level tumor masks from MRI images. The notebook covers dataset preparation, augmentation, training, evaluation, and visual inspection of predictions.

> **Research/educational project only.** Predictions are not intended for medical diagnosis or clinical decision-making.

## Results

Final test metrics reported for this notebook:

| Metric | Test result |
|---|---:|
| Loss (BCE + Dice) | **0.1206** |
| Dice score | **0.7951** |
| Intersection over Union (IoU) | **0.7075** |

The metrics are computed with a **0.5 probability threshold**, averaged per image and then across the test set. As a result, the reported mean Dice and mean IoU need not obey the single-mask Dice–IoU conversion formula.

## Dataset

**[Brain Tumor Segmentation — Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)** (`nikhilroxtomar/brain-tumor-segmentation`)

The notebook downloads the dataset with `kagglehub` and expects two directories:

```text
<dataset-root>/
├── images/
└── masks/
```

Image and mask files are paired by matching filenames. Masks are converted to binary labels (nonzero = tumor). The notebook uses a random **70% training / 15% validation / 15% test** split of matched image-mask pairs with seed `42`. This is an image-level split; it does **not** explicitly enforce patient-level separation.

## Model Architecture

A custom **U-Net** implemented in PyTorch:

- Input: 3-channel RGB image, resized to **256 × 256**
- Encoder: four double-convolution blocks with channels **64 → 128 → 256 → 512**
- Bottleneck: **1024** channels
- Decoder: transposed convolutions, skip connections, and double-convolution blocks
- Output: one-channel tumor segmentation logits, converted to probabilities with sigmoid
- Double-convolution block: `Conv2d → BatchNorm2d → ReLU` repeated twice

## Training Configuration

| Setting | Value |
|---|---|
| Image size | 256 × 256 |
| Batch size | 16 |
| Maximum epochs | 60 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss | 0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early stopping | Patience 7, based on validation Dice |
| Prediction threshold | 0.5 |
| Random seed | 42 |
| Checkpoint | `best_unet.pth` |

Training uses automatic mixed precision when CUDA is available and gradient clipping (`max_norm=1.0`). The best checkpoint is selected using **validation Dice**.

### Data augmentation

Training images use Albumentations resizing, horizontal and vertical flips, random 90° rotation, random brightness/contrast, normalization, and tensor conversion. Validation and test images use resizing, normalization, and tensor conversion only.

## Notebook Workflow

1. Download the dataset from Kaggle.
2. Import dependencies and configure training settings.
3. Inspect the image and mask directories and visualize samples.
4. Build a custom PyTorch `Dataset`, transforms, and `DataLoader`s.
5. Implement U-Net with encoder–decoder skip connections.
6. Define combined BCE + Dice loss and the AdamW optimizer.
7. Train with validation, learning-rate scheduling, early stopping, and best-model checkpointing.
8. Plot training/validation loss, Dice, and IoU curves.
9. Reload the best checkpoint and evaluate on the held-out test split.
10. Visualize original MRI images, ground-truth masks, predictions, and red overlays.

## How to Run

1. Open [`Brain_Tumor_Segmentaion.ipynb`](Brain_Tumor_Segmentaion.ipynb) in **Kaggle Notebooks** or a compatible Jupyter environment.
2. Enable a GPU if available.
3. Run cells in order. The first cells install/download dependencies and the dataset.
4. After training, the best checkpoint is saved as `best_unet.pth` in the notebook's working directory.
5. Run the final cells to load the checkpoint, print test metrics, and inspect predictions.

Core dependencies include `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `scikit-learn`, `albumentations`, `torchmetrics`, `torchsummary`, `tqdm`, and `kagglehub`.

**Note:** The notebook calls `torch.cuda.get_device_name(0)` unconditionally in its setup cell. On CPU-only machines, guard that call with `torch.cuda.is_available()` before running.

## Evaluation Metrics

For binary masks:

\[
\mathrm{Dice}=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{IoU}=\frac{TP}{TP+FP+FN}
\]

Both metrics range from 0 (no overlap) to 1 (perfect overlap). The notebook computes each metric for every image and averages across samples. Dice and IoU should be interpreted alongside qualitative predictions and results on an independent, patient-separated evaluation set.

## Example Outputs

The notebook's visualization function displays four panels for each sampled test image:

| Original MRI | Ground Truth | Predicted Mask | Prediction Overlay |
|---|---|---|---|
| Input image | Binary tumor label | Thresholded U-Net prediction with per-image Dice | Predicted tumor highlighted in red |

To add screenshots to this README, export a visualization from the notebook to an `assets/` directory and link it here.

## Limitations & Next Steps

- The current split is by image rather than verified patient identity, so leakage between patients' related scans is possible.
- Evaluation uses a fixed threshold of 0.5; threshold tuning should use validation data only.
- Investigate performance on small tumors, false positives, and empty masks.
- Add patient-level splitting and external validation before making generalization claims.
- Package preprocessing and checkpoint loading into a **Streamlit demo** for research visualization.

## Tech Stack

**Python · PyTorch · U-Net · Albumentations · scikit-learn · Matplotlib · Kaggle · Jupyter**

## Acknowledgment

Dataset: [Brain Tumor Segmentation on Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation).
