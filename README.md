# SARFormer: Neural Networks for Missing Persons Location Prediction

A multimodal deep learning framework for predicting the locations of missing persons in search and rescue (SAR) scenarios. Developed at the **AI for Search & Rescue Lab**, sponsored by the **Menlo Park Fire Protection District**.

## Overview

SARFormer fuses satellite imagery, elevation maps, and textual incident metadata to predict where a missing person is most likely to be found. The system is trained on historical search and rescue incident records spanning many geographic regions across the United States. (The training data is not distributed with this repository.)

### Input Modalities

| Modality | Description |
|---|---|
| **RGB** | NAIP aerial/satellite imagery |
| **Depth** | Elevation and terrain maps |
| **Caption** | Natural language descriptions of the incident (person profile, weather, terrain) |
| **Structured** | Tabular metadata (time, weather conditions, location features) |

### Models

- **SARFormer** -- Multimodal transformer that fuses a ConvNeXt-based spatial encoder (for imagery) with a T5 text encoder (for captions/metadata) via cross-attention, predicting a spatial heatmap of likely locations.
- **ConvNeXt Point Regression** -- ConvNeXtV2 backbone with a GatedMLP head for direct 2D coordinate regression. Available in Tiny, Base, Large, and Huge variants.
- **Baselines** -- Center-prediction and uniform-random baselines, plus gradient-boosted decision tree (GBDT) models via scikit-learn.

## Architecture

```
              RGB Image ──┐
                          ├──► ConvNeXt UNet (Spatial Encoder) ──┐
          Depth Map ──────┘                                      │
                                                                 ├──► Cross-Attention Fusion ──► Location Prediction
          Caption ────────────► T5 Encoder (Text) ───────────────┘
          Structured Data ────┘
```

The spatial encoder processes RGB and depth inputs through a patched ConvNeXt UNet with multi-scale skip connections. The T5 encoder processes tokenized text captions. Cross-attention blocks allow the text context to condition the spatial features before producing a location prediction.

## Training

- **Objective**: Lost-Person Location (LPL) prediction with a distance-weighted spatial loss, plus optional masked autoencoder (MAE) pretraining
- **Infrastructure**: Distributed training on 8x A100 GPUs via PyTorch DDP with mixed-precision (bf16/fp16) and gradient accumulation
- **Tracking**: Weights & Biases for experiment logging, metric tracking, and prediction visualization
- **Evaluation Metrics**: MSE, MAE, PSNR, MS-SSIM

## Project Structure

```
sarformer/
├── run_training_sarformer.py    # Main distributed training script
├── cfgs/                        # YAML experiment configs (model, tokenizer, pretraining)
├── src/
│   ├── models/                  # SARFormer, ConvNeXt regression, UNet, baselines
│   ├── data/                    # Multimodal dataset loader and per-modality transforms
│   └── utils/                   # Custom tokenizer, CLIP utilities, optimization
├── dataset_gen/                 # Dataset generation and preprocessing scripts
├── tif_gen/                     # GeoTIFF generation utilities
├── notebooks/                   # Analysis and visualization notebooks
└── job_scripts/                 # SLURM job submission scripts
```

## Tech Stack

PyTorch, Hugging Face Transformers, torchvision, scikit-learn, Weights & Biases, Albumentations, torchmetrics

## Setup

```bash
pip install -e .

# Optional: install xformers for memory-efficient attention
pip install -e ".[fast]"
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
