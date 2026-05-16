# Experiments

This folder documents planned CNN and Transformer experiments for deepfake detection.

## Purpose

Each experiment should be reproducible and easy to compare. The goal is to track model architecture, dataset version, preprocessing steps, training settings, and evaluation results in a clear format.

## Experiment Naming

Use short names such as:

- `cnn-baseline-v1`
- `cnn-transfer-learning-v1`
- `vision-transformer-v1`
- `hybrid-cnn-transformer-v1`

## Experiment Template

Each experiment should record:

```text
Experiment name:
Date:
Dataset version:
Preprocessing:
Model architecture:
Training epochs:
Batch size:
Learning rate:
Optimizer:
Metrics:
Notes:
```

## Planned Experiments

1. CNN baseline for real vs fake image/frame classification
2. Transfer learning with a pretrained CNN backbone
3. Vision Transformer-style classifier
4. Hybrid CNN-Transformer approach
5. Robustness test on compressed or low-quality media

## Comparison Goals

- Which model performs best overall?
- Which model generalizes better to unseen data?
- Which model has better precision/recall balance?
- Which model is more practical for deployment?
- What errors appear most often?
