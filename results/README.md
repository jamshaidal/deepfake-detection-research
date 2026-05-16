# Results

This folder will track evaluation summaries for CNN and Transformer-based deepfake detection experiments.

## Purpose

Results should be reported clearly so that model performance, limitations, and next steps are easy to understand.

## Metrics To Report

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Inference time
- Model size
- Notes on false positives and false negatives

## Results Table Template

| Experiment | Model Type | Accuracy | Precision | Recall | F1-score | ROC-AUC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn-baseline-v1 | CNN | Pending | Pending | Pending | Pending | Pending | Initial baseline |
| vision-transformer-v1 | Transformer | Pending | Pending | Pending | Pending | Pending | Planned comparison |

## Error Analysis

For each model, document:

- Examples where fake media was classified as real
- Examples where real media was classified as fake
- Impact of compression, blur, lighting, or face angle
- Whether the model overfits to dataset-specific artifacts

## Reporting Principle

Do not report inflated or unsupported performance. Results should be based on reproducible experiments and clearly documented evaluation settings.
