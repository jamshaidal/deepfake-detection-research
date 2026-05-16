# Dataset Documentation

This folder documents planned dataset sources, preprocessing decisions, and ethical notes for deepfake detection research.

## Purpose

Deepfake detection research depends on careful dataset handling. This document will track which datasets are considered, how they are prepared, and what limitations may affect model evaluation.

## Dataset Selection Criteria

- Publicly available or clearly permitted for research use
- Suitable for real vs manipulated media classification
- Includes enough diversity in identities, lighting, quality, and manipulation type
- Can support fair train, validation, and test splits
- Has documentation about source, license, and intended use

## Planned Dataset Notes

For each dataset, record:

- Dataset name
- Source link
- License or access rules
- Number of real samples
- Number of fake samples
- Manipulation types
- Resolution and compression details
- Known limitations or bias risks

## Preprocessing Plan

- Extract video frames where needed
- Apply face detection or center-crop strategy if appropriate
- Normalize image size for CNN and Transformer models
- Keep preprocessing settings consistent across experiments
- Store split information for reproducibility

## Ethical Considerations

Deepfake datasets may contain sensitive faces or manipulated media. This research should use permitted datasets, avoid exposing private data, and present the work as detection and media-authenticity research.
