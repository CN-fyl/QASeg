# QASeg

QASeg (Quality-Aware Segmentation) trains a single model for abdominal multi-organ segmentation from cross-source, partially labeled CT datasets.

The method uses mask-aligned cross-source feature-statistic intervention and learns to estimate the regional overlap of teacher predictions. The quality estimates help select and weight pseudo-labels for unannotated organs. Inference uses the student segmentation network to predict the liver, spleen, bilateral kidneys, and pancreas.

Core implementation:

- `train.py`: training workflow and default hyperparameters.
- `infer.py`: inference and evaluation.
- `qaseg/nnunet_wrapper.py`: model wrapper and source-style intervention.
- `qaseg/risk.py`: regional quality estimation and pseudo-label selection.
- `qaseg/losses.py`: training losses.

Default hyperparameters are defined in `train.py`. Organ-specific values follow the order liver, spleen, kidney, pancreas.

This repository provides the core method source. Execution requires external nnU-Net backbone construction code and project dependencies.
