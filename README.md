# ICRSeg

ICRSeg implements intervention-calibrated risk learning for abdominal multi-organ segmentation with partially labeled data.

The method combines source-style intervention, EMA and historical teachers, and a region risk estimator. Pseudo labels for unknown organs are selected using quality, precision, historical stability, and per-organ quotas. Training combines supervised, intervention, consistency, pseudo-label, and overlap exclusion losses. Inference uses the student segmentation network.

Core implementation:

- `train.py`: training workflow and default hyperparameters.
- `infer.py`: inference and evaluation.
- `icrseg/nnunet_wrapper.py`: model wrapper and source-style intervention.
- `icrseg/risk.py`: region risk estimation and pseudo-label selection.
- `icrseg/losses.py`: training losses.

Default hyperparameters are defined in `train.py`. Organ-specific values follow the order liver, spleen, kidney, pancreas.

This repository provides the core method source. Execution requires external nnU-Net backbone construction code and project dependencies.
