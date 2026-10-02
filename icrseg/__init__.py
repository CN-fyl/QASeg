"""ICRSeg: intervention-calibrated risk learning for partial-label segmentation."""

from .nnunet_wrapper import ICRSegNNUNet
from .risk import RiskEstimator

__all__ = ["ICRSegNNUNet", "RiskEstimator"]
