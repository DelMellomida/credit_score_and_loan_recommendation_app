"""
Scaling Method Enumeration
==========================

Defines the supported scaling methods for feature normalization.
"""

from enum import Enum


class ScalingMethod(Enum):
    """Supported scaling methods for feature normalization."""
    LOG_MINMAX = "log_minmax"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"