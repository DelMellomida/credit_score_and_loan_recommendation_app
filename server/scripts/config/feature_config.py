"""
Feature Configuration
====================

Defines configuration for individual features with strict caps and constraints.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..enums.scaling_method import ScalingMethod


@dataclass
class FeatureConfig:
    """Configuration for individual feature with strict caps and constraints."""
    name: str
    max_contribution_pct: float         # Maximum % this feature can contribute to its component
    weight_in_component: float          # Weight within its component (0.0 to 1.0)
    scaling_method: ScalingMethod = ScalingMethod.LOG_MINMAX
    min_value: float = 0.0
    max_value: float = 1.0
    handle_outliers: bool = True
    outlier_percentiles: Tuple[float, float] = (0.05, 0.95)
    categorical_mapping: Optional[Dict[str, float]] = None
    missing_value_strategy: str = "median"  # median, mean, mode, zero
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if not 0 <= self.weight_in_component <= 1.0:
            raise ValueError(f"Feature {self.name}: weight_in_component must be between 0 and 1")
        if not 0 <= self.max_contribution_pct <= 1.0:
            raise ValueError(f"Feature {self.name}: max_contribution_pct must be between 0 and 1")