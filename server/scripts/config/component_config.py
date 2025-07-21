"""
Component Configuration
======================

Configuration for scoring components with mathematical feature isolation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

from .feature_config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass  
class ComponentConfig:
    """Configuration for scoring components with mathematical feature isolation."""
    name: str
    max_contribution_pct: float         # Maximum % this component can contribute to final score
    features: Dict[str, FeatureConfig] = field(default_factory=dict)
    normalization_method: str = "weighted_capped"
    component_description: str = ""
    
    def add_feature(self, feature_config: FeatureConfig) -> None:
        """Add feature with validation and automatic weight normalization."""
        self.features[feature_config.name] = feature_config
        
        # Validate and normalize weights to sum to 1.0
        total_weight = sum(f.weight_in_component for f in self.features.values())
        if total_weight > 1.0:
            # Auto-normalize weights
            for feature_name, feature in self.features.items():
                feature.weight_in_component /= total_weight
            logger.info(f"Component {self.name}: Auto-normalized feature weights to sum to 1.0")
    
    def get_max_individual_contribution(self) -> float:
        """Get the maximum any single feature can contribute to this component."""
        if not self.features:
            return 0.0
        return max(f.max_contribution_pct for f in self.features.values())