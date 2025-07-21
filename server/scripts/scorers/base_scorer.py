"""
Base Component Scorer
====================

Abstract base class for component scoring with mathematical isolation.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

from ..config.component_config import ComponentConfig
from ..processors.feature_processor import FeatureProcessor


class ComponentScorer(ABC):
    """Abstract base class for component scoring with mathematical isolation."""
    
    def __init__(self, component_config: ComponentConfig, processor: FeatureProcessor):
        self.config = component_config
        self.processor = processor
    
    @abstractmethod
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate component score with feature isolation."""
        pass
    
    def _apply_feature_caps(self, feature_scores: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Apply individual feature caps to prevent any feature from dominating."""
        capped_scores = {}
        
        for feature_name, scores in feature_scores.items():
            feature_config = self.config.features[feature_name]
            max_cap = feature_config.max_contribution_pct
            
            # Cap the feature contribution
            capped_scores[feature_name] = pd.Series(
                np.clip(scores, 0, max_cap), 
                index=scores.index
            )
        
        return capped_scores
    
    def _normalize_component_score(self, combined_score: pd.Series) -> pd.Series:
        """Normalize component score and apply component-level cap."""
        # Normalize to 0-1 range first
        if combined_score.max() > 0:
            normalized = combined_score / combined_score.max()
        else:
            normalized = combined_score
        
        # Apply component-level cap
        component_cap = self.config.max_contribution_pct
        final_score = pd.Series(
            np.clip(normalized * component_cap, 0, component_cap),
            index=combined_score.index
        )
        
        return final_score