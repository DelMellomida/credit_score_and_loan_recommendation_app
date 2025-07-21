"""
Cultural Context Scorer
======================

Cultural context scorer with severe data leakage prevention.
Uses heavily constrained cultural features to prevent bias and leakage.
"""

import numpy as np
import pandas as pd

from .base_scorer import ComponentScorer
from ..config.feature_config import FeatureConfig


class CulturalContextScorer(ComponentScorer):
    """
    Cultural context scorer with severe data leakage prevention.
    
    Uses heavily constrained cultural features to prevent bias and leakage.
    """
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cultural context score with severe leakage constraints."""
        feature_scores = {}
        
        # Financial discipline (Paluwagan participation - SEVERELY CONSTRAINED)
        if 'Paluwagan_Participation' in df.columns:
            feature_scores['financial_discipline'] = self.processor.process_categorical_feature(
                df['Paluwagan_Participation'],
                self.config.features['financial_discipline']
            )
        
        # Community integration (Community role - SEVERELY CONSTRAINED)  
        if 'Has_Community_Role' in df.columns:
            feature_scores['community_integration'] = self.processor.process_categorical_feature(
                df['Has_Community_Role'],
                self.config.features['community_integration']
            )
        
        # Family stability (Housing status + Household head)
        family_stability_score = pd.Series([0.0] * len(df), index=df.index)
        
        if 'Housing_Status' in df.columns:
            housing_scores = self.processor.process_categorical_feature(
                df['Housing_Status'],
                FeatureConfig(
                    name="housing_temp",
                    max_contribution_pct=0.5,
                    weight_in_component=0.5,
                    categorical_mapping=self.processor.config.leakage_mitigation_features['housing_status']
                )
            )
            family_stability_score += housing_scores * 0.5
        
        if 'Household_Head' in df.columns:
            head_scores = self.processor.process_categorical_feature(
                df['Household_Head'],
                FeatureConfig(
                    name="head_temp",
                    max_contribution_pct=0.5,
                    weight_in_component=0.5,
                    categorical_mapping=self.processor.config.leakage_mitigation_features['household_head']
                )
            )
            family_stability_score += head_scores * 0.5
        
        feature_scores['family_stability'] = family_stability_score
        
        # Income diversification (Other income source)
        if 'Other_Income_Source' in df.columns:
            feature_scores['income_diversification'] = self.processor.process_categorical_feature(
                df['Other_Income_Source'],
                self.config.features['income_diversification']
            )
        
        # Relationship strength (Comaker relationship)
        if 'Comaker_Relationship' in df.columns:
            feature_scores['relationship_strength'] = self.processor.process_categorical_feature(
                df['Comaker_Relationship'],
                self.config.features['relationship_strength']
            )
        
        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            if feature_name in self.config.features:
                weight = self.config.features[feature_name].weight_in_component
                combined_score += scores * weight
        
        # Additional penalty for dependents (capped)
        if 'Number_of_Dependents' in df.columns:
            dependents = df['Number_of_Dependents'].fillna(0)
            dependents_penalty = np.clip(dependents * 0.01, 0, 0.05)  # Max 5% penalty
            combined_score -= dependents_penalty
        
        # Apply component-level normalization and cap
        return self._normalize_component_score(combined_score)