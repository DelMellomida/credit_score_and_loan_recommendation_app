"""
Credit Behavior Scorer
======================

Credit behavior scorer for renewing clients only.
"""

import pandas as pd

from .base_scorer import ComponentScorer


class CreditBehaviorScorer(ComponentScorer):
    """Credit behavior scorer for renewing clients only."""
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate credit behavior score for renewing clients."""
        feature_scores = {}
        
        # Payment history score (inverted - lower late payments = higher score)
        if 'Late_Payment_Count' in df.columns:
            late_payments = df['Late_Payment_Count'].fillna(0)
            max_late = max(late_payments.max(), 1)
            payment_score = 1.0 - (late_payments / max_late)
            
            feature_scores['payment_history'] = self.processor.process_numerical_feature(
                payment_score, 
                self.config.features['payment_history'],
                'payment_history'
            )
        
        # Grace period usage score (inverted - lower usage = higher score)
        if 'Grace_Period_Usage_Rate' in df.columns:
            grace_usage = df['Grace_Period_Usage_Rate'].fillna(0)
            grace_score = 1.0 - grace_usage
            
            feature_scores['grace_period_usage'] = self.processor.process_numerical_feature(
                grace_score,
                self.config.features['grace_period_usage'],
                'grace_usage'
            )
        
        # Special considerations penalty (inverted)
        if 'Had_Special_Consideration' in df.columns:
            special_considerations = df['Had_Special_Consideration'].fillna(0)
            special_score = 1.0 - special_considerations
            
            feature_scores['special_considerations'] = special_score * \
                self.config.features['special_considerations'].max_contribution_pct
        
        # Client loyalty bonus
        feature_scores['client_loyalty'] = pd.Series(
            [self.config.features['client_loyalty'].max_contribution_pct] * len(df),
            index=df.index
        )
        
        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            weight = self.config.features[feature_name].weight_in_component
            combined_score += scores * weight
        
        return self._normalize_component_score(combined_score)