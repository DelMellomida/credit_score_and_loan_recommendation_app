"""
Financial Stability Scorer
=========================

Financial stability scorer for both client types.
Uses objective financial data with robust scaling and feature isolation.
"""

import pandas as pd
from typing import Dict

from .base_scorer import ComponentScorer


class FinancialStabilityScorer(ComponentScorer):
    """
    Financial stability scorer for both client types.
    
    Uses objective financial data with robust scaling and feature isolation.
    """
    
    def calculate_component_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate financial stability score with feature isolation."""
        feature_scores = {}
        
        # Income adequacy (primary income per household member)
        if 'Net_Salary_Per_Cutoff' in df.columns:
            primary_income = df['Net_Salary_Per_Cutoff'].fillna(0)
            comaker_income = df['Comaker_Net_Salary_Per_Cutoff'].fillna(0)
            dependents = df['Number_of_Dependents'].fillna(0)
            has_comaker = (comaker_income > 0).astype(int)
            
            household_size = 1 + dependents + has_comaker
            income_per_member = (primary_income + comaker_income) / household_size
            
            feature_scores['income_adequacy'] = self.processor.process_numerical_feature(
                income_per_member,
                self.config.features['income_adequacy'],
                'net_salary'
            )
        
        # Household income capacity (total household income)
        if 'Net_Salary_Per_Cutoff' in df.columns and 'Comaker_Net_Salary_Per_Cutoff' in df.columns:
            total_household_income = df['Net_Salary_Per_Cutoff'].fillna(0) + \
                                   df['Comaker_Net_Salary_Per_Cutoff'].fillna(0)
            
            feature_scores['household_income_capacity'] = self.processor.process_numerical_feature(
                total_household_income,
                self.config.features['household_income_capacity'],
                'household_income'
            )
        
        # Employment stability
        if 'Employment_Tenure_Months' in df.columns:
            employment_tenure = df['Employment_Tenure_Months'].fillna(0)
            
            feature_scores['employment_stability'] = self.processor.process_numerical_feature(
                employment_tenure,
                self.config.features['employment_stability'],
                'employment_tenure'
            )
        
        # Address stability
        if 'Years_at_Current_Address' in df.columns:
            address_years = df['Years_at_Current_Address'].fillna(0)
            
            feature_scores['address_stability'] = self.processor.process_numerical_feature(
                address_years,
                self.config.features['address_stability'],
                'address_stability'
            )
        
        # Sector stability (if included)
        if 'sector_stability' in self.config.features and 'Employment_Sector' in df.columns:
            feature_scores['sector_stability'] = self.processor.process_categorical_feature(
                df['Employment_Sector'],
                self.config.features['sector_stability']
            )
        
        # Apply feature caps
        capped_scores = self._apply_feature_caps(feature_scores)
        
        # Combine features with weights
        combined_score = pd.Series([0.0] * len(df), index=df.index)
        for feature_name, scores in capped_scores.items():
            if feature_name in self.config.features:
                weight = self.config.features[feature_name].weight_in_component
                combined_score += scores * weight
        
        # Apply component-level normalization and cap
        return self._normalize_component_score(combined_score)