"""
Feature Processor
================

Advanced feature processor with mathematical isolation and robust scaling.
"""

import numpy as np
import pandas as pd

from ..config.enhanced_config import EnhancedCreditScoringConfig
from ..config.feature_config import FeatureConfig
from ..enums.scaling_method import ScalingMethod


class FeatureProcessor:
    """Advanced feature processor with mathematical isolation and robust scaling."""
    
    def __init__(self, config: EnhancedCreditScoringConfig):
        self.config = config
        self.scaling_cache = {}
        
    def process_numerical_feature(self, 
                                values: pd.Series, 
                                feature_config: FeatureConfig,
                                range_key: str) -> pd.Series:
        """Process numerical features with robust scaling and outlier handling."""
        # Handle missing values
        if feature_config.missing_value_strategy == "median":
            fill_value = values.median()
        elif feature_config.missing_value_strategy == "mean":
            fill_value = values.mean()
        elif feature_config.missing_value_strategy == "zero":
            fill_value = 0
        else:
            fill_value = values.median()
        
        processed_values = values.fillna(fill_value)
        
        # Handle outliers if specified
        if feature_config.handle_outliers:
            lower_pct, upper_pct = feature_config.outlier_percentiles
            lower_bound = processed_values.quantile(lower_pct)
            upper_bound = processed_values.quantile(upper_pct)
            processed_values = processed_values.clip(lower_bound, upper_bound)
        
        # Apply scaling method
        if feature_config.scaling_method == ScalingMethod.LOG_MINMAX:
            return self._log_minmax_scale(processed_values, range_key)
        elif feature_config.scaling_method == ScalingMethod.MINMAX:
            return self._minmax_scale(processed_values)
        elif feature_config.scaling_method == ScalingMethod.ZSCORE:
            return self._zscore_scale(processed_values)
        else:
            return self._robust_scale(processed_values)
    
    def _log_minmax_scale(self, values: pd.Series, range_key: str) -> pd.Series:
        """Log transform followed by min-max scaling using data-driven ranges."""
        if range_key not in self.config.data_driven_ranges:
            # Fallback to percentile-based scaling
            log_values = np.log1p(values)
            min_val = log_values.quantile(0.05)
            max_val = log_values.quantile(0.95)
        else:
            log_values = np.log1p(values)
            range_params = self.config.data_driven_ranges[range_key]
            min_val = range_params['min_log']
            max_val = range_params['max_log']
        
        if max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (log_values - min_val) / (max_val - min_val)
        return pd.Series(np.clip(scaled, 0, 1), index=values.index)
    
    def _minmax_scale(self, values: pd.Series) -> pd.Series:
        """Standard min-max scaling."""
        min_val = values.min()
        max_val = values.max()
        
        if max_val == min_val:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (values - min_val) / (max_val - min_val)
        return pd.Series(np.clip(scaled, 0, 1), index=values.index)
    
    def _zscore_scale(self, values: pd.Series) -> pd.Series:
        """Z-score normalization with sigmoid transformation."""
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val == 0:
            return pd.Series([0.5] * len(values), index=values.index)
        
        z_scores = (values - mean_val) / std_val
        sigmoid_values = 1 / (1 + np.exp(-z_scores))
        return pd.Series(sigmoid_values, index=values.index)
    
    def _robust_scale(self, values: pd.Series) -> pd.Series:
        """Robust scaling using median and IQR."""
        median_val = values.median()
        q75 = values.quantile(0.75)
        q25 = values.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            return pd.Series([0.5] * len(values), index=values.index)
        
        scaled = (values - median_val) / iqr
        sigmoid_values = 1 / (1 + np.exp(-scaled))
        return pd.Series(sigmoid_values, index=values.index)
    
    def process_categorical_feature(self, 
                                  values: pd.Series,
                                  feature_config: FeatureConfig) -> pd.Series:
        """Process categorical features with predefined mappings and missing value handling."""
        if feature_config.categorical_mapping is None:
            raise ValueError(f"Categorical feature {feature_config.name} requires categorical_mapping")
        
        # Handle missing values by mapping to lowest score
        min_score = min(feature_config.categorical_mapping.values())
        processed_values = values.fillna("__MISSING__")
        
        # Map values
        mapped_values = processed_values.map(feature_config.categorical_mapping)
        
        # Handle unmapped values (including missing)
        mapped_values = mapped_values.fillna(min_score)
        
        return mapped_values