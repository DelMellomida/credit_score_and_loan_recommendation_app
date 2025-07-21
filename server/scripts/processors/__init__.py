# """
# Feature Processing Module
# ========================

# Advanced feature processing capabilities for the credit scoring system.

# This module provides sophisticated feature transformation and scaling methods
# with built-in handling for outliers, missing values, and categorical variables.

# Key Features:
# - Multiple scaling methods (log-minmax, minmax, z-score, robust)
# - Outlier detection and handling
# - Missing value imputation strategies
# - Categorical feature mapping
# - Data-driven scaling ranges
# - Mathematical isolation of features

# Classes:
#     FeatureProcessor: Advanced feature processor with mathematical isolation and robust scaling

# Example Usage:
#     from processing import FeatureProcessor
#     from config import EnhancedCreditScoringConfig
    
#     # Initialize configuration and processor
#     config = EnhancedCreditScoringConfig()
#     processor = FeatureProcessor(config)
    
#     # Process numerical feature
#     scaled_values = processor.process_numerical_feature(
#         values=df['income'],
#         feature_config=feature_config,
#         range_key='net_salary'
#     )
    
#     # Process categorical feature
#     mapped_values = processor.process_categorical_feature(
#         values=df['employment_sector'],
#         feature_config=feature_config
#     )
# """

# from .feature_processor import FeatureProcessor

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Export main class
# __all__ = [
#     'FeatureProcessor',
# ]

# # Supported scaling methods (for reference)
# SUPPORTED_SCALING_METHODS = [
#     'log_minmax',   # Logarithmic transformation followed by min-max scaling
#     'minmax',       # Standard min-max scaling to [0, 1]
#     'zscore',       # Z-score normalization with sigmoid transformation
#     'robust',       # Robust scaling using median and IQR
# ]

# # Supported missing value strategies
# SUPPORTED_MISSING_STRATEGIES = [
#     'median',       # Fill with median value
#     'mean',         # Fill with mean value
#     'zero',         # Fill with zero
#     'mode',         # Fill with mode (most common value)
# ]

# # Utility function to create a configured processor
# def create_feature_processor(config=None):
#     """
#     Create a feature processor with the given configuration.
    
#     Args:
#         config: EnhancedCreditScoringConfig instance (optional)
#                 If None, creates a default configuration
    
#     Returns:
#         FeatureProcessor: Configured feature processor instance
#     """
#     if config is None:
#         from ..config import EnhancedCreditScoringConfig
#         config = EnhancedCreditScoringConfig()
    
#     return FeatureProcessor(config)


# # Validation helpers
# def validate_scaling_method(method):
#     """
#     Validate that a scaling method is supported.
    
#     Args:
#         method: String name of scaling method
        
#     Raises:
#         ValueError: If method is not supported
#     """
#     if method not in SUPPORTED_SCALING_METHODS:
#         raise ValueError(
#             f"Unsupported scaling method: {method}. "
#             f"Must be one of {SUPPORTED_SCALING_METHODS}"
#         )


# def validate_missing_strategy(strategy):
#     """
#     Validate that a missing value strategy is supported.
    
#     Args:
#         strategy: String name of missing value strategy
        
#     Raises:
#         ValueError: If strategy is not supported
#     """
#     if strategy not in SUPPORTED_MISSING_STRATEGIES:
#         raise ValueError(
#             f"Unsupported missing value strategy: {strategy}. "
#             f"Must be one of {SUPPORTED_MISSING_STRATEGIES}"
#         )