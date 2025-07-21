# """
# Credit Scoring Configuration Module
# ===================================

# This module provides a comprehensive configuration system for credit scoring with:
# - Client-type specific configurations (NEW vs RENEWING)
# - Feature-level contribution caps to prevent data leakage
# - Component-based architecture for modular scoring
# - Mathematical isolation of features

# Main Classes:
# - EnhancedCreditScoringConfig: Master configuration class
# - ClientTypeConfig: Configuration for specific client types
# - ComponentConfig: Configuration for scoring components
# - FeatureConfig: Configuration for individual features

# Example Usage:
#     from config import EnhancedCreditScoringConfig
    
#     # Initialize the configuration
#     config = EnhancedCreditScoringConfig()
    
#     # Get configuration for new clients
#     new_client_config = config.get_client_config(ClientType.NEW)
    
#     # Access feature caps summary
#     caps_summary = config.get_feature_caps_summary()
# """

# # Import all configuration classes
# from .client_type_config import ClientTypeConfig
# from .component_config import ComponentConfig
# from .enhanced_config import EnhancedCreditScoringConfig
# from .feature_config import FeatureConfig

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Define what should be imported with "from config import *"
# __all__ = [
#     'ClientTypeConfig',
#     'ComponentConfig', 
#     'EnhancedCreditScoringConfig',
#     'FeatureConfig',
# ]

# # Convenience function to create default configuration
# def create_default_config():
#     """
#     Create and return a default enhanced credit scoring configuration.
    
#     Returns:
#         EnhancedCreditScoringConfig: Initialized configuration with default settings
#     """
#     return EnhancedCreditScoringConfig()


# # Module-level configuration instance (singleton pattern)
# _default_config = None

# def get_default_config():
#     """
#     Get the default configuration instance (singleton).
    
#     Returns:
#         EnhancedCreditScoringConfig: The default configuration instance
#     """
#     global _default_config
#     if _default_config is None:
#         _default_config = EnhancedCreditScoringConfig()
#     return _default_config