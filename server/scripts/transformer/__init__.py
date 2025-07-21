# """
# Credit Scoring Transformers Module
# ==================================

# Main transformation pipeline for the enhanced credit scoring system.

# This module provides the core transformation logic that orchestrates the entire
# credit scoring process. It handles client-type specific scoring, feature isolation,
# and provides detailed explanations for credit decisions.

# Architecture Overview:
#     EnhancedCreditScoringTransformer
#     ├── Detects client type (New vs Renewing)
#     ├── Applies appropriate scoring configuration
#     ├── Orchestrates component scorers
#     ├── Enforces feature and component caps
#     └── Generates interpretable explanations

# Key Features:
# - Client-type specific scoring logic
# - Mathematical feature isolation with strict caps
# - Component-based architecture for modular scoring
# - Comprehensive score explanations
# - Data leakage prevention through constrained mappings
# - Transparent feature importance analysis

# Classes:
#     EnhancedCreditScoringTransformer: Main transformer with complete scoring pipeline

# Example Usage:
#     from transformers import EnhancedCreditScoringTransformer
#     import pandas as pd
    
#     # Initialize transformer
#     transformer = EnhancedCreditScoringTransformer()
    
#     # Transform dataset
#     df_transformed = transformer.transform(df)
    
#     # Get feature importance
#     importance = transformer.get_feature_importance()
    
#     # Get detailed explanation for a single applicant
#     explanation = transformer.get_score_explanation(df.iloc[0])
    
# Score Interpretation:
#     0.80 - 1.00: Excellent credit risk (High approval likelihood)
#     0.70 - 0.79: Good credit risk (Favorable terms likely)
#     0.60 - 0.69: Acceptable credit risk (Standard terms)
#     0.50 - 0.59: Marginal credit risk (Review required)
#     0.40 - 0.49: Poor credit risk (Higher risk/rates)
#     0.00 - 0.39: Very poor credit risk (Decline recommended)
# """

# from .enhanced_transformer import EnhancedCreditScoringTransformer

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Export main transformer class
# __all__ = [
#     'EnhancedCreditScoringTransformer',
# ]

# # Score interpretation thresholds
# SCORE_THRESHOLDS = {
#     'excellent': 0.80,
#     'good': 0.70,
#     'acceptable': 0.60,
#     'marginal': 0.50,
#     'poor': 0.40,
#     'very_poor': 0.00
# }

# # Component weight configurations by client type
# CLIENT_TYPE_WEIGHTS = {
#     'new': {
#         'financial_stability': 0.80,
#         'cultural_context': 0.20,
#         'credit_behavior': 0.00  # Not used for new clients
#     },
#     'renewing': {
#         'financial_stability': 0.37,
#         'cultural_context': 0.03,
#         'credit_behavior': 0.60
#     }
# }

# # Factory function to create transformer
# def create_transformer(config=None):
#     """
#     Create an enhanced credit scoring transformer.
    
#     Args:
#         config: Optional EnhancedCreditScoringConfig instance
#                 If None, uses default configuration
    
#     Returns:
#         EnhancedCreditScoringTransformer: Configured transformer instance
#     """
#     return EnhancedCreditScoringTransformer(config)


# # Utility function to interpret score
# def interpret_score(score):
#     """
#     Get human-readable interpretation of a credit risk score.
    
#     Args:
#         score (float): Credit risk score between 0 and 1
        
#     Returns:
#         tuple: (category, description) where category is the risk level
#                and description is the detailed interpretation
#     """
#     if score >= SCORE_THRESHOLDS['excellent']:
#         return ('excellent', 'Excellent credit risk profile - High approval likelihood')
#     elif score >= SCORE_THRESHOLDS['good']:
#         return ('good', 'Good credit risk profile - Favorable terms likely')
#     elif score >= SCORE_THRESHOLDS['acceptable']:
#         return ('acceptable', 'Acceptable credit risk profile - Standard terms')
#     elif score >= SCORE_THRESHOLDS['marginal']:
#         return ('marginal', 'Marginal credit risk profile - Review required')
#     elif score >= SCORE_THRESHOLDS['poor']:
#         return ('poor', 'Poor credit risk profile - Higher risk/rates')
#     else:
#         return ('very_poor', 'Very poor credit risk profile - Decline recommended')


# # Validation function
# def validate_input_dataframe(df):
#     """
#     Validate that input DataFrame has required columns for transformation.
    
#     Args:
#         df (pd.DataFrame): Input dataframe to validate
        
#     Returns:
#         tuple: (is_valid, error_messages) where is_valid is boolean
#                and error_messages is list of validation errors
#     """
#     required_columns = [
#         'Is_Renewing_Client',
#         'Net_Salary_Per_Cutoff',
#         'Number_of_Dependents'
#     ]
    
#     errors = []
    
#     # Check for required columns
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         errors.append(f"Missing required columns: {missing_columns}")
    
#     # Check data types
#     if 'Is_Renewing_Client' in df.columns:
#         if not df['Is_Renewing_Client'].isin([0, 1]).all():
#             errors.append("'Is_Renewing_Client' must contain only 0 or 1 values")
    
#     # Check for negative values in numeric columns
#     numeric_columns = ['Net_Salary_Per_Cutoff', 'Comaker_Net_Salary_Per_Cutoff', 
#                       'Employment_Tenure_Months', 'Years_at_Current_Address']
#     for col in numeric_columns:
#         if col in df.columns and (df[col] < 0).any():
#             errors.append(f"'{col}' contains negative values")
    
#     return (len(errors) == 0, errors)


# # Summary statistics function
# def get_transformation_summary(df_transformed):
#     """
#     Get summary statistics from transformed dataframe.
    
#     Args:
#         df_transformed (pd.DataFrame): Transformed dataframe from transformer
        
#     Returns:
#         dict: Summary statistics including score distributions and client counts
#     """
#     summary = {
#         'total_records': len(df_transformed),
#         'client_type_distribution': df_transformed['Client_Type'].value_counts().to_dict(),
#         'score_statistics': {
#             'mean': df_transformed['Credit_Risk_Score'].mean(),
#             'median': df_transformed['Credit_Risk_Score'].median(),
#             'std': df_transformed['Credit_Risk_Score'].std(),
#             'min': df_transformed['Credit_Risk_Score'].min(),
#             'max': df_transformed['Credit_Risk_Score'].max()
#         },
#         'score_distribution': {},
#         'component_correlations': {}
#     }
    
#     # Score distribution by category
#     for _, row in df_transformed.iterrows():
#         category, _ = interpret_score(row['Credit_Risk_Score'])
#         summary['score_distribution'][category] = summary['score_distribution'].get(category, 0) + 1
    
#     # Component correlations
#     score_columns = ['Credit_Behavior_Score', 'Financial_Stability_Score', 
#                     'Cultural_Context_Score', 'Credit_Risk_Score']
#     existing_columns = [col for col in score_columns if col in df_transformed.columns]
#     if len(existing_columns) > 1:
#         summary['component_correlations'] = df_transformed[existing_columns].corr().to_dict()
    
#     return summary