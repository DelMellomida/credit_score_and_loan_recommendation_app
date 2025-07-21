# """
# Credit Scoring Utilities Module
# ===============================

# Utility functions for the enhanced credit scoring system.

# This module provides essential utility functions for data validation, feature management,
# and data conversion to ensure consistency and reliability throughout the credit scoring
# pipeline.

# Components:
#     - Feature Management: Get lists of available features and their metadata
#     - Data Validation: Validate dataframes against the loan application schema
#     - Data Conversion: Convert dictionaries to properly formatted dataframes

# Functions:
#     get_available_features(): Returns comprehensive feature lists and metadata
#     validate_loan_application_schema(): Validates and fixes dataframe schema issues
#     create_loan_application_from_dict(): Converts API requests to dataframes

# Example Usage:
#     from utils import get_available_features, validate_loan_application_schema
#     from utils import create_loan_application_from_dict
    
#     # Get available features
#     features = get_available_features()
#     input_features = features['input_features']
    
#     # Validate dataframe
#     df_validated, issues = validate_loan_application_schema(df)
    
#     # Convert API request to dataframe
#     loan_data = {'Net_Salary_Per_Cutoff': 25000, 'Employment_Tenure_Months': 24}
#     df = create_loan_application_from_dict(loan_data)
# """

# from .features import get_available_features
# from .helpers import create_loan_application_from_dict
# from .validation import validate_loan_application_schema

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Export main functions
# __all__ = [
#     'get_available_features',
#     'validate_loan_application_schema',
#     'create_loan_application_from_dict',
# ]

# # Feature categories for quick reference
# FEATURE_CATEGORIES = {
#     'demographic': [
#         'Housing_Status', 'Household_Head', 'Number_of_Dependents',
#         'Years_at_Current_Address'
#     ],
#     'employment': [
#         'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
#         'Salary_Frequency'
#     ],
#     'comaker': [
#         'Comaker_Relationship', 'Comaker_Employment_Tenure_Months',
#         'Comaker_Net_Salary_Per_Cutoff'
#     ],
#     'cultural': [
#         'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
#         'Disaster_Preparedness'
#     ],
#     'credit_history': [
#         'Is_Renewing_Client', 'Grace_Period_Usage_Rate', 'Late_Payment_Count',
#         'Had_Special_Consideration'
#     ]
# }

# # Default values for loan applications
# DEFAULT_LOAN_APPLICATION_VALUES = {
#     'Employment_Sector': 'Private',
#     'Employment_Tenure_Months': 12,
#     'Net_Salary_Per_Cutoff': 15000.0,
#     'Salary_Frequency': 'Monthly',
#     'Housing_Status': 'Rented',
#     'Years_at_Current_Address': 1.0,
#     'Household_Head': 'No',
#     'Number_of_Dependents': 0,
#     'Comaker_Relationship': 'Friend',
#     'Comaker_Employment_Tenure_Months': 12,
#     'Comaker_Net_Salary_Per_Cutoff': 10000.0,
#     'Has_Community_Role': 'No',
#     'Paluwagan_Participation': 'No',
#     'Other_Income_Source': 'None',
#     'Disaster_Preparedness': 'None',
#     'Is_Renewing_Client': 0,
#     'Grace_Period_Usage_Rate': 0.0,
#     'Late_Payment_Count': 0,
#     'Had_Special_Consideration': 0
# }

# # Categorical value mappings
# CATEGORICAL_VALUES = {
#     'Employment_Sector': ['Public', 'Private'],
#     'Salary_Frequency': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly'],
#     'Housing_Status': ['Owned', 'Rented'],
#     'Household_Head': ['Yes', 'No'],
#     'Comaker_Relationship': ['Friend', 'Parent', 'Sibling', 'Spouse'],
#     'Has_Community_Role': ['Yes', 'No'],
#     'Paluwagan_Participation': ['Yes', 'No'],
#     'Other_Income_Source': ['None', 'Freelance', 'Business', 'OFW Remittance'],
#     'Disaster_Preparedness': ['None', 'Savings', 'Insurance', 'Community Plan']
# }

# # Utility function to get feature by category
# def get_features_by_category(category):
#     """
#     Get list of features for a specific category.
    
#     Args:
#         category (str): Category name ('demographic', 'employment', 'comaker', 
#                        'cultural', 'credit_history')
    
#     Returns:
#         list: Features in the specified category
        
#     Raises:
#         ValueError: If category is not recognized
#     """
#     if category not in FEATURE_CATEGORIES:
#         raise ValueError(
#             f"Unknown category: {category}. "
#             f"Valid categories are: {list(FEATURE_CATEGORIES.keys())}"
#         )
#     return FEATURE_CATEGORIES[category]


# # Utility function to get valid values for categorical features
# def get_categorical_values(feature_name):
#     """
#     Get valid values for a categorical feature.
    
#     Args:
#         feature_name (str): Name of the categorical feature
        
#     Returns:
#         list: Valid values for the feature
        
#     Raises:
#         ValueError: If feature is not categorical
#     """
#     if feature_name not in CATEGORICAL_VALUES:
#         raise ValueError(
#             f"'{feature_name}' is not a categorical feature or is not recognized"
#         )
#     return CATEGORICAL_VALUES[feature_name]


# # Check if value is valid for a categorical feature
# def is_valid_categorical_value(feature_name, value):
#     """
#     Check if a value is valid for a categorical feature.
    
#     Args:
#         feature_name (str): Name of the categorical feature
#         value: Value to check
        
#     Returns:
#         bool: True if valid, False otherwise
#     """
#     try:
#         valid_values = get_categorical_values(feature_name)
#         return value in valid_values
#     except ValueError:
#         return False


# # Get feature type
# def get_feature_type(feature_name):
#     """
#     Get the type of a feature (categorical, numeric, binary).
    
#     Args:
#         feature_name (str): Name of the feature
        
#     Returns:
#         str: Feature type ('categorical', 'numeric', 'binary', or 'unknown')
#     """
#     if feature_name in CATEGORICAL_VALUES:
#         return 'categorical'
#     elif feature_name in ['Is_Renewing_Client', 'Had_Special_Consideration']:
#         return 'binary'
#     elif feature_name in ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
#                          'Years_at_Current_Address', 'Number_of_Dependents',
#                          'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
#                          'Grace_Period_Usage_Rate', 'Late_Payment_Count']:
#         return 'numeric'
#     else:
#         return 'unknown'


# # Summary function for feature metadata
# def get_feature_metadata():
#     """
#     Get comprehensive metadata about all features.
    
#     Returns:
#         dict: Feature metadata including types, categories, and valid values
#     """
#     metadata = {}
#     all_features = get_available_features()['input_features']
    
#     for feature in all_features:
#         feature_info = {
#             'type': get_feature_type(feature),
#             'category': None,
#             'default_value': DEFAULT_LOAN_APPLICATION_VALUES.get(feature),
#             'valid_values': None
#         }
        
#         # Find category
#         for cat, features in FEATURE_CATEGORIES.items():
#             if feature in features:
#                 feature_info['category'] = cat
#                 break
        
#         # Add valid values for categorical
#         if feature in CATEGORICAL_VALUES:
#             feature_info['valid_values'] = CATEGORICAL_VALUES[feature]
        
#         metadata[feature] = feature_info
    
#     return metadata