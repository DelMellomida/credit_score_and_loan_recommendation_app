"""
Helper Utilities
================

Utility functions for data conversion and manipulation.
"""

import logging
import pandas as pd
from typing import Any, Dict

from .validation import validate_loan_application_schema

logger = logging.getLogger(__name__)


def create_loan_application_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create DataFrame from LoanApplicationRequest-compatible dictionary.
    
    Useful for converting API requests to DataFrame format.
    """
    # Ensure all required fields are present with defaults
    defaults = {
        'Employment_Sector': 'Private',
        'Employment_Tenure_Months': 12,
        'Net_Salary_Per_Cutoff': 15000.0,
        'Salary_Frequency': 'Monthly',
        'Housing_Status': 'Rented',
        'Years_at_Current_Address': 1.0,
        'Household_Head': 'No',
        'Number_of_Dependents': 0,
        'Comaker_Relationship': 'Friend',
        'Comaker_Employment_Tenure_Months': 12,
        'Comaker_Net_Salary_Per_Cutoff': 10000.0,
        'Has_Community_Role': 'No',
        'Paluwagan_Participation': 'No',
        'Other_Income_Source': 'None',
        'Disaster_Preparedness': 'None',
        'Is_Renewing_Client': 0,
        'Grace_Period_Usage_Rate': 0.0,
        'Late_Payment_Count': 0,
        'Had_Special_Consideration': 0
    }
    
    # Merge with defaults
    complete_data = {**defaults, **data}
    
    # Create DataFrame
    df = pd.DataFrame([complete_data])
    
    # Validate and fix any issues
    df_fixed, issues = validate_loan_application_schema(df)
    
    if issues:
        logger.warning(f"Schema issues found and fixed: {issues}")
    
    return df_fixed