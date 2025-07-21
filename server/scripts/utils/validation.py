"""
Schema Validation Utilities
===========================

Functions for validating dataframes against the loan application schema.
"""

import pandas as pd
from typing import List, Tuple


def validate_loan_application_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate dataframe against LoanApplicationRequest schema.
    
    Ensures compatibility with the API input schema.
    """
    required_columns = [
        'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
        'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
        'Household_Head', 'Number_of_Dependents', 'Comaker_Relationship',
        'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
        'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
        'Disaster_Preparedness', 'Is_Renewing_Client', 'Grace_Period_Usage_Rate',
        'Late_Payment_Count', 'Had_Special_Consideration'
    ]
    
    schema_rules = {
        'Employment_Sector': {'type': 'categorical', 'allowed': ['Public', 'Private']},
        'Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
        'Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
        'Salary_Frequency': {'type': 'categorical', 'allowed': ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']},
        'Housing_Status': {'type': 'categorical', 'allowed': ['Owned', 'Rented']},
        'Years_at_Current_Address': {'type': 'numeric', 'min': 0, 'dtype': 'float'},
        'Household_Head': {'type': 'categorical', 'allowed': ['Yes', 'No']},
        'Number_of_Dependents': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
        'Comaker_Relationship': {'type': 'categorical', 'allowed': ['Friend', 'Parent', 'Sibling', 'Spouse']},
        'Comaker_Employment_Tenure_Months': {'type': 'numeric', 'min': 1, 'dtype': 'int'},
        'Comaker_Net_Salary_Per_Cutoff': {'type': 'numeric', 'min': 0.01, 'dtype': 'float'},
        'Has_Community_Role': {'type': 'categorical', 'allowed': ['Yes', 'No']},
        'Paluwagan_Participation': {'type': 'categorical', 'allowed': ['Yes', 'No']},
        'Other_Income_Source': {'type': 'categorical', 'allowed': ['None', 'Freelance', 'Business', 'OFW Remittance']},
        'Disaster_Preparedness': {'type': 'categorical', 'allowed': ['None', 'Savings', 'Insurance', 'Community Plan']},
        'Is_Renewing_Client': {'type': 'binary', 'allowed': [0, 1]},
        'Grace_Period_Usage_Rate': {'type': 'numeric', 'min': 0.0, 'max': 1.0, 'dtype': 'float'},
        'Late_Payment_Count': {'type': 'numeric', 'min': 0, 'dtype': 'int'},
        'Had_Special_Consideration': {'type': 'binary', 'allowed': [0, 1]}
    }
    
    issues_found = []
    df_fixed = df.copy()
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues_found.extend([f"Missing required column: {col}" for col in missing_columns])
        return df_fixed, issues_found
    
    # Validate each column
    for col, rules in schema_rules.items():
        if col not in df.columns:
            continue
            
        if rules['type'] == 'categorical':
            invalid_values = df[col].dropna().unique()
            invalid_values = [v for v in invalid_values if v not in rules['allowed']]
            if invalid_values:
                issues_found.append(f"{col} has invalid values: {invalid_values}")
                # Fix by mapping to first allowed value
                df_fixed[col] = df_fixed[col].fillna(rules['allowed'][0])
                mask = ~df_fixed[col].isin(rules['allowed'])
                df_fixed.loc[mask, col] = rules['allowed'][0]
                
        elif rules['type'] in ['numeric', 'binary']:
            # Check minimum values
            if 'min' in rules:
                invalid_count = (df[col] < rules['min']).sum()
                if invalid_count > 0:
                    issues_found.append(f"{col} has {invalid_count} values below minimum {rules['min']}")
                    df_fixed[col] = df_fixed[col].clip(lower=rules['min'])
            
            # Check maximum values        
            if 'max' in rules:
                invalid_count = (df[col] > rules['max']).sum()
                if invalid_count > 0:
                    issues_found.append(f"{col} has {invalid_count} values above maximum {rules['max']}")
                    df_fixed[col] = df_fixed[col].clip(upper=rules['max'])
    
    return df_fixed, issues_found