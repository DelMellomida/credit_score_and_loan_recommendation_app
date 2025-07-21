"""
Feature Utilities
=================

Functions for managing feature lists and metadata.
"""

from typing import Dict, List


def get_available_features() -> Dict[str, List[str]]:
    """Get list of all available features in the enhanced system."""
    return {
        'input_features': [
            'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
            'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
            'Household_Head', 'Number_of_Dependents', 'Comaker_Relationship',
            'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff',
            'Has_Community_Role', 'Paluwagan_Participation', 'Other_Income_Source',
            'Disaster_Preparedness', 'Is_Renewing_Client', 'Grace_Period_Usage_Rate',
            'Late_Payment_Count', 'Had_Special_Consideration'
        ],
        'output_features': [
            'Credit_Behavior_Score', 'Financial_Stability_Score', 'Cultural_Context_Score',
            'Credit_Risk_Score', 'Client_Type'
        ],
        'component_features': {
            'credit_behavior': ['payment_history', 'grace_period_usage', 'special_considerations', 'client_loyalty'],
            'financial_stability': ['income_adequacy', 'household_income_capacity', 'employment_stability', 'address_stability', 'sector_stability'],
            'cultural_context': ['financial_discipline', 'community_integration', 'family_stability', 'income_diversification', 'relationship_strength']
        }
    }