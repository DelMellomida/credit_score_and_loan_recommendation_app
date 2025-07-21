"""
Enhanced Credit Scoring Transformer
===================================

Main transformer class with complete feature isolation and client-type specific scoring.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from ..config.enhanced_config import EnhancedCreditScoringConfig
from ..enums.client_type import ClientType
from ..processors.feature_processor import FeatureProcessor
from ..scorers.credit_scorer import CreditBehaviorScorer
from ..scorers.financial_scorer import FinancialStabilityScorer
from ..scorers.cultural_scorer import CulturalContextScorer

logger = logging.getLogger(__name__)


class EnhancedCreditScoringTransformer:
    """
    Main transformer class with complete feature isolation and client-type specific scoring.
    
    ARCHITECTURE FEATURES:
    1. Client-type specific scoring (New vs Renewing)
    2. Mathematical feature isolation with caps
    3. Component-level contribution limits
    4. Severe data leakage prevention
    5. Interpretable scoring with explanations
    """
    
    def __init__(self, config: Optional[EnhancedCreditScoringConfig] = None):
        self.config = config or EnhancedCreditScoringConfig()
        self.processor = FeatureProcessor(self.config)
        
        # Initialize component scorers
        self.scorers = {
            'credit_behavior': CreditBehaviorScorer,
            'financial_stability': FinancialStabilityScorer,
            'cultural_context': CulturalContextScorer
        }
        
        logger.info("Enhanced Credit Scoring Transformer initialized successfully")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe with client-type specific scoring and feature isolation."""
        if 'Is_Renewing_Client' not in df.columns:
            raise ValueError("Column 'Is_Renewing_Client' is required for client type detection")
        
        df_transformed = df.copy()
        
        # Separate client types
        new_clients_mask = df['Is_Renewing_Client'] == 0
        renewing_clients_mask = df['Is_Renewing_Client'] == 1
        
        new_clients_df = df[new_clients_mask]
        renewing_clients_df = df[renewing_clients_mask]
        
        # Initialize score columns
        for component in ['Credit_Behavior_Score', 'Financial_Stability_Score', 'Cultural_Context_Score']:
            df_transformed[component] = 0.0
        
        # Process new clients
        if len(new_clients_df) > 0:
            new_scores = self._score_client_type(new_clients_df, ClientType.NEW)
            for component, scores in new_scores.items():
                df_transformed.loc[new_clients_mask, component] = scores
        
        # Process renewing clients
        if len(renewing_clients_df) > 0:
            renewing_scores = self._score_client_type(renewing_clients_df, ClientType.RENEWING)
            for component, scores in renewing_scores.items():
                df_transformed.loc[renewing_clients_mask, component] = scores
        
        # Calculate final credit risk scores
        df_transformed['Credit_Risk_Score'] = self._calculate_final_scores(df_transformed)
        
        # Add client type labels for transparency
        df_transformed['Client_Type'] = df['Is_Renewing_Client'].map({
            0: 'New',
            1: 'Renewing'
        })
        
        return df_transformed
    
    def _score_client_type(self, df: pd.DataFrame, client_type: ClientType) -> Dict[str, pd.Series]:
        """Score specific client type using appropriate configuration."""
        client_config = self.config.get_client_config(client_type)
        scores = {}
        
        for component_name, component_config in client_config.components.items():
            scorer_class = self.scorers[component_name]
            scorer = scorer_class(component_config, self.processor)
            
            component_score = scorer.calculate_component_score(df)
            scores[f"{component_name.title().replace('_', '_')}_Score"] = component_score
        
        return scores
    
    def _calculate_final_scores(self, df: pd.DataFrame) -> pd.Series:
        """Calculate final credit risk scores respecting client-type specific weights."""
        final_scores = pd.Series([0.0] * len(df), index=df.index)
        
        # Process each client type separately
        new_clients_mask = df['Is_Renewing_Client'] == 0
        renewing_clients_mask = df['Is_Renewing_Client'] == 1
        
        # New clients: Financial (80%) + Cultural (20%)
        if new_clients_mask.sum() > 0:
            new_config = self.config.get_client_config(ClientType.NEW)
            new_scores = (
                df.loc[new_clients_mask, 'Financial_Stability_Score'] * 
                new_config.components['financial_stability'].max_contribution_pct +
                df.loc[new_clients_mask, 'Cultural_Context_Score'] * 
                new_config.components['cultural_context'].max_contribution_pct
            )
            final_scores.loc[new_clients_mask] = new_scores
        
        # Renewing clients: Credit (60%) + Financial (37%) + Cultural (3%)
        if renewing_clients_mask.sum() > 0:
            renewing_config = self.config.get_client_config(ClientType.RENEWING)
            renewing_scores = (
                df.loc[renewing_clients_mask, 'Credit_Behavior_Score'] * 
                renewing_config.components['credit_behavior'].max_contribution_pct +
                df.loc[renewing_clients_mask, 'Financial_Stability_Score'] * 
                renewing_config.components['financial_stability'].max_contribution_pct +
                df.loc[renewing_clients_mask, 'Cultural_Context_Score'] * 
                renewing_config.components['cultural_context'].max_contribution_pct
            )
            final_scores.loc[renewing_clients_mask] = renewing_scores
        
        # Ensure scores are in valid range
        return pd.Series(np.clip(final_scores, 0, 1), index=df.index)
    
    def get_feature_importance(self, client_type: Optional[ClientType] = None) -> Dict[str, Any]:
        """Get feature importance analysis showing caps and actual contributions."""
        if client_type:
            client_configs = {client_type: self.config.get_client_config(client_type)}
        else:
            client_configs = self.config.client_configs
        
        importance_analysis = {}
        
        for ct, config in client_configs.items():
            client_analysis = {
                'total_max_contribution': config.get_total_max_contribution(),
                'components': {}
            }
            
            for comp_name, component in config.components.items():
                comp_analysis = {
                    'component_max_pct': component.max_contribution_pct,
                    'features': {}
                }
                
                for feat_name, feature in component.features.items():
                    max_total_impact = feature.max_contribution_pct * component.max_contribution_pct
                    comp_analysis['features'][feat_name] = {
                        'max_contribution_to_component': feature.max_contribution_pct,
                        'weight_in_component': feature.weight_in_component,
                        'max_total_impact_on_final_score': max_total_impact,
                        'effective_cap_pct': max_total_impact * 100
                    }
                
                client_analysis['components'][comp_name] = comp_analysis
            
            importance_analysis[ct.value] = client_analysis
        
        return importance_analysis
    
    def get_score_explanation(self, applicant_data: pd.Series) -> Dict[str, Any]:
        """Get detailed explanation of score calculation with feature contributions."""
        # Determine client type
        is_renewing = bool(applicant_data.get('Is_Renewing_Client', 0))
        client_type = ClientType.RENEWING if is_renewing else ClientType.NEW
        
        # Convert to DataFrame for processing
        df_single = pd.DataFrame([applicant_data])
        transformed = self.transform(df_single)
        
        # Get scores
        credit_behavior_score = transformed['Credit_Behavior_Score'].iloc[0]
        financial_stability_score = transformed['Financial_Stability_Score'].iloc[0]
        cultural_context_score = transformed['Cultural_Context_Score'].iloc[0]
        final_score = transformed['Credit_Risk_Score'].iloc[0]
        
        # Get configuration
        client_config = self.config.get_client_config(client_type)
        
        explanation = {
            'final_credit_risk_score': round(final_score, 4),
            'client_type': client_type.value,
            'score_interpretation': self._interpret_score(final_score),
            
            'component_contributions': {
                'financial_stability': {
                    'score': round(financial_stability_score, 4),
                    'max_contribution_pct': client_config.components['financial_stability'].max_contribution_pct,
                    'actual_contribution': round(financial_stability_score * client_config.components['financial_stability'].max_contribution_pct, 4),
                    'features_breakdown': self._get_financial_features_breakdown(applicant_data, client_type)
                },
                'cultural_context': {
                    'score': round(cultural_context_score, 4),
                    'max_contribution_pct': client_config.components['cultural_context'].max_contribution_pct,
                    'actual_contribution': round(cultural_context_score * client_config.components['cultural_context'].max_contribution_pct, 4),
                    'features_breakdown': self._get_cultural_features_breakdown(applicant_data, client_type)
                }
            },
            
            'feature_isolation_summary': {
                'total_possible_max_contribution': client_config.get_total_max_contribution(),
                'actual_total_contribution': round(final_score, 4),
                'leakage_prevention': {
                    'community_role_impact': 'Limited to 0.5% max for renewing, 2% max for new clients',
                    'paluwagan_impact': 'Limited to 0.8% max for renewing, 3% max for new clients',
                    'individual_feature_caps': 'All features mathematically capped',
                    'component_level_caps': 'Components cannot exceed specified percentages'
                }
            },
            
            'client_type_specific_weights': {
                client_type.value: {
                    comp_name: comp_config.max_contribution_pct 
                    for comp_name, comp_config in client_config.components.items()
                }
            }
        }
        
        # Add credit behavior for renewing clients
        if client_type == ClientType.RENEWING:
            explanation['component_contributions']['credit_behavior'] = {
                'score': round(credit_behavior_score, 4),
                'max_contribution_pct': client_config.components['credit_behavior'].max_contribution_pct,
                'actual_contribution': round(credit_behavior_score * client_config.components['credit_behavior'].max_contribution_pct, 4),
                'features_breakdown': self._get_credit_features_breakdown(applicant_data)
            }
        
        return explanation
    
    def _interpret_score(self, score: float) -> str:
        """Interpret credit risk score."""
        if score >= 0.8:
            return "Excellent credit risk profile - High approval likelihood"
        elif score >= 0.7:
            return "Good credit risk profile - Favorable terms likely" 
        elif score >= 0.6:
            return "Acceptable credit risk profile - Standard terms"
        elif score >= 0.5:
            return "Marginal credit risk profile - Review required"
        elif score >= 0.4:
            return "Poor credit risk profile - Higher risk/rates"
        else:
            return "Very poor credit risk profile - Decline recommended"
    
    def _get_financial_features_breakdown(self, applicant_data: pd.Series, client_type: ClientType) -> Dict[str, Any]:
        """Get breakdown of financial features contribution."""
        primary_salary = applicant_data.get('Net_Salary_Per_Cutoff', 0)
        comaker_salary = applicant_data.get('Comaker_Net_Salary_Per_Cutoff', 0)
        employment_tenure = applicant_data.get('Employment_Tenure_Months', 0)
        address_years = applicant_data.get('Years_at_Current_Address', 0)
        dependents = applicant_data.get('Number_of_Dependents', 0)
        
        household_size = 1 + dependents + (1 if comaker_salary > 0 else 0)
        income_per_member = (primary_salary + comaker_salary) / household_size
        total_household_income = primary_salary + comaker_salary
        
        client_config = self.config.get_client_config(client_type)
        financial_config = client_config.components['financial_stability']
        
        return {
            'income_adequacy': {
                'value': f"₱{income_per_member:,.0f} per household member",
                'max_impact_pct': financial_config.features['income_adequacy'].max_contribution_pct * 100,
                'description': 'Primary income divided by household size'
            },
            'household_income_capacity': {
                'value': f"₱{total_household_income:,.0f} total household income",
                'max_impact_pct': financial_config.features['household_income_capacity'].max_contribution_pct * 100,
                'description': 'Combined primary and comaker income'
            },
            'employment_stability': {
                'value': f"{employment_tenure} months tenure",
                'max_impact_pct': financial_config.features['employment_stability'].max_contribution_pct * 100,
                'description': 'Length of current employment'
            },
            'address_stability': {
                'value': f"{address_years} years at current address",
                'max_impact_pct': financial_config.features['address_stability'].max_contribution_pct * 100,
                'description': 'Residential stability indicator'
            }
        }
    
    def _get_cultural_features_breakdown(self, applicant_data: pd.Series, client_type: ClientType) -> Dict[str, Any]:
        """Get breakdown of cultural features contribution with leakage prevention details."""
        community_role = applicant_data.get('Has_Community_Role', 'No')
        paluwagan = applicant_data.get('Paluwagan_Participation', 'No')
        housing_status = applicant_data.get('Housing_Status', 'Rented')
        household_head = applicant_data.get('Household_Head', 'No')
        other_income = applicant_data.get('Other_Income_Source', 'None')
        comaker_relationship = applicant_data.get('Comaker_Relationship', 'Friend')
        
        client_config = self.config.get_client_config(client_type)
        cultural_config = client_config.components['cultural_context']
        
        return {
            'financial_discipline': {
                'value': f"Paluwagan: {paluwagan}",
                'max_impact_pct': cultural_config.features['financial_discipline'].max_contribution_pct * 100,
                'leakage_prevention': 'SEVERELY LIMITED - was 66% default rate difference',
                'description': 'Participation in traditional savings groups'
            },
            'community_integration': {
                'value': f"Community Role: {community_role}",
                'max_impact_pct': cultural_config.features['community_integration'].max_contribution_pct * 100,
                'leakage_prevention': 'SEVERELY LIMITED - was perfect predictor (0% default)',
                'description': 'Leadership role in community'
            },
            'family_stability': {
                'value': f"Housing: {housing_status}, Head: {household_head}",
                'max_impact_pct': cultural_config.features['family_stability'].max_contribution_pct * 100,
                'description': 'Housing ownership and household leadership'
            },
            'income_diversification': {
                'value': f"Other Income: {other_income}",
                'max_impact_pct': cultural_config.features['income_diversification'].max_contribution_pct * 100,
                'description': 'Additional income sources beyond employment'
            },
            'relationship_strength': {
                'value': f"Comaker: {comaker_relationship}",
                'max_impact_pct': cultural_config.features['relationship_strength'].max_contribution_pct * 100,
                'description': 'Relationship type with loan comaker'
            }
        }
    
    def _get_credit_features_breakdown(self, applicant_data: pd.Series) -> Dict[str, Any]:
        """Get breakdown of credit behavior features (renewing clients only)."""
        late_payments = applicant_data.get('Late_Payment_Count', 0)
        grace_usage = applicant_data.get('Grace_Period_Usage_Rate', 0)
        special_consideration = applicant_data.get('Had_Special_Consideration', 0)
        
        return {
            'payment_history': {
                'value': f"{late_payments} late payments",
                'max_impact_pct': 21,  # 35% of 60% component
                'description': 'Historical payment performance'
            },
            'grace_period_usage': {
                'value': f"{grace_usage:.1%} grace period usage",
                'max_impact_pct': 12,  # 20% of 60% component
                'description': 'Frequency of payment extensions'
            },
            'special_considerations': {
                'value': 'Yes' if special_consideration else 'No',
                'max_impact_pct': 6,   # 10% of 60% component
                'description': 'Required special payment arrangements'
            },
            'client_loyalty': {
                'value': 'Renewing client',
                'max_impact_pct': 6,   # 10% of 60% component
                'description': 'Loyalty bonus for returning customers'
            }
        }