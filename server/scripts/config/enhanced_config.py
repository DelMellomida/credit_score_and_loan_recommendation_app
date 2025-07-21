"""
Enhanced Credit Scoring Configuration
=====================================

Master configuration class with complete feature isolation and data leakage prevention.
"""

import logging
from typing import Any, Dict

import numpy as np

from ..enums.client_type import ClientType
from ..enums.scaling_method import ScalingMethod
from .client_type_config import ClientTypeConfig
from .component_config import ComponentConfig
from .feature_config import FeatureConfig

logger = logging.getLogger(__name__)


class EnhancedCreditScoringConfig:
    """
    Master configuration class with complete feature isolation and data leakage prevention.
    
    CRITICAL DESIGN FEATURES:
    1. Client-type specific configurations
    2. Individual feature contribution caps
    3. Component-level contribution caps  
    4. Data-driven scaling parameters
    5. Severe constraints on leaky features
    """
    
    def __init__(self):
        self.client_configs: Dict[ClientType, ClientTypeConfig] = {}
        self.data_driven_ranges = self._get_data_driven_ranges()
        self.leakage_mitigation_features = self._get_leakage_features()
        
        # Initialize client-specific configurations
        self._setup_new_client_config()
        self._setup_renewing_client_config()
        
        # Validation
        self._validate_configurations()
        
        logger.info("Enhanced Credit Scoring Configuration initialized successfully")
    
    def _get_data_driven_ranges(self) -> Dict[str, Dict[str, float]]:
        """Data-driven scaling ranges from actual dataset analysis."""
        return {
            'net_salary': {
                'min_log': np.log1p(9000),      # ₱9,000 (5th percentile)
                'max_log': np.log1p(75000),     # ₱75,000 (95th percentile)
                'median': 25000
            },
            'comaker_salary': {
                'min_log': np.log1p(5000),
                'max_log': np.log1p(80000),
                'median': 20000
            },
            'employment_tenure': {
                'min_log': np.log1p(2),         # 2 months (5th percentile)
                'max_log': np.log1p(240),       # 240 months (95th percentile)
                'median': 36
            },
            'address_stability': {
                'min_log': np.log1p(0.1),
                'max_log': np.log1p(30),
                'median': 3
            },
            'household_income': {
                'min_log': np.log1p(15000),
                'max_log': np.log1p(150000),
                'median': 45000
            }
        }
    
    def _get_leakage_features(self) -> Dict[str, Dict[str, float]]:
        """
        Severely constrained mappings for data leakage prevention.
        
        CRITICAL: These features showed perfect/near-perfect prediction:
        - Community Role "Yes" = 0.0% default (perfect predictor)
        - Paluwagan "Yes" = 9.1% vs "No" = 75.1% (66% difference)
        """
        return {
            'has_community_role': {
                'Yes': 0.02,    # SEVERELY LIMITED from perfect predictor
                'No': 0.0       # Baseline
            },
            'paluwagan_participation': {
                'Yes': 0.03,    # SEVERELY LIMITED from 66% advantage
                'No': 0.0       # Baseline
            },
            'housing_status': {
                'Owned': 0.15,
                'Rented': 0.0
            },
            'household_head': {
                'Yes': 0.10,
                'No': 0.05
            },
            'employment_sector': {
                'Public': 0.12,
                'Private': 0.08
            },
            'disaster_preparedness': {
                'Community Plan': 0.15,
                'Insurance': 0.12,
                'Savings': 0.08,
                'None': 0.0
            },
            'other_income_source': {
                'Business': 0.20,
                'OFW Remittance': 0.15,
                'Freelance': 0.08,
                'None': 0.0
            },
            'comaker_relationship': {
                'Spouse': 0.18,
                'Parent': 0.12,
                'Sibling': 0.10,
                'Friend': 0.05
            },
            'salary_frequency': {
                'Biweekly': 0.15,
                'Weekly': 0.12,
                'Monthly': 0.08,
                'Bimonthly': 0.05
            }
        }
    
    def _setup_new_client_config(self) -> None:
        """Setup configuration for new clients (80% Financial, 20% Cultural)."""
        new_client_config = ClientTypeConfig(
            client_type=ClientType.NEW,
            description="New clients with no credit history - focus on financial capacity"
        )
        
        # Financial Stability Component (80% for new clients)
        financial_component = ComponentConfig(
            name="financial_stability",
            max_contribution_pct=0.80,
            component_description="Objective financial capacity and stability measures"
        )
        
        # Add financial features with caps
        financial_features = [
            FeatureConfig(
                name="income_adequacy",
                max_contribution_pct=0.25,      # Max 25% of component (20% of total)
                weight_in_component=0.35,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="household_income_capacity", 
                max_contribution_pct=0.25,      # Max 25% of component (20% of total)
                weight_in_component=0.30,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="employment_stability",
                max_contribution_pct=0.20,      # Max 20% of component (16% of total)
                weight_in_component=0.25,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="address_stability",
                max_contribution_pct=0.10,      # Max 10% of component (8% of total)
                weight_in_component=0.08,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="sector_stability",
                max_contribution_pct=0.05,      # Max 5% of component (4% of total)
                weight_in_component=0.02,
                categorical_mapping=self.leakage_mitigation_features['employment_sector']
            )
        ]
        
        for feature in financial_features:
            financial_component.add_feature(feature)
        
        new_client_config.add_component(financial_component)
        
        # Cultural Context Component (20% for new clients)
        cultural_component = ComponentConfig(
            name="cultural_context",
            max_contribution_pct=0.10,
            component_description="Cultural and behavioral indicators with severe leakage constraints"
        )
        
        # Add cultural features with SEVERE caps due to leakage
        cultural_features = [
            FeatureConfig(
                name="financial_discipline",     # Paluwagan 
                max_contribution_pct=0.03,      # Max 3% of component (0.6% of total)
                weight_in_component=0.25,
                categorical_mapping=self.leakage_mitigation_features['paluwagan_participation']
            ),
            FeatureConfig(
                name="community_integration",   # Community role
                max_contribution_pct=0.02,      # Max 2% of component (0.4% of total) 
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['has_community_role']
            ),
            FeatureConfig(
                name="family_stability",        # Housing + household head
                max_contribution_pct=0.08,      # Max 8% of component (1.6% of total)
                weight_in_component=0.25,
            ),
            FeatureConfig(
                name="income_diversification",  # Other income
                max_contribution_pct=0.08,      # Max 8% of component (1.6% of total)
                weight_in_component=0.20,
                categorical_mapping=self.leakage_mitigation_features['other_income_source']
            ),
            FeatureConfig(
                name="relationship_strength",   # Comaker relationship
                max_contribution_pct=0.06,      # Max 6% of component (1.2% of total)
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['comaker_relationship']
            )
        ]
        
        for feature in cultural_features:
            cultural_component.add_feature(feature)
        
        new_client_config.add_component(cultural_component)
        self.client_configs[ClientType.NEW] = new_client_config
    
    def _setup_renewing_client_config(self) -> None:
        """Setup configuration for renewing clients (60% Credit, 37% Financial, 3% Cultural)."""
        renewing_client_config = ClientTypeConfig(
            client_type=ClientType.RENEWING,
            description="Renewing clients with credit history - focus on payment behavior"
        )
        
        # Credit Behavior Component (60% for renewing clients)
        credit_component = ComponentConfig(
            name="credit_behavior",
            max_contribution_pct=0.60,
            component_description="Objective payment history and credit behavior"
        )
        
        # Add credit behavior features
        credit_features = [
            FeatureConfig(
                name="payment_history",
                max_contribution_pct=0.35,      # Max 35% of component (21% of total)
                weight_in_component=0.50,
                scaling_method=ScalingMethod.MINMAX
            ),
            FeatureConfig(
                name="grace_period_usage",
                max_contribution_pct=0.20,      # Max 20% of component (12% of total)
                weight_in_component=0.30,
                scaling_method=ScalingMethod.MINMAX
            ),
            FeatureConfig(
                name="special_considerations",
                max_contribution_pct=0.10,      # Max 10% of component (6% of total)
                weight_in_component=0.10,
            ),
            FeatureConfig(
                name="client_loyalty",
                max_contribution_pct=0.10,      # Max 10% of component (6% of total)
                weight_in_component=0.10,
            )
        ]
        
        for feature in credit_features:
            credit_component.add_feature(feature)
        
        renewing_client_config.add_component(credit_component)
        
        # Financial Stability Component (37% for renewing clients)
        financial_component = ComponentConfig(
            name="financial_stability", 
            max_contribution_pct=0.37,
            component_description="Current financial capacity and stability"
        )
        
        # Add financial features (same as new clients but different weights)
        financial_features = [
            FeatureConfig(
                name="income_adequacy",
                max_contribution_pct=0.15,      # Max 15% of component (5.5% of total)
                weight_in_component=0.35,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="household_income_capacity",
                max_contribution_pct=0.12,      # Max 12% of component (4.4% of total)
                weight_in_component=0.30,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="employment_stability", 
                max_contribution_pct=0.10,      # Max 10% of component (3.7% of total)
                weight_in_component=0.25,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="address_stability",
                max_contribution_pct=0.05,      # Max 5% of component (1.8% of total)
                weight_in_component=0.08,
                scaling_method=ScalingMethod.LOG_MINMAX
            ),
            FeatureConfig(
                name="sector_stability",
                max_contribution_pct=0.03,      # Max 3% of component (1.1% of total)
                weight_in_component=0.02,
                categorical_mapping=self.leakage_mitigation_features['employment_sector']
            )
        ]
        
        for feature in financial_features:
            financial_component.add_feature(feature)
        
        renewing_client_config.add_component(financial_component)
        
        # Cultural Context Component (3% for renewing clients - SEVERELY LIMITED)
        cultural_component = ComponentConfig(
            name="cultural_context",
            max_contribution_pct=0.03,
            component_description="SEVERELY LIMITED cultural factors to prevent data leakage"
        )
        
        # Add cultural features with EXTREME caps for renewing clients
        cultural_features = [
            FeatureConfig(
                name="financial_discipline",     # Paluwagan
                max_contribution_pct=0.008,     # Max 0.8% of component (0.024% of total)
                weight_in_component=0.25,
                categorical_mapping=self.leakage_mitigation_features['paluwagan_participation']
            ),
            FeatureConfig(
                name="community_integration",   # Community role
                max_contribution_pct=0.005,     # Max 0.5% of component (0.015% of total)
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['has_community_role']
            ),
            FeatureConfig(
                name="family_stability",        # Housing + household head
                max_contribution_pct=0.012,     # Max 1.2% of component (0.036% of total)
                weight_in_component=0.25,
            ),
            FeatureConfig(
                name="income_diversification",  # Other income
                max_contribution_pct=0.008,     # Max 0.8% of component (0.024% of total)
                weight_in_component=0.20,
                categorical_mapping=self.leakage_mitigation_features['other_income_source']
            ),
            FeatureConfig(
                name="relationship_strength",   # Comaker relationship
                max_contribution_pct=0.007,     # Max 0.7% of component (0.021% of total)
                weight_in_component=0.15,
                categorical_mapping=self.leakage_mitigation_features['comaker_relationship']
            )
        ]
        
        for feature in cultural_features:
            cultural_component.add_feature(feature)
        
        renewing_client_config.add_component(cultural_component)
        self.client_configs[ClientType.RENEWING] = renewing_client_config
    
    def _validate_configurations(self) -> None:
        """Validate all configurations for consistency and caps."""
        for client_type, config in self.client_configs.items():
            total_contribution = config.get_total_max_contribution()
            if total_contribution > 1.0:
                raise ValueError(f"Client type {client_type.value}: total max contributions exceed 100%")
            
            logger.info(f"✓ {client_type.value} client config validated (max total: {total_contribution:.1%})")
    
    def get_client_config(self, client_type: ClientType) -> ClientTypeConfig:
        """Get configuration for specific client type."""
        return self.client_configs[client_type]
    
    def get_feature_caps_summary(self) -> Dict[str, Any]:
        """Get summary of all feature caps for transparency."""
        summary = {}
        
        for client_type, config in self.client_configs.items():
            client_summary = {}
            for comp_name, component in config.components.items():
                comp_summary = {
                    'component_max_pct': component.max_contribution_pct,
                    'features': {}
                }
                for feat_name, feature in component.features.items():
                    comp_summary['features'][feat_name] = {
                        'max_contribution_pct': feature.max_contribution_pct,
                        'weight_in_component': feature.weight_in_component,
                        'max_total_impact_pct': feature.max_contribution_pct * component.max_contribution_pct
                    }
                client_summary[comp_name] = comp_summary
            summary[client_type.value] = client_summary
        
        return summary