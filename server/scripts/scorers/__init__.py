# """
# Credit Scoring Components Module
# ================================

# Component-based scoring system with mathematical feature isolation.

# This module implements a modular scoring architecture where different aspects
# of creditworthiness are evaluated by specialized scorers. Each scorer implements
# strict feature caps and mathematical isolation to prevent data leakage and ensure
# fair, explainable credit decisions.

# Architecture:
#     ComponentScorer (ABC): Base class providing common functionality
#     ├── FinancialStabilityScorer: Evaluates objective financial capacity
#     ├── CreditBehaviorScorer: Analyzes payment history (renewing clients only)
#     └── CulturalContextScorer: Considers cultural factors with severe constraints

# Key Features:
# - Feature-level contribution caps to prevent any single factor from dominating
# - Component-level caps to ensure balanced scoring
# - Mathematical isolation between components
# - Robust handling of missing values
# - Prevention of data leakage through constrained mappings

# Classes:
#     ComponentScorer: Abstract base class for all scoring components
#     FinancialStabilityScorer: Scores based on income, employment, and stability
#     CreditBehaviorScorer: Scores based on payment history and credit behavior
#     CulturalContextScorer: Scores based on cultural and behavioral indicators

# Example Usage:
#     from scorers import FinancialStabilityScorer, CreditBehaviorScorer
#     from config import EnhancedCreditScoringConfig
#     from processing import FeatureProcessor
    
#     # Initialize configuration and processor
#     config = EnhancedCreditScoringConfig()
#     processor = FeatureProcessor(config)
    
#     # Get component configuration for new clients
#     new_client_config = config.get_client_config(ClientType.NEW)
#     financial_config = new_client_config.components['financial_stability']
    
#     # Create scorer
#     financial_scorer = FinancialStabilityScorer(financial_config, processor)
    
#     # Calculate scores
#     scores = financial_scorer.calculate_component_score(df)
# """

# from .base_scorer import ComponentScorer
# from .credit_scorer import CreditBehaviorScorer
# from .cultural_scorer import CulturalContextScorer
# from .financial_scorer import FinancialStabilityScorer

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Export all scorer classes
# __all__ = [
#     'ComponentScorer',
#     'FinancialStabilityScorer',
#     'CreditBehaviorScorer', 
#     'CulturalContextScorer',
# ]

# # Component descriptions for documentation
# COMPONENT_DESCRIPTIONS = {
#     'financial_stability': """
#         Evaluates objective financial capacity including:
#         - Income adequacy (income per household member)
#         - Total household income capacity
#         - Employment tenure and stability
#         - Address stability (years at current address)
#         - Employment sector stability
#     """,
    
#     'credit_behavior': """
#         Analyzes credit history and payment behavior (renewing clients only):
#         - Payment history (late payment frequency)
#         - Grace period usage patterns
#         - Special consideration history
#         - Client loyalty bonus
#     """,
    
#     'cultural_context': """
#         Considers cultural and behavioral factors with severe constraints:
#         - Financial discipline (Paluwagan participation)
#         - Community integration (community roles)
#         - Family stability (housing status, household head)
#         - Income diversification (additional income sources)
#         - Relationship strength (comaker relationships)
#     """
# }

# # Factory function to create appropriate scorer
# def create_scorer(component_name, component_config, processor):
#     """
#     Factory function to create the appropriate scorer based on component name.
    
#     Args:
#         component_name (str): Name of the component ('financial_stability', 
#                               'credit_behavior', 'cultural_context')
#         component_config: ComponentConfig instance for the component
#         processor: FeatureProcessor instance
        
#     Returns:
#         ComponentScorer: Appropriate scorer instance
        
#     Raises:
#         ValueError: If component_name is not recognized
#     """
#     scorer_mapping = {
#         'financial_stability': FinancialStabilityScorer,
#         'credit_behavior': CreditBehaviorScorer,
#         'cultural_context': CulturalContextScorer,
#     }
    
#     if component_name not in scorer_mapping:
#         raise ValueError(
#             f"Unknown component: {component_name}. "
#             f"Must be one of {list(scorer_mapping.keys())}"
#         )
    
#     scorer_class = scorer_mapping[component_name]
#     return scorer_class(component_config, processor)


# # Utility function to get component description
# def get_component_description(component_name):
#     """
#     Get a human-readable description of a scoring component.
    
#     Args:
#         component_name (str): Name of the component
        
#     Returns:
#         str: Description of the component
#     """
#     return COMPONENT_DESCRIPTIONS.get(
#         component_name, 
#         f"No description available for component: {component_name}"
#     ).strip()


# # List all available components
# def list_available_components():
#     """
#     Get a list of all available scoring components.
    
#     Returns:
#         list: Names of available components
#     """
#     return list(COMPONENT_DESCRIPTIONS.keys())