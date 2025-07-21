# """
# Enums Package
# =============

# Contains enumeration classes for type-safe operations throughout the credit scoring system.

# This package provides strongly-typed enumerations to ensure consistency and prevent
# errors from using string literals directly in the code.

# Enumerations:
#     ClientType: Defines the types of clients (NEW, RENEWING)
#     ScalingMethod: Defines supported feature scaling methods (LOG_MINMAX, MINMAX, ZSCORE, ROBUST)

# Example Usage:
#     from enums import ClientType, ScalingMethod
    
#     # Use client type
#     if client_type == ClientType.NEW:
#         # Handle new client logic
#         pass
    
#     # Use scaling method
#     if scaling_method == ScalingMethod.LOG_MINMAX:
#         # Apply log transformation with min-max scaling
#         pass
# """

# from .client_type import ClientType
# from .scaling_method import ScalingMethod

# # Version info
# __version__ = '1.0.0'
# __author__ = 'Credit Scoring Team'

# # Export all enums
# __all__ = [
#     'ClientType',
#     'ScalingMethod',
# ]

# # Convenience functions for enum validation
# def validate_client_type(value):
#     """
#     Validate and convert a value to ClientType enum.
    
#     Args:
#         value: String or ClientType enum value
        
#     Returns:
#         ClientType: The validated enum value
        
#     Raises:
#         ValueError: If the value is not a valid client type
#     """
#     if isinstance(value, ClientType):
#         return value
    
#     if isinstance(value, str):
#         value = value.lower()
#         for client_type in ClientType:
#             if client_type.value == value:
#                 return client_type
    
#     valid_values = [ct.value for ct in ClientType]
#     raise ValueError(f"Invalid client type: {value}. Must be one of {valid_values}")


# def validate_scaling_method(value):
#     """
#     Validate and convert a value to ScalingMethod enum.
    
#     Args:
#         value: String or ScalingMethod enum value
        
#     Returns:
#         ScalingMethod: The validated enum value
        
#     Raises:
#         ValueError: If the value is not a valid scaling method
#     """
#     if isinstance(value, ScalingMethod):
#         return value
    
#     if isinstance(value, str):
#         value = value.lower()
#         for method in ScalingMethod:
#             if method.value == value:
#                 return method
    
#     valid_values = [sm.value for sm in ScalingMethod]
#     raise ValueError(f"Invalid scaling method: {value}. Must be one of {valid_values}")


# # Utility functions for getting enum descriptions
# def get_client_type_description(client_type):
#     """
#     Get a human-readable description of a client type.
    
#     Args:
#         client_type (ClientType): The client type enum
        
#     Returns:
#         str: Description of the client type
#     """
#     descriptions = {
#         ClientType.NEW: "New client with no previous credit history",
#         ClientType.RENEWING: "Existing client renewing or applying for additional credit"
#     }
#     return descriptions.get(client_type, "Unknown client type")


# def get_scaling_method_description(scaling_method):
#     """
#     Get a human-readable description of a scaling method.
    
#     Args:
#         scaling_method (ScalingMethod): The scaling method enum
        
#     Returns:
#         str: Description of the scaling method
#     """
#     descriptions = {
#         ScalingMethod.LOG_MINMAX: "Logarithmic transformation followed by min-max scaling (0-1)",
#         ScalingMethod.MINMAX: "Linear min-max scaling to range 0-1",
#         ScalingMethod.ZSCORE: "Z-score standardization (mean=0, std=1)",
#         ScalingMethod.ROBUST: "Robust scaling using median and IQR (resistant to outliers)"
#     }
#     return descriptions.get(scaling_method, "Unknown scaling method")


# # Lists for easy iteration
# CLIENT_TYPES = list(ClientType)
# SCALING_METHODS = list(ScalingMethod)