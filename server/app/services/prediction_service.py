import joblib
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
from app.schemas.loan_schema import LoanApplicationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'credit_model.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
POLY_PATH = os.path.join(MODEL_DIR, 'polynomial_features.pkl')
SELECTOR_PATH = os.path.join(MODEL_DIR, 'feature_selector.pkl')
FEATURES_INFO_PATH = os.path.join(MODEL_DIR, 'feature_info.pkl')

class PredictionService:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.poly = None
        self.selector = None
        self.features_info = None
        self._load_models()

    def _load_models(self):
        """Load all required model components with comprehensive error handling."""
        try:
            logger.info("Loading models...")
            
            # Check if model directory exists
            if not os.path.exists(MODEL_DIR):
                raise FileNotFoundError(f"Model directory '{MODEL_DIR}' does not exist.")
            
            # Check if all required files exist
            required_files = {
                'model': MODEL_PATH,
                'encoder': ENCODER_PATH,
                'scaler': SCALER_PATH,
                'polynomial_features': POLY_PATH,
                'feature_selector': SELECTOR_PATH,
                'features_info': FEATURES_INFO_PATH
            }
            
            missing_files = []
            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing_files.append(f"{name} ({path})")
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
            
            # Load models with individual error handling
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info("Model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load main model: {e}")
            
            try:
                self.encoder = joblib.load(ENCODER_PATH)
                logger.info("Encoder loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load encoder: {e}")
            
            try:
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load scaler: {e}")
            
            try:
                self.poly = joblib.load(POLY_PATH)
                logger.info("Polynomial features loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load polynomial features: {e}")
            
            try:
                self.selector = joblib.load(SELECTOR_PATH)
                logger.info("Feature selector loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load feature selector: {e}")
            
            try:
                self.features_info = joblib.load(FEATURES_INFO_PATH)
                logger.info("Features info loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load features info: {e}")
            
            # Validate features_info structure
            if not isinstance(self.features_info, dict):
                raise ValueError("features_info must be a dictionary.")
            
            required_keys = ['categorical_features', 'numerical_features']
            missing_keys = [key for key in required_keys if key not in self.features_info]
            if missing_keys:
                raise ValueError(f"features_info missing required keys: {missing_keys}")
            
            logger.info("All models loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise RuntimeError(f"Model files not found: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")
        
    def predict(self, input_data: LoanApplicationRequest) -> Dict[str, float]:
        """Make a prediction based on the input data with comprehensive error handling."""
        try:
            # Validate service is properly initialized
            if not self._is_service_ready():
                raise RuntimeError("PredictionService is not properly initialized. All model components must be loaded.")
            
            # Validate input data
            if input_data is None:
                raise ValueError("Input data cannot be None.")
            
            # Convert to DataFrame
            try:
                df = pd.DataFrame([input_data.model_dump()])
            except Exception as e:
                raise ValueError(f"Failed to convert input data to DataFrame: {e}")
            
            # Validate required features are present
            categorical_features = self.features_info['categorical_features']
            numerical_features = self.features_info['numerical_features']
            
            missing_categorical = [f for f in categorical_features if f not in df.columns]
            missing_numerical = [f for f in numerical_features if f not in df.columns]
            
            if missing_categorical:
                raise ValueError(f"Missing categorical features: {missing_categorical}")
            if missing_numerical:
                raise ValueError(f"Missing numerical features: {missing_numerical}")
            
            # Check for missing values
            if df[categorical_features + numerical_features].isnull().any().any():
                raise ValueError("Input data contains missing values.")
            
            # Transform categorical features
            try:
                encoded_categorical = self.encoder.transform(df[categorical_features])
            except Exception as e:
                raise RuntimeError(f"Failed to encode categorical features: {e}")
            
            # Transform numerical features
            try:
                scaled_numerical = self.scaler.transform(df[numerical_features])
            except Exception as e:
                raise RuntimeError(f"Failed to scale numerical features: {e}")
            
            # Combine features
            try:
                transformed_features = np.hstack([scaled_numerical, encoded_categorical.toarray()])
            except Exception as e:
                raise RuntimeError(f"Failed to combine features: {e}")
            
            # Apply polynomial transformation
            try:
                poly_features = self.poly.transform(transformed_features)
            except Exception as e:
                raise RuntimeError(f"Failed to apply polynomial transformation: {e}")
            
            # Apply feature selection
            try:
                selected_features = self.selector.transform(poly_features)
            except Exception as e:
                raise RuntimeError(f"Failed to apply feature selection: {e}")
            
            # Make prediction
            try:
                prediction_proba = self.model.predict_proba(selected_features)
                if prediction_proba.shape[1] < 2:
                    raise RuntimeError("Model does not provide probability for positive class.")
                probability_of_default = float(prediction_proba[0, 1])
            except Exception as e:
                raise RuntimeError(f"Failed to make prediction: {e}")
            
            # Validate probability is in valid range
            if not (0 <= probability_of_default <= 1):
                logger.warning(f"Probability of default {probability_of_default} is outside [0,1] range. Clipping.")
                probability_of_default = max(0, min(1, probability_of_default))
            
            return {"probability_of_default": probability_of_default}
            
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _is_service_ready(self) -> bool:
        """Check if all required components are loaded."""
        return all([
            self.model is not None,
            self.encoder is not None,
            self.scaler is not None,
            self.poly is not None,
            self.selector is not None,
            self.features_info is not None
        ])

    @staticmethod
    def transform_pod_to_credit_score(pod: float, min_score: int = 300, max_score: int = 850) -> int:
        """Transform probability of default to credit score with error handling."""
        try:
            # Validate inputs
            if not isinstance(pod, (int, float)):
                raise TypeError("Probability of default must be a number.")
            
            if not (0 <= pod <= 1):
                raise ValueError("Probability of default must be between 0 and 1.")
            
            if not isinstance(min_score, int) or not isinstance(max_score, int):
                raise TypeError("Min and max scores must be integers.")
            
            if min_score >= max_score:
                raise ValueError("Min score must be less than max score.")
            
            if min_score < 0 or max_score < 0:
                raise ValueError("Credit scores must be non-negative.")
            
            # Calculate credit score
            credit_score = min_score + (max_score - min_score) * (1 - pod)
            
            # Round and convert to integer
            credit_score = int(round(credit_score))
            
            # Ensure score is within bounds (safety check)
            credit_score = max(min_score, min(max_score, credit_score))
            
            return credit_score
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error transforming probability to credit score: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in credit score transformation: {e}")
            raise RuntimeError(f"Credit score transformation failed: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the prediction service."""
        return {
            "is_ready": self._is_service_ready(),
            "model_loaded": self.model is not None,
            "encoder_loaded": self.encoder is not None,
            "scaler_loaded": self.scaler is not None,
            "poly_loaded": self.poly is not None,
            "selector_loaded": self.selector is not None,
            "features_info_loaded": self.features_info is not None,
            "categorical_features": self.features_info.get('categorical_features', []) if self.features_info else [],
            "numerical_features": self.features_info.get('numerical_features', []) if self.features_info else []
        }

# Initialize the prediction service with proper error handling
def initialize_prediction_service() -> Optional[PredictionService]:
    """Initialize the prediction service with proper error handling."""
    try:
        logger.info("Initializing PredictionService...")
        service = PredictionService()
        logger.info("PredictionService initialized successfully.")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize PredictionService: {e}")
        return None

# Initialize the service
prediction_service = initialize_prediction_service()

# Enum definitions for your model inputs
# from enum import Enum

# class EmploymentSectorEnum(str, Enum):
#     PUBLIC = "Public"
#     PRIVATE = "Private"
#     SELF_EMPLOYED = "Self-Employed"
#     GOVERNMENT = "Government"
#     OFW = "OFW"

# class SalaryFrequencyEnum(str, Enum):
#     MONTHLY = "Monthly"
#     BI_MONTHLY = "Bi-Monthly"
#     WEEKLY = "Weekly"
#     DAILY = "Daily"

# class HousingStatusEnum(str, Enum):
#     OWNED = "Owned"
#     RENTED = "Rented"
#     FAMILY_OWNED = "Family-Owned"
#     COMPANY_PROVIDED = "Company-Provided"

# class YesNoEnum(str, Enum):
#     YES = "Yes"
#     NO = "No"

# class ComakerRelationshipEnum(str, Enum):
#     SPOUSE = "Spouse"
#     PARENT = "Parent"
#     SIBLING = "Sibling"
#     CHILD = "Child"
#     RELATIVE = "Relative"
#     FRIEND = "Friend"
#     NONE = "None"

# class OtherIncomeSourceEnum(str, Enum):
#     BUSINESS = "Business"
#     REMITTANCE = "Remittance"
#     PENSION = "Pension"
#     INVESTMENT = "Investment"
#     NONE = "None"

# class DisasterPreparednessEnum(str, Enum):
#     INSURANCE = "Insurance"
#     SAVINGS = "Savings"
#     GOVERNMENT_AID = "Government Aid"
#     FAMILY_SUPPORT = "Family Support"
#     NONE = "None"

# # Your actual model input structure
# class ModelInputData(BaseModel):
#     Employment_Sector: EmploymentSectorEnum = Field(..., description="Employment sector of the applicant")
#     Employment_Tenure_Months: int = Field(..., description="Employment tenure in months")
#     Net_Salary_Per_Cutoff: float = Field(..., description="Net salary per cutoff")
#     Salary_Frequency: SalaryFrequencyEnum = Field(..., description="Salary frequency of the applicant")
#     Housing_Status: HousingStatusEnum = Field(..., description="Housing status of the applicant")
#     Years_at_Current_Address: int = Field(..., description="Years at current address")
#     Household_Head: YesNoEnum = Field(..., description="Indicates if the applicant is the household head")
#     Number_of_Dependents: int = Field(..., description="Number of dependents of the applicant")
#     Comaker_Relationship: ComakerRelationshipEnum = Field(..., description="Relationship of the co-maker to the applicant")
#     Comaker_Employment_Tenure_Months: int = Field(..., description="Employment tenure of the co-maker in months")
#     Comaker_Net_Salary_Per_Cutoff: float = Field(..., description="Net salary of the co-maker per cutoff")
#     Has_Community_Role: YesNoEnum = Field(..., description="Indicates if the applicant has a community role")
#     Paluwagan_Participation: YesNoEnum = Field(..., description="Indicates if the applicant participates in a Paluwagan")
#     Other_Income_Source: OtherIncomeSourceEnum = Field(..., description="Other income source of the applicant")
#     Disaster_Preparedness: DisasterPreparednessEnum = Field(..., description="Disaster preparedness of the applicant")
#     Is_Renewing_Client: int = Field(..., description="Indicates if the applicant is a renewing client")
#     Grace_Period_Usage_Rate: float = Field(..., description="Grace period usage rate of the applicant")
#     Late_Payment_Count: float = Field(..., description="Late payment rate of the applicant")
#     Had_Special_Consideration: int = Field(..., description="Indicates if the applicant had special consideration in the past")
    
#     class Config:
#         use_enum_values = True
#         schema_extra = {
#             "example": {
#                 "Employment_Sector": "Public",
#                 "Employment_Tenure_Months": 48,
#                 "Net_Salary_Per_Cutoff": 18000.0,
#                 "Salary_Frequency": "Bi-Monthly",
#                 "Housing_Status": "Owned",
#                 "Years_at_Current_Address": 15,
#                 "Household_Head": "Yes",
#                 "Number_of_Dependents": 2,
#                 "Comaker_Relationship": "Spouse",
#                 "Comaker_Employment_Tenure_Months": 60,
#                 "Comaker_Net_Salary_Per_Cutoff": 20000.0,
#                 "Has_Community_Role": "No",
#                 "Paluwagan_Participation": "No",
#                 "Other_Income_Source": "None",
#                 "Disaster_Preparedness": "Insurance",
#                 "Is_Renewing_Client": 0,
#                 "Grace_Period_Usage_Rate": 0.0,
#                 "Late_Payment_Count": 0.0,
#                 "Had_Special_Consideration": 0
#             }
#         }

# # Complete workflow function for your model
# def get_credit_assessment(input_data: ModelInputData) -> Optional[Dict[str, Any]]:
#     """Complete workflow: input -> POD -> credit score."""
#     if prediction_service is None:
#         logger.error("PredictionService is not available.")
#         return None
    
#     try:
#         # Step 1: Get probability of default
#         pod_result = prediction_service.predict(input_data)
#         pod = pod_result["probability_of_default"]
        
#         # Step 2: Convert to credit score
#         credit_score = PredictionService.transform_pod_to_credit_score(pod)
        
#         # Step 3: Add risk assessment
#         if credit_score >= 750:
#             risk_level = "Low Risk"
#         elif credit_score >= 650:
#             risk_level = "Medium Risk"
#         else:
#             risk_level = "High Risk"
        
#         return {
#             "probability_of_default": pod,
#             "credit_score": credit_score,
#             "risk_level": risk_level,
#             "recommendation": "Approve" if credit_score >= 650 else "Decline"
#         }
        
#     except Exception as e:
#         logger.error(f"Credit assessment failed: {e}")
#         return None

# # Test function with your hardcoded sample data
# def test_prediction_with_sample_data():
#     """Test the prediction service with your sample data."""
#     print("=" * 60)
#     print("TESTING CREDIT PREDICTION SERVICE")
#     print("=" * 60)
    
#     # Check if service is available
#     if prediction_service is None:
#         print("❌ PredictionService is not available. Please check your model files.")
#         return False
    
#     # Get service status
#     status = prediction_service.get_service_status()
#     print(f"Service Ready: {'✅' if status['is_ready'] else '❌'}")
    
#     if not status['is_ready']:
#         print("Service Status Details:")
#         for key, value in status.items():
#             print(f"  {key}: {value}")
#         return False
    
#     print(f"Categorical Features: {status['categorical_features']}")
#     print(f"Numerical Features: {status['numerical_features']}")
#     print()
    
#     # Create sample input data based on your provided sample
#     try:
#         sample_input = ModelInputData(
#             Employment_Sector=EmploymentSectorEnum.PUBLIC,
#             Employment_Tenure_Months=48,
#             Net_Salary_Per_Cutoff=18000.0,
#             Salary_Frequency=SalaryFrequencyEnum.BI_MONTHLY,
#             Housing_Status=HousingStatusEnum.OWNED,
#             Years_at_Current_Address=15,
#             Household_Head=YesNoEnum.YES,
#             Number_of_Dependents=2,
#             Comaker_Relationship=ComakerRelationshipEnum.SPOUSE,
#             Comaker_Employment_Tenure_Months=60,
#             Comaker_Net_Salary_Per_Cutoff=20000.0,
#             Has_Community_Role=YesNoEnum.NO,
#             Paluwagan_Participation=YesNoEnum.NO,
#             Other_Income_Source=OtherIncomeSourceEnum.NONE,
#             Disaster_Preparedness=DisasterPreparednessEnum.INSURANCE,
#             Is_Renewing_Client=0,
#             Grace_Period_Usage_Rate=0.0,
#             Late_Payment_Count=0.0,
#             Had_Special_Consideration=0
#         )
        
#         print("✅ Sample input data created successfully")
#         print("Input Data:")
#         input_dict = sample_input.model_dump()
#         for key, value in input_dict.items():
#             print(f"  {key}: {value}")
#         print()
        
#     except Exception as e:
#         print(f"❌ Failed to create sample input data: {e}")
#         return False
    
#     # Test individual prediction
#     print("TESTING INDIVIDUAL PREDICTION:")
#     print("-" * 40)
#     try:
#         pod_result = prediction_service.predict(sample_input)
#         pod = pod_result["probability_of_default"]
#         print(f"✅ Probability of Default: {pod:.6f}")
        
#         # Test credit score conversion
#         credit_score = PredictionService.transform_pod_to_credit_score(pod)
#         print(f"✅ Credit Score: {credit_score}")
        
#     except Exception as e:
#         print(f"❌ Individual prediction failed: {e}")
#         return False
    
#     print()
    
#     # Test complete assessment
#     print("TESTING COMPLETE ASSESSMENT:")
#     print("-" * 40)
#     try:
#         assessment = get_credit_assessment(sample_input)
#         if assessment:
#             print("✅ Complete Assessment Results:")
#             print(f"  Probability of Default: {assessment['probability_of_default']:.6f}")
#             print(f"  Credit Score: {assessment['credit_score']}")
#             print(f"  Risk Level: {assessment['risk_level']}")
#             print(f"  Recommendation: {assessment['recommendation']}")
#         else:
#             print("❌ Complete assessment failed")
#             return False
            
#     except Exception as e:
#         print(f"❌ Complete assessment failed: {e}")
#         return False
    
#     print()
#     print("✅ ALL TESTS PASSED!")
#     return True

# # Additional test with different scenarios
# def test_edge_cases():
#     """Test edge cases and error handling."""
#     print("=" * 60)
#     print("TESTING EDGE CASES")
#     print("=" * 60)
    
#     if prediction_service is None:
#         print("❌ PredictionService not available for edge case testing")
#         return
    
#     # Test 1: High-risk scenario
#     print("Test 1: High-risk scenario")
#     try:
#         high_risk_input = ModelInputData(
#             Employment_Sector=EmploymentSectorEnum.SELF_EMPLOYED,
#             Employment_Tenure_Months=6,  # Short tenure
#             Net_Salary_Per_Cutoff=8000.0,  # Lower salary
#             Salary_Frequency=SalaryFrequencyEnum.WEEKLY,
#             Housing_Status=HousingStatusEnum.RENTED,
#             Years_at_Current_Address=1,  # Short address history
#             Household_Head=YesNoEnum.NO,
#             Number_of_Dependents=4,  # Many dependents
#             Comaker_Relationship=ComakerRelationshipEnum.NONE,
#             Comaker_Employment_Tenure_Months=0,
#             Comaker_Net_Salary_Per_Cutoff=0.0,
#             Has_Community_Role=YesNoEnum.NO,
#             Paluwagan_Participation=YesNoEnum.NO,
#             Other_Income_Source=OtherIncomeSourceEnum.NONE,
#             Disaster_Preparedness=DisasterPreparednessEnum.NONE,
#             Is_Renewing_Client=0,
#             Grace_Period_Usage_Rate=0.8,  # High grace period usage
#             Late_Payment_Count=0.6,  # High late payment rate
#             Had_Special_Consideration=1
#         )
        
#         result = get_credit_assessment(high_risk_input)
#         if result:
#             print(f"  POD: {result['probability_of_default']:.6f}")
#             print(f"  Credit Score: {result['credit_score']}")
#             print(f"  Risk Level: {result['risk_level']}")
#             print(f"  Recommendation: {result['recommendation']}")
#         else:
#             print("  ❌ High-risk test failed")
            
#     except Exception as e:
#         print(f"  ❌ High-risk test error: {e}")
    
#     print()
    
#     # Test 2: Low-risk scenario
#     print("Test 2: Low-risk scenario")
#     try:
#         low_risk_input = ModelInputData(
#             Employment_Sector=EmploymentSectorEnum.GOVERNMENT,
#             Employment_Tenure_Months=120,  # Long tenure
#             Net_Salary_Per_Cutoff=35000.0,  # Higher salary
#             Salary_Frequency=SalaryFrequencyEnum.MONTHLY,
#             Housing_Status=HousingStatusEnum.OWNED,
#             Years_at_Current_Address=20,  # Long address history
#             Household_Head=YesNoEnum.YES,
#             Number_of_Dependents=1,  # Few dependents
#             Comaker_Relationship=ComakerRelationshipEnum.SPOUSE,
#             Comaker_Employment_Tenure_Months=100,
#             Comaker_Net_Salary_Per_Cutoff=30000.0,
#             Has_Community_Role=YesNoEnum.YES,
#             Paluwagan_Participation=YesNoEnum.YES,
#             Other_Income_Source=OtherIncomeSourceEnum.BUSINESS,
#             Disaster_Preparedness=DisasterPreparednessEnum.INSURANCE,
#             Is_Renewing_Client=1,
#             Grace_Period_Usage_Rate=0.0,  # No grace period usage
#             Late_Payment_Count=0.0,  # No late payments
#             Had_Special_Consideration=0
#         )
        
#         result = get_credit_assessment(low_risk_input)
#         if result:
#             print(f"  POD: {result['probability_of_default']:.6f}")
#             print(f"  Credit Score: {result['credit_score']}")
#             print(f"  Risk Level: {result['risk_level']}")
#             print(f"  Recommendation: {result['recommendation']}")
#         else:
#             print("  ❌ Low-risk test failed")
            
#     except Exception as e:
#         print(f"  ❌ Low-risk test error: {e}")
    
#     print()
#     print("✅ Edge case testing completed!")

# # Example usage and testing
# if __name__ == "__main__":
#     # Run main test
#     test_success = test_prediction_with_sample_data()
    
#     if test_success:
#         print()
#         # Run edge case tests
#         test_edge_cases()
#     else:
#         print()
#         print("❌ Main test failed. Please check:")
#         print("1. Model files exist in './models/' directory")
#         print("2. All required .pkl files are present")
#         print("3. Feature names match between training and prediction")
#         print("4. Model files are not corrupted")