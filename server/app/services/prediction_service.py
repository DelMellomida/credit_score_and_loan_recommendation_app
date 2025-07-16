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
    def __init__(self, default_threshold: float = 0.5):
        self.model = None
        self.encoder = None
        self.scaler = None
        self.poly = None
        self.selector = None
        self.features_info = None
        self.default_threshold = default_threshold  # Threshold for binary classification
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
        
    def predict(self, input_data: LoanApplicationRequest) -> Dict[str, Any]:
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
                
                # Make binary prediction based on threshold
                binary_prediction = int(probability_of_default >= self.default_threshold)
                
            except Exception as e:
                raise RuntimeError(f"Failed to make prediction: {e}")
            
            # Validate probability is in valid range
            if not (0 <= probability_of_default <= 1):
                logger.warning(f"Probability of default {probability_of_default} is outside [0,1] range. Clipping.")
                probability_of_default = max(0, min(1, probability_of_default))
                # Recalculate binary prediction after clipping
                binary_prediction = int(probability_of_default >= self.default_threshold)
            
            return {
                "probability_of_default": probability_of_default,
                "default_prediction": binary_prediction,
                "threshold_used": self.default_threshold
            }
            
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for binary classification."""
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.default_threshold = threshold
        logger.info(f"Default threshold set to {threshold}")

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
            "current_threshold": self.default_threshold,
            "categorical_features": self.features_info.get('categorical_features', []) if self.features_info else [],
            "numerical_features": self.features_info.get('numerical_features', []) if self.features_info else []
        }

# Initialize the prediction service with proper error handling
def initialize_prediction_service(threshold: float = 0.5) -> Optional[PredictionService]:
    """Initialize the prediction service with proper error handling."""
    try:
        logger.info("Initializing PredictionService...")
        service = PredictionService(default_threshold=threshold)
        logger.info("PredictionService initialized successfully.")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize PredictionService: {e}")
        return None

# Initialize the service
prediction_service = initialize_prediction_service()

# Complete workflow function for your model (updated to include binary prediction)
def get_credit_assessment(input_data, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    """Complete workflow: input -> POD -> binary prediction -> credit score."""
    if prediction_service is None:
        logger.error("PredictionService is not available.")
        return None
    
    try:
        # Set threshold if different from current
        if prediction_service.default_threshold != threshold:
            prediction_service.set_threshold(threshold)
        
        # Step 1: Get probability of default and binary prediction
        prediction_result = prediction_service.predict(input_data)
        pod = prediction_result["probability_of_default"]
        binary_prediction = prediction_result["default_prediction"]
        
        # Step 2: Convert to credit score
        credit_score = PredictionService.transform_pod_to_credit_score(pod)
        
        # Step 3: Add risk assessment
        if credit_score >= 750:
            risk_level = "Low Risk"
        elif credit_score >= 650:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        # Step 4: Recommendation based on binary prediction
        recommendation = "Decline" if binary_prediction == 1 else "Approve"
        
        return {
            "probability_of_default": pod,
            "default_prediction": binary_prediction,
            "credit_score": credit_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "threshold_used": threshold
        }
        
    except Exception as e:
        logger.error(f"Credit assessment failed: {e}")
        return None