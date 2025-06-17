import pickle
import os
from pathlib import Path
from typing import Optional, Any

class CreditScoreModel:
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_path = Path("server/data/models/credit_score_model.pkl")
        
    def save_model(self, model: Any) -> None:
        """Save the trained model to a pickle file"""
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        self.model = model
    
    def load_model(self) -> Any:
        """Load the model from pickle file"""
        if not self.model_path.exists():
            raise FileNotFoundError("Model file not found. Please train the model first.")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        return self.model
    
    def predict(self, features: Any) -> dict:
        """Make predictions using the loaded model"""
        if self.model is None:
            self.load_model()
            
        # Add your prediction logic here
        # This is a placeholder - you'll need to implement the actual prediction
        prediction = self.model.predict(features)
        
        return {
            "credit_score": float(prediction[0]),  # Assuming first value is credit score
            "loan_eligibility": bool(prediction[1]),  # Assuming second value is eligibility
            "recommended_loan_amount": float(prediction[2]) if len(prediction) > 2 else None,
            "interest_rate": float(prediction[3]) if len(prediction) > 3 else None,
            "confidence_score": float(prediction[4]) if len(prediction) > 4 else None
        } 