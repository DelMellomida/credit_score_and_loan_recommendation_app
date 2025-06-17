from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Prediction(Document):
    user_id: str
    credit_score: float
    loan_eligibility: bool
    recommended_loan_amount: Optional[float] = None
    interest_rate: Optional[float] = None
    confidence_score: Optional[float] = None
    prediction_date: datetime = Field(default_factory=datetime.now)
    
    class Settings:
        name = "Predictions" 