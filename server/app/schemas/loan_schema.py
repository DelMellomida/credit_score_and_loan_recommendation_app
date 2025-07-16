from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any

class EmploymentSectorEnum(str, Enum):
    public = "Public"
    private = "Private"

class SalaryFrequencyEnum(str, Enum):
    monthly = "Monthly"
    biweekly = "Biweekly"
    weekly = "Weekly"

class HousingStatusEnum(str, Enum):
    owned = "Owned"
    rented = "Rented"

class ComakerRelationshipEnum(str, Enum):
    spouse = "Spouse"
    sibling = "Sibling"
    parent = "Parent"
    friend = "Friend"

class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"

class OtherIncomeSourceEnum(str, Enum):
    none = "None"
    ofw_remittance = "OFW Remittance"
    freelance = "Freelance"
    business = "Business"

class DisasterPreparednessEnum(str, Enum):
    none = "None"
    savings = "Savings"
    insurance = "Insurance"
    community_plan = "Community Plan"

class LoanApplicationRequest(BaseModel):
    """Defines all the fields a loan officer must submit for a prediction."""
    # This is the Pydantic model for the `model_input_data` field in your database model.
    Employment_Sector: EmploymentSectorEnum
    Employment_Tenure_Months: int = Field(..., gt=0)
    Net_Salary_Per_Cutoff: float = Field(..., gt=0)
    Salary_Frequency: SalaryFrequencyEnum
    Housing_Status: HousingStatusEnum
    Years_at_Current_Address: float = Field(..., ge=0)
    Household_Head: YesNoEnum
    Number_of_Dependents: int = Field(..., ge=0)
    Comaker_Relationship: ComakerRelationshipEnum
    Comaker_Employment_Tenure_Months: int = Field(..., gt=0)
    Comaker_Net_Salary_Per_Cutoff: float = Field(..., gt=0)
    Has_Community_Role: YesNoEnum
    Paluwagan_Participation: YesNoEnum
    Other_Income_Source: OtherIncomeSourceEnum
    Disaster_Preparedness: DisasterPreparednessEnum
    Is_Renewing_Client: int = Field(0, ge=0, le=1)
    Grace_Period_Usage_Rate: float = Field(0.0, ge=0.0, le=1.0)
    Late_Payment_Count: int = Field(0, ge=0)
    Had_Special_Consideration: int = Field(0, ge=0, le=1)

class ApplicantInfo(BaseModel):
    """Schema for the applicant's personal info."""
    full_name: str
    contact_number: str
    address: str
    salary: str

class CoMakerInfo(BaseModel):
    """Schema for the co-maker's personal info."""
    full_name: str
    contact_number: str

class FullLoanApplicationRequest(BaseModel):
    """The complete request body for creating a new loan application record."""
    applicant_info: ApplicantInfo
    comaker_info: CoMakerInfo
    model_input_data: LoanApplicationRequest

class PredictionResult(BaseModel):
    """Schema for the prediction result."""
    final_credit_score: int
    probability_of_default: float
    loan_recommendation: List[str]
    status: str = Field(default="Pending", description="Status of the prediction result")
    
    class Config:
        schema_extra = {
            "example": {
                "final_credit_score": 750,
                "probability_of_default": 0.05,
                "loan_recommendation": "Approved",
                "status": "Pending"
            }
        }