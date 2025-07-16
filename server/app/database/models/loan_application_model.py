from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4

from app.schemas.loan_schema import (
    EmploymentSectorEnum,
    SalaryFrequencyEnum,
    HousingStatusEnum,
    ComakerRelationshipEnum,
    YesNoEnum,
    OtherIncomeSourceEnum,
    DisasterPreparednessEnum,
    JobEnum
)

class AIExplanation(BaseModel):
    technical_explanation: str = Field(..., description="Technical explanation of the AI model's prediction")
    business_explanation: str = Field(..., description="Business explanation of the AI model's prediction")
    customer_explanation: str = Field(..., description="Customer-friendly explanation of the AI model's prediction")
    risk_factors: str = Field(..., description="List of risk factors identified by the AI model")
    recommendations: str = Field(..., description="Recommendations based on the AI model's prediction")

class ApplicantInfo(BaseModel):
    full_name: str = Field(..., description="Full name of the applicant")
    contact_number: str = Field(..., description="Contact number of the applicant")  # Changed from phone_number
    address: str = Field(..., description="Address of the applicant")
    salary: str = Field(..., description="Salary of the applicant")  # Changed from float to str
    job: JobEnum = Field(..., description="Job of the applicant")

class CoMakerInfo(BaseModel):
    full_name: str = Field(..., description="Full name of the co-maker")
    contact_number: str = Field(..., description="Contact number of the co-maker")  # Changed from phone_number

class PredictionResult(BaseModel):
    final_credit_score: int = Field(..., description="Final credit score of the applicant")
    default: int = Field(ge=0, le=1, description="Default status of the applicant (0 for no, 1 for yes)")
    probability_of_default: float = Field(..., description="Probability of default for the applicant")
    loan_recommendation: List[str] = Field(default=[], description="Loan recommendation based on the credit score and probability of default")
    status: str = Field(default="Pending", description="Status of the prediction result")

class ModelInputData(BaseModel):
    Employment_Sector: EmploymentSectorEnum = Field(..., description="Employment sector of the applicant")
    Employment_Tenure_Months: int = Field(..., description="Employment tenure in months")
    Net_Salary_Per_Cutoff: float = Field(..., description="Net salary per cutoff")
    Salary_Frequency: SalaryFrequencyEnum = Field(..., description="Salary frequency of the applicant")
    Housing_Status: HousingStatusEnum = Field(..., description="Housing status of the applicant")
    Years_at_Current_Address: float = Field(..., description="Years at current address")  # Changed from int to float
    Household_Head: YesNoEnum = Field(..., description="Indicates if the applicant is the household head")
    Number_of_Dependents: int = Field(..., description="Number of dependents of the applicant")
    Comaker_Relationship: ComakerRelationshipEnum = Field(..., description="Relationship of the co-maker to the applicant")
    Comaker_Employment_Tenure_Months: int = Field(..., description="Employment tenure of the co-maker in months")
    Comaker_Net_Salary_Per_Cutoff: float = Field(..., description="Net salary of the co-maker per cutoff")
    Has_Community_Role: YesNoEnum = Field(..., description="Indicates if the applicant has a community role")
    Paluwagan_Participation: YesNoEnum = Field(..., description="Indicates if the applicant participates in a Paluwagan")
    Other_Income_Source: OtherIncomeSourceEnum = Field(..., description="Other income source of the applicant")
    Disaster_Preparedness: DisasterPreparednessEnum = Field(..., description="Disaster preparedness of the applicant")
    Is_Renewing_Client: int = Field(..., description="Indicates if the applicant is a renewing client")
    Grace_Period_Usage_Rate: float = Field(..., description="Grace period usage rate of the applicant")
    Late_Payment_Count: int = Field(..., description="Late payment count of the applicant")  # Changed from float to int
    Had_Special_Consideration: int = Field(..., description="Indicates if the applicant had special consideration in the past")

class LoanApplication(Document):
    application_id: UUID = Field(default_factory=uuid4, description="Unique identifier of the loan application")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp when the loan application was created")
    loan_officer_id: str = Field(..., description="ID of the loan officer handling the application")

    applicant_info: ApplicantInfo = Field(..., description="Information about the loan applicant")
    comaker_info: CoMakerInfo = Field(..., description="Information about the co-maker")  # Changed from co_maker_info

    model_input_data: ModelInputData = Field(..., description="Input data for the model prediction")

    prediction_result: Optional[PredictionResult] = Field(None, description="Result of the model prediction")
    
    ai_explanation: Optional[AIExplanation] = Field(None, description="Explanation of the AI model's prediction")

    class Settings:
        name = "loan_applications"
        
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }