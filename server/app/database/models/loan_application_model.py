from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

from app.schemas.loan_schema import (
    EmploymentSectorEnum,
    SalaryFrequencyEnum,
    HousingStatusEnum,
    ComakerRelationshipEnum,
    YesNoEnum,
    OtherIncomeSourceEnum,
    DisasterPreparednessEnum
)

class ApplicantInfo(BaseModel):
    full_name: str = Field(..., description="Full name of the applicant")
    phone_number: Optional[str] = Field(None, description="Phone number of the applicant")
    address: str = Field(..., description="Address of the applicant")
    salary: float = Field(..., description="Salary of the applicant")

class CoMakerInfo(BaseModel):
    full_name: str = Field(..., description="Full name of the co-maker")
    phone_number: Optional[str] = Field(None, description="Phone number of the co-maker")

class PredictionResult(BaseModel):
    final_credit_score: int = Field(..., description="Final credit score of the applicant")
    probability_of_default: float = Field(..., description="Probability of default for the applicant")
    loan_recommendation: str = Field(..., description="Loan recommendation based on the credit score and probability of default")
    status: str = Field(default="Pending", description="Status of the prediction result")

class ModelInputData(BaseModel):
    Employment_Sector: EmploymentSectorEnum = Field(..., description="Employment sector of the applicant")
    Employment_Tenure_Months: int = Field(..., description="Employment tenure in months")
    Net_Salary_Per_Cutoff: float = Field(..., description="Net salary per cutoff")
    Salary_Frequency: SalaryFrequencyEnum = Field(..., description="Salary frequency of the applicant")
    Housing_Status: HousingStatusEnum = Field(..., description="Housing status of the applicant")
    Years_at_Current_Address: int = Field(..., description="Years at current address")
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
    Late_Payment_Count: float = Field(..., description="Late payment rate of the applicant")
    Had_Special_Consideration: int = Field(..., description="Indicates if the applicant had special consideration in the past")

class LoanApplication(Document):
    application_id: UUID = Field(default_factory=uuid4, description="Unique identifier of the loan application")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp when the loan application was created")
    loan_officer_id: str = Field(..., description="ID of the loan officer handling the application")

    applicant_info: ApplicantInfo = Field(..., description="Information about the loan applicant")
    co_maker_info: CoMakerInfo = Field(..., description="Information about the co-maker")

    model_input_data: ModelInputData = Field(..., description="Input data for the model prediction")

    prediction_result: Optional[PredictionResult] = Field(None, description="Result of the model prediction")

    class Settings:
        name = "loan_applications"  # Collection name in MongoDB
        
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str
        }