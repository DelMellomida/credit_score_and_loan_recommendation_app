from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Dict, Any, List, Optional
import logging
from uuid import UUID

from app.services.loan_service import LoanApplicationService, loan_application_service
from app.schemas.loan_schema import (
    FullLoanApplicationRequest, 
    AIExplanation, 
    FullLoanApplicationResponse, 
    ApplicantInfo, 
    CoMakerInfo, 
    LoanApplicationRequest, 
    EmploymentSectorEnum, 
    SalaryFrequencyEnum, 
    HousingStatusEnum, 
    YesNoEnum, 
    ComakerRelationshipEnum, 
    OtherIncomeSourceEnum, 
    DisasterPreparednessEnum,
    RecommendedProducts,
    PaluwaganParticipationEnum,
    CommunityRoleEnum
)
from app.database.models.loan_application_model import (
    LoanApplication, 
    ApplicantInfo as DbApplicantInfo, 
    CoMakerInfo as DbCoMakerInfo, 
    ModelInputData
)
from app.core.auth_dependencies import get_current_user, get_current_active_user

# Configure logging
logger = logging.getLogger(__name__)

def get_loan_application_service() -> LoanApplicationService:
    """
    Dependency to get the loan application service instance.
    
    Raises:
        HTTPException: If loan application service is not initialized
    """
    if loan_application_service is None:
        logger.error("Loan application service is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Loan application service is not initialized. Please contact system administrator."
        )
    
    return loan_application_service

router = APIRouter(prefix="/loans", tags=["Loan Applications"])

@router.post("/applications", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_loan_application(
    request_data: FullLoanApplicationRequest,
    current_user: Dict = Depends(get_current_active_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Creates a new loan application, runs the assessment, 
    and saves the complete record to the database.
    Returns the application details along with prediction results.
    
    Requires authentication - the loan officer ID will be taken from the authenticated user.
    """
    try:
        logger.info(f"Starting loan application creation process for user: {current_user['email']}")
        
        # Get loan officer ID from authenticated user
        loan_officer_id = current_user["id"]  # Use the authenticated user's ID
        
        logger.info("Converting request data to database models")
        
        # Convert request schemas to database model schemas
        try:
            applicant_info = DbApplicantInfo(**request_data.applicant_info.model_dump())
            logger.info("ApplicantInfo created successfully")
        except Exception as e:
            logger.error(f"Error creating ApplicantInfo: {e}")
            raise ValueError(f"Error in applicant info: {e}")
        
        try:
            comaker_info = DbCoMakerInfo(
                full_name=request_data.comaker_info.full_name,
                contact_number=request_data.comaker_info.contact_number
            )
            logger.info("CoMakerInfo created successfully")
        except Exception as e:
            logger.error(f"Error creating CoMakerInfo: {e}")
            raise ValueError(f"Error in comaker info: {e}")
        
        try:
            model_input_data = ModelInputData(**request_data.model_input_data.model_dump())
            logger.info("ModelInputData created successfully")
        except Exception as e:
            logger.error(f"Error creating ModelInputData: {e}")
            raise ValueError(f"Error in model input data: {e}")
        
        # Create the loan application
        try:
            logger.info("Creating LoanApplication document")
            loan_application = LoanApplication(
                loan_officer_id=loan_officer_id,
                applicant_info=applicant_info,
                comaker_info=comaker_info,
                model_input_data=model_input_data
            )
            logger.info("LoanApplication document created successfully")
        except Exception as e:
            logger.error(f"Error creating LoanApplication document: {e}")
            raise ValueError(f"Error creating loan application: {e}")
        
        # Run prediction using the service
        try:
            logger.info("Running prediction for loan application")
            prediction_result = await service._run_prediction(model_input_data)
            
            # Add prediction result to the loan application
            loan_application.prediction_result = prediction_result
            logger.info("Prediction completed and added to loan application")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
        
        # Get loan recommendations using the recommendation service
        try:
            recommended_products = []
            if service.recommendation_service:
                recommended_products = service.recommendation_service.get_loan_recommendations(
                    applicant_info=request_data.applicant_info,
                    model_input_data=request_data.model_input_data.model_dump()
                )
                logger.info("Recommended products are created")
            else:
                logger.warning("Recommendation service not available")
        except Exception as e:
            logger.error(f"Error during recommending products: {e}")
            raise ValueError(f"Recommending products failed: {e}")
        
        try:
            ai_explanation = await service._generate_and_save_explanation(loan_application)
            logger.info("AI explanation generated and saved successfully")
        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            print(e)
            raise RuntimeError(f"AI explanation generation failed: {e}")
        
        # Save to database
        try:
            logger.info("Saving loan application to database")
            await loan_application.save()
            logger.info("Loan application saved to database successfully")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise RuntimeError(f"Database error: {e}")
        
        logger.info(f"Loan application created successfully with ID: {loan_application.application_id}")
        
        return {
            "message": "Loan application created successfully",
            "application_id": str(loan_application.application_id),
            "timestamp": loan_application.timestamp.isoformat(),
            "status": "created",
            "prediction_result": {
                "final_credit_score": prediction_result.final_credit_score,
                "default": prediction_result.default,
                "probability_of_default": prediction_result.probability_of_default,
                "status": prediction_result.status
            },
            "recommended_products": recommended_products,
            "applicant_info": {
                "full_name": loan_application.applicant_info.full_name,
                "contact_number": loan_application.applicant_info.contact_number,
                "address": loan_application.applicant_info.address,
                "salary": loan_application.applicant_info.salary,
                "job": loan_application.applicant_info.job
            },
            "loan_officer_id": loan_application.loan_officer_id,
            "created_by": {
                "email": current_user["email"],
                "full_name": current_user["full_name"]
            },
            "ai_explanation": ai_explanation
        }
        
    except ValueError as e:
        logger.error(f"Validation error during application creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid loan application data: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Runtime error during application creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during application creation: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post("/applications/demo", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_demo_loan_application(
    # Applicant Info
    applicant_name: str,
    contact_number: str,
    address: str,
    salary: str,
    job: str,
    
    # Co-maker Info
    comaker_name: str,
    comaker_contact: str,
    
    # Key Model Inputs
    employment_sector: EmploymentSectorEnum,
    employment_tenure_months: int,
    net_salary_per_cutoff: float,
    salary_frequency: SalaryFrequencyEnum,
    housing_status: HousingStatusEnum,
    years_at_current_address: float,
    household_head: YesNoEnum,
    number_of_dependents: int,
    comaker_relationship: ComakerRelationshipEnum,
    comaker_employment_tenure_months: int,
    comaker_net_salary_per_cutoff: float,
    has_community_role: CommunityRoleEnum,
    paluwagan_participation: PaluwaganParticipationEnum,
    other_income_source: OtherIncomeSourceEnum,
    disaster_preparedness: DisasterPreparednessEnum,
    is_renewing_client: int = 0,
    grace_period_usage_rate: float = 0.0,
    late_payment_count: int = 0,
    had_special_consideration: int = 0,
    
    current_user: Dict = Depends(get_current_active_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Demo endpoint for loan application creation - all fields as individual parameters.
    This will show each field separately in Swagger UI for easy testing and demo purposes.
    Perfect for presentations where you want to fill out fields one by one.
    """
    try:
        logger.info(f"Starting demo loan application creation for user: {current_user['email']}")
        
        # Get loan officer ID from authenticated user
        loan_officer_id = current_user["id"]
        
        # Reconstruct the nested models from individual parameters
        applicant_info = ApplicantInfo(
            full_name=applicant_name,
            contact_number=contact_number,
            address=address,
            salary=salary,
            job=job
        )
        
        comaker_info = CoMakerInfo(
            full_name=comaker_name,
            contact_number=comaker_contact
        )
        
        model_input_data = LoanApplicationRequest(
            Employment_Sector=employment_sector,
            Employment_Tenure_Months=employment_tenure_months,
            Net_Salary_Per_Cutoff=net_salary_per_cutoff,
            Salary_Frequency=salary_frequency,
            Housing_Status=housing_status,
            Years_at_Current_Address=years_at_current_address,
            Household_Head=household_head,
            Number_of_Dependents=number_of_dependents,
            Comaker_Relationship=comaker_relationship,
            Comaker_Employment_Tenure_Months=comaker_employment_tenure_months,
            Comaker_Net_Salary_Per_Cutoff=comaker_net_salary_per_cutoff,
            Has_Community_Role=has_community_role,
            Paluwagan_Participation=paluwagan_participation,
            Other_Income_Source=other_income_source,
            Disaster_Preparedness=disaster_preparedness,
            Is_Renewing_Client=is_renewing_client,
            Grace_Period_Usage_Rate=grace_period_usage_rate,
            Late_Payment_Count=late_payment_count,
            Had_Special_Consideration=had_special_consideration
        )
        
        # Create full request object
        full_request = FullLoanApplicationRequest(
            applicant_info=applicant_info,
            comaker_info=comaker_info,
            model_input_data=model_input_data
        )
        
        # Convert to database models
        try:
            db_applicant_info = DbApplicantInfo(**applicant_info.model_dump())
            db_comaker_info = DbCoMakerInfo(
                full_name=comaker_info.full_name,
                contact_number=comaker_info.contact_number
            )
            db_model_input_data = ModelInputData(**model_input_data.model_dump())
        except Exception as e:
            logger.error(f"Error converting to database models: {e}")
            raise ValueError(f"Error in data conversion: {e}")
        
        # Create the loan application
        try:
            loan_application = LoanApplication(
                loan_officer_id=loan_officer_id,
                applicant_info=db_applicant_info,
                comaker_info=db_comaker_info,
                model_input_data=db_model_input_data
            )
        except Exception as e:
            logger.error(f"Error creating loan application: {e}")
            raise ValueError(f"Error creating loan application: {e}")
        
        # Run prediction using the service
        try:
            logger.info("Running prediction for demo loan application")
            prediction_result = await service._run_prediction(db_model_input_data)
            loan_application.prediction_result = prediction_result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
        
        # Get loan recommendations using the recommendation service
        try:
            recommended_products = []
            if service.recommendation_service:
                recommended_products = service.recommendation_service.get_loan_recommendations(
                    applicant_info=applicant_info,
                    model_input_data=model_input_data.model_dump()
                )
                logger.info("Recommended products are created")
            else:
                logger.warning("Recommendation service not available")
        except Exception as e:
            logger.error(f"Error during recommending products: {e}")
            raise ValueError(f"Recommending products failed: {e}")
        
        try:
            ai_explanation = await service._generate_and_save_explanation(loan_application)
        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            raise RuntimeError(f"AI explanation generation failed: {e}")
        
        # Save to database
        try:
            await loan_application.save()
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            raise RuntimeError(f"Database error: {e}")
        
        logger.info(f"Demo loan application created successfully with ID: {loan_application.application_id}")
        
        return {
            "message": "Demo loan application created successfully",
            "application_id": str(loan_application.application_id),
            "timestamp": loan_application.timestamp.isoformat(),
            "status": "created",
            "demo_mode": True,
            "prediction_result": {
                "final_credit_score": prediction_result.final_credit_score,
                "default": prediction_result.default,
                "probability_of_default": prediction_result.probability_of_default,
                "status": prediction_result.status
            },
            "recommended_products": recommended_products,
            "applicant_info": {
                "full_name": applicant_info.full_name,
                "contact_number": applicant_info.contact_number,
                "address": applicant_info.address,
                "salary": applicant_info.salary,
                "job": applicant_info.job
            },
            "loan_officer_id": loan_application.loan_officer_id,
            "created_by": {
                "email": current_user["email"],
                "full_name": current_user["full_name"]
            },
            "ai_explanation": ai_explanation
        }
        
    except ValueError as e:
        logger.error(f"Validation error during demo application creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid demo application data: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Runtime error during demo application creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during demo application creation: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected demo error: {str(e)}"
        )

@router.get("/applications/{application_id}", response_model=Dict[str, Any])
async def get_loan_application(
    application_id: UUID,
    include_ai_explanation: bool = Query(default=True, description="Include AI explanation in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve a specific loan application by ID.
    Only accessible to authenticated users.
    """
    try:
        application = await service.get_loan_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        # Build response data
        response_data = {
            "application_id": str(application.application_id),
            "timestamp": application.timestamp.isoformat(),
            "loan_officer_id": application.loan_officer_id,
            "applicant_info": application.applicant_info.model_dump(),
            "comaker_info": application.comaker_info.model_dump(),
            "model_input_data": application.model_input_data.model_dump(),
        }
        
        # Add prediction result if available
        if application.prediction_result:
            response_data["prediction_result"] = application.prediction_result.model_dump()
        
        # Add AI explanation if requested and available
        if include_ai_explanation and hasattr(application, 'ai_explanation') and application.ai_explanation:
            response_data["ai_explanation"] = application.ai_explanation.model_dump()
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the application"
        )

@router.get("/applications", response_model=List[Dict[str, Any]])
async def get_loan_applications(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of records to return"),
    loan_officer_id: Optional[str] = Query(default=None, description="Filter by loan officer ID"),
    include_ai_explanation: bool = Query(default=False, description="Include AI explanations in response"),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve loan applications with optional filtering and pagination.
    By default, returns applications created by the authenticated user.
    """
    try:
        # If no specific loan officer ID is provided, default to current user
        if loan_officer_id is None:
            loan_officer_id = current_user["id"]
        
        # Optional: Allow admins to view all applications
        # if current_user.get("role") == "admin":
        #     loan_officer_id = None  # View all applications
        
        applications = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=loan_officer_id
        )
        
        # Format response data
        response_data = []
        for app in applications:
            app_data = {
                "application_id": str(app.application_id),
                "timestamp": app.timestamp.isoformat(),
                "loan_officer_id": app.loan_officer_id,
                "applicant_info": app.applicant_info.model_dump(),
                "comaker_info": app.comaker_info.model_dump(),
            }
            
            # Add prediction result summary
            if app.prediction_result:
                app_data["prediction_result"] = {
                    "final_credit_score": app.prediction_result.final_credit_score,
                    "default": app.prediction_result.default,
                    "probability_of_default": app.prediction_result.probability_of_default,
                    "status": app.prediction_result.status,
                    "recommendation_count": len(app.prediction_result.loan_recommendation or [])
                }
            
            # Add AI explanation if requested and available
            if include_ai_explanation and hasattr(app, 'ai_explanation') and app.ai_explanation:
                app_data["ai_explanation"] = app.ai_explanation.model_dump()
            
            response_data.append(app_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error retrieving loan applications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving applications"
        )

@router.get("/my-applications", response_model=List[Dict[str, Any]])
async def get_my_loan_applications(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    include_ai_explanation: bool = Query(default=False),
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve loan applications created by the authenticated user.
    """
    try:
        applications = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=current_user["id"]
        )
        
        # Format response data
        response_data = []
        for app in applications:
            app_data = {
                "application_id": str(app.application_id),
                "timestamp": app.timestamp.isoformat(),
                "applicant_name": app.applicant_info.full_name,
                "contact_number": app.applicant_info.contact_number,
                "status": app.prediction_result.status if app.prediction_result else "Unknown",
                "credit_score": app.prediction_result.final_credit_score if app.prediction_result else None,
                "recommendation_count": len(app.prediction_result.loan_recommendation or []) if app.prediction_result else 0
            }
            
            # Add AI explanation if requested and available
            if include_ai_explanation and hasattr(app, 'ai_explanation') and app.ai_explanation:
                app_data["ai_explanation_summary"] = {
                    "has_explanation": True,
                    "recommendation": app.ai_explanation.recommendation if hasattr(app.ai_explanation, 'recommendation') else None
                }
            
            response_data.append(app_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error retrieving user's loan applications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving your applications"
        )

@router.put("/applications/{application_id}/status")
async def update_application_status(
    application_id: UUID,
    status_update: Dict[str, str],
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Update the status of a loan application.
    Only accessible to authenticated users.
    """
    try:
        new_status = status_update.get("status")
        if not new_status:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status is required"
            )
        
        # Validate status value
        valid_statuses = ["Success", "Pending", "Rejected", "Approved", "Cancelled"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid statuses are: {', '.join(valid_statuses)}"
            )
        
        updated_application = await service.update_application_status(
            application_id=application_id,
            new_status=new_status
        )
        
        if not updated_application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {
            "message": "Application status updated successfully",
            "application_id": str(application_id),
            "new_status": new_status,
            "updated_at": updated_application.timestamp.isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating application status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the application status"
        )

@router.post("/applications/{application_id}/regenerate-recommendations")
async def regenerate_loan_recommendations(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Regenerate loan recommendations for an existing application.
    """
    try:
        recommendations = await service.regenerate_loan_recommendations(application_id)
        
        if recommendations is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found or recommendation service unavailable"
            )
        
        return {
            "message": "Loan recommendations regenerated successfully",
            "application_id": str(application_id),
            "recommendation_count": len(recommendations),
            "recommendations": recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating recommendations for application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while regenerating recommendations"
        )

@router.delete("/applications/{application_id}")
async def delete_loan_application(
    application_id: UUID,
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Delete a loan application by ID.
    Only accessible to authenticated users.
    """
    try:
        # Optional: Check if user owns this application
        application = await service.get_loan_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        # Optional authorization check
        # if application.loan_officer_id != current_user["id"]:
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="You can only delete your own loan applications"
        #     )
        
        deleted = await service.delete_loan_application(application_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {
            "message": "Loan application deleted successfully",
            "application_id": str(application_id),
            "deleted_at": "now"  # You might want to add actual timestamp
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the application"
        )

@router.get("/health", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint for the loan service.
    """
    try:
        from datetime import datetime
        return {
            "status": "healthy",
            "service": "loan-api",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

@router.get("/service-status", tags=["Health Check"])
async def get_service_status(
    current_user: Dict = Depends(get_current_user),
    service: LoanApplicationService = Depends(get_loan_application_service)
) -> Dict[str, Any]:
    """
    Get the current status of the loan application service.
    Only accessible to authenticated users.
    """
    try:
        logger.info(f"Retrieving loan service status for user: {current_user['email']}")
        status_info = await service.get_service_status()
        
        logger.info("Loan service status retrieved successfully")
        return status_info
        
    except Exception as e:
        logger.error(f"Error retrieving service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve service status"
        )