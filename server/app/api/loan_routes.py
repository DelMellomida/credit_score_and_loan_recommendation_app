from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, List
import logging
from uuid import UUID

from app.services.loan_service import LoanApplicationService, loan_application_service
from app.schemas.loan_schema import FullLoanApplicationRequest, AIExplanation, FullLoanApplicationResponse
from app.database.models.loan_application_model import LoanApplication, ApplicantInfo, CoMakerInfo, ModelInputData

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
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Creates a new loan application, runs the assessment, 
    and saves the complete record to the database.
    Returns the application details along with prediction results.
    """
    try:
        logger.info("Starting loan application creation process")
        
        # In a real application, this would come from the authenticated user
        loan_officer_id = "loan_officer_abc_123"  # Replace with current_user.id
        
        logger.info("Converting request data to database models")
        
        # Convert request schemas to database model schemas
        try:
            applicant_info = ApplicantInfo(**request_data.applicant_info.model_dump())
            logger.info("ApplicantInfo created successfully")
        except Exception as e:
            logger.error(f"Error creating ApplicantInfo: {e}")
            raise ValueError(f"Error in applicant info: {e}")
        
        try:
            comaker_info = CoMakerInfo(
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
        
        try:
            recommended_products = service.get_loan_recommendations(
                applicant_info=request_data.applicant_info,
                model_input_data=request_data.model_input_data.model_dump()
            )
            logger.info("Recommended products are created")
        except Exception as e:
            logger.error("Error during recommending products")
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

@router.get("/applications/{application_id}", response_model=LoanApplication)
async def get_loan_application(
    application_id: UUID,
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve a specific loan application by ID.
    """
    try:
        application = await service.get_loan_application(application_id)
        if not application:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        return application
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving loan application {application_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving the application"
        )

@router.get("/applications", response_model=List[LoanApplication])
async def get_loan_applications(
    skip: int = 0,
    limit: int = 100,
    loan_officer_id: str = None,
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Retrieve loan applications with optional filtering and pagination.
    """
    try:
        applications = await service.get_loan_applications(
            skip=skip,
            limit=limit,
            loan_officer_id=loan_officer_id
        )
        return applications
    except Exception as e:
        logger.error(f"Error retrieving loan applications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving applications"
        )

@router.put("/applications/{application_id}/status")
async def update_application_status(
    application_id: UUID,
    status_update: Dict[str, str],
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Update the status of a loan application.
    """
    try:
        new_status = status_update.get("status")
        if not new_status:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status is required"
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
        
        return {"message": "Application status updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating application status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating the application status"
        )

@router.delete("/applications/{application_id}")
async def delete_loan_application(
    application_id: UUID,
    service: LoanApplicationService = Depends(get_loan_application_service)
):
    """
    Delete a loan application by ID.
    """
    try:
        deleted = await service.delete_loan_application(application_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Loan application with ID {application_id} not found"
            )
        
        return {"message": "Loan application deleted successfully"}
    
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
        return {
            "status": "healthy",
            "service": "loan-api",
            "timestamp": "2025-07-16T12:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

@router.get("/service-status", tags=["Health Check"])
async def get_service_status(
    service: LoanApplicationService = Depends(get_loan_application_service)
) -> Dict[str, Any]:
    """
    Get the current status of the loan application service.
    """
    try:
        logger.info("Retrieving loan service status")
        status_info = await service.get_service_status()
        
        logger.info("Loan service status retrieved successfully")
        return status_info
        
    except Exception as e:
        logger.error(f"Error retrieving service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve service status"
        )