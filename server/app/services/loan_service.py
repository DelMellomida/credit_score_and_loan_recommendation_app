from app.database.models.loan_application_model import LoanApplication, PredictionResult
from app.schemas.loan_schema import FullLoanApplicationRequest
from app.services.prediction_service import PredictionService, prediction_service
import logging
from fastapi import HTTPException, status
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

logger = logging.getLogger(__name__)

class LoanApplicationService:
    """
    Service class for handling loan application operations.
    """
    
    def __init__(self, prediction_service: PredictionService):
        """
        Initialize the loan application service with a prediction service.
        
        Args:
            prediction_service: The prediction service instance
        """
        self.prediction_service = prediction_service
        logger.info("LoanApplicationService initialized")

    async def create_loan_application(
        self, 
        request_data: FullLoanApplicationRequest, 
        loan_officer_id: str
    ) -> LoanApplication:
        """
        Create a new loan application record with prediction results.
        
        Args:
            request_data: Complete loan application request data
            loan_officer_id: ID of the loan officer handling the application
            
        Returns:
            LoanApplication: The created loan application with prediction results
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If prediction or database operations fail
        """
        try:
            logger.info(f"Creating loan application for loan officer: {loan_officer_id}")
            
            # Validate input data
            self._validate_loan_application_data(request_data)
            
            # Run prediction using the prediction service
            prediction_result = await self._run_prediction(request_data.model_input_data)
            
            # Create loan application record
            new_application = LoanApplication(
                loan_officer_id=loan_officer_id,
                applicant_info=request_data.applicant_info,
                co_maker_info=request_data.comaker_info,
                model_input_data=request_data.model_input_data,
                prediction_result=prediction_result
            )
            
            # Save to database
            await new_application.insert()
            
            logger.info(f"Loan application created successfully with ID: {new_application.application_id}")
            return {
                "application": new_application,
                "prediction_result": prediction_result
            }
            
        except ValueError as e:
            logger.error(f"Validation error in loan application creation: {e}")
            raise ValueError(f"Invalid loan application data: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating loan application: {e}")
            raise RuntimeError(f"Failed to create loan application: {str(e)}")

    async def get_loan_application(self, application_id: UUID) -> Optional[LoanApplication]:
        """
        Retrieve a loan application by its ID.
        
        Args:
            application_id: UUID of the loan application
            
        Returns:
            Optional[LoanApplication]: The loan application if found, None otherwise
        """
        try:
            logger.info(f"Retrieving loan application with ID: {application_id}")
            application = await LoanApplication.get(application_id)
            
            if application:
                logger.info(f"Loan application {application_id} found")
            else:
                logger.warning(f"Loan application {application_id} not found")
                
            return application
            
        except Exception as e:
            logger.error(f"Error retrieving loan application {application_id}: {e}")
            raise RuntimeError(f"Failed to retrieve loan application: {str(e)}")

    async def get_loan_applications(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        loan_officer_id: Optional[str] = None
    ) -> List[LoanApplication]:
        """
        Retrieve loan applications with optional filtering and pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            loan_officer_id: Optional filter by loan officer ID
            
        Returns:
            List[LoanApplication]: List of loan applications
        """
        try:
            logger.info(f"Retrieving loan applications (skip: {skip}, limit: {limit}, officer: {loan_officer_id})")
            
            # Build query
            query = LoanApplication.find()
            
            if loan_officer_id:
                query = query.find(LoanApplication.loan_officer_id == loan_officer_id)
            
            # Apply pagination
            applications = await query.skip(skip).limit(limit).to_list()
            
            logger.info(f"Retrieved {len(applications)} loan applications")
            return applications
            
        except Exception as e:
            logger.error(f"Error retrieving loan applications: {e}")
            raise RuntimeError(f"Failed to retrieve loan applications: {str(e)}")

    async def update_application_status(
        self, 
        application_id: UUID, 
        new_status: str
    ) -> Optional[LoanApplication]:
        """
        Update the status of a loan application.
        
        Args:
            application_id: UUID of the loan application
            new_status: New status to set
            
        Returns:
            Optional[LoanApplication]: Updated application if found, None otherwise
        """
        try:
            logger.info(f"Updating status for application {application_id} to: {new_status}")
            
            application = await LoanApplication.get(application_id)
            if not application:
                logger.warning(f"Application {application_id} not found for status update")
                return None
            
            # Update the status in prediction_result
            if application.prediction_result:
                application.prediction_result.status = new_status
                await application.save()
                logger.info(f"Status updated successfully for application {application_id}")
            else:
                logger.warning(f"No prediction result found for application {application_id}")
                
            return application
            
        except Exception as e:
            logger.error(f"Error updating application status: {e}")
            raise RuntimeError(f"Failed to update application status: {str(e)}")

    async def delete_loan_application(self, application_id: UUID) -> bool:
        """
        Delete a loan application by its ID.
        
        Args:
            application_id: UUID of the loan application to delete
            
        Returns:
            bool: True if deleted successfully, False if not found
        """
        try:
            logger.info(f"Deleting loan application with ID: {application_id}")
            
            application = await LoanApplication.get(application_id)
            if not application:
                logger.warning(f"Application {application_id} not found for deletion")
                return False
            
            await application.delete()
            logger.info(f"Loan application {application_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting loan application {application_id}: {e}")
            raise RuntimeError(f"Failed to delete loan application: {str(e)}")

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the loan application service.
        
        Returns:
            Dict[str, Any]: Service status information
        """
        try:
            # Check if prediction service is available
            prediction_service_status = "healthy" if self.prediction_service else "unavailable"
            
            # Get basic service info
            status_info = {
                "service": "loan-application-service",
                "status": "healthy",
                "prediction_service_status": prediction_service_status,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
            
            # Add prediction service status if available
            if self.prediction_service:
                try:
                    pred_status = self.prediction_service.get_service_status()
                    status_info["prediction_service_details"] = pred_status
                except Exception as e:
                    logger.warning(f"Could not get prediction service status: {e}")
                    status_info["prediction_service_details"] = {"error": str(e)}
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            raise RuntimeError(f"Failed to get service status: {str(e)}")

    def _validate_loan_application_data(self, request_data: FullLoanApplicationRequest) -> None:
        """
        Validate loan application data before processing.
        
        Args:
            request_data: The loan application request data
            
        Raises:
            ValueError: If validation fails
        """
        # Validate applicant info
        if not request_data.applicant_info.full_name.strip():
            raise ValueError("Applicant full name is required")
        
        if not request_data.applicant_info.contact_number.strip():
            raise ValueError("Applicant contact number is required")
        
        if not request_data.applicant_info.address.strip():
            raise ValueError("Applicant address is required")
        
        # Validate co-maker info
        if not request_data.comaker_info.full_name.strip():
            raise ValueError("Co-maker full name is required")
        
        if not request_data.comaker_info.contact_number.strip():
            raise ValueError("Co-maker contact number is required")
        
        # Validate model input data
        model_data = request_data.model_input_data
        
        if model_data.Employment_Tenure_Months <= 0:
            raise ValueError("Employment tenure must be greater than 0")
        
        if model_data.Net_Salary_Per_Cutoff <= 0:
            raise ValueError("Net salary must be greater than 0")
        
        if model_data.Years_at_Current_Address < 0:
            raise ValueError("Years at current address cannot be negative")
        
        if model_data.Number_of_Dependents < 0:
            raise ValueError("Number of dependents cannot be negative")
        
        if model_data.Comaker_Employment_Tenure_Months <= 0:
            raise ValueError("Co-maker employment tenure must be greater than 0")
        
        if model_data.Comaker_Net_Salary_Per_Cutoff <= 0:
            raise ValueError("Co-maker net salary must be greater than 0")
        
        if not (0 <= model_data.Grace_Period_Usage_Rate <= 1):
            raise ValueError("Grace period usage rate must be between 0 and 1")
        
        if model_data.Late_Payment_Count < 0:
            raise ValueError("Late payment count cannot be negative")

    async def _run_prediction(self, model_input_data) -> PredictionResult:
        """
        Run prediction using the prediction service.
        
        Args:
            model_input_data: Model input data for prediction
            
        Returns:
            PredictionResult: The prediction result
            
        Raises:
            RuntimeError: If prediction fails
        """
        try:
            logger.info("Running prediction for loan application")
            
            # Check if prediction service is available
            if not self.prediction_service:
                raise RuntimeError("Prediction service is not available")
            
            # Run prediction
            pod_result = self.prediction_service.predict(model_input_data)
            pod = pod_result.get("probability_of_default")
            
            if pod is None:
                raise RuntimeError("Prediction service returned invalid result")
            
            # Transform to credit score
            credit_score = self.prediction_service.transform_pod_to_credit_score(pod)
            
            # Generate recommendations
            loan_recommendation = self._generate_loan_recommendation(credit_score, pod)
            
            # Create prediction result
            prediction_result = PredictionResult(
                final_credit_score=credit_score,
                probability_of_default=pod,
                loan_recommendation=loan_recommendation,
                status="Success"
            )
            
            logger.info(f"Prediction completed successfully. Credit Score: {credit_score}, POD: {pod:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _generate_loan_recommendation(self, credit_score: int, pod: float) -> List[str]:
        """
        Generate loan recommendations based on credit score and probability of default.
        
        Args:
            credit_score: Calculated credit score
            pod: Probability of default
            
        Returns:
            List[str]: List of recommendation strings
        """
        recommendations = []
        
        try:
            # Credit score based recommendations
            if credit_score >= 750:
                recommendations.extend([
                    "Excellent credit profile",
                    "Eligible for premium loan products",
                    "Low interest rates available",
                    "High loan amount approval likely"
                ])
            elif credit_score >= 650:
                recommendations.extend([
                    "Good credit profile",
                    "Standard loan products available",
                    "Moderate interest rates",
                    "Standard loan amount approval"
                ])
            elif credit_score >= 550:
                recommendations.extend([
                    "Fair credit profile",
                    "Limited loan products available",
                    "Higher interest rates may apply",
                    "Lower loan amount or co-signer may be required"
                ])
            else:
                recommendations.extend([
                    "Poor credit profile",
                    "High risk applicant",
                    "Loan approval unlikely",
                    "Consider improving credit history before reapplying"
                ])
            
            # POD-specific recommendations
            if pod > 0.5:
                recommendations.append("High default risk - additional documentation required")
            elif pod > 0.3:
                recommendations.append("Moderate default risk - standard verification required")
            else:
                recommendations.append("Low default risk - expedited processing possible")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Unable to generate specific recommendations"]
        
        return recommendations


def initialize_loan_application_service() -> Optional[LoanApplicationService]:
    """
    Initialize the loan application service.
    
    Returns:
        Optional[LoanApplicationService]: The initialized service or None if failed
    """
    try:
        logger.info("Initializing LoanApplicationService...")
        
        # Check if prediction service is available
        if not prediction_service:
            logger.error("Prediction service is not available for loan service initialization")
            return None
        
        service = LoanApplicationService(prediction_service)
        logger.info("LoanApplicationService initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize LoanApplicationService: {e}")
        return None


# Initialize the service
loan_application_service = initialize_loan_application_service()