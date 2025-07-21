from app.database.models.loan_application_model import LoanApplication, PredictionResult, AIExplanation
import logging
from fastapi import HTTPException, status
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from app.schemas.loan_schema import FullLoanApplicationRequest, RecommendedProducts, ApplicantInfo as ApplicantInfoSchema
from app.services.prediction_service import PredictionService, prediction_service
from app.services.ai_service import AIExplainabilityService, ai_service
from app.services.loan_recommendation_service import LoanRecommendationService, loan_recommendation_service

logger = logging.getLogger(__name__)


class LoanApplicationService:
    """
    Service class for handling loan application operations.
    """
    
    def __init__(self, 
                 prediction_service: PredictionService, 
                 recommendation_service: Optional[LoanRecommendationService] = None):
        """
        Initialize the loan application service with required services.
        
        Args:
            prediction_service: The prediction service instance
            recommendation_service: The loan recommendation service instance
        """
        self.prediction_service = prediction_service
        self.ai_service = ai_service
        self.recommendation_service = recommendation_service
        logger.info("LoanApplicationService initialized")

    async def create_loan_application(
        self, 
        request_data: FullLoanApplicationRequest, 
        loan_officer_id: str
    ) -> Dict[str, Any]:
        """
        Create a new loan application record with prediction results.
        
        Args:
            request_data: Complete loan application request data
            loan_officer_id: ID of the loan officer handling the application
            
        Returns:
            Dict containing the created loan application and prediction result
            
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
            
            # Generate loan recommendations if service is available
            if self.recommendation_service:
                loan_recommendations = self.recommendation_service.get_loan_recommendations(
                    applicant_info=request_data.applicant_info,
                    model_input_data=request_data.model_input_data.model_dump()
                )
                prediction_result.loan_recommendation = loan_recommendations
                logger.info(f"Generated {len(loan_recommendations)} loan recommendations")
            else:
                logger.warning("Loan recommendation service not available")
                prediction_result.loan_recommendation = []
            
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
            
            # Generate and save AI explanation asynchronously
            await self._generate_and_save_explanation(new_application)
            
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
        
    async def _generate_and_save_explanation(self, application: LoanApplication) -> Optional[AIExplanation]:
        """
        Generate and save AI explanation for the loan application.
        
        Args:
            application: The loan application to generate explanation for
            
        Returns:
            Optional[AIExplanation]: The generated explanation or None if failed
        """
        if not self.ai_service:
            logger.warning("AIExplainabilityService is not available, skipping explanation generation")
            return None
        
        try:
            logger.info(f"Generating AI explanation for application ID: {application.application_id}")
            
            prediction_result_dict = application.prediction_result.model_dump()
            
            # This is a synchronous call (removed await)
            explanation_dict = self.ai_service.generate_loan_explanation(
                application_data=application.model_input_data,
                prediction_results=prediction_result_dict
            )

            # Handle potential failure from the AI service
            if not explanation_dict or "technical_explanation" not in explanation_dict:
                logger.error(f"AI service failed to return a valid explanation dict. Got: {explanation_dict}")
                return None 

            ai_explanation = AIExplanation(**explanation_dict)
            application.ai_explanation = ai_explanation
            await application.save()
            logger.info(f"AI explanation generated and saved successfully for application ID: {application.application_id}")
            return ai_explanation
        except Exception as e:
            logger.error(f"Error generating AI explanation for application {application.application_id}: {e}", exc_info=True)
            return None

    async def get_loan_application(self, application_id: UUID) -> Optional[LoanApplication]:
        """
        Retrieve a loan application by its application_id UUID.
        
        Args:
            application_id: UUID of the loan application
            
        Returns:
            Optional[LoanApplication]: The loan application if found, None otherwise
        """
        try:
            logger.info(f"Retrieving loan application with application_id: {application_id}")
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            
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
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
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
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            if not application:
                logger.warning(f"Application {application_id} not found for deletion")
                return False
            
            await application.delete()
            logger.info(f"Loan application {application_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting loan application {application_id}: {e}")
            raise RuntimeError(f"Failed to delete loan application: {str(e)}")

    async def regenerate_loan_recommendations(
        self, 
        application_id: UUID
    ) -> Optional[List[RecommendedProducts]]:
        """
        Regenerate loan recommendations for an existing application.
        
        Args:
            application_id: UUID of the loan application
            
        Returns:
            Optional[List[RecommendedProducts]]: Updated recommendations or None if failed
        """
        try:
            logger.info(f"Regenerating loan recommendations for application: {application_id}")
            
            if not self.recommendation_service:
                logger.error("Loan recommendation service not available")
                return None
            
            # Use find_one to search by application_id field instead of _id
            application = await LoanApplication.find_one(LoanApplication.application_id == application_id)
            if not application:
                logger.warning(f"Application {application_id} not found")
                return None
            
            # Generate new recommendations
            new_recommendations = self.recommendation_service.get_loan_recommendations(
                applicant_info=application.applicant_info,
                model_input_data=application.model_input_data.model_dump()
            )
            
            # Update the application
            application.prediction_result.loan_recommendation = new_recommendations
            await application.save()
            
            logger.info(f"Regenerated {len(new_recommendations)} loan recommendations for application {application_id}")
            return new_recommendations
            
        except Exception as e:
            logger.error(f"Error regenerating loan recommendations: {e}")
            return None

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the loan application service.
        
        Returns:
            Dict[str, Any]: Service status information
        """
        try:
            # Check if prediction service is available
            prediction_service_status = "healthy" if self.prediction_service else "unavailable"
            
            # Check if recommendation service is available
            recommendation_service_status = "healthy" if self.recommendation_service else "unavailable"
            
            # Get basic service info
            status_info = {
                "service": "loan-application-service",
                "status": "healthy",
                "prediction_service_status": prediction_service_status,
                "recommendation_service_status": recommendation_service_status,
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
            
            # Add recommendation service status if available
            if self.recommendation_service:
                try:
                    rec_status = self.recommendation_service.get_service_status()
                    status_info["recommendation_service_details"] = rec_status
                except Exception as e:
                    logger.warning(f"Could not get recommendation service status: {e}")
                    status_info["recommendation_service_details"] = {"error": str(e)}
            
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
        if not request_data.applicant_info.full_name.strip():
            raise ValueError("Applicant full name is required")
        
        if not request_data.applicant_info.contact_number.strip():
            raise ValueError("Applicant contact number is required")
        
        if not request_data.applicant_info.address.strip():
            raise ValueError("Applicant address is required")
        
        if not request_data.comaker_info.full_name.strip():
            raise ValueError("Co-maker full name is required")
        
        if not request_data.comaker_info.contact_number.strip():
            raise ValueError("Co-maker contact number is required")
        
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
            default = pod_result.get("default_prediction")
            
            if pod is None:
                raise RuntimeError("Prediction service returned invalid result")
            
            # Transform to credit score
            credit_score = self.prediction_service.transform_pod_to_credit_score(pod)
            
            # Create prediction result
            prediction_result = PredictionResult(
                final_credit_score=credit_score,
                default=default,
                probability_of_default=pod,
                loan_recommendation=[],  # Will be populated later
                status="Success"
            )
            
            logger.info(f"Prediction completed successfully. Credit Score: {credit_score}, POD: {pod:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")


def initialize_loan_application_service() -> Optional[LoanApplicationService]:
    """
    Initialize the loan application service.
    
    Returns:
        Optional[LoanApplicationService]: The initialized service or None if failed
    """
    try:
        logger.info("Initializing LoanApplicationService...")
        
        if not prediction_service:
            logger.error("Prediction service is not available for loan service initialization")
            return None
        
        if not ai_service:
            logger.warning("AIExplainabilityService is not available for loan service initialization")
        
        if not loan_recommendation_service:
            logger.warning("LoanRecommendationService is not available for loan service initialization")
        
        service = LoanApplicationService(
            prediction_service=prediction_service,
            recommendation_service=loan_recommendation_service
        )
        logger.info("LoanApplicationService initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize LoanApplicationService: {e}")
        return None


# Initialize the service
loan_application_service = initialize_loan_application_service()