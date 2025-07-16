from app.database.models.loan_application_model import LoanApplication, PredictionResult, AIExplanation
import logging
from fastapi import HTTPException, status
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from app.schemas.loan_schema import FullLoanApplicationRequest, RecommendedProducts, ApplicantInfo as ApplicantInfoSchema
from app.loan_product import LOAN_PRODUCTS_CATALOG

from app.services.prediction_service import PredictionService, prediction_service
from app.services.ai_service import AIExplainabilityService, ai_service

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
        self.ai_service = ai_service
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
                loan_recommendation=[],  # Will be populated separately
                status="Success"
            )
            
            logger.info(f"Prediction completed successfully. Credit Score: {credit_score}, POD: {pod:.4f}")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _calculate_max_loan_principal(self, max_amortization: float, monthly_interest_rate: float, term_in_months: int) -> float:
        """
        Calculate the maximum loan principal based on amortization capacity.
        
        Args:
            max_amortization: Maximum affordable monthly amortization
            monthly_interest_rate: Monthly interest rate (as percentage)
            term_in_months: Loan term in months
            
        Returns:
            float: Maximum loan principal
        """
        rate = monthly_interest_rate / 100
        
        # Handle edge case where rate is 0
        if rate == 0:
            return max_amortization * term_in_months
        
        # Using the present value of annuity formula
        # PV = PMT * [(1 - (1 + r)^(-n)) / r]
        discount_factor = (1 - (1 + rate) ** (-term_in_months)) / rate
        principal = max_amortization * discount_factor
        
        return principal
    
    def get_loan_recommendations(
        self,
        applicant_info: ApplicantInfoSchema,
        model_input_data: Dict[str, Any]
    ) -> List[RecommendedProducts]:
        """
        Get loan product recommendations based on applicant information and model data.
        
        Args:
            applicant_info: Applicant information schema
            model_input_data: Model input data dictionary
            
        Returns:
            List[RecommendedProducts]: List of recommended loan products
        """
        eligible_products = []
        is_renewing = model_input_data.get("Is_Renewing_Client") == 1
        
        # Get employment sector from model data
        employment_sector = model_input_data.get("Employment_Sector", "")
        
        for product in LOAN_PRODUCTS_CATALOG:
            rules = product["eligibility_rules"]

            # Rule: Check if the product is for new or existing clients
            if not rules["is_new_client_eligible"] and not is_renewing:
                continue

            # Rule: Check employment sector
            if employment_sector not in rules["employment_sector"]:
                continue

            # Rule: Check specific job (if applicable)
            if rules["job"] and hasattr(applicant_info, 'job') and applicant_info.job not in rules["job"]:
                continue

            eligible_products.append(product)

        if not eligible_products:
            logger.warning("No eligible products found for applicant")
            return []

        ranked_products = []
        net_salary_per_cutoff = model_input_data.get("Net_Salary_Per_Cutoff", 0)

        # Calculate maximum affordable amortization (50% of net salary)
        max_affordable_amortization = net_salary_per_cutoff * 0.50

        for product in eligible_products:
            # Calculate the max loan this person can get for this product
            salary_frequency = model_input_data.get("Salary_Frequency", "Bimonthly")
            
            # Convert per-cutoff amortization to monthly
            if salary_frequency in ["Biweekly", "Bimonthly"]:
                cutoffs_per_month = 2
            else:
                cutoffs_per_month = 1

            max_affordable_monthly_amortization = max_affordable_amortization * cutoffs_per_month

            max_principal = self._calculate_max_loan_principal(
                max_amortization=max_affordable_monthly_amortization,
                monthly_interest_rate=product["interest_rate_monthly"],
                term_in_months=product["max_term_months"]
            )

            # The final loanable amount cannot exceed the product's own maximum
            final_loanable_amount = min(max_principal, product["max_loanable_amount"])

            # Ensure non-negative amount
            final_loanable_amount = max(0, final_loanable_amount)

            # --- Suitability Scoring ---
            score = 100
            
            # Lower interest rate increases score
            score -= product["interest_rate_monthly"] * 10
            
            # Higher potential loan amount increases score
            score += (final_loanable_amount / 10000)
            
            # Longer term increases score (more flexibility)
            score += product["max_term_months"] / 12

            # Major bonus for specialized loans that match the job
            if (product["eligibility_rules"]["job"] and 
                hasattr(applicant_info, 'job') and 
                applicant_info.job in product["eligibility_rules"]["job"]):
                score += 20

            # Bonus for existing client products if applicable
            if not product["eligibility_rules"]["is_new_client_eligible"] and is_renewing:
                score += 10

            ranked_products.append({
                "product_data": product,
                "suitability_score": max(0, int(score)),  # Ensure non-negative score
                "final_loanable_amount": round(final_loanable_amount, -2),  # Round to nearest 100
                "estimated_amortization_per_cutoff": round(max_affordable_amortization, 2),
            })

        # Sort by score, highest first
        sorted_products = sorted(ranked_products, key=lambda x: x["suitability_score"], reverse=True)
        
        recommendation = []
        for i, item in enumerate(sorted_products):
            prod_data = item["product_data"]
            
            # Only include products with meaningful loan amounts
            if item["final_loanable_amount"] > 0:
                recommendation.append(
                    RecommendedProducts(
                        product_name=prod_data["product_name"],
                        is_top_recommendation=(i == 0),
                        max_loanable_amount=item["final_loanable_amount"],
                        interest_rate_monthly=prod_data["interest_rate_monthly"],
                        term_in_months=prod_data["max_term_months"],
                        estimated_amortization_per_cutoff=item["estimated_amortization_per_cutoff"],
                        suitability_score=item["suitability_score"]
                    )
                )
        
        return recommendation


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
        
        if not ai_service:
            logger.warning("AIExplainabilityService is not available for loan service initialization")
        
        service = LoanApplicationService(prediction_service)
        logger.info("LoanApplicationService initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize LoanApplicationService: {e}")
        return None


# Initialize the service
loan_application_service = initialize_loan_application_service()