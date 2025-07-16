from fastapi import APIRouter, HTTPException, status, Depends
from app.services.prediction_service import prediction_service, PredictionService
from app.schemas.loan_schema import LoanApplicationRequest, PredictionResult
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

def get_prediction_service() -> PredictionService:
    """
    Dependency to get the prediction service instance.
    
    Raises:
        HTTPException: If prediction service is not initialized
    """
    if prediction_service is None:
        logger.error("Prediction service is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,  # More appropriate than 500
            detail="Prediction service is not initialized. Please contact system administrator."
        )
    
    # Additional check to ensure service is ready
    if not prediction_service._is_service_ready():
        logger.error("Prediction service is not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not ready. Required model components are not loaded."
        )
    
    return prediction_service

router = APIRouter(prefix="/prediction", tags=["Predictions"])

@router.post("/predict", response_model=PredictionResult, status_code=status.HTTP_200_OK)
async def get_credit_score(
    input_data: LoanApplicationRequest,
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResult:
    """
    Generate credit score and loan recommendation based on input data.
    
    Args:
        input_data: Loan application request data
        service: Prediction service instance
        
    Returns:
        PredictionResult: Contains credit score, probability of default, and recommendation
        
    Raises:
        HTTPException: Various HTTP errors based on the type of failure
    """
    try:
        logger.info("Starting credit score prediction")
        
        # Validate input data is not None (FastAPI should handle this, but extra safety)
        if input_data is None:
            logger.error("Input data is None")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data is required"
            )
        
        # Get probability of default
        try:
            pod_result = service.predict(input_data)
            pod = pod_result.get("probability_of_default")
            
            if pod is None:
                logger.error("Prediction service returned None for probability_of_default")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Prediction service returned invalid result"
                )
                
        except ValueError as e:
            logger.error(f"Input validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid input data: {str(e)}"
            )
        except RuntimeError as e:
            logger.error(f"Runtime error during prediction: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

        # Transform probability to credit score
        try:
            credit_score = service.transform_pod_to_credit_score(pod)
        except (ValueError, TypeError) as e:
            logger.error(f"Credit score transformation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Credit score calculation failed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error in credit score transformation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Credit score calculation failed due to unexpected error"
            )

        # Generate loan recommendation based on credit score
        try:
            loan_recommendation = _generate_loan_recommendation(credit_score, pod)
        except Exception as e:
            logger.error(f"Error generating loan recommendation: {e}")
            # Don't fail the entire request for recommendation errors
            loan_recommendation = ["Unable to generate recommendation"]

        # Validate results before returning
        if not (0 <= pod <= 1):
            logger.warning(f"Probability of default {pod} is outside valid range [0,1]")
            
        if not (300 <= credit_score <= 850):  # Assuming standard credit score range
            logger.warning(f"Credit score {credit_score} is outside typical range [300,850]")

        logger.info(f"Prediction completed successfully. Credit Score: {credit_score}, POD: {pod:.4f}")
        
        return PredictionResult(
            final_credit_score=credit_score,
            probability_of_default=pod,
            loan_recommendation=loan_recommendation,
            status="Success"
        )

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the prediction"
        )

def _generate_loan_recommendation(credit_score: int, pod: float) -> list[str]:
    """
    Generate loan recommendations based on credit score and probability of default.
    
    Args:
        credit_score: Calculated credit score
        pod: Probability of default
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    try:
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
        
        # Add POD-specific recommendations
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

@router.get("/service-status", tags=["Health Check"])
async def get_service_status(
    service: PredictionService = Depends(get_prediction_service)
) -> Dict[str, Any]:
    """
    Get the current status of the prediction service.
    
    Args:
        service: Prediction service instance
        
    Returns:
        Dict containing service status information
        
    Raises:
        HTTPException: If service status cannot be retrieved
    """
    try:
        logger.info("Retrieving service status")
        status_info = service.get_service_status()
        
        # Add additional status information
        status_info.update({
            "service_health": "healthy" if status_info.get("is_ready") else "unhealthy",
            "timestamp": "2025-07-16T12:00:00Z",  # You might want to use actual timestamp
            "version": "1.0.0"  # Add your service version
        })
        
        logger.info("Service status retrieved successfully")
        return status_info
        
    except Exception as e:
        logger.error(f"Error retrieving service status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve service status"
        )

@router.get("/health", tags=["Health Check"])
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint that doesn't require service initialization.
    
    Returns:
        Dict with health status
    """
    try:
        # Basic health check without depending on prediction service
        return {
            "status": "healthy",
            "service": "prediction-api",
            "timestamp": "2025-07-16T12:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )

# Additional utility endpoint for testing
@router.post("/validate-input", tags=["Validation"])
async def validate_input_data(
    input_data: LoanApplicationRequest
) -> Dict[str, Any]:
    """
    Validate input data without running prediction.
    
    Args:
        input_data: Loan application request data
        
    Returns:
        Dict with validation results
    """
    try:
        logger.info("Validating input data")
        
        # Convert to dict for validation
        data_dict = input_data.model_dump()
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "field_count": len(data_dict),
            "data_summary": {}
        }
        
        # Add basic validation checks
        for field, value in data_dict.items():
            if value is None:
                validation_results["errors"].append(f"Field '{field}' is None")
                validation_results["is_valid"] = False
            elif isinstance(value, str) and value.strip() == "":
                validation_results["errors"].append(f"Field '{field}' is empty")
                validation_results["is_valid"] = False
        
        # Add data summary
        validation_results["data_summary"] = {
            "total_fields": len(data_dict),
            "non_null_fields": sum(1 for v in data_dict.values() if v is not None),
            "sample_fields": dict(list(data_dict.items())[:5])  # Show first 5 fields
        }
        
        logger.info(f"Input validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {str(e)}"
        )