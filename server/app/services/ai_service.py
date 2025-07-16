# app/services/ai_service.py

import logging
import json
from typing import Dict, Any, Optional

from fastapi import HTTPException, status
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core import Settings
from app.schemas.loan_schema import LoanApplicationRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FIX 1: Configure the API Key Globally at startup ---
try:
    if not Settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment or settings.")
    genai.configure(api_key=Settings.GEMINI_API_KEY)
    _is_service_configured = True
    logger.info("Google AI SDK configured successfully.")
except (ValueError, AttributeError) as e:
    logger.critical(f"CRITICAL: Gemini API Key is not configured. AI Service will be disabled. Error: {e}")
    _is_service_configured = False


class AIExplainabilityService:
    """Service for generating AI-powered explanations of loan decisions."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        """Initialize the AI explainability service."""
        # --- FIX 2: Instantiate GenerativeModel, not the low-level Client ---
        try:
            if not _is_service_configured:
                raise RuntimeError("AI Service is not configured due to missing API key.")
            
            self.model = genai.GenerativeModel(model_name)
            logger.info(f"AIExplainabilityService initialized successfully with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize AI service model: {e}", exc_info=True)
            raise RuntimeError(f"AI service initialization failed: {e}")

    def generate_loan_explanation(
        self,
        application_data: LoanApplicationRequest,
        prediction_results: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive explanation of loan decision using AI.
        Returns a dictionary of explanations.
        """
        try:
            analysis_data = self._prepare_analysis_data(
                application_data, prediction_results, feature_importance
            )
            explanations = {
                "technical_explanation": self._call_ai_model(self._generate_technical_explanation(analysis_data)),
                "business_explanation": self._call_ai_model(self._generate_business_explanation(analysis_data)),
                "customer_explanation": self._call_ai_model(self._generate_customer_explanation(analysis_data)),
                "risk_factors": self._call_ai_model(self._generate_risk_factors_explanation(analysis_data)),
                "recommendations": self._call_ai_model(self._generate_recommendations(analysis_data)),
            }
            return explanations
        except Exception as e:
            logger.error(f"Failed to generate full loan explanation set: {e}", exc_info=True)
            error_msg = f"Error generating explanation: {e}"
            return {key: error_msg for key in ["technical_explanation", "business_explanation", "customer_explanation", "risk_factors", "recommendations"]}

    def _call_ai_model(self, prompt: str) -> str:
        """Call the AI model with error handling."""
        try:
            generation_config = GenerationConfig(
                temperature=0.3
            )
            
            # --- FIX 3: Call generate_content directly on the self.model object ---
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            logger.error(f"AI model call failed: {e}", exc_info=True)
            return f"Error generating explanation: {str(e)}"

    def _prepare_analysis_data(
        self,
        application_data: LoanApplicationRequest,
        prediction_results: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Prepare data for AI analysis."""
        app_dict = application_data.model_dump()
        monthly_salary = self._calculate_monthly_salary(
            app_dict.get("Net_Salary_Per_Cutoff", 0),
            app_dict.get("Salary_Frequency", "Monthly")
        )
        
        return {
            "application_data": app_dict,
            "prediction_results": prediction_results,
            "feature_importance": feature_importance or {},
            "derived_metrics": {
                "monthly_salary": monthly_salary,
            },
            "risk_assessment": self._assess_risk_factors(app_dict),
            "model_info": {
                "model_type": "Logistic Regression with Polynomial Features",
                "threshold": prediction_results.get("threshold_used", 0.5),
            }
        }

    # All prompt-generation methods remain the same
    def _generate_technical_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "System Instruction: Technical Explanation\n"
            "This applicant’s credit score was calculated using a penalized logistic regression model. It evaluates key financial and behavioral features such as: "
            "income consistency, payment habits, address stability, co-maker credentials, and participation in community or informal lending systems. "
            "The model adds polynomial combinations and selects only statistically strong predictors. This means the score reflects a reliable subset of traits that have shown correlation with default risk. "
            "Each applicant is compared against historical repayment patterns to determine their Probability of Default (POD).\n\n"
            f"Model Inputs and Evaluation Snapshot: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_business_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "System Instruction: Business-Level Interpretation\n"
            "Based on the model output, this score reflects the applicant’s expected repayment reliability. A lower POD suggests low risk of default and potential eligibility for standard lending terms. "
            "A higher POD may trigger stricter conditions or require deeper review. The system supports decision-making by standardizing risk assessment across diverse applicants, "
            "while also adapting to observed trends such as growing credit reliability in previously underserved groups.\n\n"
            f"Business Risk Indicators: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_customer_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "System Instruction: Customer-Focused Summary\n"
            "Your score is based on how closely your financial behavior matches those of previous applicants who repaid successfully. "
            "It looks at things like whether payments were made on time, how steady your income appears, how long you’ve lived at your current address, and your participation in savings groups or community loans. "
            "This score doesn’t label you—it helps us know where we can support you better. If you didn’t score highly this time, it’s not permanent. There are ways forward.\n\n"
            f"Behavioral Summary: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_risk_factors_explanation(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "System Instruction: Default Risk Explanation\n"
            "Based on the applicant’s profile, the model identifies the likelihood they might default on a loan. The most influential risk factors include: "
            "late payments, high dependency ratio, low disaster preparedness, unstable co-maker background, and lack of formal income records. "
            "However, it also accounts for cultural practices—such as paluwagan participation, household leadership roles, and non-traditional sources of income—"
            "which may offer stability not seen in formal systems.\n\n"
            f"Risk Evaluation Summary: {json.dumps(analysis_data, default=str)}"
        )

    def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> str:
        return (
            "System Instruction: Actionable Guidance\n"
            "To improve this score, the applicant may focus on: reducing late payments, strengthening documentation of income (even informal sources), "
            "limiting frequent loan applications, and increasing financial visibility through community lending or cooperatives. "
            "Applicants should also consider tracking paluwagan activity or disaster readiness, which can signal financial resilience in future assessments.\n\n"
            f"Recommended Steps: {json.dumps(analysis_data, default=str)}"
        )

    def _calculate_monthly_salary(self, salary_per_cutoff: float, frequency: str) -> float:
        multipliers = {"Monthly": 1, "Biweekly": 2, "Weekly": 4.33}
        return salary_per_cutoff * multipliers.get(frequency, 1)

    def _assess_risk_factors(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"employment_stability": "High" if app_data.get("Employment_Tenure_Months", 0) > 24 else "Medium"}


# Initialize the service
def initialize_ai_service() -> Optional[AIExplainabilityService]:
    """Initialize the AI explainability service."""
    if not _is_service_configured:
        logger.error("AI service will not be initialized because it was not configured.")
        return None
    try:
        return AIExplainabilityService()
    except Exception as e:
        logger.error(f"Failed to initialize AIExplainabilityService instance: {e}", exc_info=True)
        return None

# Global service instance
ai_service = initialize_ai_service()