import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Credit Score and Loan Recommendation"


settings = Settings()