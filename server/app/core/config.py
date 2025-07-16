import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Credit Score and Loan Recommendation"
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()