from beanie import Document, before_event, Update 
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class User(Document):
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(..., unique=True)
    phone: str = Field(..., min_length=10, max_length=10)
    password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @before_event(Update)
    def update_timestamp(self):
        self.updated_at = datetime.now()

    class Settings:
        name = "User"