from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr = Field(..., unique=True)
    phone: str = Field(..., min_length=10, max_length=10)
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v
    
class UserLogin(BaseModel):
    email: EmailStr = Field(..., unique=True)
    password: str = Field(..., min_length=8)
    
class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    phone: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

class UserDelete(BaseModel):
    id: str