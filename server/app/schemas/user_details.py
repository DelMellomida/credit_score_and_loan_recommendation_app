from pydantic import BaseModel, Field
from typing import Optional

class UserDetailsCreate(BaseModel):
    user_id: str
    address: str
    occupation: str
    income: float

class UserDetailsUpdate(BaseModel):
    address: Optional[str] = None
    occupation: Optional[str] = None
    income: Optional[float] = None

class UserDetailOut(BaseModel):
    user_id: str
    address: Optional[str] = None
    occupation: Optional[str] = None
    income: Optional[float] = None

class UserDetailDelete(BaseModel):
    user_id: str

    class Config:
        from_attributes = True