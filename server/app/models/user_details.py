from beanie import Document, before_event, Update
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class UserDetails(Document):
    user_id: str
    occupation: Optional[str] = None
    income: Optional[float] = None 
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @before_event(Update)
    def update_timestamp(self):
        self.updated_at = datetime.now()

    class Settings:
        name = "UserDetails"