from pydantic import BaseModel, Field
from datetime import date
from typing import Optional
import uuid

class License(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product: str
    key: str
    expires_at: Optional[date] = None
    active: bool = True
