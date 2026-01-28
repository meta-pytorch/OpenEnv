"""
Settings database model
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .base import Base

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .base import Base

class Settings(Base):
    """Settings database model"""

    __tablename__ = "settings"

    id = Column(String(255), primary_key=True, nullable=False)  # e.g., "timezone"
    value = Column(String(255), nullable=False)  # e.g., "Asia/Karachi"
    etag = Column(String(255), nullable=True)
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=True, index=True)

    # Relationships
    user = relationship("User")

    def __repr__(self):
        return f"<Setting(id='{self.id}', value='{self.value}', user_id='{self.user_id}')>"
