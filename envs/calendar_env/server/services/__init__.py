"""
Services package for calendar application
"""

from .notification_service import NotificationService, get_notification_service

__all__ = [
    "NotificationService",
    "get_notification_service"
]