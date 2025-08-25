"""
A2A Developer Portal Services
"""

from .email_service import EmailService, create_email_service, EmailMessage, EmailProvider

__all__ = [
    'EmailService',
    'create_email_service',
    'EmailMessage',
    'EmailProvider'
]