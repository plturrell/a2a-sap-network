"""
Production Configuration for A2A Developer Portal
All sensitive values should come from environment variables
"""

import os
from typing import Dict, Any


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
class ProductionConfig:
    """Production configuration with secure defaults"""

    # Application settings
    DEBUG = False
    TESTING = False
    PRODUCTION = True

    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")

    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ALGORITHM = "RS256"  # Use RSA for production
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400 * 30  # 30 days
    JWT_PUBLIC_KEY_PATH = os.environ.get('JWT_PUBLIC_KEY_PATH', '/app/certs/public.pem')
    JWT_PRIVATE_KEY_PATH = os.environ.get('JWT_PRIVATE_KEY_PATH', '/app/certs/private.pem')

    # SAP BTP Configuration
    XSUAA_SERVICE_URL = os.environ.get('XSUAA_SERVICE_URL')
    XSUAA_CLIENT_ID = os.environ.get('XSUAA_CLIENT_ID')
    XSUAA_CLIENT_SECRET = os.environ.get('XSUAA_CLIENT_SECRET')

    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DATABASE_POOL_SIZE = int(os.environ.get('DATABASE_POOL_SIZE', '10'))
    DATABASE_MAX_OVERFLOW = int(os.environ.get('DATABASE_MAX_OVERFLOW', '20'))

    # Redis Configuration
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
    REDIS_SSL = os.environ.get('REDIS_SSL', 'false').lower() == 'true'
    REDIS_CONNECTION_POOL_SIZE = int(os.environ.get('REDIS_CONNECTION_POOL_SIZE', '10'))

    # Email Configuration
    EMAIL_SERVICE = os.environ.get('EMAIL_SERVICE', 'smtp')  # smtp, sendgrid, ses
    EMAIL_FROM = os.environ.get('EMAIL_FROM', 'noreply@a2a-portal.com')

    # SMTP Configuration
    SMTP_HOST = os.environ.get('SMTP_HOST')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD')
    SMTP_USE_TLS = os.environ.get('SMTP_USE_TLS', 'true').lower() == 'true'

    # SendGrid Configuration
    SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')

    # AWS SES Configuration
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

    # Deployment Configuration
    KUBERNETES_NAMESPACE = os.environ.get('KUBERNETES_NAMESPACE', 'a2a-agents')
    DOCKER_REGISTRY = os.environ.get('DOCKER_REGISTRY')
    DOCKER_REGISTRY_USERNAME = os.environ.get('DOCKER_REGISTRY_USERNAME')
    DOCKER_REGISTRY_PASSWORD = os.environ.get('DOCKER_REGISTRY_PASSWORD')

    # SAP Destination Service
    DESTINATION_SERVICE_URL = os.environ.get('DESTINATION_SERVICE_URL')
    DESTINATION_CLIENT_ID = os.environ.get('DESTINATION_CLIENT_ID')
    DESTINATION_CLIENT_SECRET = os.environ.get('DESTINATION_CLIENT_SECRET')

    # Monitoring
    ENABLE_TELEMETRY = os.environ.get('ENABLE_TELEMETRY', 'true').lower() == 'true'
    OTEL_EXPORTER_ENDPOINT = os.environ.get('OTEL_EXPORTER_ENDPOINT', os.getenv("A2A_SERVICE_HOST"))

    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' https://ui5.sap.com; style-src 'self' 'unsafe-inline' https://ui5.sap.com; img-src 'self' data: https:; font-src 'self' https://ui5.sap.com;"
    }

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required_vars = [
            'SECRET_KEY',
            'DATABASE_URL',
            'XSUAA_SERVICE_URL',
            'XSUAA_CLIENT_ID',
            'XSUAA_CLIENT_SECRET'
        ]

        missing = []
        for var in required_vars:
            if not os.environ.get(var):
                missing.append(var)

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Validate email configuration
        if cls.EMAIL_SERVICE == 'smtp':
            smtp_required = ['SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD']
            smtp_missing = [var for var in smtp_required if not os.environ.get(var)]
            if smtp_missing:
                raise ValueError(f"SMTP configuration incomplete. Missing: {', '.join(smtp_missing)}")
        elif cls.EMAIL_SERVICE == 'sendgrid' and not cls.SENDGRID_API_KEY:
            raise ValueError("SENDGRID_API_KEY required when EMAIL_SERVICE=sendgrid")
        elif cls.EMAIL_SERVICE == 'ses':
            ses_required = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
            ses_missing = [var for var in ses_required if not os.environ.get(var)]
            if ses_missing:
                raise ValueError(f"AWS SES configuration incomplete. Missing: {', '.join(ses_missing)}")

# Validate configuration on import
ProductionConfig.validate()
