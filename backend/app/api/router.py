from fastapi import APIRouter

from .endpoints import auth, data, health, users, rateLimits, authentication, securityMonitoring, securityTesting, apiKeys, sessions, compliance

api_router = APIRouter()

# Include all API endpoints
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(authentication.router, tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(rateLimits.router, prefix="/admin", tags=["Rate Limiting"])
api_router.include_router(securityMonitoring.router, prefix="/admin", tags=["Security Monitoring"])
api_router.include_router(securityTesting.router, prefix="/admin", tags=["Security Testing"])
api_router.include_router(apiKeys.router, prefix="/admin", tags=["API Keys"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
api_router.include_router(compliance.router, prefix="/admin", tags=["Compliance"])