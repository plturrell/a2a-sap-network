"""
Secure Agent Base Class
Provides security-hardened base implementation for all A2A agents
"""

import asyncio
import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, Request

from .security_middleware import (
    SecurityMiddleware, SecurityConfig, InputValidator,
    get_secure_logger, require_auth, validate_input
)
from ..sdk.types import A2AMessage, MessagePart, MessageRole, AgentCard
from ..sdk.agentBase import A2AAgentBase


class SecureAgentConfig(BaseModel):
    """Configuration for secure agent"""
    
    # Agent identification
    agent_id: str
    agent_name: str
    agent_version: str = "1.0.0"
    description: str = ""
    base_url: str = "http://localhost:4004"
    
    # Security settings
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    max_request_size: int = 1024 * 1024  # 1MB
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Allowed operations
    allowed_operations: Set[str] = Field(default_factory=set)
    
    # API keys (loaded from environment)
    api_keys: Dict[str, str] = Field(default_factory=dict)
    
    @validator('api_keys', pre=True, always=True)
    def load_api_keys(cls, v, values):
        """Load API keys from environment variables"""
        if not v:
            v = {}
        
        # Load common API keys from environment
        api_key_env_vars = [
            'GROK_API_KEY',
            'OPENAI_API_KEY', 
            'ANTHROPIC_API_KEY',
            'PERPLEXITY_API_KEY',
            'XAI_API_KEY'
        ]
        
        for env_var in api_key_env_vars:
            if os.getenv(env_var):
                # Store without exposing in config
                v[env_var] = os.getenv(env_var)
        
        return v
    
    class Config:
        # Prevent API keys from being serialized
        fields = {
            'api_keys': {'exclude': True}
        }


class SecureA2AAgent(A2AAgentBase):
    """
    Security-hardened base class for A2A agents
    Provides authentication, rate limiting, input validation, and secure logging
    """
    
    def __init__(self, config: SecureAgentConfig):
        """Initialize secure agent with security middleware"""
        # Initialize parent SDK
        super().__init__(
            agent_id=config.agent_id,
            name=config.agent_name,  # Changed from agent_name to name
            description=config.description or f"Secure {config.agent_name}",
            base_url=config.base_url,
            capabilities=list(config.allowed_operations),
            version=config.agent_version
        )
        
        self.config = config
        
        # Initialize secure logger
        self.logger = get_secure_logger(f"a2a.agents.{config.agent_id}")
        
        # Initialize security middleware
        security_config = SecurityConfig(
            auth_enabled=config.enable_authentication,
            rate_limit_enabled=config.enable_rate_limiting,
            rate_limit_requests=config.rate_limit_requests,
            rate_limit_window_seconds=config.rate_limit_window,
            max_payload_size=config.max_request_size
        )
        self.security = SecurityMiddleware(security_config)
        
        # Input validator
        self.validator = InputValidator()
        
        # Track active operations for cleanup
        self._active_operations: Set[asyncio.Task] = set()
        
        self.logger.info(
            f"Secure agent initialized",
            extra={
                "agent_id": config.agent_id,
                "security_enabled": {
                    "auth": config.enable_authentication,
                    "rate_limit": config.enable_rate_limiting,
                    "input_validation": config.enable_input_validation
                }
            }
        )
    
    def secure_handler(self, operation: str, require_auth: bool = True):
        """
        Decorator for secure message handlers
        Provides authentication, rate limiting, and input validation
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
                # 1. Check if operation is allowed
                if operation not in self.config.allowed_operations:
                    self.logger.warning(
                        f"Unauthorized operation attempted",
                        extra={"operation": operation, "sender": message.sender_id}
                    )
                    raise HTTPException(
                        status_code=403,
                        detail=f"Operation '{operation}' not allowed for this agent"
                    )
                
                # 2. Validate message structure
                try:
                    self._validate_message(message)
                except ValueError as e:
                    self.logger.error(
                        f"Invalid message structure",
                        extra={"error": str(e), "sender": message.sender_id}
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid message: {str(e)}"
                    )
                
                # 3. Extract and validate input data
                data = {}
                if message.parts:
                    raw_data = message.parts[0].data or {}
                    if self.config.enable_input_validation:
                        data = self._validate_input_data(raw_data, operation)
                    else:
                        data = raw_data
                
                # 4. Check rate limiting
                if self.config.enable_rate_limiting:
                    client_id = f"{message.sender_id}:{operation}"
                    if not await self.security.rate_limiter.check_rate_limit(client_id):
                        self.logger.warning(
                            f"Rate limit exceeded",
                            extra={"sender": message.sender_id, "operation": operation}
                        )
                        return {
                            "status": "error",
                            "error": "Rate limit exceeded. Please try again later."
                        }
                
                # 5. Log operation (securely)
                self.logger.info(
                    f"Processing secure operation",
                    extra={
                        "operation": operation,
                        "sender": message.sender_id,
                        "context_id": context_id,
                        "data_size": len(str(data))
                    }
                )
                
                # 6. Execute handler with timeout
                try:
                    # Create task for tracking
                    task = asyncio.create_task(func(self, message, context_id, data))
                    self._active_operations.add(task)
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(task, timeout=300)  # 5 minute timeout
                    
                    # Clean up
                    self._active_operations.discard(task)
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.logger.error(
                        f"Operation timeout",
                        extra={"operation": operation, "context_id": context_id}
                    )
                    return {
                        "status": "error",
                        "error": "Operation timed out"
                    }
                except Exception as e:
                    self.logger.error(
                        f"Operation failed",
                        extra={
                            "operation": operation,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    return {
                        "status": "error",
                        "error": f"Operation failed: {str(e)}"
                    }
            
            # Store handler for later registration
            if not hasattr(self, '_handlers'):
                self._handlers = {}
            self._handlers[operation] = wrapper
            return wrapper
        
        return decorator
    
    def _validate_message(self, message: A2AMessage) -> None:
        """Validate A2A message structure"""
        if not message.sender_id:
            raise ValueError("Message must have sender_id")
        
        if not message.parts:
            raise ValueError("Message must have at least one part")
        
        # Validate message parts
        for part in message.parts:
            if part.content:
                # Validate content length
                if len(part.content) > self.config.max_request_size:
                    raise ValueError(f"Message content exceeds maximum size")
                
                # Check for script injection in content
                self.validator.validate_no_scripts(part.content, "message content")
    
    def _validate_input_data(self, data: Any, operation: str) -> Any:
        """Validate input data based on operation"""
        # Define validation schemas per operation
        validation_schemas = {
            "calculate": {
                "expression": {"type": "string", "max_length": 1000, "required": True},
                "variables": {"type": "dict", "max_length": 100, "required": False}
            },
            "analyze_code": {
                "code": {"type": "string", "max_length": 100000, "required": True},
                "language": {"type": "string", "max_length": 20, "required": False}
            },
            "process_data": {
                "data": {"type": "any", "required": True},
                "format": {"type": "string", "max_length": 20, "required": False}
            }
        }
        
        # Get schema for operation
        schema = validation_schemas.get(operation, {})
        
        # Validate against schema
        if isinstance(data, dict):
            validated = {}
            for field, rules in schema.items():
                value = data.get(field)
                
                # Check required fields
                if rules.get("required") and value is None:
                    raise ValueError(f"Required field '{field}' is missing")
                
                if value is not None:
                    # Validate type
                    field_type = rules.get("type")
                    if field_type == "string":
                        value = self.validator.validate_string(
                            value, field, rules.get("max_length", 10000)
                        )
                    elif field_type == "dict" and not isinstance(value, dict):
                        raise ValueError(f"Field '{field}' must be a dictionary")
                    
                    validated[field] = value
            
            # Include any additional fields not in schema (after validation)
            for key, value in data.items():
                if key not in validated:
                    if isinstance(value, str):
                        value = self.validator.validate_string(value, key)
                    validated[key] = value
            
            return validated
        else:
            # For non-dict data, apply basic validation
            return self.security.validate_request_data(data)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Securely retrieve API key for external service"""
        # Map service names to environment variable names
        service_map = {
            'grok': 'GROK_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
            'xai': 'XAI_API_KEY'
        }
        
        env_var = service_map.get(service.lower())
        if env_var:
            return self.config.api_keys.get(env_var)
        
        return None
    
    async def make_external_request(self, service: str, endpoint: str, data: Any) -> Dict[str, Any]:
        """
        Make secure external API request through A2A protocol
        This ensures all external requests are logged and audited
        """
        # Get API key
        api_key = self.get_api_key(service)
        if not api_key:
            raise ValueError(f"No API key configured for service: {service}")
        
        # Create A2A message for external request (as per protocol compliance)
        external_request = {
            "service": service,
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log external request (without exposing API key)
        self.logger.info(
            f"External API request",
            extra={
                "service": service,
                "endpoint": endpoint,
                "data_size": len(str(data))
            }
        )
        
        # Route through A2A protocol (not direct HTTP)
        # This is a placeholder - implement according to your A2A protocol
        return {"status": "success", "data": {}}
    
    async def shutdown(self) -> None:
        """Clean shutdown of agent"""
        self.logger.info(f"Shutting down secure agent: {self.config.agent_id}")
        
        # Cancel any active operations
        for task in self._active_operations:
            task.cancel()
        
        # Wait for operations to complete
        if self._active_operations:
            await asyncio.gather(*self._active_operations, return_exceptions=True)
        
        # Call parent shutdown
        await super().shutdown()
    
    def create_secure_response(self, data: Any, status: str = "success") -> Dict[str, Any]:
        """Create a secure response with proper structure"""
        response = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.config.agent_id,
            "version": self.config.agent_version
        }
        
        if status == "success":
            response["data"] = data
        else:
            response["error"] = data
        
        return response


# Example usage for converting existing agents
def convert_to_secure_agent(agent_class):
    """
    Decorator to convert existing agent to use secure base
    """
    class SecureAgent(SecureA2AAgent, agent_class):
        def __init__(self, *args, **kwargs):
            # Extract config
            config = SecureAgentConfig(
                agent_id=kwargs.get('agent_id', 'unknown'),
                agent_name=kwargs.get('agent_name', 'Unknown Agent'),
                allowed_operations=kwargs.get('allowed_operations', set())
            )
            
            # Initialize secure base
            SecureA2AAgent.__init__(self, config)
            
            # Initialize original agent
            agent_class.__init__(self, *args, **kwargs)
    
    return SecureAgent