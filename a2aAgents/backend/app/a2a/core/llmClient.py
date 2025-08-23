"""
Unified LLM Client for A2A Agents
Integrates SAP AI Core SDK with automatic failover and mode switching
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import sys
from pathlib import Path

# Add A2A path to import the SAP AI Core SDK
aiq_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "coverage" / "src"
if str(aiq_path) not in sys.path:
    sys.path.insert(0, str(aiq_path))

try:
    from aiq.llm.sap_ai_core import LLMService, ExecutionMode, Message
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("SAP AI Core SDK not found. Please ensure A2A dependencies are installed.")
    raise

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """
    Unified LLM client that integrates SAP AI Core SDK for A2A agents.
    
    Features:
    - Development: Grok4 → LNN fallback
    - Production: SAP AI Core (Claude Opus 4) → LNN fallback
    - Automatic mode detection and switching
    - Compatible with existing GrokClient interface
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 800,
                 **kwargs):
        """Initialize unified LLM client.
        
        Args:
            api_key: API key (will use appropriate key based on mode)
            base_url: Base URL (optional)
            model: Model name (optional, will use mode defaults)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_override = model
        
        # Initialize LLM service
        self.llm_service = LLMService()
        
        # Log current mode
        mode = self.llm_service.get_current_mode()
        logger.info(f"UnifiedLLMClient initialized in {mode.value} mode")
        
        # Validate connections
        connections = self.llm_service.validate_connection()
        logger.info(f"Available LLM connections: {connections}")
        
    async def complete(self,
                      prompt: str,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      response_format: str = "text",
                      **kwargs) -> 'LLMResponse':
        """
        Generate completion for a prompt - compatible with GrokClient interface.
        
        Args:
            prompt: The prompt to complete
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: Response format (text or json)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object compatible with GrokClient
        """
        try:
            # Convert to messages format
            messages = [
                Message(role="user", content=prompt)
            ]
            
            # Add system message if needed for JSON response
            if response_format == "json":
                messages.insert(0, Message(
                    role="system",
                    content="You are a helpful AI assistant. Always respond with valid JSON when requested."
                ))
            
            # Generate response using LLM service
            response = await self.llm_service.generate(
                messages=messages,
                model=self.model_override,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )
            
            # Convert to GrokClient-compatible response
            return LLMResponse(
                content=response.content,
                model=response.model,
                usage=response.usage,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            # Return a basic error response
            return LLMResponse(
                content=json.dumps({
                    "error": str(e),
                    "fallback": True
                }) if response_format == "json" else f"Error: {str(e)}",
                model="error",
                usage={"total_tokens": 0}
            )
    
    async def chat(self,
                   messages: List[Dict[str, str]],
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   **kwargs) -> 'LLMResponse':
        """
        Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        try:
            # Convert to Message objects
            message_objects = [
                Message(role=msg["role"], content=msg["content"])
                for msg in messages
            ]
            
            # Generate response
            response = await self.llm_service.generate(
                messages=message_objects,
                model=self.model_override,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.content,
                model=response.model,
                usage=response.usage,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return LLMResponse(
                content=f"Error in chat: {str(e)}",
                model="error",
                usage={"total_tokens": 0}
            )
    
    async def analyze_code(self, code: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze code - specialized method for code analysis.
        
        Args:
            code: Code to analyze
            analysis_type: Type of analysis
            
        Returns:
            Analysis results
        """
        prompt = f"""Analyze the following code ({analysis_type} analysis):

```
{code}
```

Provide a detailed analysis including:
1. Code quality assessment
2. Potential issues or bugs
3. Security concerns
4. Performance considerations
5. Suggestions for improvement

Respond in JSON format."""

        response = await self.complete(
            prompt=prompt,
            response_format="json",
            temperature=0.3  # Lower temperature for code analysis
        )
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "analysis": response.content,
                "type": analysis_type,
                "format": "text"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and service."""
        return self.llm_service.get_info()
    
    def set_mode(self, mode: str):
        """
        Manually set execution mode.
        
        Args:
            mode: 'development', 'production', or 'auto'
        """
        try:
            exec_mode = ExecutionMode(mode.lower())
            self.llm_service.set_mode(exec_mode)
            logger.info(f"Switched to {mode} mode")
        except ValueError:
            logger.error(f"Invalid mode: {mode}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class LLMResponse:
    """Response object compatible with GrokClient."""
    
    def __init__(self, 
                 content: str, 
                 model: str,
                 usage: Optional[Dict[str, int]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.model = model
        self.usage = usage or {"total_tokens": 0}
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


# Compatibility alias for drop-in replacement
GrokClient = UnifiedLLMClient


# Factory function for creating appropriate client
def create_llm_client(**kwargs) -> UnifiedLLMClient:
    """
    Create an LLM client with automatic configuration.
    
    Returns:
        UnifiedLLMClient instance
    """
    # Auto-detect configuration from environment
    config = {}
    
    # Check for model override
    if "GROK_MODEL" in os.environ:
        config["model"] = os.environ["GROK_MODEL"]
    
    # Temperature setting
    if "LLM_TEMPERATURE" in os.environ:
        config["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    
    # Max tokens
    if "LLM_MAX_TOKENS" in os.environ:
        config["max_tokens"] = int(os.environ["LLM_MAX_TOKENS"])
    
    # Merge with provided kwargs
    config.update(kwargs)
    
    return UnifiedLLMClient(**config)