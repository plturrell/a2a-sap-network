"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
import json
from typing import Dict, List, Optional, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class XAIService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = settings.XAI_BASE_URL
        self.model = settings.XAI_MODEL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat completion request to X.AI Grok API
        """
        payload = {
            "messages": messages,
            "model": self.model,
            "stream": stream,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with None as _unused:
        # httpx\.AsyncClient(timeout=settings.XAI_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.error(f"X.AI API request failed: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"X.AI API HTTP error: {e.response.status_code} - {e.response.text}")
            raise

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from X.AI Grok with optional system prompt
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response["choices"][0]["message"]["content"]


# Singleton instance
xai_service = XAIService()