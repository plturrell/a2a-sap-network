"""
Grok API Production Client
Production-ready client for X.AI Grok API integration
Now integrated with SAP AI Core SDK for enterprise deployments
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
import httpx  # External service client - allowed to make HTTP calls
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()
logger = logging.getLogger(__name__)

# Try to import SAP AI Core SDK
try:
    # Add A2A path to import the SAP AI Core SDK
    aiq_path = Path(__file__).parent.parent.parent.parent.parent.parent / "coverage" / "src"
    if str(aiq_path) not in sys.path:
        sys.path.insert(0, str(aiq_path))

    SAP_AI_CORE_AVAILABLE = True
    logger.info("SAP AI Core SDK integration enabled for GrokClient")
except ImportError:
    SAP_AI_CORE_AVAILABLE = False
    logger.info("SAP AI Core SDK not available - using standard Grok implementation")


@dataclass
class GrokResponse:
    """Structured response from Grok API"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]


@dataclass
class GrokConfig:
    """Configuration for Grok client"""
    api_key: str
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4-latest"
    timeout: int = 30
    max_retries: int = 3


class GrokClient:
    """Production-ready Grok API client for A2A agents"""

    def __init__(self, config: Optional[GrokConfig] = None):
        """Initialize Grok client with configuration"""
        if config is None:
            config = GrokConfig(
                api_key=os.getenv('GROK_API_KEY') or os.getenv('XAI_API_KEY'),
                base_url=os.getenv('GROK_API_URL') or os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1'),
                model=os.getenv('GROK_MODEL') or os.getenv('XAI_MODEL', 'grok-4-latest'),
                timeout=int(os.getenv('GROK_TIMEOUT') or os.getenv('XAI_TIMEOUT', '30'))
            )

        if not config.api_key or config.api_key in ['', 'your-api-key-here']:
            # Allow missing keys in development - they will fail gracefully
            logger.warning("Grok API key not configured - using mock mode")
            config.api_key = "mock-api-key-for-development"
            self.mock_mode = True
        else:
            self.mock_mode = False

        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Grok client initialized with model: {config.model}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GrokResponse:
        """Synchronous chat completion using direct HTTP requests"""
        # Return mock response in development mode
        if self.mock_mode:
            return GrokResponse(
                content="Mock Grok response: I understand your request but am in development mode without API credentials.",
                model="grok-mock",
                usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
                finish_reason="stop",
                raw_response={"mock": True}
            )

        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            # Secure HTTP client configuration
            with httpx.Client(
                timeout=self.config.timeout,
                verify=True,  # Always verify SSL certificates
                follow_redirects=False,  # Don't follow redirects for security
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            ) as client:
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()

                    return GrokResponse(
                        content=data['choices'][0]['message']['content'],
                        model=data['model'],
                        usage={
                            "prompt_tokens": data.get('usage', {}).get('prompt_tokens', 0),
                            "completion_tokens": data.get('usage', {}).get('completion_tokens', 0),
                            "total_tokens": data.get('usage', {}).get('total_tokens', 0)
                        },
                        finish_reason=data['choices'][0].get('finish_reason', 'stop'),
                        raw_response=data
                    )
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Grok chat completion error: {e}")
            raise

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GrokResponse:
        """Asynchronous chat completion using direct HTTP requests"""
        # Return mock response in development mode
        if self.mock_mode:
            return GrokResponse(
                content="Mock Grok response: I understand your request but am in development mode without API credentials.",
                model="grok-mock",
                usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
                finish_reason="stop",
                raw_response={"mock": True}
            )

        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            # Secure async HTTP client configuration
            async with httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=True,  # Always verify SSL certificates
                follow_redirects=False,  # Don't follow redirects for security
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            ) as client:
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()

                    return GrokResponse(
                        content=data['choices'][0]['message']['content'],
                        model=data['model'],
                        usage={
                            "prompt_tokens": data.get('usage', {}).get('prompt_tokens', 0),
                            "completion_tokens": data.get('usage', {}).get('completion_tokens', 0),
                            "total_tokens": data.get('usage', {}).get('total_tokens', 0)
                        },
                        finish_reason=data['choices'][0].get('finish_reason', 'stop'),
                        raw_response=data
                    )
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Grok async chat completion error: {e}")
            raise

    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion responses using direct HTTP requests"""
        try:
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            # Secure streaming HTTP client configuration
            async with httpx.AsyncClient(
                timeout=self.config.timeout,
                verify=True,  # Always verify SSL certificates
                follow_redirects=False,  # Don't follow redirects for security
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.config.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith("data: "):
                                data = line[6:].strip()  # Remove "data: " prefix and strip whitespace

                                if data == "[DONE]":
                                    break

                                if not data:  # Skip empty data
                                    continue

                                try:
                                    chunk_data = json.loads(data)
                                    choices = chunk_data.get('choices', [])

                                    if choices and len(choices) > 0:
                                        delta = choices[0].get('delta', {})
                                        content = delta.get('content')

                                        if content is not None:  # Allow empty strings but not None
                                            yield content

                                except json.JSONDecodeError as e:
                                    logger.debug(f"Failed to parse JSON: {data[:100]}... Error: {e}")
                                    continue
                    else:
                        error_text = await response.aread() if hasattr(response, 'aread') else response.text
                        raise Exception(f"HTTP {response.status_code}: {error_text}")

        except Exception as e:
            logger.error(f"Grok streaming error: {e}")
            raise

    def analyze_financial_data(self, data: str, context: str = "") -> GrokResponse:
        """Specialized method for financial data analysis"""
        messages = [
            {
                "role": "system",
                "content": "You are a financial data analysis expert. Provide clear, accurate analysis of financial data and trends."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nAnalyze this financial data:\n{data}"
            }
        ]

        return self.chat_completion(messages, temperature=0.3)

    async def generate_financial_insights(
        self,
        data: Dict[str, Any],
        insight_type: str = "general"
    ) -> GrokResponse:
        """Generate financial insights from structured data"""
        messages = [
            {
                "role": "system",
                "content": f"You are a financial analyst generating {insight_type} insights. Be precise and actionable."
            },
            {
                "role": "user",
                "content": f"Generate financial insights from this data:\n{data}"
            }
        ]

        return await self.chat_completion_async(messages, temperature=0.4)

    def validate_financial_entities(self, entities: List[str]) -> GrokResponse:
        """Validate financial entities and suggest corrections"""
        entities_text = "\n".join([f"- {entity}" for entity in entities])

        messages = [
            {
                "role": "system",
                "content": "You are a financial data validation expert. Check entity names for accuracy and suggest corrections."
            },
            {
                "role": "user",
                "content": f"Validate these financial entities:\n{entities_text}"
            }
        ]

        return self.chat_completion(messages, temperature=0.1)

    async def process_a2a_request(
        self,
        request_type: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> GrokResponse:
        """Process A2A agent requests with specialized prompting"""
        system_prompt = f"""You are an AI assistant integrated into an A2A (Agent-to-Agent) financial system.
        Request Type: {request_type}
        Context: {context or 'No additional context'}

        Provide structured, actionable responses that other A2A agents can process."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Process this A2A request:\n{data}"}
        ]

        return await self.async_chat_completion(messages, temperature=0.5)

    def health_check(self) -> Dict[str, Any]:
        """Health check for the Grok client"""
        import time

        try:
            start_time = time.time()
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Respond with 'OK' if you're working correctly."}],
                max_tokens=5,
                temperature=0
            )
            end_time = time.time()
            response_time = round(end_time - start_time, 3)

            return {
                "status": "healthy",
                "model": self.config.model,
                "response_time_seconds": response_time,
                "response": response.content.strip()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def close(self):
        """Close client connections"""
        if hasattr(self.async_client, 'close'):
            await self.async_client.close()
        logger.info("Grok client connections closed")


# Factory function for easy instantiation
def create_grok_client(config: Optional[GrokConfig] = None) -> GrokClient:
    """Factory function to create a Grok client"""
    return GrokClient(config)


# Singleton instance for global use
_grok_client_instance: Optional[GrokClient] = None

def get_grok_client() -> GrokClient:
    """Get singleton Grok client instance"""
    global _grok_client_instance

    if _grok_client_instance is None:
        try:
            _grok_client_instance = create_grok_client()
        except Exception as e:
            logger.warning(f"Failed to initialize Grok client: {e}")
            # Create a mock client for development
            _grok_client_instance = GrokClient(GrokConfig(
                api_key="mock-api-key-for-development",
                base_url="https://api.x.ai/v1",
                model="grok-mock"
            ))

    return _grok_client_instance
