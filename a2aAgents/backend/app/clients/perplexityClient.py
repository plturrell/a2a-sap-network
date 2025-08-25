"""
Perplexity AI Production Client
Production-ready client for Perplexity API integration with real-time information retrieval
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import httpx  # External service client - allowed to make HTTP calls
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class PerplexityResponse:
    """Structured response from Perplexity API"""
    content: str
    model: str
    usage: Dict[str, int]
    citations: List[Dict[str, Any]]
    finish_reason: str
    raw_response: Dict[str, Any]


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity client"""
    api_key: str
    base_url: str = "https://api.perplexity.ai"
    model: str = "sonar"  # Valid 2025 model name
    timeout: int = 30
    max_retries: int = 3


class PerplexityClient:
    """Production-ready Perplexity API client for A2A agents"""

    def __init__(self, config: Optional[PerplexityConfig] = None):
        """Initialize Perplexity client with configuration"""
        if config is None:
            config = PerplexityConfig(
                api_key=os.getenv('PERPLEXITY_API_KEY'),
                base_url="https://api.perplexity.ai",
                model="sonar"
            )

        if not config.api_key:
            raise ValueError("Perplexity API key is required")

        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Perplexity client initialized with model: {config.model}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Asynchronous chat completion with real-time information access"""
        try:
            payload = {
                "model": model or self.config.model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()

                    return PerplexityResponse(
                        content=data['choices'][0]['message']['content'],
                        model=data['model'],
                        usage=data.get('usage', {}),
                        citations=data.get('citations', []),
                        finish_reason=data['choices'][0].get('finish_reason', 'stop'),
                        raw_response=data
                    )
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Perplexity chat completion error: {e}")
            raise

    async def search_real_time(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: int = 500
    ) -> PerplexityResponse:
        """Search for real-time information"""
        system_prompt = "You are a research assistant with access to real-time information. Provide accurate, up-to-date information with proper citations."

        user_message = f"Context: {context}\n\nSearch query: {query}" if context else query

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            model="sonar"
        )

    async def analyze_financial_news(
        self,
        topic: str,
        time_range: str = "recent",
        analysis_type: str = "summary"
    ) -> PerplexityResponse:
        """Analyze recent financial news on a specific topic"""
        query = f"Latest {time_range} financial news and analysis about {topic}. Provide {analysis_type} with key insights and market implications."

        messages = [
            {
                "role": "system",
                "content": "You are a financial news analyst with access to real-time market information. Provide comprehensive analysis with proper citations."
            },
            {"role": "user", "content": query}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            model="sonar-pro"
        )

    async def research_company(
        self,
        company_name: str,
        research_focus: List[str] = None
    ) -> PerplexityResponse:
        """Research company information with real-time data"""
        if research_focus is None:
            research_focus = ["financial performance", "recent news", "market position"]

        focus_areas = ", ".join(research_focus)
        query = f"Research {company_name} focusing on: {focus_areas}. Include recent developments, financial metrics, and market analysis."

        messages = [
            {
                "role": "system",
                "content": "You are a financial research analyst with access to real-time company data and market information."
            },
            {"role": "user", "content": query}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            model="sonar-pro"
        )

    async def get_market_insights(
        self,
        market_sector: str,
        insight_type: str = "trends"
    ) -> PerplexityResponse:
        """Get real-time market insights for specific sectors"""
        query = f"Current {insight_type} and analysis for {market_sector} sector. Include recent developments, key players, and market outlook."

        messages = [
            {
                "role": "system",
                "content": "You are a market analyst with access to real-time financial data and market intelligence."
            },
            {"role": "user", "content": query}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=700,
            model="sonar"
        )

    async def validate_financial_information(
        self,
        information: str,
        validation_type: str = "accuracy"
    ) -> PerplexityResponse:
        """Validate financial information against real-time sources"""
        query = f"Validate this financial information for {validation_type}: {information}. Check against current sources and provide verification status."

        messages = [
            {
                "role": "system",
                "content": "You are a fact-checker with access to real-time financial data sources. Verify information accuracy and provide evidence."
            },
            {"role": "user", "content": query}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            model="sonar"
        )

    async def process_a2a_research_request(
        self,
        research_topic: str,
        request_context: Dict[str, Any],
        output_format: str = "structured"
    ) -> PerplexityResponse:
        """Process A2A agent research requests"""
        context_str = f"A2A Context: {request_context}"
        query = f"{context_str}\n\nResearch topic: {research_topic}\nOutput format: {output_format}"

        messages = [
            {
                "role": "system",
                "content": f"You are an AI research assistant integrated into an A2A financial system. Provide {output_format} responses that other A2A agents can process efficiently."
            },
            {"role": "user", "content": query}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=600
        )

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the Perplexity client"""
        try:
            response = await self.chat_completion(
                messages=[
                    {"role": "user", "content": "What is the current year? Respond with just the year number."}
                ],
                max_tokens=10,
                temperature=0
            )

            return {
                "status": "healthy",
                "model": self.config.model,
                "citations_available": len(response.citations) > 0,
                "response": response.content.strip(),
                "usage": response.usage
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def close(self):
        """Close client connections (if any persistent connections exist)"""
        logger.info("Perplexity client connections closed")


# Factory function for easy instantiation
def create_perplexity_client(config: Optional[PerplexityConfig] = None) -> PerplexityClient:
    """Factory function to create a Perplexity client"""
    return PerplexityClient(config)


# Singleton instance for global use
_perplexity_client_instance: Optional[PerplexityClient] = None

def get_perplexity_client() -> PerplexityClient:
    """Get singleton Perplexity client instance"""
    global _perplexity_client_instance

    if _perplexity_client_instance is None:
        _perplexity_client_instance = create_perplexity_client()

    return _perplexity_client_instance
