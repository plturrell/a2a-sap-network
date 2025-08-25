"""
Grok Reasoning
Uses xAI Grok-4 API for reasoning capabilities
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directory to path to import clients
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from app.clients.grokClient import GrokClient, GrokConfig
except ImportError:
    try:
        from app.a2a.core.grokClient import GrokClient, GrokConfig
    except ImportError:
        # Try direct import for testing
        sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend')
        from app.clients.grokClient import GrokClient, GrokConfig


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies

logger = logging.getLogger(__name__)


class GrokReasoning:
    """Grok-4 reasoning capabilities using xAI API"""

    def __init__(self):
        self.grok_client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize xAI Grok client"""
        try:
            config = GrokConfig(
                api_key=os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'),
                model='grok-4-latest'
            )
            self.grok_client = GrokClient(config)
            logger.info("Grok-4 client initialized")

        except Exception as e:
            logger.warning(f"Could not initialize Grok-4 client: {e}")
            self.grok_client = None

    async def decompose_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Decompose question using Grok-4"""
        if not self.grok_client:
            return {"success": False, "reason": "Grok-4 not available"}

        try:
            prompt = f"""
Analyze and decompose this question:

Question: {question}
{f"Context: {context}" if context else ""}

Provide:
1. Main concepts
2. Sub-questions (ordered by importance)
3. Reasoning approach
4. Expected answer structure

Format as JSON.
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            if response and response.content:
                result = json.loads(response.content)
                return {
                    "success": True,
                    "decomposition": result,
                    "model": response.model
                }

        except Exception as e:
            logger.error(f"Grok-4 decomposition error: {e}")

        return {"success": False, "reason": "Decomposition failed"}

    async def analyze_patterns(self, text: str, existing_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze patterns using Grok-4"""
        if not self.grok_client:
            return {"success": False, "patterns": existing_patterns or []}

        try:
            prompt = f"""
Analyze patterns in this text:

Text: {text}
{f"Existing patterns: {json.dumps(existing_patterns, indent=2)}" if existing_patterns else ""}

Identify:
1. Semantic patterns
2. Logical relationships
3. Key insights
4. Reasoning frameworks

Return as JSON.
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                response_format={"type": "json_object"}
            )

            if response and response.content:
                patterns = json.loads(response.content)
                return {
                    "success": True,
                    "patterns": patterns,
                    "model": response.model
                }

        except Exception as e:
            logger.error(f"Grok-4 pattern analysis error: {e}")

        return {"success": False, "patterns": existing_patterns or []}

    async def synthesize_answer(self, sub_answers: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """Synthesize answer using Grok-4"""
        if not self.grok_client:
            return {"success": False, "reason": "Grok-4 not available"}

        try:
            prompt = f"""
Synthesize a comprehensive answer:

Original Question: {original_question}

Sub-Answers:
{json.dumps(sub_answers, indent=2)}

Create an answer that:
1. Integrates all information
2. Maintains logical flow
3. Directly addresses the question
4. Includes confidence assessment
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )

            if response and response.content:
                return {
                    "success": True,
                    "synthesis": response.content,
                    "model": response.model
                }

        except Exception as e:
            logger.error(f"Grok-4 synthesis error: {e}")

        return {"success": False, "reason": "Synthesis failed"}

    async def optimize_routing(self, message_content: str, available_skills: List[str],
                             current_loads: Dict[str, float] = None) -> Dict[str, Any]:
        """Optimize skill routing using Grok-4"""
        if not self.grok_client:
            return {
                "success": False,
                "recommended_skill": available_skills[0] if available_skills else None
            }

        try:
            prompt = f"""
Optimize routing for this message:

Message: {message_content}
Available Skills: {available_skills}
{f"Current Loads: {current_loads}" if current_loads else ""}

Determine the best skill based on:
1. Message content alignment
2. Load balancing
3. Skill capabilities

Return JSON with: recommended_skill, confidence, reasoning
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            if response and response.content:
                result = json.loads(response.content)
                result['success'] = True
                return result

        except Exception as e:
            logger.error(f"Grok-4 routing error: {e}")

        return {
            "success": False,
            "recommended_skill": available_skills[0] if available_skills else None,
            "confidence": 0.3
        }

    async def analyze_reasoning_quality(self, reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reasoning chain quality using Grok-4"""
        if not self.grok_client:
            return {"success": False}

        try:
            prompt = f"""
Analyze this reasoning chain:

{json.dumps(reasoning_chain, indent=2)}

Evaluate:
1. Logical consistency
2. Evidence strength
3. Inference validity
4. Overall coherence

Provide improvement suggestions.
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )

            if response and response.content:
                return {
                    "success": True,
                    "analysis": response.content,
                    "model": response.model
                }

        except Exception as e:
            logger.error(f"Grok-4 quality analysis error: {e}")

        return {"success": False}

    async def suggest_improvements(self, question: str, current_approach: str,
                                 metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get improvement suggestions from Grok-4"""
        if not self.grok_client:
            return {"success": False, "suggestions": []}

        try:
            prompt = f"""
Suggest improvements for this reasoning approach:

Question: {question}
Current Approach: {current_approach}
{f"Performance Metrics: {json.dumps(metrics, indent=2)}" if metrics else ""}

Provide specific suggestions to improve:
1. Accuracy
2. Efficiency
3. Answer quality
4. Confidence calibration

Format as JSON with actionable suggestions.
"""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            if response and response.content:
                suggestions = json.loads(response.content)
                return {
                    "success": True,
                    "suggestions": suggestions,
                    "model": response.model
                }

        except Exception as e:
            logger.error(f"Grok-4 suggestions error: {e}")

        return {"success": False, "suggestions": []}


# Example usage
if __name__ == "__main__":
    async def test_grok_reasoning():
        grok = GrokReasoning()

        # Test decomposition
        result = await grok.decompose_question(
            "What are the implications of quantum computing on cryptography?"
        )
        print("Decomposition:", json.dumps(result, indent=2))

        # Test pattern analysis
        patterns = await grok.analyze_patterns(
            "Quantum computers can break RSA encryption using Shor's algorithm"
        )
        print("\nPatterns:", json.dumps(patterns, indent=2))

        # Test routing
        routing = await grok.optimize_routing(
            "Analyze system performance metrics",
            ["optimizer", "analyzer", "monitor"],
            {"optimizer": 0.3, "analyzer": 0.7, "monitor": 0.5}
        )
        print("\nRouting:", json.dumps(routing, indent=2))

    asyncio.run(test_grok_reasoning())