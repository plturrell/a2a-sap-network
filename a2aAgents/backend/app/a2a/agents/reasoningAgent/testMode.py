"""
Test Mode for Reasoning Agent
Provides test utilities and mock capabilities for testing without API keys
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class TestGrokClient:
    """Test Grok client for testing without API keys"""

    def __init__(self):
        self.call_count = 0
        self.test_mode = True

    async def decompose_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test question decomposition"""
        self.call_count += 1

        # Generate deterministic response based on question
        words = question.lower().split()

        main_concepts = []
        if "what" in words:
            main_concepts.append("definition")
        if "how" in words:
            main_concepts.append("process")
        if "why" in words:
            main_concepts.append("causation")

        sub_questions = []
        if len(words) > 5:
            mid = len(words) // 2
            sub_questions.append(" ".join(words[:mid]) + "?")
            sub_questions.append(" ".join(words[mid:]) + "?")
        else:
            sub_questions.append(question)

        return {
            "success": True,
            "decomposition": {
                "main_concepts": main_concepts or ["general"],
                "sub_questions": sub_questions,
                "reasoning_approach": "analytical",
                "expected_answer_structure": "explanation"
            },
            "model": "test-grok-4",
            "test_mode": True
        }

    async def analyze_patterns(self, text: str, existing_patterns: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test pattern analysis"""
        self.call_count += 1

        # Simple pattern detection
        patterns = []

        if "because" in text.lower() or "therefore" in text.lower():
            patterns.append({"type": "causal", "confidence": 0.8})

        if "if" in text.lower() or "then" in text.lower():
            patterns.append({"type": "conditional", "confidence": 0.7})

        if "and" in text.lower() or "or" in text.lower():
            patterns.append({"type": "logical", "confidence": 0.6})

        return {
            "success": True,
            "patterns": {
                "detected_patterns": patterns,
                "complexity": "moderate" if len(patterns) > 1 else "simple",
                "entities": [w for w in text.split() if len(w) > 5][:3]
            },
            "test_mode": True
        }

    async def synthesize_answer(self, sub_answers: List[Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """Test answer synthesis"""
        self.call_count += 1

        # Combine sub-answers
        combined_content = " ".join([
            ans.get("content", "") for ans in sub_answers
        ])

        synthesis = f"Test synthesis for '{original_question[:30]}...': {combined_content[:100]}..."

        return {
            "success": True,
            "synthesis": synthesis,
            "confidence": 0.75,
            "test_mode": True
        }

    async def async_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Test chat completion"""
        self.call_count += 1

        # Simple response based on last message
        last_message = messages[-1]["content"] if messages else ""

        class TestResponse:
            def __init__(self, content):
                self.content = content
                self.model = "test-grok-4"

        response_text = f"Test response to: {last_message[:50]}..."
        return TestResponse(response_text)


class TestMode:
    """Manager for test mode operations"""

    def __init__(self):
        self.enabled = False
        self.test_clients = {}
        self.operation_delays = {
            "decompose_question": 0.01,
            "analyze_patterns": 0.005,
            "synthesize_answer": 0.008,
            "reason": 0.02
        }

    def enable(self):
        """Enable test mode"""
        self.enabled = True
        logger.info("Test mode enabled - using test responses")

    def disable(self):
        """Disable test mode"""
        self.enabled = False
        logger.info("Test mode disabled - using real APIs")

    def get_test_grok_client(self) -> TestGrokClient:
        """Get or create test Grok client"""
        if "grok" not in self.test_clients:
            self.test_clients["grok"] = TestGrokClient()
        return self.test_clients["grok"]

    async def simulate_delay(self, operation: str):
        """Simulate realistic operation delay"""
        delay = self.operation_delays.get(operation, 0.01)
        await asyncio.sleep(delay)

    def generate_test_reasoning_result(
        self,
        question: str,
        architecture: str,
        confidence_base: float = 0.7
    ) -> Dict[str, Any]:
        """Generate test reasoning result"""
        # Deterministic but varied results
        question_length = len(question)
        complexity_factor = min(1.0, question_length / 100)

        confidence = confidence_base + (0.2 * complexity_factor)

        return {
            "answer": f"Test {architecture} reasoning result for: {question[:50]}...",
            "confidence": min(0.95, confidence),
            "reasoning_type": architecture,
            "test_mode": True,
            "complexity": "high" if complexity_factor > 0.7 else "moderate",
            "execution_time": 0.05 + complexity_factor * 0.1,
            "steps": [
                {"step": 1, "action": "analyze", "result": "analyzed"},
                {"step": 2, "action": "reason", "result": "reasoned"},
                {"step": 3, "action": "synthesize", "result": "synthesized"}
            ]
        }

    def should_use_test_mode(self, api_key: Optional[str] = None) -> bool:
        """Determine if test mode should be used"""
        if self.enabled:
            return True

        if not api_key or api_key == "test" or api_key.startswith("test-"):
            return True

        if api_key.startswith("your-xai-api-key-here") and len(api_key) > 10:
            return False  # Looks like a real API key

        return False  # Default to real mode


# Global test mode instance
_test_mode = TestMode()


def get_test_mode() -> TestMode:
    """Get global test mode instance"""
    return _test_mode


def enable_test_mode():
    """Enable test mode globally"""
    _test_mode.enable()


def disable_test_mode():
    """Disable test mode globally"""
    _test_mode.disable()


def is_test_mode() -> bool:
    """Check if test mode is enabled"""
    return _test_mode.enabled