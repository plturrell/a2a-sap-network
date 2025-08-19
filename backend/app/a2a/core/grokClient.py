"""
GrokClient - AI-powered analysis and evaluation client
Provides intelligent evaluation capabilities for calculation testing
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


class GrokClient:
    """
    Client for AI-powered analysis using Grok or compatible AI models
    Provides intelligent evaluation of calculation results and methodology
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("GROK_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("GROK_MODEL", "gpt-4")
        self.client = httpx.AsyncClient(timeout=60.0)

        # Fallback to local analysis if no API key
        self.use_local_analysis = not bool(self.api_key)

        if self.use_local_analysis:
            logger.warning("No API key found, using local rule-based analysis")
        else:
            logger.info(f"Initialized GrokClient with model: {self.model}")

    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze content using AI model or local fallback
        """
        if self.use_local_analysis:
            return await self._local_analysis(prompt, context)
        else:
            return await self._ai_analysis(prompt, context)

    async def _ai_analysis(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform AI-powered analysis using external API
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert mathematical evaluator. Analyze calculation results for accuracy, methodology quality, and explanation clarity. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]

            if context:
                messages[0]["content"] += f" Additional context: {json.dumps(context)}"

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000,
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    if "{" in content and "}" in content:
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        json_str = content[start:end]
                        json.loads(json_str)  # Validate JSON
                        return json_str
                    else:
                        # Fallback to local analysis if no JSON
                        return await self._local_analysis(prompt, context)
                except json.JSONDecodeError:
                    logger.warning("AI response was not valid JSON, using local analysis")
                    return await self._local_analysis(prompt, context)
            else:
                logger.error(f"AI API error: {response.status_code}")
                return await self._local_analysis(prompt, context)

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return await self._local_analysis(prompt, context)

    async def _local_analysis(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform local rule-based analysis as fallback
        """
        try:
            # Extract key information from prompt
            prompt_lower = prompt.lower()

            # Initialize scores
            accuracy_score = 70  # Default moderate score
            methodology_score = 70
            explanation_score = 70

            # Analyze for mathematical keywords and complexity
            math_keywords = ["derivative", "integral", "equation", "solve", "calculate", "formula"]
            if any(keyword in prompt_lower for keyword in math_keywords):
                methodology_score += 10

            # Check for step-by-step content
            if "step" in prompt_lower or "steps:" in prompt_lower:
                explanation_score += 15

            # Check for mathematical expressions
            if any(char in prompt for char in ["=", "+", "-", "*", "/", "^", "x", "y"]):
                accuracy_score += 10

            # Look for detailed explanations
            if len(prompt) > 500:  # Longer explanations get higher scores
                explanation_score += 10
                methodology_score += 5

            # Check for error indicators
            error_keywords = ["error", "wrong", "incorrect", "failed", "undefined"]
            if any(keyword in prompt_lower for keyword in error_keywords):
                accuracy_score -= 20
                methodology_score -= 15

            # Check for methodology terms
            methodology_terms = ["approach", "method", "technique", "algorithm", "process"]
            if any(term in prompt_lower for term in methodology_terms):
                methodology_score += 10

            # Ensure scores are within bounds
            accuracy_score = max(0, min(100, accuracy_score))
            methodology_score = max(0, min(100, methodology_score))
            explanation_score = max(0, min(100, explanation_score))

            # Calculate overall score
            overall_score = accuracy_score * 0.5 + methodology_score * 0.3 + explanation_score * 0.2

            # Generate feedback
            feedback_parts = []
            if accuracy_score >= 80:
                feedback_parts.append("Strong mathematical accuracy")
            elif accuracy_score >= 60:
                feedback_parts.append("Adequate mathematical accuracy")
            else:
                feedback_parts.append("Mathematical accuracy needs improvement")

            if methodology_score >= 80:
                feedback_parts.append("well-explained methodology")
            elif methodology_score >= 60:
                feedback_parts.append("clear methodology")
            else:
                feedback_parts.append("methodology could be clearer")

            if explanation_score >= 80:
                feedback_parts.append("excellent step-by-step explanation")
            elif explanation_score >= 60:
                feedback_parts.append("good explanation")
            else:
                feedback_parts.append("explanation needs more detail")

            feedback = ". ".join(feedback_parts).capitalize() + "."

            result = {
                "accuracy_score": accuracy_score,
                "methodology_score": methodology_score,
                "explanation_score": explanation_score,
                "overall_score": round(overall_score, 1),
                "feedback": feedback,
                "passed": overall_score >= 70,
                "analysis_type": "local_rule_based",
            }

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Local analysis failed: {e}")
            # Return minimal fallback result
            return json.dumps(
                {
                    "accuracy_score": 50,
                    "methodology_score": 50,
                    "explanation_score": 50,
                    "overall_score": 50,
                    "feedback": "Analysis unavailable due to error",
                    "passed": False,
                    "analysis_type": "error_fallback",
                }
            )

    async def evaluate_calculation(
        self,
        question: str,
        answer: Any,
        methodology: str,
        steps: List[Dict[str, Any]],
        expected_answer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Specialized method for evaluating calculation results
        """
        evaluation_prompt = f"""
        Evaluate this mathematical calculation:
        
        Question: {question}
        Provided Answer: {answer}
        Methodology: {methodology}
        Steps: {json.dumps(steps, indent=2)}
        Expected Answer: {expected_answer if expected_answer else "Not provided"}
        
        Provide evaluation scores (0-100) for:
        1. Accuracy of the answer
        2. Quality of methodology explanation
        3. Clarity of step-by-step explanation
        
        Return JSON with: accuracy_score, methodology_score, explanation_score, overall_score, feedback, passed (true/false)
        """

        result = await self.analyze(evaluation_prompt)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error("Failed to parse evaluation result as JSON")
            return {
                "accuracy_score": 50,
                "methodology_score": 50,
                "explanation_score": 50,
                "overall_score": 50,
                "feedback": "Evaluation parsing failed",
                "passed": False,
            }

    async def generate_test_feedback(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive feedback on test results
        """
        if not test_results:
            return {
                "summary": "No test results to analyze",
                "recommendations": [],
                "overall_performance": "insufficient_data",
            }

        # Calculate statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.get("passed", False))
        avg_accuracy = sum(result.get("accuracy_score", 0) for result in test_results) / total_tests
        avg_methodology = (
            sum(result.get("methodology_score", 0) for result in test_results) / total_tests
        )
        avg_explanation = (
            sum(result.get("explanation_score", 0) for result in test_results) / total_tests
        )

        # Generate recommendations
        recommendations = []
        if avg_accuracy < 70:
            recommendations.append("Focus on improving calculation accuracy")
        if avg_methodology < 70:
            recommendations.append("Enhance methodology explanations")
        if avg_explanation < 70:
            recommendations.append("Provide more detailed step-by-step explanations")

        # Determine overall performance
        overall_performance = (
            "excellent"
            if avg_accuracy >= 90
            else (
                "good"
                if avg_accuracy >= 80
                else "satisfactory" if avg_accuracy >= 70 else "needs_improvement"
            )
        )

        return {
            "summary": f"Analyzed {total_tests} tests with {passed_tests} passed ({passed_tests/total_tests*100:.1f}%)",
            "average_scores": {
                "accuracy": round(avg_accuracy, 1),
                "methodology": round(avg_methodology, 1),
                "explanation": round(avg_explanation, 1),
            },
            "recommendations": recommendations,
            "overall_performance": overall_performance,
            "pass_rate": round(passed_tests / total_tests * 100, 1),
        }

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "client"):
            await self.client.aclose()
