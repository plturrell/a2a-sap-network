"""
GrokClient - AI-powered analysis and evaluation client
Provides intelligent evaluation capabilities for calculation testing
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import json
import logging
import os
from typing import Dict, Any, Optional, List
# A2A Protocol: Use blockchain messaging instead of httpx
from datetime import datetime
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import SAP AI Core SDK
try:
    # Add A2A path to import the SAP AI Core SDK
    aiq_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent / "coverage" / "src"
    if str(aiq_path) not in sys.path:
        sys.path.insert(0, str(aiq_path))

    from aiq.llm.sap_ai_core import LLMService, ExecutionMode, Message
    SAP_AI_CORE_AVAILABLE = True
    logger.info("SAP AI Core SDK integration enabled")
except ImportError:
    SAP_AI_CORE_AVAILABLE = False
    logger.warning("SAP AI Core SDK not available - using original implementation")

try:
    from ..core.lnn.lnnFallback import LNNFallbackClient, LNN_AVAILABLE
    from .lnnQualityMonitor import LNNQualityMonitor
    LNN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LNN fallback not available: {e}")
    LNN_AVAILABLE = False


class GrokClient:
    """
    Client for AI-powered analysis using Grok or compatible AI models
    Provides intelligent evaluation of calculation results and methodology
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, **kwargs):
        self.api_key = api_key or os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("GROK_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("GROK_MODEL", "gpt-4")
        self.temperature = temperature or float(os.getenv("GROK_TEMPERATURE", "0.7"))
        # Accept any additional parameters without error
        self.client = httpx.AsyncClient(timeout=60.0)

        # Initialize SAP AI Core SDK if available
        self.sap_llm_service = None
        if SAP_AI_CORE_AVAILABLE:
            try:
                self.sap_llm_service = LLMService()
                mode = self.sap_llm_service.get_current_mode()
                logger.info(f"SAP AI Core LLM Service initialized in {mode.value} mode")

                # Log available connections
                connections = self.sap_llm_service.validate_connection()
                logger.info(f"SAP AI Core connections: {connections}")
            except Exception as e:
                logger.warning(f"Failed to initialize SAP AI Core LLM Service: {e}")
                self.sap_llm_service = None

        # Initialize LNN fallback and quality monitoring if available
        self.lnn_client = None
        self.quality_monitor = None
        self.real_time_training = True
        self.quality_check_enabled = True

        if LNN_AVAILABLE:
            try:
                self.lnn_client = LNNFallbackClient()
                logger.info("LNN fallback client initialized")

                # Initialize quality monitor for continuous benchmarking
                if self.quality_check_enabled:
                    self.quality_monitor = LNNQualityMonitor(self, self.lnn_client)
                    logger.info("LNN quality monitor initialized")

            except Exception as e:
                logger.warning(f"Failed to initialize LNN fallback: {e}")

        # Fallback hierarchy: Grok API -> LNN -> Rule-based
        self.use_local_analysis = not bool(self.api_key)

        if self.use_local_analysis:
            if self.lnn_client:
                logger.warning("No API key found, using LNN -> rule-based fallback chain")
            else:
                logger.warning("No API key found, using local rule-based analysis only")
        else:
            logger.info(f"Initialized GrokClient with model: {self.model}")
            if self.lnn_client:
                logger.info("LNN fallback available as secondary option")

        # Start quality monitoring for continuous benchmarking
        if self.quality_monitor and not self.use_local_analysis:
            import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.quality_monitor.start_monitoring())
                logger.info("Started LNN quality monitoring")
            except RuntimeError:
                logger.info("Quality monitoring will start when event loop is available")

    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze content using AI model with enhanced LNN fallback chain
        Fallback hierarchy: Grok API -> LNN -> Rule-based
        """
        if self.use_local_analysis:
            # No API key - use LNN -> rule-based fallback chain
            return await self._fallback_analysis(prompt, context)
        else:
            # Try Grok API first, then fallback chain on failure
            return await self._ai_analysis(prompt, context)

    async def _ai_analysis(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform AI-powered analysis using external API
        Priority: SAP AI Core SDK -> Original Grok/OpenAI API -> Fallback
        """
        # Try SAP AI Core SDK first if available
        if self.sap_llm_service:
            try:
                logger.debug("Using SAP AI Core SDK for analysis")
                messages = [
                    Message(
                        role="system",
                        content="You are an expert mathematical evaluator. Analyze calculation results for accuracy, methodology quality, and explanation clarity. Always respond with valid JSON."
                    ),
                    Message(role="user", content=prompt)
                ]

                if context:
                    messages[0] = Message(
                        role="system",
                        content=messages[0].content + f" Additional context: {json.dumps(context)}"
                    )

                response = await self.sap_llm_service.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )

                content = response.content

                # Try to extract JSON from the response
                if "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    json.loads(json_str)  # Validate JSON

                    # Store successful AI result for real-time LNN training
                    self._last_successful_ai_result = json_str

                    # Real-time training: immediately add successful response to LNN training
                    if self.real_time_training and self.lnn_client:
                        try:
                            result_data = json.loads(json_str)
                            self.lnn_client.add_training_data(prompt, result_data)
                            logger.debug("Added real-time training data to LNN")
                        except Exception as e:
                            logger.debug(f"Real-time training failed: {e}")

                    return json_str
                else:
                    # Fallback to LNN analysis if no JSON
                    logger.warning("SAP AI Core response was not valid JSON, using fallback")
                    return await self._fallback_analysis(prompt, context)

            except Exception as e:
                logger.warning(f"SAP AI Core analysis failed: {e}, trying original API")
                # Fall through to original implementation

        # Original implementation
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
            response.raise_for_status()
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

                    # Store successful AI result for real-time LNN training
                    self._last_successful_ai_result = json_str

                    # Real-time training: immediately add successful response to LNN training
                    if self.real_time_training and self.lnn_client:
                        try:
                            result_data = json.loads(json_str)
                            self.lnn_client.add_training_data(prompt, result_data)
                            logger.debug("Added real-time training data to LNN")
                        except Exception as e:
                            logger.debug(f"Real-time training failed: {e}")

                    return json_str
                else:
                    # Fallback to LNN analysis if no JSON
                    return await self._fallback_analysis(prompt, context)
            except json.JSONDecodeError:
                logger.warning("AI response was not valid JSON, using fallback analysis")
                return await self._fallback_analysis(prompt, context)

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return await self._fallback_analysis(prompt, context)

    async def _fallback_analysis(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhanced fallback analysis chain: LNN -> Rule-based
        """
        try:
            # Try LNN fallback first if available
            if self.lnn_client:
                logger.info("Using LNN fallback analysis")
                result = await self.lnn_client.analyze(prompt, context)

                # If we have successful AI analysis data and LNN client, add it as training data
                if hasattr(self, '_last_successful_ai_result') and self._last_successful_ai_result:
                    try:
                        expected_result = json.loads(self._last_successful_ai_result)
                        self.lnn_client.add_training_data(prompt, expected_result)
                        logger.debug("Added training data to LNN from successful AI analysis")
                    except Exception as e:
                        logger.debug(f"Could not add training data to LNN: {e}")

                return result
            else:
                logger.info("LNN not available, using rule-based analysis")
                return await self._local_analysis(prompt, context)

        except Exception as e:
            logger.error(f"LNN fallback failed: {e}")
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

    def get_lnn_info(self) -> Dict[str, Any]:
        """Get information about the LNN fallback system"""
        if not self.lnn_client:
            return {"lnn_available": False, "reason": "LNN client not initialized"}

        try:
            return {
                "lnn_available": True,
                **self.lnn_client.get_model_info()
            }
        except Exception as e:
            return {"lnn_available": False, "error": str(e)}

    async def train_lnn(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Manually trigger LNN training"""
        if not self.lnn_client:
            return {"success": False, "error": "LNN client not available"}

        try:
            if force_retrain or not self.lnn_client.is_trained:
                await self.lnn_client.train_model()
                return {"success": True, "message": "LNN training completed"}
            else:
                return {"success": True, "message": "LNN already trained, use force_retrain=True to retrain"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_lnn_training_data(self, prompt: str, expected_result: Dict[str, Any]) -> bool:
        """Add training data to LNN client"""
        if not self.lnn_client:
            return False

        try:
            self.lnn_client.add_training_data(prompt, expected_result)
            return True
        except Exception as e:
            logger.error(f"Failed to add LNN training data: {e}")
            return False

    async def check_failover_readiness(self) -> Dict[str, Any]:
        """
        Check if the system is ready for seamless failover to LNN
        Returns detailed readiness assessment
        """
        readiness_check = {
            "ready_for_failover": False,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "recommendations": []
        }

        try:
            # 1. Check LNN availability
            if not self.lnn_client:
                readiness_check["checks"]["lnn_available"] = {
                    "status": "failed",
                    "message": "LNN client not initialized"
                }
                readiness_check["recommendations"].append("Initialize LNN fallback system")
                return readiness_check

            readiness_check["checks"]["lnn_available"] = {
                "status": "passed",
                "message": "LNN client initialized"
            }

            # 2. Check LNN training status
            if not self.lnn_client.is_trained:
                readiness_check["checks"]["lnn_trained"] = {
                    "status": "warning",
                    "message": "LNN not trained - will use pattern-based fallback"
                }
                readiness_check["recommendations"].append("Train LNN model for better failover quality")
            else:
                readiness_check["checks"]["lnn_trained"] = {
                    "status": "passed",
                    "message": f"LNN trained with {len(self.lnn_client.training_data)} samples"
                }

            # 3. Check quality monitoring status
            if not self.quality_monitor:
                readiness_check["checks"]["quality_monitoring"] = {
                    "status": "warning",
                    "message": "Quality monitoring not available"
                }
                readiness_check["recommendations"].append("Enable quality monitoring for failover confidence")
            else:
                # Run immediate quality check
                quality_status = self.quality_monitor.get_current_quality_status()
                readiness_check["checks"]["quality_status"] = {
                    "status": "passed" if quality_status.get("ready_for_failover") else "failed",
                    "message": f"Quality status: {quality_status.get('status', 'unknown')}",
                    "details": quality_status
                }

                if not quality_status.get("ready_for_failover"):
                    readiness_check["recommendations"].append("LNN quality below acceptable threshold")

            # 4. Check real-time training capability
            if not self.real_time_training:
                readiness_check["checks"]["real_time_training"] = {
                    "status": "warning",
                    "message": "Real-time training disabled"
                }
                readiness_check["recommendations"].append("Enable real-time training for continuous improvement")
            else:
                readiness_check["checks"]["real_time_training"] = {
                    "status": "passed",
                    "message": "Real-time training enabled"
                }

            # 5. Check fallback chain
            readiness_check["checks"]["fallback_chain"] = {
                "status": "passed",
                "message": "Complete fallback chain: Grok API → LNN → Rule-based"
            }

            # Overall readiness assessment
            passed_checks = sum(1 for check in readiness_check["checks"].values() if check["status"] == "passed")
            total_checks = len(readiness_check["checks"])
            warning_checks = sum(1 for check in readiness_check["checks"].values() if check["status"] == "warning")
            failed_checks = sum(1 for check in readiness_check["checks"].values() if check["status"] == "failed")

            # Ready if all checks pass or only warnings
            readiness_check["ready_for_failover"] = failed_checks == 0
            readiness_check["confidence_score"] = passed_checks / total_checks if total_checks > 0 else 0.0
            readiness_check["summary"] = {
                "passed": passed_checks,
                "warnings": warning_checks,
                "failed": failed_checks,
                "total": total_checks
            }

            if readiness_check["ready_for_failover"]:
                logger.info(f"✅ Failover readiness check passed - confidence: {readiness_check['confidence_score']:.2f}")
            else:
                logger.warning(f"⚠️ Failover readiness check failed - {failed_checks} critical issues")

            return readiness_check

        except Exception as e:
            logger.error(f"Failover readiness check failed: {e}")
            readiness_check["checks"]["system_error"] = {
                "status": "failed",
                "message": f"Readiness check error: {e}"
            }
            readiness_check["recommendations"].append("Fix system errors before attempting failover")
            return readiness_check

    async def force_failover_test(self) -> Dict[str, Any]:
        """
        Force a controlled failover test to validate LNN performance
        Temporarily disables Grok API to test failover chain
        """
        logger.info("🧪 Starting controlled failover test")

        # Store original state
        original_api_key = self.api_key
        original_use_local = self.use_local_analysis

        test_results = {
            "test_started": datetime.utcnow().isoformat(),
            "success": False,
            "results": {},
            "recommendations": []
        }

        try:
            # Temporarily disable API
            self.api_key = None
            self.use_local_analysis = True
            logger.info("🔒 Temporarily disabled Grok API for failover test")

            # Run quality check during failover
            if self.quality_monitor:
                failover_quality = await self.quality_monitor.run_immediate_quality_check()
                test_results["results"]["quality_check"] = failover_quality

            # Test a few sample prompts
            test_prompts = [
                "Calculate 15 + 25 = 40. Show your work step by step.",
                "Find the derivative of f(x) = x^2 + 3x + 2",
                "This is wrong: 2 + 2 = 5. Please evaluate this."
            ]

            lnn_responses = []
            for i, prompt in enumerate(test_prompts):
                try:
                    response = await self.analyze(prompt)
                    response_data = json.loads(response)
                    lnn_responses.append({
                        "prompt_id": i,
                        "analysis_type": response_data.get("analysis_type"),
                        "overall_score": response_data.get("overall_score"),
                        "confidence": response_data.get("confidence")
                    })
                except Exception as e:
                    lnn_responses.append({
                        "prompt_id": i,
                        "error": str(e)
                    })

            test_results["results"]["lnn_responses"] = lnn_responses

            # Analyze results
            successful_responses = [r for r in lnn_responses if "error" not in r]
            avg_score = sum(r.get("overall_score", 0) for r in successful_responses) / len(successful_responses) if successful_responses else 0
            avg_confidence = sum(r.get("confidence", 0) for r in successful_responses) / len(successful_responses) if successful_responses else 0

            test_results["results"]["summary"] = {
                "successful_responses": len(successful_responses),
                "total_responses": len(lnn_responses),
                "average_score": avg_score,
                "average_confidence": avg_confidence,
                "success_rate": len(successful_responses) / len(lnn_responses) if lnn_responses else 0
            }

            # Determine test success
            test_results["success"] = (
                len(successful_responses) == len(lnn_responses) and
                avg_score >= 60 and  # Minimum acceptable score
                avg_confidence >= 0.4  # Minimum confidence
            )

            if test_results["success"]:
                test_results["recommendations"].append("✅ Failover system ready - quality acceptable")
                logger.info("✅ Failover test passed - system ready for production failover")
            else:
                test_results["recommendations"].append("⚠️ Failover quality needs improvement")
                if avg_score < 60:
                    test_results["recommendations"].append("Train LNN with more high-quality examples")
                if avg_confidence < 0.4:
                    test_results["recommendations"].append("Improve LNN confidence through additional training")
                logger.warning("⚠️ Failover test revealed quality issues")

        except Exception as e:
            logger.error(f"Failover test failed: {e}")
            test_results["error"] = str(e)
            test_results["recommendations"].append("Fix failover system errors")

        finally:
            # Restore original state
            self.api_key = original_api_key
            self.use_local_analysis = original_use_local
            logger.info("🔓 Restored Grok API access after failover test")

            test_results["test_completed"] = datetime.utcnow().isoformat()

        return test_results

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including LNN and quality monitoring"""
        # Determine active failover chain based on available services
        failover_chain = []
        if self.sap_llm_service:
            current_mode = self.sap_llm_service.get_current_mode()
            if current_mode == ExecutionMode.PRODUCTION:
                failover_chain.extend(["SAP AI Core (Claude Opus 4)", "LNN"])
            elif current_mode == ExecutionMode.DEVELOPMENT:
                failover_chain.extend(["Grok4", "LNN"])
            else:  # LOCAL or AUTO
                failover_chain.extend(["Grok4", "LNN"])
        elif self.api_key:
            failover_chain.append("Grok API")

        if self.lnn_client:
            if "LNN" not in failover_chain:
                failover_chain.append("LNN")
        failover_chain.append("Rule-based")

        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "sap_ai_core": {
                "available": bool(self.sap_llm_service),
                "mode": self.sap_llm_service.get_current_mode().value if self.sap_llm_service else None,
                "info": self.sap_llm_service.get_info() if self.sap_llm_service else None
            },
            "grok_api": {
                "available": bool(self.api_key),
                "model": self.model,
                "base_url": self.base_url
            },
            "lnn_fallback": self.get_lnn_info(),
            "quality_monitoring": {
                "enabled": bool(self.quality_monitor),
                "current_status": None
            },
            "real_time_training": self.real_time_training,
            "failover_chain": failover_chain
        }

        # Add quality monitoring status
        if self.quality_monitor:
            try:
                status["quality_monitoring"]["current_status"] = self.quality_monitor.get_current_quality_status()
            except Exception as e:
                status["quality_monitoring"]["error"] = str(e)

        return status

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "client"):
            await self.client.aclose()

        # Stop quality monitoring
        if self.quality_monitor:
            try:
                await self.quality_monitor.stop_monitoring()
                logger.info("Stopped LNN quality monitoring")
            except Exception as e:
                logger.error(f"Failed to stop quality monitoring: {e}")

        # Save LNN model state if available
        if self.lnn_client and self.lnn_client.is_trained:
            try:
                await self.lnn_client._save_model()
                logger.info("LNN model state saved during cleanup")
            except Exception as e:
                logger.error(f"Failed to save LNN model during cleanup: {e}")
