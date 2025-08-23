"""
Calculation Testing Integration Skills
Provides skills for CalcTesting agent to interact with CalculationAgent and evaluate responses
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



import asyncio
import json
import uuid
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import logging
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
# Import GrokClient for AI evaluation
try:
    from app.a2a.core.grokClient import GrokClient
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    logging.warning("GrokClient not available. AI evaluation will be limited.")

# Import A2A components
from app.a2a.sdk import A2AMessage, MessageRole, MessagePart, create_agent_id

logger = logging.getLogger(__name__)


class TestQuestion(BaseModel):
    """Model for a test question"""
    question_id: str = Field(default_factory=lambda: f"q_{uuid.uuid4().hex[:8]}")
    question: str
    category: str  # mathematical, financial, graph, statistical
    difficulty: str  # easy, medium, hard
    expected_methodology: Optional[str] = None
    expected_steps: Optional[List[str]] = None
    tolerance: Optional[float] = 0.01


class CalculationResult(BaseModel):
    """Model for calculation result from CalculationAgent"""
    answer: Any
    methodology: str
    steps: List[Dict[str, Any]]
    confidence: float
    computation_time: float


class EvaluationScore(BaseModel):
    """Model for evaluation score"""
    question_id: str
    accuracy_score: float  # 0-100
    methodology_score: float  # 0-100
    explanation_score: float  # 0-100
    overall_score: float  # 0-100
    feedback: str
    passed: bool


class Scoreboard(BaseModel):
    """Model for tracking overall performance"""
    total_questions: int = 0
    correct_answers: int = 0
    accuracy_rate: float = 0.0
    average_methodology_score: float = 0.0
    average_explanation_score: float = 0.0
    category_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    difficulty_scores: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class CalcTestingIntegrationSkills:
    """Skills for CalcTesting agent to interact with CalculationAgent"""
    
    def __init__(self, agent):
        self.agent = agent
        self.grok_client = GrokClient() if GROK_AVAILABLE else None
        self.scoreboard = Scoreboard()
        # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        self.http_client = None  # Disabled for A2A protocol compliance
        # self.http_client = httpx.AsyncClient(timeout=60.0)
        # Agent endpoints will be discovered dynamically
        self._agent_endpoints = {}
        self._discovered_agents = {}
        
    async def _discover_agent_endpoint(self, agent_id: str) -> Optional[str]:
        """Discover agent endpoint via Catalog Manager"""
        if agent_id in self._agent_endpoints:
            return self._agent_endpoints[agent_id]
            
        try:
            # Use catalog manager to discover agent
            result = await self.agent._call_catalog_manager("discover_agent", {
                "agent_id": agent_id
            })
            
            if result and not result.get("error"):
                endpoint = result.get("endpoint")
                if endpoint:
                    self._agent_endpoints[agent_id] = endpoint
                    return endpoint
        except Exception as e:
            logger.error(f"Failed to discover agent {agent_id}: {e}")
            
        # Fallback to environment variable
        env_key = f"{agent_id.upper()}_URL"
        return os.getenv(env_key)
    
    async def _wait_for_response(self, message_id: str) -> Dict[str, Any]:
        """Wait for response message from CalculationAgent"""
        # Store pending response in a class-level dict
        if not hasattr(self, '_pending_responses'):
            self._pending_responses = {}
        
        # Create a future for this message
        future = asyncio.get_event_loop().create_future()
        self._pending_responses[message_id] = future
        
        # Wait for the response
        try:
            return await future
        finally:
            # Clean up
            if message_id in self._pending_responses:
                del self._pending_responses[message_id]
    
    def _handle_calculation_response(self, message_id: str, response: Dict[str, Any]):
        """Handle incoming response from CalculationAgent"""
        if hasattr(self, '_pending_responses') and message_id in self._pending_responses:
            future = self._pending_responses[message_id]
            if not future.done():
                future.set_result(response)
    
    async def _handle_blockchain_response(self, blockchain_msg: Dict[str, Any]):
        """Handle blockchain response message from CalculationAgent"""
        try:
            # Parse the content
            content = blockchain_msg.get('content', {})
            if isinstance(content, str):
                content = json.loads(content)
            
            # Extract message ID from content
            original_message_id = content.get('in_reply_to') or blockchain_msg.get('id')
            
            # Handle the response
            self._handle_calculation_response(original_message_id, content)
            
        except Exception as e:
            logger.error(f"Error handling blockchain response: {e}")
    
    async def _call_calculation_agent(self, message: A2AMessage) -> Dict[str, Any]:
        """Send message to CalculationAgent via A2A blockchain network"""
        try:
            # Import A2A blockchain integration from a2aNetwork
            import sys
            sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
            
            from pythonSdk.blockchain.agentIntegration import BlockchainAgentIntegration
            
            # Initialize blockchain integration if not already done
            if not hasattr(self, '_blockchain_integration'):
                self._blockchain_integration = BlockchainAgentIntegration(
                    agent_name=self.agent.agent_id,
                    agent_endpoint=self.agent.base_url,
                    capabilities=["calculation_testing", "evaluation"]
                )
                
                # Initialize the blockchain integration
                await self._blockchain_integration.initialize()
                
                # Register message handler for responses
                self._blockchain_integration.register_message_handler(
                    "calculation_response", 
                    self._handle_blockchain_response
                )
            
            # Find calculation agent on blockchain
            calc_agents = await self._blockchain_integration.find_agents_by_capability("calculation")
            if not calc_agents:
                return {"error": "No calculation agent found on blockchain"}
            
            calc_agent_address = calc_agents[0]['address']
            
            # Send message through blockchain
            message_id = await self._blockchain_integration.send_message(
                to_agent_address=calc_agent_address,
                content=json.dumps(message.dict()),
                message_type="calculation_request"
            )
            
            if not message_id:
                return {"error": "Failed to send message through blockchain"}
            
            # Wait for response (with timeout)
            try:
                result = await asyncio.wait_for(
                    self._wait_for_response(message_id),
                    timeout=30.0
                )
                return result
            except asyncio.TimeoutError:
                return {"error": "Timeout waiting for CalculationAgent response"}
                    
        except Exception as e:
            logger.error(f"Failed to send message to CalculationAgent via blockchain: {e}")
            return {"error": str(e)}
        
    async def dispatch_test_question(self, question: TestQuestion) -> Dict[str, Any]:
        """
        Dispatch a test question to CalculationAgent via A2A protocol
        """
        try:
            # Create A2A message for CalculationAgent with proper format
            message = A2AMessage(
                messageId=f"test_{question.question_id}",
                role=MessageRole.USER,
                parts=[
                    MessagePart(
                        kind="text",
                        text=f"Please calculate: {question.question}"
                    ),
                    MessagePart(
                        kind="data",
                        data={
                            "method": "intelligent_dispatch",
                            "request": question.question,
                            "metadata": {
                                "category": question.category,
                                "difficulty": question.difficulty,
                                "test_mode": True,
                                "require_explanation": True
                            },
                            "sender": self.agent.agent_id,
                            "receiver": "calculation_agent"
                        }
                    )
                ],
                taskId=question.question_id,
                contextId=f"test_context_{question.question_id}"
            )
            
            # Sign the message with BDC smart contract
            if hasattr(self.agent, 'trust_identity') and self.agent.trust_identity:
                try:
                    import sys
                    sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')
                    from trustSystem.smartContractTrust import sign_a2a_message
                    signature = sign_a2a_message(
                        self.agent.agent_id,
                        message.dict()
                    )
                    message.signature = signature.get("signature", "")
                except ImportError:
                    logger.warning("Trust system not available, sending unsigned message")
            
            # Send to CalculationAgent via A2A protocol
            logger.info(f"Dispatching test question {question.question_id} to CalculationAgent")
            
            # Send via A2A protocol using agent's communication pattern
            response = await self._call_calculation_agent(message)
            
            if response and response.get("success"):
                result = response.get("data", {})
                
                # Extract calculation result
                calc_result = CalculationResult(
                    answer=result.get("result", {}).get("answer"),
                    methodology=result.get("result", {}).get("methodology", ""),
                    steps=result.get("result", {}).get("steps", []),
                    confidence=result.get("result", {}).get("confidence", 0.0),
                    computation_time=result.get("computation_time", 0.0)
                )
                
                # Store raw response for evaluation
                await self._store_calculation_response(question.question_id, calc_result)
                
                return {
                    "status": "success",
                    "question_id": question.question_id,
                    "calculation_result": calc_result.dict()
                }
            else:
                error_msg = response.get("error", "Failed to get response from CalculationAgent") if response else "No response received"
                logger.error(f"Failed to dispatch question: {error_msg}")
                return {
                    "status": "error",
                    "question_id": question.question_id,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Error dispatching test question: {str(e)}")
            return {
                "status": "error",
                "question_id": question.question_id,
                "error": str(e)
            }
    
    async def evaluate_calculation_answer(
        self, 
        question: TestQuestion, 
        calc_result: CalculationResult,
        expected_answer: Optional[Any] = None
    ) -> EvaluationScore:
        """
        Use GrokClient to evaluate the answer from CalculationAgent
        """
        try:
            evaluation_prompt = f"""
            Evaluate the following calculation result:
            
            Question: {question.question}
            Category: {question.category}
            Difficulty: {question.difficulty}
            
            Provided Answer: {calc_result.answer}
            Methodology: {calc_result.methodology}
            Steps: {json.dumps(calc_result.steps, indent=2)}
            
            Expected Answer: {expected_answer if expected_answer else "Not provided - evaluate correctness based on mathematical principles"}
            Expected Methodology: {question.expected_methodology if question.expected_methodology else "Any valid approach"}
            
            Please evaluate:
            1. Accuracy of the answer (0-100 score)
            2. Quality of methodology explanation (0-100 score)
            3. Clarity of step-by-step explanation (0-100 score)
            4. Overall assessment
            
            Consider:
            - Mathematical correctness
            - Logical flow of steps
            - Clarity of explanation
            - Appropriate methodology for the problem type
            
            Return a JSON response with:
            {{
                "accuracy_score": <0-100>,
                "methodology_score": <0-100>,
                "explanation_score": <0-100>,
                "overall_score": <0-100>,
                "feedback": "<detailed feedback>",
                "passed": <true/false>
            }}
            """
            
            if self.grok_client and GROK_AVAILABLE:
                # Use Grok for intelligent evaluation
                evaluation_data = await self.grok_client.evaluate_calculation(
                    question=question.question,
                    answer=calc_result.answer,
                    methodology=calc_result.methodology,
                    steps=calc_result.steps,
                    expected_answer=expected_answer
                )
            else:
                # Fallback to rule-based evaluation
                evaluation_data = await self._rule_based_evaluation(
                    question, calc_result, expected_answer
                )
            
            # Create evaluation score
            evaluation = EvaluationScore(
                question_id=question.question_id,
                accuracy_score=evaluation_data.get("accuracy_score", 0),
                methodology_score=evaluation_data.get("methodology_score", 0),
                explanation_score=evaluation_data.get("explanation_score", 0),
                overall_score=evaluation_data.get("overall_score", 0),
                feedback=evaluation_data.get("feedback", ""),
                passed=evaluation_data.get("passed", False)
            )
            
            # Update scoreboard
            await self._update_scoreboard(question, evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating calculation answer: {str(e)}")
            return EvaluationScore(
                question_id=question.question_id,
                accuracy_score=0,
                methodology_score=0,
                explanation_score=0,
                overall_score=0,
                feedback=f"Evaluation error: {str(e)}",
                passed=False
            )
    
    async def _rule_based_evaluation(
        self, 
        question: TestQuestion, 
        calc_result: CalculationResult,
        expected_answer: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Fallback rule-based evaluation when Grok is not available
        """
        scores = {
            "accuracy_score": 0,
            "methodology_score": 0,
            "explanation_score": 0,
            "overall_score": 0,
            "feedback": "",
            "passed": False
        }
        
        # Check accuracy if expected answer provided
        if expected_answer is not None:
            try:
                if isinstance(expected_answer, (int, float)):
                    # Numerical comparison with tolerance
                    if abs(float(calc_result.answer) - float(expected_answer)) <= question.tolerance:
                        scores["accuracy_score"] = 100
                    else:
                        scores["accuracy_score"] = max(0, 100 - abs(float(calc_result.answer) - float(expected_answer)) * 10)
                else:
                    # String/symbolic comparison
                    scores["accuracy_score"] = 100 if str(calc_result.answer) == str(expected_answer) else 0
            except:
                scores["accuracy_score"] = 0
        else:
            # No expected answer - give benefit of doubt
            scores["accuracy_score"] = 80 if calc_result.confidence > 0.8 else 60
        
        # Evaluate methodology
        if calc_result.methodology:
            methodology_length = len(calc_result.methodology)
            if methodology_length > 50:
                scores["methodology_score"] = min(100, 50 + methodology_length / 10)
            else:
                scores["methodology_score"] = methodology_length
        
        # Evaluate explanation steps
        if calc_result.steps:
            step_count = len(calc_result.steps)
            if step_count >= 3:
                scores["explanation_score"] = min(100, 60 + step_count * 10)
            else:
                scores["explanation_score"] = step_count * 20
        
        # Calculate overall score
        scores["overall_score"] = (
            scores["accuracy_score"] * 0.5 +
            scores["methodology_score"] * 0.3 +
            scores["explanation_score"] * 0.2
        )
        
        # Determine if passed
        scores["passed"] = scores["overall_score"] >= 70
        
        # Generate feedback
        feedback_parts = []
        if scores["accuracy_score"] < 70:
            feedback_parts.append("Answer accuracy needs improvement")
        if scores["methodology_score"] < 70:
            feedback_parts.append("Methodology explanation could be more detailed")
        if scores["explanation_score"] < 70:
            feedback_parts.append("Step-by-step explanation needs more clarity")
        
        scores["feedback"] = ". ".join(feedback_parts) if feedback_parts else "Good performance overall"
        
        return scores
    
    async def _update_scoreboard(self, question: TestQuestion, evaluation: EvaluationScore):
        """
        Update the scoreboard with evaluation results
        """
        # Update total questions
        self.scoreboard.total_questions += 1
        
        # Update correct answers
        if evaluation.passed:
            self.scoreboard.correct_answers += 1
        
        # Update accuracy rate
        self.scoreboard.accuracy_rate = (
            self.scoreboard.correct_answers / self.scoreboard.total_questions * 100
        )
        
        # Update average scores
        n = self.scoreboard.total_questions
        self.scoreboard.average_methodology_score = (
            (self.scoreboard.average_methodology_score * (n - 1) + evaluation.methodology_score) / n
        )
        self.scoreboard.average_explanation_score = (
            (self.scoreboard.average_explanation_score * (n - 1) + evaluation.explanation_score) / n
        )
        
        # Update category scores
        if question.category not in self.scoreboard.category_scores:
            self.scoreboard.category_scores[question.category] = {
                "total": 0,
                "passed": 0,
                "average_score": 0
            }
        
        cat_scores = self.scoreboard.category_scores[question.category]
        cat_scores["total"] += 1
        if evaluation.passed:
            cat_scores["passed"] += 1
        cat_scores["average_score"] = (
            (cat_scores["average_score"] * (cat_scores["total"] - 1) + evaluation.overall_score) / 
            cat_scores["total"]
        )
        
        # Update difficulty scores
        if question.difficulty not in self.scoreboard.difficulty_scores:
            self.scoreboard.difficulty_scores[question.difficulty] = {
                "total": 0,
                "passed": 0,
                "average_score": 0
            }
        
        diff_scores = self.scoreboard.difficulty_scores[question.difficulty]
        diff_scores["total"] += 1
        if evaluation.passed:
            diff_scores["passed"] += 1
        diff_scores["average_score"] = (
            (diff_scores["average_score"] * (diff_scores["total"] - 1) + evaluation.overall_score) / 
            diff_scores["total"]
        )
        
        # Store scoreboard in Data Manager
        await self._store_scoreboard()
    
    async def get_scoreboard_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive scoreboard report
        """
        report = {
            "summary": {
                "total_questions": self.scoreboard.total_questions,
                "correct_answers": self.scoreboard.correct_answers,
                "accuracy_rate": f"{self.scoreboard.accuracy_rate:.2f}%",
                "average_methodology_score": f"{self.scoreboard.average_methodology_score:.2f}",
                "average_explanation_score": f"{self.scoreboard.average_explanation_score:.2f}",
                "timestamp": self.scoreboard.timestamp.isoformat()
            },
            "by_category": {},
            "by_difficulty": {}
        }
        
        # Add category breakdown
        for category, scores in self.scoreboard.category_scores.items():
            report["by_category"][category] = {
                "total": scores["total"],
                "passed": scores["passed"],
                "pass_rate": f"{(scores['passed'] / scores['total'] * 100):.2f}%" if scores['total'] > 0 else "0%",
                "average_score": f"{scores['average_score']:.2f}"
            }
        
        # Add difficulty breakdown
        for difficulty, scores in self.scoreboard.difficulty_scores.items():
            report["by_difficulty"][difficulty] = {
                "total": scores["total"],
                "passed": scores["passed"],
                "pass_rate": f"{(scores['passed'] / scores['total'] * 100):.2f}%" if scores['total'] > 0 else "0%",
                "average_score": f"{scores['average_score']:.2f}"
            }
        
        return report
    
    async def _store_calculation_response(self, question_id: str, calc_result: CalculationResult):
        """
        Store calculation response in Data Manager
        """
        try:
            # Use agent's data manager integration
            await self.agent._call_data_manager("data_create", {
                "data": calc_result.dict(),
                "storage_backend": "filesystem",
                "service_level": "silver",
                "metadata": {
                    "data_type": "calc_test_result",
                    "question_id": question_id,
                    "timestamp": datetime.now().isoformat(),
                    "agent": self.agent.agent_id
                }
            })
        except Exception as e:
            logger.error(f"Failed to store calculation response: {str(e)}")
    
    async def _store_scoreboard(self):
        """
        Store scoreboard in Data Manager
        """
        try:
            # Use agent's data manager integration
            await self.agent._call_data_manager("data_update", {
                "key": "calc_test_scoreboard_current",
                "data": self.scoreboard.dict(),
                "storage_backend": "filesystem",
                "metadata": {
                    "data_type": "scoreboard",
                    "timestamp": datetime.now().isoformat(),
                    "agent": self.agent.agent_id
                }
            })
        except Exception as e:
            logger.error(f"Failed to store scoreboard: {str(e)}")
    
    async def create_test_suite(self, test_config: Dict[str, Any]) -> List[TestQuestion]:
        """
        Create a test suite based on configuration
        """
        test_questions = []
        
        # Get test templates from Data Manager
        try:
            result = await self.agent._call_data_manager("data_read", {
                "key": "test_templates_calculation",
                "storage_backend": "filesystem"
            })
            
            if result and not result.get("error"):
                templates = result.get("data", {})
            else:
                templates = self._get_default_templates()
        except:
            templates = self._get_default_templates()
        
        # Generate test questions based on config
        for category in test_config.get("categories", ["mathematical"]):
            for difficulty in test_config.get("difficulties", ["easy", "medium", "hard"]):
                count = test_config.get("questions_per_combination", 3)
                
                for i in range(count):
                    question_template = templates.get(category, {}).get(difficulty, [])
                    if question_template:
                        # Select random template
                        import random


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                        template = secrets.choice(question_template)
                        
                        test_questions.append(TestQuestion(
                            question=template["question"],
                            category=category,
                            difficulty=difficulty,
                            expected_methodology=template.get("methodology"),
                            expected_steps=template.get("steps"),
                            tolerance=template.get("tolerance", 0.01)
                        ))
        
        return test_questions
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """
        Get default test templates
        """
        return {
            "mathematical": {
                "easy": [
                    {
                        "question": "Calculate the derivative of x^2 + 3x + 5",
                        "methodology": "Power rule differentiation",
                        "steps": ["Apply power rule", "Differentiate each term", "Combine results"]
                    },
                    {
                        "question": "Solve the equation 2x + 5 = 15",
                        "methodology": "Linear equation solving",
                        "steps": ["Isolate variable", "Simplify", "Calculate result"]
                    }
                ],
                "medium": [
                    {
                        "question": "Find the integral of sin(x) * cos(x)",
                        "methodology": "Substitution method",
                        "steps": ["Identify substitution", "Apply substitution", "Integrate", "Back-substitute"]
                    }
                ],
                "hard": [
                    {
                        "question": "Solve the system: x + y = 10, x^2 + y^2 = 58",
                        "methodology": "Substitution and quadratic solving",
                        "steps": ["Express one variable", "Substitute", "Solve quadratic", "Find both solutions"]
                    }
                ]
            },
            "financial": {
                "easy": [
                    {
                        "question": "Calculate compound interest on $1000 at 5% for 3 years",
                        "methodology": "Compound interest formula",
                        "steps": ["Apply formula A = P(1 + r)^t", "Calculate result"]
                    }
                ],
                "medium": [
                    {
                        "question": "Price a 5-year bond with 4% coupon and 3% yield",
                        "methodology": "Bond pricing formula",
                        "steps": ["Calculate present value of coupons", "Calculate present value of principal", "Sum values"]
                    }
                ]
            }
        }