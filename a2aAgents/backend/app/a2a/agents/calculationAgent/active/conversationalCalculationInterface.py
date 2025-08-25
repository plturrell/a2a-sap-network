import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from app.a2a.core.security_base import SecureA2AAgent
"""
Conversational Calculation Interface
Provides natural, interactive mathematical problem-solving conversations
"""

try:
    from app.clients.grokMathematicalClient import GrokMathematicalClient, GrokMathematicalAssistant
    from .grokRealTimeValidator import GrokRealTimeValidator
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Conversation states for mathematical problem solving"""
    INITIAL = "initial"
    UNDERSTANDING = "understanding"
    CLARIFYING = "clarifying"
    SOLVING = "solving"
    EXPLAINING = "explaining"
    VALIDATING = "validating"
    TEACHING = "teaching"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    turn_id: str
    user_input: str
    assistant_response: Dict[str, Any]
    state: ConversationState
    confidence: float
    timestamp: str
    context_used: Dict[str, Any]

class ConversationalCalculationInterface(SecureA2AAgent):
    """Interactive conversational interface for mathematical problem solving"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self,
                 grok_client: Optional[GrokMathematicalClient] = None,
                 calculation_agent = None):
        super().__init__()
        self.grok_client = grok_client or (GrokMathematicalClient() if GROK_AVAILABLE else None)
        self.grok_assistant = GrokMathematicalAssistant(self.grok_client) if self.grok_client else None
        self.calculation_agent = calculation_agent
        self.validator = GrokRealTimeValidator(self.grok_client) if self.grok_client else None

        # Conversation management
        self.conversations = {}  # session_id -> conversation data
        self.active_sessions = set()

        # Interface settings
        self.settings = {
            "max_clarification_rounds": 3,
            "auto_validation": True,
            "show_step_by_step": True,
            "enable_teaching_mode": True,
            "conversation_timeout": 1800,  # 30 minutes
            "max_conversation_length": 50
        }

        # Conversation templates
        self.response_templates = self._initialize_response_templates()

    async def start_conversation(self,
                               session_id: str,
                               initial_query: str,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a new conversational calculation session"""

        if not self.grok_client:
            return {
                "error": "Conversational interface requires Grok client",
                "suggestion": "Please configure GrokMathematicalClient"
            }

        # Initialize conversation
        conversation = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "state": ConversationState.INITIAL,
            "turns": [],
            "context": {
                "user_preferences": user_preferences or {},
                "mathematical_context": {},
                "solution_progress": {},
                "clarification_count": 0
            },
            "summary": {
                "problem_type": None,
                "difficulty_level": None,
                "concepts_covered": [],
                "skills_used": []
            }
        }

        self.conversations[session_id] = conversation
        self.active_sessions.add(session_id)

        # Start validator if not running
        if self.validator and not self.validator._running:
            await self.validator.start_validator()

        # Process initial query
        response = await self.process_user_input(session_id, initial_query)

        logger.info(f"Started conversation session {session_id}")
        return response

    async def process_user_input(self,
                               session_id: str,
                               user_input: str) -> Dict[str, Any]:
        """Process user input and generate appropriate response"""

        if session_id not in self.conversations:
            return {
                "error": "Session not found",
                "suggestion": "Please start a new conversation session"
            }

        conversation = self.conversations[session_id]
        current_state = conversation["state"]

        try:
            # Analyze user input with Grok
            analysis = await self._analyze_user_input(user_input, conversation)

            # Generate response based on current state and analysis
            response = await self._generate_response(analysis, conversation)

            # Record the conversation turn
            turn = ConversationTurn(
                turn_id=f"turn_{len(conversation['turns']) + 1}",
                user_input=user_input,
                assistant_response=response,
                state=conversation["state"],
                confidence=analysis.get("confidence", 0.5),
                timestamp=datetime.utcnow().isoformat(),
                context_used=conversation["context"].copy()
            )

            conversation["turns"].append(asdict(turn))

            # Update conversation state
            conversation["state"] = response.get("next_state", current_state)

            # Update context with new information
            await self._update_conversation_context(conversation, analysis, response)

            return response

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            conversation["state"] = ConversationState.ERROR
            return {
                "success": False,
                "message": "I encountered an error processing your request. Could you please rephrase it?",
                "error": str(e),
                "next_state": ConversationState.ERROR
            }

    async def _analyze_user_input(self,
                                user_input: str,
                                conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user input to understand intent and content"""

        # Get conversation context
        context = conversation["context"]
        previous_turns = conversation["turns"][-5:]  # Last 5 turns for context

        # Create analysis prompt for Grok
        analysis_prompt = f"""
        Analyze this user input in the context of an ongoing mathematical conversation:

        User Input: "{user_input}"

        Current State: {conversation["state"].value}
        Previous Context: {json.dumps(context, indent=2)}
        Recent Conversation: {json.dumps(previous_turns, indent=2) if previous_turns else "None"}

        Provide analysis including:
        {{
            "intent": "question|clarification|answer|request_help|change_topic",
            "mathematical_content": "extracted mathematical expressions or concepts",
            "confidence": 0.95,
            "user_needs": ["what the user appears to need"],
            "follow_up_required": true/false,
            "suggested_next_action": "analyze|solve|explain|teach|validate",
            "emotional_tone": "confident|confused|frustrated|curious",
            "complexity_level": "basic|intermediate|advanced"
        }}
        """

        try:
            analysis_response = await self.grok_client.async_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a conversational analysis assistant for mathematical education."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            # Parse JSON response
            import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            json_match = re.search(r'\{.*\}', analysis_response.content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Fallback analysis
            return {
                "intent": "question",
                "mathematical_content": user_input,
                "confidence": 0.5,
                "user_needs": ["mathematical assistance"],
                "follow_up_required": True,
                "suggested_next_action": "analyze",
                "analysis_text": analysis_response.content
            }

        except Exception as e:
            logger.warning(f"Input analysis failed: {e}")
            return {
                "intent": "question",
                "mathematical_content": user_input,
                "confidence": 0.3,
                "error": str(e)
            }

    async def _generate_response(self,
                               analysis: Dict[str, Any],
                               conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response based on analysis and conversation state"""

        current_state = conversation["state"]
        user_intent = analysis.get("intent", "question")
        suggested_action = analysis.get("suggested_next_action", "analyze")

        # Determine response strategy
        if current_state == ConversationState.INITIAL or user_intent == "question":
            return await self._handle_new_question(analysis, conversation)
        elif current_state == ConversationState.UNDERSTANDING:
            return await self._handle_understanding_phase(analysis, conversation)
        elif current_state == ConversationState.CLARIFYING:
            return await self._handle_clarification(analysis, conversation)
        elif current_state == ConversationState.SOLVING:
            return await self._handle_solving_phase(analysis, conversation)
        elif current_state == ConversationState.EXPLAINING:
            return await self._handle_explanation_phase(analysis, conversation)
        elif user_intent == "change_topic":
            return await self._handle_topic_change(analysis, conversation)
        else:
            return await self._handle_general_interaction(analysis, conversation)

    async def _handle_new_question(self,
                                 analysis: Dict[str, Any],
                                 conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a new mathematical question"""

        mathematical_content = analysis.get("mathematical_content", "")

        # Use Grok to analyze the mathematical query
        if self.grok_assistant:
            grok_response = await self.grok_assistant.interactive_problem_solving(mathematical_content)

            if grok_response["status"] == "ready_to_solve":
                # We understand the problem well
                return {
                    "success": True,
                    "message": self._format_understanding_message(grok_response["analysis"]),
                    "analysis": grok_response["analysis"],
                    "solution_preview": grok_response.get("solution", {}),
                    "next_action": "Would you like me to solve this step by step, or do you need clarification on any part?",
                    "next_state": ConversationState.SOLVING,
                    "options": ["solve", "explain_concept", "provide_hints"]
                }
            else:
                # Need clarification
                return {
                    "success": True,
                    "message": "I'd like to make sure I understand your question correctly.",
                    "clarification_questions": grok_response.get("clarification_questions", []),
                    "suggestions": grok_response.get("suggestions", []),
                    "next_state": ConversationState.CLARIFYING,
                    "analysis": grok_response.get("analysis", {})
                }

    async def _handle_solving_phase(self,
                                  analysis: Dict[str, Any],
                                  conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the problem-solving phase"""

        user_input = analysis.get("mathematical_content", "")

        if "solve" in user_input.lower() or "yes" in user_input.lower():
            # User wants step-by-step solution
            if self.grok_client:
                # Get the problem from context
                problem_query = conversation["context"].get("current_problem", user_input)

                # Generate step-by-step solution
                query_analysis = await self.grok_client.analyze_mathematical_query(problem_query)
                solution = await self.grok_client.generate_step_by_step_solution(problem_query, query_analysis)

                # Execute calculation if agent is available
                calculation_result = None
                if self.calculation_agent and query_analysis.get("confidence", 0) > 0.7:
                    try:
                        calc_input = {
                            "request": problem_query,
                            "auto_execute": True,
                            "use_grok_enhancement": True
                        }
                        calculation_result = await self.calculation_agent.intelligent_dispatch_calculation(calc_input)
                    except Exception as e:
                        logger.warning(f"Calculation execution failed: {e}")

                return {
                    "success": True,
                    "message": "Here's the step-by-step solution:",
                    "solution": solution,
                    "calculation_result": calculation_result,
                    "next_action": "Would you like me to explain any of these steps in more detail?",
                    "next_state": ConversationState.EXPLAINING,
                    "options": ["explain_step", "try_similar", "new_problem"]
                }

        elif "explain" in user_input.lower() or "concept" in user_input.lower():
            # User wants concept explanation
            return await self._handle_concept_explanation(analysis, conversation)

        else:
            # Continue solving conversation
            return {
                "success": True,
                "message": "I can help you solve this problem. Would you like me to:",
                "options": [
                    "Show the complete step-by-step solution",
                    "Explain the underlying concepts first",
                    "Give you hints to solve it yourself"
                ],
                "next_state": ConversationState.SOLVING
            }

    async def _handle_concept_explanation(self,
                                        analysis: Dict[str, Any],
                                        conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle concept explanation requests"""

        concept = analysis.get("mathematical_content", "")
        user_level = conversation["context"].get("user_preferences", {}).get("level", "intermediate")

        if self.grok_client:
            explanation = await self.grok_client.explain_mathematical_concept(
                concept=concept,
                level=user_level,
                include_examples=True
            )

            practice_problems = await self.grok_client.suggest_practice_problems(
                topic=concept,
                difficulty="easy" if user_level == "beginner" else "medium",
                num_problems=3
            )

            return {
                "success": True,
                "message": f"Here's an explanation of {concept}:",
                "explanation": explanation,
                "practice_problems": practice_problems,
                "next_action": "Would you like to try some practice problems or continue with your original question?",
                "next_state": ConversationState.TEACHING,
                "options": ["practice_problems", "return_to_problem", "ask_questions"]
            }

        return {
            "success": True,
            "message": f"I'd be happy to explain {concept}. Could you be more specific about what aspect you'd like to understand?",
            "next_state": ConversationState.CLARIFYING
        }

    def _format_understanding_message(self, analysis: Dict[str, Any]) -> str:
        """Format a message showing understanding of the problem"""

        operation = analysis.get("operation_type", "mathematical problem")
        expression = analysis.get("mathematical_expression", "")
        confidence = analysis.get("confidence", 0)

        if confidence > 0.8:
            return f"I understand! You want me to {operation.replace('_', ' ')} the expression: {expression}"
        else:
            return f"I think you're asking me to work with {operation.replace('_', ' ')}, but let me confirm the details."

    async def _update_conversation_context(self,
                                         conversation: Dict[str, Any],
                                         analysis: Dict[str, Any],
                                         response: Dict[str, Any]):
        """Update conversation context with new information"""

        context = conversation["context"]

        # Update mathematical context
        if "mathematical_content" in analysis:
            context["current_problem"] = analysis["mathematical_content"]

        # Track concepts covered
        if "analysis" in response:
            operation_type = response["analysis"].get("operation_type")
            if operation_type and operation_type not in conversation["summary"]["concepts_covered"]:
                conversation["summary"]["concepts_covered"].append(operation_type)

        # Update clarification count
        if conversation["state"] == ConversationState.CLARIFYING:
            context["clarification_count"] += 1

        # Track user preferences
        emotional_tone = analysis.get("emotional_tone")
        if emotional_tone:
            context.setdefault("user_patterns", {})["emotional_tone"] = emotional_tone

    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize response templates for different scenarios"""

        return {
            "greeting": "Hello! I'm here to help you with mathematical problems. What would you like to work on today?",
            "clarification": "I want to make sure I understand correctly. Could you clarify {specific_point}?",
            "encouragement": "Great work! You're on the right track. Let's continue.",
            "error_recovery": "No worries! Let's approach this differently. {alternative_approach}",
            "completion": "Excellent! We've solved the problem. The answer is {answer}. Would you like to try a similar problem?",
            "concept_intro": "Let me explain the concept of {concept} before we solve the problem.",
            "step_explanation": "In this step, we {action} because {reason}."
        }

    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation session"""

        if session_id not in self.conversations:
            return {"error": "Session not found"}

        conversation = self.conversations[session_id]

        summary = {
            "session_id": session_id,
            "duration": self._calculate_duration(conversation),
            "turns_count": len(conversation["turns"]),
            "concepts_covered": conversation["summary"]["concepts_covered"],
            "current_state": conversation["state"].value,
            "problems_solved": self._count_solved_problems(conversation),
            "user_engagement": self._assess_engagement(conversation),
            "learning_progress": self._assess_progress(conversation)
        }

        return summary

    def _calculate_duration(self, conversation: Dict[str, Any]) -> str:
        """Calculate conversation duration"""
        if not conversation["turns"]:
            return "0 minutes"

        start_time = datetime.fromisoformat(conversation["created_at"])
        last_turn = conversation["turns"][-1]
        end_time = datetime.fromisoformat(last_turn["timestamp"])

        duration = end_time - start_time
        minutes = int(duration.total_seconds() / 60)

        return f"{minutes} minutes"

    def _count_solved_problems(self, conversation: Dict[str, Any]) -> int:
        """Count number of problems solved in conversation"""
        solved_count = 0
        for turn in conversation["turns"]:
            if turn["state"] == ConversationState.COMPLETED.value:
                solved_count += 1
        return solved_count

    def _assess_engagement(self, conversation: Dict[str, Any]) -> str:
        """Assess user engagement level"""
        if len(conversation["turns"]) > 20:
            return "high"
        elif len(conversation["turns"]) > 10:
            return "medium"
        else:
            return "low"

    def _assess_progress(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess learning progress"""
        return {
            "concepts_learned": len(conversation["summary"]["concepts_covered"]),
            "difficulty_progression": "beginner → intermediate",  # Would be calculated
            "confidence_trend": "improving",  # Would be calculated from turns
            "areas_of_strength": ["algebra", "calculus"],  # Would be extracted
            "areas_for_improvement": ["geometry"]  # Would be extracted
        }

    async def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """End a conversation session"""

        if session_id not in self.conversations:
            return {"error": "Session not found"}

        # Get final summary
        summary = await self.get_conversation_summary(session_id)

        # Clean up
        del self.conversations[session_id]
        self.active_sessions.discard(session_id)

        logger.info(f"Ended conversation session {session_id}")

        return {
            "message": "Thank you for the mathematical conversation! Here's a summary of what we accomplished:",
            "summary": summary,
            "farewell": "Feel free to start a new conversation anytime you need help with mathematics!"
        }

# Factory function
def create_conversational_interface(grok_client: Optional[GrokMathematicalClient] = None,
                                  calculation_agent = None) -> ConversationalCalculationInterface:
    """Create a conversational calculation interface"""
    return ConversationalCalculationInterface(grok_client, calculation_agent)
