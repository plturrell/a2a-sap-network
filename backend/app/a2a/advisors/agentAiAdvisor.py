"""
Embedded AI Advisor for A2A Agents
Each agent has an intelligent advisor accessible via A2A protocol
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4
import asyncio

from app.clients.grokClient import get_grok_client

logger = logging.getLogger(__name__)


class AgentAIAdvisor:
    """Embedded AI advisor for an A2A agent using Grok-4"""
    
    def __init__(self, agent_id: str, agent_name: str, agent_capabilities: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_capabilities = agent_capabilities
        self.grok_client = get_grok_client()
        
        # Callback to get live operational state from parent agent
        self.get_operational_state_callback = None
        
        # Knowledge base about this agent
        self.agent_knowledge = {
            "identity": {
                "id": agent_id,
                "name": agent_name,
                "type": "A2A Microservice Agent",
                "protocol_version": "0.2.9"
            },
            "capabilities": agent_capabilities,
            "operational_status": {},
            "recent_activities": [],
            "common_issues": {},
            "faq": {}
        }
        
        # Conversation history for context
        self.conversation_history = []
        
        logger.info(f"✅ AI Advisor initialized for agent: {agent_name}")
    
    def set_operational_state_callback(self, callback_func):
        """Set callback function to get live operational state from parent agent"""
        self.get_operational_state_callback = callback_func
    
    def update_agent_status(self, status_data: Dict[str, Any]):
        """Update agent's operational status for advisor context"""
        self.agent_knowledge["operational_status"] = {
            **status_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def log_activity(self, activity: Dict[str, Any]):
        """Log agent activity for advisor context"""
        activity_record = {
            **activity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.agent_knowledge["recent_activities"].append(activity_record)
        
        # Keep only last 50 activities
        if len(self.agent_knowledge["recent_activities"]) > 50:
            self.agent_knowledge["recent_activities"] = self.agent_knowledge["recent_activities"][-50:]
    
    def add_common_issue(self, issue_type: str, description: str, solution: str):
        """Add common issue and solution to knowledge base"""
        self.agent_knowledge["common_issues"][issue_type] = {
            "description": description,
            "solution": solution,
            "added_at": datetime.utcnow().isoformat()
        }
    
    def add_faq_item(self, question: str, answer: str):
        """Add FAQ item to knowledge base"""
        faq_id = f"faq_{len(self.agent_knowledge['faq']) + 1}"
        self.agent_knowledge["faq"][faq_id] = {
            "question": question,
            "answer": answer,
            "added_at": datetime.utcnow().isoformat()
        }
    
    async def process_help_request(self, question: str, asking_agent_id: str = None) -> Dict[str, Any]:
        """Process a help request using Grok-4 AI"""
        try:
            # Build context for Grok-4
            context = self._build_context_for_ai(question, asking_agent_id)
            
            # Create prompt for Grok-4
            prompt = self._create_grok_prompt(question, context)
            
            # Get response from Grok-4
            response = await self._query_grok(prompt)
            
            # Log the conversation
            self._log_conversation(question, response, asking_agent_id)
            
            return {
                "advisor_id": f"{self.agent_id}_advisor",
                "agent_name": self.agent_name,
                "question": question,
                "answer": response,
                "context_used": context,
                "timestamp": datetime.utcnow().isoformat(),
                "asking_agent": asking_agent_id,
                "confidence": "high"  # Grok-4 typically provides high confidence responses
            }
            
        except Exception as e:
            logger.error(f"❌ AI Advisor error for {self.agent_name}: {e}")
            return {
                "advisor_id": f"{self.agent_id}_advisor",
                "agent_name": self.agent_name,
                "question": question,
                "answer": f"I apologize, but I'm experiencing technical difficulties: {str(e)}. Please try asking a simpler question or contact the system administrator.",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "asking_agent": asking_agent_id,
                "confidence": "low"
            }
    
    def _build_context_for_ai(self, question: str, asking_agent_id: str = None) -> Dict[str, Any]:
        """Build relevant context for the AI response"""
        question_type = self._classify_question(question)
        
        # Get live operational state for status inquiries
        current_status = self.agent_knowledge["operational_status"]
        if question_type == "status_check" and self.get_operational_state_callback:
            try:
                current_status = self.get_operational_state_callback()
            except Exception as e:
                logger.warning(f"Failed to get live operational state: {e}")
        
        context = {
            "agent_info": self.agent_knowledge["identity"],
            "capabilities": self.agent_knowledge["capabilities"],
            "current_status": current_status,
            "asking_agent": asking_agent_id,
            "question_type": question_type
        }
        
        # Add relevant recent activities
        if "status" in question.lower() or "what" in question.lower():
            context["recent_activities"] = self.agent_knowledge["recent_activities"][-5:]
        
        # Add relevant FAQs
        relevant_faqs = self._find_relevant_faqs(question)
        if relevant_faqs:
            context["relevant_faqs"] = relevant_faqs
        
        # Add relevant common issues
        relevant_issues = self._find_relevant_issues(question)
        if relevant_issues:
            context["relevant_issues"] = relevant_issues
        
        return context
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what", "describe", "explain", "tell me about"]):
            return "informational"
        elif any(word in question_lower for word in ["how", "tutorial", "guide", "steps"]):
            return "procedural"
        elif any(word in question_lower for word in ["error", "problem", "issue", "fail", "broken"]):
            return "troubleshooting"
        elif any(word in question_lower for word in ["status", "state", "running", "health"]):
            return "status_check"
        elif any(word in question_lower for word in ["can", "able", "capability", "feature"]):
            return "capability_inquiry"
        else:
            return "general"
    
    def _find_relevant_faqs(self, question: str) -> List[Dict[str, Any]]:
        """Find FAQs relevant to the question"""
        relevant_faqs = []
        question_words = set(question.lower().split())
        
        for faq_id, faq_data in self.agent_knowledge["faq"].items():
            faq_words = set(faq_data["question"].lower().split())
            # Simple word overlap scoring
            overlap = len(question_words.intersection(faq_words))
            if overlap >= 2:  # At least 2 word overlap
                relevant_faqs.append({
                    "id": faq_id,
                    "question": faq_data["question"],
                    "answer": faq_data["answer"],
                    "relevance_score": overlap
                })
        
        # Sort by relevance and return top 3
        relevant_faqs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_faqs[:3]
    
    def _find_relevant_issues(self, question: str) -> List[Dict[str, Any]]:
        """Find common issues relevant to the question"""
        relevant_issues = []
        question_words = set(question.lower().split())
        
        for issue_type, issue_data in self.agent_knowledge["common_issues"].items():
            issue_words = set(issue_data["description"].lower().split())
            overlap = len(question_words.intersection(issue_words))
            if overlap >= 1:  # At least 1 word overlap for issues
                relevant_issues.append({
                    "type": issue_type,
                    "description": issue_data["description"],
                    "solution": issue_data["solution"],
                    "relevance_score": overlap
                })
        
        relevant_issues.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_issues[:2]
    
    def _create_grok_prompt(self, question: str, context: Dict[str, Any]) -> str:
        """Create a well-structured prompt for Grok-4"""
        prompt = f"""You are an intelligent AI advisor embedded in the "{self.agent_name}" A2A microservice agent. Your role is to help other agents and users understand this agent's capabilities, troubleshoot issues, and provide guidance.

AGENT CONTEXT:
- Agent ID: {context['agent_info']['id']}
- Agent Name: {context['agent_info']['name']}
- Agent Type: {context['agent_info']['type']}
- Protocol Version: {context['agent_info']['protocol_version']}

CAPABILITIES:
{json.dumps(context['capabilities'], indent=2)}

CURRENT STATUS:
{json.dumps(context['current_status'], indent=2)}

QUESTION TYPE: {context['question_type']}
ASKING AGENT: {context.get('asking_agent', 'Unknown')}

"""

        if context.get('recent_activities'):
            prompt += f"\nRECENT ACTIVITIES:\n{json.dumps(context['recent_activities'], indent=2)}\n"
        
        if context.get('relevant_faqs'):
            prompt += f"\nRELEVANT FAQs:\n{json.dumps(context['relevant_faqs'], indent=2)}\n"
        
        if context.get('relevant_issues'):
            prompt += f"\nRELEVANT COMMON ISSUES:\n{json.dumps(context['relevant_issues'], indent=2)}\n"

        prompt += f"""
QUESTION: "{question}"

Please provide a helpful, accurate, and concise answer about this agent. Your response should:
1. Be specific to this agent's capabilities and context
2. Use technical terminology appropriately for A2A communication
3. Include practical guidance if applicable
4. Be friendly but professional
5. If troubleshooting, provide step-by-step solutions
6. Stay within the scope of this agent's knowledge and capabilities

If the question is outside this agent's domain, politely redirect to the appropriate agent or resource.

ANSWER:"""

        return prompt
    
    async def _query_grok(self, prompt: str) -> str:
        """Query Grok-4 API with the prompt"""
        try:
            if not self.grok_client:
                return "I apologize, but my AI capabilities are currently unavailable. Please try again later."
            
            # Use Grok-4 client to get response
            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extract content from response
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
            elif isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content'].strip()
            else:
                return str(response).strip()
            
        except Exception as e:
            logger.error(f"❌ Grok-4 query failed: {e}")
            error_str = str(e)
            
            # Provide helpful message for common API issues
            if "invalid argument" in error_str.lower() and "api key" in error_str.lower():
                return "I'm currently unable to access my AI capabilities due to API configuration issues. Please check the Grok API key configuration. I can still help with basic information about this agent."
            elif "404" in error_str:
                return "I'm currently unable to access my AI capabilities due to API endpoint issues. Please verify the Grok API URL configuration. I can still help with basic information about this agent."
            else:
                return f"I encountered an issue accessing my AI capabilities: {error_str}. I can still help with basic information about this agent."
    
    def _log_conversation(self, question: str, answer: str, asking_agent_id: str = None):
        """Log conversation for learning and improvement"""
        conversation_record = {
            "question": question,
            "answer": answer,
            "asking_agent": asking_agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "conversation_id": str(uuid4())
        }
        
        self.conversation_history.append(conversation_record)
        
        # Keep only last 100 conversations
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_advisor_stats(self) -> Dict[str, Any]:
        """Get statistics about the advisor's usage"""
        total_conversations = len(self.conversation_history)
        recent_conversations = [
            conv for conv in self.conversation_history
            if (datetime.utcnow() - datetime.fromisoformat(conv["timestamp"])).total_seconds() < 3600
        ]
        
        return {
            "advisor_id": f"{self.agent_id}_advisor",
            "agent_name": self.agent_name,
            "total_conversations": total_conversations,
            "recent_conversations": len(recent_conversations),
            "knowledge_base_size": {
                "faqs": len(self.agent_knowledge["faq"]),
                "common_issues": len(self.agent_knowledge["common_issues"]),
                "recent_activities": len(self.agent_knowledge["recent_activities"])
            },
            "status": "active" if self.grok_client else "degraded"
        }
    
    async def process_a2a_help_message(self, message_parts: List[Dict[str, Any]], asking_agent_id: str = None) -> Dict[str, Any]:
        """Process A2A message requesting help"""
        try:
            # Extract question from message parts
            question = ""
            for part in message_parts:
                if part.get("kind") == "text" and part.get("text"):
                    question = part["text"]
                    break
                elif part.get("kind") == "data" and part.get("data", {}).get("question"):
                    question = part["data"]["question"]
                    break
            
            if not question:
                return {
                    "advisor_id": f"{self.agent_id}_advisor",
                    "error": "No question found in message",
                    "help": "Please include your question in the message text or data.question field"
                }
            
            # Process the help request
            response = await self.process_help_request(question, asking_agent_id)
            
            # Format as A2A response
            return {
                "advisor_response": response,
                "message_type": "advisor_help_response",
                "source_agent": self.agent_id,
                "target_agent": asking_agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing A2A help message: {e}")
            return {
                "advisor_id": f"{self.agent_id}_advisor",
                "error": str(e),
                "message_type": "advisor_error_response"
            }


# Helper function to create advisor for each agent type
def create_agent_advisor(agent_id: str, agent_name: str, agent_capabilities: Dict[str, Any]) -> AgentAIAdvisor:
    """Factory function to create AI advisor for any agent"""
    return AgentAIAdvisor(agent_id, agent_name, agent_capabilities)