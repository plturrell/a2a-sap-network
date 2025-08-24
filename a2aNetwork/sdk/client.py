"""
A2A Client for communicating with other agents
"""

# A2A Protocol: Use blockchain messaging instead of httpx
import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

from .types import (
    A2AMessage, MessagePart, MessageRole, AgentCard,
    SkillExecutionRequest, SkillExecutionResponse,
    MessageHandlerRequest, MessageHandlerResponse
)
from ..core.telemetry import trace_async, add_span_attributes, get_trace_context

logger = logging.getLogger(__name__)


class A2AClient:
    """
    Client for communicating with A2A agents
    Provides high-level interface for agent-to-agent communication
    """
    
    def __init__(
        self,
        agent_id: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        registry_url: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.registry_url = registry_url
        
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": f"A2AClient/{agent_id}",
                "Content-Type": "application/json"
            }
        )
        
        # Agent discovery cache
        self._agent_cache: Dict[str, Dict[str, Any]] = {}
        
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
    
    @trace_async("send_message")
    async def send_message(
        self,
        target_agent_url: str,
        message: A2AMessage,
        context_id: str,
        use_rpc: bool = True
    ) -> Dict[str, Any]:
        """
        Send message to another agent
        
        Args:
            target_agent_url: URL of target agent
            message: A2A message to send
            context_id: Context identifier
            use_rpc: Whether to use JSON-RPC endpoint (default) or REST
        
        Returns:
            Response from target agent
        """
        
        add_span_attributes({
            "client.target_agent": target_agent_url,
            "client.message_id": message.messageId,
            "client.use_rpc": use_rpc,
            "client.context_id": context_id
        })
        
        # Add trace context to headers
        headers = get_trace_context()
        headers.update({
            "X-A2A-Agent-Id": self.agent_id,
            "X-A2A-Context-Id": context_id,
            "X-A2A-Message-Id": message.messageId
        })
        
        for attempt in range(self.max_retries):
            try:
                if use_rpc:
                    # Use JSON-RPC endpoint
                    endpoint = f"{target_agent_url}/rpc"
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "agent.processMessage",
                        "params": {
                            "message": message.model_dump(),
                            "contextId": context_id
                        },
                        "id": message.messageId
                    }
                else:
                    # Use REST endpoint
                    endpoint = f"{target_agent_url}/messages"
                    payload = {
                        "message": message.model_dump(),
                        "contextId": context_id
                    }
                
                response = await self.http_client.post(
                    endpoint,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Handle JSON-RPC response
                if use_rpc:
                    if "error" in result:
                        error = result["error"]
                        raise Exception(f"RPC Error {error['code']}: {error['message']}")
                    return result.get("result", {})
                else:
                    return result
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    @trace_async("execute_skill")
    async def execute_skill(
        self,
        target_agent_url: str,
        skill_name: str,
        input_data: Dict[str, Any],
        context_id: Optional[str] = None
    ) -> SkillExecutionResponse:
        """
        Execute a skill on a remote agent
        
        Args:
            target_agent_url: URL of target agent
            skill_name: Name of skill to execute
            input_data: Input data for skill
            context_id: Optional context identifier
        
        Returns:
            Skill execution response
        """
        
        context_id = context_id or f"skill_{skill_name}_{datetime.utcnow().timestamp()}"
        
        add_span_attributes({
            "client.skill_name": skill_name,
            "client.target_agent": target_agent_url,
            "client.context_id": context_id
        })
        
        # Create message for skill execution
        message = A2AMessage(
            role=MessageRole.AGENT,
            parts=[
                MessagePart(
                    kind="data",
                    data={
                        "method": "execute_skill",
                        "skill_name": skill_name,
                        "input_data": input_data
                    }
                )
            ],
            contextId=context_id
        )
        
        try:
            result = await self.send_message(target_agent_url, message, context_id)
            return SkillExecutionResponse(
                success=result.get("success", False),
                result=result.get("result"),
                error=result.get("error"),
                skill=skill_name,
                timestamp=result.get("timestamp", datetime.utcnow().isoformat())
            )
        except Exception as e:
            return SkillExecutionResponse(
                success=False,
                error=str(e),
                skill=skill_name,
                timestamp=datetime.utcnow().isoformat()
            )
    
    @trace_async("get_agent_card")
    async def get_agent_card(self, agent_url: str) -> AgentCard:
        """
        Get agent card from remote agent
        
        Args:
            agent_url: URL of agent
        
        Returns:
            Agent card
        """
        
        add_span_attributes({
            "client.target_agent": agent_url
        })
        
        response = await self.http_client.get(f"{agent_url}/.well-known/agent.json")
        response.raise_for_status()
        
        return AgentCard(**response.json())
    
    @trace_async("discover_agents")
    async def discover_agents(
        self,
        required_skills: List[str] = None,
        required_capabilities: List[str] = None,
        minimum_trust_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Discover agents in the network
        
        Args:
            required_skills: List of required skills
            required_capabilities: List of required capabilities
            minimum_trust_score: Minimum trust score requirement
        
        Returns:
            List of matching agents
        """
        
        if not self.registry_url:
            raise ValueError("Registry URL not configured for agent discovery")
        
        add_span_attributes({
            "client.required_skills": required_skills or [],
            "client.required_capabilities": required_capabilities or [],
            "client.minimum_trust_score": minimum_trust_score
        })
        
        params = {
            "required_skills": required_skills or [],
            "required_capabilities": required_capabilities or [],
            "minimum_trust_score": minimum_trust_score
        }
        
        response = await self.http_client.post(
            f"{self.registry_url}/api/v1/agents/discover",
            json=params
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("agents", [])
    
    @trace_async("health_check")
    async def health_check(self, agent_url: str) -> Dict[str, Any]:
        """
        Check health of remote agent
        
        Args:
            agent_url: URL of agent
        
        Returns:
            Health status
        """
        
        add_span_attributes({
            "client.target_agent": agent_url
        })
        
        try:
            response = await self.http_client.get(
                f"{agent_url}/health",
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def list_agent_skills(self, agent_url: str) -> List[Dict[str, Any]]:
        """
        List skills available on remote agent
        
        Args:
            agent_url: URL of agent
        
        Returns:
            List of available skills
        """
        
        response = await self.http_client.get(f"{agent_url}/skills")
        response.raise_for_status()
        
        result = response.json()
        return result.get("skills", [])
    
    def create_message(
        self,
        text: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        file: Optional[Dict[str, Any]] = None,
        role: MessageRole = MessageRole.AGENT,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> A2AMessage:
        """
        Create A2A message
        
        Args:
            text: Text content
            data: Data payload
            file: File attachment
            role: Message role
            context_id: Context identifier
            task_id: Task identifier
        
        Returns:
            A2A message
        """
        
        parts = []
        
        if text:
            parts.append(MessagePart(kind="text", text=text))
        
        if data:
            parts.append(MessagePart(kind="data", data=data))
        
        if file:
            parts.append(MessagePart(kind="file", file=file))
        
        return A2AMessage(
            role=role,
            parts=parts,
            contextId=context_id,
            taskId=task_id
        )
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()