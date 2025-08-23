"""
Standardized message templates for A2A agent communication
Provides consistent message formatting across all agents
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class MessageStatus(str, Enum):
    """Standard status values for messages"""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageTemplate:
    """Standard message templates for A2A communication"""
    
    @staticmethod
    def create_request(
        skill: str,
        parameters: Dict[str, Any],
        sender_id: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized request message"""
        return {
            "request_id": request_id or str(datetime.utcnow().timestamp()),
            "skill": skill,
            "parameters": parameters,
            "sender": sender_id,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
    
    @staticmethod
    def create_response(
        request_id: str,
        status: MessageStatus,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized response message"""
        response = {
            "request_id": request_id,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result is not None:
            response["result"] = result
        
        if error:
            response["error"] = error
        
        if agent_id:
            response["agent_id"] = agent_id
            
        if metadata:
            response["metadata"] = metadata
            
        return response
    
    @staticmethod
    def create_task_update(
        task_id: str,
        status: MessageStatus,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a task status update message"""
        update = {
            "task_id": task_id,
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if progress is not None:
            update["progress"] = progress
            
        if message:
            update["message"] = message
            
        if result is not None:
            update["result"] = result
            
        if error:
            update["error"] = error
            
        return update
    
    @staticmethod
    def create_error_response(
        error_message: str,
        error_code: Optional[str] = None,
        request_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized error response"""
        error_response = {
            "status": MessageStatus.ERROR.value,
            "error": {
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if error_code:
            error_response["error"]["code"] = error_code
            
        if request_id:
            error_response["request_id"] = request_id
            
        if agent_id:
            error_response["agent_id"] = agent_id
            
        if details:
            error_response["error"]["details"] = details
            
        return error_response
    
    @staticmethod
    def create_inter_agent_message(
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a message for inter-agent communication"""
        return {
            "message_id": str(datetime.utcnow().timestamp()),
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "payload": payload,
            "correlation_id": correlation_id,
            "reply_to": reply_to,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def create_broadcast_message(
        from_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        target_roles: Optional[List[str]] = None,
        target_capabilities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a broadcast message for multiple agents"""
        message = {
            "message_id": str(datetime.utcnow().timestamp()),
            "from": from_agent,
            "type": message_type,
            "broadcast": True,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if target_roles:
            message["target_roles"] = target_roles
            
        if target_capabilities:
            message["target_capabilities"] = target_capabilities
            
        return message


class ReasoningMessageTemplate(MessageTemplate):
    """Extended templates specific to reasoning operations"""
    
    @staticmethod
    def create_reasoning_request(
        question: str,
        sender_id: str,
        architecture: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a reasoning request message"""
        return MessageTemplate.create_request(
            skill="multi_agent_reasoning",
            parameters={
                "question": question,
                "architecture": architecture,
                "context": context or {},
                "constraints": constraints or {}
            },
            sender_id=sender_id
        )
    
    @staticmethod
    def create_sub_task_assignment(
        task_id: str,
        agent_role: str,
        task_type: str,
        parameters: Dict[str, Any],
        deadline: Optional[str] = None,
        priority: float = 1.0
    ) -> Dict[str, Any]:
        """Create a sub-task assignment message"""
        return {
            "task_id": task_id,
            "assignment": {
                "role": agent_role,
                "type": task_type,
                "parameters": parameters,
                "priority": priority
            },
            "deadline": deadline,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def create_evidence_request(
        query: str,
        context: Dict[str, Any],
        relevance_threshold: float = 0.7,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Create an evidence retrieval request"""
        return {
            "operation": "retrieve_evidence",
            "query": query,
            "context": context,
            "filters": {
                "relevance_threshold": relevance_threshold,
                "max_results": max_results
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def create_debate_message(
        debate_id: str,
        round: int,
        position: str,
        arguments: List[Dict[str, Any]],
        agent_id: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Create a debate contribution message"""
        return {
            "debate_id": debate_id,
            "round": round,
            "position": position,
            "arguments": arguments,
            "agent_id": agent_id,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }


class MessageValidator:
    """Validate messages against expected formats"""
    
    @staticmethod
    def validate_request(message: Dict[str, Any]) -> bool:
        """Validate request message format"""
        required_fields = ["skill", "parameters", "sender", "timestamp"]
        return all(field in message for field in required_fields)
    
    @staticmethod
    def validate_response(message: Dict[str, Any]) -> bool:
        """Validate response message format"""
        required_fields = ["status", "timestamp"]
        return all(field in message for field in required_fields)
    
    @staticmethod
    def validate_task_update(message: Dict[str, Any]) -> bool:
        """Validate task update format"""
        required_fields = ["task_id", "status", "timestamp"]
        return all(field in message for field in required_fields)