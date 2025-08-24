# Message Template Standardization Guide

This guide helps migrate agents to use standardized message templates from `app.a2a.sdk.messageTemplates`.

## Migration Steps

### 1. Import Message Templates

Add the following import to your agent:

```python
from app.a2a.sdk.messageTemplates import (
    MessageTemplate, MessageStatus, ReasoningMessageTemplate
)
```

### 2. Replace Custom Error Responses

**Before:**
```python
def _create_error_response(self, message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": message,
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": self.agent_id
    }
```

**After:**
```python
def _create_error_response(self, message: str) -> Dict[str, Any]:
    return MessageTemplate.create_error_response(
        error_message=message,
        agent_id=self.agent_id
    )
```

### 3. Replace Custom Success Responses

**Before:**
```python
def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": self.agent_id
    }
```

**After:**
```python
def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
    return MessageTemplate.create_response(
        request_id=str(datetime.utcnow().timestamp()),
        status=MessageStatus.SUCCESS,
        result=data,
        agent_id=self.agent_id
    )
```

### 4. Replace Inter-Agent Messages

**Before:**
```python
message_content = {
    "skill": skill,
    "parameters": parameters,
    "timestamp": datetime.utcnow().isoformat(),
    "sender": self.agent_id
}
```

**After:**
```python
message_content = MessageTemplate.create_request(
    skill=skill,
    parameters=parameters,
    sender_id=self.agent_id
)
```

### 5. Replace Task Updates

**Before:**
```python
task_update = {
    "task_id": task_id,
    "status": "completed",
    "result": result,
    "message": "Task completed successfully"
}
```

**After:**
```python
task_update = MessageTemplate.create_task_update(
    task_id=task_id,
    status=MessageStatus.COMPLETED,
    result=result,
    message="Task completed successfully"
)
```

## Message Status Values

Use the `MessageStatus` enum for consistent status values:

- `MessageStatus.SUCCESS` - Operation completed successfully
- `MessageStatus.ERROR` - Operation failed with error
- `MessageStatus.PENDING` - Operation is queued
- `MessageStatus.IN_PROGRESS` - Operation is running
- `MessageStatus.COMPLETED` - Operation finished
- `MessageStatus.FAILED` - Operation failed

## For Reasoning Agents

Use `ReasoningMessageTemplate` for reasoning-specific messages:

```python
# Create reasoning request
request = ReasoningMessageTemplate.create_reasoning_request(
    question="What is the impact of AI on healthcare?",
    sender_id=self.agent_id,
    architecture="hierarchical",
    context={"domain": "healthcare"}
)

# Create sub-task assignment
assignment = ReasoningMessageTemplate.create_sub_task_assignment(
    task_id=task_id,
    agent_role="evidence_retriever",
    task_type="search",
    parameters={"query": "AI healthcare applications"}
)

# Create debate message
debate_msg = ReasoningMessageTemplate.create_debate_message(
    debate_id=debate_id,
    round=1,
    position="supporting",
    arguments=[{"claim": "AI improves diagnosis", "evidence": "..."}],
    agent_id=self.agent_id,
    confidence=0.85
)
```

## Validation

Use `MessageValidator` to ensure message format compliance:

```python
from app.a2a.sdk.messageTemplates import MessageValidator

# Validate request format
if MessageValidator.validate_request(message):
    # Process valid request
    pass
```

## Benefits

1. **Consistency**: All agents use the same message format
2. **Maintainability**: Changes to message format only need to be made in one place
3. **Type Safety**: Enums prevent typos in status values
4. **Extensibility**: Easy to add new message types
5. **Validation**: Built-in validation ensures format compliance

## Example Migration

See the following agents for reference implementations:
- `calculationAgentSdk.py` - Basic request/response patterns
- `reasoningAgent.py` - Advanced inter-agent communication
- `agentManagerAgent.py` - Task management patterns