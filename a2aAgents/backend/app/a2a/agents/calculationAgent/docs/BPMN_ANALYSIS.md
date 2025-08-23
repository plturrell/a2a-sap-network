# Calculation Agent BPMN Analysis

## 1. Internal Skills Structure (Current Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│                    CALCULATION AGENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Entry Points:                                              │
│  ┌─────────────┐        ┌──────────────────┐              │
│  │ Blockchain  │───────►│ Message Router   │              │
│  │  Events     │        └────────┬─────────┘              │
│  └─────────────┘                 │                         │
│                                  ▼                         │
│                    ┌──────────────────────────┐           │
│                    │    Handler Gateway       │           │
│                    └────────────┬─────────────┘           │
│                                 │                          │
│         ┌───────────────────────┼────────────────┐        │
│         ▼                       ▼                ▼        │
│  ┌──────────────┐    ┌───────────────┐  ┌─────────────┐ │
│  │calculation_  │    │test_calculation│  │  default    │ │
│  │  request     │    │   _request    │  │  handler    │ │
│  └──────┬───────┘    └───────┬───────┘  └─────────────┘ │
│         │                     │                           │
│         └──────────┬──────────┘                          │
│                    ▼                                      │
│           ┌─────────────────┐                            │
│           │ @a2a_handler    │                            │
│           │  ("calculate")  │                            │
│           └────────┬────────┘                            │
│                    │                                      │
│                    ▼                                      │
│         ┌──────────────────────┐                         │
│         │   Skills Gateway     │                         │
│         └──────────────────────┘                         │
│                    │                                      │
│    ┌───────────────┴────────────────────────────┐       │
│    ▼               ▼               ▼            ▼       │
│ ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────┐  │
│ │evaluate│   │ solve  │   │calculus│   │intelligent │  │
│ │_calc   │   │equation│   │        │   │ dispatch   │  │
│ └────────┘   └────────┘   └────────┘   └─────┬──────┘  │
│                                               │          │
│                                               ▼          │
│                                    ┌───────────────────┐ │
│                                    │ Enhanced Skills  │ │
│                                    │ (with steps)     │ │
│                                    └───────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 2. Agent Delegation Flow (Via Blockchain)

```
┌──────────────┐                    ┌─────────────────┐
│CalcValidation├───────────────────►│   Blockchain    │
│    Agent     │ test_calculation   │    Network      │
└──────────────┘    _request        └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Calculation     │
                                    │    Agent        │
                                    └────────┬────────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                 ▼
                            ┌──────────────┐  ┌──────────────┐
                            │Process Calc  │  │Store Results │
                            └──────────────┘  └──────┬───────┘
                                                     │
                                                     ▼
┌──────────────┐            ┌─────────────────┐     │
│ Data Manager │◄───────────│   Blockchain    │◄────┘
│    Agent     │            │    Network      │
└──────────────┘            └─────────────────┘
```

## 3. Missing Connections & Issues Found

### ❌ Issue 1: Missing HTTP Client Cleanup
The agent initializes an HTTP client but never uses it (removed blockchain delegation).

**Fix needed:**
```python
async def shutdown(self) -> None:
    """Cleanup agent resources"""
    logger.info("Shutting down Calculation Agent...")
    
    # Cleanup blockchain integration
    if hasattr(self, 'blockchain_integration'):
        await self.blockchain_integration.cleanup()
    
    # Remove unused HTTP client
    # self.http_client is no longer needed
    
    logger.info("Calculation Agent shutdown complete")
```

### ❌ Issue 2: Missing Response Tracking
The `_pending_requests` dict can grow indefinitely if responses never arrive.

**Fix needed:**
```python
# Add periodic cleanup of stale requests
async def _cleanup_pending_requests(self):
    """Clean up stale pending requests"""
    current_time = time.time()
    stale_requests = []
    
    for msg_id, (future, timestamp) in self._pending_requests.items():
        if current_time - timestamp > 60:  # 60 second timeout
            if not future.done():
                future.set_exception(TimeoutError("Request timed out"))
            stale_requests.append(msg_id)
    
    for msg_id in stale_requests:
        self._pending_requests.pop(msg_id, None)
```

### ❌ Issue 3: No Agent Manager or Catalog Manager Usage
The delegation methods exist but are never called by any skill.

**Current state:**
- `_call_agent_manager()` - defined but unused
- `_call_catalog_manager()` - defined but unused

**Fix needed:** Either remove these methods or add skills that use them.

### ❌ Issue 4: Missing Error Propagation in Test Handler
The `_handle_test_calculation_request` catches errors but doesn't send error responses.

**Fix needed:**
```python
except Exception as e:
    logger.error(f"Error handling test calculation request: {e}")
    # Send error response back
    error_response = {
        'in_reply_to': content.get('messageId', blockchain_msg.get('id')),
        'status': 'error',
        'error': str(e),
        'agent_id': self.agent_id,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    signed_error = sign_a2a_message(error_response, self.agent_id)
    
    await self.blockchain_integration.send_message(
        to_agent_address=blockchain_msg.get('from'),
        content=json.dumps(signed_error),
        message_type="test_calculation_response"
    )
```

### ❌ Issue 5: Trust Verification Not Enforced
The agent logs warnings for untrusted senders but still processes their requests.

**Fix needed in _handle_test_calculation_request:**
```python
# Verify sender is trusted CalcValidation agent
sender = blockchain_msg.get('from')
if sender not in self.trusted_agents and 'calc_validation' not in sender:
    logger.warning(f"Received test request from untrusted agent: {sender}")
    return  # Should send rejection response instead
```

### ✅ Correctly Implemented:
1. Blockchain event listener registration
2. Message signing and verification
3. Data Manager delegation for storage
4. Enhanced calculation skills integration
5. Batch processing support

## 4. Required Fixes Summary

1. Remove unused HTTP client
2. Add pending request cleanup mechanism
3. Remove or implement Agent/Catalog Manager usage
4. Add error response handling in test handler
5. Enforce trust verification with proper rejection responses