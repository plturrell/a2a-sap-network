# Enhanced Queue Management Integration Guide

This guide explains how to integrate the new Enhanced Queue Management system into A2A agents.

## Overview

The Enhanced Queue Manager provides unified queue management with advanced features:
- Multiple queue types (Priority, FIFO, LIFO, Delayed, Broadcast, Partitioned)
- Built-in retry mechanisms with exponential backoff
- Dead letter queue support
- Message TTL and expiration
- Comprehensive metrics and monitoring
- Event-driven architecture
- Pluggable processors

## Quick Start

### 1. Import Required Components

```python
from app.a2a.core.enhancedQueueManager import (
    queue_manager, EnhancedQueue, QueueType, QueuePriority,
    QueueProcessor, QueueMessage
)
from app.a2a.core.queueProcessors import (
    BatchProcessor, HTTPProcessor, TransformProcessor,
    AnalyticsProcessor, RouterProcessor, ValidationProcessor
)
```

### 2. Create a Queue

```python
# In your agent's __init__ method
async def initialize(self):
    # Create a priority queue for incoming messages
    self.message_queue = queue_manager.create_queue(
        name=f"{self.agent_id}_messages",
        queue_type=QueueType.PRIORITY,
        max_size=1000,
        ttl=timedelta(hours=1),
        group="agent_queues"
    )
    
    # Create a dead letter queue for failed messages
    self.dlq = queue_manager.create_queue(
        name=f"{self.agent_id}_dlq",
        queue_type=QueueType.FIFO,
        group="dead_letter_queues"
    )
    
    # Link dead letter queue
    self.message_queue.dead_letter_queue = self.dlq
```

### 3. Add Processors

```python
# Add batch processor for efficiency
batch_processor = BatchProcessor(
    batch_size=10,
    batch_timeout=5.0,
    processor_func=self.process_message_batch
)
self.message_queue.add_processor(batch_processor)

# Add validation processor
validator = ValidationProcessor(
    schema={
        'skill': str,
        'parameters': dict,
        'sender': str
    },
    fix_func=self.fix_invalid_message
)
self.message_queue.add_processor(validator)

# Add analytics processor
analytics = AnalyticsProcessor(self.queue_metrics)
self.message_queue.add_processor(analytics)
```

### 4. Enqueue Messages

```python
# Basic enqueue
message_id = await self.message_queue.enqueue(
    payload={'skill': 'calculate', 'parameters': {'expression': '2+2'}},
    priority=QueuePriority.MEDIUM
)

# Delayed message
await self.message_queue.enqueue(
    payload={'task': 'scheduled_cleanup'},
    priority=QueuePriority.LOW,
    delay=timedelta(minutes=30)
)

# Partitioned message (for load balancing)
await self.message_queue.enqueue(
    payload={'data': large_dataset},
    partition_key=f"partition_{hash(user_id) % 10}"
)
```

### 5. Handle Queue Events

```python
# Register event handlers
self.message_queue.on('message_enqueued', self.on_message_received)
self.message_queue.on('message_processed', self.on_message_completed)
self.message_queue.on('message_failed', self.on_message_failed)
self.message_queue.on('message_dead_lettered', self.on_message_dlq)

async def on_message_failed(self, message: QueueMessage):
    logger.warning(f"Message {message.id} failed, retry {message.retry_count}")
    # Send alert if critical
    if message.priority == QueuePriority.CRITICAL:
        await self.send_alert(f"Critical message failed: {message.id}")
```

## Advanced Usage

### Custom Processor Implementation

```python
class AgentMessageProcessor(QueueProcessor):
    def __init__(self, agent: A2AAgentBase):
        self.agent = agent
        
    async def process(self, message: QueueMessage) -> bool:
        try:
            # Extract A2A message
            a2a_msg = A2AMessage(**message.payload)
            
            # Process with agent
            result = await self.agent.process_message(
                a2a_msg,
                message.correlation_id or str(uuid4())
            )
            
            # Check success
            return result.get('success', False)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
```

### Message Routing

```python
# Create router processor
router = RouterProcessor({
    'calculation_rule': lambda msg: 'calc_queue' if 'calculate' in str(msg.payload) else None,
    'validation_rule': lambda msg: 'validation_queue' if msg.priority == QueuePriority.HIGH else None,
    'default_rule': lambda msg: 'general_queue'
})

# Add to main queue
self.intake_queue.add_processor(router)
```

### Broadcast Messaging

```python
# Broadcast to all agent queues
await queue_manager.broadcast(
    payload={'event': 'system_shutdown', 'grace_period': 30},
    group='agent_queues',
    exclude={self.agent_id}
)
```

### Queue Monitoring

```python
# Get queue metrics
metrics = self.message_queue.get_metrics()
logger.info(f"Queue depth: {metrics['queue_depth']}")
logger.info(f"Processing rate: {metrics['messages_processed']}/s")

# Get global metrics
global_metrics = queue_manager.get_global_metrics()
logger.info(f"Total queues: {global_metrics['total_queues']}")
logger.info(f"Total messages: {global_metrics['total_messages_enqueued']}")
```

## Migration from Existing Queue Systems

### From MessageQueue

```python
# Old
from app.a2a.core.messageQueue import MessageQueue, MessagePriority
self.queue = MessageQueue()
await self.queue.push(message, priority)

# New
from app.a2a.core.enhancedQueueManager import queue_manager, QueuePriority
self.queue = queue_manager.create_queue(f"{self.agent_id}_queue")
await self.queue.enqueue(message, priority=QueuePriority.MEDIUM)
```

### From BlockchainQueueManager

```python
# Old
from app.a2a.core.blockchainQueueManager import BlockchainQueueManager
self.bc_queue = BlockchainQueueManager()
await self.bc_queue.create_task(task_data)

# New - with blockchain processor
class BlockchainProcessor(QueueProcessor):
    async def process(self, message: QueueMessage) -> bool:
        # Submit to blockchain
        tx_hash = await self.submit_to_blockchain(message.payload)
        message.metadata['tx_hash'] = tx_hash
        return bool(tx_hash)

self.queue.add_processor(BlockchainProcessor())
```

## Best Practices

1. **Use Appropriate Queue Types**
   - PRIORITY: For messages with varying importance
   - FIFO: For order-sensitive processing
   - DELAYED: For scheduled tasks
   - PARTITIONED: For load distribution

2. **Set Reasonable Limits**
   - Max queue size to prevent memory issues
   - TTL for time-sensitive messages
   - Retry limits to avoid infinite loops

3. **Monitor Queue Health**
   - Track queue depth trends
   - Alert on high failure rates
   - Monitor processing latency

4. **Use Dead Letter Queues**
   - Always configure DLQ for critical queues
   - Regularly review DLQ contents
   - Implement recovery mechanisms

5. **Batch Processing**
   - Use BatchProcessor for high-volume queues
   - Balance batch size vs latency
   - Consider timeout for partial batches

## Example: Complete Agent Integration

```python
class EnhancedAgent(A2AAgentBase):
    async def initialize(self):
        # Create queues
        self.task_queue = queue_manager.create_queue(
            f"{self.agent_id}_tasks",
            QueueType.PRIORITY,
            max_size=500
        )
        
        # Setup processors
        self.task_queue.add_processor(
            ValidationProcessor(
                schema={'task_type': str, 'data': dict}
            )
        )
        
        self.task_queue.add_processor(
            TransformProcessor(
                transform_func=self.enrich_task,
                target_queue=None  # Process in-place
            )
        )
        
        self.task_queue.add_processor(
            ChainProcessor([
                AnalyticsProcessor(self.metrics),
                AgentMessageProcessor(self)
            ])
        )
        
        # Start processing
        asyncio.create_task(self.task_queue.start_processing())
    
    async def enrich_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to task"""
        return {
            **task,
            'enriched_at': datetime.utcnow().isoformat(),
            'agent_version': self.version
        }
```

## Performance Considerations

1. **Queue Sizing**: Monitor memory usage with large queues
2. **Processor Efficiency**: Keep processors lightweight
3. **Async Processing**: Use async processors for I/O operations
4. **Metrics Overhead**: Disable detailed metrics in production if needed
5. **Event Handlers**: Keep event handlers fast to avoid blocking

## Troubleshooting

1. **Messages Not Processing**
   - Check if processing is enabled: `queue.start_processing()`
   - Verify processors are added
   - Check for processor exceptions

2. **High Memory Usage**
   - Set max_size on queues
   - Implement TTL for messages
   - Monitor queue depth

3. **Messages Going to DLQ**
   - Check processor return values
   - Review retry configuration
   - Examine processor logs

4. **Performance Issues**
   - Use batch processing
   - Optimize processor logic
   - Consider partitioned queues