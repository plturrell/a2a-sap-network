const amqp = require('amqplib');
const EventEmitter = require('events');
const crypto = require('crypto');

/**
 * Production-ready Message Queue Service
 * Implements reliable message delivery with RabbitMQ
 * Includes retry logic, dead letter queues, and monitoring
 */
class MessageQueueService extends EventEmitter {
    constructor() {
        super();
        
        // RabbitMQ configuration
        this.config = {
            url: process.env.RABBITMQ_URL || 'amqp://localhost',
            reconnectDelay: 5000,
            prefetchCount: parseInt(process.env.MQ_PREFETCH || '10'),
            messageExpiry: parseInt(process.env.MQ_MESSAGE_EXPIRY || '3600000'), // 1 hour
            maxRetries: parseInt(process.env.MQ_MAX_RETRIES || '3')
        };

        // Queue definitions
        this.queues = {
            notifications: {
                name: 'a2a.notifications',
                durable: true,
                deadLetter: 'a2a.notifications.dlq'
            },
            chat: {
                name: 'a2a.chat.messages',
                durable: true,
                deadLetter: 'a2a.chat.dlq',
                priority: true
            },
            agent: {
                name: 'a2a.agent.messages',
                durable: true,
                deadLetter: 'a2a.agent.dlq'
            }
        };

        // Exchanges
        this.exchanges = {
            direct: 'a2a.direct',
            topic: 'a2a.topic',
            fanout: 'a2a.fanout',
            delayed: 'a2a.delayed'
        };

        // Connection state
        this.connection = null;
        this.channel = null;
        this.consumers = new Map();
        this.publishers = new Map();
        
        // Monitoring
        this.stats = {
            messagesPublished: 0,
            messagesConsumed: 0,
            messagesRetried: 0,
            messagesFailed: 0,
            connectionRetries: 0
        };

        // Initialize connection
        this.connect();
    }

    /**
     * Connect to RabbitMQ with retry logic
     */
    async connect() {
        try {
            console.log('🐰 Connecting to RabbitMQ...');
            
            this.connection = await amqp.connect(this.config.url);
            
            // Handle connection events
            this.connection.on('error', (err) => {
                console.error('RabbitMQ connection error:', err);
                this.handleConnectionError();
            });
            
            this.connection.on('close', () => {
                console.log('RabbitMQ connection closed');
                this.handleConnectionClose();
            });

            // Create channel
            this.channel = await this.connection.createChannel();
            await this.channel.prefetch(this.config.prefetchCount);

            // Setup infrastructure
            await this.setupExchanges();
            await this.setupQueues();

            console.log('✅ Connected to RabbitMQ successfully');
            this.emit('connected');

        } catch (error) {
            console.error('Failed to connect to RabbitMQ:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Setup exchanges
     */
    async setupExchanges() {
        for (const [type, name] of Object.entries(this.exchanges)) {
            if (type === 'delayed') {
                // Delayed message exchange (requires plugin)
                await this.channel.assertExchange(name, 'x-delayed-message', {
                    durable: true,
                    arguments: {
                        'x-delayed-type': 'direct'
                    }
                });
            } else {
                await this.channel.assertExchange(name, type, {
                    durable: true
                });
            }
        }
    }

    /**
     * Setup queues with dead letter support
     */
    async setupQueues() {
        for (const [key, config] of Object.entries(this.queues)) {
            // Setup dead letter queue first
            if (config.deadLetter) {
                await this.channel.assertQueue(config.deadLetter, {
                    durable: true,
                    arguments: {
                        'x-message-ttl': 7 * 24 * 60 * 60 * 1000 // 7 days
                    }
                });
            }

            // Setup main queue
            const queueArgs = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': config.deadLetter
            };

            if (config.priority) {
                queueArgs['x-max-priority'] = 10;
            }

            await this.channel.assertQueue(config.name, {
                durable: config.durable,
                arguments: queueArgs
            });

            // Bind to exchanges
            await this.channel.bindQueue(
                config.name,
                this.exchanges.direct,
                config.name
            );
        }
    }

    /**
     * Publish message with reliability guarantees
     */
    async publish(queue, message, options = {}) {
        if (!this.channel) {
            throw new Error('Not connected to message queue');
        }

        const messageId = options.messageId || this.generateMessageId();
        const timestamp = new Date().toISOString();

        const messageData = {
            id: messageId,
            timestamp,
            retryCount: 0,
            data: message,
            metadata: options.metadata || {}
        };

        const publishOptions = {
            persistent: true,
            messageId,
            timestamp: Date.now(),
            contentType: 'application/json',
            headers: {
                'x-retry-count': 0,
                'x-max-retries': this.config.maxRetries,
                ...options.headers
            }
        };

        // Priority support
        if (options.priority !== undefined) {
            publishOptions.priority = Math.min(Math.max(0, options.priority), 10);
        }

        // Delayed message support
        if (options.delay) {
            publishOptions.headers['x-delay'] = options.delay;
            
            return this.channel.publish(
                this.exchanges.delayed,
                queue,
                Buffer.from(JSON.stringify(messageData)),
                publishOptions
            );
        }

        // Direct publish
        const published = this.channel.sendToQueue(
            queue,
            Buffer.from(JSON.stringify(messageData)),
            publishOptions
        );

        if (published) {
            this.stats.messagesPublished++;
            this.emit('message:published', { queue, messageId });
        } else {
            throw new Error('Message could not be published');
        }

        return messageId;
    }

    /**
     * Subscribe to queue with error handling
     */
    async subscribe(queue, handler, options = {}) {
        if (!this.channel) {
            throw new Error('Not connected to message queue');
        }

        const consumerId = options.consumerId || this.generateConsumerId();

        const wrappedHandler = async (msg) => {
            if (!msg) return;

            try {
                const content = JSON.parse(msg.content.toString());
                const headers = msg.properties.headers;

                // Call handler
                await handler(content.data, {
                    messageId: content.id,
                    timestamp: content.timestamp,
                    retryCount: headers['x-retry-count'] || 0,
                    metadata: content.metadata,
                    ack: () => this.channel.ack(msg),
                    nack: (requeue = true) => this.channel.nack(msg, false, requeue),
                    reject: () => this.channel.reject(msg, false)
                });

                // Auto-ack if not explicitly handled
                if (!options.noAck) {
                    this.channel.ack(msg);
                }

                this.stats.messagesConsumed++;
                this.emit('message:consumed', { queue, messageId: content.id });

            } catch (error) {
                console.error(`Error processing message from ${queue}:`, error);
                
                // Handle retry logic
                await this.handleMessageError(msg, error, queue);
            }
        };

        // Start consuming
        const consumerTag = await this.channel.consume(
            queue,
            wrappedHandler,
            {
                noAck: false,
                consumerTag: consumerId,
                ...options
            }
        );

        this.consumers.set(consumerId, {
            queue,
            handler: wrappedHandler,
            tag: consumerTag.consumerTag
        });

        console.log(`📨 Subscribed to queue: ${queue} (${consumerId})`);
        
        return consumerId;
    }

    /**
     * Handle message processing errors with retry
     */
    async handleMessageError(msg, error, queue) {
        const headers = msg.properties.headers;
        const retryCount = (headers['x-retry-count'] || 0) + 1;
        const maxRetries = headers['x-max-retries'] || this.config.maxRetries;

        if (retryCount <= maxRetries) {
            // Retry with exponential backoff
            const delay = Math.min(1000 * Math.pow(2, retryCount), 60000);
            
            console.log(`Retrying message (attempt ${retryCount}/${maxRetries}) after ${delay}ms`);
            
            // Update retry count
            headers['x-retry-count'] = retryCount;
            
            // Republish with delay
            setTimeout(() => {
                this.channel.sendToQueue(
                    queue,
                    msg.content,
                    {
                        ...msg.properties,
                        headers
                    }
                );
                
                // Acknowledge original message
                this.channel.ack(msg);
                this.stats.messagesRetried++;
                
            }, delay);
            
        } else {
            // Max retries exceeded - send to DLQ
            console.error(`Message failed after ${maxRetries} retries, sending to DLQ`);
            
            this.channel.reject(msg, false);
            this.stats.messagesFailed++;
            
            this.emit('message:failed', {
                queue,
                messageId: msg.properties.messageId,
                error: error.message
            });
        }
    }

    /**
     * Unsubscribe consumer
     */
    async unsubscribe(consumerId) {
        const consumer = this.consumers.get(consumerId);
        
        if (consumer && this.channel) {
            await this.channel.cancel(consumer.tag);
            this.consumers.delete(consumerId);
            console.log(`🔕 Unsubscribed consumer: ${consumerId}`);
        }
    }

    /**
     * Handle connection errors
     */
    handleConnectionError() {
        this.stats.connectionRetries++;
        this.scheduleReconnect();
    }

    /**
     * Handle connection close
     */
    handleConnectionClose() {
        this.connection = null;
        this.channel = null;
        this.scheduleReconnect();
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        console.log(`Reconnecting in ${this.config.reconnectDelay}ms...`);
        
        setTimeout(() => {
            this.connect();
        }, this.config.reconnectDelay);
    }

    /**
     * Get queue statistics
     */
    async getQueueStats(queueName) {
        if (!this.channel) {
            throw new Error('Not connected to message queue');
        }

        const queueInfo = await this.channel.checkQueue(queueName);
        
        return {
            messages: queueInfo.messageCount,
            consumers: queueInfo.consumerCount,
            ...this.stats
        };
    }

    /**
     * Purge queue (dangerous!)
     */
    async purgeQueue(queueName) {
        if (!this.channel) {
            throw new Error('Not connected to message queue');
        }

        const result = await this.channel.purgeQueue(queueName);
        console.log(`🗑️ Purged ${result.messageCount} messages from ${queueName}`);
        
        return result.messageCount;
    }

    /**
     * Generate unique message ID
     */
    generateMessageId() {
        return `msg_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
    }

    /**
     * Generate unique consumer ID
     */
    generateConsumerId() {
        return `consumer_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    }

    /**
     * Graceful shutdown
     */
    async shutdown() {
        console.log('📪 Shutting down message queue service...');
        
        // Cancel all consumers
        for (const [consumerId, consumer] of this.consumers) {
            await this.unsubscribe(consumerId);
        }

        // Close channel and connection
        if (this.channel) {
            await this.channel.close();
        }
        
        if (this.connection) {
            await this.connection.close();
        }

        console.log('✅ Message queue service shut down');
    }
}

// Export singleton instance
const messageQueue = new MessageQueueService();

// Graceful shutdown handler
process.on('SIGINT', async () => {
    await messageQueue.shutdown();
    process.exit(0);
});

module.exports = {
    MessageQueueService,
    getInstance: () => messageQueue
};