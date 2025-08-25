
/**
 * Blockchain Event Adapter
 * Provides WebSocket-like interface using blockchain events
 */
class BlockchainEventAdapter {
    constructor(options = {}) {
        this.eventStream = require('./blockchain-event-stream');
        this.connections = new Map();
        this.logger = options.logger || console;
    }

    on(event, handler) {
        if (event === 'connection' || event === 'blockchain-connection') {
            // Handle new connections via blockchain
            this.eventStream.on('new-subscriber', async (subscriberId) => {
                const connection = {
                    id: subscriberId,
                    send: (data) => this.sendToSubscriber(subscriberId, data),
                    on: (evt, fn) => this.eventStream.subscribe(subscriberId, [evt], fn),
                    close: () => this.eventStream.unsubscribe(subscriberId)
                };
                this.connections.set(subscriberId, connection);
                handler(connection);
            });
        }
    }

    async sendToSubscriber(subscriberId, data) {
        try {
            await this.eventStream.publishEvent('message', {
                to: subscriberId,
                data: typeof data === 'string' ? data : JSON.stringify(data)
            });
        } catch (error) {
            this.logger.error('Failed to send via blockchain:', error);
        }
    }
}

// WebSocket compatibility layer
class BlockchainEventServer extends BlockchainEventAdapter {
    constructor(options) {
        super(options);
        this.port = options.port;
        this.path = options.path;
        this.logger.info(`Blockchain event server replacing WebSocket on port ${this.port}`);
    }
}

class BlockchainEventClient extends BlockchainEventAdapter {
    constructor() {
        super();
        this.readyState = 1; // OPEN - for compatibility
    }

    send(data) {
        this.publishEvent('message', { data });
    }

    close() {
        this.eventStream.disconnect();
        this.readyState = 3; // CLOSED
    }
}

module.exports = { BlockchainEventServer, BlockchainEventClient };
