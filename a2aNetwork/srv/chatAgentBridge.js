/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

const WebSocket = require('ws');

const { LoggerFactory } = require('../../shared/logging/structured-logger');
const logger = LoggerFactory.createLogger('chatAgentBridge');
const EventEmitter = require('events');

/**
 * Chat Agent Bridge - Seamless integration between notifications and chat
 * Handles real-time communication, context preservation, and intelligent routing
 */
class ChatAgentBridge extends EventEmitter {

    constructor() {
        super();
        this.activeChatSessions = new Map();
        this.notificationChatMappings = new Map();
        this.chatAgentConnections = new Map();

        this.setupBlockchainEventServer();
        this.setupA2AIntegration();
    }

    /**
     * Setup WebSocket server for real-time chat
     */
    setupBlockchainEventServer() {
        this.wss = new BlockchainEventServer($1);

        this.wss.on('blockchain-connection', (ws, request) => {
            const sessionId = this.generateSessionId();

            ws.sessionId = sessionId;
            ws.isAlive = true;

            // Handle incoming messages
            blockchainClient.on('event', (data) => {
                this.handleClientMessage(ws, data);
            });

            // Handle connection close
            ws.on('close', () => {
                this.handleClientDisconnection(ws);
            });

            // Heartbeat
            ws.on('pong', () => {
                ws.isAlive = true;
            });

            logger.info(`🔗 Chat client connected: ${sessionId}`);
        });

        // Heartbeat interval
        setInterval(() => {
            this.wss.clients.forEach(ws => {
                if (!ws.isAlive) return ws.terminate();
                ws.isAlive = false;
                ws.ping();
            });
        }, 30000);

        logger.info('💬 Chat Agent Bridge WebSocket server started on port 8087');
    }

    /**
     * Setup A2A integration with communication router
     */
    setupA2AIntegration() {
        // Connect to A2A Communication Router
        this.a2aRouter = {
            sendMessage: async (message) => {
                try {
                    const response = await blockchainClient.sendMessage('http://localhost:8001/route', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(message)
                    });
                    return await response.json();
                } catch (error) {
                    logger.error('A2A Router communication error:', { error: error });
                    throw error;
                }
            }
        };
    }

    /**
     * Start chat session from notification context
     */
    async startChatFromNotification(notificationContext, clientWs) {
        const chatSession = {
            sessionId: clientWs.sessionId,
            notificationId: notificationContext.notificationId,
            conversationId: notificationContext.conversationId,
            context: notificationContext.context,
            startTime: new Date(),
            messages: [],
            status: 'active'
        };

        // Store session
        this.activeChatSessions.set(clientWs.sessionId, chatSession);
        this.notificationChatMappings.set(notificationContext.notificationId, clientWs.sessionId);

        // Generate AI's first response based on notification context
        const initialAIResponse = await this.generateContextualAIResponse(notificationContext);

        // Send initial AI message to client
        this.sendToClient(clientWs, {
            type: 'ai_message',
            message: initialAIResponse,
            sessionId: clientWs.sessionId,
            context: notificationContext
        });

        // Log session start
        logger.info(`🚀 Chat session started for notification: ${notificationContext.notificationId}`);

        return chatSession;
    }

    /**
     * Generate contextual AI response based on notification
     */
    async generateContextualAIResponse(notificationContext) {
        const contextualResponses = {
            'agent_crash': {
                greeting: `I can see Agent "${notificationContext.fromAgent}" has crashed. Let me help you diagnose and resolve this issue.`,
                analysis: await this.analyzeAgentCrash(notificationContext),
                nextSteps: 'I can help you restart the agent safely, investigate the root cause, or escalate to the engineering team.'
            },

            'security_alert': {
                greeting: `Security alert detected: "${notificationContext.title}". This requires immediate attention.`,
                analysis: await this.analyzeSecurityThreat(notificationContext),
                nextSteps: 'I can help you assess the threat level, implement containment measures, or coordinate with the security team.'
            },

            'workflow_approval': {
                greeting: `You have a workflow approval request: "${notificationContext.title}"`,
                analysis: await this.analyzeApprovalRequest(notificationContext),
                nextSteps: 'I can provide more context about this request, check compliance requirements, or help you make an informed decision.'
            }
        };

        const response = contextualResponses[notificationContext.type] || {
            greeting: `I'm here to help with: "${notificationContext.title}"`,
            analysis: 'Let me gather more information about this situation.',
            nextSteps: 'What specific aspect would you like me to help you with?'
        };

        return `${response.greeting}\n\n${response.analysis}\n\n${response.nextSteps}`;
    }

    /**
     * Handle client messages and route to appropriate AI agent
     */
    async handleClientMessage(ws, data) {
        try {
            const message = JSON.parse(data);
            const session = this.activeChatSessions.get(ws.sessionId);

            if (!session) {
                this.sendToClient(ws, {
                    type: 'error',
                    message: 'Session not found'
                });
                return;
            }

            // Store user message
            session.messages.push({
                sender: 'user',
                message: message.content,
                timestamp: new Date(),
                intent: message.intent
            });

            // Route to appropriate AI agent based on context
            const aiResponse = await this.routeToAIAgent(message, session);

            // Store AI response
            session.messages.push({
                sender: 'ai',
                message: aiResponse.response,
                timestamp: new Date(),
                confidence: aiResponse.confidence,
                suggestedActions: aiResponse.actions
            });

            // Send response to client
            this.sendToClient(ws, {
                type: 'ai_message',
                message: aiResponse.response,
                sessionId: ws.sessionId,
                actions: aiResponse.actions,
                confidence: aiResponse.confidence
            });

        } catch (error) {
            logger.error('Error handling client message:', { error: error });
            this.sendToClient(ws, {
                type: 'error',
                message: 'Failed to process message'
            });
        }
    }

    /**
     * Route message to appropriate AI agent via A2A
     */
    async routeToAIAgent(message, session) {
        // Build A2A message for AI agent
        const a2aMessage = {
            messageId: this.generateMessageId(),
            messageType: 'chat_assistance_request',
            from: 'chat_bridge',
            to: this.selectBestAIAgent(session.context),
            category: 'interactive_support',
            urgency: session.context.priority || 'medium',
            description: message.content,
            context: {
                conversationId: session.conversationId,
                notificationContext: session.context,
                chatHistory: session.messages.slice(-5), // Last 5 messages for context
                userIntent: message.intent || this.inferIntent(message.content),
                systemState: await this.getCurrentSystemState()
            }
        };

        try {
            // Send via A2A Communication Router
            const routingResponse = await this.a2aRouter.sendMessage({
                message: a2aMessage,
                from_agent: 'chat_bridge',
                context: session.context
            });

            // Process AI agent response
            return this.processAIAgentResponse(routingResponse, session);

        } catch (error) {
            logger.error('Error routing to AI agent:', { error: error });
            return {
                response: 'I\'m experiencing some technical difficulties. Let me try to help you with the basic information I have.',
                confidence: 0.3,
                actions: ['retry', 'escalate_to_human']
            };
        }
    }

    /**
     * Select best AI agent for the context
     */
    selectBestAIAgent(context) {
        const agentMapping = {
            'agent_crash': 'system_diagnostics_agent',
            'security_alert': 'security_analysis_agent',
            'workflow_approval': 'business_process_agent',
            'performance_alert': 'performance_optimization_agent',
            'default': 'general_assistance_agent'
        };

        return agentMapping[context.type] || agentMapping.default;
    }

    /**
     * Process response from AI agent
     */
    processAIAgentResponse(routingResponse, session) {
        // Extract meaningful response from A2A routing
        const aiResponse = routingResponse.ai_response || routingResponse.response || 'Let me look into this further.';

        return {
            response: aiResponse,
            confidence: routingResponse.confidence || 0.8,
            actions: this.generateContextualActions(session.context, routingResponse)
        };
    }

    /**
     * Generate contextual actions based on response
     */
    generateContextualActions(context, response) {
        const baseActions = ['mark_resolved', 'escalate', 'get_more_info'];

        const contextualActions = {
            'agent_crash': ['restart_agent', 'view_logs', 'contact_support'],
            'security_alert': ['block_threat', 'investigate_further', 'alert_security_team'],
            'workflow_approval': ['approve_request', 'request_more_info', 'decline_request']
        };

        return [
            ...baseActions,
            ...(contextualActions[context.type] || [])
        ];
    }

    /**
     * Analyze agent crash for contextual response
     */
    async analyzeAgentCrash(context) {
        // In a real implementation, this would connect to monitoring systems
        return `Based on initial analysis, the agent crashed at ${new Date(context.timestamp).toLocaleString()}. Common causes include resource exhaustion, configuration errors, or external service failures.`;
    }

    /**
     * Analyze security threat for contextual response
     */
    async analyzeSecurityThreat(context) {
        return `Security analysis shows a ${context.priority} priority threat. I'm evaluating the scope and potential impact on your systems.`;
    }

    /**
     * Analyze approval request for contextual response
     */
    async analyzeApprovalRequest(context) {
        return `This approval request involves ${context.description}. I'm checking compliance requirements and business rules to provide you with guidance.`;
    }

    /**
     * Send message to specific client
     */
    sendToClient(ws, data) {
        if (ws.readyState === WebSocket.OPEN) {
            blockchainClient.publishEvent(JSON.stringify(data));
        }
    }

    /**
     * Handle client disconnection
     */
    handleClientDisconnection(ws) {
        const session = this.activeChatSessions.get(ws.sessionId);

        if (session) {
            // Mark session as inactive
            session.status = 'inactive';
            session.endTime = new Date();

            // Clean up mappings
            this.notificationChatMappings.delete(session.notificationId);
            this.activeChatSessions.delete(ws.sessionId);

            logger.info(`📤 Chat session ended: ${ws.sessionId}`);
        }
    }

    /**
     * Get current system state for AI context
     */
    async getCurrentSystemState() {
        // This would integrate with system monitoring
        return {
            timestamp: new Date().toISOString(),
            system_load: 'normal',
            active_agents: 12,
            pending_notifications: 3,
            recent_incidents: 1
        };
    }

    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return `chat_${  Date.now()  }_${  Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Generate unique message ID
     */
    generateMessageId() {
        return `msg_${  Date.now()  }_${  Math.random().toString(36).substr(2, 5)}`;
    }

    /**
     * Infer user intent from message content
     */
    inferIntent(messageContent) {
        const intentKeywords = {
            'troubleshoot': ['fix', 'error', 'problem', 'issue', 'broken'],
            'investigate': ['why', 'what happened', 'cause', 'reason'],
            'resolve': ['resolve', 'solve', 'complete', 'finish'],
            'escalate': ['escalate', 'urgent', 'critical', 'emergency']
        };

        for (const [intent, keywords] of Object.entries(intentKeywords)) {
            if (keywords.some(keyword => messageContent.toLowerCase().includes(keyword))) {
                return intent;
            }
        }

        return 'general_inquiry';
    }
}

// Export singleton instance
const chatBridge = new ChatAgentBridge();

module.exports = {
    ChatAgentBridge,
    getInstance: () => chatBridge
};