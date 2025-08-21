/**
 * A2A Human Message Bridge Service
 * Bridges A2A protocol messages between agents and humans
 * Routes messages based on Python communication configuration
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');
const cds = require('@sap/cds');

class A2AHumanMessageBridge extends EventEmitter {
    constructor() {
        super();
        this.logger = cds.log('a2a-human-bridge');
        this.isInitialized = false;
        
        // Bridge configuration
        this.config = {
            pythonRouterUrl: process.env.A2A_ROUTER_URL || 'http://localhost:8001/a2a/route',
            agentConfigUrl: process.env.AGENT_CONFIG_URL || 'http://localhost:8001/a2a/agents',
            notificationServiceUrl: process.env.NOTIFICATION_SERVICE_URL || 'http://localhost:8080/notifications',
            bridgeEnabled: process.env.A2A_BRIDGE_ENABLED !== 'false'
        };
        
        // Message tracking
        this.pendingMessages = new Map(); // messageId -> message data
        this.messageHistory = [];
        this.agentProfiles = new Map(); // agentId -> profile
        this.humanResponses = new Map(); // messageId -> response data
        
        // Statistics
        this.stats = {
            messagesRouted: 0,
            messagesToHumans: 0,
            messagesToAgents: 0,
            escalatedMessages: 0,
            averageResponseTime: 0
        };
        
        this.logger.info('A2A Human Message Bridge initialized');
    }
    
    async initialize() {
        try {
            if (!this.config.bridgeEnabled) {
                this.logger.info('A2A Bridge disabled by configuration');
                return;
            }
            
            // Load agent communication profiles
            await this.loadAgentProfiles();
            
            // Test connection to Python router
            await this.testPythonRouterConnection();
            
            // Set up message processing
            this.setupMessageProcessing();
            
            this.isInitialized = true;
            this.logger.info('✅ A2A Human Message Bridge ready');
            
        } catch (error) {
            this.logger.error('❌ Failed to initialize A2A Human Message Bridge:', error);
            throw error;
        }
    }
    
    async loadAgentProfiles() {
        try {
            const response = await axios.get(`${this.config.agentConfigUrl}/profiles`, {
                timeout: 5000
            });
            
            if (response.data && response.data.profiles) {
                Object.entries(response.data.profiles).forEach(([agentId, profile]) => {
                    this.agentProfiles.set(agentId, profile);
                });
                
                this.logger.info(`📋 Loaded ${this.agentProfiles.size} agent communication profiles`);
            }
            
        } catch (error) {
            this.logger.error('Failed to load agent profiles:', error);
            // Continue with default profiles
        }
    }
    
    async testPythonRouterConnection() {
        try {
            const response = await axios.get(`${this.config.pythonRouterUrl}/health`, {
                timeout: 3000
            });
            
            this.logger.info('✅ Python A2A router connection verified');
            
        } catch (error) {
            this.logger.warn('⚠️ Python A2A router not available, using fallback rules');
        }
    }
    
    setupMessageProcessing() {
        // Set up periodic cleanup of old messages
        setInterval(() => {
            this.cleanupOldMessages();
        }, 60000); // Every minute
        
        // Set up statistics update
        setInterval(() => {
            this.updateStatistics();
        }, 30000); // Every 30 seconds
        
        this.logger.info('📡 Message processing setup complete');
    }
    
    async processA2AMessage(message, fromAgent, context = {}) {
        try {
            if (!this.isInitialized || !this.config.bridgeEnabled) {
                this.logger.debug('Bridge not initialized or disabled, skipping message');
                return { target: 'agent', reason: 'bridge_disabled' };
            }
            
            // Generate unique message ID if not provided
            if (!message.messageId) {
                message.messageId = this.generateMessageId();
            }
            
            // Enhance message with context
            const enhancedMessage = {
                ...message,
                fromAgent,
                timestamp: new Date().toISOString(),
                bridgeProcessedAt: new Date().toISOString(),
                ...context
            };
            
            // Add agent profile data
            const agentProfile = this.agentProfiles.get(fromAgent);
            if (agentProfile) {
                enhancedMessage.agentProfile = agentProfile;
            }
            
            // Route the message using Python configuration
            const routingDecision = await this.routeMessage(enhancedMessage, fromAgent);
            
            // Process based on routing decision
            await this.handleRoutingDecision(enhancedMessage, routingDecision);
            
            // Update statistics
            this.stats.messagesRouted++;
            if (routingDecision.target === 'human' || routingDecision.target === 'both') {
                this.stats.messagesToHumans++;
            }
            if (routingDecision.target === 'agent' || routingDecision.target === 'both') {
                this.stats.messagesToAgents++;
            }
            if (routingDecision.target === 'escalate') {
                this.stats.escalatedMessages++;
            }
            
            // Store message in history
            this.messageHistory.push({
                messageId: message.messageId,
                fromAgent,
                routingDecision,
                timestamp: new Date().toISOString(),
                processed: true
            });
            
            // Keep only last 1000 messages
            if (this.messageHistory.length > 1000) {
                this.messageHistory = this.messageHistory.slice(-1000);
            }
            
            this.logger.info(`📨 A2A message ${message.messageId} routed to ${routingDecision.target} using rule ${routingDecision.rule_applied?.rule_id}`);
            
            return routingDecision;
            
        } catch (error) {
            this.logger.error(`❌ Failed to process A2A message: ${error.message}`);
            
            // Fail-safe: route to human
            return {
                target: 'human',
                error: error.message,
                reason: 'processing_error',
                timestamp: new Date().toISOString()
            };
        }
    }
    
    async routeMessage(message, fromAgent) {
        try {
            // Try to use Python router first
            const response = await axios.post(this.config.pythonRouterUrl, {
                message,
                from_agent: fromAgent,
                context: {
                    timestamp: new Date().toISOString(),
                    bridge_version: '1.0.0'
                }
            }, {
                timeout: 5000,
                headers: { 'Content-Type': 'application/json' }
            });
            
            return response.data;
            
        } catch (error) {
            this.logger.warn('Python router unavailable, using JavaScript fallback rules');
            return this.fallbackRouting(message, fromAgent);
        }
    }
    
    fallbackRouting(message, fromAgent) {
        // JavaScript fallback routing rules
        const messageType = message.message_type || message.type;
        const urgency = message.urgency || 'medium';
        const category = message.category;
        
        // Critical conditions always go to human
        if (urgency === 'critical' || urgency === 'emergency') {
            return {
                target: 'human',
                rule_applied: {
                    rule_id: 'fallback_critical',
                    name: 'Fallback Critical',
                    description: 'Critical messages always routed to humans'
                },
                requires_human_confirmation: true,
                human_response_timeout_minutes: 30,
                reason: 'critical_urgency'
            };
        }
        
        // Security alerts to human
        if (category === 'security' || messageType === 'alert') {
            return {
                target: 'human',
                rule_applied: {
                    rule_id: 'fallback_security',
                    name: 'Fallback Security',
                    description: 'Security events routed to humans'
                },
                requires_human_confirmation: true,
                reason: 'security_event'
            };
        }
        
        // Approval requests to human
        if (messageType === 'approval_request') {
            return {
                target: 'human',
                rule_applied: {
                    rule_id: 'fallback_approval',
                    name: 'Fallback Approval',
                    description: 'Approval requests routed to humans'
                },
                requires_human_confirmation: true,
                human_response_timeout_minutes: 240,
                reason: 'approval_required'
            };
        }
        
        // Default to agent routing for routine messages
        return {
            target: 'agent',
            rule_applied: {
                rule_id: 'fallback_default',
                name: 'Fallback Default',
                description: 'Default routing to agents'
            },
            reason: 'routine_message'
        };
    }
    
    async handleRoutingDecision(message, routingDecision) {
        const target = routingDecision.target;
        
        try {
            if (target === 'human' || target === 'both') {
                await this.sendToHuman(message, routingDecision);
            }
            
            if (target === 'agent' || target === 'both') {
                await this.sendToAgent(message, routingDecision);
            }
            
            if (target === 'escalate') {
                await this.scheduleEscalation(message, routingDecision);
            }
            
        } catch (error) {
            this.logger.error(`Failed to handle routing decision: ${error.message}`);
        }
    }
    
    async sendToHuman(message, routingDecision) {
        try {
            // Convert A2A message to human-readable notification
            const humanNotification = this.convertA2AToHumanNotification(message, routingDecision);
            
            // Send to notification system
            await this.sendNotificationToHuman(humanNotification);
            
            // Track pending message if response required
            if (routingDecision.requires_human_confirmation) {
                this.pendingMessages.set(message.messageId, {
                    originalMessage: message,
                    routingDecision,
                    sentToHumanAt: new Date().toISOString(),
                    awaiting_response: true
                });
                
                // Set timeout for human response
                this.setHumanResponseTimeout(message.messageId, routingDecision.human_response_timeout_minutes || 60);
            }
            
            this.logger.info(`👤 Message ${message.messageId} sent to human via notification system`);
            
        } catch (error) {
            this.logger.error(`Failed to send message to human: ${error.message}`);
        }
    }
    
    convertA2AToHumanNotification(message, routingDecision) {
        // Convert A2A protocol message to human-friendly notification
        const humanReadableMessage = this.generateHumanReadableMessage(message);
        
        return {
            id: this.generateNotificationId(),
            title: this.generateHumanTitle(message),
            message: humanReadableMessage,
            type: this.mapA2ATypeToNotificationType(message),
            priority: this.mapA2APriorityToNotificationPriority(message.urgency),
            category: 'agent_message',
            messageType: message.message_type || message.type,
            source: message.fromAgent,
            sourceAgent: message.fromAgent,
            status: 'unread',
            createdAt: new Date().toISOString(),
            metadata: {
                messageId: message.messageId,
                fromAgent: message.fromAgent,
                toAgent: 'human',
                originalA2AMessage: message,
                translatedMessage: humanReadableMessage,
                conversationId: message.conversationId || this.generateConversationId(message.fromAgent),
                requiresResponse: routingDecision.requires_human_confirmation,
                routingRule: routingDecision.rule_applied
            },
            actions: this.generateHumanActions(message, routingDecision),
            requiresUserAttention: routingDecision.requires_human_confirmation,
            // A2A specific properties for enhanced UI
            a2aMessage: {
                originalProtocol: message.protocol?.version || 'A2A-v1.0',
                messageStructure: message.protocol || {},
                semanticContext: message.context || {},
                intentClassification: this.classifyIntent(message),
                urgencyLevel: message.urgency || 'medium',
                expectedResponseTime: routingDecision.human_response_timeout_minutes
            }
        };
    }
    
    generateHumanReadableMessage(message) {
        const messageType = message.message_type || message.type;
        const fromAgent = this.getAgentDisplayName(message.fromAgent);
        
        // Use human message if provided
        if (message.humanMessage || message.human_message) {
            return message.humanMessage || message.human_message;
        }
        
        // Generate based on message type and content
        switch (messageType) {
            case 'approval_request':
                return `${fromAgent} is requesting your approval for: ${message.description || message.action || 'an operation'}`;
            
            case 'data_request':
                return `${fromAgent} needs access to data: ${message.data_description || message.resource || 'unspecified data'}`;
            
            case 'error':
            case 'alert':
                return `${fromAgent} encountered an issue: ${message.error_description || message.description || 'requires attention'}`;
            
            case 'status_update':
                return `${fromAgent} status update: ${message.status || message.description || 'status changed'}`;
            
            case 'request':
                return `${fromAgent} is requesting: ${message.description || message.request || 'assistance'}`;
            
            case 'notification':
                return `${fromAgent} reports: ${message.description || message.message || 'notification'}`;
            
            default:
                return `${fromAgent} sent a message: ${message.description || message.message || JSON.stringify(message).substring(0, 100)}`;
        }
    }
    
    generateHumanTitle(message) {
        const messageType = (message.message_type || message.type || 'message').toUpperCase();
        const fromAgent = this.getAgentDisplayName(message.fromAgent);
        
        return `A2A ${messageType}: ${fromAgent} → Human`;
    }
    
    mapA2ATypeToNotificationType(message) {
        const messageType = message.message_type || message.type;
        const urgency = message.urgency;
        
        if (urgency === 'critical' || urgency === 'emergency') return 'error';
        if (messageType === 'error' || messageType === 'alert') return 'warning';
        if (messageType === 'approval_request') return 'warning';
        if (messageType === 'status_update' && message.status === 'success') return 'success';
        
        return 'info';
    }
    
    mapA2APriorityToNotificationPriority(urgency) {
        const priorityMap = {
            'emergency': 'critical',
            'critical': 'critical', 
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        };
        
        return priorityMap[urgency] || 'medium';
    }
    
    generateHumanActions(message, routingDecision) {
        const actions = [];
        const messageType = message.message_type || message.type;
        
        // Always add reply action for requests
        if (routingDecision.requires_human_confirmation) {
            actions.push({
                text: "Reply",
                action: "reply_to_agent",
                type: "Emphasized"
            });
        }
        
        // Type-specific actions
        if (messageType === 'approval_request') {
            actions.push(
                {
                    text: "Approve",
                    action: "approve_request",
                    type: "Accept"
                },
                {
                    text: "Reject", 
                    action: "reject_request",
                    type: "Reject"
                }
            );
        }
        
        if (messageType === 'data_request') {
            actions.push(
                {
                    text: "Grant Access",
                    action: "grant_data_access",
                    type: "Accept"
                },
                {
                    text: "Deny Access",
                    action: "deny_data_access", 
                    type: "Reject"
                }
            );
        }
        
        // Generic actions
        actions.push({
            text: "Delegate",
            action: "delegate_to_agent",
            type: "Default"
        });
        
        return actions;
    }
    
    classifyIntent(message) {
        const messageType = message.message_type || message.type;
        const content = (message.description || message.message || '').toLowerCase();
        
        if (messageType === 'approval_request') return 'approval_request';
        if (messageType === 'data_request') return 'data_request';
        if (messageType === 'error') return 'error_report';
        if (messageType === 'status_update') return 'status_update';
        
        // Content-based classification
        if (content.includes('approve') || content.includes('permission')) return 'approval_request';
        if (content.includes('data') || content.includes('access')) return 'data_request';
        if (content.includes('error') || content.includes('problem')) return 'error_report';
        
        return 'general_request';
    }
    
    async sendNotificationToHuman(notification) {
        try {
            // Send to notification service
            await axios.post(`${this.config.notificationServiceUrl}/send`, {
                notification,
                source: 'a2a_bridge',
                timestamp: new Date().toISOString()
            });
            
            // Emit event for WebSocket forwarding
            this.emit('human_notification', notification);
            
        } catch (error) {
            this.logger.error(`Failed to send notification: ${error.message}`);
            throw error;
        }
    }
    
    async sendToAgent(message, routingDecision) {
        try {
            // Determine target agents
            const targetAgents = this.determineTargetAgents(message, routingDecision);
            
            // Send to each target agent
            for (const targetAgent of targetAgents) {
                await this.forwardMessageToAgent(message, targetAgent);
            }
            
            this.logger.info(`🤖 Message ${message.messageId} sent to ${targetAgents.length} agents`);
            
        } catch (error) {
            this.logger.error(`Failed to send message to agents: ${error.message}`);
        }
    }
    
    determineTargetAgents(message, routingDecision) {
        // Use routing decision if it specifies agents
        if (routingDecision.agent_routing?.target_agents) {
            return routingDecision.agent_routing.target_agents;
        }
        
        // Fallback logic based on message category
        const category = message.category;
        
        if (category === 'data_processing') {
            return ['data_product_agent_0', 'sql_agent'];
        } else if (category === 'security') {
            return ['security_monitor'];
        } else if (category === 'workflow') {
            return ['workflow_engine'];
        } else {
            return ['agent_manager']; // Default to agent manager
        }
    }
    
    async forwardMessageToAgent(message, targetAgent) {
        // This would integrate with the agent communication system
        // For now, emit an event that can be picked up by the agent runtime
        this.emit('agent_message', {
            message,
            targetAgent,
            timestamp: new Date().toISOString()
        });
    }
    
    async scheduleEscalation(message, routingDecision) {
        const escalationDelay = routingDecision.escalation_timeout_minutes || 15;
        
        setTimeout(async () => {
            try {
                // Check if message was handled by agents
                if (!this.wasMessageHandledByAgents(message.messageId)) {
                    this.logger.info(`⏰ Escalating message ${message.messageId} to human after ${escalationDelay} minutes`);
                    
                    // Route to human with escalation context
                    const escalationRouting = {
                        ...routingDecision,
                        target: 'human',
                        escalated: true,
                        escalation_reason: `No agent response after ${escalationDelay} minutes`
                    };
                    
                    await this.sendToHuman(message, escalationRouting);
                }
            } catch (error) {
                this.logger.error(`Escalation failed: ${error.message}`);
            }
        }, escalationDelay * 60 * 1000);
        
        this.logger.info(`⏳ Scheduled escalation for message ${message.messageId} in ${escalationDelay} minutes`);
    }
    
    wasMessageHandledByAgents(messageId) {
        // Check if any agents responded to this message
        // This would integrate with agent response tracking
        return false; // Simplified for now
    }
    
    setHumanResponseTimeout(messageId, timeoutMinutes) {
        setTimeout(() => {
            const pendingMessage = this.pendingMessages.get(messageId);
            if (pendingMessage && pendingMessage.awaiting_response) {
                this.logger.warn(`⏰ Human response timeout for message ${messageId}`);
                
                // Mark as timed out and potentially re-route
                pendingMessage.timedOut = true;
                pendingMessage.timedOutAt = new Date().toISOString();
                
                this.emit('human_response_timeout', {
                    messageId,
                    originalMessage: pendingMessage.originalMessage
                });
            }
        }, timeoutMinutes * 60 * 1000);
    }
    
    async processHumanResponse(messageId, response, respondedBy = 'human') {
        try {
            const pendingMessage = this.pendingMessages.get(messageId);
            if (!pendingMessage) {
                this.logger.error(`No pending message found for ID: ${messageId}`);
                return false;
            }
            
            // Convert human response to A2A format
            const a2aResponse = this.convertHumanResponseToA2A(response, pendingMessage.originalMessage, respondedBy);
            
            // Send response back to originating agent
            await this.sendA2AResponseToAgent(a2aResponse, pendingMessage.originalMessage.fromAgent);
            
            // Update pending message
            pendingMessage.human_response = response;
            pendingMessage.responded_at = new Date().toISOString();
            pendingMessage.responded_by = respondedBy;
            pendingMessage.awaiting_response = false;
            
            // Store in human responses
            this.humanResponses.set(messageId, {
                messageId,
                response,
                respondedBy,
                respondedAt: new Date().toISOString(),
                a2aResponse
            });
            
            this.logger.info(`✅ Human response processed for message ${messageId}`);
            return true;
            
        } catch (error) {
            this.logger.error(`Failed to process human response: ${error.message}`);
            return false;
        }
    }
    
    convertHumanResponseToA2A(humanResponse, originalMessage, respondedBy) {
        return {
            messageId: this.generateMessageId(),
            conversationId: originalMessage.conversationId || this.generateConversationId(originalMessage.fromAgent),
            timestamp: new Date().toISOString(),
            fromAgent: 'human',
            toAgent: originalMessage.fromAgent,
            messageType: 'response',
            inReplyTo: originalMessage.messageId,
            protocol: {
                version: 'A2A-v1.0',
                intent: this.determineResponseIntent(humanResponse, originalMessage),
                context: 'human_response'
            },
            payload: {
                humanResponse: humanResponse.text || humanResponse.message,
                responseType: humanResponse.action || 'text_response',
                approved: humanResponse.approved,
                data: humanResponse.data || {},
                instructions: humanResponse.instructions || []
            },
            metadata: {
                respondedBy,
                originalMessageId: originalMessage.messageId,
                translatedAt: new Date().toISOString(),
                translationMethod: 'human-to-a2a-bridge'
            }
        };
    }
    
    determineResponseIntent(humanResponse, originalMessage) {
        if (humanResponse.approved === true) return 'approval_granted';
        if (humanResponse.approved === false) return 'approval_denied';
        if (humanResponse.action === 'approve_request') return 'approval_granted';
        if (humanResponse.action === 'reject_request') return 'approval_denied';
        if (humanResponse.action === 'grant_data_access') return 'data_access_granted';
        if (humanResponse.action === 'deny_data_access') return 'data_access_denied';
        
        return 'general_response';
    }
    
    async sendA2AResponseToAgent(a2aResponse, targetAgent) {
        // Send A2A formatted response back to the agent
        this.emit('a2a_response', {
            response: a2aResponse,
            targetAgent,
            timestamp: new Date().toISOString()
        });
        
        this.logger.info(`🔄 A2A response sent to agent ${targetAgent}`);
    }
    
    // Utility methods
    generateMessageId() {
        return 'bridge_msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    generateNotificationId() {
        return 'bridge_notif_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    generateConversationId(agentId) {
        return `conv_${agentId}_${Date.now()}`;
    }
    
    getAgentDisplayName(agentId) {
        const profile = this.agentProfiles.get(agentId);
        if (profile && profile.agent_name) {
            return profile.agent_name;
        }
        
        // Fallback name mapping
        const nameMap = {
            'data_product_agent_0': 'Data Product Agent',
            'security_monitor': 'Security Monitor',
            'workflow_engine': 'Workflow Engine',
            'reasoning_agent': 'Reasoning Agent',
            'sql_agent': 'SQL Agent'
        };
        
        return nameMap[agentId] || agentId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    cleanupOldMessages() {
        const cutoffTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
        
        // Clean up old pending messages
        for (const [messageId, messageData] of this.pendingMessages.entries()) {
            const messageTime = new Date(messageData.sentToHumanAt).getTime();
            if (messageTime < cutoffTime) {
                this.pendingMessages.delete(messageId);
            }
        }
        
        // Clean up old responses
        for (const [messageId, responseData] of this.humanResponses.entries()) {
            const responseTime = new Date(responseData.respondedAt).getTime();
            if (responseTime < cutoffTime) {
                this.humanResponses.delete(messageId);
            }
        }
        
        // Clean up message history
        this.messageHistory = this.messageHistory.filter(msg => {
            const msgTime = new Date(msg.timestamp).getTime();
            return msgTime >= cutoffTime;
        });
    }
    
    updateStatistics() {
        // Calculate average response time
        const responses = Array.from(this.humanResponses.values());
        if (responses.length > 0) {
            const totalResponseTime = responses.reduce((sum, response) => {
                const sent = new Date(response.sentAt || 0).getTime();
                const responded = new Date(response.respondedAt).getTime();
                return sum + (responded - sent);
            }, 0);
            
            this.stats.averageResponseTime = totalResponseTime / responses.length;
        }
        
        this.emit('statistics_updated', this.stats);
    }
    
    getStatus() {
        return {
            initialized: this.isInitialized,
            enabled: this.config.bridgeEnabled,
            agentProfiles: this.agentProfiles.size,
            pendingMessages: this.pendingMessages.size,
            statistics: this.stats,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = A2AHumanMessageBridge;