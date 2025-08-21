sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/TextArea",
    "sap/m/Input",
    "sap/m/VBox",
    "sap/m/HBox"
], function (Controller, JSONModel, MessageToast, MessageBox, Fragment, Dialog, Button, TextArea, Input, VBox, HBox) {
    "use strict";

    return Controller.extend("a2a.network.controller.EnhancedNotificationCenter", {
        
        onInit: function () {
            // Initialize enhanced models
            this.initializeEnhancedModels();
            
            // Initialize WebSocket with enhanced event handling
            this.initializeEnhancedWebSocket();
            
            // Load initial notifications
            this.loadNotifications();
            
            // Set up auto-refresh with smarter intervals
            this.setupSmartAutoRefresh();
            
            // Initialize A2A message bridge
            this.initializeA2AMessageBridge();
            
            // Set up real-time statistics updates
            this.setupRealtimeStatistics();
        },

        initializeEnhancedModels: function () {
            // Enhanced notification data model
            this.oNotificationModel = new JSONModel({
                notifications: [],
                filteredNotifications: [],
                filteredCount: 0,
                stats: {
                    total: 0,
                    unread: 0,
                    byType: {
                        info: 0,
                        warning: 0,
                        error: 0,
                        success: 0,
                        system: 0
                    },
                    byPriority: {
                        low: 0,
                        medium: 0,
                        high: 0,
                        critical: 0
                    },
                    byCategory: {
                        agent_message: 0,
                        agent_crash: 0,
                        security: 0,
                        workflow: 0,
                        transaction: 0,
                        system: 0
                    },
                    byStatus: {
                        unread: 0,
                        read: 0,
                        dismissed: 0
                    },
                    categoryChart: []
                },
                connectionStatus: 'disconnected',
                lastUpdated: new Date().toISOString(),
                totalEventsProcessed: 0,
                availableAgents: []
            });
            this.getView().setModel(this.oNotificationModel, "notificationModel");

            // Enhanced filter model with smart filters
            this.oFilterModel = new JSONModel({
                status: "",
                type: "",
                priority: "",
                category: "",
                messageType: "",
                timeRange: "24h",
                sourceAgents: [],
                searchText: "",
                // Smart toggle filters
                unreadOnly: false,
                highPriorityOnly: false,
                agentMessagesOnly: false,
                securityOnly: false,
                // Advanced filters
                hasActions: null,
                requiresApproval: null,
                isFromAgent: null
            });
            this.getView().setModel(this.oFilterModel, "filterModel");

            // Enhanced pagination model
            this.oPaginationModel = new JSONModel({
                limit: 50, // Increased for better UX
                offset: 0,
                currentPage: 1,
                totalPages: 1,
                total: 0
            });
            this.getView().setModel(this.oPaginationModel, "paginationModel");

            // Sort model
            this.oSortModel = new JSONModel({
                field: "createdAt",
                ascending: false
            });
            this.getView().setModel(this.oSortModel, "sortModel");

            // Settings model
            this.oSettingsModel = new JSONModel({
                autoRefresh: true,
                refreshInterval: 10000,
                showToastNotifications: true,
                groupByCategory: false,
                compactView: false,
                enableA2ABridge: true,
                agentReplyTimeout: 30000
            });
            this.getView().setModel(this.oSettingsModel, "settingsModel");

            // A2A Message Bridge Model
            this.oA2AModel = new JSONModel({
                bridgeStatus: 'inactive',
                pendingMessages: [],
                agentConversations: {},
                messageHistory: []
            });
            this.getView().setModel(this.oA2AModel, "a2aModel");
        },

        initializeEnhancedWebSocket: function () {
            const wsUrl = this.getWebSocketUrl();
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = this.onEnhancedWebSocketOpen.bind(this);
            this.ws.onmessage = this.onEnhancedWebSocketMessage.bind(this);
            this.ws.onclose = this.onEnhancedWebSocketClose.bind(this);
            this.ws.onerror = this.onEnhancedWebSocketError.bind(this);

            // Enhanced reconnection with exponential backoff
            this.wsUrl = wsUrl;
            this.reconnectAttempts = 0;
            this.maxReconnectAttempts = 10;
            this.baseReconnectDelay = 1000;
            this.maxReconnectDelay = 30000;
        },

        onEnhancedWebSocketOpen: function () {
            console.log("🔗 Enhanced WebSocket connection established");
            
            this.oNotificationModel.setProperty("/connectionStatus", "connected");
            this.reconnectAttempts = 0;
            
            // Subscribe to enhanced event types
            this.subscribeToEnhancedEvents();
            
            MessageToast.show("Connected to A2A notification stream", {
                duration: 2000,
                at: "center top"
            });
        },

        subscribeToEnhancedEvents: function () {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                const subscriptions = [
                    // Real agent crash events
                    "agent.crashed",
                    "agent.recovered", 
                    "agent.degraded",
                    // Real security events
                    "security.alert",
                    "security.threat_detected",
                    // Real workflow events
                    "workflow.approval_required",
                    "workflow.task_completed",
                    // A2A agent communications
                    "a2a.message.request",
                    "a2a.message.response",
                    "a2a.message.notification",
                    // System events
                    "system.alert",
                    "transaction.completed",
                    "transaction.failed"
                ];

                this.ws.send(JSON.stringify({
                    type: 'subscribe',
                    events: subscriptions,
                    clientId: this.generateClientId()
                }));

                console.log("📡 Subscribed to enhanced A2A events:", subscriptions);
            }
        },

        onEnhancedWebSocketMessage: function (message) {
            try {
                const data = JSON.parse(message.data);
                console.log("📨 Enhanced notification received:", data);
                
                // Increment total events processed
                const currentTotal = this.oNotificationModel.getProperty("/totalEventsProcessed");
                this.oNotificationModel.setProperty("/totalEventsProcessed", currentTotal + 1);
                
                switch (data.type) {
                    case 'agent.crashed':
                        this.handleAgentCrashNotification(data);
                        break;
                    case 'agent.recovered':
                        this.handleAgentRecoveryNotification(data);
                        break;
                    case 'security.alert':
                        this.handleSecurityAlertNotification(data);
                        break;
                    case 'workflow.approval_required':
                        this.handleWorkflowApprovalNotification(data);
                        break;
                    case 'a2a.message.request':
                    case 'a2a.message.response':
                    case 'a2a.message.notification':
                        this.handleA2AMessageNotification(data);
                        break;
                    default:
                        this.handleGenericNotification(data);
                }
                
                // Update statistics and UI
                this.updateRealtimeStatistics();
                this.applyCurrentFilters();
                this.updateLastUpdated();
                
            } catch (error) {
                console.error("❌ Failed to process enhanced notification:", error);
            }
        },

        handleAgentCrashNotification: function (data) {
            const notification = {
                id: this.generateNotificationId(),
                title: `Agent Crash: ${data.data.agentName}`,
                message: `Agent ${data.data.agentName} has crashed and is no longer responding. Previous status: ${data.data.previousStatus}`,
                type: "error",
                priority: "critical",
                category: "agent_crash",
                source: "Agent Monitor",
                sourceAgent: data.data.agentId,
                status: "unread",
                createdAt: new Date().toISOString(),
                metadata: {
                    agentId: data.data.agentId,
                    previousStatus: data.data.previousStatus,
                    currentStatus: data.data.currentStatus,
                    healthScore: data.data.healthScore,
                    lastActivity: data.data.lastActivity,
                    environment: data.data.details?.environment
                },
                actions: [
                    {
                        text: "Restart Agent",
                        action: "restart_agent",
                        type: "Emphasized"
                    },
                    {
                        text: "View Logs",
                        action: "view_logs",
                        type: "Default"
                    }
                ],
                requiresUserAttention: true
            };
            
            this.addNotification(notification);
            
            // Show critical toast
            MessageToast.show(`🚨 CRITICAL: Agent ${data.data.agentName} has crashed!`, {
                duration: 5000,
                at: "center center"
            });
        },

        handleSecurityAlertNotification: function (data) {
            const notification = {
                id: this.generateNotificationId(),
                title: `Security Alert: ${data.data.title}`,
                message: data.data.message,
                type: "error",
                priority: data.data.severity === "critical" ? "critical" : "high",
                category: "security",
                source: "Security Monitor",
                sourceAgent: "security_monitor",
                status: "unread",
                createdAt: new Date().toISOString(),
                metadata: {
                    threatType: data.data.threat,
                    sourceIp: data.data.source,
                    userId: data.data.userId,
                    indicators: data.data.metadata?.indicators || [],
                    responseActions: data.data.metadata?.actions || []
                },
                actions: data.data.metadata?.responseActions?.map(action => ({
                    text: action.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    action: action,
                    type: action.includes('block') ? 'Reject' : 'Default'
                })) || [],
                requiresUserAttention: data.data.severity === "critical"
            };
            
            this.addNotification(notification);
            
            if (data.data.severity === "critical") {
                MessageToast.show(`🛡️ SECURITY ALERT: ${data.data.message}`, {
                    duration: 5000,
                    at: "center center"
                });
            }
        },

        handleWorkflowApprovalNotification: function (data) {
            const notification = {
                id: this.generateNotificationId(),
                title: `Approval Required: ${data.data.workflowName || data.data.taskName}`,
                message: `A workflow task requires your approval: ${data.data.description || data.data.taskName}`,
                type: "warning",
                priority: data.data.priority || "medium",
                category: "workflow", 
                source: "Workflow Engine",
                sourceAgent: "workflow_engine",
                status: "unread",
                createdAt: new Date().toISOString(),
                metadata: {
                    taskId: data.data.taskId,
                    workflowId: data.data.workflowId,
                    assignee: data.data.assignee,
                    dueDate: data.data.dueDate,
                    formFields: data.data.formFields || []
                },
                actions: [
                    {
                        text: "Approve",
                        action: "approve_task",
                        type: "Accept"
                    },
                    {
                        text: "Reject", 
                        action: "reject_task",
                        type: "Reject"
                    },
                    {
                        text: "View Details",
                        action: "view_task_details",
                        type: "Default"
                    }
                ],
                requiresApproval: true,
                requiresUserAttention: true
            };
            
            this.addNotification(notification);
            
            MessageToast.show(`📋 Approval Required: ${notification.title}`, {
                duration: 3000,
                at: "center top"
            });
        },

        handleA2AMessageNotification: function (data) {
            const messageType = data.type.split('.').pop(); // 'request', 'response', 'notification'
            
            const notification = {
                id: this.generateNotificationId(),
                title: `A2A ${messageType.toUpperCase()}: ${data.data.fromAgent} → Human`,
                message: this.formatA2AMessageForHuman(data.data),
                type: messageType === 'request' ? 'warning' : 'info',
                priority: data.data.priority || 'medium',
                category: 'agent_message',
                messageType: messageType,
                source: data.data.fromAgent,
                sourceAgent: data.data.fromAgent,
                status: 'unread',
                createdAt: new Date().toISOString(),
                metadata: {
                    messageId: data.data.messageId,
                    fromAgent: data.data.fromAgent,
                    toAgent: data.data.toAgent || 'human',
                    originalA2AMessage: data.data.originalMessage,
                    translatedMessage: data.data.humanMessage,
                    conversationId: data.data.conversationId,
                    requiresResponse: messageType === 'request'
                },
                actions: messageType === 'request' ? [
                    {
                        text: "Reply",
                        action: "reply_to_agent",
                        type: "Emphasized"
                    },
                    {
                        text: "Delegate",
                        action: "delegate_to_agent", 
                        type: "Default"
                    }
                ] : [],
                requiresUserAttention: messageType === 'request',
                // Enhanced A2A specific properties
                a2aMessage: {
                    originalProtocol: data.data.protocol || 'A2A-v1.0',
                    messageStructure: data.data.structure,
                    semanticContext: data.data.context,
                    intentClassification: data.data.intent,
                    urgencyLevel: data.data.urgency,
                    expectedResponseTime: data.data.responseTimeExpected
                }
            };
            
            this.addNotification(notification);
            
            // Add to A2A conversation history
            this.addToA2AConversation(data.data);
            
            if (messageType === 'request') {
                MessageToast.show(`🤖 Agent ${data.data.fromAgent} needs your input`, {
                    duration: 3000,
                    at: "center top"
                });
            }
        },

        formatA2AMessageForHuman: function (messageData) {
            // This is where we bridge A2A protocol messages to human-readable format
            if (messageData.humanMessage) {
                return messageData.humanMessage;
            }
            
            // Fallback: try to interpret A2A message structure
            const originalMessage = messageData.originalMessage || messageData.message || {};
            
            if (originalMessage.intent) {
                switch (originalMessage.intent) {
                    case 'data_request':
                        return `Agent ${messageData.fromAgent} is requesting data: ${originalMessage.description || 'Data access needed for processing'}`;
                    case 'approval_request':
                        return `Agent ${messageData.fromAgent} needs approval for: ${originalMessage.action || 'an operation'}`;
                    case 'error_report':
                        return `Agent ${messageData.fromAgent} encountered an error: ${originalMessage.error || 'Processing failed'}`;
                    case 'status_update':
                        return `Agent ${messageData.fromAgent} reports: ${originalMessage.status || 'Status update'}`;
                    default:
                        return `Agent ${messageData.fromAgent} sent: ${JSON.stringify(originalMessage)}`;
                }
            }
            
            // Generic fallback
            return `Agent ${messageData.fromAgent} sent a message: ${JSON.stringify(originalMessage).substring(0, 100)}...`;
        },

        addNotification: function (notification) {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            notifications.unshift(notification); // Add to beginning for reverse chronological order
            
            this.oNotificationModel.setProperty("/notifications", notifications);
            
            // Update available agents list
            this.updateAvailableAgents(notification.sourceAgent);
        },

        updateAvailableAgents: function (agentId) {
            if (!agentId || agentId === 'human') return;
            
            const availableAgents = this.oNotificationModel.getProperty("/availableAgents");
            const existingAgent = availableAgents.find(agent => agent.agentId === agentId);
            
            if (!existingAgent) {
                availableAgents.push({
                    agentId: agentId,
                    name: this.formatAgentDisplayName(agentId),
                    lastSeen: new Date().toISOString()
                });
                
                this.oNotificationModel.setProperty("/availableAgents", availableAgents);
            }
        },

        formatAgentDisplayName: function (agentId) {
            // Convert agent IDs to human-readable names
            const agentNames = {
                'security_monitor': 'Security Monitor',
                'workflow_engine': 'Workflow Engine',
                'data_product_agent_0': 'Data Product Agent',
                'agent_1_standardization': 'Data Standardization Agent',
                'agent_2_ai_preparation': 'AI Preparation Agent',
                'reasoning_agent': 'Reasoning Agent',
                'sql_agent': 'SQL Agent'
            };
            
            return agentNames[agentId] || agentId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        },

        updateRealtimeStatistics: function () {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            
            // Calculate statistics
            const stats = {
                total: notifications.length,
                unread: 0,
                byType: { info: 0, warning: 0, error: 0, success: 0, system: 0 },
                byPriority: { low: 0, medium: 0, high: 0, critical: 0 },
                byCategory: { 
                    agent_message: 0, 
                    agent_crash: 0, 
                    security: 0, 
                    workflow: 0, 
                    transaction: 0, 
                    system: 0 
                },
                byStatus: { unread: 0, read: 0, dismissed: 0 },
                categoryChart: []
            };
            
            notifications.forEach(notification => {
                // Count by status
                stats.byStatus[notification.status]++;
                if (notification.status === 'unread') {
                    stats.unread++;
                }
                
                // Count by type
                if (stats.byType[notification.type] !== undefined) {
                    stats.byType[notification.type]++;
                }
                
                // Count by priority
                if (stats.byPriority[notification.priority] !== undefined) {
                    stats.byPriority[notification.priority]++;
                }
                
                // Count by category
                if (stats.byCategory[notification.category] !== undefined) {
                    stats.byCategory[notification.category]++;
                }
            });
            
            // Create chart data for categories
            Object.keys(stats.byCategory).forEach(category => {
                if (stats.byCategory[category] > 0) {
                    stats.categoryChart.push({
                        category: category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                        count: stats.byCategory[category],
                        color: this.getCategoryColor(category)
                    });
                }
            });
            
            this.oNotificationModel.setProperty("/stats", stats);
        },

        getCategoryColor: function (category) {
            const colors = {
                'agent_message': 'Good',
                'agent_crash': 'Error', 
                'security': 'Critical',
                'workflow': 'Neutral',
                'transaction': 'Good',
                'system': 'Neutral'
            };
            return colors[category] || 'Neutral';
        },

        applyCurrentFilters: function () {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            const filters = this.oFilterModel.getData();
            
            let filtered = notifications.filter(notification => {
                // Apply all filter criteria
                if (filters.unreadOnly && notification.status !== 'unread') return false;
                if (filters.highPriorityOnly && !['high', 'critical'].includes(notification.priority)) return false;
                if (filters.agentMessagesOnly && notification.category !== 'agent_message') return false;
                if (filters.securityOnly && notification.category !== 'security') return false;
                
                if (filters.status && notification.status !== filters.status) return false;
                if (filters.type && notification.type !== filters.type) return false;
                if (filters.priority && notification.priority !== filters.priority) return false;
                if (filters.category && notification.category !== filters.category) return false;
                if (filters.messageType && notification.messageType !== filters.messageType) return false;
                
                if (filters.sourceAgents.length > 0 && !filters.sourceAgents.includes(notification.sourceAgent)) return false;
                
                if (filters.searchText) {
                    const searchLower = filters.searchText.toLowerCase();
                    if (!notification.title.toLowerCase().includes(searchLower) &&
                        !notification.message.toLowerCase().includes(searchLower) &&
                        !notification.source.toLowerCase().includes(searchLower)) {
                        return false;
                    }
                }
                
                // Time range filter
                if (filters.timeRange && filters.timeRange !== 'all') {
                    const now = new Date();
                    const notificationTime = new Date(notification.createdAt);
                    const diffHours = (now - notificationTime) / (1000 * 60 * 60);
                    
                    switch (filters.timeRange) {
                        case '1h': if (diffHours > 1) return false; break;
                        case '24h': if (diffHours > 24) return false; break;
                        case '7d': if (diffHours > 168) return false; break;  
                        case '30d': if (diffHours > 720) return false; break;
                    }
                }
                
                return true;
            });
            
            this.oNotificationModel.setProperty("/filteredNotifications", filtered);
            this.oNotificationModel.setProperty("/filteredCount", filtered.length);
        },

        // Event Handlers
        onToggleUnreadFilter: function () {
            this.applyCurrentFilters();
        },

        onToggleHighPriorityFilter: function () {
            this.applyCurrentFilters();
        },

        onToggleAgentMessagesFilter: function () {
            this.applyCurrentFilters();
        },

        onToggleSecurityFilter: function () {
            this.applyCurrentFilters();
        },

        onReplyToAgent: function (oEvent) {
            const bindingContext = oEvent.getSource().getBindingContext("notificationModel");
            const notification = bindingContext.getObject();
            
            this.openA2AReplyDialog(notification);
        },

        openA2AReplyDialog: function (notification) {
            if (!this.oA2AReplyDialog) {
                this.oA2AReplyDialog = new Dialog({
                    title: "Reply to Agent: " + notification.sourceAgent,
                    contentWidth: "600px",
                    contentHeight: "400px",
                    resizable: true,
                    draggable: true,
                    content: [
                        new VBox({
                            items: [
                                new sap.m.Text({
                                    text: "Original A2A Message:",
                                    class: "sapUiMediumMarginBottom"
                                }).addStyleClass("sapUiMediumMarginBottom"),
                                new sap.m.Panel({
                                    content: new sap.m.FormattedText({
                                        htmlText: "{a2aReplyModel>/originalMessage}"
                                    }),
                                    class: "sapUiSmallMarginBottom"
                                }),
                                new sap.m.Label({
                                    text: "Your Reply (will be translated to A2A format):",
                                    class: "sapUiSmallMarginTop"
                                }),
                                new TextArea({
                                    value: "{a2aReplyModel>/replyText}",
                                    rows: 6,
                                    placeholder: "Type your response here... It will be automatically converted to A2A protocol format.",
                                    class: "sapUiMediumMarginTop"
                                })
                            ]
                        })
                    ],
                    buttons: [
                        new Button({
                            text: "Send A2A Reply",
                            type: "Emphasized",
                            press: this.onSendA2AReply.bind(this)
                        }),
                        new Button({
                            text: "Cancel",
                            press: this.onCloseA2AReplyDialog.bind(this)
                        })
                    ]
                });
            }
            
            // Set the data for the dialog
            const a2aReplyModel = new JSONModel({
                notification: notification,
                originalMessage: JSON.stringify(notification.metadata.originalA2AMessage, null, 2),
                replyText: "",
                targetAgent: notification.sourceAgent
            });
            
            this.oA2AReplyDialog.setModel(a2aReplyModel, "a2aReplyModel");
            this.oA2AReplyDialog.open();
        },

        onSendA2AReply: function () {
            const replyData = this.oA2AReplyDialog.getModel("a2aReplyModel").getData();
            
            if (!replyData.replyText.trim()) {
                MessageToast.show("Please enter a reply message");
                return;
            }
            
            // Convert human message to A2A format
            const a2aMessage = this.convertHumanMessageToA2A(replyData.replyText, replyData.targetAgent);
            
            // Send via WebSocket to the A2A bridge
            this.sendA2AMessage(a2aMessage);
            
            this.oA2AReplyDialog.close();
            MessageToast.show(`Reply sent to agent ${replyData.targetAgent}`);
        },

        convertHumanMessageToA2A: function (humanMessage, targetAgent) {
            // A2A Message Bridge: Convert human language to A2A protocol format
            return {
                version: "A2A-v1.0",
                messageId: this.generateMessageId(),
                timestamp: new Date().toISOString(),
                fromAgent: "human",
                toAgent: targetAgent,
                messageType: "response",
                protocol: {
                    intent: "human_response",
                    context: "notification_reply",
                    urgency: "normal"
                },
                payload: {
                    originalHumanMessage: humanMessage,
                    translatedContent: this.translateHumanMessageToA2AStructure(humanMessage),
                    responseType: "instruction",
                    requiresAcknowledgment: true
                },
                metadata: {
                    translatedAt: new Date().toISOString(),
                    translationMethod: "human-to-a2a-bridge",
                    sourceInterface: "notification_center"
                }
            };
        },

        translateHumanMessageToA2AStructure: function (humanMessage) {
            // Intelligent translation from human language to A2A structured format
            const lowerMessage = humanMessage.toLowerCase();
            
            // Intent detection
            let intent = "general_instruction";
            if (lowerMessage.includes("approve") || lowerMessage.includes("yes") || lowerMessage.includes("proceed")) {
                intent = "approval_granted";
            } else if (lowerMessage.includes("deny") || lowerMessage.includes("no") || lowerMessage.includes("stop")) {
                intent = "approval_denied";
            } else if (lowerMessage.includes("data") || lowerMessage.includes("information")) {
                intent = "data_instruction";
            } else if (lowerMessage.includes("error") || lowerMessage.includes("problem")) {
                intent = "error_acknowledgment";
            }
            
            return {
                intent: intent,
                instruction: humanMessage,
                parameters: this.extractParametersFromHumanMessage(humanMessage),
                confidence: 0.85, // Could be enhanced with NLP
                requiresConfirmation: lowerMessage.includes("please confirm") || lowerMessage.includes("verify")
            };
        },

        extractParametersFromHumanMessage: function (message) {
            // Simple parameter extraction - could be enhanced with NLP
            const parameters = {};
            
            // Extract common patterns
            const patterns = {
                amount: /\$?(\d+(?:\.\d{2})?)/g,
                percentage: /(\d+)%/g,
                date: /(\d{1,2}\/\d{1,2}\/\d{4})/g,
                time: /(\d{1,2}:\d{2}(?:\s?[AP]M)?)/gi
            };
            
            Object.keys(patterns).forEach(key => {
                const matches = message.match(patterns[key]);
                if (matches) {
                    parameters[key] = matches;
                }
            });
            
            return parameters;
        },

        sendA2AMessage: function (a2aMessage) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'a2a_message_bridge',
                    message: a2aMessage,
                    clientId: this.generateClientId()
                }));
                
                // Add to conversation history
                this.addToA2AConversation({
                    messageId: a2aMessage.messageId,
                    fromAgent: a2aMessage.fromAgent,
                    toAgent: a2aMessage.toAgent,
                    message: a2aMessage,
                    timestamp: a2aMessage.timestamp,
                    type: 'outgoing'
                });
            }
        },

        addToA2AConversation: function (messageData) {
            const conversations = this.oA2AModel.getProperty("/agentConversations");
            const agentId = messageData.fromAgent === 'human' ? messageData.toAgent : messageData.fromAgent;
            
            if (!conversations[agentId]) {
                conversations[agentId] = {
                    agentId: agentId,
                    agentName: this.formatAgentDisplayName(agentId),
                    messages: [],
                    lastActivity: messageData.timestamp,
                    status: 'active'
                };
            }
            
            conversations[agentId].messages.push({
                messageId: messageData.messageId,
                timestamp: messageData.timestamp,
                direction: messageData.fromAgent === 'human' ? 'outgoing' : 'incoming',
                content: messageData.message,
                humanMessage: messageData.humanMessage,
                status: 'sent'
            });
            
            conversations[agentId].lastActivity = messageData.timestamp;
            
            this.oA2AModel.setProperty("/agentConversations", conversations);
        },

        initializeA2AMessageBridge: function () {
            // Initialize the A2A message bridge service
            this.oA2AModel.setProperty("/bridgeStatus", "active");
            console.log("🤖 A2A Message Bridge initialized");
        },

        setupSmartAutoRefresh: function () {
            // Smart refresh based on activity
            const settings = this.oSettingsModel.getData();
            if (settings.autoRefresh) {
                this.refreshInterval = setInterval(() => {
                    if (document.visibilityState === 'visible') {
                        this.loadNotifications();
                    }
                }, settings.refreshInterval);
            }
        },

        setupRealtimeStatistics: function () {
            // Update statistics every 5 seconds
            this.statisticsInterval = setInterval(() => {
                this.updateLastUpdated();
            }, 5000);
        },

        updateLastUpdated: function () {
            this.oNotificationModel.setProperty("/lastUpdated", new Date().toLocaleString());
        },

        // Utility methods
        generateNotificationId: function () {
            return 'notif_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },

        generateClientId: function () {
            return 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },

        generateMessageId: function () {
            return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },

        getWebSocketUrl: function () {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const port = process.env.WS_PORT || '8080';
            return `${protocol}//${host}:${port}/notifications`;
        },

        // Formatters (will be used by the view)
        formatEnhancedNotificationIcon: function (type, priority, category) {
            if (category === 'agent_crash') return 'sap-icon://error';
            if (category === 'security') return 'sap-icon://shield';
            if (category === 'workflow') return 'sap-icon://approvals';
            if (category === 'agent_message') return 'sap-icon://collaborate';
            
            const icons = {
                error: 'sap-icon://error',
                warning: 'sap-icon://alert',
                info: 'sap-icon://information',
                success: 'sap-icon://accept'
            };
            return icons[type] || 'sap-icon://bell';
        },

        formatEnhancedNotificationColor: function (priority, type) {
            if (priority === 'critical') return 'Critical';
            if (priority === 'high') return 'Negative';
            if (priority === 'medium') return 'Neutral';
            if (type === 'success') return 'Positive';
            return 'Neutral';
        },

        formatRelativeTime: function (timestamp) {
            const now = new Date();
            const time = new Date(timestamp);
            const diff = now - time;
            
            const minutes = Math.floor(diff / 60000);
            const hours = Math.floor(diff / 3600000);
            const days = Math.floor(diff / 86400000);
            
            if (minutes < 1) return 'Just now';
            if (minutes < 60) return `${minutes}m ago`;
            if (hours < 24) return `${hours}h ago`;
            return `${days}d ago`;
        },

        formatAbsoluteTime: function (timestamp) {
            return new Date(timestamp).toLocaleString();
        },

        onDestroy: function () {
            if (this.ws) {
                this.ws.close();
            }
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
            }
            if (this.statisticsInterval) {
                clearInterval(this.statisticsInterval);
            }
        }
    });
});