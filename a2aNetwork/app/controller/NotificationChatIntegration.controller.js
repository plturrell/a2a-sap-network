sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/m/Dialog",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Text",
    "sap/m/TextArea",
    "sap/m/Button"
], function (Controller, MessageToast, MessageBox, Fragment, Dialog, VBox, HBox, Text, TextArea, Button) {
    "use strict";

    return Controller.extend("a2a.network.controller.NotificationChatIntegration", {

        /**
         * Open contextual chat based on notification
         */
        onOpenContextualChat: function(oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("notificationModel");
            if (!oBindingContext) {
                MessageToast.show("Unable to get notification context");
                return;
            }

            const oNotification = oBindingContext.getObject();
            const oChatContext = this.buildChatContext(oNotification);
            
            this.openChatDialog(oChatContext);
        },

        /**
         * Build comprehensive chat context from notification
         */
        buildChatContext: function(oNotification) {
            return {
                // Core notification info
                notificationId: oNotification.id,
                title: oNotification.title,
                type: oNotification.type,
                priority: oNotification.priority,
                status: oNotification.status,
                
                // A2A specific context
                fromAgent: oNotification.fromAgent,
                toAgent: "human", // User is the target
                conversationId: this.generateConversationId(),
                
                // Rich context for AI agent
                context: {
                    triggeredFrom: "notification_center",
                    userIntent: this.inferUserIntent(oNotification),
                    systemState: this.getCurrentSystemState(),
                    relatedAgents: this.getRelatedAgents(oNotification),
                    historicalContext: this.getNotificationHistory(oNotification.type)
                },
                
                // Pre-built conversation starter
                initialMessage: this.generateContextualStarter(oNotification),
                
                // Suggested follow-up questions
                suggestedQuestions: this.getSuggestedQuestions(oNotification)
            };
        },

        /**
         * Generate conversation ID for tracking
         */
        generateConversationId: function() {
            return "chat_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Infer user intent from notification type and priority
         */
        inferUserIntent: function(oNotification) {
            const intentMap = {
                "agent_crash": "troubleshooting",
                "security_alert": "investigation", 
                "workflow_approval": "decision_support",
                "system_health": "monitoring",
                "performance_alert": "optimization"
            };
            
            return intentMap[oNotification.type] || "general_assistance";
        },

        /**
         * Generate contextual conversation starter
         */
        generateContextualStarter: function(oNotification) {
            const starters = {
                "agent_crash": `🚨 Agent "${oNotification.fromAgent}" has crashed. I need help understanding what went wrong and how to get it back online safely.`,
                
                "security_alert": `🔒 Security alert detected: "${oNotification.title}". Can you analyze the threat level and recommend immediate actions?`,
                
                "workflow_approval": `📋 Approval needed for: "${oNotification.title}". I need context about this request and guidance on whether to approve it.`,
                
                "system_health": `⚡ System health issue: "${oNotification.title}". Help me understand the impact and required actions.`,
                
                "performance_alert": `📈 Performance issue detected: "${oNotification.title}". I need analysis and optimization recommendations.`
            };

            return starters[oNotification.type] || 
                   `I need assistance with this notification: "${oNotification.title}". Can you help me understand what I should do?`;
        },

        /**
         * Get suggested follow-up questions
         */
        getSuggestedQuestions: function(oNotification) {
            const suggestions = {
                "agent_crash": [
                    "What caused this agent to crash?",
                    "How can I prevent this in the future?", 
                    "Is there any data loss?",
                    "What's the recovery procedure?"
                ],
                
                "security_alert": [
                    "How critical is this security threat?",
                    "What immediate actions should I take?",
                    "Are other systems affected?",
                    "How do I contain this issue?"
                ],
                
                "workflow_approval": [
                    "Who initiated this request?",
                    "What are the business implications?",
                    "Are there any compliance concerns?",
                    "What happens if I decline?"
                ]
            };

            return suggestions[oNotification.type] || [
                "What does this notification mean?",
                "What action should I take?",
                "Is this urgent?",
                "How do I resolve this?"
            ];
        },

        /**
         * Open chat dialog with rich context
         */
        openChatDialog: function(oChatContext) {
            if (!this._chatDialog) {
                this._chatDialog = new Dialog({
                    title: "💬 Contextual AI Assistant",
                    contentWidth: "60rem",
                    contentHeight: "40rem",
                    resizable: true,
                    draggable: true,
                    
                    content: [
                        new VBox({
                            class: "chatContainer sapUiMediumMargin",
                            items: [
                                // Context header
                                new HBox({
                                    class: "chatContextHeader",
                                    items: [
                                        new Text({
                                            text: "🎯 Context: {chatModel>/title}",
                                            class: "contextTitle"
                                        }),
                                        new Text({
                                            text: "Priority: {chatModel>/priority}",
                                            class: "contextPriority"
                                        })
                                    ]
                                }),
                                
                                // Chat messages area  
                                new VBox({
                                    class: "chatMessages",
                                    items: {
                                        path: "chatModel>/messages",
                                        template: new HBox({
                                            class: "chatMessage {path: 'chatModel>sender', formatter: '.formatMessageClass'}",
                                            items: [
                                                new Text({
                                                    text: "{chatModel>message}",
                                                    class: "messageText"
                                                })
                                            ]
                                        })
                                    }
                                }),
                                
                                // Suggested questions
                                new VBox({
                                    class: "suggestedQuestions",
                                    visible: "{= ${chatModel>/messages}.length === 1}",
                                    items: {
                                        path: "chatModel>/suggestedQuestions",
                                        template: new Button({
                                            text: "{chatModel>}",
                                            type: "Transparent",
                                            press: "onSelectSuggestedQuestion",
                                            class: "suggestionButton"
                                        })
                                    }
                                }),
                                
                                // Input area
                                new HBox({
                                    class: "chatInput",
                                    items: [
                                        new TextArea({
                                            value: "{chatModel>/currentMessage}",
                                            placeholder: "Ask me anything about this notification...",
                                            rows: 3,
                                            width: "100%",
                                            submit: "onSendMessage"
                                        }),
                                        new VBox({
                                            items: [
                                                new Button({
                                                    icon: "sap-icon://paper-plane",
                                                    type: "Emphasized", 
                                                    press: "onSendMessage",
                                                    tooltip: "Send Message"
                                                }),
                                                new Button({
                                                    icon: "sap-icon://action",
                                                    type: "Default",
                                                    press: "onShowQuickActions",
                                                    tooltip: "Quick Actions"
                                                })
                                            ]
                                        })
                                    ]
                                })
                            ]
                        })
                    ],
                    
                    beginButton: new Button({
                        text: "Resolve & Close",
                        type: "Accept",
                        press: "onResolveAndClose"
                    }),
                    
                    endButton: new Button({
                        text: "Close Chat", 
                        press: "onCloseChat"
                    })
                });

                this.getView().addDependent(this._chatDialog);
            }

            // Set chat context model
            this._chatDialog.setModel(new sap.ui.model.json.JSONModel(oChatContext), "chatModel");
            
            // Initialize first message from AI
            this.initializeChatConversation(oChatContext);
            
            this._chatDialog.open();
        },

        /**
         * Initialize chat with AI's first response
         */
        initializeChatConversation: function(oChatContext) {
            const oModel = this._chatDialog.getModel("chatModel");
            const aMessages = [{
                sender: "ai",
                message: `Hello! I can see you need help with: "${oChatContext.title}"\n\n${oChatContext.initialMessage}\n\nI have access to your system context and can help you understand what's happening and what actions you should take. What would you like to know?`,
                timestamp: new Date().toISOString()
            }];
            
            oModel.setProperty("/messages", aMessages);
            oModel.setProperty("/currentMessage", "");
        },

        /**
         * Handle sending messages to AI agent
         */
        onSendMessage: function() {
            const oModel = this._chatDialog.getModel("chatModel");
            const sCurrentMessage = oModel.getProperty("/currentMessage");
            
            if (!sCurrentMessage.trim()) {
                MessageToast.show("Please enter a message");
                return;
            }

            // Add user message
            const aMessages = oModel.getProperty("/messages");
            aMessages.push({
                sender: "user",
                message: sCurrentMessage,
                timestamp: new Date().toISOString()
            });

            // Clear input
            oModel.setProperty("/currentMessage", "");
            oModel.setProperty("/messages", aMessages);

            // Send to AI agent and get response
            this.sendToAIAgent(sCurrentMessage, oModel.getData());
        },

        /**
         * Send message to AI agent via A2A bridge
         */
        sendToAIAgent: function(sMessage, oChatContext) {
            // Build A2A message for the chat agent
            const oA2AMessage = {
                messageId: this.generateMessageId(),
                messageType: "human_chat_request",
                from: "human_user",
                to: "chat_assistant_agent", 
                category: "interactive_support",
                urgency: oChatContext.priority || "medium",
                description: sMessage,
                context: {
                    conversation_id: oChatContext.conversationId,
                    notification_context: oChatContext,
                    user_intent: oChatContext.context.userIntent,
                    system_state: oChatContext.context.systemState
                }
            };

            // Send via A2A message bridge
            jQuery.ajax({
                url: "/a2a/route", // A2A Communication Router endpoint
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    message: oA2AMessage,
                    from_agent: "notification_center_ui",
                    context: oChatContext.context
                }),
                success: (response) => {
                    this.handleAIResponse(response);
                },
                error: (error) => {
                    MessageToast.show("Error communicating with AI agent");
                    console.error("Chat AI error:", error);
                }
            });
        },

        /**
         * Handle AI agent response
         */
        handleAIResponse: function(response) {
            const oModel = this._chatDialog.getModel("chatModel");
            const aMessages = oModel.getProperty("/messages");
            
            // Add AI response
            aMessages.push({
                sender: "ai",
                message: response.ai_response || "I'm processing your request...",
                timestamp: new Date().toISOString(),
                actions: response.suggested_actions || []
            });

            oModel.setProperty("/messages", aMessages);
            
            // Auto-scroll to latest message
            this.scrollToLatestMessage();
        },

        /**
         * Handle suggested question selection
         */
        onSelectSuggestedQuestion: function(oEvent) {
            const sQuestion = oEvent.getSource().getText();
            const oModel = this._chatDialog.getModel("chatModel");
            
            oModel.setProperty("/currentMessage", sQuestion);
            this.onSendMessage();
        },

        /**
         * Resolve notification and close chat
         */
        onResolveAndClose: function() {
            const oChatContext = this._chatDialog.getModel("chatModel").getData();
            
            // Mark notification as resolved
            this.markNotificationAsResolved(oChatContext.notificationId);
            
            MessageToast.show("Notification resolved and chat closed");
            this.onCloseChat();
        },

        /**
         * Close chat dialog
         */
        onCloseChat: function() {
            this._chatDialog.close();
        },

        /**
         * Get current system state for context
         */
        getCurrentSystemState: function() {
            return {
                timestamp: new Date().toISOString(),
                active_agents: this.getActiveAgentsCount(),
                system_load: this.getSystemLoad(),
                recent_alerts: this.getRecentAlertsCount()
            };
        },

        /**
         * Get related agents for context
         */
        getRelatedAgents: function(oNotification) {
            // Return agents related to this notification
            return [oNotification.fromAgent].filter(Boolean);
        },

        /**
         * Generate unique message ID
         */
        generateMessageId: function() {
            return "msg_" + Date.now() + "_" + Math.random().toString(36).substr(2, 5);
        }
    });
});