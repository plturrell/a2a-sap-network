sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment",
    "sap/ui/core/BusyIndicator"
], function (Controller, MessageBox, MessageToast, JSONModel, Fragment, BusyIndicator) {
    "use strict";

    return Controller.extend("a2a.portal.controller.A2ANetworkManager", {
        
        onInit: function () {
            // Initialize models
            this._initializeModels();
            
            // Connect to A2A Network on initialization
            this._connectToA2ANetwork();
            
            // Setup WebSocket for real-time updates
            this._setupWebSocket();
            
            // Load initial data
            this._loadNetworkData();
        },

        _initializeModels: function() {
            // Network connection model
            var oNetworkModel = new JSONModel({
                connected: false,
                network: "mainnet",
                chainId: null,
                blockNumber: null,
                address: null,
                contracts: []
            });
            this.getView().setModel(oNetworkModel, "network");

            // Agents model
            var oAgentsModel = new JSONModel({
                agents: [],
                totalAgents: 0,
                selectedAgent: null,
                loading: false
            });
            this.getView().setModel(oAgentsModel, "agents");

            // Messages model
            var oMessagesModel = new JSONModel({
                messages: [],
                unreadCount: 0
            });
            this.getView().setModel(oMessagesModel, "messages");

            // Webhooks model
            var oWebhooksModel = new JSONModel({
                subscriptions: [],
                eventTypes: [
                    { key: "agent_registered", text: "Agent Registered" },
                    { key: "agent_updated", text: "Agent Updated" },
                    { key: "agent_status_changed", text: "Agent Status Changed" },
                    { key: "message_sent", text: "Message Sent" },
                    { key: "message_received", text: "Message Received" }
                ]
            });
            this.getView().setModel(oWebhooksModel, "webhooks");
        },

        _connectToA2ANetwork: function() {
            BusyIndicator.show(0);
            
            // Get connection config from settings
            var sPrivateKey = localStorage.getItem("a2a_private_key");
            var sNetwork = localStorage.getItem("a2a_network") || "mainnet";
            
            jQuery.ajax({
                url: "/api/a2a-network/connect",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    network: sNetwork,
                    private_key: sPrivateKey
                }),
                success: function(data) {
                    BusyIndicator.hide();
                    
                    // Update network model
                    var oNetworkModel = this.getView().getModel("network");
                    oNetworkModel.setData(data);
                    
                    MessageToast.show("Connected to A2A Network");
                    
                    // Load agents after connection
                    this._loadAgents();
                }.bind(this),
                error: function(xhr) {
                    BusyIndicator.hide();
                    MessageBox.error("Failed to connect to A2A Network: " + xhr.responseJSON.detail);
                }
            });
        },

        _setupWebSocket: function() {
            // Setup real-time updates via WebSocket
            var wsUrl = window.location.protocol === "https:" ? "wss://" : "ws://";
            wsUrl += window.location.host + "/ws";
            
            this._ws = new WebSocket(wsUrl);
            
            this._ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                this._handleWebSocketMessage(data);
            }.bind(this);
            
            this._ws.onerror = function(error) {
                console.error("WebSocket error:", error);
            };
        },

        _handleWebSocketMessage: function(data) {
            switch(data.type) {
                case "agent_registered":
                case "agent_updated":
                case "agent_status_changed":
                    // Reload agents
                    this._loadAgents();
                    MessageToast.show("Agent " + data.type.replace("_", " "));
                    break;
                    
                case "message_sent":
                case "message_received":
                    // Update messages
                    this._loadMessages();
                    break;
            }
        },

        _loadNetworkData: function() {
            // Load network status
            jQuery.ajax({
                url: "/api/a2a-network/status",
                type: "GET",
                success: function(data) {
                    if (data.connected) {
                        var oNetworkModel = this.getView().getModel("network");
                        oNetworkModel.setProperty("/connected", true);
                    }
                }.bind(this)
            });
        },

        _loadAgents: function() {
            var oAgentsModel = this.getView().getModel("agents");
            oAgentsModel.setProperty("/loading", true);
            
            jQuery.ajax({
                url: "/api/a2a-network/agents?limit=100",
                type: "GET",
                success: function(data) {
                    oAgentsModel.setProperty("/agents", data.data.agents);
                    oAgentsModel.setProperty("/totalAgents", data.data.total);
                    oAgentsModel.setProperty("/loading", false);
                }.bind(this),
                error: function() {
                    oAgentsModel.setProperty("/loading", false);
                    MessageBox.error("Failed to load agents");
                }
            });
        },

        onRegisterAgent: function() {
            if (!this._oRegisterDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.RegisterAgentDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oRegisterDialog = oDialog;
                    this.getView().addDependent(this._oRegisterDialog);
                    
                    // Set initial model
                    var oModel = new JSONModel({
                        name: "",
                        description: "",
                        endpoint: "",
                        capabilities: {
                            messaging: true,
                            workflow: false,
                            analytics: false,
                            ai: false
                        }
                    });
                    this._oRegisterDialog.setModel(oModel, "newAgent");
                    
                    this._oRegisterDialog.open();
                }.bind(this));
            } else {
                this._oRegisterDialog.open();
            }
        },

        onRegisterAgentConfirm: function() {
            var oModel = this._oRegisterDialog.getModel("newAgent");
            var oData = oModel.getData();
            
            // Validate
            if (!oData.name || !oData.description || !oData.endpoint) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            BusyIndicator.show(0);
            
            jQuery.ajax({
                url: "/api/a2a-network/agents/register",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    BusyIndicator.hide();
                    MessageBox.success("Agent registered successfully!\nAgent ID: " + data.agent_id);
                    this._oRegisterDialog.close();
                    this._loadAgents();
                }.bind(this),
                error: function(xhr) {
                    BusyIndicator.hide();
                    MessageBox.error("Registration failed: " + xhr.responseJSON.detail);
                }
            });
        },

        onRegisterAgentCancel: function() {
            this._oRegisterDialog.close();
        },

        onAgentSelect: function(oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oAgent = oItem.getBindingContext("agents").getObject();
            
            // Load agent profile
            BusyIndicator.show(0);
            
            jQuery.ajax({
                url: "/api/a2a-network/agents/" + oAgent.id + "/profile",
                type: "GET",
                success: function(data) {
                    BusyIndicator.hide();
                    
                    var oAgentsModel = this.getView().getModel("agents");
                    oAgentsModel.setProperty("/selectedAgent", data);
                    
                    // Navigate to agent detail view
                    this._showAgentDetail(data);
                }.bind(this),
                error: function() {
                    BusyIndicator.hide();
                    MessageBox.error("Failed to load agent profile");
                }
            });
        },

        _showAgentDetail: function(oAgent) {
            if (!this._oAgentDetailDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.AgentDetailDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oAgentDetailDialog = oDialog;
                    this.getView().addDependent(this._oAgentDetailDialog);
                    
                    var oModel = new JSONModel(oAgent);
                    this._oAgentDetailDialog.setModel(oModel, "agentDetail");
                    
                    this._oAgentDetailDialog.open();
                }.bind(this));
            } else {
                var oModel = this._oAgentDetailDialog.getModel("agentDetail");
                oModel.setData(oAgent);
                this._oAgentDetailDialog.open();
            }
        },

        onCloseAgentDetail: function() {
            this._oAgentDetailDialog.close();
        },

        onToggleAgentStatus: function() {
            var oModel = this._oAgentDetailDialog.getModel("agentDetail");
            var oAgent = oModel.getData();
            
            BusyIndicator.show(0);
            
            jQuery.ajax({
                url: "/api/a2a-network/agents/" + oAgent.id + "/status",
                type: "PATCH",
                contentType: "application/json",
                data: JSON.stringify({
                    is_active: !oAgent.isActive
                }),
                success: function(data) {
                    BusyIndicator.hide();
                    MessageToast.show("Agent status updated");
                    oModel.setProperty("/isActive", !oAgent.isActive);
                    this._loadAgents();
                }.bind(this),
                error: function() {
                    BusyIndicator.hide();
                    MessageBox.error("Failed to update agent status");
                }
            });
        },

        onSendMessage: function() {
            var oAgent = this._oAgentDetailDialog.getModel("agentDetail").getData();
            
            if (!this._oMessageDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.SendMessageDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oMessageDialog = oDialog;
                    this.getView().addDependent(this._oMessageDialog);
                    
                    var oModel = new JSONModel({
                        recipientId: oAgent.id,
                        recipientName: oAgent.name,
                        content: "",
                        messageType: "text"
                    });
                    this._oMessageDialog.setModel(oModel, "message");
                    
                    this._oMessageDialog.open();
                }.bind(this));
            } else {
                var oModel = this._oMessageDialog.getModel("message");
                oModel.setProperty("/recipientId", oAgent.id);
                oModel.setProperty("/recipientName", oAgent.name);
                oModel.setProperty("/content", "");
                this._oMessageDialog.open();
            }
        },

        onSendMessageConfirm: function() {
            var oModel = this._oMessageDialog.getModel("message");
            var oData = oModel.getData();
            
            if (!oData.content) {
                MessageBox.error("Please enter a message");
                return;
            }
            
            BusyIndicator.show(0);
            
            jQuery.ajax({
                url: "/api/a2a-network/messages/send",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    recipient_id: oData.recipientId,
                    content: oData.content,
                    message_type: oData.messageType
                }),
                success: function(data) {
                    BusyIndicator.hide();
                    MessageToast.show("Message sent successfully");
                    this._oMessageDialog.close();
                }.bind(this),
                error: function() {
                    BusyIndicator.hide();
                    MessageBox.error("Failed to send message");
                }
            });
        },

        onSendMessageCancel: function() {
            this._oMessageDialog.close();
        },

        onSearchAgents: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            
            if (!sQuery) {
                this._loadAgents();
                return;
            }
            
            var oAgentsModel = this.getView().getModel("agents");
            oAgentsModel.setProperty("/loading", true);
            
            jQuery.ajax({
                url: "/api/a2a-network/agents?search=" + encodeURIComponent(sQuery),
                type: "GET",
                success: function(data) {
                    oAgentsModel.setProperty("/agents", data.data.agents);
                    oAgentsModel.setProperty("/loading", false);
                }.bind(this),
                error: function() {
                    oAgentsModel.setProperty("/loading", false);
                }
            });
        },

        onManageWebhooks: function() {
            if (!this._oWebhooksDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.WebhooksDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oWebhooksDialog = oDialog;
                    this.getView().addDependent(this._oWebhooksDialog);
                    
                    this._loadWebhooks();
                    this._oWebhooksDialog.open();
                }.bind(this));
            } else {
                this._loadWebhooks();
                this._oWebhooksDialog.open();
            }
        },

        _loadWebhooks: function() {
            jQuery.ajax({
                url: "/api/a2a-network/webhooks/subscriptions",
                type: "GET",
                success: function(data) {
                    var oWebhooksModel = this.getView().getModel("webhooks");
                    oWebhooksModel.setProperty("/subscriptions", data.subscriptions);
                }.bind(this)
            });
        },

        onAddWebhook: function() {
            if (!this._oAddWebhookDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.AddWebhookDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oAddWebhookDialog = oDialog;
                    this.getView().addDependent(this._oAddWebhookDialog);
                    
                    var oModel = new JSONModel({
                        event_type: "agent_registered",
                        webhook_url: "",
                        active: true
                    });
                    this._oAddWebhookDialog.setModel(oModel, "newWebhook");
                    
                    this._oAddWebhookDialog.open();
                }.bind(this));
            } else {
                this._oAddWebhookDialog.open();
            }
        },

        onAddWebhookConfirm: function() {
            var oModel = this._oAddWebhookDialog.getModel("newWebhook");
            var oData = oModel.getData();
            
            if (!oData.webhook_url) {
                MessageBox.error("Please enter a webhook URL");
                return;
            }
            
            BusyIndicator.show(0);
            
            jQuery.ajax({
                url: "/api/a2a-network/webhooks/subscribe",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    BusyIndicator.hide();
                    MessageToast.show("Webhook subscription created");
                    this._oAddWebhookDialog.close();
                    this._loadWebhooks();
                }.bind(this),
                error: function() {
                    BusyIndicator.hide();
                    MessageBox.error("Failed to create webhook subscription");
                }
            });
        },

        onAddWebhookCancel: function() {
            this._oAddWebhookDialog.close();
        },

        onDeleteWebhook: function(oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oWebhook = oItem.getBindingContext("webhooks").getObject();
            
            MessageBox.confirm("Delete this webhook subscription?", {
                onClose: function(sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        jQuery.ajax({
                            url: "/api/a2a-network/webhooks/subscriptions/" + oWebhook.subscription_id,
                            type: "DELETE",
                            success: function() {
                                MessageToast.show("Webhook deleted");
                                this._loadWebhooks();
                            }.bind(this),
                            error: function() {
                                MessageBox.error("Failed to delete webhook");
                            }
                        });
                    }
                }.bind(this)
            });
        },

        onCloseWebhooks: function() {
            this._oWebhooksDialog.close();
        },

        onRefreshAgents: function() {
            this._loadAgents();
        },

        onNetworkSettings: function() {
            if (!this._oNetworkSettingsDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.portal.fragment.NetworkSettingsDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oNetworkSettingsDialog = oDialog;
                    this.getView().addDependent(this._oNetworkSettingsDialog);
                    
                    var oModel = new JSONModel({
                        network: localStorage.getItem("a2a_network") || "mainnet",
                        privateKey: localStorage.getItem("a2a_private_key") || "",
                        rpcUrl: localStorage.getItem("a2a_rpc_url") || ""
                    });
                    this._oNetworkSettingsDialog.setModel(oModel, "settings");
                    
                    this._oNetworkSettingsDialog.open();
                }.bind(this));
            } else {
                this._oNetworkSettingsDialog.open();
            }
        },

        onSaveNetworkSettings: function() {
            var oModel = this._oNetworkSettingsDialog.getModel("settings");
            var oData = oModel.getData();
            
            // Save to localStorage
            localStorage.setItem("a2a_network", oData.network);
            if (oData.privateKey) {
                localStorage.setItem("a2a_private_key", oData.privateKey);
            }
            if (oData.rpcUrl) {
                localStorage.setItem("a2a_rpc_url", oData.rpcUrl);
            }
            
            this._oNetworkSettingsDialog.close();
            
            // Reconnect with new settings
            this._connectToA2ANetwork();
        },

        onCancelNetworkSettings: function() {
            this._oNetworkSettingsDialog.close();
        },

        onExit: function() {
            // Cleanup WebSocket
            if (this._ws) {
                this._ws.close();
            }
        },

        // Formatters
        formatDate: function(date) {
            if (!date) return "";
            const d = new Date(date);
            return d.toLocaleDateString();
        },

        formatTimestamp: function(timestamp) {
            if (!timestamp) return "";
            const d = new Date(timestamp);
            return d.toLocaleString();
        },

        onNavBack: function() {
            const oHistory = sap.ui.core.routing.History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();
            
            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getOwnerComponent().getRouter().navTo("projects");
            }
        }
    });
});