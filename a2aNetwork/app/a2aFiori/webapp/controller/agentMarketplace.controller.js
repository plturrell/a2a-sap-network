/**
 * A2A Agent Marketplace Controller
 * Handles agent discovery, service requests, and agent management
 */

sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/Sorter"
], (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Fragment, Sorter) => {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.AgentMarketplace", {

        onInit() {
            BaseController.prototype.onInit.apply(this, arguments);

            // Initialize models
            this._initializeModels();

            // Load marketplace data
            this._loadMarketplaceData();

            // Set up real-time updates
            this._setupRealtimeUpdates();

            // Initialize WebSocket connection
            this._initializeWebSocket();
        },

        _initializeModels() {
            // Agent marketplace model
            this.oMarketplaceModel = new JSONModel({
                agents: [],
                userRequests: [],
                userAgents: [],
                categories: [],
                capabilities: [],
                requestStatuses: [],
                sortOptions: [],
                stats: {
                    totalAgents: 0,
                    activeServices: 0,
                    totalTransactions: 0,
                    averageRating: 0
                },
                requestStats: {
                    pending: 0,
                    active: 0,
                    completed: 0,
                    pendingCount: 0,
                    activeCount: 0,
                    completedCount: 0
                },
                filters: {
                    category: "",
                    capabilities: [],
                    priceMin: 0,
                    priceMax: 1000,
                    minRating: 0,
                    availableOnly: true
                },
                selectedAgent: null,
                serviceRequest: {
                    serviceType: "",
                    parameters: "",
                    deadline: null,
                    estimatedCost: 0,
                    escrowAmount: 0
                },
                newAgent: {
                    name: "",
                    description: "",
                    category: "",
                    endpoint: "",
                    capabilities: [],
                    pricing: 0,
                    minimumStake: 0
                },
                viewMode: "grid",
                sortBy: "rating"
            });
            this.getView().setModel(this.oMarketplaceModel, "marketplace");

            // Update UI model
            this.oUIModel.setProperty("/agentMarketplaceView", "browse");
        },

        _loadMarketplaceData() {
            this.showSkeletonLoading(this.getResourceBundle().getText("agentMarketplace.loading"));

            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";

            Promise.all([
                this._loadAgents(),
                this._loadUserRequests(),
                this._loadUserAgents(),
                this._loadStaticData()
            ]).then(() => {
                this._updateStats();
                this.hideLoading();
            }).catch(error => {
                console.error("Failed to load agent marketplace data:", error);
                this._loadFallbackData();
                this.hideLoading();
                this.showErrorMessage(this.getResourceBundle().getText("agentMarketplace.loadError"));
            });
        },

        _loadAgents() {
            return new Promise((resolve) => {
                // Mock agent data - replace with actual API call
                const agents = [
                    {
                        id: "agent_001",
                        name: "Agent-2 (AI Preparation)",
                        owner: "A2A Platform",
                        description: "Advanced AI-powered document preparation and enhancement",
                        category: "AI/ML",
                        capabilities: ["Document Processing", "AI Enhancement", "Content Analysis"],
                        endpoint: "http://localhost:8002",
                        pricing: 25.0,
                        rating: 4.8,
                        status: "Available",
                        lastActivity: new Date().toISOString(),
                        totalEarnings: 12450.00,
                        activeServices: 15,
                        successRate: 0.96
                    },
                    {
                        id: "agent_002",
                        name: "Agent-3 (Vector Processing)",
                        owner: "A2A Platform",
                        description: "High-performance vector processing and similarity search",
                        category: "Analytics",
                        capabilities: ["Vector Search", "Similarity Analysis", "Data Processing"],
                        endpoint: "http://localhost:8003",
                        pricing: 35.0,
                        rating: 4.7,
                        status: "Available",
                        lastActivity: new Date().toISOString(),
                        totalEarnings: 18900.00,
                        activeServices: 23,
                        successRate: 0.94
                    },
                    {
                        id: "agent_003",
                        name: "Calculator Agent",
                        owner: "A2A Platform",
                        description: "Complex mathematical calculations and financial modeling",
                        category: "Finance",
                        capabilities: ["Mathematical Modeling", "Financial Analysis", "Statistical Computing"],
                        endpoint: "http://localhost:8004",
                        pricing: 15.0,
                        rating: 4.9,
                        status: "Available",
                        lastActivity: new Date().toISOString(),
                        totalEarnings: 8750.00,
                        activeServices: 31,
                        successRate: 0.98
                    },
                    {
                        id: "agent_004",
                        name: "Security Agent",
                        owner: "A2A Platform",
                        description: "Security analysis and threat detection services",
                        category: "Security",
                        capabilities: ["Threat Detection", "Security Analysis", "Risk Assessment"],
                        endpoint: "http://localhost:8005",
                        pricing: 50.0,
                        rating: 4.6,
                        status: "Busy",
                        lastActivity: new Date().toISOString(),
                        totalEarnings: 22100.00,
                        activeServices: 8,
                        successRate: 0.92
                    }
                ];

                this.oMarketplaceModel.setProperty("/agents", agents);
                resolve(agents);
            });
        },

        _loadUserRequests() {
            return new Promise((resolve) => {
                // Mock user requests data
                const userRequests = [
                    {
                        requestId: "REQ_001",
                        agentName: "Agent-2 (AI Preparation)",
                        serviceName: "Document Enhancement",
                        status: "Active",
                        requestDate: new Date(Date.now() - 86400000).toISOString(),
                        completionDate: null,
                        cost: 25.0,
                        rated: false
                    },
                    {
                        requestId: "REQ_002",
                        agentName: "Calculator Agent",
                        serviceName: "Financial Modeling",
                        status: "Completed",
                        requestDate: new Date(Date.now() - 259200000).toISOString(),
                        completionDate: new Date(Date.now() - 172800000).toISOString(),
                        cost: 15.0,
                        rated: true
                    },
                    {
                        requestId: "REQ_003",
                        agentName: "Agent-3 (Vector Processing)",
                        serviceName: "Similarity Search",
                        status: "Pending",
                        requestDate: new Date().toISOString(),
                        completionDate: null,
                        cost: 35.0,
                        rated: false
                    }
                ];

                this.oMarketplaceModel.setProperty("/userRequests", userRequests);
                this._updateRequestStats(userRequests);
                resolve(userRequests);
            });
        },

        _loadUserAgents() {
            return new Promise((resolve) => {
                // Mock user agents data (agents owned by current user)
                const userAgents = [
                    {
                        id: "user_agent_001",
                        name: "Custom Analytics Agent",
                        description: "Specialized analytics for e-commerce data",
                        category: "Analytics",
                        status: "Available",
                        rating: 4.5,
                        totalEarnings: 3450.00,
                        activeServices: 12,
                        lastActivity: new Date().toISOString()
                    }
                ];

                this.oMarketplaceModel.setProperty("/userAgents", userAgents);
                resolve(userAgents);
            });
        },

        _loadStaticData() {
            return new Promise((resolve) => {
                const staticData = {
                    categories: [
                        { key: "", text: "All Categories" },
                        { key: "ai-ml", text: "AI/ML" },
                        { key: "analytics", text: "Analytics" },
                        { key: "blockchain", text: "Blockchain" },
                        { key: "security", text: "Security" },
                        { key: "finance", text: "Finance" },
                        { key: "iot", text: "IoT" },
                        { key: "operations", text: "Operations" }
                    ],
                    capabilities: [
                        { key: "document-processing", text: "Document Processing" },
                        { key: "ai-enhancement", text: "AI Enhancement" },
                        { key: "vector-search", text: "Vector Search" },
                        { key: "similarity-analysis", text: "Similarity Analysis" },
                        { key: "mathematical-modeling", text: "Mathematical Modeling" },
                        { key: "financial-analysis", text: "Financial Analysis" },
                        { key: "threat-detection", text: "Threat Detection" },
                        { key: "security-analysis", text: "Security Analysis" }
                    ],
                    requestStatuses: [
                        { key: "", text: "All Status" },
                        { key: "pending", text: "Pending" },
                        { key: "active", text: "Active" },
                        { key: "completed", text: "Completed" },
                        { key: "cancelled", text: "Cancelled" }
                    ],
                    sortOptions: [
                        { key: "rating", text: "Rating (High to Low)" },
                        { key: "price_asc", text: "Price (Low to High)" },
                        { key: "price_desc", text: "Price (High to Low)" },
                        { key: "name", text: "Name (A to Z)" },
                        { key: "availability", text: "Availability" }
                    ]
                };

                this.oMarketplaceModel.setProperty("/categories", staticData.categories);
                this.oMarketplaceModel.setProperty("/capabilities", staticData.capabilities);
                this.oMarketplaceModel.setProperty("/requestStatuses", staticData.requestStatuses);
                this.oMarketplaceModel.setProperty("/sortOptions", staticData.sortOptions);

                resolve(staticData);
            });
        },

        _loadFallbackData() {
            this.oMarketplaceModel.setProperty("/agents", []);
            this.oMarketplaceModel.setProperty("/userRequests", []);
            this.oMarketplaceModel.setProperty("/userAgents", []);
        },

        _updateStats() {
            const agents = this.oMarketplaceModel.getProperty("/agents");
            const requests = this.oMarketplaceModel.getProperty("/userRequests");

            const stats = {
                totalAgents: agents.length,
                activeServices: agents.reduce((sum, agent) => sum + agent.activeServices, 0),
                totalTransactions: requests.length,
                averageRating: agents.length > 0 ?
                    (agents.reduce((sum, agent) => sum + agent.rating, 0) / agents.length).toFixed(1) : 0
            };

            this.oMarketplaceModel.setProperty("/stats", stats);
        },

        _updateRequestStats(requests) {
            const stats = {
                pendingCount: requests.filter(r => r.status === "Pending").length,
                activeCount: requests.filter(r => r.status === "Active").length,
                completedCount: requests.filter(r => r.status === "Completed").length
            };

            const total = requests.length;
            stats.pending = total > 0 ? Math.round((stats.pendingCount / total) * 100) : 0;
            stats.active = total > 0 ? Math.round((stats.activeCount / total) * 100) : 0;
            stats.completed = total > 0 ? Math.round((stats.completedCount / total) * 100) : 0;

            this.oMarketplaceModel.setProperty("/requestStats", stats);
        },

        // Event Handlers

        onSearchAgents(oEvent) {
            const sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            this._applyAgentFilters({ search: sQuery });
        },

        onCategoryFilter(oEvent) {
            const sCategory = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/filters/category", sCategory);
            this._applyAgentFilters();
        },

        onCapabilityFilter(oEvent) {
            const aCapabilities = oEvent.getParameter("selectedItems").map(item => item.getKey());
            this.oMarketplaceModel.setProperty("/filters/capabilities", aCapabilities);
            this._applyAgentFilters();
        },

        onPriceRangeChange(oEvent) {
            const fMin = oEvent.getParameter("value");
            const fMax = oEvent.getParameter("value2");
            this.oMarketplaceModel.setProperty("/filters/priceMin", fMin);
            this.oMarketplaceModel.setProperty("/filters/priceMax", fMax);
            this._applyAgentFilters();
        },

        onRatingFilter(oEvent) {
            const iRating = oEvent.getParameter("value");
            this.oMarketplaceModel.setProperty("/filters/minRating", iRating);
            this._applyAgentFilters();
        },

        onAvailabilityFilter(oEvent) {
            const bAvailableOnly = oEvent.getParameter("state");
            this.oMarketplaceModel.setProperty("/filters/availableOnly", bAvailableOnly);
            this._applyAgentFilters();
        },

        onViewToggle(oEvent) {
            const sViewMode = oEvent.getParameter("key") || oEvent.getSource().getSelectedKey();
            this.oMarketplaceModel.setProperty("/viewMode", sViewMode);
        },

        onSortChange(oEvent) {
            const sSortKey = oEvent.getParameter("selectedItem").getKey();
            this.oMarketplaceModel.setProperty("/sortBy", sSortKey);
            this._applySorting(sSortKey);
        },

        onAgentPress(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._showAgentDetails(oAgent);
        },

        onViewAgentDetails(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();
            this._showAgentDetails(oAgent);
        },

        onRequestService(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();
            this.oMarketplaceModel.setProperty("/selectedAgent", oAgent);
            this._resetServiceRequest();
            this.byId("serviceRequestDialog").open();
        },

        onSubmitServiceRequest() {
            const oRequest = this.oMarketplaceModel.getProperty("/serviceRequest");
            const oAgent = this.oMarketplaceModel.getProperty("/selectedAgent");

            if (!oRequest.serviceType || !oRequest.deadline) {
                MessageBox.error(this.getResourceBundle().getText("agentMarketplace.validation.required"));
                return;
            }

            // Mock service request submission
            const newRequest = {
                requestId: `REQ_${Date.now()}`,
                agentName: oAgent.name,
                serviceName: oRequest.serviceType,
                status: "Pending",
                requestDate: new Date().toISOString(),
                completionDate: null,
                cost: oRequest.estimatedCost,
                rated: false
            };

            // Add to user requests
            const userRequests = this.oMarketplaceModel.getProperty("/userRequests");
            userRequests.unshift(newRequest);
            this.oMarketplaceModel.setProperty("/userRequests", userRequests);
            this._updateRequestStats(userRequests);

            this.byId("serviceRequestDialog").close();
            MessageToast.show(this.getResourceBundle().getText("agentMarketplace.request.submitted"));

            // Send WebSocket update
            this._sendWebSocketMessage({
                type: "service_request_created",
                data: newRequest
            });
        },

        onCancelServiceRequest() {
            this.byId("serviceRequestDialog").close();
        },

        onRegisterAgent() {
            this._resetNewAgent();
            this.byId("agentRegistrationDialog").open();
        },

        onConfirmAgentRegistration() {
            const oNewAgent = this.oMarketplaceModel.getProperty("/newAgent");

            if (!oNewAgent.name || !oNewAgent.description || !oNewAgent.category || !oNewAgent.endpoint) {
                MessageBox.error(this.getResourceBundle().getText("agentMarketplace.validation.required"));
                return;
            }

            // Mock agent registration
            const registeredAgent = {
                id: `agent_${Date.now()}`,
                name: oNewAgent.name,
                description: oNewAgent.description,
                category: oNewAgent.category,
                status: "Pending Approval",
                rating: 0,
                totalEarnings: 0,
                activeServices: 0,
                lastActivity: new Date().toISOString()
            };

            // Add to user agents
            const userAgents = this.oMarketplaceModel.getProperty("/userAgents");
            userAgents.push(registeredAgent);
            this.oMarketplaceModel.setProperty("/userAgents", userAgents);

            this.byId("agentRegistrationDialog").close();
            MessageToast.show(this.getResourceBundle().getText("agentMarketplace.agent.registered"));

            // Send WebSocket update
            this._sendWebSocketMessage({
                type: "agent_registered",
                data: registeredAgent
            });
        },

        onCancelAgentRegistration() {
            this.byId("agentRegistrationDialog").close();
        },

        onCancelRequest(oEvent) {
            const oRequest = oEvent.getSource().getBindingContext("marketplace").getObject();

            MessageBox.confirm(
                this.getResourceBundle().getText("agentMarketplace.request.cancelConfirm"),
                {
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            oRequest.status = "Cancelled";
                            this.oMarketplaceModel.refresh();
                            MessageToast.show(this.getResourceBundle().getText("agentMarketplace.request.cancelled"));
                        }
                    }
                }
            );
        },

        onRateService(oEvent) {
            const oRequest = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`Rate service: ${oRequest.serviceName}`);
            // Implement rating dialog
        },

        onViewRequestDetails(oEvent) {
            const oRequest = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`View details for request: ${oRequest.requestId}`);
            // Implement request details dialog
        },

        onEditAgent(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();
            MessageToast.show(`Edit agent: ${oAgent.name}`);
            // Implement agent editing dialog
        },

        onToggleAgentStatus(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();
            oAgent.status = oAgent.status === "Available" ? "Paused" : "Available";
            this.oMarketplaceModel.refresh();
            MessageToast.show(this.getResourceBundle().getText("agentMarketplace.agent.statusChanged"));
        },

        onDeleteAgent(oEvent) {
            const oAgent = oEvent.getSource().getBindingContext("marketplace").getObject();

            MessageBox.confirm(
                this.getResourceBundle().getText("agentMarketplace.agent.deleteConfirm"),
                {
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            const userAgents = this.oMarketplaceModel.getProperty("/userAgents");
                            const index = userAgents.findIndex(a => a.id === oAgent.id);
                            if (index > -1) {
                                userAgents.splice(index, 1);
                                this.oMarketplaceModel.refresh();
                                MessageToast.show(this.getResourceBundle().getText("agentMarketplace.agent.deleted"));
                            }
                        }
                    }
                }
            );
        },

        // Helper Methods

        _applyAgentFilters(additionalFilters = {}) {
            const oFilters = this.oMarketplaceModel.getProperty("/filters");
            const agents = this.oMarketplaceModel.getProperty("/agents");

            agents.forEach(agent => {
                let visible = true;

                // Search filter
                const searchQuery = additionalFilters.search || "";
                if (searchQuery) {
                    const query = searchQuery.toLowerCase();
                    visible = agent.name.toLowerCase().includes(query) ||
                             agent.description.toLowerCase().includes(query) ||
                             agent.owner.toLowerCase().includes(query);
                }

                // Category filter
                if (visible && oFilters.category) {
                    visible = agent.category.toLowerCase() === oFilters.category.toLowerCase();
                }

                // Capability filter
                if (visible && oFilters.capabilities.length > 0) {
                    visible = oFilters.capabilities.some(cap =>
                        agent.capabilities.some(agentCap =>
                            agentCap.toLowerCase().includes(cap.toLowerCase())
                        )
                    );
                }

                // Price range filter
                if (visible) {
                    visible = agent.pricing >= oFilters.priceMin && agent.pricing <= oFilters.priceMax;
                }

                // Rating filter
                if (visible && oFilters.minRating > 0) {
                    visible = agent.rating >= oFilters.minRating;
                }

                // Availability filter
                if (visible && oFilters.availableOnly) {
                    visible = agent.status === "Available";
                }

                agent.visible = visible;
            });

            this.oMarketplaceModel.refresh();
        },

        _applySorting(sSortKey) {
            const agents = this.oMarketplaceModel.getProperty("/agents");

            let sorter;
            switch (sSortKey) {
            case "rating":
                sorter = new Sorter("rating", true); // descending
                break;
            case "price_asc":
                sorter = new Sorter("pricing", false);
                break;
            case "price_desc":
                sorter = new Sorter("pricing", true);
                break;
            case "name":
                sorter = new Sorter("name", false);
                break;
            case "availability":
                sorter = new Sorter("status", false);
                break;
            default:
                sorter = new Sorter("rating", true);
            }

            agents.sort((a, b) => {
                if (sorter.fnCompare) {
                    return sorter.fnCompare(a, b);
                }

                const prop = sorter.sPath;
                const desc = sorter.bDescending;

                if (a[prop] < b[prop]) {return desc ? 1 : -1;}
                if (a[prop] > b[prop]) {return desc ? -1 : 1;}
                return 0;
            });

            this.oMarketplaceModel.setProperty("/agents", agents);
        },

        _showAgentDetails(oAgent) {
            MessageToast.show(`Agent details: ${oAgent.name}`);
            // Implement agent details dialog
        },

        _resetServiceRequest() {
            this.oMarketplaceModel.setProperty("/serviceRequest", {
                serviceType: "",
                parameters: "",
                deadline: null,
                estimatedCost: 0,
                escrowAmount: 0
            });
        },

        _resetNewAgent() {
            this.oMarketplaceModel.setProperty("/newAgent", {
                name: "",
                description: "",
                category: "",
                endpoint: "",
                capabilities: [],
                pricing: 0,
                minimumStake: 0
            });
        },

        // WebSocket Implementation
        _initializeWebSocket() {
            if (window.WebSocket) {
                const wsUrl = `ws://localhost:8000/api/v1/marketplace/ws/user_${Date.now()}`;

                try {
                    this.websocket = new WebSocket(wsUrl);

                    this.websocket.onopen = () => {
                        console.log("Agent marketplace WebSocket connected");
                        this._sendWebSocketMessage({
                            type: "subscribe",
                            subscriptions: ["agent_updates", "service_requests", "marketplace_stats"]
                        });
                    };

                    this.websocket.onmessage = (event) => {
                        this._handleWebSocketMessage(JSON.parse(event.data));
                    };

                    this.websocket.onerror = (error) => {
                        console.warn("Agent marketplace WebSocket error:", error);
                    };

                    this.websocket.onclose = () => {
                        console.log("Agent marketplace WebSocket disconnected");
                        // Attempt reconnection after 5 seconds
                        setTimeout(() => this._initializeWebSocket(), 5000);
                    };

                } catch (error) {
                    console.warn("Failed to initialize WebSocket:", error);
                }
            }
        },

        _sendWebSocketMessage(message) {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify(message));
            }
        },

        _handleWebSocketMessage(message) {
            switch (message.type) {
            case "marketplace_update":
                this._updateStats();
                break;
            case "agent_status_changed":
                this._handleAgentStatusChange(message.data);
                break;
            case "service_request_update":
                this._handleServiceRequestUpdate(message.data);
                break;
            case "subscription_confirmed":
                console.log("WebSocket subscriptions confirmed:", message.subscriptions);
                break;
            }
        },

        _handleAgentStatusChange(agentData) {
            const agents = this.oMarketplaceModel.getProperty("/agents");
            const agent = agents.find(a => a.id === agentData.id);
            if (agent) {
                agent.status = agentData.status;
                agent.lastActivity = agentData.lastActivity;
                this.oMarketplaceModel.refresh();
            }
        },

        _handleServiceRequestUpdate(requestData) {
            const requests = this.oMarketplaceModel.getProperty("/userRequests");
            const request = requests.find(r => r.requestId === requestData.requestId);
            if (request) {
                Object.assign(request, requestData);
                this._updateRequestStats(requests);
                this.oMarketplaceModel.refresh();

                // Show notification
                MessageToast.show(`Request ${requestData.requestId} updated: ${requestData.status}`);
            }
        },

        _setupRealtimeUpdates() {
            // Periodic updates for agent status
            this._updateInterval = setInterval(() => {
                this._updateStats();
            }, 30000); // Update every 30 seconds
        },

        // Formatters
        formatStatusColor(sStatus) {
            switch (sStatus) {
            case "Available":
                return "Good";
            case "Busy":
                return "Critical";
            case "Offline":
                return "Error";
            default:
                return "Neutral";
            }
        },

        formatStatusState(sStatus) {
            switch (sStatus) {
            case "Available":
                return "Success";
            case "Busy":
                return "Warning";
            case "Offline":
                return "Error";
            default:
                return "None";
            }
        },

        formatRequestStatusState(sStatus) {
            switch (sStatus) {
            case "Completed":
                return "Success";
            case "Active":
                return "Information";
            case "Pending":
                return "Warning";
            case "Cancelled":
                return "Error";
            default:
                return "None";
            }
        },

        onNavBack() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onExit() {
            // Clean up
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }

            if (this.websocket) {
                this.websocket.close();
            }
        }
    });
});