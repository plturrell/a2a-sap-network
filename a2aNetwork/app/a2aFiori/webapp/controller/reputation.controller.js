sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/ui/fl/variants/VariantManagement",
    "sap/m/p13n/Engine",
    "sap/m/p13n/SelectionController",
    "sap/m/p13n/SortController",
    "sap/m/p13n/FilterController",
    "sap/m/p13n/GroupController",
    "sap/ui/export/Spreadsheet",
    "../utils/constants",
    "../utils/errorHandler"
], (Controller, JSONModel, MessageBox, MessageToast, Fragment, Filter, FilterOperator,
    VariantManagement, Engine, SelectionController, SortController, FilterController,
    GroupController, Spreadsheet, constants, errorHandler) => {
    "use strict";

    return Controller.extend("a2a.controller.reputation", {

        onInit() {
            // Initialize models
            this.oReputationModel = new JSONModel({
                currentAgent: {},
                reputationHistory: [],
                endorsements: {
                    received: [],
                    given: []
                },
                analytics: {
                    trend: "STABLE",
                    weeklyChange: 0,
                    monthlyChange: 0,
                    successRate: 0
                },
                endorsementForm: {
                    toAgentId: "",
                    amount: 5,
                    reason: "",
                    description: ""
                },
                milestones: [],
                badges: {
                    current: {},
                    all: []
                },
                limits: {
                    dailyRemaining: 50,
                    maxEndorsementAmount: 5
                }
            });

            this.getView().setModel(this.oReputationModel, "reputation");

            // Initialize P13n model
            this.oP13nModel = new JSONModel({
                columns: [],
                sortItems: [],
                filterItems: [],
                groupItems: []
            });
            this.getView().setModel(this.oP13nModel, "p13n");

            // Load initial data
            this._loadReputationData();

            // Subscribe to reputation events
            this._subscribeToEvents();

            // Set up refresh timer
            this._startAutoRefresh();

            // Initialize P13n and Variant Management
            this._initializeP13n();
            this._initializeVariantManagement();
        },

        /**
         * Load reputation data for current agent
         */
        _loadReputationData() {
            const sAgentId = this.getOwnerComponent().getModel("global").getProperty("/currentAgentId");
            if (!sAgentId) {
                return;
            }

            // Load multiple data points in parallel
            Promise.all([
                this._loadAgentReputation(sAgentId),
                this._loadReputationHistory(sAgentId),
                this._loadEndorsements(sAgentId),
                this._loadReputationAnalytics(sAgentId),
                this._loadEndorsementLimits()
            ]).then(() => {
                this._updateReputationDisplay();
            }).catch(error => {
                errorHandler.handleError(error, "Failed to load reputation data");
            });
        },

        /**
         * Load agent reputation details
         */
        _loadAgentReputation(sAgentId) {
            return new Promise((resolve, reject) => {
                const oModel = this.getOwnerComponent().getModel();

                oModel.callFunction("/calculateReputation", {
                    urlParameters: {
                        agentId: sAgentId
                    },
                    success: (oData) => {
                        this.oReputationModel.setProperty("/currentAgent", {
                            id: sAgentId,
                            reputation: oData.reputation,
                            successRate: oData.successRate,
                            endorsementScore: oData.endorsementScore,
                            trustScore: oData.trustScore,
                            badge: oData.badge,
                            performance: oData.performance
                        });

                        // Update badge display
                        this._updateBadgeDisplay(oData.badge);

                        resolve();
                    },
                    error: reject
                });
            });
        },

        /**
         * Load reputation history
         */
        _loadReputationHistory(sAgentId) {
            return new Promise((resolve, reject) => {
                const oModel = this.getOwnerComponent().getModel();
                const dEndDate = new Date();
                const dStartDate = new Date();
                dStartDate.setDate(dStartDate.getDate() - 30); // Last 30 days

                oModel.callFunction("/getReputationHistory", {
                    urlParameters: {
                        agentId: sAgentId,
                        startDate: dStartDate.toISOString().split("T")[0],
                        endDate: dEndDate.toISOString().split("T")[0]
                    },
                    success: (oData) => {
                        this.oReputationModel.setProperty("/reputationHistory", oData.results);
                        this._calculateTrend(oData.results);
                        resolve();
                    },
                    error: reject
                });
            });
        },

        /**
         * Load endorsements
         */
        _loadEndorsements(sAgentId) {
            const oModel = this.getOwnerComponent().getModel();

            return Promise.all([
                // Load received endorsements
                new Promise((resolve, reject) => {
                    oModel.read("/PeerEndorsements", {
                        filters: [
                            new sap.ui.model.Filter("toAgent_ID", "EQ", sAgentId)
                        ],
                        sorters: [
                            new sap.ui.model.Sorter("createdAt", true)
                        ],
                        success: (oData) => {
                            this.oReputationModel.setProperty("/endorsements/received", oData.results);
                            resolve();
                        },
                        error: reject
                    });
                }),

                // Load given endorsements
                new Promise((resolve, reject) => {
                    oModel.read("/PeerEndorsements", {
                        filters: [
                            new sap.ui.model.Filter("fromAgent_ID", "EQ", sAgentId)
                        ],
                        sorters: [
                            new sap.ui.model.Sorter("createdAt", true)
                        ],
                        success: (oData) => {
                            this.oReputationModel.setProperty("/endorsements/given", oData.results);
                            resolve();
                        },
                        error: reject
                    });
                })
            ]);
        },

        /**
         * Load reputation analytics
         */
        _loadReputationAnalytics(sAgentId) {
            return new Promise((resolve, reject) => {
                const oModel = this.getOwnerComponent().getModel();

                oModel.callFunction("/getReputationAnalytics", {
                    urlParameters: {
                        agentId: sAgentId,
                        period: "MONTHLY"
                    },
                    success: (oData) => {
                        this.oReputationModel.setProperty("/analytics", {
                            trend: this._calculateTrendDirection(oData),
                            weeklyChange: oData.weeklyChange || 0,
                            monthlyChange: oData.endingReputation - oData.startingReputation,
                            successRate: oData.taskSuccessRate || 0,
                            totalEarned: oData.totalEarned,
                            totalLost: oData.totalLost,
                            uniqueEndorsers: oData.uniqueEndorsers
                        });
                        resolve();
                    },
                    error: reject
                });
            });
        },

        /**
         * Load endorsement limits for current user
         */
        _loadEndorsementLimits() {
            // This would typically call a service to get current limits
            const iCurrentReputation = this.oReputationModel.getProperty("/currentAgent/reputation") || 100;
            const iMaxAmount = this._getMaxEndorsementAmount(iCurrentReputation);

            this.oReputationModel.setProperty("/limits/maxEndorsementAmount", iMaxAmount);

            // TODO: Get actual daily remaining from service
            return Promise.resolve();
        },

        /**
         * Handle endorsement submission
         */
        onEndorsePeer() {
            const oForm = this.oReputationModel.getProperty("/endorsementForm");

            // Validate form
            if (!oForm.toAgentId || !oForm.reason) {
                MessageBox.error("Please fill in all required fields");
                return;
            }

            // Prepare context
            const _oContext = {
                description: oForm.description,
                timestamp: new Date().toISOString()
            };

            const oModel = this.getOwnerComponent().getModel();
            const sFromAgentId = this.getOwnerComponent().getModel("global").getProperty("/currentAgentId");

            // Call endorsement service
            oModel.callFunction("/endorsePeer", {
                urlParameters: {
                    fromAgentId: sFromAgentId,
                    toAgentId: oForm.toAgentId,
                    amount: oForm.amount,
                    reason: oForm.reason,
                    context: JSON.stringify(oContext)
                },
                success: (oData) => {
                    if (oData.success) {
                        MessageToast.show(`Successfully endorsed agent with ${oForm.amount} reputation points`);

                        // Reset form
                        this.oReputationModel.setProperty("/endorsementForm", {
                            toAgentId: "",
                            amount: 5,
                            reason: "",
                            description: ""
                        });

                        // Reload endorsements
                        this._loadEndorsements(sFromAgentId);

                        // Update limits
                        const iRemaining = this.oReputationModel.getProperty("/limits/dailyRemaining");
                        this.oReputationModel.setProperty("/limits/dailyRemaining", iRemaining - oForm.amount);
                    }
                },
                error: (oError) => {
                    errorHandler.handleError(oError, "Failed to endorse peer");
                }
            });
        },

        /**
         * Calculate maximum endorsement amount based on reputation
         */
        _getMaxEndorsementAmount(iReputation) {
            if (iReputation <= 50) {
                return 3;
            }
            if (iReputation <= 100) {
                return 5;
            }
            if (iReputation <= 150) {
                return 7;
            }
            return 10;
        },

        /**
         * Update badge display
         */
        _updateBadgeDisplay(oBadge) {
            this.oReputationModel.setProperty("/badges/current", oBadge);

            // Define all badges for progress display
            const aBadges = [
                { name: "NEWCOMER", min: 0, max: 49, icon: "🌱", color: "gray", unlocked: true },
                { name: "ESTABLISHED", min: 50, max: 99, icon: "⭐", color: "bronze", unlocked: oBadge.name !== "NEWCOMER" },
                { name: "TRUSTED", min: 100, max: 149, icon: "🏆", color: "silver", unlocked: ["TRUSTED", "EXPERT", "LEGENDARY"].includes(oBadge.name) },
                { name: "EXPERT", min: 150, max: 199, icon: "💎", color: "gold", unlocked: ["EXPERT", "LEGENDARY"].includes(oBadge.name) },
                { name: "LEGENDARY", min: 200, max: 200, icon: "👑", color: "platinum", unlocked: oBadge.name === "LEGENDARY" }
            ];

            this.oReputationModel.setProperty("/badges/all", aBadges);
        },

        /**
         * Calculate reputation trend
         */
        _calculateTrend(aHistory) {
            if (!aHistory || aHistory.length < 2) {
                return;
            }

            // Calculate trend over last 7 days
            const dWeekAgo = new Date();
            dWeekAgo.setDate(dWeekAgo.getDate() - 7);

            const aWeekHistory = aHistory.filter(oTx => new Date(oTx.createdAt) >= dWeekAgo);
            const iWeeklyChange = aWeekHistory.reduce((sum, oTx) => sum + oTx.amount, 0);

            this.oReputationModel.setProperty("/analytics/weeklyChange", iWeeklyChange);
        },

        /**
         * Calculate trend direction
         */
        _calculateTrendDirection(oAnalytics) {
            const iChange = oAnalytics.endingReputation - oAnalytics.startingReputation;
            if (iChange > 5) {
                return "UP";
            }
            if (iChange < -5) {
                return "DOWN";
            }
            return "STABLE";
        },

        /**
         * Update reputation display formatting
         */
        _updateReputationDisplay() {
            const iReputation = this.oReputationModel.getProperty("/currentAgent/reputation");
            const sFormattedRep = `${iReputation}/200`;
            this.oReputationModel.setProperty("/currentAgent/formattedReputation", sFormattedRep);

            // Update progress percentage
            const iProgress = (iReputation / 200) * 100;
            this.oReputationModel.setProperty("/currentAgent/progressPercentage", iProgress);

            // Update state based on reputation level
            let sState = "None";
            if (iReputation >= 150) {
                sState = "Success";
            } else if (iReputation >= 100) {
                sState = "Information";
            } else if (iReputation >= 50) {
                sState = "Warning";
            } else {
                sState = "Error";
            }

            this.oReputationModel.setProperty("/currentAgent/reputationState", sState);
        },

        /**
         * Subscribe to reputation events
         */
        _subscribeToEvents() {
            const oEventBus = sap.ui.getCore().getEventBus();

            // Listen for reputation changes
            oEventBus.subscribe("reputation", "changed", (sChannel, sEvent, oData) => {
                if (oData.agentId === this.oReputationModel.getProperty("/currentAgent/id")) {
                    // Reload reputation data
                    this._loadReputationData();

                    // Show notification
                    MessageToast.show(`Reputation ${oData.change > 0 ? "increased" : "decreased"} by ${Math.abs(oData.change)} points`);
                }
            });

            // Listen for new endorsements
            oEventBus.subscribe("reputation", "endorsed", (sChannel, sEvent, oData) => {
                if (oData.toAgent === this.oReputationModel.getProperty("/currentAgent/id")) {
                    // Reload endorsements
                    this._loadEndorsements(oData.toAgent);

                    // Show notification
                    MessageToast.show(`You received a ${oData.amount} point endorsement!`);
                }
            });
        },

        /**
         * Start auto refresh timer
         */
        _startAutoRefresh() {
            // Refresh data every 30 seconds
            this._refreshInterval = setInterval(() => {
                this._loadReputationData();
            }, 30000);
        },

        /**
         * Navigate to reputation details
         */
        onNavigateToDetails(oEvent) {
            const oRouter = this.getOwnerComponent().getRouter();
            const sAgentId = oEvent.getSource().data("agentId");

            oRouter.navTo("reputationDetails", {
                agentId: sAgentId
            });
        },

        /**
         * Export reputation history
         */
        onExportHistory() {
            const aHistory = this.oReputationModel.getProperty("/reputationHistory");

            // Convert to CSV
            const sCSV = this._convertToCSV(aHistory);

            // Download file
            const blob = new Blob([sCSV], { type: "text/csv" });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `reputation_history_${new Date().toISOString().split("T")[0]}.csv`;
            a.click();

            MessageToast.show("Reputation history exported");
        },

        /**
         * Convert data to CSV
         */
        _convertToCSV(aData) {
            const aHeaders = ["Date", "Type", "Amount", "Reason", "New Balance"];
            const aRows = aData.map(oTx => [
                new Date(oTx.createdAt).toLocaleString(),
                oTx.transactionType,
                oTx.amount,
                oTx.reason,
                oTx.newBalance || ""
            ]);

            return [aHeaders, ...aRows].map(row => row.join(",")).join("\n");
        },

        onExit() {
            // Clean up
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }

            const oEventBus = sap.ui.getCore().getEventBus();
            oEventBus.unsubscribe("reputation", "changed");
            oEventBus.unsubscribe("reputation", "endorsed");
        },

        /* =========================================================== */
        /* Variant Management                                          */
        /* =========================================================== */

        onVariantSelect(oEvent) {
            const sVariantKey = oEvent.getParameter("key");
            this._applyVariant(sVariantKey);
        },

        onVariantSave(oEvent) {
            const sVariantName = oEvent.getParameter("name");
            const _bDefault = oEvent.getParameter("def");
            const _bPublic = oEvent.getParameter("public");
            const bOverwrite = oEvent.getParameter("overwrite");

            this._saveVariant(sVariantName, bDefault, bPublic, bOverwrite);
        },

        onVariantManage(oEvent) {
            // Handle variant management - would typically open a dialog to manage variants
            const _oVariantManagement = oEvent.getSource();
            // Implementation for managing variants
        },

        /* =========================================================== */
        /* P13n Dialog Handlers                                        */
        /* =========================================================== */

        onTableSettings(oEvent) {
            const oTable = this.byId("rankingsTable");
            this._openP13nDialog(oTable);
        },

        onDisputeTableSettings(oEvent) {
            const oTable = this.byId("disputesTable");
            this._openP13nDialog(oTable);
        },

        onP13nDialogOK(oEvent) {
            this._applyP13nSettings();
            this.byId("p13nDialog").close();
        },

        onP13nDialogCancel(oEvent) {
            this.byId("p13nDialog").close();
        },

        onP13nDialogReset(oEvent) {
            this._resetP13nSettings();
        },

        onChangeColumnsItems(oEvent) {
            // Handle column changes
            const aItems = oEvent.getParameter("items");
            this.oP13nModel.setProperty("/columns", aItems);
        },

        onChangeSortItems(oEvent) {
            // Handle sort changes
            const aItems = oEvent.getParameter("sortItems");
            this.oP13nModel.setProperty("/sortItems", aItems);
        },

        onChangeFilterItems(oEvent) {
            // Handle filter changes
            const aItems = oEvent.getParameter("filterItems");
            this.oP13nModel.setProperty("/filterItems", aItems);
        },

        onChangeGroupItems(oEvent) {
            // Handle group changes
            const aItems = oEvent.getParameter("groupItems");
            this.oP13nModel.setProperty("/groupItems", aItems);
        },

        /* =========================================================== */
        /* P13n Internal Methods                                       */
        /* =========================================================== */

        _initializeP13n() {
            // Initialize P13n Engine for tables
            const oRankingsTable = this.byId("rankingsTable");
            if (oRankingsTable) {
                Engine.getInstance().register(oRankingsTable, {
                    helper: {
                        name: "sap.m.p13n.TableHelper",
                        payload: {
                            column: this._getColumnConfiguration()
                        }
                    },
                    controller: {
                        Columns: new SelectionController({
                            targetAggregation: "columns"
                        }),
                        Sorter: new SortController({
                            targetAggregation: "sorter"
                        }),
                        Filter: new FilterController({
                            targetAggregation: "filter"
                        }),
                        Group: new GroupController({
                            targetAggregation: "group"
                        })
                    }
                });
            }

            const oDisputesTable = this.byId("disputesTable");
            if (oDisputesTable) {
                Engine.getInstance().register(oDisputesTable, {
                    helper: {
                        name: "sap.m.p13n.TableHelper",
                        payload: {
                            column: this._getDisputeColumnConfiguration()
                        }
                    },
                    controller: {
                        Columns: new SelectionController({
                            targetAggregation: "columns"
                        }),
                        Sorter: new SortController({
                            targetAggregation: "sorter"
                        }),
                        Filter: new FilterController({
                            targetAggregation: "filter"
                        })
                    }
                });
            }
        },

        _initializeVariantManagement() {
            // Initialize variant management for tables
            const oRankingsVariant = this.byId("rankingsVariant");
            if (oRankingsVariant) {
                this._setupVariantManagement(oRankingsVariant, "rankingsTable");
            }

            const oDisputesVariant = this.byId("disputesVariant");
            if (oDisputesVariant) {
                this._setupVariantManagement(oDisputesVariant, "disputesTable");
            }
        },

        _setupVariantManagement(oVariantManagement, sTableId) {
            if (!oVariantManagement) {
                return;
            }

            // Load saved variants from backend or local storage
            const aVariants = this._loadVariants(sTableId);
            oVariantManagement.setModel(new JSONModel({
                variants: aVariants
            }), "variants");
        },

        _getColumnConfiguration() {
            return [
                { key: "rank", label: "Rank", dataType: "sap.ui.model.type.Integer" },
                { key: "name", label: "Agent Name", dataType: "sap.ui.model.type.String" },
                { key: "category", label: "Category", dataType: "sap.ui.model.type.String" },
                { key: "reputationScore", label: "Reputation Score", dataType: "sap.ui.model.type.Integer" },
                { key: "totalReviews", label: "Total Reviews", dataType: "sap.ui.model.type.Integer" },
                { key: "successRate", label: "Success Rate", dataType: "sap.ui.model.type.Float" },
                { key: "avgResponseTime", label: "Avg Response Time", dataType: "sap.ui.model.type.Integer" },
                { key: "lastActive", label: "Last Active", dataType: "sap.ui.model.odata.type.DateTimeOffset" }
            ];
        },

        _getDisputeColumnConfiguration() {
            return [
                { key: "disputeId", label: "Dispute ID", dataType: "sap.ui.model.type.String" },
                { key: "transactionId", label: "Transaction", dataType: "sap.ui.model.type.String" },
                { key: "disputerName", label: "Disputer", dataType: "sap.ui.model.type.String" },
                { key: "disputedParty", label: "Disputed Party", dataType: "sap.ui.model.type.String" },
                { key: "reason", label: "Reason", dataType: "sap.ui.model.type.String" },
                { key: "status", label: "Status", dataType: "sap.ui.model.type.String" },
                { key: "submittedDate", label: "Submitted Date", dataType: "sap.ui.model.odata.type.DateTimeOffset" }
            ];
        },

        _openP13nDialog(oTable) {
            const oP13nDialog = this.byId("p13nDialog");

            // Prepare P13n data
            this._prepareP13nData(oTable);

            // Open dialog
            oP13nDialog.open();
        },

        _prepareP13nData(oTable) {
            const _aColumns = oTable.getColumns();
            const aP13nColumns = [];

            aColumns.forEach((oColumn, iIndex) => {
                aP13nColumns.push({
                    columnKey: oColumn.getId(),
                    text: oColumn.getHeader().getText(),
                    visible: oColumn.getVisible(),
                    index: iIndex
                });
            });

            this.oP13nModel.setProperty("/columns", aP13nColumns);
        },

        _applyP13nSettings() {
            // Apply personalization settings
            const _aColumns = this.oP13nModel.getProperty("/columns");
            const _aSortItems = this.oP13nModel.getProperty("/sortItems");
            const _aFilterItems = this.oP13nModel.getProperty("/filterItems");
            const _aGroupItems = this.oP13nModel.getProperty("/groupItems");

            // Apply column visibility and order
            // Apply sort settings
            // Apply filter settings
            // Apply group settings

            MessageToast.show("Table settings applied");
        },

        _resetP13nSettings() {
            // Reset to default settings
            this._prepareP13nData(this._currentTable);
            MessageToast.show("Table settings reset to default");
        },

        _loadVariants(sTableId) {
            // Load saved variants from backend or local storage
            // This is a mock implementation
            return [
                {
                    key: "default",
                    text: "Default View",
                    isDefault: true
                },
                {
                    key: "highPerformers",
                    text: "High Performers",
                    isDefault: false
                },
                {
                    key: "recentActivity",
                    text: "Recently Active",
                    isDefault: false
                }
            ];
        },

        _applyVariant(sVariantKey) {
            // Apply selected variant
            // This would load saved settings and apply them to the table
            MessageToast.show(`Variant applied: ${ sVariantKey}`);
        },

        _saveVariant(sName, bDefault, bPublic, bOverwrite) {
            // Save current state as variant
            // This would save the current table configuration
            MessageToast.show(`Variant saved: ${ sName}`);
        }
    });
});