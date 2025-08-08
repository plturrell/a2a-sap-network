sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log",
    "sap/viz/ui5/format/ChartFormatter",
    "sap/viz/ui5/api/env/Format"
], function(BaseController, MessageToast, MessageBox, formatter, JSONModel, Filter, FilterOperator, Log, ChartFormatter, Format) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.AgentVisualization", {
        formatter: formatter,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit: function() {
            // Call base controller initialization
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Initialize agent visualization model
            var oAgentVizModel = new JSONModel({
                viewMode: "grid",
                filterType: "all",
                filterStatus: "all",
                minReputation: 0,
                totalAgents: 0,
                activeAgents: 0,
                avgReputation: 0,
                agents: []
            });
            this.setModel(oAgentVizModel, "agentViz");

            // Initialize formatters for charts
            Format.numericFormatter(ChartFormatter.getInstance());
            var formatPattern = ChartFormatter.DefaultPattern;

            // Attach route matched handler
            this.getRouter().getRoute("agentVisualization").attachPatternMatched(this._onRouteMatched, this);

            Log.info("Agent Visualization controller initialized");
        },

        /* =========================================================== */
        /* event handlers                                              */
        /* =========================================================== */

        /**
         * Event handler for filter change.
         * @public
         */
        onFilterChange: function() {
            this._applyFilters();
        },

        /**
         * Event handler for agent card press.
         * @param {sap.ui.base.Event} oEvent the press event
         * @public
         */
        onAgentCardPress: function(oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("agentViz");
            var sAgentId = oBindingContext.getProperty("id");
            
            Log.debug("Agent card pressed", { agentId: sAgentId });
            
            this.getRouter().navTo("agentDetail", {
                agentId: sAgentId
            });
        },

        /**
         * Event handler for view details button.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onViewDetails: function(oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("agentViz");
            var sAgentId = oBindingContext.getProperty("id");
            
            this.getRouter().navTo("agentDetail", {
                agentId: sAgentId
            });
        },

        /**
         * Event handler for interact button.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onInteract: function(oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("agentViz");
            var oAgent = oBindingContext.getObject();
            
            Log.info("Interact with agent", { agentName: oAgent.name });
            
            // Open interaction dialog
            this._openInteractionDialog(oAgent);
        },

        /**
         * Event handler for search field.
         * @param {sap.ui.base.Event} oEvent the search event
         * @public
         */
        onSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("newValue");
            this._applyFilters(sQuery);
        },

        /**
         * Event handler for sort button.
         * @public
         */
        onSort: function() {
            Log.debug("Sort dialog requested");
            
            if (!this._oSortDialog) {
                this._createSortDialog();
            }
            this._oSortDialog.open();
        },

        /**
         * Event handler for export button.
         * @public
         */
        onExport: function() {
            Log.info("Export agents data requested");
            
            MessageBox.information(
                this.getResourceBundle().getText("exportFeatureComingSoon"),
                {
                    title: this.getResourceBundle().getText("export")
                }
            );
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Binds the view to the object path.
         * @function
         * @param {sap.ui.base.Event} oEvent pattern match event
         * @private
         */
        _onRouteMatched: function(oEvent) {
            Log.debug("Agent visualization route matched");
            
            // Load agents data
            this._loadAgents();
            
            // Initialize charts if in analytics view
            if (this.getModel("agentViz").getProperty("/viewMode") === "analytics") {
                this._initializeCharts();
            }
        },

        /**
         * Loads agents from the backend.
         * @private
         */
        _loadAgents: function() {
            var oModel = this.getModel();
            var oAgentVizModel = this.getModel("agentViz");
            
            // Show skeleton loading for agent data
            this.showSkeletonLoading(this.getResourceBundle().getText("loadingAgents"));
            
            oModel.read("/Agents", {
                success: function(oData) {
                    this.hideLoading();
                    
                    // Process agents data
                    var aAgents = oData.results.map(function(oAgent) {
                        return {
                            id: oAgent.id,
                            name: oAgent.name,
                            type: oAgent.type || "general",
                            status: oAgent.status || "idle",
                            reputation: oAgent.reputation || 100,
                            address: oAgent.address,
                            availability: oAgent.availability || 85,
                            serviceCount: oAgent.services ? oAgent.services.length : 0,
                            successRate: oAgent.successRate || 92,
                            avgResponseTime: oAgent.avgResponseTime || 250
                        };
                    });
                    
                    // Calculate summary statistics
                    var iTotal = aAgents.length;
                    var iActive = aAgents.filter(function(a) { return a.status === "active"; }).length;
                    var fAvgRep = aAgents.reduce(function(sum, a) { return sum + a.reputation; }, 0) / iTotal;
                    
                    // Update model
                    oAgentVizModel.setProperty("/agents", aAgents);
                    oAgentVizModel.setProperty("/totalAgents", iTotal);
                    oAgentVizModel.setProperty("/activeAgents", iActive);
                    oAgentVizModel.setProperty("/avgReputation", Math.round(fAvgRep));
                    
                    Log.info("Agents loaded successfully", { count: iTotal });
                    
                    // Initialize charts if in analytics view
                    if (oAgentVizModel.getProperty("/viewMode") === "analytics") {
                        this._initializeCharts();
                    }
                }.bind(this),
                error: function(oError) {
                    var sMessage = this._createErrorMessage ? this._createErrorMessage(oError) : "Failed to load agents";
                    this.showError(sMessage);
                    Log.error("Failed to load agents", oError);
                }.bind(this)
            });
        },

        /**
         * Applies filters to the agents.
         * @param {string} sSearchQuery optional search query
         * @private
         */
        _applyFilters: function(sSearchQuery) {
            var oAgentVizModel = this.getModel("agentViz");
            var sFilterType = oAgentVizModel.getProperty("/filterType");
            var sFilterStatus = oAgentVizModel.getProperty("/filterStatus");
            var iMinReputation = oAgentVizModel.getProperty("/minReputation");
            
            // Get the appropriate binding based on view mode
            var oBinding;
            if (oAgentVizModel.getProperty("/viewMode") === "grid") {
                // Grid view uses Cards - need to handle differently
                return; // Filtering in grid view would require custom implementation
            } else if (oAgentVizModel.getProperty("/viewMode") === "list") {
                oBinding = this.byId("agentListTable").getBinding("items");
            }
            
            if (!oBinding) {
                return;
            }
            
            var aFilters = [];
            
            // Type filter
            if (sFilterType !== "all") {
                aFilters.push(new Filter("type", FilterOperator.EQ, sFilterType));
            }
            
            // Status filter
            if (sFilterStatus !== "all") {
                aFilters.push(new Filter("status", FilterOperator.EQ, sFilterStatus));
            }
            
            // Reputation filter
            if (iMinReputation > 0) {
                aFilters.push(new Filter("reputation", FilterOperator.GE, iMinReputation));
            }
            
            // Search filter
            if (sSearchQuery) {
                aFilters.push(new Filter({
                    filters: [
                        new Filter("name", FilterOperator.Contains, sSearchQuery),
                        new Filter("address", FilterOperator.Contains, sSearchQuery)
                    ],
                    and: false
                }));
            }
            
            oBinding.filter(aFilters, "Application");
        },

        /**
         * Creates availability item for HarveyBall chart.
         * @param {string} sId the ID
         * @param {object} oContext the binding context
         * @returns {sap.suite.ui.microchart.HarveyBallMicroChartItem} the item
         * @private
         */
        _createAvailabilityItem: function(sId, oContext) {
            var iAvailability = oContext.getProperty("availability");
            
            return new sap.suite.ui.microchart.HarveyBallMicroChartItem({
                value: iAvailability,
                color: iAvailability > 90 ? "Good" : iAvailability > 70 ? "Critical" : "Error"
            });
        },

        /**
         * Initializes visualization charts.
         * @private
         */
        _initializeCharts: function() {
            var oAgentVizModel = this.getModel("agentViz");
            var aAgents = oAgentVizModel.getProperty("/agents");
            
            // Initialize agent type distribution chart
            this._initAgentTypeChart(aAgents);
            
            // Initialize performance chart
            this._initPerformanceChart(aAgents);
            
            // Initialize reputation distribution chart
            this._initReputationChart(aAgents);
            
            // Initialize activity heatmap
            this._initActivityChart(aAgents);
        },

        /**
         * Initializes agent type distribution chart.
         * @param {array} aAgents the agents data
         * @private
         */
        _initAgentTypeChart: function(aAgents) {
            var oVizFrame = this.byId("agentTypeChart");
            if (!oVizFrame) return;
            
            // Prepare data
            var oTypeCount = {};
            aAgents.forEach(function(agent) {
                oTypeCount[agent.type] = (oTypeCount[agent.type] || 0) + 1;
            });
            
            var aChartData = Object.keys(oTypeCount).map(function(type) {
                return {
                    Type: type,
                    Count: oTypeCount[type]
                };
            });
            
            // Create model and dataset
            var oModel = new JSONModel({ data: aChartData });
            var oDataset = new sap.viz.ui5.data.FlattenedDataset({
                dimensions: [{
                    name: "Type",
                    value: "{Type}"
                }],
                measures: [{
                    name: "Count",
                    value: "{Count}"
                }],
                data: {
                    path: "/data"
                }
            });
            
            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(oModel);
            
            // Set viz properties
            oVizFrame.setVizProperties({
                plotArea: {
                    colorPalette: ["#5899DA", "#E8743B", "#19A979", "#ED4A7B"]
                },
                title: {
                    visible: false
                }
            });
            
            // Feed dimensions and measures
            var oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "size",
                type: "Measure",
                values: ["Count"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "color",
                type: "Dimension",
                values: ["Type"]
            });
            
            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);
        },

        /**
         * Initializes performance overview chart.
         * @param {array} aAgents the agents data
         * @private
         */
        _initPerformanceChart: function(aAgents) {
            var oVizFrame = this.byId("performanceChart");
            if (!oVizFrame) return;
            
            // Get top 10 agents by reputation
            var aTopAgents = aAgents
                .sort(function(a, b) { return b.reputation - a.reputation; })
                .slice(0, 10)
                .map(function(agent) {
                    return {
                        Name: agent.name,
                        Reputation: agent.reputation,
                        SuccessRate: agent.successRate
                    };
                });
            
            // Create model and dataset
            var oModel = new JSONModel({ data: aTopAgents });
            var oDataset = new sap.viz.ui5.data.FlattenedDataset({
                dimensions: [{
                    name: "Name",
                    value: "{Name}"
                }],
                measures: [{
                    name: "Reputation",
                    value: "{Reputation}"
                }, {
                    name: "Success Rate",
                    value: "{SuccessRate}"
                }],
                data: {
                    path: "/data"
                }
            });
            
            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(oModel);
            oVizFrame.setVizType("column");
            
            // Set viz properties
            oVizFrame.setVizProperties({
                plotArea: {
                    colorPalette: ["#5899DA", "#19A979"]
                },
                title: {
                    visible: false
                },
                categoryAxis: {
                    title: {
                        visible: false
                    }
                },
                valueAxis: {
                    title: {
                        visible: false
                    }
                }
            });
            
            // Feed dimensions and measures
            var oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Reputation", "Success Rate"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis",
                type: "Dimension",
                values: ["Name"]
            });
            
            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);
        },

        /**
         * Initializes reputation distribution chart.
         * @param {array} aAgents the agents data
         * @private
         */
        _initReputationChart: function(aAgents) {
            var oVizFrame = this.byId("reputationChart");
            if (!oVizFrame) return;
            
            // Create reputation ranges
            var aRanges = [
                { range: "0-50", min: 0, max: 50, count: 0 },
                { range: "51-100", min: 51, max: 100, count: 0 },
                { range: "101-150", min: 101, max: 150, count: 0 },
                { range: "151-200", min: 151, max: 200, count: 0 }
            ];
            
            // Count agents in each range
            aAgents.forEach(function(agent) {
                var rep = agent.reputation;
                aRanges.forEach(function(range) {
                    if (rep >= range.min && rep <= range.max) {
                        range.count++;
                    }
                });
            });
            
            var aChartData = aRanges.map(function(range) {
                return {
                    Range: range.range,
                    Count: range.count
                };
            });
            
            // Create model and dataset
            var oModel = new JSONModel({ data: aChartData });
            var oDataset = new sap.viz.ui5.data.FlattenedDataset({
                dimensions: [{
                    name: "Range",
                    value: "{Range}"
                }],
                measures: [{
                    name: "Count",
                    value: "{Count}"
                }],
                data: {
                    path: "/data"
                }
            });
            
            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(oModel);
            
            // Set viz properties
            oVizFrame.setVizProperties({
                plotArea: {
                    colorPalette: ["#E8743B"]
                },
                title: {
                    visible: false
                }
            });
            
            // Feed dimensions and measures
            var oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Count"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis",
                type: "Dimension",
                values: ["Range"]
            });
            
            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);
        },

        /**
         * Initializes activity heatmap chart.
         * @param {array} aAgents the agents data
         * @private
         */
        _initActivityChart: function(aAgents) {
            var oVizFrame = this.byId("activityChart");
            if (!oVizFrame) return;
            
            // Generate sample activity data for last 7 days
            var aActivityData = [];
            var aDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
            var aHours = ["00", "04", "08", "12", "16", "20"];
            
            aDays.forEach(function(day) {
                aHours.forEach(function(hour) {
                    aActivityData.push({
                        Day: day,
                        Hour: hour,
                        Activity: Math.floor(Math.random() * 100)
                    });
                });
            });
            
            // Create model and dataset
            var oModel = new JSONModel({ data: aActivityData });
            var oDataset = new sap.viz.ui5.data.FlattenedDataset({
                dimensions: [{
                    name: "Day",
                    value: "{Day}"
                }, {
                    name: "Hour",
                    value: "{Hour}"
                }],
                measures: [{
                    name: "Activity",
                    value: "{Activity}"
                }],
                data: {
                    path: "/data"
                }
            });
            
            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(oModel);
            
            // Set viz properties
            oVizFrame.setVizProperties({
                plotArea: {
                    colorPalette: ["#FAFAFA", "#5899DA"]
                },
                title: {
                    visible: false
                }
            });
            
            // Feed dimensions and measures
            var oFeedColor = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "color",
                type: "Measure",
                values: ["Activity"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis",
                type: "Dimension",
                values: ["Day"]
            });
            var oFeedCategoryAxis2 = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis2",
                type: "Dimension",
                values: ["Hour"]
            });
            
            oVizFrame.addFeed(oFeedColor);
            oVizFrame.addFeed(oFeedCategoryAxis);
            oVizFrame.addFeed(oFeedCategoryAxis2);
        },

        /**
         * Opens interaction dialog for an agent.
         * @param {object} oAgent the agent object
         * @private
         */
        _openInteractionDialog: function(oAgent) {
            if (!this._oInteractionDialog) {
                this._oInteractionDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.view.fragments.AgentInteraction",
                    this
                );
                this.getView().addDependent(this._oInteractionDialog);
            }
            
            // Set agent data
            var oDialogModel = new JSONModel({
                agent: oAgent,
                message: "",
                serviceType: ""
            });
            this._oInteractionDialog.setModel(oDialogModel, "interaction");
            
            this._oInteractionDialog.open();
        },

        /**
         * Creates sort dialog.
         * @private
         */
        _createSortDialog: function() {
            this._oSortDialog = sap.ui.xmlfragment(
                "a2a.network.fiori.view.fragments.AgentSort",
                this
            );
            this.getView().addDependent(this._oSortDialog);
        }
    });
});