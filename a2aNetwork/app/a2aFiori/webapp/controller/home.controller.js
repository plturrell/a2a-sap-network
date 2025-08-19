sap.ui.define([
    "./BaseController",
    "sap/ui/core/routing/History",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter",
    "sap/base/Log",
    "sap/ui/model/json/JSONModel"
], function(BaseController, History, MessageToast, MessageBox, formatter, Log, JSONModel) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Home", {
        formatter,
        _iRefreshInterval: null,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit() {
            // Control state model
            const oViewModel = new JSONModel({
                busy: false,
                delay: 0,
                dashboardTitle: "",
                networkStatus: "Unknown",
                activeAgents: 0,
                totalAgents: 0,
                pendingMessages: 0,
                completedWorkflows: 0
            });
            this.setModel(oViewModel, "homeView");

            // Attach route matched handler
            this.getRouter().getRoute("home").attachPatternMatched(this._onRouteMatched, this);

            // Set refresh interval for dashboard (30 seconds)
            this._iRefreshInterval = setInterval(function() {
                this._refreshDashboard();
            }.bind(this), 30000);

            // Register for cleanup
            this._registerForCleanup(function() {
                if (this._iRefreshInterval) {
                    clearInterval(this._iRefreshInterval);
                    this._iRefreshInterval = null;
                }
            }.bind(this));

            Log.info("Home controller initialized");
        },

        /**
         * Called when the controller is destroyed.
         * @public
         */
        onExit() {
            // Clear refresh interval
            if (this._iRefreshInterval) {
                clearInterval(this._iRefreshInterval);
                this._iRefreshInterval = null;
                Log.debug("Dashboard refresh interval cleared");
            }

            // Call base implementation
            BaseController.prototype.onExit.apply(this, arguments);
        },

        /* =========================================================== */
        /* event handlers                                              */
        /* =========================================================== */

        /**
         * Event handler for blockchain sync button press.
         * @public
         */
        onSyncBlockchain() {
            const oModel = this.getModel();
            const oViewModel = this.getModel("homeView");

            if (!oModel) {
                Log.error("Model not available");
                return;
            }

            Log.info("Blockchain sync initiated from dashboard");

            // Set busy state
            oViewModel.setProperty("/busy", true);

            // Call sync function
            oModel.callFunction("/syncBlockchain", {
                method: "POST",
                success: function(oData) {
                    oViewModel.setProperty("/busy", false);

                    const oResult = oData.syncBlockchain;
                    if (oResult) {
                        const sMessage = this.getResourceBundle().getText("syncCompleteMessage",
                            [oResult.synced || 0, oResult.failed || 0]);
                        MessageToast.show(sMessage);
                        Log.info("Blockchain sync completed", oResult);
                    }

                    // Refresh dashboard data
                    this._refreshDashboard();
                }.bind(this),
                error: function(oError) {
                    oViewModel.setProperty("/busy", false);
                    const sMessage = this._createErrorMessage(oError);
                    MessageBox.error(sMessage);
                    Log.error("Blockchain sync failed", sMessage);
                }.bind(this)
            });
        },

        /**
         * Navigate to Agents view.
         * @public
         */
        onNavToAgents() {
            Log.debug("Navigating to agents");
            this.getRouter().navTo("agents");
        },

        /**
         * Navigate to Services view.
         * @public
         */
        onNavToServices() {
            Log.debug("Navigating to services");
            this.getRouter().navTo("services");
        },

        /**
         * Navigate to Workflows view.
         * @public
         */
        onNavToWorkflows() {
            Log.debug("Navigating to workflows");
            this.getRouter().navTo("workflows");
        },

        /**
         * Navigate to Analytics view.
         * @public
         */
        onNavToAnalytics() {
            Log.debug("Navigating to analytics");
            this.getRouter().navTo("analytics");
        },

        /**
         * Event handler for agent tile press.
         * @param {sap.ui.base.Event} oEvent the tile press event
         * @public
         */
        onAgentPress(oEvent) {
            const oItem = oEvent.getSource();
            const _oContext = oItem.getBindingContext();

            if (!oContext) {
                Log.error("No binding context found");
                return;
            }

            const sAgentId = oContext.getProperty("ID");
            Log.debug("Navigating to agent detail", { agentId: sAgentId });

            this.getRouter().navTo("agentDetail", {
                agentId: sAgentId
            });
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Binds the view to the object path.
         * @function
         * @param {sap.ui.base.Event} oEvent pattern match event in route 'home'
         * @private
         */
        _onRouteMatched(oEvent) {
            Log.debug("Home route matched");

            // Set dashboard title
            const oViewModel = this.getModel("homeView");
            oViewModel.setProperty("/dashboardTitle",
                this.getResourceBundle().getText("dashboardTitle"));

            // Refresh dashboard data
            this._refreshDashboard();
        },

        /**
         * Refreshes all dashboard data.
         * @private
         */
        _refreshDashboard() {
            Log.debug("Refreshing dashboard data");

            const oModel = this.getModel();
            if (!oModel) {
                Log.warning("Model not available for refresh");
                return;
            }

            // Refresh all bindings
            oModel.refresh();

            // Update network statistics
            this._loadNetworkStats();

            // Update dashboard KPIs
            this._updateDashboardKPIs();
        },

        /**
         * Loads network statistics.
         * @private
         */
        _loadNetworkStats() {
            const oComponent = this.getOwnerComponent();
            if (oComponent && oComponent._loadNetworkStats) {
                oComponent._loadNetworkStats();
                Log.debug("Network stats refresh triggered");
            }
        },

        /**
         * Updates dashboard KPI values.
         * @private
         */
        _updateDashboardKPIs() {
            const oModel = this.getModel();
            const oViewModel = this.getModel("homeView");

            if (!oModel || !oModel.read) {
                Log.warning("OData model not available");
                return;
            }

            // Get active agents count
            oModel.read("/Agents/$count", {
                filters: [new sap.ui.model.Filter("isActive", sap.ui.model.FilterOperator.EQ, true)],
                success(iCount) {
                    oViewModel.setProperty("/activeAgents", iCount);
                    Log.debug("Active agents count updated", iCount);
                },
                error(oError) {
                    Log.error("Failed to read active agents count", oError);
                }
            });

            // Get total agents count
            oModel.read("/Agents/$count", {
                success(iCount) {
                    oViewModel.setProperty("/totalAgents", iCount);
                    Log.debug("Total agents count updated", iCount);
                },
                error(oError) {
                    Log.error("Failed to read total agents count", oError);
                }
            });

            // Get pending messages count
            oModel.read("/Messages/$count", {
                filters: [new sap.ui.model.Filter("status", sap.ui.model.FilterOperator.EQ, "pending")],
                success(iCount) {
                    oViewModel.setProperty("/pendingMessages", iCount);
                    Log.debug("Pending messages count updated", iCount);
                },
                error(oError) {
                    Log.error("Failed to read pending messages count", oError);
                }
            });

            // Get completed workflows count (last 24 hours)
            const dYesterday = new Date();
            dYesterday.setDate(dYesterday.getDate() - 1);

            oModel.read("/WorkflowExecutions/$count", {
                filters: [
                    new sap.ui.model.Filter("status", sap.ui.model.FilterOperator.EQ, "completed"),
                    new sap.ui.model.Filter("completedAt", sap.ui.model.FilterOperator.GT, dYesterday)
                ],
                success(iCount) {
                    oViewModel.setProperty("/completedWorkflows", iCount);
                    Log.debug("Completed workflows count updated", iCount);
                },
                error(oError) {
                    Log.error("Failed to read completed workflows count", oError);
                }
            });
        }
    });
});