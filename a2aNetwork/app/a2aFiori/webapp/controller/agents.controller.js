sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log"
], (BaseController, MessageToast, MessageBox, formatter, Filter, FilterOperator, Log) => {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Agents", {
        formatter,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit() {
            // Control state model
            const oViewModel = new sap.ui.model.json.JSONModel({
                busy: false,
                delay: 0,
                countAll: 0,
                countActive: 0,
                countInactive: 0,
                reputationFilters: {
                    minReputation: 0,
                    maxReputation: 200,
                    badgeFilter: ""
                }
            });
            this.setModel(oViewModel, "agentsView");

            // Attach route matched handler
            this.getRouter().getRoute("agents").attachPatternMatched(this._onRouteMatched, this);

            Log.info("Agents controller initialized");
        },

        /* =========================================================== */
        /* event handlers                                              */
        /* =========================================================== */

        /**
         * Event handler when the share by E-Mail button has been clicked
         * @public
         */
        onShareEmailPress() {
            const oViewModel = this.getModel("agentsView");
            sap.m.URLHelper.triggerEmail(
                null,
                oViewModel.getProperty("/shareSendEmailSubject"),
                oViewModel.getProperty("/shareSendEmailMessage")
            );
        },

        /**
         * Event handler for refresh event. Re-reads data from the backend.
         * @public
         */
        onRefresh() {
            const oTable = this.byId("agentsTable");
            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }

            const oBinding = oTable.getBinding("items");
            if (oBinding) {
                oBinding.refresh();
                Log.debug("Agents list refreshed");
            }
        },

        /**
         * Event handler for search field liveChange event.
         * @param {sap.ui.base.Event} oEvent pattern match event
         * @public
         */
        onSearch(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("agentsTable");

            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }

            const oBinding = oTable.getBinding("items");
            const aFilters = [];

            if (sQuery) {
                aFilters.push(new Filter({
                    filters: [
                        new Filter("name", FilterOperator.Contains, sQuery),
                        new Filter("address", FilterOperator.Contains, sQuery)
                    ],
                    and: false
                }));
            }

            oBinding.filter(aFilters, "Application");
            Log.debug("Search applied", { query: sQuery, filters: aFilters.length });
        },

        /**
         * Event handler for filter select change.
         * @param {sap.ui.base.Event} oEvent the select change event
         * @public
         */
        onFilterChange(oEvent) {
            const sKey = oEvent.getSource().getSelectedKey();
            const oTable = this.byId("agentsTable");

            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }

            const oBinding = oTable.getBinding("items");
            const aFilters = [];

            switch (sKey) {
            case "active":
                aFilters.push(new Filter("isActive", FilterOperator.EQ, true));
                break;
            case "inactive":
                aFilters.push(new Filter("isActive", FilterOperator.EQ, false));
                break;
            default:
                // No filter for "all"
                break;
            }

            oBinding.filter(aFilters, "Application");
            Log.debug("Filter applied", { filter: sKey });
        },

        /**
         * Event handler for register agent button press.
         * @public
         */
        onRegisterAgent() {
            Log.info("Register agent initiated");

            // Navigate to agent registration dialog or view
            if (!this._oRegisterDialog) {
                this._createRegisterDialog();
            }
            this._oRegisterDialog.open();
        },

        /**
         * Event handler for table item press.
         * @param {sap.ui.base.Event} oEvent the table item press event
         * @public
         */
        onAgentPress(oEvent) {
            const oItem = oEvent.getSource();
            const oContext = oItem.getBindingContext();

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

        /**
         * Event handler for edit action button press.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onEdit(oEvent) {
            const oContext = oEvent.getSource().getBindingContext();

            if (!oContext) {
                Log.error("No binding context found");
                return;
            }

            const sAgentName = oContext.getProperty("name");
            Log.info("Edit agent initiated", { agent: sAgentName });

            MessageToast.show(this.getResourceBundle().getText("editAgentMessage", [sAgentName]));
        },

        /**
         * Event handler for sync action button press.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onSync(oEvent) {
            const oContext = oEvent.getSource().getBindingContext();

            if (!oContext) {
                Log.error("No binding context found");
                return;
            }

            const sAgentId = oContext.getProperty("ID");
            const oModel = this.getModel();
            const oViewModel = this.getModel("agentsView");

            Log.info("Blockchain sync initiated", { agentId: sAgentId });

            // Set busy state
            oViewModel.setProperty("/busy", true);

            // Call blockchain sync action
            oModel.callFunction(`/Agents(${ sAgentId })/registerOnBlockchain`, {
                method: "POST",
                success: function(_oData) {
                    oViewModel.setProperty("/busy", false);
                    MessageToast.show(this.getResourceBundle().getText("agentSyncSuccess"));
                    Log.info("Agent synced with blockchain", { agentId: sAgentId });
                }.bind(this),
                error: function(oError) {
                    oViewModel.setProperty("/busy", false);
                    const sMessage = this._createErrorMessage(oError);
                    MessageBox.error(sMessage);
                    Log.error("Agent sync failed", { agentId: sAgentId, error: sMessage });
                }.bind(this)
            });
        },

        /**
         * Event handler for export button press.
         * @public
         */
        onExport() {
            Log.info("Export functionality requested");
            MessageToast.show(this.getResourceBundle().getText("exportComingSoon"));
        },

        /**
         * Event handler for table settings button press.
         * @public
         */
        onTableSettings() {
            Log.info("Table settings requested");
            MessageToast.show(this.getResourceBundle().getText("tableSettingsComingSoon"));
        },

        /**
         * Event handler for endorsing an agent.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onEndorseAgent(oEvent) {
            const oContext = oEvent.getSource().getBindingContext();

            if (!oContext) {
                Log.error("No binding context found");
                return;
            }

            const sAgentId = oContext.getProperty("ID");
            const sAgentName = oContext.getProperty("name");

            // Open endorsement dialog
            this._openEndorsementDialog(sAgentId, sAgentName);
        },

        /**
         * Event handler for viewing reputation details.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onViewReputationDetails(oEvent) {
            const oContext = oEvent.getSource().getBindingContext();

            if (!oContext) {
                Log.error("No binding context found");
                return;
            }

            const sAgentId = oContext.getProperty("ID");

            // Navigate to reputation details page
            this.getRouter().navTo("reputationDetails", {
                agentId: sAgentId
            });
        },

        /**
         * Event handler for reputation filter changes.
         * @param {sap.ui.base.Event} oEvent the filter change event
         * @public
         */
        onReputationFilterChange(oEvent) {
            this._applyReputationFilters();
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Binds the view to the object path.
         * @function
         * @param {sap.ui.base.Event} oEvent pattern match event in route 'agents'
         * @private
         */
        _onRouteMatched(_oEvent) {
            Log.debug("Agents route matched");

            // Refresh the binding
            this.onRefresh();

            // Update counts
            this._updateListItemCounts();
        },

        /**
         * Updates the item counts in the view model.
         * @private
         */
        _updateListItemCounts() {
            const oModel = this.getModel();
            const oViewModel = this.getModel("agentsView");

            if (!oModel || !oModel.read) {
                Log.warning("OData model not available");
                return;
            }

            // Read counts
            oModel.read("/Agents/$count", {
                success(iCount) {
                    oViewModel.setProperty("/countAll", iCount);
                },
                error(oError) {
                    Log.error("Failed to read total count", oError);
                }
            });

            oModel.read("/Agents/$count", {
                filters: [new Filter("isActive", FilterOperator.EQ, true)],
                success(iCount) {
                    oViewModel.setProperty("/countActive", iCount);
                },
                error(oError) {
                    Log.error("Failed to read active count", oError);
                }
            });

            oModel.read("/Agents/$count", {
                filters: [new Filter("isActive", FilterOperator.EQ, false)],
                success(iCount) {
                    oViewModel.setProperty("/countInactive", iCount);
                },
                error(oError) {
                    Log.error("Failed to read inactive count", oError);
                }
            });
        },

        /**
         * Creates the agent registration dialog.
         * @private
         */
        _createRegisterDialog() {
            Log.debug("Creating register dialog");

            // Create dialog via fragment
            sap.ui.core.Fragment.load({
                id: this.getView().getId(),
                name: "a2a.network.fiori.view.fragments.RegisterAgent",
                controller: this
            }).then((oDialog) => {
                this._oRegisterDialog = oDialog;
                this.getView().addDependent(oDialog);

                // Set initial model
                const oDialogModel = new sap.ui.model.json.JSONModel({
                    name: "",
                    address: "",
                    endpoint: "",
                    description: ""
                });
                oDialog.setModel(oDialogModel, "register");

                Log.debug("Register dialog created");
            });
        },

        /**
         * Opens the endorsement dialog for a specific agent.
         * @param {string} sAgentId the agent ID to endorse
         * @param {string} sAgentName the agent name
         * @private
         */
        _openEndorsementDialog(sAgentId, sAgentName) {
            if (!this._oEndorsementDialog) {
                this._createEndorsementDialog();
            }

            // Set dialog model data
            const oDialogModel = new sap.ui.model.json.JSONModel({
                toAgentId: sAgentId,
                toAgentName: sAgentName,
                amount: 5,
                reason: "",
                description: "",
                maxAmount: 10 // This should be calculated based on endorser's reputation
            });

            this._oEndorsementDialog.setModel(oDialogModel, "endorsement");
            this._oEndorsementDialog.open();
        },

        /**
         * Creates the endorsement dialog.
         * @private
         */
        _createEndorsementDialog() {
            Log.debug("Creating endorsement dialog");

            // Create dialog via fragment
            sap.ui.core.Fragment.load({
                id: this.getView().getId(),
                name: "a2a.network.fiori.view.fragments.EndorseAgent",
                controller: this
            }).then((oDialog) => {
                this._oEndorsementDialog = oDialog;
                this.getView().addDependent(oDialog);

                Log.debug("Endorsement dialog created");
            });
        },

        /**
         * Handles endorsement dialog confirmation.
         * @public
         */
        onEndorsementConfirm() {
            const oDialogModel = this._oEndorsementDialog.getModel("endorsement");
            const oData = oDialogModel.getData();

            if (!oData.reason) {
                MessageBox.error("Please select a reason for endorsement");
                return;
            }

            const oModel = this.getModel();
            const oViewModel = this.getModel("agentsView");

            // Set busy state
            oViewModel.setProperty("/busy", true);

            // Call endorsement action
            oModel.callFunction(`/Agents(${ oData.toAgentId })/endorsePeer`, {
                method: "POST",
                urlParameters: {
                    toAgentId: oData.toAgentId,
                    amount: oData.amount,
                    reason: oData.reason,
                    description: oData.description
                },
                success: function(oResponse) {
                    oViewModel.setProperty("/busy", false);
                    MessageToast.show("Agent endorsed successfully!");
                    this._oEndorsementDialog.close();
                    this.onRefresh(); // Refresh to show updated reputation
                }.bind(this),
                error: function(oError) {
                    oViewModel.setProperty("/busy", false);
                    const sMessage = this._createErrorMessage(oError);
                    MessageBox.error(sMessage);
                }.bind(this)
            });
        },

        /**
         * Handles endorsement dialog cancellation.
         * @public
         */
        onEndorsementCancel() {
            this._oEndorsementDialog.close();
        },

        /**
         * Applies reputation-based filters to the agents table.
         * @private
         */
        _applyReputationFilters() {
            const oTable = this.byId("agentsTable");
            const oBinding = oTable.getBinding("items");
            const oViewModel = this.getModel("agentsView");
            const oFilters = oViewModel.getProperty("/reputationFilters");

            const aFilters = [];

            // Min reputation filter
            if (oFilters.minReputation > 0) {
                aFilters.push(new Filter("reputation", FilterOperator.GE, oFilters.minReputation));
            }

            // Max reputation filter
            if (oFilters.maxReputation < 200) {
                aFilters.push(new Filter("reputation", FilterOperator.LE, oFilters.maxReputation));
            }

            // Badge filter (would need to implement badge calculation)
            if (oFilters.badgeFilter) {
                // This would require a custom filter function or server-side implementation
                Log.info("Badge filter not yet implemented", { badge: oFilters.badgeFilter });
            }

            oBinding.filter(aFilters);
            Log.info("Reputation filters applied", { filters: oFilters });
        }
    });
});