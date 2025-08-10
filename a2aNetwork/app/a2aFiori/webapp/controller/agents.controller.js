sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "../model/formatter",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log"
], function(BaseController, MessageToast, MessageBox, formatter, Filter, FilterOperator, Log) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Agents", {
        formatter: formatter,

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         * @public
         */
        onInit: function() {
            // Control state model
            var oViewModel = new sap.ui.model.json.JSONModel({
                busy: false,
                delay: 0,
                countAll: 0,
                countActive: 0,
                countInactive: 0
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
        onShareEmailPress: function() {
            var oViewModel = this.getModel("agentsView");
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
        onRefresh: function() {
            var oTable = this.byId("agentsTable");
            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }
            
            var oBinding = oTable.getBinding("items");
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
        onSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            var oTable = this.byId("agentsTable");
            
            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }
            
            var oBinding = oTable.getBinding("items");
            var aFilters = [];
            
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
        onFilterChange: function(oEvent) {
            var sKey = oEvent.getSource().getSelectedKey();
            var oTable = this.byId("agentsTable");
            
            if (!oTable) {
                Log.error("Agents table not found");
                return;
            }
            
            var oBinding = oTable.getBinding("items");
            var aFilters = [];
            
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
        onRegisterAgent: function() {
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
        onAgentPress: function(oEvent) {
            var oItem = oEvent.getSource();
            var oContext = oItem.getBindingContext();
            
            if (!oContext) {
                Log.error("No binding context found");
                return;
            }
            
            var sAgentId = oContext.getProperty("ID");
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
        onEdit: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            
            if (!oContext) {
                Log.error("No binding context found");
                return;
            }
            
            var sAgentName = oContext.getProperty("name");
            Log.info("Edit agent initiated", { agent: sAgentName });
            
            MessageToast.show(this.getResourceBundle().getText("editAgentMessage", [sAgentName]));
        },

        /**
         * Event handler for sync action button press.
         * @param {sap.ui.base.Event} oEvent the button press event
         * @public
         */
        onSync: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            
            if (!oContext) {
                Log.error("No binding context found");
                return;
            }
            
            var sAgentId = oContext.getProperty("ID");
            var oModel = this.getModel();
            var oViewModel = this.getModel("agentsView");
            
            Log.info("Blockchain sync initiated", { agentId: sAgentId });
            
            // Set busy state
            oViewModel.setProperty("/busy", true);
            
            // Call blockchain sync action
            oModel.callFunction("/Agents(" + sAgentId + ")/registerOnBlockchain", {
                method: "POST",
                success: function(_oData) {
                    oViewModel.setProperty("/busy", false);
                    MessageToast.show(this.getResourceBundle().getText("agentSyncSuccess"));
                    Log.info("Agent synced with blockchain", { agentId: sAgentId });
                }.bind(this),
                error: function(oError) {
                    oViewModel.setProperty("/busy", false);
                    var sMessage = this._createErrorMessage(oError);
                    MessageBox.error(sMessage);
                    Log.error("Agent sync failed", { agentId: sAgentId, error: sMessage });
                }.bind(this)
            });
        },

        /**
         * Event handler for export button press.
         * @public
         */
        onExport: function() {
            Log.info("Export functionality requested");
            MessageToast.show(this.getResourceBundle().getText("exportComingSoon"));
        },

        /**
         * Event handler for table settings button press.
         * @public
         */
        onTableSettings: function() {
            Log.info("Table settings requested");
            MessageToast.show(this.getResourceBundle().getText("tableSettingsComingSoon"));
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
        _onRouteMatched: function(_oEvent) {
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
        _updateListItemCounts: function() {
            var oModel = this.getModel();
            var oViewModel = this.getModel("agentsView");
            
            if (!oModel || !oModel.read) {
                Log.warning("OData model not available");
                return;
            }
            
            // Read counts
            oModel.read("/Agents/$count", {
                success: function(iCount) {
                    oViewModel.setProperty("/countAll", iCount);
                },
                error: function(oError) {
                    Log.error("Failed to read total count", oError);
                }
            });
            
            oModel.read("/Agents/$count", {
                filters: [new Filter("isActive", FilterOperator.EQ, true)],
                success: function(iCount) {
                    oViewModel.setProperty("/countActive", iCount);
                },
                error: function(oError) {
                    Log.error("Failed to read active count", oError);
                }
            });
            
            oModel.read("/Agents/$count", {
                filters: [new Filter("isActive", FilterOperator.EQ, false)],
                success: function(iCount) {
                    oViewModel.setProperty("/countInactive", iCount);
                },
                error: function(oError) {
                    Log.error("Failed to read inactive count", oError);
                }
            });
        },

        /**
         * Creates the agent registration dialog.
         * @private
         */
        _createRegisterDialog: function() {
            Log.debug("Creating register dialog");
            
            // Create dialog via fragment
            sap.ui.core.Fragment.load({
                id: this.getView().getId(),
                name: "a2a.network.fiori.view.fragments.RegisterAgent",
                controller: this
            }).then(function(oDialog) {
                this._oRegisterDialog = oDialog;
                this.getView().addDependent(oDialog);
                
                // Set initial model
                var oDialogModel = new sap.ui.model.json.JSONModel({
                    name: "",
                    address: "",
                    endpoint: "",
                    description: ""
                });
                oDialog.setModel(oDialogModel, "register");
                
                Log.debug("Register dialog created");
            }.bind(this));
        }
    });
});