sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment",
    "sap/ui/core/format/DateFormat",
    "../utils/OfflineManager"
], (BaseController, MessageToast, MessageBox, JSONModel, Fragment, DateFormat, OfflineManager) => {
    "use strict";
/* global  */

    /**
     * Conflict Resolution Controller
     * Handles conflict resolution for offline data synchronization
     */
    return BaseController.extend("sap.a2a.controller.ConflictResolutionController", {

        /* =========================================================== */
        /* Lifecycle Methods                                          */
        /* =========================================================== */

        onInit: function () {
            // Initialize conflict model
            this._initializeConflictModel();
            
            // Setup formatters
            this._setupFormatters();
        },

        /* =========================================================== */
        /* Event Handlers                                             */
        /* =========================================================== */

        /**
         * Handle conflict selection change
         */
        onConflictSelectionChange: function (oEvent) {
            const oSelectedItem = oEvent.getParameter("listItem");
            const oConflictModel = this.getModel("conflict");
            
            if (oSelectedItem) {
                const oBindingContext = oSelectedItem.getBindingContext("conflict");
                const oSelectedConflict = oBindingContext.getObject();
                
                oConflictModel.setProperty("/selectedConflict", oSelectedConflict);
            } else {
                oConflictModel.setProperty("/selectedConflict", null);
            }
        },

        /**
         * Handle conflict item press (for mobile)
         */
        onConflictItemPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("conflict");
            const oSelectedConflict = oBindingContext.getObject();
            const oConflictModel = this.getModel("conflict");
            
            oConflictModel.setProperty("/selectedConflict", oSelectedConflict);
            
            // Expand conflict details panel
            const oPanel = this.byId("conflictDetailsPanel");
            if (oPanel) {
                oPanel.setExpanded(true);
            }
        },

        /**
         * Handle resolve conflict button press
         */
        onResolveConflict: function (oEvent) {
            const oButton = oEvent.getSource();
            const sResolution = oButton.data("resolution");
            const oConflictModel = this.getModel("conflict");
            const oSelectedConflict = oConflictModel.getProperty("/selectedConflict");
            
            if (!oSelectedConflict) {
                MessageToast.show(this.getResourceBundle().getText("offline.noConflictSelected"));
                return;
            }
            
            this._resolveConflict(oSelectedConflict, sResolution);
        },

        /**
         * Handle manual merge button press
         */
        onManualMerge: function () {
            const oConflictModel = this.getModel("conflict");
            const oSelectedConflict = oConflictModel.getProperty("/selectedConflict");
            
            if (!oSelectedConflict) {
                MessageToast.show(this.getResourceBundle().getText("offline.noConflictSelected"));
                return;
            }
            
            this._openManualMergeDialog(oSelectedConflict);
        },

        /**
         * Handle skip conflict button press
         */
        onSkipConflict: function () {
            const oConflictModel = this.getModel("conflict");
            const oSelectedConflict = oConflictModel.getProperty("/selectedConflict");
            
            if (!oSelectedConflict) {
                MessageToast.show(this.getResourceBundle().getText("offline.noConflictSelected"));
                return;
            }
            
            this._skipConflict(oSelectedConflict);
        },

        /**
         * Handle resolve all conflicts button press
         */
        onResolveAllConflicts: function () {
            const oAutoResolutionGroup = this.byId("autoResolutionGroup");
            const iSelectedIndex = oAutoResolutionGroup.getSelectedIndex();
            const bApplyToAll = this.byId("applyToAllCheckBox").getSelected();
            
            if (!bApplyToAll) {
                MessageBox.warning(this.getResourceBundle().getText("offline.enableApplyToAllWarning"));
                return;
            }
            
            let sStrategy = "manual";
            switch (iSelectedIndex) {
                case 0: sStrategy = "clientWins"; break;
                case 1: sStrategy = "serverWins"; break;
                case 2: sStrategy = "timestampWins"; break;
                case 3: sStrategy = "manual"; break;
            }
            
            if (sStrategy === "manual") {
                MessageBox.warning(this.getResourceBundle().getText("offline.cannotAutoResolveManual"));
                return;
            }
            
            this._resolveAllConflicts(sStrategy);
        },

        /**
         * Handle close conflict dialog
         */
        onCloseConflictDialog: function () {
            const oDialog = this.byId("conflictResolutionDialog");
            if (oDialog) {
                oDialog.close();
            }
        },

        /* =========================================================== */
        /* Public Methods                                             */
        /* =========================================================== */

        /**
         * Open conflict resolution dialog
         */
        openConflictResolutionDialog: function (aConflicts) {
            this._loadConflicts(aConflicts);
            
            if (!this._oConflictDialog) {
                Fragment.load({
                    name: "sap.a2a.view.fragments.ConflictResolutionDialog",
                    controller: this
                }).then((oDialog) => {
                    this._oConflictDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._oConflictDialog.open();
            }
        },

        /* =========================================================== */
        /* Private Methods                                            */
        /* =========================================================== */

        /**
         * Initialize conflict model
         */
        _initializeConflictModel: function () {
            const oConflictModel = new JSONModel({
                conflicts: [],
                selectedConflict: null,
                resolutionInProgress: false
            });
            this.setModel(oConflictModel, "conflict");
        },

        /**
         * Setup formatters for UI binding
         */
        _setupFormatters: function () {
            // Date formatter
            this.formatDate = function (oDate) {
                if (!oDate) {
return "";
}
                const oDateFormat = DateFormat.getDateInstance({
                    style: "medium"
                });
                return oDateFormat.format(new Date(oDate));
            };

            // Date-time formatter
            this.formatDateTime = function (oDate) {
                if (!oDate) {
return "";
}
                const oDateTimeFormat = DateFormat.getDateTimeInstance({
                    style: "medium"
                });
                return oDateTimeFormat.format(new Date(oDate));
            };

            // Conflict status formatter
            this.formatConflictStatus = function (sStatus) {
                switch (sStatus) {
                    case "pending": return "Warning";
                    case "resolved": return "Success";
                    case "skipped": return "Information";
                    default: return "None";
                }
            };

            // Conflict data formatter (for display)
            this.formatConflictData = function (oData) {
                if (!oData) {
return "";
}
                
                try {
                    const sJson = JSON.stringify(oData, null, 2);
                    return this._syntaxHighlightJson(sJson);
                } catch (error) {
                    return String(oData);
                }
            }.bind(this);

            // Conflicts table title formatter
            this.formatConflictsTableTitle = function (aConflicts, _sTemplate) {
                const iCount = aConflicts ? aConflicts.length : 0;
                return this.getResourceBundle().getText("offline.conflictsTableTitle", [iCount]);
            }.bind(this);
        },

        /**
         * Load conflicts into model
         */
        _loadConflicts: function (aConflicts) {
            const oConflictModel = this.getModel("conflict");
            
            // Process conflicts to ensure proper structure
            const aProcessedConflicts = (aConflicts || []).map((oConflict, iIndex) => {
                return {
                    id: oConflict.id || `conflict_${iIndex}`,
                    entityType: oConflict.entityType || "Unknown",
                    entityId: oConflict.entityId || "Unknown",
                    timestamp: oConflict.timestamp || new Date(),
                    status: oConflict.status || "pending",
                    clientData: oConflict.clientData || {},
                    serverData: oConflict.serverData || {},
                    conflictReason: oConflict.conflictReason || "Data mismatch"
                };
            });
            
            oConflictModel.setProperty("/conflicts", aProcessedConflicts);
            oConflictModel.setProperty("/selectedConflict", null);
        },

        /**
         * Resolve a single conflict
         */
        _resolveConflict: function (oConflict, sResolution) {
            const oConflictModel = this.getModel("conflict");
            
            this.showBusyIndicator(0);
            oConflictModel.setProperty("/resolutionInProgress", true);
            
            let oResolvedData;
            switch (sResolution) {
                case "useClient":
                    oResolvedData = oConflict.clientData;
                    break;
                case "useServer":
                    oResolvedData = oConflict.serverData;
                    break;
                default:
                    this.hideBusyIndicator();
                    oConflictModel.setProperty("/resolutionInProgress", false);
                    MessageBox.error(this.getResourceBundle().getText("offline.invalidResolutionStrategy"));
                    return;
            }
            
            // Apply resolution through OfflineManager
            OfflineManager.resolveConflict(oConflict.id, oResolvedData)
                .then(() => {
                    MessageToast.show(this.getResourceBundle().getText("offline.conflictResolved"));
                    this._removeResolvedConflict(oConflict.id);
                })
                .catch((oError) => {
                    MessageBox.error(this.getResourceBundle().getText("offline.conflictResolutionError", [oError.message]));
                })
                .finally(() => {
                    this.hideBusyIndicator();
                    oConflictModel.setProperty("/resolutionInProgress", false);
                });
        },

        /**
         * Skip a conflict (mark as manually resolved)
         */
        _skipConflict: function (oConflict) {
            const _oConflictModel = this.getModel("conflict");
            
            MessageBox.confirm(this.getResourceBundle().getText("offline.skipConflictConfirm"), {
                title: this.getResourceBundle().getText("offline.skipConflictTitle"),
                onClose: (sAction) => {
                    if (sAction === MessageBox.Action.OK) {
                        this.showBusyIndicator(0);
                        
                        OfflineManager.skipConflict(oConflict.id)
                            .then(() => {
                                MessageToast.show(this.getResourceBundle().getText("offline.conflictSkipped"));
                                this._removeResolvedConflict(oConflict.id);
                            })
                            .catch((oError) => {
                                MessageBox.error(this.getResourceBundle().getText("offline.skipConflictError", [oError.message]));
                            })
                            .finally(() => {
                                this.hideBusyIndicator();
                            });
                    }
                }
            });
        },

        /**
         * Resolve all conflicts using auto-resolution strategy
         */
        _resolveAllConflicts: function (sStrategy) {
            const oConflictModel = this.getModel("conflict");
            const aConflicts = oConflictModel.getProperty("/conflicts");
            
            if (!aConflicts || aConflicts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("offline.noConflictsToResolve"));
                return;
            }
            
            const sConfirmMessage = this.getResourceBundle().getText("offline.resolveAllConfirm", [
                aConflicts.length,
                this._getStrategyDisplayName(sStrategy)
            ]);
            
            MessageBox.confirm(sConfirmMessage, {
                title: this.getResourceBundle().getText("offline.resolveAllTitle"),
                onClose: (sAction) => {
                    if (sAction === MessageBox.Action.OK) {
                        this._performBulkResolution(aConflicts, sStrategy);
                    }
                }
            });
        },

        /**
         * Perform bulk conflict resolution
         */
        _performBulkResolution: function (aConflicts, sStrategy) {
            const oConflictModel = this.getModel("conflict");
            
            this.showBusyIndicator(0);
            oConflictModel.setProperty("/resolutionInProgress", true);
            
            let iResolved = 0;
            let iErrors = 0;
            
            const aPromises = aConflicts.map((oConflict) => {
                const oResolvedData = this._getResolvedDataByStrategy(oConflict, sStrategy);
                
                return OfflineManager.resolveConflict(oConflict.id, oResolvedData)
                    .then(() => {
                        iResolved++;
                    })
                    .catch((oError) => {
                        console.error("Failed to resolve conflict:", oConflict.id, oError);
                        iErrors++;
                    });
            });
            
            Promise.allSettled(aPromises)
                .then(() => {
                    const sMessage = this.getResourceBundle().getText("offline.bulkResolutionComplete", [
                        iResolved,
                        iErrors
                    ]);
                    
                    if (iErrors === 0) {
                        MessageToast.show(sMessage);
                    } else {
                        MessageBox.warning(sMessage);
                    }
                    
                    // Refresh conflict list
                    this._refreshConflictList();
                })
                .finally(() => {
                    this.hideBusyIndicator();
                    oConflictModel.setProperty("/resolutionInProgress", false);
                });
        },

        /**
         * Get resolved data based on strategy
         */
        _getResolvedDataByStrategy: function (oConflict, sStrategy) {
            switch (sStrategy) {
                case "clientWins":
                    return oConflict.clientData;
                case "serverWins":
                    return oConflict.serverData;
                case "timestampWins": {
                    const clientTime = new Date(oConflict.clientData.modifiedAt || 0).getTime();
                    const serverTime = new Date(oConflict.serverData.modifiedAt || 0).getTime();
                    return clientTime > serverTime ? oConflict.clientData : oConflict.serverData;
                }
                default:
                    return oConflict.serverData; // Default to server
            }
        },

        /**
         * Get display name for resolution strategy
         */
        _getStrategyDisplayName: function (sStrategy) {
            switch (sStrategy) {
                case "clientWins": return this.getResourceBundle().getText("offline.clientWins");
                case "serverWins": return this.getResourceBundle().getText("offline.serverWins");
                case "timestampWins": return this.getResourceBundle().getText("offline.timestampWins");
                default: return sStrategy;
            }
        },

        /**
         * Open manual merge dialog
         */
        _openManualMergeDialog: function (_oConflict) {
            // This would open a separate dialog for manual field-by-field merging
            MessageToast.show(this.getResourceBundle().getText("offline.manualMergeNotImplemented"));
            // TODO: Implement manual merge dialog
        },

        /**
         * Remove resolved conflict from list
         */
        _removeResolvedConflict: function (sConflictId) {
            const oConflictModel = this.getModel("conflict");
            const aConflicts = oConflictModel.getProperty("/conflicts");
            
            const aFilteredConflicts = aConflicts.filter(oConflict => oConflict.id !== sConflictId);
            oConflictModel.setProperty("/conflicts", aFilteredConflicts);
            
            // Clear selection if the resolved conflict was selected
            const oSelectedConflict = oConflictModel.getProperty("/selectedConflict");
            if (oSelectedConflict && oSelectedConflict.id === sConflictId) {
                oConflictModel.setProperty("/selectedConflict", null);
            }
            
            // Close dialog if no more conflicts
            if (aFilteredConflicts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("offline.allConflictsResolved"));
                this.onCloseConflictDialog();
            }
        },

        /**
         * Refresh conflict list from OfflineManager
         */
        _refreshConflictList: function () {
            OfflineManager.getConflicts()
                .then((aConflicts) => {
                    this._loadConflicts(aConflicts);
                })
                .catch((oError) => {
                    console.error("Failed to refresh conflict list:", oError);
                });
        },

        /**
         * Apply JSON syntax highlighting
         */
        _syntaxHighlightJson: function (sJson) {
            return sJson
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, 
                    (match) => {
                        let cls = 'json-number';
                        if (/^"/.test(match)) {
                            if (/:$/.test(match)) {
                                cls = 'json-key';
                            } else {
                                cls = 'json-string';
                            }
                        } else if (/true|false/.test(match)) {
                            cls = 'json-boolean';
                        } else if (/null/.test(match)) {
                            cls = 'json-null';
                        }
                        return `<span class="${  cls  }">${  match  }</span>`;
                    });
        }
    });
});