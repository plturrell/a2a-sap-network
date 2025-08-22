sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent11.ext.controller.ObjectPageExt", {
        
        // Execute Query Action
        onExecuteQuery: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.executionStatus === 'executing') {
                MessageToast.show(this.getResourceBundle().getText("msg.queryAlreadyExecuting"));
                return;
            }

            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLStatement"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.executeQueryConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeQuery(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Validate Query Action
        onValidateQuery: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLStatement"));
                return;
            }

            this._validateQuery(oContext);
        },

        // Optimize Query Action
        onOptimizeQuery: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLToOptimize"));
                return;
            }

            if (!this._queryOptimizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.QueryOptimizer",
                    controller: this
                }).then(function(oDialog) {
                    this._queryOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadOptimizationData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadOptimizationData(oContext);
                this._queryOptimizer.open();
            }
        },

        // Explain Plan Action
        onExplainPlan: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLStatement"));
                return;
            }

            if (!this._explainPlanDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.ExplainPlan",
                    controller: this
                }).then(function(oDialog) {
                    this._explainPlanDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._generateExplainPlan(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._generateExplainPlan(oContext);
                this._explainPlanDialog.open();
            }
        },

        // Format Query Action
        onFormatQuery: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLToFormat"));
                return;
            }

            this._formatQuery(oContext);
        },

        // Save as Template Action
        onSaveAsTemplate: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLToSave"));
                return;
            }

            if (!this._saveTemplateDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.SaveTemplate",
                    controller: this
                }).then(function(oDialog) {
                    this._saveTemplateDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._saveTemplateDialog.open();
            }
        },

        // Export Results Action
        onExportResults: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.executionStatus !== 'completed' || !oData.rowsReturned || oData.rowsReturned === 0) {
                MessageToast.show(this.getResourceBundle().getText("error.noResultsToExport"));
                return;
            }

            if (!this._exportResultsDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.ExportResults",
                    controller: this
                }).then(function(oDialog) {
                    this._exportResultsDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._exportResultsDialog.open();
            }
        },

        // Schedule Query Action
        onScheduleQuery: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noSQLToSchedule"));
                return;
            }

            if (!this._scheduleQueryDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.ScheduleQuery",
                    controller: this
                }).then(function(oDialog) {
                    this._scheduleQueryDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._scheduleQueryDialog.open();
            }
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeQueryMonitoring();
        },

        _initializeQueryMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const queryId = oContext.getObject().queryId;
            
            // Subscribe to query execution updates for this specific query
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = new EventSource(`http://localhost:8011/sql/${queryId}/stream`);
                
                this._eventSource.addEventListener('query-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateQueryProgress(data);
                });

                this._eventSource.addEventListener('query-completed', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleQueryCompleted(data);
                });

                this._eventSource.addEventListener('query-error', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleQueryError(data);
                });

                this._eventSource.addEventListener('optimization-update', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleOptimizationUpdate(data);
                });

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(queryId);
            }
        },

        _initializePolling: function(queryId) {
            this._pollInterval = setInterval(() => {
                this._refreshQueryData();
            }, 2000);
        },

        _executeQuery: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = oContext.getObject().queryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.queryExecutionStarted"));
            
            oModel.callFunction("/ExecuteQuery", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.queryExecuted"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.queryExecutionFailed", [error.message]));
                }.bind(this)
            });
        },

        _validateQuery: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = oContext.getObject().queryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.queryValidationStarted"));
            
            oModel.callFunction("/ValidateQuery", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.queryValidated"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.validationFailed"));
                }.bind(this)
            });
        },

        _formatQuery: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = oContext.getObject().queryId;
            
            oModel.callFunction("/FormatQuery", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.queryFormatted"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.formattingFailed"));
                }.bind(this)
            });
        },

        _loadOptimizationData: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = oContext.getObject().queryId;
            
            oModel.callFunction("/GetOptimizationSuggestions", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    this._displayOptimizationSuggestions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingOptimizationData"));
                }.bind(this)
            });
        },

        _generateExplainPlan: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = oContext.getObject().queryId;
            
            oModel.callFunction("/GenerateExplainPlan", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    this._displayExplainPlan(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.generatingExplainPlan"));
                }.bind(this)
            });
        },

        _updateQueryProgress: function(data) {
            // Update progress indicators for query execution
            const oProgressIndicator = this.getView().byId("queryProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _handleQueryCompleted: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.queryExecuted"));
            this._refreshQueryData();
            
            // Show execution metrics if available
            if (data.executionTime) {
                MessageToast.show(this.getResourceBundle().getText("msg.executionCompleted", [data.executionTime]));
            }
        },

        _handleQueryError: function(data) {
            MessageBox.error(this.getResourceBundle().getText("error.queryExecutionFailed", [data.error]));
            this._refreshQueryData();
        },

        _handleOptimizationUpdate: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.optimizationUpdate", [data.improvement]));
            this._refreshQueryData();
        },

        _refreshQueryData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayOptimizationSuggestions: function(data) {
            // Display optimization suggestions in dialog
        },

        _displayExplainPlan: function(data) {
            // Display query execution plan
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._eventSource) {
                this._eventSource.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});