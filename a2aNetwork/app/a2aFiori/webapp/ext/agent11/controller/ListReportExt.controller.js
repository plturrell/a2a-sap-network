sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "a2a/network/agent11/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent11.ext.controller.ListReportExt", {
        
        // SQL Dashboard Action
        onSQLDashboard: function() {
            if (!this._sqlDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.SQLDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._sqlDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDashboardData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._sqlDashboard.open();
            }
        },

        // Create New SQL Query
        onCreateQuery: function() {
            if (!this._createQueryDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.CreateSQLQuery",
                    controller: this
                }).then(function(oDialog) {
                    this._createQueryDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._createQueryDialog.open();
            }
        },

        // Query Builder Action
        onQueryBuilder: function() {
            if (!this._queryBuilder) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.QueryBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._queryBuilder = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadSchemaData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadSchemaData();
                this._queryBuilder.open();
            }
        },

        // Connection Manager Action
        onConnectionManager: function() {
            if (!this._connectionManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.ConnectionManager",
                    controller: this
                }).then(function(oDialog) {
                    this._connectionManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadConnectionData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadConnectionData();
                this._connectionManager.open();
            }
        },

        // Template Library Action
        onTemplateLibrary: function() {
            if (!this._templateLibrary) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.TemplateLibrary",
                    controller: this
                }).then(function(oDialog) {
                    this._templateLibrary = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadTemplateData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadTemplateData();
                this._templateLibrary.open();
            }
        },

        // Performance Analyzer Action
        onPerformanceAnalyzer: function() {
            if (!this._performanceAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPerformanceData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadPerformanceData();
                this._performanceAnalyzer.open();
            }
        },

        // Schema Explorer Action
        onSchemaExplorer: function() {
            if (!this._schemaExplorer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent11.ext.fragment.SchemaExplorer",
                    controller: this
                }).then(function(oDialog) {
                    this._schemaExplorer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadSchemaStructure();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadSchemaStructure();
                this._schemaExplorer.open();
            }
        },

        // Query Optimizer Action
        onQueryOptimizer: function() {
            const oBinding = this.base.getView().byId("fe::table::SQLQueries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectQueriesFirst"));
                return;
            }

            this._runBatchOptimization(aSelectedContexts);
        },

        // Execute Selected Queries
        onExecuteSelectedQueries: function() {
            const oBinding = this.base.getView().byId("fe::table::SQLQueries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectQueriesFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.executeQueriesConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeBatchQueries(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        // Validate Selected Queries
        onValidateSelectedQueries: function() {
            const oBinding = this.base.getView().byId("fe::table::SQLQueries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectQueriesFirst"));
                return;
            }

            this._validateBatchQueries(aSelectedContexts);
        },

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = SecurityUtils.createSecureWebSocket('ws://localhost:8011/sql/updates', {
                    onMessage: function(data) {
                        this._handleSQLUpdate(data);
                    }.bind(this)
                });

                this._ws.onclose = function() {
                    setTimeout(() => this._initializeWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshQueryData();
            }, 3000);
        },

        _handleSQLUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'QUERY_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.queryExecutionStarted"));
                    break;
                case 'QUERY_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.queryExecuted"));
                    this._refreshQueryData();
                    break;
                case 'QUERY_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("error.queryExecutionFailed", [SecurityUtils.escapeHTML(data.error)]));
                    break;
                case 'CONNECTION_STATUS':
                    this._updateConnectionStatus(data.connections);
                    break;
                case 'PERFORMANCE_UPDATE':
                    this._updatePerformanceMetrics(data.metrics);
                    break;
            }
        },

        _loadDashboardData: function() {
            const oModel = this.getView().getModel();
            
            // Load SQL execution statistics
            SecurityUtils.secureCallFunction(oModel, "/GetSQLStatistics", {
                success: function(data) {
                    this._updateDashboardCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingStatistics"));
                }.bind(this)
            });
        },

        _loadSchemaData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetDatabaseSchemas", {
                success: function(data) {
                    this._updateSchemaData(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingSchemaData"));
                }.bind(this)
            });
        },

        _loadConnectionData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetConnectionStatus", {
                success: function(data) {
                    this._updateConnectionData(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingConnectionData"));
                }.bind(this)
            });
        },

        _loadTemplateData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetQueryTemplates", {
                success: function(data) {
                    this._updateTemplateData(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingTemplateData"));
                }.bind(this)
            });
        },

        _loadPerformanceData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetPerformanceMetrics", {
                success: function(data) {
                    this._updatePerformanceCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingPerformanceData"));
                }.bind(this)
            });
        },

        _loadSchemaStructure: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetSchemaStructure", {
                success: function(data) {
                    this._updateSchemaTree(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingSchemaStructure"));
                }.bind(this)
            });
        },

        _runBatchOptimization: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aQueryIds = aSelectedContexts.map(ctx => ctx.getObject().queryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.optimizationStarted"));
            
            SecurityUtils.secureCallFunction(oModel, "/OptimizeQueries", {
                urlParameters: {
                    queryIds: SecurityUtils.sanitizeSQLParameter(aQueryIds.join(','))
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.queryOptimized"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.optimizationFailed"));
                }.bind(this)
            });
        },

        _executeBatchQueries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aQueryIds = aSelectedContexts.map(ctx => ctx.getObject().queryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.batchExecutionStarted", [aQueryIds.length]));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteBatchQueries", {
                urlParameters: {
                    queryIds: SecurityUtils.sanitizeSQLParameter(aQueryIds.join(','))
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.batchExecutionCompleted"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.batchExecutionFailed"));
                }.bind(this)
            });
        },

        _validateBatchQueries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aQueryIds = aSelectedContexts.map(ctx => ctx.getObject().queryId);
            
            SecurityUtils.secureCallFunction(oModel, "/ValidateQueries", {
                urlParameters: {
                    queryIds: SecurityUtils.sanitizeSQLParameter(aQueryIds.join(','))
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

        _refreshQueryData: function() {
            const oBinding = this.base.getView().byId("fe::table::SQLQueries::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateDashboardCharts: function(data) {
            // Update execution trends chart
            // Update database utilization chart
            // Update query type distribution chart
        },

        _updateSchemaData: function(data) {
            // Update schema dropdown options
        },

        _updateConnectionData: function(data) {
            // Update connection status indicators
        },

        _updateTemplateData: function(data) {
            // Update template library
        },

        _updatePerformanceCharts: function(data) {
            // Update performance metrics charts
        },

        _updateSchemaTree: function(data) {
            // Update schema tree structure
        },

        _updateConnectionStatus: function(connections) {
            // Real-time connection status updates
        },

        _updatePerformanceMetrics: function(metrics) {
            // Real-time performance metric updates
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._ws) {
                this._ws.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});