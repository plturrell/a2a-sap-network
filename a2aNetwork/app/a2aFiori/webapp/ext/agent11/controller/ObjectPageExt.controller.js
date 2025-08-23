sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent11/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent11.ext.controller.ObjectPageExt", {

        override: {
            onInit: function () {
                this._initializeCreateModel();
            }
        },

        _initializeCreateModel: function() {
            var oCreateData = {
                queryName: "",
                description: "",
                queryType: "",
                databaseType: "",
                priority: "medium",
                complexity: "moderate",
                queryNameState: "",
                queryNameStateText: "",
                queryTypeState: "",
                queryTypeStateText: "",
                databaseTypeState: "",
                databaseTypeStateText: "",
                sqlStatement: "",
                queryLanguage: "sql",
                dialectVersion: "",
                estimatedCost: "",
                indexUsage: "auto",
                connectionString: "",
                schemaName: "",
                databaseVersion: "",
                connectionPool: 10,
                timeoutSettings: 30,
                transactionMode: "auto",
                isolationLevel: "read_committed",
                autoCommit: true,
                dataClassification: "internal"
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.getView().setModel(oCreateModel, "create");
        },

        onCreateSQLQuery: function() {
            var oView = this.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent11.ext.fragment.CreateSQLQuery",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onQueryNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!sValue || sValue.length < 3) {
                oData.queryNameState = "Error";
                oData.queryNameStateText = "Query name must be at least 3 characters";
            } else if (sValue.length > 100) {
                oData.queryNameState = "Error";
                oData.queryNameStateText = "Query name must not exceed 100 characters";
            } else {
                oData.queryNameState = "Success";
                oData.queryNameStateText = "Valid query name";
            }
            
            oCreateModel.setData(oData);
        },

        onQueryTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.queryTypeState = "Success";
                oData.queryTypeStateText = "Query type selected";
                
                // Smart suggestions based on query type
                switch (sValue) {
                    case "select":
                        oData.transactionMode = "read_only";
                        oData.autoCommit = true;
                        oData.isolationLevel = "read_committed";
                        break;
                    case "insert":
                    case "update":
                    case "delete":
                        oData.transactionMode = "read_write";
                        oData.autoCommit = false;
                        oData.isolationLevel = "repeatable_read";
                        break;
                    case "create":
                    case "alter":
                    case "drop":
                        oData.transactionMode = "manual";
                        oData.autoCommit = false;
                        oData.isolationLevel = "serializable";
                        break;
                    case "procedure":
                    case "function":
                        oData.complexity = "complex";
                        oData.timeoutSettings = 60;
                        break;
                }
            } else {
                oData.queryTypeState = "Error";
                oData.queryTypeStateText = "Please select a query type";
            }
            
            oCreateModel.setData(oData);
        },

        onDatabaseTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.databaseTypeState = "Success";
                oData.databaseTypeStateText = "Database type selected";
                
                // Set dialect version based on database type
                switch (sValue) {
                    case "mysql":
                        oData.dialectVersion = "MySQL 8.0";
                        oData.queryLanguage = "mysql";
                        break;
                    case "postgresql":
                        oData.dialectVersion = "PostgreSQL 14";
                        oData.queryLanguage = "postgresql";
                        break;
                    case "oracle":
                        oData.dialectVersion = "Oracle 19c";
                        oData.queryLanguage = "plsql";
                        break;
                    case "sqlserver":
                        oData.dialectVersion = "SQL Server 2019";
                        oData.queryLanguage = "tsql";
                        break;
                    case "hana":
                        oData.dialectVersion = "HANA 2.0";
                        oData.queryLanguage = "sql";
                        oData.connectionPool = 20;
                        break;
                    default:
                        oData.queryLanguage = "sql";
                }
            } else {
                oData.databaseTypeState = "Error";
                oData.databaseTypeStateText = "Please select a database type";
            }
            
            oCreateModel.setData(oData);
        },

        onSQLStatementChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Basic SQL validation
            if (sValue && sValue.trim().length > 0) {
                // Validate SQL with SecurityUtils
                var validation = SecurityUtils.validateSQL(sValue);
                if (!validation.isValid) {
                    MessageToast.show("SQL validation warning: " + validation.errors.join(", "));
                }
                
                // Estimate complexity based on SQL content
                var upperSQL = sValue.toUpperCase();
                if (upperSQL.includes("JOIN") && upperSQL.includes("SUBQUERY")) {
                    oData.complexity = "very_complex";
                } else if (upperSQL.includes("JOIN") || upperSQL.includes("UNION")) {
                    oData.complexity = "complex";
                } else if (upperSQL.includes("WHERE") || upperSQL.includes("GROUP BY")) {
                    oData.complexity = "moderate";
                } else {
                    oData.complexity = "simple";
                }
                
                // Estimate cost (simplified)
                oData.estimatedCost = this._estimateQueryCost(sValue);
            }
            
            oCreateModel.setData(oData);
        },

        onQueryLanguageChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Update code editor type based on language
            var oCodeEditor = this.getView().byId("sqlEditor");
            if (oCodeEditor) {
                oCodeEditor.setType(sValue === "nosql" ? "javascript" : "sql");
            }
            
            oCreateModel.setData(oData);
        },

        _estimateQueryCost: function(sql) {
            // Simplified cost estimation
            var cost = 10;
            var upperSQL = sql.toUpperCase();
            
            if (upperSQL.includes("JOIN")) cost += 20;
            if (upperSQL.includes("SUBQUERY") || upperSQL.includes("IN (")) cost += 30;
            if (upperSQL.includes("ORDER BY")) cost += 10;
            if (upperSQL.includes("GROUP BY")) cost += 15;
            if (upperSQL.includes("DISTINCT")) cost += 10;
            if (upperSQL.includes("UNION")) cost += 25;
            
            return cost.toString();
        },

        onCancelCreateSQLQuery: function() {
            this._oCreateDialog.close();
            this._resetCreateModel();
        },

        onConfirmCreateSQLQuery: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Final validation
            if (!this._validateCreateData(oData)) {
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Sanitize data for security
            var oSanitizedData = {
                queryName: SecurityUtils.sanitizeSQLParameter(oData.queryName),
                description: SecurityUtils.sanitizeSQLParameter(oData.description),
                queryType: oData.queryType,
                databaseType: oData.databaseType,
                priority: oData.priority,
                complexity: oData.complexity,
                sqlStatement: oData.sqlStatement, // Already validated by SecurityUtils
                queryLanguage: oData.queryLanguage,
                dialectVersion: oData.dialectVersion,
                estimatedCost: oData.estimatedCost,
                indexUsage: oData.indexUsage,
                connectionString: oData.connectionString,
                schemaName: oData.schemaName,
                databaseVersion: oData.databaseVersion,
                connectionPool: parseInt(oData.connectionPool) || 10,
                timeoutSettings: parseInt(oData.timeoutSettings) || 30,
                transactionMode: oData.transactionMode,
                isolationLevel: oData.isolationLevel,
                autoCommit: !!oData.autoCommit,
                dataClassification: oData.dataClassification
            };
            
            SecurityUtils.secureCallFunction(this.getView().getModel(), "/CreateSQLQuery", {
                urlParameters: oSanitizedData,
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show(this.getResourceBundle().getText("msg.queryCreated"));
                    this._refreshQueryData();
                    this._resetCreateModel();
                }.bind(this),
                error: function(error) {
                    this._oCreateDialog.setBusy(false);
                    var errorMsg = SecurityUtils.escapeHTML(error.message || "Unknown error");
                    MessageBox.error(this.getResourceBundle().getText("error.createQueryFailed") + ": " + errorMsg);
                }.bind(this)
            });
        },

        _validateCreateData: function(oData) {
            if (!oData.queryName || oData.queryName.length < 3) {
                MessageBox.error(this.getResourceBundle().getText("validation.queryNameRequired"));
                return false;
            }
            
            if (!oData.queryType) {
                MessageBox.error(this.getResourceBundle().getText("validation.queryTypeRequired"));
                return false;
            }
            
            if (!oData.databaseType) {
                MessageBox.error(this.getResourceBundle().getText("validation.databaseTypeRequired"));
                return false;
            }
            
            if (!oData.sqlStatement || oData.sqlStatement.trim() === "") {
                MessageBox.error(this.getResourceBundle().getText("validation.sqlStatementRequired"));
                return false;
            }
            
            // Validate SQL with SecurityUtils
            var validation = SecurityUtils.validateSQL(oData.sqlStatement);
            if (!validation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.sqlValidationFailed", [validation.errors.join(", ")]));
                return false;
            }
            
            return true;
        },

        _resetCreateModel: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.queryName = "";
            oData.description = "";
            oData.queryType = "";
            oData.databaseType = "";
            oData.priority = "medium";
            oData.complexity = "moderate";
            oData.queryNameState = "";
            oData.queryNameStateText = "";
            oData.queryTypeState = "";
            oData.queryTypeStateText = "";
            oData.databaseTypeState = "";
            oData.databaseTypeStateText = "";
            oData.sqlStatement = "";
            
            oCreateModel.setData(oData);
        },
        

        // Enhanced Execute Query Action with comprehensive security
        onExecuteQuery: function() {
            const userId = SecurityUtils._getCurrentUser();
            
            // Check rate limits first
            SecurityUtils.checkRateLimit(userId, 'execute').then(function(allowed) {
                if (!allowed) {
                    MessageToast.show(this.getResourceBundle().getText("error.rateLimitExceeded"));
                    return;
                }
                
                SecurityUtils.checkSQLAuth('execute', 'query').then(function(authorized) {
                    if (!authorized) {
                        MessageToast.show(this.getResourceBundle().getText("error.notAuthorized"));
                        return;
                    }
                    
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
                    
                    // Enhanced SQL validation with comprehensive security checks
                    const validation = SecurityUtils.validateSQL(oData.sqlStatement, {}, {
                        minSecurityScore: 70,
                        allowedOperations: this._getAllowedOperationsForUser(userId),
                        complexityLimits: {
                            maxJoins: 8,
                            maxSubqueries: 4,
                            maxComplexity: 40
                        }
                    });
                    
                    if (!validation.isValid) {
                        MessageBox.error(
                            this.getResourceBundle().getText("error.sqlValidationFailed", [validation.errors.join(', ')]) +
                            (validation.detectedPatterns.length > 0 ? 
                                "\\nSecurity issues: " + validation.detectedPatterns.join(', ') : "")
                        );
                        
                        // Audit log security validation failure
                        SecurityUtils.auditLog('SQL_VALIDATION_FAILED', {
                            queryId: oData.queryId,
                            queryHash: validation.queryHash,
                            securityScore: validation.securityScore,
                            riskLevel: validation.riskLevel,
                            errors: validation.errors,
                            detectedPatterns: validation.detectedPatterns
                        }, 'SECURITY_VIOLATION');
                        
                        return;
                    }
                    
                    // Check query complexity with enhanced limits
                    const complexity = SecurityUtils.validateQueryComplexity(oData.sqlStatement, {
                        maxJoins: 10,
                        maxSubqueries: 5,
                        maxComplexity: 50,
                        maxQueryLength: 5000
                    });
                    
                    if (!complexity.isValid) {
                        MessageBox.error(
                            this.getResourceBundle().getText("error.queryTooComplex", [complexity.issues.join(', ')]) +
                            "\\nComplexity Score: " + complexity.complexity + 
                            "\\nEstimated Execution Time: " + complexity.estimatedExecutionTime + "ms"
                        );
                        
                        // Audit log complexity validation failure
                        SecurityUtils.auditLog('COMPLEXITY_VALIDATION_FAILED', {
                            queryId: oData.queryId,
                            complexityScore: complexity.complexity,
                            riskLevel: complexity.riskLevel,
                            issues: complexity.issues
                        }, 'COMPLEXITY_EXCEEDED');
                        
                        return;
                    }

                    // Show enhanced confirmation dialog with security information
                    const confirmMessage = this.getResourceBundle().getText("msg.executeQueryConfirm") +
                        "\\n\\nSecurity Score: " + validation.securityScore + "/100" +
                        "\\nRisk Level: " + validation.riskLevel +
                        "\\nComplexity: " + complexity.riskLevel +
                        "\\nEstimated Time: " + complexity.estimatedExecutionTime + "ms";

                    MessageBox.confirm(confirmMessage, {
                        title: this.getResourceBundle().getText("dialog.executeQueryConfirm"),
                        onClose: function(oAction) {
                            if (oAction === MessageBox.Action.OK) {
                                this._executeQuerySecure(oContext, validation, complexity);
                            }
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this));
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
                const sanitizedQueryId = SecurityUtils.sanitizeSQLParameter(queryId);
                this._eventSource = SecurityUtils.createSecureEventSource(`http://localhost:8011/sql/${sanitizedQueryId}/stream`);
                
                this._eventSource.addEventListener('query-progress', (event) => {
                    this._updateQueryProgress(event.data);
                });

                this._eventSource.addEventListener('query-completed', (event) => {
                    this._handleQueryCompleted(event.data);
                });

                this._eventSource.addEventListener('query-error', (event) => {
                    this._handleQueryError(event.data);
                });

                this._eventSource.addEventListener('optimization-update', (event) => {
                    this._handleOptimizationUpdate(event.data);
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
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oContext.getObject().queryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.queryExecutionStarted"));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteQuery", {
                urlParameters: {
                    queryId: sQueryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.queryExecuted"));
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.queryExecutionFailed", [SecurityUtils.escapeHTML(error.message)]));
                }.bind(this)
            });
        },

        _validateQuery: function(oContext) {
            const oModel = this.getView().getModel();
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oContext.getObject().queryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.queryValidationStarted"));
            
            SecurityUtils.secureCallFunction(oModel, "/ValidateQuery", {
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
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oContext.getObject().queryId);
            
            SecurityUtils.secureCallFunction(oModel, "/FormatQuery", {
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
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oContext.getObject().queryId);
            
            SecurityUtils.secureCallFunction(oModel, "/GetOptimizationSuggestions", {
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
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oContext.getObject().queryId);
            
            SecurityUtils.secureCallFunction(oModel, "/GenerateExplainPlan", {
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
            MessageBox.error(this.getResourceBundle().getText("error.queryExecutionFailed", [SecurityUtils.escapeHTML(data.error)]));
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
        },

        /**
         * Get allowed SQL operations for user based on role/permissions
         * @private
         */
        _getAllowedOperationsForUser: function(userId) {
            // In a real implementation, this would check user roles/permissions
            // For now, return a default safe set
            return ['SELECT', 'COUNT', 'SHOW', 'DESCRIBE', 'EXPLAIN'];
        },

        /**
         * Enhanced secure query execution with comprehensive monitoring
         * @private
         */
        _executeQuerySecure: function(oContext, validation, complexity) {
            const oModel = this.getView().getModel();
            const oData = oContext.getObject();
            const sQueryId = SecurityUtils.sanitizeSQLParameter(oData.queryId);
            const startTime = Date.now();
            
            // Pre-execution audit log
            SecurityUtils.auditLog('QUERY_EXECUTION_STARTED', {
                queryId: sQueryId,
                queryHash: validation.queryHash,
                queryType: this._detectQueryType(oData.sqlStatement),
                securityScore: validation.securityScore,
                riskLevel: validation.riskLevel,
                complexityScore: complexity.complexity,
                estimatedExecutionTime: complexity.estimatedExecutionTime,
                tables: this._extractTableNames(oData.sqlStatement)
            }, 'INITIATED');
            
            MessageToast.show(this.getResourceBundle().getText("msg.queryExecutionStarted"));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteQuery", {
                urlParameters: {
                    queryId: sQueryId,
                    securityScore: validation.securityScore,
                    complexityScore: complexity.complexity,
                    validationResult: JSON.stringify({
                        isValid: validation.isValid,
                        riskLevel: validation.riskLevel,
                        queryHash: validation.queryHash
                    })
                },
                success: function(data) {
                    const executionTime = Date.now() - startTime;
                    MessageToast.show(this.getResourceBundle().getText("msg.queryExecuted"));
                    
                    // Success audit log with execution metrics
                    SecurityUtils.auditLog('QUERY_EXECUTION_COMPLETED', {
                        queryId: sQueryId,
                        executionTime: executionTime,
                        rowsReturned: data.rowsReturned || 0,
                        executionResult: 'SUCCESS'
                    }, 'SUCCESS');
                    
                    this._refreshQueryData();
                }.bind(this),
                error: function(error) {
                    const executionTime = Date.now() - startTime;
                    const sanitizedError = SecurityUtils.escapeHTML(error.message || 'Unknown error');
                    
                    // Error audit log
                    SecurityUtils.auditLog('QUERY_EXECUTION_FAILED', {
                        queryId: sQueryId,
                        executionTime: executionTime,
                        errorMessage: sanitizedError,
                        errorType: this._classifyError(error)
                    }, 'FAILURE');
                    
                    MessageToast.show(this.getResourceBundle().getText("error.queryExecutionFailed", [sanitizedError]));
                }.bind(this)
            });
        },

        /**
         * Detect query type from SQL statement
         * @private
         */
        _detectQueryType: function(sql) {
            const lowerSQL = sql.toLowerCase().trim();
            if (lowerSQL.startsWith('select')) return 'SELECT';
            if (lowerSQL.startsWith('insert')) return 'INSERT';
            if (lowerSQL.startsWith('update')) return 'UPDATE';
            if (lowerSQL.startsWith('delete')) return 'DELETE';
            if (lowerSQL.startsWith('create')) return 'CREATE';
            if (lowerSQL.startsWith('alter')) return 'ALTER';
            if (lowerSQL.startsWith('drop')) return 'DROP';
            if (lowerSQL.startsWith('show')) return 'SHOW';
            if (lowerSQL.startsWith('describe')) return 'DESCRIBE';
            return 'UNKNOWN';
        },

        /**
         * Extract table names from SQL statement
         * @private
         */
        _extractTableNames: function(sql) {
            const tables = [];
            const patterns = [
                /from\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi,
                /join\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi,
                /into\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi,
                /update\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi
            ];
            
            patterns.forEach(pattern => {
                let match;
                while ((match = pattern.exec(sql)) !== null) {
                    const tableName = match[1];
                    if (!tables.includes(tableName)) {
                        tables.push(tableName);
                    }
                }
            });
            
            return tables;
        },

        /**
         * Classify error type for better audit logging
         * @private
         */
        _classifyError: function(error) {
            if (!error || !error.message) return 'UNKNOWN_ERROR';
            
            const errorMsg = error.message.toLowerCase();
            
            if (errorMsg.includes('timeout')) return 'TIMEOUT_ERROR';
            if (errorMsg.includes('permission') || errorMsg.includes('denied')) return 'PERMISSION_ERROR';
            if (errorMsg.includes('syntax')) return 'SYNTAX_ERROR';
            if (errorMsg.includes('table') && errorMsg.includes('not found')) return 'TABLE_NOT_FOUND';
            if (errorMsg.includes('column') && errorMsg.includes('not found')) return 'COLUMN_NOT_FOUND';
            if (errorMsg.includes('connection')) return 'CONNECTION_ERROR';
            if (errorMsg.includes('resource')) return 'RESOURCE_ERROR';
            
            return 'EXECUTION_ERROR';
        }
    });
});