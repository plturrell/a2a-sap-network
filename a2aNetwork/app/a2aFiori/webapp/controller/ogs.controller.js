sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/export/Spreadsheet",
    "sap/ui/export/library"
], function (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Spreadsheet, exportLibrary) {
    "use strict";

    var EdmType = exportLibrary.EdmType;

    return BaseController.extend("a2a.network.fiori.controller.Logs", {

        onInit: function () {
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Initialize models
            this._initializeModels();
            
            // Set up real-time log streaming
            this._setupRealtimeLogging();
            
            // Load initial data
            this._loadLogs();
            
            // Set up auto-refresh
            this._startAutoRefresh();
            
            // Initialize filter persistence
            this._restoreSavedFilters();
        },

        _initializeModels: function() {
            // Logs model
            this.oLogsModel = new JSONModel({
                entries: [],
                sources: [],
                statistics: {
                    total: 0,
                    errors: 0,
                    warnings: 0,
                    info: 0,
                    debug: 0,
                    trace: 0
                },
                totalCount: 0,
                selectedLog: null,
                filters: {
                    levels: [],
                    sources: [],
                    dateFrom: null,
                    dateTo: null,
                    correlationId: "",
                    searchQuery: ""
                }
            });
            this.getView().setModel(this.oLogsModel, "logs");
            
            // Permissions model
            var oPermissionsModel = new JSONModel({
                canClearLogs: this._checkPermission("LOGS_CLEAR"),
                canExportLogs: this._checkPermission("LOGS_EXPORT"),
                canViewDetails: true
            });
            this.getView().setModel(oPermissionsModel, "permissions");
        },

        _setupRealtimeLogging: function() {
            // WebSocket connection for real-time logs
            if (window.WebSocket) {
                try {
                    this._wsConnection = new WebSocket("ws://localhost:4004/ws/logs");
                    
                    this._wsConnection.onopen = function() {
                        console.log("Real-time log connection established");
                    }.bind(this);
                    
                    this._wsConnection.onmessage = function(event) {
                        if (this.oUIModel.getProperty("/realtimeEnabled")) {
                            var logEntry = JSON.parse(event.data);
                            this._addRealtimeLog(logEntry);
                        }
                    }.bind(this);
                    
                    this._wsConnection.onerror = function(error) {
                        console.error("WebSocket error:", error);
                    };
                    
                    this._wsConnection.onclose = function() {
                        console.log("Real-time log connection closed");
                        // Attempt reconnect after 5 seconds
                        setTimeout(this._setupRealtimeLogging.bind(this), 5000);
                    }.bind(this);
                } catch (error) {
                    console.error("Failed to establish WebSocket connection:", error);
                }
            }
        },

        _addRealtimeLog: function(logEntry) {
            var aEntries = this.oLogsModel.getProperty("/entries");
            aEntries.unshift(logEntry);
            
            // Limit to 1000 entries in memory
            if (aEntries.length > 1000) {
                aEntries.pop();
            }
            
            this.oLogsModel.setProperty("/entries", aEntries);
            this._updateStatistics();
            
            // Auto-scroll if enabled
            if (this.oUIModel.getProperty("/autoScroll")) {
                var oTable = this.byId("logsTable");
                oTable.scrollToIndex(0);
            }
        },

        _loadLogs: function() {
            this.showSkeletonLoading(this.getResourceBundle().getText("logs.loading"));
            
            // Simulate loading logs - in production, call backend service
            setTimeout(function() {
                var aLogs = this._generateSampleLogs();
                this.oLogsModel.setProperty("/entries", aLogs);
                this.oLogsModel.setProperty("/totalCount", aLogs.length);
                this._updateStatistics();
                this._extractSources();
                this.hideLoading();
            }.bind(this), 1000);
        },

        _generateSampleLogs: function() {
            var aLogs = [];
            var aSources = ["AgentService", "BlockchainService", "AuthService", "WorkflowEngine", "MessageQueue", "CacheManager"];
            var aLevels = ["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"];
            var aUsers = ["system", "admin@sap.com", "agent_scheduler", "blockchain_monitor", "api_gateway"];
            
            for (var i = 0; i < 250; i++) {
                var level = aLevels[Math.floor(Math.random() * aLevels.length)];
                var timestamp = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
                
                aLogs.push({
                    id: "log_" + i,
                    timestamp: timestamp,
                    level: level,
                    source: aSources[Math.floor(Math.random() * aSources.length)],
                    message: this._generateLogMessage(level),
                    user: aUsers[Math.floor(Math.random() * aUsers.length)],
                    correlationId: Math.random() > 0.3 ? "corr_" + Math.floor(Math.random() * 100) : null,
                    sessionId: "session_" + Math.floor(Math.random() * 20),
                    duration: level === "INFO" && Math.random() > 0.5 ? Math.floor(Math.random() * 5000) : null,
                    stackTrace: level === "ERROR" && Math.random() > 0.5 ? this._generateStackTrace() : null,
                    context: Math.random() > 0.7 ? {
                        agentId: "agent_" + Math.floor(Math.random() * 50),
                        transactionHash: "0x" + Math.random().toString(16).substr(2, 64),
                        blockNumber: Math.floor(Math.random() * 100000)
                    } : null
                });
            }
            
            return aLogs.sort((a, b) => b.timestamp - a.timestamp);
        },

        _generateLogMessage: function(level) {
            var messages = {
                ERROR: [
                    "Failed to connect to blockchain node: Connection timeout",
                    "Agent execution failed: Insufficient gas",
                    "Database connection error: Too many connections",
                    "Authentication failed for user: Invalid credentials",
                    "Smart contract deployment failed: Contract already exists"
                ],
                WARNING: [
                    "High memory usage detected: 85% threshold exceeded",
                    "Slow query detected: Execution time > 2000ms",
                    "Rate limit approaching for API endpoint",
                    "Certificate expires in 30 days",
                    "Deprecated API version used"
                ],
                INFO: [
                    "Agent registered successfully",
                    "Transaction confirmed on blockchain",
                    "User login successful",
                    "Workflow execution completed",
                    "Cache cleared successfully"
                ],
                DEBUG: [
                    "Entering method: executeAgent()",
                    "Query parameters: {limit: 100, offset: 0}",
                    "Cache hit for key: agent_list",
                    "WebSocket connection established",
                    "Request headers validated"
                ],
                TRACE: [
                    "Method call stack: [init, validate, execute]",
                    "Variable state: {status: 'pending', retries: 0}",
                    "Event emitted: agent.status.changed",
                    "Lock acquired for resource: workflow_123",
                    "Memory allocation: 256MB"
                ]
            };
            
            var levelMessages = messages[level] || messages.INFO;
            return levelMessages[Math.floor(Math.random() * levelMessages.length)];
        },

        _generateStackTrace: function() {
            return `Error: Connection timeout
    at BlockchainService.connect (/srv/blockchain-service.js:145:15)
    at async AgentExecutor.deployContract (/srv/agent-executor.js:89:9)
    at async WorkflowEngine.executeStep (/srv/workflow-engine.js:234:17)
    at async MessageQueue.processMessage (/srv/message-queue.js:78:13)
    at async Server.<anonymous> (/srv/server.js:156:5)`;
        },

        _updateStatistics: function() {
            var aEntries = this.oLogsModel.getProperty("/entries");
            var oStats = {
                total: aEntries.length,
                errors: 0,
                warnings: 0,
                info: 0,
                debug: 0,
                trace: 0
            };
            
            aEntries.forEach(function(entry) {
                switch(entry.level) {
                    case "ERROR": oStats.errors++; break;
                    case "WARNING": oStats.warnings++; break;
                    case "INFO": oStats.info++; break;
                    case "DEBUG": oStats.debug++; break;
                    case "TRACE": oStats.trace++; break;
                }
            });
            
            this.oLogsModel.setProperty("/statistics", oStats);
        },

        _extractSources: function() {
            var aEntries = this.oLogsModel.getProperty("/entries");
            var oSourceMap = {};
            
            aEntries.forEach(function(entry) {
                oSourceMap[entry.source] = true;
            });
            
            var aSources = Object.keys(oSourceMap).map(function(source) {
                return { name: source };
            });
            
            this.oLogsModel.setProperty("/sources", aSources);
        },

        onSearchLogs: function(oEvent) {
            var sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            this.oLogsModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onFilterLogs: function() {
            this._applyFilters();
            this._saveFilters();
        },

        _applyFilters: function() {
            var oTable = this.byId("logsTable");
            var oBinding = oTable.getBinding("items");
            var aFilters = [];
            
            // Level filter
            var aLevelFilters = this.byId("levelFilter").getSelectedKeys();
            if (aLevelFilters.length > 0) {
                var aLevelFilterObjects = aLevelFilters.map(function(level) {
                    return new Filter("level", FilterOperator.EQ, level);
                });
                aFilters.push(new Filter({
                    filters: aLevelFilterObjects,
                    and: false
                }));
            }
            
            // Source filter
            var aSourceFilters = this.byId("sourceFilter").getSelectedKeys();
            if (aSourceFilters.length > 0) {
                var aSourceFilterObjects = aSourceFilters.map(function(source) {
                    return new Filter("source", FilterOperator.EQ, source);
                });
                aFilters.push(new Filter({
                    filters: aSourceFilterObjects,
                    and: false
                }));
            }
            
            // Date range filter
            var oDateRange = this.byId("dateRangeFilter");
            if (oDateRange.getDateValue() && oDateRange.getSecondDateValue()) {
                aFilters.push(new Filter("timestamp", FilterOperator.BT, 
                    oDateRange.getDateValue(), oDateRange.getSecondDateValue()));
            }
            
            // Correlation ID filter
            var sCorrelationId = this.byId("correlationFilter").getValue();
            if (sCorrelationId) {
                aFilters.push(new Filter("correlationId", FilterOperator.Contains, sCorrelationId));
            }
            
            // Search filter
            var sSearchQuery = this.oLogsModel.getProperty("/filters/searchQuery");
            if (sSearchQuery) {
                var aSearchFilters = [
                    new Filter("message", FilterOperator.Contains, sSearchQuery),
                    new Filter("source", FilterOperator.Contains, sSearchQuery),
                    new Filter("user", FilterOperator.Contains, sSearchQuery),
                    new Filter("correlationId", FilterOperator.Contains, sSearchQuery)
                ];
                aFilters.push(new Filter({
                    filters: aSearchFilters,
                    and: false
                }));
            }
            
            oBinding.filter(aFilters);
        },

        onClearFilters: function() {
            this.byId("levelFilter").setSelectedKeys([]);
            this.byId("sourceFilter").setSelectedKeys([]);
            this.byId("dateRangeFilter").setValue("");
            this.byId("correlationFilter").setValue("");
            this.byId("logSearchField").setValue("");
            this.oLogsModel.setProperty("/filters/searchQuery", "");
            this._applyFilters();
            this._saveFilters();
        },

        onLogPress: function(oEvent) {
            var oContext = oEvent.getSource().getBindingContext("logs");
            var oLog = oContext.getObject();
            
            // Format context for display
            if (oLog.context) {
                oLog.contextFormatted = JSON.stringify(oLog.context, null, 2);
            }
            
            this.oLogsModel.setProperty("/selectedLog", oLog);
            this.byId("logDetailDialog").open();
        },

        onCloseLogDetail: function() {
            this.byId("logDetailDialog").close();
        },

        onCopyLogDetail: function() {
            var oLog = this.oLogsModel.getProperty("/selectedLog");
            var sLogText = this._formatLogForClipboard(oLog);
            
            if (navigator.clipboard) {
                navigator.clipboard.writeText(sLogText).then(function() {
                    MessageToast.show(this.getResourceBundle().getText("logs.copy.success"));
                }.bind(this)).catch(function() {
                    MessageToast.show(this.getResourceBundle().getText("logs.copy.error"));
                });
            }
        },

        _formatLogForClipboard: function(oLog) {
            var sText = "=== Log Entry ===\n";
            sText += "Timestamp: " + new Date(oLog.timestamp).toISOString() + "\n";
            sText += "Level: " + oLog.level + "\n";
            sText += "Source: " + oLog.source + "\n";
            sText += "Message: " + oLog.message + "\n";
            sText += "User: " + oLog.user + "\n";
            sText += "Correlation ID: " + (oLog.correlationId || "N/A") + "\n";
            sText += "Session ID: " + oLog.sessionId + "\n";
            
            if (oLog.duration) {
                sText += "Duration: " + oLog.duration + " ms\n";
            }
            
            if (oLog.stackTrace) {
                sText += "\nStack Trace:\n" + oLog.stackTrace + "\n";
            }
            
            if (oLog.context) {
                sText += "\nContext:\n" + JSON.stringify(oLog.context, null, 2) + "\n";
            }
            
            return sText;
        },

        onViewRelatedLogs: function() {
            var oLog = this.oLogsModel.getProperty("/selectedLog");
            if (oLog && oLog.correlationId) {
                this.byId("logDetailDialog").close();
                this.byId("correlationFilter").setValue(oLog.correlationId);
                this._applyFilters();
                MessageToast.show(this.getResourceBundle().getText("logs.relatedLogs.filtered"));
            }
        },

        onCorrelationPress: function(oEvent) {
            var sCorrelationId = oEvent.getSource().getText();
            this.byId("correlationFilter").setValue(sCorrelationId);
            this._applyFilters();
        },

        onLogSelectionChange: function(oEvent) {
            var aSelectedItems = oEvent.getSource().getSelectedItems();
            var aSelectedLogs = aSelectedItems.map(function(item) {
                return item.getBindingContext("logs").getObject();
            });
            this.oUIModel.setProperty("/selectedLogs", aSelectedLogs);
        },

        onDownloadLogs: function() {
            var aSelectedLogs = this.oUIModel.getProperty("/selectedLogs");
            if (aSelectedLogs && aSelectedLogs.length > 0) {
                this._downloadLogsAsFile(aSelectedLogs);
            }
        },

        _downloadLogsAsFile: function(aLogs) {
            var sContent = aLogs.map(function(log) {
                return this._formatLogForFile(log);
            }.bind(this)).join("\n\n");
            
            var blob = new Blob([sContent], { type: "text/plain;charset=utf-8" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = "a2a_logs_" + new Date().toISOString().replace(/:/g, "-") + ".log";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            MessageToast.show(this.getResourceBundle().getText("logs.download.success"));
        },

        _formatLogForFile: function(oLog) {
            var sTimestamp = new Date(oLog.timestamp).toISOString();
            var sLevel = oLog.level.padEnd(7);
            var sSource = ("[" + oLog.source + "]").padEnd(20);
            return sTimestamp + " " + sLevel + " " + sSource + " " + oLog.message;
        },

        onClearLogs: function() {
            MessageBox.confirm(
                this.getResourceBundle().getText("logs.clear.confirm"),
                {
                    title: this.getResourceBundle().getText("logs.clear.title"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._clearLogs();
                        }
                    }.bind(this)
                }
            );
        },

        _clearLogs: function() {
            this.showSpinnerLoading(this.getResourceBundle().getText("logs.clearing"));
            
            // In production, call backend service to clear logs
            setTimeout(function() {
                this.oLogsModel.setProperty("/entries", []);
                this.oLogsModel.setProperty("/totalCount", 0);
                this._updateStatistics();
                this.hideLoading();
                MessageToast.show(this.getResourceBundle().getText("logs.clear.success"));
            }.bind(this), 1000);
        },

        onRefreshLogs: function() {
            this._loadLogs();
        },

        onExportLogs: function() {
            var aColumns = this._createExportColumns();
            var aRows = this._prepareExportData();
            
            var oSettings = {
                workbook: {
                    columns: aColumns
                },
                dataSource: aRows,
                fileName: "A2A_Network_Logs_" + new Date().toISOString().split('T')[0] + ".xlsx",
                worker: true
            };
            
            var oSpreadsheet = new Spreadsheet(oSettings);
            oSpreadsheet.build()
                .then(function() {
                    MessageToast.show(this.getResourceBundle().getText("logs.export.success"));
                }.bind(this))
                .catch(function(sMessage) {
                    MessageToast.show(this.getResourceBundle().getText("logs.export.error"));
                }.bind(this));
        },

        _createExportColumns: function() {
            return [
                { label: "Timestamp", property: "timestamp", type: EdmType.DateTime },
                { label: "Level", property: "level", type: EdmType.String },
                { label: "Source", property: "source", type: EdmType.String },
                { label: "Message", property: "message", type: EdmType.String },
                { label: "User", property: "user", type: EdmType.String },
                { label: "Correlation ID", property: "correlationId", type: EdmType.String },
                { label: "Session ID", property: "sessionId", type: EdmType.String },
                { label: "Duration (ms)", property: "duration", type: EdmType.Number }
            ];
        },

        _prepareExportData: function() {
            var oTable = this.byId("logsTable");
            var oBinding = oTable.getBinding("items");
            var aContexts = oBinding.getContexts();
            
            return aContexts.map(function(oContext) {
                return oContext.getObject();
            });
        },

        onToggleRealtime: function(oEvent) {
            var bEnabled = oEvent.getParameter("state");
            if (bEnabled) {
                MessageToast.show(this.getResourceBundle().getText("logs.realtime.enabled"));
            } else {
                MessageToast.show(this.getResourceBundle().getText("logs.realtime.disabled"));
            }
        },

        _startAutoRefresh: function() {
            this._refreshInterval = setInterval(function() {
                if (!this.oUIModel.getProperty("/realtimeEnabled")) {
                    this._loadLogs();
                }
            }.bind(this), 30000); // Refresh every 30 seconds
        },

        _saveFilters: function() {
            var oFilters = {
                levels: this.byId("levelFilter").getSelectedKeys(),
                sources: this.byId("sourceFilter").getSelectedKeys(),
                dateRange: {
                    from: this.byId("dateRangeFilter").getDateValue(),
                    to: this.byId("dateRangeFilter").getSecondDateValue()
                },
                correlationId: this.byId("correlationFilter").getValue()
            };
            
            localStorage.setItem("a2a_log_filters", JSON.stringify(oFilters));
        },

        _restoreSavedFilters: function() {
            var sSavedFilters = localStorage.getItem("a2a_log_filters");
            if (sSavedFilters) {
                try {
                    var oFilters = JSON.parse(sSavedFilters);
                    if (oFilters.levels) {
                        this.byId("levelFilter").setSelectedKeys(oFilters.levels);
                    }
                    if (oFilters.sources) {
                        this.byId("sourceFilter").setSelectedKeys(oFilters.sources);
                    }
                    if (oFilters.correlationId) {
                        this.byId("correlationFilter").setValue(oFilters.correlationId);
                    }
                } catch (e) {
                    console.error("Failed to restore saved filters:", e);
                }
            }
        },

        _checkPermission: function(sPermission) {
            // In production, check actual user permissions
            return true; // For demo, all permissions granted
        },

        onNavBack: function() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onRetry: function() {
            this._loadLogs();
        },

        onExit: function() {
            // Clean up
            if (this._wsConnection) {
                this._wsConnection.close();
            }
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
        }
    });
});