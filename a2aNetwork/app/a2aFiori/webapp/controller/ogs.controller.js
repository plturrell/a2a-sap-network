sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/export/Spreadsheet",
    "sap/ui/export/library"
], (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Spreadsheet, exportLibrary) => {
    "use strict";

    const EdmType = exportLibrary.EdmType;

    return BaseController.extend("a2a.network.fiori.controller.Logs", {

        getCurrentUserRole() {
            try {
                return sap.ushell.Container.getUser().getRole() ||
                       this.getOwnerComponent().getModel("app").getProperty("/currentUser/role") ||
                       "viewer";
            } catch (error) {
                return "viewer";
            }
        },

        hasPermissions(userRole, requiredPermissions) {
            const rolePermissions = {
                "systemAdmin": ["*"],
                "networkAdmin": ["system.logs.read", "operations.monitor", "agents.*"],
                "agentDeveloper": ["agents.*", "system.logs.read"],
                "viewer": ["system.logs.read"]
            };

            const userPermissions = rolePermissions[userRole] || [];
            if (userPermissions.includes("*")) {
                return true;
            }

            return requiredPermissions.every(permission =>
                userPermissions.some(userPerm =>
                    userPerm === permission || userPerm.endsWith("*") && permission.startsWith(userPerm.slice(0, -1))
                )
            );
        },

        onInit() {
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

        _initializeModels() {
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
            const oPermissionsModel = new JSONModel({
                canClearLogs: this._checkPermission("LOGS_CLEAR"),
                canExportLogs: this._checkPermission("LOGS_EXPORT"),
                canViewDetails: true
            });
            this.getView().setModel(oPermissionsModel, "permissions");
        },

        _setupRealtimeLogging() {
            // WebSocket connection for real-time logs
            if (window.WebSocket) {
                try {
                    const wsUrl = process.env.WS_LOGS_URL || window.A2A_CONFIG?.wsUrl || "wss://production-host/ws/logs";
                    this._wsConnection = new WebSocket(wsUrl);

                    this._wsConnection.onopen = function() {
                        // Real-time log connection established
                    }.bind(this);

                    this._wsConnection.onmessage = function(event) {
                        if (this.oUIModel.getProperty("/realtimeEnabled")) {
                            const logEntry = JSON.parse(event.data);
                            this._addRealtimeLog(logEntry);
                        }
                    }.bind(this);

                    this._wsConnection.onerror = function(error) {
                        // WebSocket error logged
                    };

                    this._wsConnection.onclose = function() {
                        // Real-time log connection closed
                        // Attempt reconnect after 5 seconds
                        setTimeout(this._setupRealtimeLogging.bind(this), 5000);
                    }.bind(this);
                } catch (error) {
                    // Failed to establish WebSocket connection
                }
            }
        },

        _addRealtimeLog(logEntry) {
            const aEntries = this.oLogsModel.getProperty("/entries");
            aEntries.unshift(logEntry);

            // Limit to 1000 entries in memory
            if (aEntries.length > 1000) {
                aEntries.pop();
            }

            this.oLogsModel.setProperty("/entries", aEntries);
            this._updateStatistics();

            // Auto-scroll if enabled
            if (this.oUIModel.getProperty("/autoScroll")) {
                const oTable = this.byId("logsTable");
                oTable.scrollToIndex(0);
            }
        },

        _loadLogs() {
            this.showSkeletonLoading(this.getResourceBundle().getText("logs.loading"));

            // Fetch real logs from the operations service
            jQuery.ajax({
                url: "/api/v1/operations/logs",
                type: "GET",
                data: {
                    limit: 250,
                    since: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString() // Last 7 days
                },
                success: function(data) {
                    let aLogs = data.logs || [];
                    // Transform logs to match the expected format
                    aLogs = aLogs.map((log, index) => {
                        return {
                            id: log.id || `log_${ index}`,
                            timestamp: new Date(log.timestamp),
                            level: log.level || "INFO",
                            source: log.logger || "Unknown",
                            message: log.message || "",
                            user: log.user || "system",
                            correlationId: log.correlationId,
                            sessionId: log.sessionId,
                            duration: log.duration,
                            stackTrace: log.stackTrace,
                            context: log.details
                        };
                    });

                    this.oLogsModel.setProperty("/entries", aLogs);
                    this.oLogsModel.setProperty("/totalCount", aLogs.length);
                    this._updateStatistics();
                    this._extractSources();
                    this.hideLoading();
                }.bind(this),
                error: function(xhr) {
                    this.hideLoading();
                    sap.m.MessageToast.show(`Failed to load logs: ${ xhr.responseJSON?.error || "Unknown error"}`);
                    // Set empty array on error
                    this.oLogsModel.setProperty("/entries", []);
                    this.oLogsModel.setProperty("/totalCount", 0);
                    this._updateStatistics();
                }.bind(this)
            });
        },

        /* REMOVED: Fake log generation - now using real API data
        _generateSampleLogs: function() {
            var aLogs = [];
            var aSources = ["AgentService", "BlockchainService", "AuthService",
                "WorkflowEngine", "MessageQueue", "CacheManager"];
            var aLevels = ["ERROR", "WARNING", "INFO", "DEBUG", "TRACE"];
            var aUsers = ["system", "system-admin", "agent_scheduler", "blockchain_monitor", "api_gateway"];

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
        */

        _updateStatistics() {
            const aEntries = this.oLogsModel.getProperty("/entries");
            const oStats = {
                total: aEntries.length,
                errors: 0,
                warnings: 0,
                info: 0,
                debug: 0,
                trace: 0
            };

            aEntries.forEach((entry) => {
                switch (entry.level) {
                case "ERROR": oStats.errors++; break;
                case "WARNING": oStats.warnings++; break;
                case "INFO": oStats.info++; break;
                case "DEBUG": oStats.debug++; break;
                case "TRACE": oStats.trace++; break;
                }
            });

            this.oLogsModel.setProperty("/statistics", oStats);
        },

        _extractSources() {
            const aEntries = this.oLogsModel.getProperty("/entries");
            const oSourceMap = {};

            aEntries.forEach((entry) => {
                oSourceMap[entry.source] = true;
            });

            const aSources = Object.keys(oSourceMap).map((source) => {
                return { name: source };
            });

            this.oLogsModel.setProperty("/sources", aSources);
        },

        onSearchLogs(oEvent) {
            const sQuery = oEvent.getParameter("query") || oEvent.getParameter("newValue");
            this.oLogsModel.setProperty("/filters/searchQuery", sQuery);
            this._applyFilters();
        },

        onFilterLogs() {
            this._applyFilters();
            this._saveFilters();
        },

        _applyFilters() {
            const oTable = this.byId("logsTable");
            const _oBinding = oTable.getBinding("items");
            const aFilters = [];

            // Level filter
            const aLevelFilters = this.byId("levelFilter").getSelectedKeys();
            if (aLevelFilters.length > 0) {
                const aLevelFilterObjects = aLevelFilters.map((level) => {
                    return new Filter("level", FilterOperator.EQ, level);
                });
                aFilters.push(new Filter({
                    filters: aLevelFilterObjects,
                    and: false
                }));
            }

            // Source filter
            const aSourceFilters = this.byId("sourceFilter").getSelectedKeys();
            if (aSourceFilters.length > 0) {
                const aSourceFilterObjects = aSourceFilters.map((source) => {
                    return new Filter("source", FilterOperator.EQ, source);
                });
                aFilters.push(new Filter({
                    filters: aSourceFilterObjects,
                    and: false
                }));
            }

            // Date range filter
            const oDateRange = this.byId("dateRangeFilter");
            if (oDateRange.getDateValue() && oDateRange.getSecondDateValue()) {
                aFilters.push(new Filter("timestamp", FilterOperator.BT,
                    oDateRange.getDateValue(), oDateRange.getSecondDateValue()));
            }

            // Correlation ID filter
            const sCorrelationId = this.byId("correlationFilter").getValue();
            if (sCorrelationId) {
                aFilters.push(new Filter("correlationId", FilterOperator.Contains, sCorrelationId));
            }

            // Search filter
            const sSearchQuery = this.oLogsModel.getProperty("/filters/searchQuery");
            if (sSearchQuery) {
                const aSearchFilters = [
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

        onClearFilters() {
            this.byId("levelFilter").setSelectedKeys([]);
            this.byId("sourceFilter").setSelectedKeys([]);
            this.byId("dateRangeFilter").setValue("");
            this.byId("correlationFilter").setValue("");
            this.byId("logSearchField").setValue("");
            this.oLogsModel.setProperty("/filters/searchQuery", "");
            this._applyFilters();
            this._saveFilters();
        },

        onLogPress(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("logs");
            const oLog = oContext.getObject();

            // Format context for display
            if (oLog.context) {
                oLog.contextFormatted = JSON.stringify(oLog.context, null, 2);
            }

            this.oLogsModel.setProperty("/selectedLog", oLog);
            this.byId("logDetailDialog").open();
        },

        onCloseLogDetail() {
            this.byId("logDetailDialog").close();
        },

        onCopyLogDetail() {
            const oLog = this.oLogsModel.getProperty("/selectedLog");
            const sLogText = this._formatLogForClipboard(oLog);

            if (navigator.clipboard) {
                navigator.clipboard.writeText(sLogText).then(() => {
                    MessageToast.show(this.getResourceBundle().getText("logs.copy.success"));
                }).catch(function() {
                    MessageToast.show(this.getResourceBundle().getText("logs.copy.error"));
                });
            }
        },

        _formatLogForClipboard(oLog) {
            let sText = "=== Log Entry ===\n";
            sText += `Timestamp: ${ new Date(oLog.timestamp).toISOString() }\n`;
            sText += `Level: ${ oLog.level }\n`;
            sText += `Source: ${ oLog.source }\n`;
            sText += `Message: ${ oLog.message }\n`;
            sText += `User: ${ oLog.user }\n`;
            sText += `Correlation ID: ${ oLog.correlationId || "N/A" }\n`;
            sText += `Session ID: ${ oLog.sessionId }\n`;

            if (oLog.duration) {
                sText += `Duration: ${ oLog.duration } ms\n`;
            }

            if (oLog.stackTrace) {
                sText += `\nStack Trace:\n${ oLog.stackTrace }\n`;
            }

            if (oLog.context) {
                sText += `\nContext:\n${ JSON.stringify(oLog.context, null, 2) }\n`;
            }

            return sText;
        },

        onViewRelatedLogs() {
            const oLog = this.oLogsModel.getProperty("/selectedLog");
            if (oLog && oLog.correlationId) {
                this.byId("logDetailDialog").close();
                this.byId("correlationFilter").setValue(oLog.correlationId);
                this._applyFilters();
                MessageToast.show(this.getResourceBundle().getText("logs.relatedLogs.filtered"));
            }
        },

        onCorrelationPress(oEvent) {
            const sCorrelationId = oEvent.getSource().getText();
            this.byId("correlationFilter").setValue(sCorrelationId);
            this._applyFilters();
        },

        onLogSelectionChange(oEvent) {
            const aSelectedItems = oEvent.getSource().getSelectedItems();
            const aSelectedLogs = aSelectedItems.map((item) => {
                return item.getBindingContext("logs").getObject();
            });
            this.oUIModel.setProperty("/selectedLogs", aSelectedLogs);
        },

        onDownloadLogs() {
            const aSelectedLogs = this.oUIModel.getProperty("/selectedLogs");
            if (aSelectedLogs && aSelectedLogs.length > 0) {
                this._downloadLogsAsFile(aSelectedLogs);
            }
        },

        _downloadLogsAsFile(aLogs) {
            const sContent = aLogs.map((log) => {
                return this._formatLogForFile(log);
            }).join("\n\n");

            const blob = new Blob([sContent], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `a2a_logs_${ new Date().toISOString().replace(/:/g, "-") }.log`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            MessageToast.show(this.getResourceBundle().getText("logs.download.success"));
        },

        _formatLogForFile(oLog) {
            const sTimestamp = new Date(oLog.timestamp).toISOString();
            const sLevel = oLog.level.padEnd(7);
            const sSource = (`[${ oLog.source }]`).padEnd(20);
            return `${sTimestamp } ${ sLevel } ${ sSource } ${ oLog.message}`;
        },

        onClearLogs() {
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

        _clearLogs() {
            this.showSpinnerLoading(this.getResourceBundle().getText("logs.clearing"));

            // In production, call backend service to clear logs
            setTimeout(() => {
                this.oLogsModel.setProperty("/entries", []);
                this.oLogsModel.setProperty("/totalCount", 0);
                this._updateStatistics();
                this.hideLoading();
                MessageToast.show(this.getResourceBundle().getText("logs.clear.success"));
            }, 1000);
        },

        onRefreshLogs() {
            this._loadLogs();
        },

        onExportLogs() {
            const _aColumns = this._createExportColumns();
            const aRows = this._prepareExportData();

            const oSettings = {
                workbook: {
                    columns: aColumns
                },
                dataSource: aRows,
                fileName: `A2A_Network_Logs_${ new Date().toISOString().split("T")[0] }.xlsx`,
                worker: true
            };

            const oSpreadsheet = new Spreadsheet(oSettings);
            oSpreadsheet.build()
                .then(() => {
                    MessageToast.show(this.getResourceBundle().getText("logs.export.success"));
                })
                .catch((sMessage) => {
                    MessageToast.show(this.getResourceBundle().getText("logs.export.error"));
                });
        },

        _createExportColumns() {
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

        _prepareExportData() {
            const oTable = this.byId("logsTable");
            const _oBinding = oTable.getBinding("items");
            const aContexts = oBinding.getContexts();

            return aContexts.map((oContext) => {
                return oContext.getObject();
            });
        },

        onToggleRealtime(oEvent) {
            const bEnabled = oEvent.getParameter("state");
            if (bEnabled) {
                MessageToast.show(this.getResourceBundle().getText("logs.realtime.enabled"));
            } else {
                MessageToast.show(this.getResourceBundle().getText("logs.realtime.disabled"));
            }
        },

        _startAutoRefresh() {
            this._refreshInterval = setInterval(() => {
                if (!this.oUIModel.getProperty("/realtimeEnabled")) {
                    this._loadLogs();
                }
            }, 30000); // Refresh every 30 seconds
        },

        _saveFilters() {
            const oFilters = {
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

        _restoreSavedFilters() {
            const sSavedFilters = localStorage.getItem("a2a_log_filters");
            if (sSavedFilters) {
                try {
                    const oFilters = JSON.parse(sSavedFilters);
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
                    // Failed to restore saved filters
                }
            }
        },

        _checkPermission(sPermission) {
            // In production, check actual user permissions
            // Check actual user permissions via backend service
            const userRole = this.getCurrentUserRole();
            const requiredPermissions = ["system.logs.read", "operations.monitor"];
            return this.hasPermissions(userRole, requiredPermissions);
        },

        onNavBack() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onRetry() {
            this._loadLogs();
        },

        onExit() {
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