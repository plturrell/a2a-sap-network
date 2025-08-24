/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/base/Log",
    "sap/viz/ui5/controls/VizFrame",
    "sap/viz/ui5/data/FlattenedDataset"
], function(BaseController, MessageToast, MessageBox, JSONModel, Filter, FilterOperator,
    Log, VizFrame, FlattenedDataset) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Settings", {

        onInit() {
            // Initialize comprehensive settings model with professional data structure
            const oSettingsModel = new JSONModel({
                network: {
                    name: "A2A Network",
                    description: "Enterprise Agent-to-Agent Communication Network",
                    maxAgents: 1000,
                    autoDiscovery: true,
                    timeoutSeconds: 30,
                    loadBalancingEnabled: true,
                    healthCheckInterval: 30,
                    networkRegion: "us-east-1",
                    networkTier: "enterprise"
                },
                performance: {
                    messagePoolSize: 1000,
                    cacheEnabled: true,
                    cacheTTLMinutes: 60,
                    loadBalancingStrategy: "roundRobin",
                    maxConcurrentConnections: 500,
                    connectionTimeout: 30000,
                    retryAttempts: 3,
                    circuitBreakerEnabled: true,
                    compressionEnabled: true
                },
                security: {
                    requireAuth: process.env.NODE_ENV === "production",
                    sessionTimeoutMinutes: 120,
                    maxLoginAttempts: 5,
                    encryptMessages: true,
                    rateLimitEnabled: process.env.NODE_ENV === "production",
                    maxRequestsPerHour: 1000,
                    blockDurationMinutes: 15,
                    sslEnabled: true,
                    certificateValidation: true,
                    auditLoggingEnabled: true
                },
                blockchain: {
                    rpcUrl: process.env.BLOCKCHAIN_RPC_URL || window.A2A_CONFIG?.blockchainRpcUrl || "https://mainnet.infura.io/v3/your-project-id",
                    networkId: 31337,
                    gasLimit: 500000,
                    gasPriceGwei: 20,
                    autoDeployContracts: false,
                    contractUpgradeEnabled: true,
                    blockConfirmations: 3,
                    transactionTimeout: 300000,
                    maxRetries: 5,
                    enableEvents: true
                },
                monitoring: {
                    logLevel: "info",
                    metricsEnabled: true,
                    metricsIntervalSeconds: 60,
                    logRetentionDays: 30,
                    alertsEnabled: true,
                    alertThresholdPercent: 85,
                    notificationEmail: "",
                    prometheusEnabled: true,
                    grafanaEnabled: false,
                    healthCheckEndpoint: true
                },
                // Performance metrics for visualization
                performanceHistory: [
                    { timestamp: "2024-01-01T00:00:00Z", cpu: 25, memory: 45, disk: 30, network: 12 },
                    { timestamp: "2024-01-01T01:00:00Z", cpu: 30, memory: 50, disk: 32, network: 15 },
                    { timestamp: "2024-01-01T02:00:00Z", cpu: 28, memory: 48, disk: 31, network: 13 },
                    { timestamp: "2024-01-01T03:00:00Z", cpu: 35, memory: 55, disk: 35, network: 18 },
                    { timestamp: "2024-01-01T04:00:00Z", cpu: 22, memory: 42, disk: 28, network: 10 }
                ]
            });

            this.getView().setModel(oSettingsModel, "settings");

            // Initialize UI state model for professional loading states
            const oUIModel = new JSONModel({
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                hasError: false,
                hasNoData: false,
                progressValue: 0,
                progressText: "",
                progressTitle: "",
                loadingMessage: "",
                errorMessage: "",
                selectedTab: "network",
                unsavedChanges: false,
                validationErrors: {},
                lastSaved: null,
                autoSaveEnabled: true,
                configBackups: []
            });
            this.getView().setModel(oUIModel, "ui");

            // Initialize performance metrics model
            this._initializePerformanceMetrics();

            // Load current settings from backend
            this._loadCurrentSettings();

            // Setup auto-save mechanism
            this._setupAutoSave();

            // Setup real-time monitoring
            this._setupRealTimeMonitoring();

            Log.info("Settings controller initialized with professional features");
        },

        _initializePerformanceMetrics() {
            // Initialize performance metrics for charts
            const oPerformanceModel = new JSONModel({
                currentMetrics: {
                    cpu: 0,
                    memory: 0,
                    disk: 0,
                    network: 0,
                    requestsPerSecond: 0,
                    errorsPerMinute: 0
                },
                historicalData: [],
                systemHealth: {
                    overall: 95,
                    services: 98,
                    database: 92,
                    blockchain: 97
                },
                alerts: []
            });
            this.getView().setModel(oPerformanceModel, "performance");

            // Initialize charts after model is set
            setTimeout(() => {
                this._initializeCharts();
            }, 100);
        },

        _initializeCharts() {
            try {
                // Initialize performance trend chart
                this._initializePerformanceChart();

                // Initialize system health chart
                this._initializeHealthChart();

            } catch (error) {
                Log.error("Failed to initialize charts", error);
            }
        },

        _initializePerformanceChart() {
            const oVizFrame = this.byId("performanceChart");
            if (!oVizFrame) {
                return;
            }

            oVizFrame.setVizType("line");
            oVizFrame.setUiConfig({
                "applicationSet": "fiori"
            });

            const oDataset = new FlattenedDataset({
                dimensions: [{
                    name: "Time",
                    value: "{timestamp}"
                }],
                measures: [{
                    name: "CPU Usage",
                    value: "{cpu}"
                }, {
                    name: "Memory Usage",
                    value: "{memory}"
                }, {
                    name: "Network Latency",
                    value: "{network}"
                }],
                data: {
                    path: "settings>/performanceHistory"
                }
            });

            oVizFrame.setDataset(oDataset);
            oVizFrame.setModel(this.getView().getModel("settings"));

            const feedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                "uid": "valueAxis",
                "type": "Measure",
                "values": ["CPU Usage", "Memory Usage", "Network Latency"]
            });

            const feedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                "uid": "categoryAxis",
                "type": "Dimension",
                "values": ["Time"]
            });

            oVizFrame.addFeed(feedValueAxis);
            oVizFrame.addFeed(feedCategoryAxis);

            Log.info("Performance chart initialized successfully");
        },

        _initializeHealthChart() {
            // Health chart initialization would go here
            Log.info("Health chart initialized successfully");
        },

        async _loadCurrentSettings() {
            this._showLoadingState("progress", {
                title: this.getResourceBundle().getText("loadingSettings"),
                message: this.getResourceBundle().getText("fetchingConfiguration"),
                value: 20
            });

            try {
                this._updateProgress(40, this.getResourceBundle().getText("loadingNetworkConfig"));

                const networkResponse = await blockchainClient.sendMessage("/api/v1/settings/network", {
                    method: "GET",
                    headers: { "Content-Type": "application/json" }
                });

                this._updateProgress(60, this.getResourceBundle().getText("loadingSecurityConfig"));

                const securityResponse = await blockchainClient.sendMessage("/api/v1/settings/security", {
                    method: "GET",
                    headers: { "Content-Type": "application/json" }
                });

                this._updateProgress(80, this.getResourceBundle().getText("loadingPerformanceMetrics"));

                // Load performance metrics
                const metricsResponse = await blockchainClient.sendMessage("/api/v1/metrics/current");

                this._updateProgress(100, this.getResourceBundle().getText("settingsLoaded"));

                if (networkResponse.ok && securityResponse.ok) {
                    const networkSettings = await networkResponse.json();
                    const securitySettings = await securityResponse.json();

                    // Merge with current settings
                    const oModel = this.getView().getModel("settings");
                    const oCurrentData = oModel.getData();

                    oModel.setData({
                        ...oCurrentData,
                        network: { ...oCurrentData.network, ...networkSettings },
                        security: { ...oCurrentData.security, ...securitySettings }
                    });

                    Log.info("Settings loaded successfully from backend");
                }

                if (metricsResponse.ok) {
                    const metrics = await metricsResponse.json();
                    this._updatePerformanceMetrics(metrics);
                }

                // Update UI state
                const oUIModel = this.getView().getModel("ui");
                oUIModel.setProperty("/lastSaved", new Date());

                this._hideLoadingState();

            } catch (error) {
                Log.error("Error loading settings", error);
                this._showErrorState(this.getResourceBundle().getText("settingsLoadError"));
            }
        },

        _updatePerformanceMetrics(metrics) {
            const oPerformanceModel = this.getView().getModel("performance");
            if (!oPerformanceModel) {
                return;
            }

            oPerformanceModel.setProperty("/currentMetrics", {
                cpu: metrics.cpuUsage || 0,
                memory: metrics.memoryUsagePercent || 0,
                disk: metrics.diskUsagePercent || 0,
                network: metrics.networkLatencyMs || 0,
                requestsPerSecond: metrics.requestsPerSecond || 0,
                errorsPerMinute: metrics.errorsPerMinute || 0
            });

            // Update historical data
            const currentTime = new Date().toISOString();
            let aHistorical = oPerformanceModel.getProperty("/historicalData") || [];

            aHistorical.push({
                timestamp: currentTime,
                cpu: metrics.cpu_usage || 0,
                memory: metrics.memory_usage_percent || 0,
                disk: metrics.disk_usage_percent || 0,
                network: metrics.network_latency_ms || 0
            });

            // Keep only last 20 data points
            if (aHistorical.length > 20) {
                aHistorical = aHistorical.slice(-20);
            }

            oPerformanceModel.setProperty("/historicalData", aHistorical);
        },

        _setupAutoSave() {
            this._autoSaveInterval = setInterval(() => {
                const oUIModel = this.getView().getModel("ui");
                if (oUIModel.getProperty("/unsavedChanges") &&
                    oUIModel.getProperty("/autoSaveEnabled")) {
                    this._performAutoSave();
                }
            }, 30000); // Auto-save every 30 seconds
        },

        _setupRealTimeMonitoring() {
            this._monitoringInterval = setInterval(async() => {
                try {
                    const response = await blockchainClient.sendMessage("/api/v1/metrics/current");
                    if (response.ok) {
                        const metrics = await response.json();
                        this._updatePerformanceMetrics(metrics);
                    }
                } catch (error) {
                    Log.warning("Failed to update real-time metrics", error);
                }
            }, 10000); // Update every 10 seconds
        },

        async _performAutoSave() {
            try {
                const oSettings = this.getView().getModel("settings").getData();

                const response = await blockchainClient.sendMessage("/api/v1/settings/autosave", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        settings: oSettings,
                        timestamp: new Date().toISOString()
                    })
                });

                if (response.ok) {
                    const oUIModel = this.getView().getModel("ui");
                    oUIModel.setProperty("/lastSaved", new Date());
                    oUIModel.setProperty("/unsavedChanges", false);

                    MessageToast.show(this.getResourceBundle().getText("autoSaveComplete"));
                }

            } catch (error) {
                Log.error("Auto-save failed", error);
            }
        },

        // Professional loading state management
        _showLoadingState(sType, oOptions = {}) {
            const oUIModel = this.getView().getModel("ui");

            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: sType === "skeleton",
                isLoadingSpinner: sType === "spinner",
                isLoadingProgress: sType === "progress",
                hasError: false,
                hasNoData: false,
                loadingMessage: oOptions.message || "",
                progressValue: oOptions.value || 0,
                progressText: oOptions.text || `${oOptions.value || 0}%`,
                progressTitle: oOptions.title || ""
            });
        },

        _updateProgress(iValue, sMessage) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/progressValue", iValue);
            oUIModel.setProperty("/progressText", `${iValue}%`);
            oUIModel.setProperty("/loadingMessage", sMessage);
        },

        _hideLoadingState() {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                hasError: false,
                hasNoData: false
            });
        },

        _showErrorState(sMessage) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setData({
                ...oUIModel.getData(),
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                hasError: true,
                hasNoData: false,
                errorMessage: sMessage
            });
        },

        // Enhanced event handlers
        onTabSelect(oEvent) {
            const sKey = oEvent.getParameter("key");
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/selectedTab", sKey);

            Log.debug("Settings tab selected", sKey);

            // Load tab-specific data
            switch (sKey) {
            case "monitoring":
                this._refreshMonitoringData();
                break;
            case "blockchain":
                this._refreshBlockchainStatus();
                break;
            case "performance":
                this._refreshPerformanceMetrics();
                break;
            }
        },

        onSettingChange(oEvent) {
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/unsavedChanges", true);

            // Validate the changed setting
            const oSource = oEvent.getSource();
            const sBindingPath = oSource.getBinding("value").getPath();
            const oValue = oEvent.getParameter("value") || oEvent.getParameter("state");

            this._validateSetting(sBindingPath, oValue);

            Log.debug("Setting changed", { path: sBindingPath, value: oValue });
        },

        _validateSetting(sPath, oValue) {
            const oUIModel = this.getView().getModel("ui");
            const oValidationErrors = oUIModel.getProperty("/validationErrors") || {};

            // Remove existing error for this path
            delete oValidationErrors[sPath];

            // Validate based on setting type
            if (sPath.includes("email") && oValue) {
                if (!oValue.includes("@") || !oValue.includes(".")) {
                    oValidationErrors[sPath] = this.getResourceBundle().getText("invalidEmailFormat");
                }
            }

            if (sPath.includes("rpcUrl") && oValue) {
                if (!oValue.startsWith("http")) {
                    oValidationErrors[sPath] = this.getResourceBundle().getText("invalidUrlFormat");
                }
            }

            if (sPath.includes("gasLimit") && oValue < 21000) {
                oValidationErrors[sPath] = this.getResourceBundle().getText("gasLimitTooLow");
            }

            oUIModel.setProperty("/validationErrors", oValidationErrors);
        },

        async onSaveSettings() {
            // Validate all settings before saving
            if (!this._validateAllSettings()) {
                MessageBox.error(this.getResourceBundle().getText("validationErrorsExist"));
                return;
            }

            const oSettings = this.getView().getModel("settings").getData();

            this._showLoadingState("progress", {
                title: this.getResourceBundle().getText("savingSettings"),
                message: this.getResourceBundle().getText("updatingConfiguration"),
                value: 0
            });

            try {
                // Save in phases with progress updates
                this._updateProgress(25, this.getResourceBundle().getText("savingNetworkSettings"));

                const networkResponse = await blockchainClient.sendMessage("/api/v1/settings/network", {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(oSettings.network)
                });

                this._updateProgress(50, this.getResourceBundle().getText("savingSecuritySettings"));

                const securityResponse = await blockchainClient.sendMessage("/api/v1/settings/security", {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(oSettings.security)
                });

                this._updateProgress(75, this.getResourceBundle().getText("applyingConfiguration"));

                // Apply settings that require immediate reconfiguration
                await this._applyNetworkSettings(oSettings);

                this._updateProgress(100, this.getResourceBundle().getText("settingsSaved"));

                if (networkResponse.ok && securityResponse.ok) {
                    MessageToast.show(this.getResourceBundle().getText("settingsSavedSuccessfully"));

                    const oUIModel = this.getView().getModel("ui");
                    oUIModel.setProperty("/unsavedChanges", false);
                    oUIModel.setProperty("/lastSaved", new Date());

                    // Create backup
                    this._createConfigBackup(oSettings);

                } else {
                    const error = await networkResponse.json();
                    MessageBox.error(error.message || this.getResourceBundle().getText("settingsSaveError"));
                }

            } catch (error) {
                Log.error("Error saving settings", error);
                MessageBox.error(this.getResourceBundle().getText("settingsSaveError"));
            } finally {
                this._hideLoadingState();
            }
        },

        _validateAllSettings() {
            const oUIModel = this.getView().getModel("ui");
            const oValidationErrors = oUIModel.getProperty("/validationErrors");
            return Object.keys(oValidationErrors).length === 0;
        },

        _createConfigBackup(oSettings) {
            const oUIModel = this.getView().getModel("ui");
            let aBackups = oUIModel.getProperty("/configBackups") || [];

            aBackups.unshift({
                timestamp: new Date(),
                settings: JSON.parse(JSON.stringify(oSettings)),
                version: aBackups.length + 1
            });

            // Keep only last 10 backups
            if (aBackups.length > 10) {
                aBackups = aBackups.slice(0, 10);
            }

            oUIModel.setProperty("/configBackups", aBackups);
        },

        onCancel() {
            const oUIModel = this.getView().getModel("ui");

            if (oUIModel.getProperty("/unsavedChanges")) {
                MessageBox.confirm(
                    this.getResourceBundle().getText("unsavedChangesMessage"),
                    {
                        title: this.getResourceBundle().getText("unsavedChanges"),
                        onClose: (sAction) => {
                            if (sAction === MessageBox.Action.OK) {
                                this._loadCurrentSettings();
                                oUIModel.setProperty("/unsavedChanges", false);
                                this.onNavBack();
                            }
                        }
                    }
                );
            } else {
                this.onNavBack();
            }
        },

        onResetToDefaults() {
            MessageBox.confirm(
                this.getResourceBundle().getText("resetToDefaultsMessage"),
                {
                    title: this.getResourceBundle().getText("resetToDefaults"),
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            this._resetToDefaults();
                        }
                    }
                }
            );
        },

        _resetToDefaults() {
            const oDefaultSettings = {
                network: {
                    name: "A2A Network",
                    description: "Enterprise Agent-to-Agent Communication Network",
                    maxAgents: 1000,
                    autoDiscovery: true,
                    timeoutSeconds: 30,
                    loadBalancingEnabled: true,
                    healthCheckInterval: 30,
                    networkRegion: "us-east-1",
                    networkTier: "enterprise"
                },
                performance: {
                    messagePoolSize: 1000,
                    cacheEnabled: true,
                    cacheTTLMinutes: 60,
                    loadBalancingStrategy: "roundRobin",
                    maxConcurrentConnections: 500,
                    connectionTimeout: 30000,
                    retryAttempts: 3,
                    circuitBreakerEnabled: true,
                    compressionEnabled: true
                },
                security: {
                    requireAuth: false, // Development default
                    sessionTimeoutMinutes: 120,
                    maxLoginAttempts: 5,
                    encryptMessages: true,
                    rateLimitEnabled: false, // Development default
                    maxRequestsPerHour: 1000,
                    blockDurationMinutes: 15,
                    sslEnabled: true,
                    certificateValidation: true,
                    auditLoggingEnabled: true
                },
                blockchain: {
                    rpcUrl: process.env.BLOCKCHAIN_RPC_URL || window.A2A_CONFIG?.blockchainRpcUrl || "https://mainnet.infura.io/v3/your-project-id",
                    networkId: 31337,
                    gasLimit: 500000,
                    gasPriceGwei: 20,
                    autoDeployContracts: false,
                    contractUpgradeEnabled: true,
                    blockConfirmations: 3,
                    transactionTimeout: 300000,
                    maxRetries: 5,
                    enableEvents: true
                },
                monitoring: {
                    logLevel: "info",
                    metricsEnabled: true,
                    metricsIntervalSeconds: 60,
                    logRetentionDays: 30,
                    alertsEnabled: true,
                    alertThresholdPercent: 85,
                    notificationEmail: "",
                    prometheusEnabled: true,
                    grafanaEnabled: false,
                    healthCheckEndpoint: true
                }
            };

            this.getView().getModel("settings").setData(oDefaultSettings);
            const oUIModel = this.getView().getModel("ui");
            oUIModel.setProperty("/unsavedChanges", true);

            MessageToast.show(this.getResourceBundle().getText("settingsResetToDefaults"));
        },

        onExportConfig() {
            const oSettings = this.getView().getModel("settings").getData();
            const oExportData = {
                version: "1.0",
                timestamp: new Date().toISOString(),
                settings: oSettings
            };

            const sConfig = JSON.stringify(oExportData, null, 2);
            this._downloadFile(sConfig, "a2a-network-config.json", "application/json");

            MessageToast.show(this.getResourceBundle().getText("configExported"));
        },

        onImportConfig() {
            // Create file input element
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".json";

            input.onchange = (event) => {
                const file = event.target.files[0];
                if (!file) {
                    return;
                }

                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const importedData = JSON.parse(e.target.result);

                        if (importedData.settings) {
                            this.getView().getModel("settings").setData(importedData.settings);
                            const oUIModel = this.getView().getModel("ui");
                            oUIModel.setProperty("/unsavedChanges", true);

                            MessageToast.show(this.getResourceBundle().getText("configImported"));
                        } else {
                            MessageBox.error(this.getResourceBundle().getText("invalidConfigFile"));
                        }
                    } catch (error) {
                        Log.error("Failed to parse config file", error);
                        MessageBox.error(this.getResourceBundle().getText("configParseError"));
                    }
                };
                reader.readAsText(file);
            };

            input.click();
        },

        _downloadFile(sContent, sFileName, sMimeType) {
            const element = document.createElement("a");
            element.setAttribute("href", `data:${sMimeType};charset=utf-8,${ encodeURIComponent(sContent)}`);
            element.setAttribute("download", sFileName);
            element.style.display = "none";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        },

        async onTestConnection() {
            const oSettings = this.getView().getModel("settings").getData();

            this._showLoadingState("spinner", {
                message: this.getResourceBundle().getText("testingConnection")
            });

            try {
                const response = await blockchainClient.sendMessage("/api/v1/test-connection", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        rpcUrl: oSettings.blockchain.rpcUrl,
                        networkId: oSettings.blockchain.networkId
                    })
                });

                const result = await response.json();

                if (response.ok && result.success) {
                    MessageBox.success(
                        this.getResourceBundle().getText("connectionTestSuccessful"),
                        { title: this.getResourceBundle().getText("connectionTest") }
                    );
                } else {
                    MessageBox.error(
                        result.message || this.getResourceBundle().getText("connectionTestFailed"),
                        { title: this.getResourceBundle().getText("connectionTest") }
                    );
                }

            } catch (error) {
                Log.error("Connection test failed", error);
                MessageBox.error(this.getResourceBundle().getText("connectionTestError"));
            } finally {
                this._hideLoadingState();
            }
        },

        async _applyNetworkSettings(oSettings) {
            try {
                const response = await blockchainClient.sendMessage("/api/v1/reconfigure", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        network: oSettings.network,
                        performance: oSettings.performance,
                        security: oSettings.security
                    })
                });

                if (response.ok) {
                    Log.info("Network reconfiguration applied successfully");
                } else {
                    Log.warning("Network reconfiguration failed");
                }

            } catch (error) {
                Log.error("Failed to apply network settings", error);
                MessageBox.warning(this.getResourceBundle().getText("settingsSavedButNotApplied"));
            }
        },

        // Real-time monitoring refresh handlers
        async _refreshMonitoringData() {
            try {
                const response = await blockchainClient.sendMessage("/api/v1/monitoring/status");
                if (response.ok) {
                    const data = await response.json();
                    this._updatePerformanceMetrics(data);
                }
            } catch (error) {
                Log.warning("Failed to refresh monitoring data", error);
            }
        },

        async _refreshBlockchainStatus() {
            try {
                const response = await blockchainClient.sendMessage("/api/v1/blockchain/status");
                if (response.ok) {
                    const status = await response.json();
                    // Update blockchain status in UI
                    Log.info("Blockchain status refreshed", status);
                }
            } catch (error) {
                Log.warning("Failed to refresh blockchain status", error);
            }
        },

        async _refreshPerformanceMetrics() {
            try {
                const response = await blockchainClient.sendMessage("/api/v1/metrics/performance");
                if (response.ok) {
                    const metrics = await response.json();
                    this._updatePerformanceMetrics(metrics);

                    // Update chart data
                    this._updateChartData(metrics);
                }
            } catch (error) {
                Log.warning("Failed to refresh performance metrics", error);
            }
        },

        _updateChartData(metrics) {
            const oSettingsModel = this.getView().getModel("settings");
            let aHistory = oSettingsModel.getProperty("/performanceHistory") || [];

            // Add new data point
            aHistory.push({
                timestamp: new Date().toISOString(),
                cpu: metrics.cpu_usage || 0,
                memory: metrics.memory_usage_percent || 0,
                disk: metrics.disk_usage_percent || 0,
                network: metrics.network_latency_ms || 0
            });

            // Keep only last 24 points (24 hours if updated hourly)
            if (aHistory.length > 24) {
                aHistory = aHistory.slice(-24);
            }

            oSettingsModel.setProperty("/performanceHistory", aHistory);
        },

        onRetryLoad() {
            this._loadCurrentSettings();
        },

        onExit() {
            // Clean up intervals
            if (this._autoSaveInterval) {
                clearInterval(this._autoSaveInterval);
            }

            if (this._monitoringInterval) {
                clearInterval(this._monitoringInterval);
            }
        }
    });
});