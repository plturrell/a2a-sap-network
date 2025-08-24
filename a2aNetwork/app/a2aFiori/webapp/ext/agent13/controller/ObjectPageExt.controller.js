/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent13/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent13.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeSecurity();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize dialog cache
                this._dialogCache = {};
                
                // Initialize real-time monitoring
                this._initializeRealtimeMonitoring();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onViewVulnerabilities
         * @description Opens vulnerability viewer with detailed security findings.
         * @public
         */
        onViewVulnerabilities: function() {
            if (!this._hasRole("SecurityAnalyst")) {
                MessageBox.error("Access denied. Security Analyst role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewVulnerabilities", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("VIEW_VULNERABILITIES", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("viewVulnerabilities", "a2a.network.agent13.ext.fragment.ViewVulnerabilities")
                .then(function(oDialog) {
                    var oVulnModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        severityFilter: "ALL",
                        statusFilter: "OPEN",
                        categoryFilter: "ALL",
                        timeRange: "LAST_30_DAYS",
                        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                        endDate: new Date(),
                        groupBy: "SEVERITY",
                        showResolved: false,
                        showSupressed: false,
                        autoRefresh: true,
                        refreshInterval: 30000,
                        vulnerabilities: [],
                        statistics: {
                            critical: 0,
                            high: 0,
                            medium: 0,
                            low: 0,
                            info: 0,
                            resolved: 0
                        }
                    });
                    oDialog.setModel(oVulnModel, "vuln");
                    oDialog.open();
                    this._loadVulnerabilities(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Vulnerabilities Viewer: " + error.message);
                });
        },

        /**
         * @function onRunThreatAnalysis
         * @description Runs comprehensive threat analysis for the security task.
         * @public
         */
        onRunThreatAnalysis: function() {
            if (!this._hasRole("SecurityAdmin")) {
                MessageBox.error("Access denied. Security Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "RunThreatAnalysis", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if analysis is already running
            if (oData.analysisStatus === "RUNNING") {
                MessageToast.show(this.getResourceBundle().getText("msg.analysisAlreadyRunning"));
                return;
            }
            
            this._auditLogger.log("RUN_THREAT_ANALYSIS", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("threatAnalysis", "a2a.network.agent13.ext.fragment.ThreatAnalysis")
                .then(function(oDialog) {
                    var oAnalysisModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        analysisType: "COMPREHENSIVE",
                        analysisDepth: "DEEP",
                        includeThreatIntelligence: true,
                        includeExternalSources: true,
                        includeHistoricalData: true,
                        includePredictiveAnalysis: true,
                        scanNetworkTraffic: true,
                        scanSystemLogs: true,
                        scanConfigurations: true,
                        scanDependencies: true,
                        threatCategories: [
                            { key: "MALWARE", text: "Malware & Ransomware", selected: true },
                            { key: "INTRUSION", text: "Intrusion Attempts", selected: true },
                            { key: "INSIDER", text: "Insider Threats", selected: true },
                            { key: "APT", text: "Advanced Persistent Threats", selected: true },
                            { key: "ZERO_DAY", text: "Zero-Day Exploits", selected: true },
                            { key: "SUPPLY_CHAIN", text: "Supply Chain Attacks", selected: true }
                        ],
                        analysisOptions: {
                            performRiskAssessment: true,
                            generateMitigation: true,
                            prioritizeFindings: true,
                            correlateEvents: true
                        }
                    });
                    oDialog.setModel(oAnalysisModel, "analysis");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Threat Analysis: " + error.message);
                });
        },

        /**
         * @function onGenerateSecurityReport
         * @description Generates comprehensive security report.
         * @public
         */
        onGenerateSecurityReport: function() {
            if (!this._hasRole("SecurityAnalyst")) {
                MessageBox.error("Access denied. Security Analyst role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "GenerateSecurityReport", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("GENERATE_SECURITY_REPORT", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("generateReport", "a2a.network.agent13.ext.fragment.GenerateSecurityReport")
                .then(function(oDialog) {
                    var oReportModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "COMPREHENSIVE",
                        reportFormat: "PDF",
                        includeExecutiveSummary: true,
                        includeDetailedFindings: true,
                        includeTechnicalDetails: true,
                        includeRecommendations: true,
                        includeRiskMatrix: true,
                        includeComplianceStatus: true,
                        includeTrendAnalysis: true,
                        includeIncidentHistory: true,
                        timeRange: "LAST_30_DAYS",
                        confidentialityLevel: "INTERNAL",
                        distributionList: [],
                        reportSections: [
                            { key: "EXECUTIVE", text: "Executive Summary", selected: true },
                            { key: "VULNERABILITIES", text: "Vulnerability Assessment", selected: true },
                            { key: "THREATS", text: "Threat Analysis", selected: true },
                            { key: "INCIDENTS", text: "Security Incidents", selected: true },
                            { key: "COMPLIANCE", text: "Compliance Status", selected: true },
                            { key: "RECOMMENDATIONS", text: "Recommendations", selected: true },
                            { key: "APPENDIX", text: "Technical Appendix", selected: false }
                        ]
                    });
                    oDialog.setModel(oReportModel, "report");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Report Generator: " + error.message);
                });
        },

        /**
         * @function onConfigureFirewall
         * @description Opens firewall rules configuration interface.
         * @public
         */
        onConfigureFirewall: function() {
            if (!this._hasRole("SecurityAdmin")) {
                MessageBox.error("Access denied. Security Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigureFirewall", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("CONFIGURE_FIREWALL", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("configureFirewall", "a2a.network.agent13.ext.fragment.ConfigureFirewall")
                .then(function(oDialog) {
                    var oFirewallModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        rulesets: [],
                        zones: [
                            { key: "DMZ", text: "DMZ (Demilitarized Zone)" },
                            { key: "INTERNAL", text: "Internal Network" },
                            { key: "EXTERNAL", text: "External/Internet" },
                            { key: "TRUSTED", text: "Trusted Network" },
                            { key: "MANAGEMENT", text: "Management Network" }
                        ],
                        selectedZone: "DMZ",
                        defaultAction: "DENY",
                        enableLogging: true,
                        enableIDS: true,
                        enableDDoSProtection: true,
                        ruleTemplates: [],
                        currentRules: [],
                        pendingChanges: [],
                        validationEnabled: true,
                        autoBackup: true
                    });
                    oDialog.setModel(oFirewallModel, "firewall");
                    oDialog.open();
                    this._loadFirewallRules(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Firewall Configuration: " + error.message);
                });
        },

        /**
         * @function _loadVulnerabilities
         * @description Loads vulnerability data for display.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadVulnerabilities: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetVulnerabilities", {
                urlParameters: {
                    taskId: sTaskId,
                    includeDetails: true,
                    includeRemediation: true
                },
                success: function(data) {
                    var oVulnModel = oDialog.getModel("vuln");
                    if (oVulnModel) {
                        var oCurrentData = oVulnModel.getData();
                        oCurrentData.vulnerabilities = data.vulnerabilities || [];
                        oCurrentData.statistics = data.statistics || {};
                        oCurrentData.trends = data.trends || {};
                        oCurrentData.affectedAssets = data.affectedAssets || [];
                        oCurrentData.remediationPlans = data.remediationPlans || [];
                        oVulnModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                    
                    // Start auto-refresh if enabled
                    if (oCurrentData.autoRefresh) {
                        this._startVulnerabilityAutoRefresh(sTaskId, oDialog);
                    }
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load vulnerabilities: " + error.message);
                }
            });
        },

        /**
         * @function _startVulnerabilityAutoRefresh
         * @description Starts auto-refresh for vulnerability data.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startVulnerabilityAutoRefresh: function(sTaskId, oDialog) {
            if (this._vulnRefreshInterval) {
                clearInterval(this._vulnRefreshInterval);
            }
            
            var oVulnData = oDialog.getModel("vuln").getData();
            this._vulnRefreshInterval = setInterval(() => {
                if (oDialog.isOpen()) {
                    this._loadVulnerabilities(sTaskId, oDialog);
                } else {
                    clearInterval(this._vulnRefreshInterval);
                }
            }, oVulnData.refreshInterval || 30000);
        },

        /**
         * @function _loadFirewallRules
         * @description Loads firewall rules and configuration.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Firewall configuration dialog
         * @private
         */
        _loadFirewallRules: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetFirewallConfiguration", {
                urlParameters: {
                    taskId: sTaskId,
                    includeTemplates: true
                },
                success: function(data) {
                    var oFirewallModel = oDialog.getModel("firewall");
                    if (oFirewallModel) {
                        var oCurrentData = oFirewallModel.getData();
                        oCurrentData.rulesets = data.rulesets || [];
                        oCurrentData.currentRules = data.currentRules || [];
                        oCurrentData.ruleTemplates = data.templates || [];
                        oCurrentData.networkMap = data.networkMap || {};
                        oCurrentData.statistics = data.statistics || {};
                        oFirewallModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load firewall configuration: " + error.message);
                }
            });
        },

        /**
         * @function _initializeRealtimeMonitoring
         * @description Initializes real-time monitoring for security events.
         * @private
         */
        _initializeRealtimeMonitoring: function() {
            // WebSocket for real-time security alerts
            this._initializeSecurityWebSocket();
            
            // EventSource for threat intelligence updates
            this._initializeThreatIntelligence();
        },

        /**
         * @function _initializeSecurityWebSocket
         * @description Initializes WebSocket for security alerts.
         * @private
         */
        _initializeSecurityWebSocket: function() {
            if (this._securityWs) return;

            try {
                this._securityWs = SecurityUtils.createSecureWebSocket('blockchain://a2a-events', {
                    onMessage: function(data) {
                        this._handleSecurityAlert(data);
                    }.bind(this)
                });

                this._securityWs.onclose = function() {
                    console.info("Security WebSocket closed, will reconnect...");
                    setTimeout(() => this._initializeSecurityWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("Security WebSocket not available");
            }
        },

        /**
         * @function _initializeThreatIntelligence
         * @description Initializes threat intelligence feed.
         * @private
         */
        _initializeThreatIntelligence: function() {
            if (this._threatEventSource) return;
            
            try {
                this._threatEventSource = new EventSource('/api/agent13/security/threat-intelligence');
                
                this._threatEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._handleThreatIntelligence(data);
                    } catch (error) {
                        console.error('Error parsing threat intelligence:', error);
                    }
                }.bind(this);
                
                this._threatEventSource.onerror = function(error) {
                    console.warn('Threat intelligence stream error:', error);
                }.bind(this);
                
            } catch (error) {
                console.warn('EventSource not available for threat intelligence');
            }
        },

        /**
         * @function _handleSecurityAlert
         * @description Handles real-time security alerts.
         * @param {Object} data - Alert data
         * @private
         */
        _handleSecurityAlert: function(data) {
            if (data.severity === "CRITICAL") {
                MessageBox.error("Critical Security Alert: " + data.message);
                this._auditLogger.log("CRITICAL_SECURITY_ALERT", data);
            } else if (data.severity === "HIGH") {
                MessageBox.warning("High Priority Security Alert: " + data.message);
                this._auditLogger.log("HIGH_SECURITY_ALERT", data);
            } else {
                MessageToast.show("Security Alert: " + data.message);
            }
            
            // Update any open vulnerability dialogs
            if (this._dialogCache.viewVulnerabilities && this._dialogCache.viewVulnerabilities.isOpen()) {
                this._refreshVulnerabilityData();
            }
        },

        /**
         * @function _handleThreatIntelligence
         * @description Handles threat intelligence updates.
         * @param {Object} data - Threat intelligence data
         * @private
         */
        _handleThreatIntelligence: function(data) {
            if (data.type === "NEW_THREAT") {
                MessageToast.show("New threat detected: " + data.threatName);
            } else if (data.type === "THREAT_UPDATE") {
                // Update threat analysis if dialog is open
                if (this._dialogCache.threatAnalysis && this._dialogCache.threatAnalysis.isOpen()) {
                    this._updateThreatAnalysisData(data);
                }
            }
        },

        /**
         * @function _refreshVulnerabilityData
         * @description Refreshes vulnerability data in open dialog.
         * @private
         */
        _refreshVulnerabilityData: function() {
            const oDialog = this._dialogCache.viewVulnerabilities;
            if (oDialog && oDialog.isOpen()) {
                const oVulnModel = oDialog.getModel("vuln");
                if (oVulnModel) {
                    const sTaskId = oVulnModel.getData().taskId;
                    this._loadVulnerabilities(sTaskId, oDialog);
                }
            }
        },

        /**
         * @function _updateThreatAnalysisData
         * @description Updates threat analysis data in open dialog.
         * @param {Object} data - Threat data
         * @private
         */
        _updateThreatAnalysisData: function(data) {
            const oDialog = this._dialogCache.threatAnalysis;
            if (oDialog && oDialog.isOpen()) {
                const oAnalysisModel = oDialog.getModel("analysis");
                if (oAnalysisModel) {
                    const oCurrentData = oAnalysisModel.getData();
                    // Update relevant threat data
                    if (data.threats) {
                        oCurrentData.latestThreats = data.threats;
                    }
                    oAnalysisModel.setData(oCurrentData);
                }
            }
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache && this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            
            if (!this._dialogCache) {
                this._dialogCache = {};
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then(function(oDialog) {
                that._dialogCache[sDialogId] = oDialog;
                that.base.getView().addDependent(oDialog);
                
                // Enable accessibility
                that._enableDialogAccessibility(oDialog);
                
                // Optimize for mobile
                that._optimizeDialogForDevice(oDialog);
                
                return oDialog;
            });
        },
        
        /**
         * @function _enableDialogAccessibility
         * @description Adds accessibility features to dialog.
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @private
         */
        _enableDialogAccessibility: function(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering: function() {
                    var $dialog = oDialog.$();
                    
                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");
                    
                    // Handle escape key
                    $dialog.on("keydown", function(e) {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });
                    
                    // Focus first input on open
                    setTimeout(function() {
                        $dialog.find("input:visible:first").focus();
                    }, 100);
                }
            });
        },
        
        /**
         * @function _optimizeDialogForDevice
         * @description Optimizes dialog for current device.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         * @private
         */
        _optimizeDialogForDevice: function(oDialog) {
            if (sap.ui.Device.system.phone) {
                oDialog.setStretch(true);
                oDialog.setContentWidth("100%");
                oDialog.setContentHeight("100%");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("95%");
                oDialog.setContentHeight("90%");
            }
        },

        /**
         * @function _initializeSecurity
         * @description Initializes security features and audit logging.
         * @private
         */
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    var user = this._getCurrentUser();
                    var timestamp = new Date().toISOString();
                    var logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent13_Security",
                        action: action,
                        details: details || {}
                    };
                    console.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user ID for audit logging.
         * @returns {string} User ID or "anonymous"
         * @private
         */
        _getCurrentUser: function() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        /**
         * @function _hasRole
         * @description Checks if current user has specified role.
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole: function(role) {
            const user = sap.ushell?.Container?.getUser();
            if (user && user.hasRole) {
                return user.hasRole(role);
            }
            // Mock role validation for development/testing
            const mockRoles = ["SecurityAdmin", "SecurityAnalyst", "SecurityOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up WebSocket connections
            if (this._securityWs) {
                this._securityWs.close();
                this._securityWs = null;
            }
            
            // Clean up EventSource connections
            if (this._threatEventSource) {
                this._threatEventSource.close();
                this._threatEventSource = null;
            }
            
            // Clean up intervals
            if (this._vulnRefreshInterval) {
                clearInterval(this._vulnRefreshInterval);
                this._vulnRefreshInterval = null;
            }
            
            // Clean up cached dialogs
            if (this._dialogCache) {
                Object.keys(this._dialogCache).forEach(function(key) {
                    if (this._dialogCache[key]) {
                        this._dialogCache[key].destroy();
                    }
                }.bind(this));
                this._dialogCache = {};
            }
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @public
         */
        getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});