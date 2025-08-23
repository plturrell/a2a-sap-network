sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/security/encodeURL",
    "sap/base/Log",
    "sap/ui/core/routing/Router",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, encodeURL, Log, Router, escapeRegExp, sanitizeHTML) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent6.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._initializeSecurity();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize create dialog model
                this._initializeCreateModel();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },
        
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent6_QualityControl",
                        action: action,
                        details: details
                    };
                    Log.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },
        
        _getCurrentUser: function() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },
        
        _hasRole: function(role) {
            const user = sap.ushell?.Container?.getUser();
            return user && user.hasRole && user.hasRole(role);
        },
        
        _validateInput: function(input, type) {
            if (!input || typeof input !== 'string') return false;
            
            switch(type) {
                case 'taskId':
                    return /^[a-zA-Z0-9\-_]{1,36}$/.test(input);
                case 'reason':
                    return input.length <= 500;
                case 'agent':
                    return /^[a-zA-Z0-9_]{1,50}$/.test(input);
                case 'url':
                    return /^https:\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(input);
                default:
                    return input.length > 0 && input.length <= 255;
            }
        },
        
        _sanitizeInput: function(input) {
            if (!input) return "";
            return encodeXML(input.toString().trim());
        },
        
        _validateEventSourceUrl: function(url) {
            return this._validateInput(url, 'url') && url.startsWith('/a2a/agent6/v1/');
        },
        
        _validateDownloadUrl: function(url) {
            return this._validateInput(url, 'url') && 
                   url.startsWith('/a2a/agent6/v1/') &&
                   !url.includes('..');
        },
        
        _sanitizeFilename: function(filename) {
            if (!filename) return "download";
            return filename.replace(/[^a-zA-Z0-9._-]/g, '_').substring(0, 100);
        },
        
        _getCsrfToken: function() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/a2a/agent6/v1/csrf-token",
                    type: "GET",
                    headers: {
                        "X-CSRF-Token": "Fetch"
                    },
                    success: function(data, textStatus, xhr) {
                        const token = xhr.getResponseHeader("X-CSRF-Token");
                        resolve(token);
                    },
                    error: function(xhr) {
                        reject(new Error("Failed to fetch CSRF token"));
                    }
                });
            });
        },
        
        _secureAjaxCall: function(options) {
            return this._getCsrfToken().then(token => {
                return new Promise((resolve, reject) => {
                    const secureOptions = Object.assign({}, options, {
                        headers: Object.assign({
                            "X-CSRF-Token": token,
                            "Content-Type": "application/json"
                        }, options.headers || {}),
                        success: function(data, textStatus, xhr) {
                            resolve({ data, textStatus, xhr });
                        },
                        error: function(xhr, textStatus, errorThrown) {
                            reject({ xhr, textStatus, errorThrown });
                        }
                    });
                    
                    jQuery.ajax(secureOptions);
                });
            });
        },
        
        _cleanupResources: function() {
            if (this._eventSource) {
                this._eventSource.close();
                this._eventSource = null;
            }
            if (this._oResultsDialog) {
                this._oResultsDialog.destroy();
                this._oResultsDialog = null;
            }
            if (this._oRoutingDecisionDialog) {
                this._oRoutingDecisionDialog.destroy();
                this._oRoutingDecisionDialog = null;
            }
            if (this._oTrustResultsDialog) {
                this._oTrustResultsDialog.destroy();
                this._oTrustResultsDialog = null;
            }
            if (this._oReportDialog) {
                this._oReportDialog.destroy();
                this._oReportDialog = null;
            }
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
        },
        
        _initializeCreateModel: function() {
            this._oCreateModel = new JSONModel({
                taskName: "",
                description: "",
                qualityGate: "",
                dataSource: "",
                processingPipeline: "",
                evaluationCriteria: {
                    compliance: true,
                    performance: false,
                    security: false,
                    reliability: false,
                    usability: false,
                    maintainability: false
                },
                thresholds: {
                    minQualityScore: 80,
                    maxIssues: 5,
                    minTrustScore: 85
                },
                routingStrategy: "QUALITY_BASED",
                autoRouteThreshold: 90,
                enableFallback: true,
                requireManualApproval: false,
                notifyOnRouting: true,
                weights: {
                    quality: 40,
                    performance: 30,
                    trust: 30
                },
                trustLevel: "STANDARD",
                verification: {
                    blockchain: false,
                    reputation: true,
                    integrity: true,
                    consensus: false,
                    signature: false
                },
                trustFactors: {
                    performance: 30,
                    dataQuality: 40,
                    security: 30
                },
                // Validation states
                taskNameState: "None",
                qualityGateState: "None",
                dataSourceState: "None",
                taskNameStateText: "",
                qualityGateStateText: "",
                dataSourceStateText: ""
            });
        },

        onStartAssessment: function() {
            if (!this._hasRole("QualityManager")) {
                MessageBox.error("Access denied: Insufficient privileges for starting assessments");
                this._auditLogger.log("ASSESSMENT_ACCESS_DENIED", { action: "start_assessment" });
                return;
            }
            
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            var sTaskName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._validateInput(sTaskId, 'taskId')) {
                MessageBox.error("Invalid task ID format");
                return;
            }
            
            MessageBox.confirm("Start quality assessment for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startQualityAssessment(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startQualityAssessment: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/assess",
                type: "POST"
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                MessageToast.show("Quality assessment started");
                this._extensionAPI.refresh();
                
                this._startRealtimeMonitoring(sTaskId);
                
                this._auditLogger.log("ASSESSMENT_STARTED", { taskId: sTaskId });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to start assessment: " + errorMsg);
                this._auditLogger.log("ASSESSMENT_START_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _startRealtimeMonitoring: function(sTaskId) {
            const streamUrl = "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/stream";
            
            if (!this._validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid monitoring stream URL");
                return;
            }
            
            this._eventSource = new EventSource(streamUrl);
            
            this._eventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "progress") {
                        this._updateAssessmentProgress(data);
                    } else if (data.type === "complete") {
                        this._eventSource.close();
                        this._extensionAPI.refresh();
                        MessageBox.success("Quality assessment completed successfully!");
                        this._showAssessmentResults(data.results);
                        this._auditLogger.log("ASSESSMENT_COMPLETED", { taskId: sTaskId });
                    } else if (data.type === "error") {
                        this._eventSource.close();
                        const errorMsg = this._sanitizeInput(data.error || "Unknown error");
                        MessageBox.error("Assessment failed: " + errorMsg);
                        this._auditLogger.log("ASSESSMENT_FAILED", { taskId: sTaskId, error: errorMsg });
                    }
                } catch (e) {
                    this._eventSource.close();
                    MessageBox.error("Invalid data received from assessment stream");
                    this._auditLogger.log("ASSESSMENT_STREAM_ERROR", { taskId: sTaskId, error: e.message });
                }
            }.bind(this);
            
            this._eventSource.onerror = function() {
                if (this._eventSource) {
                    this._eventSource.close();
                    this._eventSource = null;
                }
                MessageBox.error("Lost connection to assessment process");
                this._auditLogger.log("ASSESSMENT_CONNECTION_LOST", { taskId: sTaskId });
            }.bind(this);
        },

        _updateAssessmentProgress: function(data) {
            MessageToast.show(data.component + ": " + data.progress + "%");
        },

        _showAssessmentResults: function(results) {
            var oView = this.base.getView();
            
            if (!this._oResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.AssessmentResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oResultsDialog = oDialog;
                    oView.addDependent(this._oResultsDialog);
                    
                    var oModel = new JSONModel(results);
                    this._oResultsDialog.setModel(oModel, "results");
                    this._oResultsDialog.open();
                    
                    this._createResultsVisualizations(results);
                }.bind(this));
            } else {
                var oModel = new JSONModel(results);
                this._oResultsDialog.setModel(oModel, "results");
                this._oResultsDialog.open();
                this._createResultsVisualizations(results);
            }
        },

        _createResultsVisualizations: function(results) {
            this._createQualityScoreRadar(results.scores);
            this._createIssueBreakdown(results.issues);
        },

        _createQualityScoreRadar: function(scores) {
            var oRadarChart = this._oResultsDialog.byId("qualityRadarChart");
            if (!oRadarChart || !scores) return;
            
            var aRadarData = Object.keys(scores).map(function(key) {
                return {
                    Component: key,
                    Score: scores[key]
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                radarData: aRadarData
            });
            oRadarChart.setModel(oChartModel);
        },

        onMakeRoutingDecision: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var oQualityScore = oContext.getProperty("overallQuality");
            
            if (!this._oRoutingDecisionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.RoutingDecisionDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oRoutingDecisionDialog = oDialog;
                    this.base.getView().addDependent(this._oRoutingDecisionDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        currentQuality: oQualityScore,
                        decision: "",
                        targetAgent: "",
                        confidence: 0,
                        reason: "",
                        priority: "NORMAL"
                    });
                    this._oRoutingDecisionDialog.setModel(oModel, "routing");
                    this._oRoutingDecisionDialog.open();
                    
                    this._loadRoutingOptions(sTaskId);
                }.bind(this));
            } else {
                this._oRoutingDecisionDialog.open();
                this._loadRoutingOptions(sTaskId);
            }
        },

        _loadRoutingOptions: function(sTaskId) {
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/routing-options",
                type: "GET"
            }).then(result => {
                var oModel = this._oRoutingDecisionDialog.getModel("routing");
                var oData = oModel.getData();
                
                oData.availableAgents = this._sanitizeArray(result.data.agents);
                oData.recommendations = this._sanitizeArray(result.data.recommendations);
                oData.routingHistory = this._sanitizeArray(result.data.history);
                
                oModel.setData(oData);
                
                this._auditLogger.log("ROUTING_OPTIONS_LOADED", { taskId: sTaskId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load routing options: " + errorMsg);
                this._auditLogger.log("ROUTING_OPTIONS_LOAD_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onConfirmRoutingDecision: function() {
            var oModel = this._oRoutingDecisionDialog.getModel("routing");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.decision)) {
                MessageBox.error("Please select a valid decision");
                return;
            }
            
            if (!this._validateInput(oData.targetAgent, 'agent')) {
                MessageBox.error("Please select a valid target agent");
                return;
            }
            
            if (!this._validateInput(oData.reason, 'reason')) {
                MessageBox.error("Invalid reason format (max 500 characters)");
                return;
            }
            
            const sanitizedData = {
                decision: this._sanitizeInput(oData.decision),
                targetAgent: this._sanitizeInput(oData.targetAgent),
                confidence: Math.max(0, Math.min(100, parseInt(oData.confidence) || 0)),
                reason: this._sanitizeInput(oData.reason),
                priority: this._sanitizeInput(oData.priority)
            };
            
            this._oRoutingDecisionDialog.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(oData.taskId) + "/route",
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oRoutingDecisionDialog.setBusy(false);
                this._oRoutingDecisionDialog.close();
                
                const data = result.data;
                MessageBox.success(
                    "Routing decision made successfully!\\n" +
                    "Task routed to: " + this._sanitizeInput(sanitizedData.targetAgent) + "\\n" +
                    "Estimated processing time: " + this._sanitizeInput(data.estimatedTime) + " minutes"
                );
                
                this._extensionAPI.refresh();
                
                this._auditLogger.log("ROUTING_DECISION_MADE", {
                    taskId: oData.taskId,
                    decision: sanitizedData.decision,
                    targetAgent: sanitizedData.targetAgent,
                    confidence: sanitizedData.confidence
                });
            }).catch(error => {
                this._oRoutingDecisionDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to make routing decision: " + errorMsg);
                this._auditLogger.log("ROUTING_DECISION_FAILED", { taskId: oData.taskId, error: errorMsg });
            });
        },

        onVerifyTrust: function() {
            if (!this._hasRole("TrustVerifier")) {
                MessageBox.error("Access denied: Insufficient privileges for trust verification");
                this._auditLogger.log("TRUST_VERIFY_ACCESS_DENIED", { action: "verify_trust" });
                return;
            }
            
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            
            if (!this._validateInput(sTaskId, 'taskId')) {
                MessageBox.error("Invalid task ID format");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/verify-trust",
                type: "POST"
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                this._showTrustVerificationResults(result.data);
                
                this._auditLogger.log("TRUST_VERIFICATION_COMPLETED", { taskId: sTaskId });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Trust verification failed: " + errorMsg);
                this._auditLogger.log("TRUST_VERIFICATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showTrustVerificationResults: function(verificationData) {
            var oView = this.base.getView();
            
            if (!this._oTrustResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.TrustVerificationResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oTrustResultsDialog = oDialog;
                    oView.addDependent(this._oTrustResultsDialog);
                    
                    var oModel = new JSONModel(verificationData);
                    this._oTrustResultsDialog.setModel(oModel, "trust");
                    this._oTrustResultsDialog.open();
                    
                    this._createTrustVisualizations(verificationData);
                }.bind(this));
            } else {
                var oModel = new JSONModel(verificationData);
                this._oTrustResultsDialog.setModel(oModel, "trust");
                this._oTrustResultsDialog.open();
                this._createTrustVisualizations(verificationData);
            }
        },

        _createTrustVisualizations: function(data) {
            var oTrustChart = this._oTrustResultsDialog.byId("trustFactorsChart");
            if (!oTrustChart || !data.factors) return;
            
            var aChartData = data.factors.map(function(factor) {
                return {
                    Factor: factor.name,
                    Score: factor.score,
                    Weight: factor.weight
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                factorData: aChartData
            });
            oTrustChart.setModel(oChartModel);
        },

        onGenerateQualityReport: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.QualityReport",
                    controller: this
                }).then(function(oDialog) {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "COMPREHENSIVE",
                        includeCharts: true,
                        includeRecommendations: true,
                        format: "PDF"
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                }.bind(this));
            } else {
                this._oReportDialog.open();
            }
        },

        onExecuteReportGeneration: function() {
            var oModel = this._oReportDialog.getModel("report");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.taskId, 'taskId')) {
                MessageBox.error("Invalid task ID format");
                return;
            }
            
            const allowedFormats = ['PDF', 'EXCEL', 'JSON'];
            if (!allowedFormats.includes(oData.format)) {
                MessageBox.error("Invalid report format");
                return;
            }
            
            const sanitizedData = {
                reportType: this._sanitizeInput(oData.reportType),
                includeCharts: Boolean(oData.includeCharts),
                includeRecommendations: Boolean(oData.includeRecommendations),
                format: this._sanitizeInput(oData.format)
            };
            
            this._oReportDialog.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(oData.taskId) + "/report",
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oReportDialog.setBusy(false);
                this._oReportDialog.close();
                
                const data = result.data;
                const sanitizedFilename = this._sanitizeFilename(data.filename);
                
                if (!this._validateDownloadUrl(data.downloadUrl)) {
                    MessageBox.error("Invalid download URL received");
                    return;
                }
                
                MessageBox.information(
                    "Quality report generated successfully!\\n\\nDownload: " + sanitizedFilename,
                    {
                        actions: ["Download", MessageBox.Action.CLOSE],
                        onClose: function(oAction) {
                            if (oAction === "Download") {
                                window.open(data.downloadUrl, "_blank");
                            }
                        }.bind(this)
                    }
                );
                
                this._auditLogger.log("REPORT_GENERATED", {
                    taskId: oData.taskId,
                    reportType: sanitizedData.reportType,
                    format: sanitizedData.format
                });
            }).catch(error => {
                this._oReportDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Report generation failed: " + errorMsg);
                this._auditLogger.log("REPORT_GENERATION_FAILED", { taskId: oData.taskId, error: errorMsg });
            });
        },

        onOptimizeWorkflow: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm(
                "Analyze and optimize workflow for this task? This will identify bottlenecks and suggest improvements.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeWorkflow(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeWorkflow: function(sTaskId) {
            const requestData = {
                analysisDepth: "COMPREHENSIVE",
                includeResourceOptimization: true,
                applyOptimizations: false
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/optimize",
                type: "POST",
                data: JSON.stringify(requestData)
            }).then(result => {
                this._showOptimizationResults(result.data);
                
                this._auditLogger.log("WORKFLOW_OPTIMIZATION_ANALYZED", {
                    taskId: sTaskId,
                    analysisDepth: requestData.analysisDepth
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Workflow optimization failed: " + errorMsg);
                this._auditLogger.log("WORKFLOW_OPTIMIZATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showOptimizationResults: function(optimizationData) {
            var sMessage = "Workflow Analysis Results:\\n\\n";
            
            if (optimizationData.bottlenecks && optimizationData.bottlenecks.length > 0) {
                sMessage += "Bottlenecks Found:\\n";
                optimizationData.bottlenecks.forEach(function(bottleneck) {
                    sMessage += "• " + bottleneck.stage + ": " + bottleneck.impact + "\\n";
                });
                sMessage += "\\n";
            }
            
            if (optimizationData.recommendations && optimizationData.recommendations.length > 0) {
                sMessage += "Optimization Recommendations:\\n";
                optimizationData.recommendations.forEach(function(rec) {
                    sMessage += "• " + rec.description + " (Impact: " + rec.expectedImprovement + ")\\n";
                });
            }
            
            MessageBox.information(sMessage, {
                actions: ["Apply Optimizations", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Apply Optimizations") {
                        this._applyOptimizations(optimizationData.optimizations);
                    }
                }.bind(this)
            });
        },

        _applyOptimizations: function(optimizations) {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            
            if (!this._validateInput(sTaskId, 'taskId')) {
                MessageBox.error("Invalid task ID format");
                return;
            }
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/apply-optimizations",
                type: "POST",
                data: JSON.stringify({ optimizations: optimizations })
            }).then(result => {
                MessageBox.success("Workflow optimizations applied successfully!");
                this._extensionAPI.refresh();
                
                this._auditLogger.log("WORKFLOW_OPTIMIZATIONS_APPLIED", {
                    taskId: sTaskId,
                    optimizationCount: optimizations?.length || 0
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to apply optimizations: " + errorMsg);
                this._auditLogger.log("WORKFLOW_OPTIMIZATIONS_APPLY_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onEscalateIssues: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var aIssues = oContext.getProperty("issuesFound");
            
            if (!aIssues || aIssues.length === 0) {
                MessageBox.information("No issues found to escalate.");
                return;
            }
            
            MessageBox.confirm(
                "Escalate " + aIssues.length + " issues for immediate attention?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._escalateIssues(sTaskId, aIssues);
                        }
                    }.bind(this)
                }
            );
        },

        _escalateIssues: function(sTaskId, aIssues) {
            if (aIssues.length > 100) {
                MessageBox.error("Too many issues to escalate at once (max 100)");
                return;
            }
            
            const sanitizedIssues = this._sanitizeArray(aIssues);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/escalate",
                type: "POST",
                data: JSON.stringify({
                    issues: sanitizedIssues,
                    priority: "HIGH",
                    notifyStakeholders: true
                })
            }).then(result => {
                const data = result.data;
                MessageBox.success(
                    "Issues escalated successfully!\\n" +
                    "Escalation ID: " + this._sanitizeInput(data.escalationId) + "\\n" +
                    "Stakeholders notified: " + this._sanitizeInput(data.notifiedCount)
                );
                
                this._auditLogger.log("ISSUES_ESCALATED", {
                    taskId: sTaskId,
                    issueCount: sanitizedIssues.length,
                    escalationId: data.escalationId
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to escalate issues: " + errorMsg);
                this._auditLogger.log("ISSUES_ESCALATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onViewQualityMetrics: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            
            if (!this._validateInput(sTaskId, 'taskId')) {
                MessageBox.error("Invalid task ID format");
                return;
            }
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks/" + encodeURL(sTaskId) + "/metrics",
                type: "GET"
            }).then(result => {
                this._showQualityMetrics(result.data);
                
                this._auditLogger.log("QUALITY_METRICS_VIEWED", { taskId: sTaskId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load quality metrics: " + errorMsg);
                this._auditLogger.log("QUALITY_METRICS_LOAD_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showQualityMetrics: function(metricsData) {
            var sMessage = "Quality Metrics Summary:\\n\\n";
            
            const sanitizedData = this._sanitizeObject(metricsData);
            Object.keys(sanitizedData).forEach(function(metric) {
                var value = sanitizedData[metric];
                sMessage += this._sanitizeInput(metric) + ": " + this._sanitizeInput(value.toString()) + "\\n";
            }.bind(this));
            
            MessageBox.information(sMessage);
        },
        
        _sanitizeObject: function(obj) {
            if (!obj || typeof obj !== 'object') return {};
            const sanitized = {};
            Object.keys(obj).forEach(key => {
                if (typeof obj[key] === 'string') {
                    sanitized[key] = this._sanitizeInput(obj[key]);
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === 'object') {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        },
        
        _sanitizeArray: function(arr) {
            if (!Array.isArray(arr)) return [];
            return arr.map(item => {
                if (typeof item === 'string') {
                    return this._sanitizeInput(item);
                } else if (typeof item === 'object') {
                    return this._sanitizeObject(item);
                } else {
                    return item;
                }
            });
        },
        
        // Create Quality Task Dialog Methods
        onCreateQualityTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.CreateQualityTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.setModel(this._oCreateModel, "create");
                    this._oCreateDialog.open();
                    
                    this._auditLogger.log("CREATE_QUALITY_DIALOG_OPENED", { action: "create_quality_task" });
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },
        
        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },
        
        onConfirmCreateTask: function() {
            var oData = this._oCreateModel.getData();
            
            // Validate form
            if (!this._validateForm()) {
                MessageBox.error("Please correct the validation errors before creating the task.");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            var oSanitizedData = {
                taskName: this._sanitizeInput(oData.taskName),
                description: this._sanitizeInput(oData.description),
                qualityGate: this._sanitizeInput(oData.qualityGate),
                dataSource: this._sanitizeInput(oData.dataSource),
                processingPipeline: this._sanitizeInput(oData.processingPipeline),
                evaluationCriteria: oData.evaluationCriteria,
                thresholds: oData.thresholds,
                routingStrategy: oData.routingStrategy,
                autoRouteThreshold: oData.autoRouteThreshold,
                enableFallback: oData.enableFallback,
                requireManualApproval: oData.requireManualApproval,
                notifyOnRouting: oData.notifyOnRouting,
                weights: oData.weights,
                trustLevel: oData.trustLevel,
                verification: oData.verification,
                trustFactors: oData.trustFactors
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks",
                type: "POST",
                data: JSON.stringify(oSanitizedData)
            }).then(result => {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show("Quality control task created successfully");
                this._extensionAPI.refresh();
                
                this._auditLogger.log("QUALITY_TASK_CREATED", { taskName: oSanitizedData.taskName });
            }).catch(error => {
                this._oCreateDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to create quality task: " + errorMsg);
                this._auditLogger.log("QUALITY_TASK_CREATE_FAILED", { error: errorMsg });
            });
        },
        
        // Validation Event Handlers
        onTaskNameChange: function() {
            var sValue = this._oCreateModel.getProperty("/taskName");
            var oValidation = this._validateTaskName(sValue);
            
            this._oCreateModel.setProperty("/taskNameState", oValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oValidation.message);
        },
        
        onQualityGateChange: function() {
            var sValue = this._oCreateModel.getProperty("/qualityGate");
            var oValidation = this._validateQualityGate(sValue);
            
            this._oCreateModel.setProperty("/qualityGateState", oValidation.state);
            this._oCreateModel.setProperty("/qualityGateStateText", oValidation.message);
        },
        
        onDataSourceChange: function() {
            var sValue = this._oCreateModel.getProperty("/dataSource");
            var oValidation = this._validateDataSource(sValue);
            
            this._oCreateModel.setProperty("/dataSourceState", oValidation.state);
            this._oCreateModel.setProperty("/dataSourceStateText", oValidation.message);
        },
        
        onRoutingStrategyChange: function() {
            // Update auto-suggestion for routing options
            var sStrategy = this._oCreateModel.getProperty("/routingStrategy");
            if (sStrategy === "QUALITY_BASED") {
                this._oCreateModel.setProperty("/autoRouteThreshold", 85);
            } else if (sStrategy === "LOAD_BALANCED") {
                this._oCreateModel.setProperty("/autoRouteThreshold", 70);
            }
        },
        
        onTrustLevelChange: function() {
            // Update verification requirements based on trust level
            var sLevel = this._oCreateModel.getProperty("/trustLevel");
            var oVerification = this._oCreateModel.getProperty("/verification");
            
            switch (sLevel) {
                case "BASIC":
                    oVerification.reputation = true;
                    oVerification.integrity = false;
                    oVerification.blockchain = false;
                    break;
                case "STANDARD":
                    oVerification.reputation = true;
                    oVerification.integrity = true;
                    oVerification.blockchain = false;
                    break;
                case "ENHANCED":
                    oVerification.reputation = true;
                    oVerification.integrity = true;
                    oVerification.blockchain = true;
                    oVerification.signature = true;
                    break;
                case "MAXIMUM":
                    oVerification.reputation = true;
                    oVerification.integrity = true;
                    oVerification.blockchain = true;
                    oVerification.consensus = true;
                    oVerification.signature = true;
                    break;
            }
            
            this._oCreateModel.setProperty("/verification", oVerification);
        },
        
        // Validation Methods
        _validateTaskName: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Task name is required" };
            }
            if (sValue.length < 3) {
                return { state: "Warning", message: "Task name should be at least 3 characters" };
            }
            if (sValue.length > 100) {
                return { state: "Error", message: "Task name must not exceed 100 characters" };
            }
            if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sValue)) {
                return { state: "Error", message: "Task name contains invalid characters" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateQualityGate: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Quality gate is required" };
            }
            var aValidGates = ["COMPLIANCE", "PERFORMANCE", "SECURITY", "RELIABILITY", "USABILITY", "MAINTAINABILITY"];
            if (!aValidGates.includes(sValue.toUpperCase())) {
                return { state: "Warning", message: "Please select a valid quality gate" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateDataSource: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Information", message: "Data source is optional but recommended" };
            }
            if (sValue.length > 500) {
                return { state: "Error", message: "Data source path is too long" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateForm: function() {
            var oData = this._oCreateModel.getData();
            var bValid = true;
            
            // Validate task name
            var oTaskNameValidation = this._validateTaskName(oData.taskName);
            this._oCreateModel.setProperty("/taskNameState", oTaskNameValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oTaskNameValidation.message);
            if (oTaskNameValidation.state === "Error") bValid = false;
            
            // Validate quality gate
            var oQualityGateValidation = this._validateQualityGate(oData.qualityGate);
            this._oCreateModel.setProperty("/qualityGateState", oQualityGateValidation.state);
            this._oCreateModel.setProperty("/qualityGateStateText", oQualityGateValidation.message);
            if (oQualityGateValidation.state === "Error") bValid = false;
            
            return bValid;
        }
    });
});