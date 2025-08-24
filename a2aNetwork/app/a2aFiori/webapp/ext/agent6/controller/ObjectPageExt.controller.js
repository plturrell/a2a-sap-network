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
    "sap/base/security/sanitizeHTML",
    "../utils/SecurityUtils"
], (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, encodeURL, Log, Router, escapeRegExp, sanitizeHTML, SecurityUtils) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent6.ext.controller.ObjectPageExt", {

        override: {
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._resourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                this._initializeSecurity();

                // Initialize device model for responsive behavior
                const oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");

                // Initialize create dialog model
                this._initializeCreateModel();
            },

            onExit() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        _initializeSecurity() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const logEntry = {
                        timestamp,
                        user,
                        agent: "Agent6_QualityControl",
                        action,
                        details
                    };
                    Log.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
                }.bind(this)
            };
        },

        _getCurrentUser() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        _hasRole(role) {
            const user = sap.ushell?.Container?.getUser();
            return user && user.hasRole && user.hasRole(role);
        },

        /**
         * @function _getQualityThresholds
         * @description Gets quality thresholds from configuration
         * @returns {Object} Quality thresholds configuration
         * @private
         */
        _getQualityThresholds() {
            // Get from model configuration or default to secure values
            const oConfigModel = this.base.getView().getModel("config");
            if (oConfigModel) {
                const thresholds = oConfigModel.getProperty("/qualityThresholds");
                if (thresholds && this._securityUtils.validateQualityThreshold(thresholds)) {
                    return thresholds;
                }
            }

            // Return secure defaults if no valid configuration
            return {
                minQualityScore: 80,
                maxIssues: 5,
                minTrustScore: 85,
                maxDefects: 10,
                maxWarnings: 20
            };
        },

        _validateInput(input, type) {
            if (!input || typeof input !== "string") {return false;}

            switch (type) {
            case "taskId":
                return /^[a-zA-Z0-9\-_]{1,36}$/.test(input);
            case "reason":
                return input.length <= 500;
            case "agent":
                return /^[a-zA-Z0-9_]{1,50}$/.test(input);
            case "url":
                return /^https:\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(input);
            default:
                return input.length > 0 && input.length <= 255;
            }
        },

        _sanitizeInput(input) {
            if (!input) {return "";}
            return encodeXML(input.toString().trim());
        },

        _validateEventSourceUrl(url) {
            return this._validateInput(url, "url") && url.startsWith("/a2a/agent6/v1/");
        },

        _validateDownloadUrl(url) {
            return this._validateInput(url, "url") &&
                   url.startsWith("/a2a/agent6/v1/") &&
                   !url.includes("..");
        },

        _sanitizeFilename(filename) {
            if (!filename) {return "download";}
            return filename.replace(/[^a-zA-Z0-9._-]/g, "_").substring(0, 100);
        },

        _getCsrfToken() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/a2a/agent6/v1/csrf-token",
                    type: "GET",
                    headers: {
                        "X-CSRF-Token": "Fetch"
                    },
                    success(data, textStatus, xhr) {
                        const token = xhr.getResponseHeader("X-CSRF-Token");
                        resolve(token);
                    },
                    error(xhr) {
                        reject(new Error("Failed to fetch CSRF token"));
                    }
                });
            });
        },

        _secureAjaxCall(options) {
            return this._getCsrfToken().then(token => {
                return new Promise((resolve, reject) => {
                    const secureOptions = Object.assign({}, options, {
                        headers: Object.assign({
                            "X-CSRF-Token": token,
                            "Content-Type": "application/json"
                        }, options.headers || {}),
                        success(data, textStatus, xhr) {
                            resolve({ data, textStatus, xhr });
                        },
                        error(xhr, textStatus, errorThrown) {
                            reject({ xhr, textStatus, errorThrown });
                        }
                    });

                    jQuery.ajax(secureOptions);
                });
            });
        },

        _cleanupResources() {
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

        _initializeCreateModel() {
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
                thresholds: this._getQualityThresholds(),
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

        onStartAssessment() {
            if (!this._hasRole("QualityManager")) {
                MessageBox.error("Access denied: Insufficient privileges for starting assessments");
                this._auditLogger.log("ASSESSMENT_ACCESS_DENIED", { action: "start_assessment" });
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            const sTaskName = this._sanitizeInput(oContext.getProperty("taskName"));

            if (!this._validateInput(sTaskId, "taskId")) {
                MessageBox.error("Invalid task ID format");
                return;
            }

            MessageBox.confirm(this._resourceBundle.getText("msg.confirmStartAssessment", [sTaskName]), {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startQualityAssessment(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startQualityAssessment(sTaskId) {
            this._extensionAPI.getView().setBusy(true);

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/assess`,
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
                MessageBox.error(this._resourceBundle.getText("error.startAssessmentFailed", [errorMsg]));
                this._auditLogger.log("ASSESSMENT_START_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _startRealtimeMonitoring(sTaskId) {
            const streamUrl = `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/stream`;

            if (!this._validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid monitoring stream URL");
                return;
            }

            this._eventSource = new EventSource(streamUrl);

            this._eventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);

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
                        MessageBox.error(`Assessment failed: ${ errorMsg}`);
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

        _updateAssessmentProgress(data) {
            MessageToast.show(this._resourceBundle.getText("msg.assessmentProgress", [this._securityUtils.escapeHTML(data.component), data.progress]));
        },

        _showAssessmentResults(results) {
            const oView = this.base.getView();

            if (!this._oResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.AssessmentResults",
                    controller: this
                }).then((oDialog) => {
                    this._oResultsDialog = oDialog;
                    oView.addDependent(this._oResultsDialog);

                    const oModel = new JSONModel(results);
                    this._oResultsDialog.setModel(oModel, "results");
                    this._oResultsDialog.open();

                    this._createResultsVisualizations(results);
                });
            } else {
                const oModel = new JSONModel(results);
                this._oResultsDialog.setModel(oModel, "results");
                this._oResultsDialog.open();
                this._createResultsVisualizations(results);
            }
        },

        _createResultsVisualizations(results) {
            this._createQualityScoreRadar(results.scores);
            this._createIssueBreakdown(results.issues);
        },

        _createQualityScoreRadar(scores) {
            const oRadarChart = this._oResultsDialog.byId("qualityRadarChart");
            if (!oRadarChart || !scores) {return;}

            const aRadarData = Object.keys(scores).map((key) => {
                return {
                    Component: key,
                    Score: scores[key]
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                radarData: aRadarData
            });
            oRadarChart.setModel(oChartModel);
        },

        onMakeRoutingDecision() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const oQualityScore = oContext.getProperty("overallQuality");

            if (!this._oRoutingDecisionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.RoutingDecisionDialog",
                    controller: this
                }).then((oDialog) => {
                    this._oRoutingDecisionDialog = oDialog;
                    this.base.getView().addDependent(this._oRoutingDecisionDialog);

                    const oModel = new JSONModel({
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
                });
            } else {
                this._oRoutingDecisionDialog.open();
                this._loadRoutingOptions(sTaskId);
            }
        },

        _loadRoutingOptions(sTaskId) {
            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/routing-options`,
                type: "GET"
            }).then(result => {
                const oModel = this._oRoutingDecisionDialog.getModel("routing");
                const oData = oModel.getData();

                oData.availableAgents = this._sanitizeArray(result.data.agents);
                oData.recommendations = this._sanitizeArray(result.data.recommendations);
                oData.routingHistory = this._sanitizeArray(result.data.history);

                oModel.setData(oData);

                this._auditLogger.log("ROUTING_OPTIONS_LOADED", { taskId: sTaskId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Failed to load routing options: ${ errorMsg}`);
                this._auditLogger.log("ROUTING_OPTIONS_LOAD_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onConfirmRoutingDecision() {
            const oModel = this._oRoutingDecisionDialog.getModel("routing");
            const oData = oModel.getData();

            if (!this._validateInput(oData.decision)) {
                MessageBox.error("Please select a valid decision");
                return;
            }

            if (!this._validateInput(oData.targetAgent, "agent")) {
                MessageBox.error("Please select a valid target agent");
                return;
            }

            if (!this._validateInput(oData.reason, "reason")) {
                MessageBox.error("Invalid reason format (max 500 characters)");
                return;
            }

            const sanitizedData = {
                decision: this._sanitizeInput(oData.decision),
                targetAgent: this._sanitizeInput(oData.targetAgent),
                confidence: Math.max(0, Math.min(100, parseInt(oData.confidence, 10) || 0)),
                reason: this._sanitizeInput(oData.reason),
                priority: this._sanitizeInput(oData.priority)
            };

            this._oRoutingDecisionDialog.setBusy(true);

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(oData.taskId) }/route`,
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oRoutingDecisionDialog.setBusy(false);
                this._oRoutingDecisionDialog.close();

                const data = result.data;
                MessageBox.success(
                    "Routing decision made successfully!\\n" +
                    `Task routed to: ${ this._sanitizeInput(sanitizedData.targetAgent) }\\n` +
                    `Estimated processing time: ${ this._sanitizeInput(data.estimatedTime) } minutes`
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
                MessageBox.error(`Failed to make routing decision: ${ errorMsg}`);
                this._auditLogger.log("ROUTING_DECISION_FAILED", { taskId: oData.taskId, error: errorMsg });
            });
        },

        onVerifyTrust() {
            if (!this._hasRole("TrustVerifier")) {
                MessageBox.error("Access denied: Insufficient privileges for trust verification");
                this._auditLogger.log("TRUST_VERIFY_ACCESS_DENIED", { action: "verify_trust" });
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = this._sanitizeInput(oContext.getProperty("ID"));

            if (!this._validateInput(sTaskId, "taskId")) {
                MessageBox.error("Invalid task ID format");
                return;
            }

            this._extensionAPI.getView().setBusy(true);

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/verify-trust`,
                type: "POST"
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                this._showTrustVerificationResults(result.data);

                this._auditLogger.log("TRUST_VERIFICATION_COMPLETED", { taskId: sTaskId });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Trust verification failed: ${ errorMsg}`);
                this._auditLogger.log("TRUST_VERIFICATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showTrustVerificationResults(verificationData) {
            const oView = this.base.getView();

            if (!this._oTrustResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.TrustVerificationResults",
                    controller: this
                }).then((oDialog) => {
                    this._oTrustResultsDialog = oDialog;
                    oView.addDependent(this._oTrustResultsDialog);

                    const oModel = new JSONModel(verificationData);
                    this._oTrustResultsDialog.setModel(oModel, "trust");
                    this._oTrustResultsDialog.open();

                    this._createTrustVisualizations(verificationData);
                });
            } else {
                const oModel = new JSONModel(verificationData);
                this._oTrustResultsDialog.setModel(oModel, "trust");
                this._oTrustResultsDialog.open();
                this._createTrustVisualizations(verificationData);
            }
        },

        _createTrustVisualizations(data) {
            const oTrustChart = this._oTrustResultsDialog.byId("trustFactorsChart");
            if (!oTrustChart || !data.factors) {return;}

            const aChartData = data.factors.map((factor) => {
                return {
                    Factor: factor.name,
                    Score: factor.score,
                    Weight: factor.weight
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                factorData: aChartData
            });
            oTrustChart.setModel(oChartModel);
        },

        onGenerateQualityReport() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.QualityReport",
                    controller: this
                }).then((oDialog) => {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);

                    const oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "COMPREHENSIVE",
                        includeCharts: true,
                        includeRecommendations: true,
                        format: "PDF"
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                });
            } else {
                this._oReportDialog.open();
            }
        },

        onExecuteReportGeneration() {
            const oModel = this._oReportDialog.getModel("report");
            const oData = oModel.getData();

            if (!this._validateInput(oData.taskId, "taskId")) {
                MessageBox.error("Invalid task ID format");
                return;
            }

            const allowedFormats = ["PDF", "EXCEL", "JSON"];
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
                url: `/a2a/agent6/v1/tasks/${ encodeURL(oData.taskId) }/report`,
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
                    `Quality report generated successfully!\\n\\nDownload: ${ sanitizedFilename}`,
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
                MessageBox.error(`Report generation failed: ${ errorMsg}`);
                this._auditLogger.log("REPORT_GENERATION_FAILED", { taskId: oData.taskId, error: errorMsg });
            });
        },

        onOptimizeWorkflow() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

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

        _optimizeWorkflow(sTaskId) {
            const requestData = {
                analysisDepth: "COMPREHENSIVE",
                includeResourceOptimization: true,
                applyOptimizations: false
            };

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/optimize`,
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
                MessageBox.error(`Workflow optimization failed: ${ errorMsg}`);
                this._auditLogger.log("WORKFLOW_OPTIMIZATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showOptimizationResults(optimizationData) {
            let sMessage = "Workflow Analysis Results:\\n\\n";

            if (optimizationData.bottlenecks && optimizationData.bottlenecks.length > 0) {
                sMessage += "Bottlenecks Found:\\n";
                optimizationData.bottlenecks.forEach((bottleneck) => {
                    sMessage += `• ${ bottleneck.stage }: ${ bottleneck.impact }\\n`;
                });
                sMessage += "\\n";
            }

            if (optimizationData.recommendations && optimizationData.recommendations.length > 0) {
                sMessage += "Optimization Recommendations:\\n";
                optimizationData.recommendations.forEach((rec) => {
                    sMessage += `• ${ rec.description } (Impact: ${ rec.expectedImprovement })\\n`;
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

        _applyOptimizations(optimizations) {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = this._sanitizeInput(oContext.getProperty("ID"));

            if (!this._validateInput(sTaskId, "taskId")) {
                MessageBox.error("Invalid task ID format");
                return;
            }

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/apply-optimizations`,
                type: "POST",
                data: JSON.stringify({ optimizations })
            }).then(result => {
                MessageBox.success("Workflow optimizations applied successfully!");
                this._extensionAPI.refresh();

                this._auditLogger.log("WORKFLOW_OPTIMIZATIONS_APPLIED", {
                    taskId: sTaskId,
                    optimizationCount: optimizations?.length || 0
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Failed to apply optimizations: ${ errorMsg}`);
                this._auditLogger.log("WORKFLOW_OPTIMIZATIONS_APPLY_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onEscalateIssues() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const aIssues = oContext.getProperty("issuesFound");

            if (!aIssues || aIssues.length === 0) {
                MessageBox.information("No issues found to escalate.");
                return;
            }

            MessageBox.confirm(
                `Escalate ${ aIssues.length } issues for immediate attention?`,
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._escalateIssues(sTaskId, aIssues);
                        }
                    }.bind(this)
                }
            );
        },

        _escalateIssues(sTaskId, aIssues) {
            if (aIssues.length > 100) {
                MessageBox.error("Too many issues to escalate at once (max 100)");
                return;
            }

            const sanitizedIssues = this._sanitizeArray(aIssues);

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/escalate`,
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
                    `Escalation ID: ${ this._sanitizeInput(data.escalationId) }\\n` +
                    `Stakeholders notified: ${ this._sanitizeInput(data.notifiedCount)}`
                );

                this._auditLogger.log("ISSUES_ESCALATED", {
                    taskId: sTaskId,
                    issueCount: sanitizedIssues.length,
                    escalationId: data.escalationId
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Failed to escalate issues: ${ errorMsg}`);
                this._auditLogger.log("ISSUES_ESCALATION_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        onViewQualityMetrics() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = this._sanitizeInput(oContext.getProperty("ID"));

            if (!this._validateInput(sTaskId, "taskId")) {
                MessageBox.error("Invalid task ID format");
                return;
            }

            this._secureAjaxCall({
                url: `/a2a/agent6/v1/tasks/${ encodeURL(sTaskId) }/metrics`,
                type: "GET"
            }).then(result => {
                this._showQualityMetrics(result.data);

                this._auditLogger.log("QUALITY_METRICS_VIEWED", { taskId: sTaskId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Failed to load quality metrics: ${ errorMsg}`);
                this._auditLogger.log("QUALITY_METRICS_LOAD_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        _showQualityMetrics(metricsData) {
            let sMessage = "Quality Metrics Summary:\\n\\n";

            const sanitizedData = this._sanitizeObject(metricsData);
            Object.keys(sanitizedData).forEach((metric) => {
                const value = sanitizedData[metric];
                sMessage += `${this._sanitizeInput(metric) }: ${ this._sanitizeInput(value.toString()) }\\n`;
            });

            MessageBox.information(sMessage);
        },

        _sanitizeObject(obj) {
            if (!obj || typeof obj !== "object") {return {};}
            const sanitized = {};
            Object.keys(obj).forEach(key => {
                if (typeof obj[key] === "string") {
                    sanitized[key] = this._sanitizeInput(obj[key]);
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === "object") {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        },

        _sanitizeArray(arr) {
            if (!Array.isArray(arr)) {return [];}
            return arr.map(item => {
                if (typeof item === "string") {
                    return this._sanitizeInput(item);
                } else if (typeof item === "object") {
                    return this._sanitizeObject(item);
                }
                return item;

            });
        },

        // Create Quality Task Dialog Methods
        onCreateQualityTask() {
            const oView = this.base.getView();

            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.CreateQualityTask",
                    controller: this
                }).then((oDialog) => {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.setModel(this._oCreateModel, "create");
                    this._oCreateDialog.open();

                    this._auditLogger.log("CREATE_QUALITY_DIALOG_OPENED", { action: "create_quality_task" });
                });
            } else {
                this._oCreateDialog.open();
            }
        },

        onCancelCreateTask() {
            this._oCreateDialog.close();
        },

        onConfirmCreateTask() {
            const oData = this._oCreateModel.getData();

            // Validate form
            if (!this._validateForm()) {
                MessageBox.error("Please correct the validation errors before creating the task.");
                return;
            }

            this._oCreateDialog.setBusy(true);

            const oSanitizedData = {
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
                MessageBox.error(`Failed to create quality task: ${ errorMsg}`);
                this._auditLogger.log("QUALITY_TASK_CREATE_FAILED", { error: errorMsg });
            });
        },

        // Validation Event Handlers
        onTaskNameChange() {
            const sValue = this._oCreateModel.getProperty("/taskName");
            const oValidation = this._validateTaskName(sValue);

            this._oCreateModel.setProperty("/taskNameState", oValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oValidation.message);
        },

        onQualityGateChange() {
            const sValue = this._oCreateModel.getProperty("/qualityGate");
            const oValidation = this._validateQualityGate(sValue);

            this._oCreateModel.setProperty("/qualityGateState", oValidation.state);
            this._oCreateModel.setProperty("/qualityGateStateText", oValidation.message);
        },

        onDataSourceChange() {
            const sValue = this._oCreateModel.getProperty("/dataSource");
            const oValidation = this._validateDataSource(sValue);

            this._oCreateModel.setProperty("/dataSourceState", oValidation.state);
            this._oCreateModel.setProperty("/dataSourceStateText", oValidation.message);
        },

        onRoutingStrategyChange() {
            // Update auto-suggestion for routing options
            const sStrategy = this._oCreateModel.getProperty("/routingStrategy");
            if (sStrategy === "QUALITY_BASED") {
                this._oCreateModel.setProperty("/autoRouteThreshold", 85);
            } else if (sStrategy === "LOAD_BALANCED") {
                this._oCreateModel.setProperty("/autoRouteThreshold", 70);
            }
        },

        onTrustLevelChange() {
            // Update verification requirements based on trust level
            const sLevel = this._oCreateModel.getProperty("/trustLevel");
            const oVerification = this._oCreateModel.getProperty("/verification");

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
        _validateTaskName(sValue) {
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

        _validateQualityGate(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Quality gate is required" };
            }
            const aValidGates = ["COMPLIANCE", "PERFORMANCE", "SECURITY", "RELIABILITY", "USABILITY", "MAINTAINABILITY"];
            if (!aValidGates.includes(sValue.toUpperCase())) {
                return { state: "Warning", message: "Please select a valid quality gate" };
            }
            return { state: "Success", message: "" };
        },

        _validateDataSource(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Information", message: "Data source is optional but recommended" };
            }
            if (sValue.length > 500) {
                return { state: "Error", message: "Data source path is too long" };
            }
            return { state: "Success", message: "" };
        },

        _validateForm() {
            const oData = this._oCreateModel.getData();
            let bValid = true;

            // Validate task name
            const oTaskNameValidation = this._validateTaskName(oData.taskName);
            this._oCreateModel.setProperty("/taskNameState", oTaskNameValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oTaskNameValidation.message);
            if (oTaskNameValidation.state === "Error") {bValid = false;}

            // Validate quality gate
            const oQualityGateValidation = this._validateQualityGate(oData.qualityGate);
            this._oCreateModel.setProperty("/qualityGateState", oQualityGateValidation.state);
            this._oCreateModel.setProperty("/qualityGateStateText", oQualityGateValidation.message);
            if (oQualityGateValidation.state === "Error") {bValid = false;}

            return bValid;
        },

        /**
         * @function onStartQualityCheck
         * @description Starts quality check for the current task with confirmation.
         * @public
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onStartQualityCheck() {
            const oContext = this._extensionAPI.getBindingContext();
            if (!oContext) {
                MessageBox.error("No quality task selected");
                return;
            }

            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            MessageBox.confirm(`Start quality check for '${ this._securityUtils.escapeHTML(sTaskName) }'?`, {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startQualityCheckProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        /**
         * @function onPauseQualityCheck
         * @description Pauses the currently running quality check.
         * @public
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onPauseQualityCheck() {
            const oContext = this._extensionAPI.getBindingContext();
            if (!oContext) {
                MessageBox.error("No quality task selected");
                return;
            }

            const sTaskId = oContext.getProperty("ID");

            MessageBox.confirm("Pause the quality check process?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._pauseQualityCheck(sTaskId);
                    }
                }.bind(this)
            });
        },

        /**
         * @function onViewResults
         * @description Views quality check results with detailed analysis.
         * @public
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onViewResults() {
            const oContext = this._extensionAPI.getBindingContext();
            if (!oContext) {
                MessageBox.error("No quality task selected");
                return;
            }

            const sTaskId = oContext.getProperty("ID");
            this._showQualityResults(sTaskId);
        },

        /**
         * @function onExportReport
         * @description Exports quality check report in various formats.
         * @public
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onExportReport() {
            const oContext = this._extensionAPI.getBindingContext();
            if (!oContext) {
                MessageBox.error("No quality task selected");
                return;
            }

            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            this._showExportDialog(sTaskId, sTaskName);
        },

        /**
         * @function _startQualityCheckProcess
         * @description Starts the quality check process for given task ID.
         * @param {string} sTaskId - Task ID to start quality check for
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _startQualityCheckProcess(sTaskId) {
            this._extensionAPI.getView().setBusy(true);

            this._securityUtils.secureAjaxRequest({
                url: `/a2a/agent6/v1/tasks/${ encodeURIComponent(sTaskId) }/start`,
                type: "POST",
                contentType: "application/json",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Quality check process started");
                    this._extensionAPI.refresh();

                    // Start monitoring progress
                    this._startProgressMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error(`Failed to start quality check: ${ this._securityUtils.escapeHTML(xhr.responseText || "Unknown error")}`);
                }.bind(this)
            });
        },

        /**
         * @function _pauseQualityCheck
         * @description Pauses the quality check process for given task ID.
         * @param {string} sTaskId - Task ID to pause quality check for
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _pauseQualityCheck(sTaskId) {
            this._securityUtils.secureAjaxRequest({
                url: `/a2a/agent6/v1/tasks/${ encodeURIComponent(sTaskId) }/pause`,
                type: "POST",
                success: function() {
                    MessageToast.show("Quality check process paused");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error(`Failed to pause quality check: ${ this._securityUtils.escapeHTML(xhr.responseText || "Unknown error")}`);
                }.bind(this)
            });
        },

        /**
         * @function _showQualityResults
         * @description Shows detailed quality check results in a dialog.
         * @param {string} sTaskId - Task ID to show results for
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _showQualityResults(sTaskId) {
            const oView = this.base.getView();

            if (!this._oResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.QualityResults",
                    controller: this
                }).then((oDialog) => {
                    this._oResultsDialog = oDialog;
                    oView.addDependent(this._oResultsDialog);
                    this._loadQualityResults(sTaskId);
                    this._oResultsDialog.open();
                });
            } else {
                this._loadQualityResults(sTaskId);
                this._oResultsDialog.open();
            }
        },

        /**
         * @function _showExportDialog
         * @description Shows export options dialog for quality reports.
         * @param {string} sTaskId - Task ID to export
         * @param {string} sTaskName - Task name for display
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _showExportDialog(sTaskId, sTaskName) {
            const oView = this.base.getView();

            if (!this._oExportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.ExportQualityReport",
                    controller: this
                }).then((oDialog) => {
                    this._oExportDialog = oDialog;
                    oView.addDependent(this._oExportDialog);

                    const oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: this._securityUtils.escapeHTML(sTaskName),
                        format: "PDF",
                        includeCharts: true,
                        includeDetails: true,
                        includeRecommendations: true
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                });
            } else {
                const oModel = new JSONModel({
                    taskId: sTaskId,
                    taskName: this._securityUtils.escapeHTML(sTaskName),
                    format: "PDF",
                    includeCharts: true,
                    includeDetails: true,
                    includeRecommendations: true
                });
                this._oExportDialog.setModel(oModel, "export");
                this._oExportDialog.open();
            }
        },

        /**
         * @function _loadQualityResults
         * @description Loads quality check results from backend.
         * @param {string} sTaskId - Task ID to load results for
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _loadQualityResults(sTaskId) {
            this._securityUtils.secureAjaxRequest({
                url: `/a2a/agent6/v1/tasks/${ encodeURIComponent(sTaskId) }/results`,
                type: "GET",
                success: function(data) {
                    const oModel = new JSONModel(data);
                    this._oResultsDialog.setModel(oModel, "results");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error(`Failed to load quality results: ${ this._securityUtils.escapeHTML(xhr.responseText || "Unknown error")}`);
                }.bind(this)
            });
        },

        /**
         * @function _startProgressMonitoring
         * @description Starts monitoring quality check progress with real-time updates.
         * @param {string} sTaskId - Task ID to monitor
         * @private
         * @memberof a2a.network.agent6.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        _startProgressMonitoring(sTaskId) {
            // Poll for progress updates every 2 seconds
            this._progressInterval = setInterval(() => {
                this._securityUtils.secureAjaxRequest({
                    url: `/a2a/agent6/v1/tasks/${ encodeURIComponent(sTaskId) }/progress`,
                    type: "GET",
                    success: function(data) {
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(this._progressInterval);
                            this._extensionAPI.refresh();

                            if (data.status === "COMPLETED") {
                                MessageBox.success("Quality check completed successfully!");
                            } else {
                                MessageBox.error(`Quality check failed: ${ this._securityUtils.escapeHTML(data.error || "Unknown error")}`);
                            }
                        }
                    }.bind(this),
                    error: function() {
                        clearInterval(this._progressInterval);
                    }.bind(this)
                });
            }, 2000);
        }
    });
});