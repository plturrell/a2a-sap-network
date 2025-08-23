sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent9/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent9.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeCreateModel();
            }
        },

        _initializeCreateModel: function() {
            var oCreateData = {
                taskName: "",
                description: "",
                reasoningType: "",
                problemDomain: "",
                priority: "MEDIUM",
                taskNameState: "",
                taskNameStateText: "",
                reasoningTypeState: "",
                reasoningTypeStateText: "",
                problemDomainState: "",
                problemDomainStateText: "",
                reasoningEngine: "",
                confidenceThreshold: 0.8,
                maxInferenceDepth: 10,
                chainingStrategy: "BREADTH_FIRST",
                uncertaintyHandling: "PROBABILISTIC",
                parallelReasoning: false,
                cacheResults: true,
                generateExplanations: true,
                validateConsistency: true,
                problemDomains: [],
                availableEngines: []
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.base.getView().setModel(oCreateModel, "create");
        },

        onCreateReasoningTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.CreateReasoningTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._loadCreateDialogData();
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._loadCreateDialogData();
                this._oCreateDialog.open();
            }
        },

        _loadCreateDialogData: function() {
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Load problem domains
            oData.problemDomains = [
                { domainId: "GENERAL", domainName: "General Problem Solving", complexity: "Low" },
                { domainId: "BUSINESS", domainName: "Business Logic", complexity: "Medium" },
                { domainId: "TECHNICAL", domainName: "Technical Analysis", complexity: "High" },
                { domainId: "SCIENTIFIC", domainName: "Scientific Research", complexity: "Very High" }
            ];
            
            // Load available engines
            oData.availableEngines = [
                { engineId: "FORWARD", engineName: "Forward Chaining", engineType: "Rule-based" },
                { engineId: "BACKWARD", engineName: "Backward Chaining", engineType: "Goal-driven" },
                { engineId: "BAYESIAN", engineName: "Bayesian Network", engineType: "Probabilistic" },
                { engineId: "FUZZY", engineName: "Fuzzy Logic", engineType: "Approximate" }
            ];
            
            oCreateModel.setData(oData);
        },

        onTaskNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!sValue || sValue.length < 3) {
                oData.taskNameState = "Error";
                oData.taskNameStateText = "Task name must be at least 3 characters";
            } else if (sValue.length > 100) {
                oData.taskNameState = "Error";
                oData.taskNameStateText = "Task name must not exceed 100 characters";
            } else {
                oData.taskNameState = "Success";
                oData.taskNameStateText = "Valid task name";
            }
            
            oCreateModel.setData(oData);
        },

        onReasoningTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.reasoningTypeState = "Success";
                oData.reasoningTypeStateText = "Reasoning type selected";
                
                // Smart suggestions based on reasoning type
                switch (sValue) {
                    case "DEDUCTIVE":
                        oData.reasoningEngine = "FORWARD";
                        oData.chainingStrategy = "BREADTH_FIRST";
                        oData.uncertaintyHandling = "CRISP";
                        break;
                    case "INDUCTIVE":
                        oData.reasoningEngine = "BAYESIAN";
                        oData.chainingStrategy = "BEST_FIRST";
                        oData.uncertaintyHandling = "PROBABILISTIC";
                        break;
                    case "PROBABILISTIC":
                        oData.reasoningEngine = "BAYESIAN";
                        oData.uncertaintyHandling = "PROBABILISTIC";
                        oData.confidenceThreshold = 0.7;
                        break;
                    case "FUZZY":
                        oData.reasoningEngine = "FUZZY";
                        oData.uncertaintyHandling = "FUZZY";
                        break;
                }
            } else {
                oData.reasoningTypeState = "Error";
                oData.reasoningTypeStateText = "Please select a reasoning type";
            }
            
            oCreateModel.setData(oData);
        },

        onProblemDomainChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.problemDomainState = "Success";
                oData.problemDomainStateText = "Problem domain selected";
                
                // Adjust settings based on domain complexity
                switch (sValue) {
                    case "GENERAL":
                        oData.maxInferenceDepth = 5;
                        oData.confidenceThreshold = 0.9;
                        break;
                    case "BUSINESS":
                        oData.maxInferenceDepth = 10;
                        oData.confidenceThreshold = 0.8;
                        break;
                    case "TECHNICAL":
                        oData.maxInferenceDepth = 20;
                        oData.confidenceThreshold = 0.7;
                        oData.parallelReasoning = true;
                        break;
                    case "SCIENTIFIC":
                        oData.maxInferenceDepth = 50;
                        oData.confidenceThreshold = 0.6;
                        oData.parallelReasoning = true;
                        oData.generateExplanations = true;
                        break;
                }
            } else {
                oData.problemDomainState = "Error";
                oData.problemDomainStateText = "Please select a problem domain";
            }
            
            oCreateModel.setData(oData);
        },

        onCancelCreateReasoningTask: function() {
            this._oCreateDialog.close();
            this._resetCreateModel();
        },

        onConfirmCreateReasoningTask: function() {
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Final validation
            if (!this._validateCreateData(oData)) {
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Sanitize data for security
            var oSanitizedData = {
                taskName: this._securityUtils.sanitizeInput(oData.taskName),
                description: this._securityUtils.sanitizeInput(oData.description),
                reasoningType: oData.reasoningType,
                problemDomain: oData.problemDomain,
                priority: oData.priority,
                reasoningEngine: oData.reasoningEngine,
                confidenceThreshold: parseFloat(oData.confidenceThreshold) || 0.8,
                maxInferenceDepth: parseInt(oData.maxInferenceDepth) || 10,
                chainingStrategy: oData.chainingStrategy,
                uncertaintyHandling: oData.uncertaintyHandling,
                parallelReasoning: !!oData.parallelReasoning,
                cacheResults: !!oData.cacheResults,
                generateExplanations: !!oData.generateExplanations,
                validateConsistency: !!oData.validateConsistency
            };
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oSanitizedData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Reasoning task created successfully");
                    this._extensionAPI.refresh();
                    this._resetCreateModel();
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to create reasoning task: " + errorMsg);
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        _validateCreateData: function(oData) {
            if (!oData.taskName || oData.taskName.length < 3) {
                MessageBox.error("Task name is required and must be at least 3 characters");
                return false;
            }
            
            if (!oData.reasoningType) {
                MessageBox.error("Reasoning type is required");
                return false;
            }
            
            if (!oData.problemDomain) {
                MessageBox.error("Problem domain is required");
                return false;
            }
            
            return true;
        },

        _resetCreateModel: function() {
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.taskName = "";
            oData.description = "";
            oData.reasoningType = "";
            oData.problemDomain = "";
            oData.priority = "MEDIUM";
            oData.taskNameState = "";
            oData.taskNameStateText = "";
            oData.reasoningTypeState = "";
            oData.reasoningTypeStateText = "";
            oData.problemDomainState = "";
            oData.problemDomainStateText = "";
            
            oCreateModel.setData(oData);
        },


        onStartReasoning: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            const safeTaskName = this._securityUtils.encodeHTML(sTaskName);
            MessageBox.confirm("Start reasoning process for '" + safeTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startReasoningProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startReasoningProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/reason",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Reasoning process started");
                    this._extensionAPI.refresh();
                    
                    this._startReasoningMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to start reasoning: " + errorMsg);
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        _startReasoningMonitoring: function(sTaskId) {
            this._reasoningEventSource = new EventSource("/a2a/agent9/v1/tasks/" + sTaskId + "/stream");
            
            this._reasoningEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "progress") {
                    this._updateReasoningProgress(data);
                } else if (data.type === "inference") {
                    const safeConclusion = this._securityUtils.encodeHTML(data.conclusion || 'Unknown conclusion');
                    MessageToast.show("New inference: " + safeConclusion);
                } else if (data.type === "complete") {
                    this._reasoningEventSource.close();
                    this._extensionAPI.refresh();
                    this._showReasoningResults(data.results);
                } else if (data.type === "error") {
                    this._reasoningEventSource.close();
                    const safeError = this._securityUtils.sanitizeErrorMessage(data.error);
                    MessageBox.error("Reasoning failed: " + safeError);
                }
            }.bind(this);
            
            this._reasoningEventSource.onerror = function() {
                this._reasoningEventSource.close();
                MessageBox.error("Lost connection to reasoning process");
            }.bind(this);
        },

        _updateReasoningProgress: function(data) {
            const safeStage = this._securityUtils.encodeHTML(data.stage || 'Processing');
            const safeProgress = parseInt(data.progress) || 0;
            const safeConfidence = parseFloat(data.confidence) || 0;
            MessageToast.show(safeStage + ": " + safeProgress + "% (Confidence: " + safeConfidence + "%)");
        },

        _showReasoningResults: function(results) {
            var oView = this.base.getView();
            
            if (!this._oResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.ReasoningResults",
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
            this._createInferenceNetwork(results.inferences);
            this._createConfidenceBreakdown(results.confidence);
        },

        onGenerateInferences: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/infer",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showInferenceResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Inference generation failed: " + errorMsg);
                }.bind(this)
            });
        },

        _showInferenceResults: function(inferenceData) {
            var oView = this.base.getView();
            
            if (!this._oInferenceResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.InferenceResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oInferenceResultsDialog = oDialog;
                    oView.addDependent(this._oInferenceResultsDialog);
                    
                    var oModel = new JSONModel(inferenceData);
                    this._oInferenceResultsDialog.setModel(oModel, "inference");
                    this._oInferenceResultsDialog.open();
                }.bind(this));
            } else {
                var oModel = new JSONModel(inferenceData);
                this._oInferenceResultsDialog.setModel(oModel, "inference");
                this._oInferenceResultsDialog.open();
            }
        },

        onMakeDecision: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oDecisionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.MakeDecision",
                    controller: this
                }).then(function(oDialog) {
                    this._oDecisionDialog = oDialog;
                    this.base.getView().addDependent(this._oDecisionDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        decisionCriteria: [],
                        weightingStrategy: "EQUAL",
                        riskTolerance: "MEDIUM",
                        timeHorizon: "SHORT_TERM",
                        stakeholderPriorities: []
                    });
                    this._oDecisionDialog.setModel(oModel, "decision");
                    this._oDecisionDialog.open();
                    
                    this._loadDecisionOptions(sTaskId);
                }.bind(this));
            } else {
                this._oDecisionDialog.open();
                this._loadDecisionOptions(sTaskId);
            }
        },

        _loadDecisionOptions: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/decision-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oDecisionDialog.getModel("decision");
                    var oData = oModel.getData();
                    oData.availableAlternatives = data.alternatives;
                    oData.availableCriteria = data.criteria;
                    oData.stakeholders = data.stakeholders;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load decision options: " + errorMsg);
                }.bind(this)
            });
        },

        onValidateConclusion: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/validate",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showValidationResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Validation failed: " + errorMsg);
                }.bind(this)
            });
        },

        _showValidationResults: function(validationData) {
            var sMessage = "Conclusion Validation Results:\\n\\n";
            
            sMessage += "Validation Status: " + this._securityUtils.encodeHTML(validationData.status || 'Unknown') + "\\n";
            sMessage += "Confidence Score: " + (parseFloat(validationData.confidence) || 0) + "%\\n";
            sMessage += "Logical Consistency: " + this._securityUtils.encodeHTML(validationData.consistency || 'Unknown') + "\\n";
            sMessage += "Supporting Evidence: " + (parseInt(validationData.supportingEvidence) || 0) + " facts\\n\\n";
            
            if (validationData.issues && validationData.issues.length > 0) {
                sMessage += "Validation Issues:\\n";
                validationData.issues.forEach(function(issue) {
                    const safeType = this._securityUtils.encodeHTML(issue.type || 'Unknown');
                    const safeDesc = this._securityUtils.encodeHTML(issue.description || 'No description');
                    sMessage += "• " + safeType + ": " + safeDesc + "\\n";
                }.bind(this));
            }
            
            if (validationData.recommendations && validationData.recommendations.length > 0) {
                sMessage += "\\nRecommendations:\\n";
                validationData.recommendations.forEach(function(rec) {
                    const safeRec = this._securityUtils.encodeHTML(rec || '');
                    sMessage += "• " + safeRec + "\\n";
                }.bind(this));
            }
            
            MessageBox.information(sMessage);
        },

        onExplainReasoning: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oExplanationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.ReasoningExplanation",
                    controller: this
                }).then(function(oDialog) {
                    this._oExplanationDialog = oDialog;
                    this.base.getView().addDependent(this._oExplanationDialog);
                    this._oExplanationDialog.open();
                    this._loadExplanationData(sTaskId);
                }.bind(this));
            } else {
                this._oExplanationDialog.open();
                this._loadExplanationData(sTaskId);
            }
        },

        _loadExplanationData: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/explain",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        explanation: data.explanation,
                        reasoningChain: data.reasoningChain,
                        factJustification: data.factJustification,
                        ruleApplications: data.ruleApplications,
                        confidenceBreakdown: data.confidenceBreakdown
                    });
                    this._oExplanationDialog.setModel(oModel, "explanation");
                    this._createExplanationVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load explanation: " + errorMsg);
                }.bind(this)
            });
        },

        _createExplanationVisualizations: function(data) {
            // Create reasoning chain diagram
            this._createReasoningChainDiagram(data.reasoningChain);
            // Create confidence breakdown chart
            this._createConfidenceChart(data.confidenceBreakdown);
        },

        onOptimizeEngine: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sReasoningEngine = oContext.getProperty("reasoningEngine");
            
            MessageBox.confirm(
                "Optimize reasoning engine '" + sReasoningEngine + "' for this task?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeEngine(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeEngine: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    optimizationStrategy: "PERFORMANCE",
                    targetMetric: "INFERENCE_SPEED",
                    constraintHandling: "SOFT"
                }),
                success: function(data) {
                    const safePerformance = parseFloat(data.performanceImprovement) || 0;
                    const safeMemory = parseFloat(data.memoryReduction) || 0;
                    const safeAccuracy = parseFloat(data.accuracyMaintained) || 0;
                    MessageBox.success(
                        "Engine optimization completed!\\n" +
                        "Performance improvement: " + safePerformance + "%\\n" +
                        "Memory reduction: " + safeMemory + "%\\n" +
                        "Accuracy maintained: " + safeAccuracy + "%"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Engine optimization failed: " + errorMsg);
                }.bind(this)
            });
        },

        onUpdateKnowledge: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oKnowledgeUpdateDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.UpdateKnowledge",
                    controller: this
                }).then(function(oDialog) {
                    this._oKnowledgeUpdateDialog = oDialog;
                    this.base.getView().addDependent(this._oKnowledgeUpdateDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        updateType: "INCREMENTAL",
                        knowledgeSource: "EXTERNAL",
                        validationLevel: "STRICT",
                        conflictResolution: "MANUAL"
                    });
                    this._oKnowledgeUpdateDialog.setModel(oModel, "update");
                    this._oKnowledgeUpdateDialog.open();
                }.bind(this));
            } else {
                this._oKnowledgeUpdateDialog.open();
            }
        },

        onAnalyzeContradictions: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/analyze-contradictions",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showContradictionAnalysis(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Contradiction analysis failed: " + errorMsg);
                }.bind(this)
            });
        },

        _showContradictionAnalysis: function(analysisData) {
            var sMessage = "Contradiction Analysis Results:\\n\\n";
            
            if (analysisData.contradictions.length === 0) {
                sMessage += "No contradictions found in the knowledge base.\\n";
                sMessage += "Logical consistency: " + (parseFloat(analysisData.consistencyScore) || 0) + "%";
            } else {
                sMessage += "Found " + analysisData.contradictions.length + " contradictions:\\n\\n";
                
                analysisData.contradictions.slice(0, 5).forEach(function(contradiction, index) {
                    const safeDesc = this._securityUtils.encodeHTML(contradiction.description || 'Unknown contradiction');
                    const safeFacts = contradiction.facts ? 
                        contradiction.facts.map(f => this._securityUtils.encodeHTML(f)).join(", ") : 
                        'No facts';
                    const safeSeverity = this._securityUtils.encodeHTML(contradiction.severity || 'Unknown');
                    
                    sMessage += (index + 1) + ". " + safeDesc + "\\n";
                    sMessage += "   Conflicting facts: " + safeFacts + "\\n";
                    sMessage += "   Severity: " + safeSeverity + "\\n\\n";
                }.bind(this));
                
                if (analysisData.contradictions.length > 5) {
                    sMessage += "... and " + (analysisData.contradictions.length - 5) + " more contradictions\\n\\n";
                }
                
                sMessage += "Resolution strategies:\\n";
                analysisData.resolutionStrategies.forEach(function(strategy) {
                    const safeStrategy = this._securityUtils.encodeHTML(strategy || '');
                    sMessage += "• " + safeStrategy + "\\n";
                }.bind(this));
            }
            
            MessageBox.information(sMessage, {
                actions: ["Resolve Contradictions", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Resolve Contradictions") {
                        this._resolveContradictions(analysisData.contradictions);
                    }
                }.bind(this)
            });
        },

        _resolveContradictions: function(contradictions) {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/resolve-contradictions",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    contradictions: contradictions,
                    resolutionStrategy: "CONFIDENCE_BASED",
                    preserveConsistency: true
                }),
                success: function(data) {
                    const safeResolved = parseInt(data.resolvedCount) || 0;
                    const safeRemaining = parseInt(data.remainingCount) || 0;
                    const safeConsistency = parseFloat(data.newConsistencyScore) || 0;
                    MessageBox.success(
                        "Contradictions resolved successfully!\\n" +
                        "Resolved: " + safeResolved + "\\n" +
                        "Remaining: " + safeRemaining + "\\n" +
                        "Consistency improved to: " + safeConsistency + "%"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to resolve contradictions: " + errorMsg);
                }.bind(this)
            });
        },

        onConfirmDecision: function() {
            var oModel = this._oDecisionDialog.getModel("decision");
            var oData = oModel.getData();
            
            if (!oData.decisionCriteria || oData.decisionCriteria.length === 0) {
                MessageBox.error("Please define decision criteria");
                return;
            }
            
            this._oDecisionDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + oData.taskId + "/decide",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oDecisionDialog.setBusy(false);
                    this._oDecisionDialog.close();
                    
                    const safeAction = this._securityUtils.encodeHTML(data.recommendedAction || 'Unknown');
                    const safeConfidence = parseFloat(data.confidence) || 0;
                    const safeOutcome = this._securityUtils.encodeHTML(data.expectedOutcome || 'Unknown');
                    MessageBox.success(
                        "Decision made successfully!\\n" +
                        "Recommended action: " + safeAction + "\\n" +
                        "Confidence: " + safeConfidence + "%\\n" +
                        "Expected outcome: " + safeOutcome
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oDecisionDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Decision making failed: " + errorMsg);
                }.bind(this)
            });
        },

        onConfirmKnowledgeUpdate: function() {
            var oModel = this._oKnowledgeUpdateDialog.getModel("update");
            var oData = oModel.getData();
            
            this._oKnowledgeUpdateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + oData.taskId + "/update-knowledge",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oKnowledgeUpdateDialog.setBusy(false);
                    this._oKnowledgeUpdateDialog.close();
                    
                    const safeFacts = parseInt(data.factsAdded) || 0;
                    const safeRules = parseInt(data.rulesUpdated) || 0;
                    const safeScore = parseFloat(data.consistencyScore) || 0;
                    MessageBox.success(
                        "Knowledge base updated successfully!\\n" +
                        "New facts added: " + safeFacts + "\\n" +
                        "Rules updated: " + safeRules + "\\n" +
                        "Consistency score: " + safeScore + "%"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oKnowledgeUpdateDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Knowledge update failed: " + errorMsg);
                }.bind(this)
            });
        },

        _createInferenceNetwork: function(inferences) {
            // Placeholder for creating a network visualization of inferences
            // This would typically use a graph library like D3.js or vis.js
        },

        _createConfidenceBreakdown: function(confidence) {
            // Placeholder for creating confidence visualization
        },

        _createReasoningChainDiagram: function(chain) {
            // Placeholder for creating reasoning chain visualization
        },

        _createConfidenceChart: function(breakdown) {
            // Placeholder for creating confidence breakdown chart
        }
    });
});