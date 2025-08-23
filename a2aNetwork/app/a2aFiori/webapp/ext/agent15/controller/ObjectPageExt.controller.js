sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent15/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent15.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._initializeCreateModel();
            }
        },
        
        _initializeCreateModel: function() {
            var oCreateData = {
                workflowName: "",
                description: "",
                workflowType: "",
                priority: "medium",
                version: "1.0.0",
                orchestrationMode: "centralized",
                executionStrategy: "sequential",
                parallelization: false,
                maxConcurrency: 4,
                taskDistribution: "roundRobin",
                loadBalancing: true,
                failoverStrategy: "immediate",
                retryPolicy: "exponential",
                circuitBreaker: true,
                triggerType: "manual",
                schedule: "",
                timeout: 300,
                maxDuration: 3600,
                checkpointEnabled: true,
                rollbackEnabled: true,
                compensationEnabled: false,
                transactional: false,
                agentSelectionMode: "automatic",
                communicationProtocol: "websocket",
                messageQueuing: true,
                eventBus: true,
                coordinationPattern: "masterSlave",
                consensusAlgorithm: "raft",
                conflictResolution: "priority",
                dataConsistency: "eventual",
                monitoringEnabled: true,
                metricsCollection: true,
                tracingEnabled: true,
                loggingLevel: "info",
                alertingEnabled: true,
                healthCheckInterval: 30,
                performanceThresholds: "",
                anomalyDetection: false,
                workflowNameState: "",
                workflowNameStateText: "",
                workflowTypeState: "",
                workflowTypeStateText: "",
                orchestrationModeState: "",
                orchestrationModeStateText: "",
                canCreate: false
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.getView().setModel(oCreateModel, "create");
        },

        // Create Workflow Action
        onCreateWorkflow: function() {
            var oView = this.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent15.ext.fragment.WorkflowCreator",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._resetCreateModel();
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._resetCreateModel();
                this._oCreateDialog.open();
            }
        },
        
        // Create Dialog Lifecycle
        onCreateWorkflowDialogAfterOpen: function() {
            // Focus on first input field
            var oWorkflowNameInput = this.getView().byId("workflowNameInput");
            if (oWorkflowNameInput) {
                oWorkflowNameInput.focus();
            }
            
            // Start real-time validation
            this._startCreateValidationInterval();
        },
        
        onCreateWorkflowDialogAfterClose: function() {
            this._stopCreateValidationInterval();
        },
        
        _startCreateValidationInterval: function() {
            if (this.createValidationInterval) {
                clearInterval(this.createValidationInterval);
            }
            
            this.createValidationInterval = setInterval(function() {
                this._validateCreateForm();
            }.bind(this), 1000);
        },
        
        _stopCreateValidationInterval: function() {
            if (this.createValidationInterval) {
                clearInterval(this.createValidationInterval);
                this.createValidationInterval = null;
            }
        },
        
        _validateCreateForm: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            var bCanCreate = true;
            
            // Workflow name validation
            if (!oData.workflowName || oData.workflowName.trim().length < 3) {
                oData.workflowNameState = "Error";
                oData.workflowNameStateText = "Workflow name is required and must be at least 3 characters";
                bCanCreate = false;
            } else if (!SecurityUtils.isValidWorkflowName(oData.workflowName)) {
                oData.workflowNameState = "Error";
                oData.workflowNameStateText = "Workflow name contains invalid characters";
                bCanCreate = false;
            } else {
                oData.workflowNameState = "Success";
                oData.workflowNameStateText = "";
            }
            
            // Workflow type validation
            if (!oData.workflowType) {
                oData.workflowTypeState = "Warning";
                oData.workflowTypeStateText = "Please select a workflow type";
                bCanCreate = false;
            } else {
                oData.workflowTypeState = "Success";
                oData.workflowTypeStateText = "";
            }
            
            // Orchestration mode validation
            if (!oData.orchestrationMode) {
                oData.orchestrationModeState = "Warning";
                oData.orchestrationModeStateText = "Please select an orchestration mode";
                bCanCreate = false;
            } else {
                oData.orchestrationModeState = "Success";
                oData.orchestrationModeStateText = "";
            }
            
            oData.canCreate = bCanCreate;
            oCreateModel.setData(oData);
        },
        
        _resetCreateModel: function() {
            this._initializeCreateModel();
        },
        
        // Field Change Handlers
        onWorkflowNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.workflowName = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onDescriptionChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.description = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onWorkflowTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.workflowType = sValue;
            
            // Auto-suggest based on workflow type
            switch (sValue) {
                case "parallel":
                    oData.parallelization = true;
                    oData.executionStrategy = "parallel";
                    oData.maxConcurrency = 8;
                    break;
                case "eventDriven":
                    oData.triggerType = "event";
                    oData.eventBus = true;
                    oData.communicationProtocol = "mqtt";
                    break;
                case "microservice":
                    oData.orchestrationMode = "distributed";
                    oData.communicationProtocol = "http";
                    oData.coordinationPattern = "peer2peer";
                    break;
                case "serverless":
                    oData.executionStrategy = "optimistic";
                    oData.triggerType = "api";
                    oData.scalingEnabled = true;
                    break;
                case "batch":
                    oData.executionStrategy = "resilient";
                    oData.checkpointEnabled = true;
                    oData.rollbackEnabled = true;
                    break;
                case "streaming":
                    oData.executionStrategy = "bestEffort";
                    oData.communicationProtocol = "kafka";
                    oData.eventBus = true;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onPriorityChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.priority = sValue;
            
            // Auto-adjust settings based on priority
            switch (sValue) {
                case "critical":
                case "urgent":
                    oData.failoverStrategy = "immediate";
                    oData.retryPolicy = "immediate";
                    oData.healthCheckInterval = 15;
                    oData.alertingEnabled = true;
                    break;
                case "high":
                    oData.failoverStrategy = "graceful";
                    oData.healthCheckInterval = 20;
                    break;
                case "low":
                    oData.healthCheckInterval = 60;
                    oData.alertingEnabled = false;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onVersionChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.version = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onOrchestrationModeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.orchestrationMode = sValue;
            
            // Auto-adjust settings based on orchestration mode
            switch (sValue) {
                case "distributed":
                    oData.consensusAlgorithm = "raft";
                    oData.coordinationPattern = "peer2peer";
                    oData.conflictResolution = "voting";
                    oData.dataConsistency = "eventual";
                    break;
                case "federated":
                    oData.consensusAlgorithm = "paxos";
                    oData.coordinationPattern = "publish";
                    oData.conflictResolution = "priority";
                    break;
                case "autonomous":
                    oData.consensusAlgorithm = "gossip";
                    oData.coordinationPattern = "scatter";
                    oData.agentSelectionMode = "capability";
                    break;
                case "centralized":
                    oData.consensusAlgorithm = "none";
                    oData.coordinationPattern = "masterSlave";
                    oData.conflictResolution = "priority";
                    oData.dataConsistency = "strong";
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onExecutionStrategyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.executionStrategy = sValue;
            
            // Auto-adjust settings based on execution strategy
            switch (sValue) {
                case "parallel":
                    oData.parallelization = true;
                    oData.maxConcurrency = Math.max(oData.maxConcurrency, 4);
                    break;
                case "failfast":
                    oData.retryPolicy = "none";
                    oData.failoverStrategy = "disabled";
                    break;
                case "resilient":
                    oData.retryPolicy = "exponential";
                    oData.circuitBreaker = true;
                    oData.rollbackEnabled = true;
                    break;
                case "guaranteedDelivery":
                    oData.retryPolicy = "exponential";
                    oData.checkpointEnabled = true;
                    oData.transactional = true;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        // Continue with remaining change handlers
        onParallelizationChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.parallelization = bValue;
            if (bValue && oData.maxConcurrency < 2) {
                oData.maxConcurrency = 4;
            }
            oCreateModel.setData(oData);
        },
        
        onMaxConcurrencyChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.maxConcurrency = iValue;
            oCreateModel.setData(oData);
        },
        
        onTaskDistributionChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.taskDistribution = sValue;
            oCreateModel.setData(oData);
        },
        
        onLoadBalancingChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.loadBalancing = bValue;
            oCreateModel.setData(oData);
        },
        
        onFailoverStrategyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.failoverStrategy = sValue;
            oCreateModel.setData(oData);
        },
        
        onRetryPolicyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.retryPolicy = sValue;
            oCreateModel.setData(oData);
        },
        
        onCircuitBreakerChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.circuitBreaker = bValue;
            oCreateModel.setData(oData);
        },
        
        onTriggerTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.triggerType = sValue;
            
            // Auto-enable/disable schedule field based on trigger type
            if (sValue === "scheduled") {
                // Focus schedule input if available
                var oScheduleInput = this.getView().byId("scheduleInput");
                if (oScheduleInput) {
                    setTimeout(function() { oScheduleInput.focus(); }, 100);
                }
            }
            
            oCreateModel.setData(oData);
        },
        
        onScheduleChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.schedule = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onTimeoutChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.timeout = iValue;
            oCreateModel.setData(oData);
        },
        
        onMaxDurationChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.maxDuration = iValue;
            oCreateModel.setData(oData);
        },
        
        onCheckpointEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.checkpointEnabled = bValue;
            oCreateModel.setData(oData);
        },
        
        onRollbackEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.rollbackEnabled = bValue;
            oCreateModel.setData(oData);
        },
        
        onCompensationEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.compensationEnabled = bValue;
            oCreateModel.setData(oData);
        },
        
        onTransactionalChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.transactional = bValue;
            
            // Auto-enable related settings when transactional is enabled
            if (bValue) {
                oData.rollbackEnabled = true;
                oData.checkpointEnabled = true;
            }
            
            oCreateModel.setData(oData);
        },
        
        onAgentSelectionModeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.agentSelectionMode = sValue;
            oCreateModel.setData(oData);
        },
        
        onCommunicationProtocolChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.communicationProtocol = sValue;
            
            // Auto-adjust settings based on protocol
            switch (sValue) {
                case "mqtt":
                case "amqp":
                case "kafka":
                    oData.messageQueuing = true;
                    oData.eventBus = true;
                    break;
                case "websocket":
                    oData.eventBus = true;
                    break;
                case "http":
                    oData.messageQueuing = false;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onMessageQueuingChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.messageQueuing = bValue;
            oCreateModel.setData(oData);
        },
        
        onEventBusChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.eventBus = bValue;
            oCreateModel.setData(oData);
        },
        
        onCoordinationPatternChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.coordinationPattern = sValue;
            
            // Auto-adjust consensus algorithm based on pattern
            switch (sValue) {
                case "peer2peer":
                    oData.consensusAlgorithm = "gossip";
                    break;
                case "masterSlave":
                    oData.consensusAlgorithm = "none";
                    break;
                case "publish":
                    oData.consensusAlgorithm = "raft";
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onConsensusAlgorithmChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.consensusAlgorithm = sValue;
            oCreateModel.setData(oData);
        },
        
        onConflictResolutionChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.conflictResolution = sValue;
            oCreateModel.setData(oData);
        },
        
        onDataConsistencyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.dataConsistency = sValue;
            oCreateModel.setData(oData);
        },
        
        onMonitoringEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.monitoringEnabled = bValue;
            
            // Auto-disable related monitoring features when disabled
            if (!bValue) {
                oData.metricsCollection = false;
                oData.tracingEnabled = false;
                oData.alertingEnabled = false;
                oData.anomalyDetection = false;
            }
            
            oCreateModel.setData(oData);
        },
        
        onMetricsCollectionChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.metricsCollection = bValue;
            oCreateModel.setData(oData);
        },
        
        onTracingEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.tracingEnabled = bValue;
            oCreateModel.setData(oData);
        },
        
        onLoggingLevelChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.loggingLevel = sValue;
            oCreateModel.setData(oData);
        },
        
        onAlertingEnabledChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.alertingEnabled = bValue;
            oCreateModel.setData(oData);
        },
        
        onHealthCheckIntervalChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.healthCheckInterval = iValue;
            oCreateModel.setData(oData);
        },
        
        onPerformanceThresholdsChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.performanceThresholds = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onAnomalyDetectionChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.anomalyDetection = bValue;
            oCreateModel.setData(oData);
        },
        
        // Dialog Action Handlers
        onConfirmCreateWorkflow: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!oData.canCreate) {
                MessageToast.show(this._getResourceBundle().getText("msg.fixValidationErrors"));
                return;
            }
            
            MessageBox.confirm(this._getResourceBundle().getText("msg.confirmWorkflowCreation"), {
                title: this._getResourceBundle().getText("title.confirmWorkflowCreation"),
                onOK: function() {
                    this._createWorkflow(oData);
                }.bind(this)
            });
        },
        
        onCancelCreateWorkflow: function() {
            if (this._oCreateDialog) {
                this._oCreateDialog.close();
            }
        },
        
        _createWorkflow: function(oData) {
            // Simulate workflow creation
            MessageToast.show(this._getResourceBundle().getText("msg.workflowCreationStarted"));
            
            setTimeout(function() {
                MessageToast.show(this._getResourceBundle().getText("msg.workflowCreated", [oData.workflowName]));
                if (this._oCreateDialog) {
                    this._oCreateDialog.close();
                }
            }.bind(this), 2000);
        },
        
        // Execute Workflow Action
        onExecuteWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'running') {
                MessageToast.show(this.getResourceBundle().getText("error.workflowAlreadyRunning"));
                return;
            }

            if (oData.status === 'completed') {
                MessageBox.confirm(
                    this.getResourceBundle().getText("msg.restartWorkflowConfirm"),
                    {
                        onClose: function(oAction) {
                            if (oAction === MessageBox.Action.OK) {
                                this._executeWorkflow(oContext);
                            }
                        }.bind(this)
                    }
                );
            } else {
                this._executeWorkflow(oContext);
            }
        },

        // Pause Workflow Action
        onPauseWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status !== 'running') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotPauseWorkflow"));
                return;
            }

            this._pauseWorkflow(oContext);
        },

        // Resume Workflow Action
        onResumeWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status !== 'paused') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotResumeWorkflow"));
                return;
            }

            this._resumeWorkflow(oContext);
        },

        // Stop Workflow Action
        onStopWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status !== 'running' && oData.status !== 'paused') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotStopWorkflow"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.stopWorkflowConfirm"),
                {
                    icon: MessageBox.Icon.WARNING,
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._stopWorkflow(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Clone Workflow Action
        onCloneWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._cloneWorkflowDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.CloneWorkflow",
                    controller: this
                }).then(function(oDialog) {
                    this._cloneWorkflowDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeCloneDialog(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeCloneDialog(oContext);
                this._cloneWorkflowDialog.open();
            }
        },

        // Configure Agents Action
        onConfigureAgents: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._agentConfigDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.AgentConfiguration",
                    controller: this
                }).then(function(oDialog) {
                    this._agentConfigDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadAgentConfiguration(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadAgentConfiguration(oContext);
                this._agentConfigDialog.open();
            }
        },

        // Monitor Execution Action
        onMonitorExecution: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._executionMonitor) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.ExecutionMonitor",
                    controller: this
                }).then(function(oDialog) {
                    this._executionMonitor = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadExecutionData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadExecutionData(oContext);
                this._executionMonitor.open();
            }
        },

        // Optimize Performance Action
        onOptimizePerformance: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'running') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotOptimizeRunningWorkflow"));
                return;
            }

            if (!this._performanceOptimizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.PerformanceOptimizer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._analyzeWorkflowPerformance(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._analyzeWorkflowPerformance(oContext);
                this._performanceOptimizer.open();
            }
        },

        // Schedule Workflow Action
        onScheduleWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._schedulerDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.WorkflowScheduler",
                    controller: this
                }).then(function(oDialog) {
                    this._schedulerDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadScheduleOptions(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadScheduleOptions(oContext);
                this._schedulerDialog.open();
            }
        },

        // View Coordination Pattern Action
        onViewCoordinationPattern: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._coordinationViewer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.CoordinationViewer",
                    controller: this
                }).then(function(oDialog) {
                    this._coordinationViewer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadCoordinationPattern(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadCoordinationPattern(oContext);
                this._coordinationViewer.open();
            }
        },

        // Export Workflow Action
        onExportWorkflow: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!this._exportDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.ExportWorkflow",
                    controller: this
                }).then(function(oDialog) {
                    this._exportDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._exportDialog.open();
            }
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeWorkflowMonitoring();
        },

        _initializeWorkflowMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const workflowId = oContext.getObject().workflowId;
            
            // Subscribe to workflow updates for this specific workflow
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = SecurityUtils.createSecureEventSource(`https://localhost:8015/orchestrator/workflow/${workflowId}/stream`, {
                    'workflow-progress': (event) => {
                        const data = JSON.parse(event.data);
                        this._updateWorkflowProgress(data);
                    },
                    'step-completed': (event) => {
                        const data = JSON.parse(event.data);
                        this._handleStepCompleted(data);
                    },
                    'agent-status': (event) => {
                        const data = JSON.parse(event.data);
                        this._updateAgentStatus(data);
                    },
                    'coordination-event': (event) => {
                        const data = JSON.parse(event.data);
                        this._handleCoordinationEvent(data);
                    },
                    'performance-alert': (event) => {
                        const data = JSON.parse(event.data);
                        this._handlePerformanceAlert(data);
                    }
                });
                
                // Event handlers configured in createSecureEventSource

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(workflowId);
            }
        },

        _initializePolling: function(workflowId) {
            this._pollInterval = setInterval(() => {
                this._refreshWorkflowData();
            }, 2000);
        },

        _executeWorkflow: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.workflowStarting"));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteWorkflow", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowStarted"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowExecutionFailed"));
                }.bind(this)
            });
        },

        _pauseWorkflow: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            SecurityUtils.secureCallFunction(oModel, "/PauseWorkflow", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowPaused"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowPauseFailed"));
                }.bind(this)
            });
        },

        _resumeWorkflow: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            SecurityUtils.secureCallFunction(oModel, "/ResumeWorkflow", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowResumed"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowResumeFailed"));
                }.bind(this)
            });
        },

        _stopWorkflow: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/StopWorkflow", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowStopped"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowStopFailed"));
                }.bind(this)
            });
        },

        _initializeCloneDialog: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/GetWorkflowTemplate", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._setupCloneDialog(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingWorkflowTemplate"));
                }.bind(this)
            });
        },

        _loadAgentConfiguration: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/GetWorkflowAgents", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._displayAgentConfiguration(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingAgentConfiguration"));
                }.bind(this)
            });
        },

        _loadExecutionData: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/GetExecutionMetrics", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._displayExecutionMonitor(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingExecutionData"));
                }.bind(this)
            });
        },

        _analyzeWorkflowPerformance: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/AnalyzeWorkflowPerformance", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._displayPerformanceAnalysis(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.performanceAnalysisFailed"));
                }.bind(this)
            });
        },

        _loadScheduleOptions: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/GetScheduleOptions", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._displayScheduleOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingScheduleOptions"));
                }.bind(this)
            });
        },

        _loadCoordinationPattern: function(oContext) {
            const oModel = this.getView().getModel();
            const sWorkflowId = oContext.getObject().workflowId;
            
            oModel.callFunction("/GetCoordinationPattern", {
                urlParameters: {
                    workflowId: sWorkflowId
                },
                success: function(data) {
                    this._displayCoordinationPattern(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingCoordinationPattern"));
                }.bind(this)
            });
        },

        _updateWorkflowProgress: function(data) {
            // Update workflow progress indicators
            const oProgressIndicator = this.getView().byId("workflowProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.completionRate);
                oProgressIndicator.setDisplayValue(`${data.completedSteps}/${data.totalSteps} steps`);
            }
        },

        _handleStepCompleted: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.stepCompleted", [data.stepName]));
            this._refreshWorkflowData();
        },

        _updateAgentStatus: function(data) {
            // Update agent status indicators
        },

        _handleCoordinationEvent: function(data) {
            // Handle coordination events
        },

        _handlePerformanceAlert: function(data) {
            MessageBox.warning(
                data.message,
                {
                    title: this.getResourceBundle().getText("msg.performanceAlert")
                }
            );
        },

        _refreshWorkflowData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _setupCloneDialog: function(data) {
            // Setup clone dialog with workflow template
        },

        _displayAgentConfiguration: function(data) {
            // Display agent configuration options
        },

        _displayExecutionMonitor: function(data) {
            // Display execution monitoring data
        },

        _displayPerformanceAnalysis: function(data) {
            // Display performance analysis results
        },

        _displayScheduleOptions: function(data) {
            // Display scheduling options
        },

        _displayCoordinationPattern: function(data) {
            // Display coordination pattern visualization
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            // Clean up create dialog
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
            
            this._stopCreateValidationInterval();
            
            // Clean up other dialogs
            if (this._cloneWorkflowDialog) {
                this._cloneWorkflowDialog.destroy();
            }
            if (this._agentConfigDialog) {
                this._agentConfigDialog.destroy();
            }
            if (this._executionMonitor) {
                this._executionMonitor.destroy();
            }
            if (this._performanceOptimizer) {
                this._performanceOptimizer.destroy();
            }
            if (this._schedulerDialog) {
                this._schedulerDialog.destroy();
            }
            if (this._coordinationViewer) {
                this._coordinationViewer.destroy();
            }
            if (this._exportDialog) {
                this._exportDialog.destroy();
            }
            
            // Clean up monitoring
            if (this._eventSource) {
                this._eventSource.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});