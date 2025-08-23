sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "a2a/network/agent15/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent15.ext.controller.ObjectPageExt", {
        
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
            if (this._eventSource) {
                this._eventSource.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});