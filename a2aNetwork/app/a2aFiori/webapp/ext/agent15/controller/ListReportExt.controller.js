sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "a2a/network/agent15/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent15.ext.controller.ListReportExt", {
        
        // Orchestration Dashboard Action
        onOrchestrationDashboard: function() {
            if (!this._orchestrationDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.OrchestrationDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._orchestrationDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadOrchestrationData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadOrchestrationData();
                this._orchestrationDashboard.open();
            }
        },

        // Create New Workflow
        onCreateWorkflow: function() {
            if (!this._workflowCreator) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.WorkflowCreator",
                    controller: this
                }).then(function(oDialog) {
                    this._workflowCreator = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._workflowCreator.open();
            }
        },

        // Start Workflow Action
        onStartWorkflow: function() {
            const oBinding = this.base.getView().byId("fe::table::Workflows::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectWorkflowsFirst"));
                return;
            }

            const aSelectedWorkflows = aSelectedContexts.map(ctx => ctx.getObject());
            const aPendingWorkflows = aSelectedWorkflows.filter(wf => wf.status === 'pending');
            
            if (aPendingWorkflows.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("error.noExecutableWorkflows"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.startWorkflowConfirm", [aPendingWorkflows.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeWorkflows(aPendingWorkflows);
                        }
                    }.bind(this)
                }
            );
        },

        // Pause Workflow Action
        onPauseWorkflow: function() {
            const oBinding = this.base.getView().byId("fe::table::Workflows::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectWorkflowsFirst"));
                return;
            }

            const aRunningWorkflows = aSelectedContexts
                .map(ctx => ctx.getObject())
                .filter(wf => wf.status === 'running');
            
            if (aRunningWorkflows.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("error.noPausableWorkflows"));
                return;
            }

            this._pauseWorkflows(aRunningWorkflows);
        },

        // Stop Workflow Action
        onStopWorkflow: function() {
            const oBinding = this.base.getView().byId("fe::table::Workflows::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectWorkflowsFirst"));
                return;
            }

            const aActiveWorkflows = aSelectedContexts
                .map(ctx => ctx.getObject())
                .filter(wf => wf.status === 'running' || wf.status === 'paused');
            
            if (aActiveWorkflows.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("error.noStoppableWorkflows"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.stopWorkflowConfirm", [aActiveWorkflows.length]),
                {
                    icon: MessageBox.Icon.WARNING,
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._stopWorkflows(aActiveWorkflows);
                        }
                    }.bind(this)
                }
            );
        },

        // Agent Coordinator Action
        onAgentCoordinator: function() {
            if (!this._agentCoordinator) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.AgentCoordinator",
                    controller: this
                }).then(function(oDialog) {
                    this._agentCoordinator = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadAgentNetworkData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadAgentNetworkData();
                this._agentCoordinator.open();
            }
        },

        // Performance Analyzer Action
        onPerformanceAnalyzer: function() {
            if (!this._performanceAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._analyzeSystemPerformance();
                    oDialog.open();
                }.bind(this));
            } else {
                this._analyzeSystemPerformance();
                this._performanceAnalyzer.open();
            }
        },

        // Pipeline Manager Action
        onPipelineManager: function() {
            if (!this._pipelineManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.PipelineManager",
                    controller: this
                }).then(function(oDialog) {
                    this._pipelineManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPipelineData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadPipelineData();
                this._pipelineManager.open();
            }
        },

        // Queue Manager Action
        onQueueManager: function() {
            if (!this._queueManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.QueueManager",
                    controller: this
                }).then(function(oDialog) {
                    this._queueManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadQueueData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadQueueData();
                this._queueManager.open();
            }
        },

        // Event Stream Viewer Action
        onEventStreamViewer: function() {
            if (!this._eventStreamViewer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent15.ext.fragment.EventStreamViewer",
                    controller: this
                }).then(function(oDialog) {
                    this._eventStreamViewer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadEventStream();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadEventStream();
                this._eventStreamViewer.open();
            }
        },

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = SecurityUtils.createSecureWebSocket('wss://localhost:8015/orchestrator/updates', {
                    onmessage: function(event) {
                        const data = JSON.parse(event.data);
                        this._handleOrchestrationUpdate(data);
                    }.bind(this),
                    onerror: function(error) {
                        console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });
                
                if (this._ws) {
                    this._ws.onclose = function() {
                        setTimeout(() => this._initializeWebSocket(), 5000);
                    }.bind(this);
                }

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshWorkflowData();
            }, 3000);
        },

        _handleOrchestrationUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'WORKFLOW_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowStarted"));
                    this._refreshWorkflowData();
                    break;
                case 'WORKFLOW_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowCompleted"));
                    this._refreshWorkflowData();
                    break;
                case 'WORKFLOW_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowFailed"));
                    this._refreshWorkflowData();
                    break;
                case 'AGENT_STATUS_CHANGED':
                    this._updateAgentStatus(data);
                    break;
                case 'SYSTEM_ALERT':
                    this._handleSystemAlert(data);
                    break;
                case 'PERFORMANCE_THRESHOLD':
                    MessageToast.show(this.getResourceBundle().getText("msg.thresholdExceeded"));
                    break;
                case 'COORDINATION_EVENT':
                    this._updateCoordinationStatus(data);
                    break;
            }
        },

        _loadOrchestrationData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetOrchestrationMetrics", {
                success: function(data) {
                    this._updateOrchestrationDashboard(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingData"));
                }.bind(this)
            });
        },

        _executeWorkflows: function(aWorkflows) {
            const oModel = this.getView().getModel();
            const aWorkflowIds = aWorkflows.map(wf => wf.workflowId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.workflowsStarting"));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteWorkflows", {
                urlParameters: {
                    workflowIds: aWorkflowIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowsStarted"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowExecutionFailed"));
                }.bind(this)
            });
        },

        _pauseWorkflows: function(aWorkflows) {
            const oModel = this.getView().getModel();
            const aWorkflowIds = aWorkflows.map(wf => wf.workflowId);
            
            SecurityUtils.secureCallFunction(oModel, "/PauseWorkflows", {
                urlParameters: {
                    workflowIds: aWorkflowIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowsPaused"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowPauseFailed"));
                }.bind(this)
            });
        },

        _stopWorkflows: function(aWorkflows) {
            const oModel = this.getView().getModel();
            const aWorkflowIds = aWorkflows.map(wf => wf.workflowId);
            
            SecurityUtils.secureCallFunction(oModel, "/StopWorkflows", {
                urlParameters: {
                    workflowIds: aWorkflowIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.workflowsStopped"));
                    this._refreshWorkflowData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.workflowStopFailed"));
                }.bind(this)
            });
        },

        _loadAgentNetworkData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetAgentNetworkStatus", {
                success: function(data) {
                    this._updateAgentCoordinator(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingAgentData"));
                }.bind(this)
            });
        },

        _analyzeSystemPerformance: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/AnalyzeSystemPerformance", {
                success: function(data) {
                    this._displayPerformanceAnalysis(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.performanceAnalysisFailed"));
                }.bind(this)
            });
        },

        _loadPipelineData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetPipelineDefinitions", {
                success: function(data) {
                    this._updatePipelineManager(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingPipelineData"));
                }.bind(this)
            });
        },

        _loadQueueData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetTaskQueues", {
                success: function(data) {
                    this._updateQueueManager(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingQueueData"));
                }.bind(this)
            });
        },

        _loadEventStream: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetOrchestrationEvents", {
                success: function(data) {
                    this._updateEventStream(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingEventData"));
                }.bind(this)
            });
        },

        _refreshWorkflowData: function() {
            const oBinding = this.base.getView().byId("fe::table::Workflows::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateAgentStatus: function(data) {
            // Update agent status in real-time
        },

        _handleSystemAlert: function(data) {
            MessageBox.warning(
                data.message,
                {
                    title: this.getResourceBundle().getText("msg.systemAlert")
                }
            );
        },

        _updateCoordinationStatus: function(data) {
            // Update coordination status indicators
        },

        _updateOrchestrationDashboard: function(data) {
            // Update dashboard with orchestration metrics
        },

        _updateAgentCoordinator: function(data) {
            // Update agent coordinator with network status
        },

        _displayPerformanceAnalysis: function(data) {
            // Display performance analysis results
        },

        _updatePipelineManager: function(data) {
            // Update pipeline manager with pipeline data
        },

        _updateQueueManager: function(data) {
            // Update queue manager with queue data
        },

        _updateEventStream: function(data) {
            // Update event stream viewer with events
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._ws) {
                this._ws.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});