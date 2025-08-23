sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent9/ext/utils/SecurityUtils"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent9.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 9 List Report - Advanced Reasoning and Decision Making.
     * Provides sophisticated AI reasoning capabilities including knowledge management, inference engines,
     * logical analysis, decision making, and problem solving with enterprise-grade security and accessibility.
     */
    return ControllerExtension.extend("a2a.network.agent9.ext.controller.ListReportExt", {
        
        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model, dialog caching, and real-time updates.
             * @override
             */
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeUpdates();
            },
            
            /**
             * @function onExit
             * @description Cleanup resources on controller destruction.
             * @override
             */
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },
        
        // Dialog caching for performance
        _dialogCache: {},
        
        // Error recovery configuration
        _errorRecoveryConfig: {
            maxRetries: 3,
            retryDelay: 1000,
            exponentialBackoff: true
        },

        /**
         * @function _initializeDeviceModel
         * @description Sets up device model for responsive design.
         * @private
         */
        _initializeDeviceModel: function() {
            var oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
            this.base.getView().setModel(oDeviceModel, "device");
        },
        
        /**
         * @function _initializeDialogCache
         * @description Initializes dialog cache for performance.
         * @private
         */
        _initializeDialogCache: function() {
            this._dialogCache = {};
        },
        
        /**
         * @function _initializePerformanceOptimizations
         * @description Sets up performance optimization features.
         * @private
         */
        _initializePerformanceOptimizations: function() {
            // Throttle dashboard updates
            this._throttledDashboardUpdate = this._throttle(this._loadDashboardData.bind(this), 1000);
            // Debounce search operations
            this._debouncedSearch = this._debounce(this._performSearch.bind(this), 300);
        },
        
        /**
         * @function _throttle
         * @description Creates a throttled function.
         * @param {Function} fn - Function to throttle
         * @param {number} limit - Time limit in milliseconds
         * @returns {Function} Throttled function
         * @private
         */
        _throttle: function(fn, limit) {
            var inThrottle;
            return function() {
                var args = arguments;
                var context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(function() { inThrottle = false; }, limit);
                }
            };
        },
        
        /**
         * @function _debounce
         * @description Creates a debounced function.
         * @param {Function} fn - Function to debounce
         * @param {number} delay - Delay in milliseconds
         * @returns {Function} Debounced function
         * @private
         */
        _debounce: function(fn, delay) {
            var timeoutId;
            return function() {
                var context = this;
                var args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
                    fn.apply(context, args);
                }, delay);
            };
        },
        
        /**
         * @function _performSearch
         * @description Performs search operation (placeholder for search functionality).
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic for reasoning tasks and knowledge base
        },
        
        /**
         * @function onCreateReasoningTask
         * @description Opens dialog to create new reasoning task with AI parameters.
         * @public
         */
        onCreateReasoningTask: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("createReasoningTask", "a2a.network.agent9.ext.fragment.CreateReasoningTask")
                .then(function(oDialog) {
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        reasoningType: "DEDUCTIVE",
                        problemDomain: "",
                        reasoningEngine: "FORWARD_CHAINING",
                        priority: "MEDIUM",
                        confidenceThreshold: 0.85,
                        maxInferenceDepth: 10,
                        chainingStrategy: "BREADTH_FIRST",
                        uncertaintyHandling: "PROBABILISTIC",
                        parallelReasoning: true
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadReasoningOptions(oDialog);
                }.bind(this));
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
            
            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
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
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery.
         * @param {Function} fnOperation - Operation to execute
         * @param {Object} oOptions - Recovery options
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery: function(fnOperation, oOptions) {
            var that = this;
            var oConfig = Object.assign({}, this._errorRecoveryConfig, oOptions);
            
            function attempt(retriesLeft, delay) {
                return fnOperation().catch(function(error) {
                    if (retriesLeft > 0) {
                        var oBundle = that.base.getView().getModel("i18n").getResourceBundle();
                        var sRetryMsg = oBundle.getText("recovery.retrying") || "Network error. Retrying...";
                        MessageToast.show(sRetryMsg);
                        
                        return new Promise(function(resolve) {
                            setTimeout(resolve, delay);
                        }).then(function() {
                            var nextDelay = oConfig.exponentialBackoff ? delay * 2 : delay;
                            return attempt(retriesLeft - 1, nextDelay);
                        });
                    }
                    throw error;
                });
            }
            
            return attempt(oConfig.maxRetries, oConfig.retryDelay);
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up event sources
            if (this._realtimeEventSource) {
                this._realtimeEventSource.close();
                this._realtimeEventSource = null;
            }
            if (this._reasoningEventSource) {
                this._reasoningEventSource.close();
                this._reasoningEventSource = null;
            }
            
            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach(function(key) {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            }.bind(this));
            this._dialogCache = {};
        },

        /**
         * @function _loadReasoningOptions
         * @description Loads reasoning engine options and AI parameters.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadReasoningOptions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createReasoningTask"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/reasoning-options",
                        type: "GET",
                        success: function(data) {
                            var oModel = oTargetDialog.getModel("create");
                            var oData = oModel.getData();
                            oData.availableEngines = data.engines;
                            oData.problemDomains = data.domains;
                            oData.reasoningTypes = data.types;
                            oModel.setData(oData);
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load reasoning options: " + errorMsg));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onReasoningDashboard
         * @description Opens comprehensive reasoning analytics dashboard.
         * @public
         */
        onReasoningDashboard: function() {
            this._getOrCreateDialog("reasoningDashboard", "a2a.network.agent9.ext.fragment.ReasoningDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this));
        },

        /**
         * @function _loadDashboardData
         * @description Loads reasoning dashboard data with AI metrics and performance analytics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/dashboard",
                        type: "GET",
                        success: function(data) {
                            var oDashboardModel = new JSONModel({
                                summary: data.summary,
                                reasoningMetrics: data.reasoningMetrics,
                                knowledgeBase: data.knowledgeBase,
                                enginePerformance: data.enginePerformance,
                                inferenceTrends: data.inferenceTrends,
                                decisionAccuracy: data.decisionAccuracy
                            });
                            
                            oTargetDialog.setModel(oDashboardModel, "dashboard");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load dashboard data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createDashboardCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createDashboardCharts
         * @description Creates reasoning analytics visualization charts.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) return;
            
            this._createConfidenceTrendsChart(data.inferenceTrends, oTargetDialog);
            this._createEnginePerformanceChart(data.enginePerformance, oTargetDialog);
            this._createDecisionAccuracyChart(data.decisionAccuracy, oTargetDialog);
        },

        /**
         * @function _createConfidenceTrendsChart
         * @description Creates confidence trends visualization chart.
         * @param {Object} trendsData - Inference trends data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createConfidenceTrendsChart: function(trendsData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) return;
            
            var oVizFrame = oTargetDialog.byId("confidenceTrendsChart");
            if (!oVizFrame || !trendsData) return;
            
            var aChartData = trendsData.map(function(trend) {
                return {
                    Time: trend.timestamp,
                    Confidence: trend.averageConfidence,
                    Inferences: trend.inferencesGenerated,
                    Accuracy: trend.accuracy
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                trendsData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.time") || "Time" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.confidencePercent") || "Confidence %" }
                },
                title: {
                    text: oBundle.getText("chart.confidenceTrends") || "Reasoning Confidence Trends"
                }
            });
        },

        /**
         * @function onKnowledgeManager
         * @description Opens knowledge management interface for facts, rules, and ontologies.
         * @public
         */
        onKnowledgeManager: function() {
            this._getOrCreateDialog("knowledgeManager", "a2a.network.agent9.ext.fragment.KnowledgeManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadKnowledgeData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Knowledge Manager: " + error.message);
                });
        },

        /**
         * @function _loadKnowledgeData
         * @description Loads knowledge base data including facts, rules, and ontologies.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadKnowledgeData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["knowledgeManager"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/knowledge-base",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                facts: data.facts,
                                rules: data.rules,
                                ontologies: data.ontologies,
                                consistency: data.consistency,
                                completeness: data.completeness,
                                domains: data.domains
                            });
                            oTargetDialog.setModel(oModel, "knowledge");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load knowledge data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createKnowledgeVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createKnowledgeVisualizations
         * @description Creates knowledge base growth and distribution visualizations.
         * @param {Object} data - Knowledge data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createKnowledgeVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["knowledgeManager"];
            if (!oTargetDialog) return;
            
            var oKnowledgeChart = oTargetDialog.byId("knowledgeGrowthChart");
            if (!oKnowledgeChart || !data.growth) return;
            
            var aChartData = data.growth.map(function(point) {
                return {
                    Date: point.date,
                    Facts: point.factsCount,
                    Rules: point.rulesCount,
                    Inferences: point.inferencesCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                growthData: aChartData
            });
            oKnowledgeChart.setModel(oChartModel);
            
            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oKnowledgeChart.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.time") || "Time" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.count") || "Count" }
                },
                title: {
                    text: oBundle.getText("chart.knowledgeGrowth") || "Knowledge Base Growth"
                }
            });
        },

        /**
         * @function onRuleEngine
         * @description Opens rule engine configuration interface for managing inference rules.
         * @public
         */
        onRuleEngine: function() {
            this._getOrCreateDialog("ruleEngine", "a2a.network.agent9.ext.fragment.RuleEngine")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadRuleData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Rule Engine: " + error.message);
                });
        },

        /**
         * @function _loadRuleData
         * @description Loads rule engine data including rules and performance metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadRuleData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["ruleEngine"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/rules",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                rules: data.rules,
                                ruleTypes: data.ruleTypes,
                                conflictResolution: data.conflictResolution,
                                rulePerformance: data.performance
                            });
                            oTargetDialog.setModel(oModel, "rules");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load rule data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createRuleVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onInferenceEngine
         * @description Opens inference engine interface for managing logical inferences.
         * @public
         */
        onInferenceEngine: function() {
            this._getOrCreateDialog("inferenceEngine", "a2a.network.agent9.ext.fragment.InferenceEngine")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadInferenceData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Inference Engine: " + error.message);
                });
        },

        /**
         * @function _loadInferenceData
         * @description Loads inference engine data including inference chains and validation.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadInferenceData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/inferences",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                inferences: data.inferences,
                                inferenceChains: data.chains,
                                confidence: data.confidence,
                                validation: data.validation
                            });
                            oTargetDialog.setModel(oModel, "inference");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load inference data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createInferenceVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createInferenceVisualizations
         * @description Creates inference chain and confidence distribution visualizations.
         * @param {Object} data - Inference data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createInferenceVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog) return;
            
            this._createInferenceChainDiagram(data.chains, oTargetDialog);
            this._createConfidenceDistribution(data.confidence, oTargetDialog);
        },

        /**
         * @function _createInferenceChainDiagram
         * @description Creates network diagram showing inference relationships.
         * @param {Array} chains - Inference chains data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createInferenceChainDiagram: function(chains, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog || !chains) return;
            
            var oChainChart = oTargetDialog.byId("inferenceChainChart");
            if (!oChainChart) return;
            
            var aChartData = chains.map(function(chain) {
                return {
                    ChainId: chain.id,
                    Steps: chain.steps.length,
                    Confidence: chain.confidence,
                    Depth: chain.depth
                };
            });
            
            var oChainModel = new sap.ui.model.json.JSONModel({
                chainData: aChartData
            });
            oChainChart.setModel(oChainModel);
        },

        /**
         * @function onDecisionMaker
         * @description Opens decision-making interface for multi-criteria analysis.
         * @public
         */
        onDecisionMaker: function() {
            this._getOrCreateDialog("decisionMaker", "a2a.network.agent9.ext.fragment.DecisionMaker")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDecisionData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Decision Maker: " + error.message);
                });
        },

        /**
         * @function _loadDecisionData
         * @description Loads decision-making data including criteria and alternatives.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDecisionData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["decisionMaker"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/decisions",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                decisions: data.decisions,
                                criteria: data.criteria,
                                alternatives: data.alternatives,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "decision");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load decision data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createDecisionVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onProblemSolver
         * @description Opens problem-solving interface for complex reasoning scenarios.
         * @public
         */
        onProblemSolver: function() {
            this._getOrCreateDialog("problemSolver", "a2a.network.agent9.ext.fragment.ProblemSolver")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadProblemData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Problem Solver: " + error.message);
                });
        },

        /**
         * @function _loadProblemData
         * @description Loads problem-solving data including strategies and complexity metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadProblemData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["problemSolver"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/problems",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                problems: data.problems,
                                solutions: data.solutions,
                                strategies: data.strategies,
                                complexity: data.complexity
                            });
                            oTargetDialog.setModel(oModel, "problem");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load problem data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createProblemVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onLogicalAnalyzer
         * @description Opens logical analyzer for consistency checks and contradiction detection.
         * @public
         */
        onLogicalAnalyzer: function() {
            this._getOrCreateDialog("logicalAnalyzer", "a2a.network.agent9.ext.fragment.LogicalAnalyzer")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadAnalysisData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Logical Analyzer: " + error.message);
                });
        },

        /**
         * @function _loadAnalysisData
         * @description Loads logical analysis data including contradictions and consistency checks.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadAnalysisData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["logicalAnalyzer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/analysis",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                contradictions: data.contradictions,
                                consistencyChecks: data.consistencyChecks,
                                logicalErrors: data.logicalErrors,
                                optimization: data.optimization
                            });
                            oTargetDialog.setModel(oModel, "analysis");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load analysis data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createAnalysisVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for reasoning progress and events.
         * @private
         */
        _startRealtimeUpdates: function() {
            // Validate EventSource URL for security
            if (!this._securityUtils.validateEventSourceUrl("/a2a/agent9/v1/realtime-updates")) {
                MessageBox.error("Invalid real-time update URL");
                return;
            }
            
            this._realtimeEventSource = new EventSource("/a2a/agent9/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "reasoning_complete") {
                        const safeTaskName = this._securityUtils.encodeHTML(data.taskName || 'Unknown task');
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sMessage = oBundle.getText("msg.reasoningCompleted") || "Reasoning completed";
                        MessageToast.show(sMessage + ": " + safeTaskName);
                        this._extensionAPI.refresh();
                    } else if (data.type === "inference_generated") {
                        const safeInference = this._securityUtils.encodeHTML(data.inference || 'Unknown inference');
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sMessage = oBundle.getText("msg.newInference") || "New inference";
                        MessageToast.show(sMessage + ": " + safeInference);
                    } else if (data.type === "contradiction_detected") {
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sMessage = oBundle.getText("msg.contradictionDetected") || "Contradiction detected and resolved";
                        MessageToast.show(sMessage);
                    }
                } catch (error) {
                    console.error("Error processing real-time update:", error);
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("msg.realtimeDisconnected") || "Real-time updates disconnected";
                MessageToast.show(sMessage);
            }.bind(this);
        },

        /**
         * @function onConfirmCreateTask
         * @description Confirms and creates a new reasoning task with validation.
         * @public
         */
        onConfirmCreateTask: function() {
            var oDialog = this._dialogCache["createReasoningTask"];
            if (!oDialog) {
                MessageBox.error("Create dialog not found");
                return;
            }
            
            var oModel = oDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validate required fields
            const validation = this._validateCreateTaskData(oData);
            if (!validation.isValid) {
                MessageBox.error(validation.message);
                return;
            }
            
            // Validate reasoning parameters
            const paramValidation = this._securityUtils.validateReasoningParameters(oData);
            if (!paramValidation.isValid) {
                MessageBox.error("Invalid reasoning parameters: " + paramValidation.message);
                return;
            }
            
            // Sanitize input data
            const sanitizedData = this._sanitizeCreateTaskData(oData);
            
            oDialog.setBusy(true);
            
            // Create secure AJAX configuration
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/reasoning-tasks",
                type: "POST",
                data: JSON.stringify(sanitizedData),
                success: function(data) {
                    oDialog.setBusy(false);
                    oDialog.close();
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sMessage = oBundle.getText("msg.taskCreated") || "Reasoning task created successfully";
                    MessageToast.show(sMessage);
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog('REASONING_TASK_CREATED', { taskName: sanitizedData.taskName });
                }.bind(this),
                error: function(xhr) {
                    oDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sErrorMsg = oBundle.getText("error.createTaskFailed") || "Failed to create task";
                    MessageBox.error(sErrorMsg + ": " + errorMsg);
                    this._securityUtils.auditLog('REASONING_TASK_CREATE_FAILED', { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        /**
         * @function onCancelCreateTask
         * @description Cancels reasoning task creation and closes dialog.
         * @public
         */
        onCancelCreateTask: function() {
            var oDialog = this._dialogCache["createReasoningTask"];
            if (oDialog) {
                oDialog.close();
            }
        },

        /**
         * @function _createRuleVisualizations
         * @description Creates rule engine performance and distribution visualizations.
         * @param {Object} data - Rule data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createRuleVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["ruleEngine"];
            if (!oTargetDialog || !data.performance) return;
            
            var oRuleChart = oTargetDialog.byId("rulePerformanceChart");
            if (!oRuleChart) return;
            
            var aChartData = data.performance.map(function(perf) {
                return {
                    RuleType: perf.type,
                    ExecutionTime: perf.avgExecutionTime,
                    Accuracy: perf.accuracy,
                    Usage: perf.usageCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                performanceData: aChartData
            });
            oRuleChart.setModel(oChartModel);
        },

        /**
         * @function _createDecisionVisualizations
         * @description Creates decision criteria and alternatives visualizations.
         * @param {Object} data - Decision data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDecisionVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["decisionMaker"];
            if (!oTargetDialog || !data.criteria) return;
            
            var oDecisionChart = oTargetDialog.byId("decisionCriteriaChart");
            if (!oDecisionChart) return;
            
            var aChartData = data.criteria.map(function(criterion) {
                return {
                    Criterion: criterion.name,
                    Weight: criterion.weight,
                    Impact: criterion.impact
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                criteriaData: aChartData
            });
            oDecisionChart.setModel(oChartModel);
        },

        /**
         * @function _createProblemVisualizations
         * @description Creates problem complexity and solution visualizations.
         * @param {Object} data - Problem data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createProblemVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["problemSolver"];
            if (!oTargetDialog || !data.complexity) return;
            
            var oProblemChart = oTargetDialog.byId("problemComplexityChart");
            if (!oProblemChart) return;
            
            var aChartData = data.complexity.map(function(complex) {
                return {
                    Domain: complex.domain,
                    Complexity: complex.score,
                    SolutionTime: complex.avgSolutionTime,
                    Success: complex.successRate
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                complexityData: aChartData
            });
            oProblemChart.setModel(oChartModel);
        },

        /**
         * @function _createAnalysisVisualizations
         * @description Creates logical analysis and consistency visualizations.
         * @param {Object} data - Analysis data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createAnalysisVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["logicalAnalyzer"];
            if (!oTargetDialog || !data.consistencyChecks) return;
            
            var oAnalysisChart = oTargetDialog.byId("consistencyTrendsChart");
            if (!oAnalysisChart) return;
            
            var aChartData = data.consistencyChecks.map(function(check) {
                return {
                    Timestamp: check.timestamp,
                    Consistency: check.score,
                    Contradictions: check.contradictionCount,
                    Resolved: check.resolvedCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                consistencyData: aChartData
            });
            oAnalysisChart.setModel(oChartModel);
        },

        /**
         * @function _createConfidenceDistribution
         * @description Creates confidence level distribution visualization.
         * @param {Object} confidence - Confidence data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createConfidenceDistribution: function(confidence, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog || !confidence) return;
            
            var oConfidenceChart = oTargetDialog.byId("confidenceDistributionChart");
            if (!oConfidenceChart) return;
            
            var aChartData = confidence.distribution.map(function(dist) {
                return {
                    Range: dist.range,
                    Count: dist.count,
                    Percentage: dist.percentage
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                distributionData: aChartData
            });
            oConfidenceChart.setModel(oChartModel);
        },

        /**
         * @function _validateCreateTaskData
         * @description Validates reasoning task creation data.
         * @param {Object} oData - Task data to validate
         * @returns {Object} Validation result with isValid flag and message
         * @private
         */
        _validateCreateTaskData: function(oData) {
            if (!oData.taskName || !oData.taskName.trim()) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("validation.taskNameRequired") || "Task name is required";
                return { isValid: false, message: sMessage };
            }
            
            const taskNameValidation = this._securityUtils.validateInput(oData.taskName, 'text', {
                required: true,
                minLength: 3,
                maxLength: 100
            });
            if (!taskNameValidation.isValid) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sPrefix = oBundle.getText("field.taskName") || "Task name";
                return { isValid: false, message: sPrefix + ": " + taskNameValidation.message };
            }
            
            if (!oData.reasoningType || !oData.reasoningType.trim()) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("validation.reasoningTypeRequired") || "Reasoning type is required";
                return { isValid: false, message: sMessage };
            }
            
            const reasoningTypeValidation = this._securityUtils.validateReasoningType(oData.reasoningType);
            if (!reasoningTypeValidation.isValid) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sPrefix = oBundle.getText("field.reasoningType") || "Reasoning type";
                return { isValid: false, message: sPrefix + ": " + reasoningTypeValidation.message };
            }
            
            if (!oData.problemDomain || !oData.problemDomain.trim()) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("validation.problemDomainRequired") || "Problem domain is required";
                return { isValid: false, message: sMessage };
            }
            
            return { isValid: true };
        },

        _sanitizeCreateTaskData: function(oData) {
            return {
                taskName: this._securityUtils.sanitizeInput(oData.taskName),
                description: this._securityUtils.sanitizeInput(oData.description || ''),
                reasoningType: this._securityUtils.sanitizeInput(oData.reasoningType),
                problemDomain: this._securityUtils.sanitizeInput(oData.problemDomain),
                reasoningEngine: this._securityUtils.sanitizeInput(oData.reasoningEngine || 'FORWARD_CHAINING'),
                priority: this._securityUtils.sanitizeInput(oData.priority || 'MEDIUM'),
                confidenceThreshold: Math.max(0, Math.min(1, parseFloat(oData.confidenceThreshold) || 0.85)),
                maxInferenceDepth: Math.max(1, Math.min(50, parseInt(oData.maxInferenceDepth) || 10)),
                chainingStrategy: this._securityUtils.sanitizeInput(oData.chainingStrategy || 'BREADTH_FIRST'),
                uncertaintyHandling: this._securityUtils.sanitizeInput(oData.uncertaintyHandling || 'PROBABILISTIC'),
                parallelReasoning: Boolean(oData.parallelReasoning)
            };
        }
    });
});