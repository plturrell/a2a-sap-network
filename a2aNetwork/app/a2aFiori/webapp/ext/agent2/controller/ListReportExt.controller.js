sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, encodeXML, escapeRegExp, sanitizeHTML) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent2.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                // Initialize dialog cache and security settings
                this._dialogCache = {};
                this._csrfToken = null;
                this._initializeCSRFToken();
            }
        },

        /**
         * Opens the AI preparation task creation dialog with comprehensive configuration options.
         * Supports multiple ML frameworks, data types, and advanced AI preparation settings.
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateAITask: function() {
            // Initialize create model before opening dialog
            this._initializeCreateModel();
            this._openCachedDialog("createAITask", "a2a.network.agent2.ext.fragment.CreateAIPreparationTask");
        },

        /**
         * Opens the data profiler dialog for AI readiness assessment.
         * Provides comprehensive data quality analysis, statistical insights, and AI preparation recommendations.
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onOpenDataProfiler: function() {
            var oView = this.base.getView();
            
            if (!this._oDataProfiler) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.DataProfiler",
                    controller: this
                }).then(function(oDialog) {
                    this._oDataProfiler = oDialog;
                    oView.addDependent(this._oDataProfiler);
                    this._oDataProfiler.open();
                    this._loadDataProfile();
                }.bind(this));
            } else {
                this._oDataProfiler.open();
                this._loadDataProfile();
            }
        },

        /**
         * Loads comprehensive data profiling information from the backend service.
         * Includes dataset statistics, data quality metrics, and AI readiness recommendations.
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        _loadDataProfile: function() {
            // Show loading
            this._oDataProfiler.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/data-profile",
                type: "GET",
                headers: this._getSecureHeaders(),
                success: function(data) {
                    this._oDataProfiler.setBusy(false);
                    
                    // Create visualization model
                    var oProfileModel = new JSONModel({
                        datasets: data.datasets,
                        statistics: data.statistics,
                        dataQuality: data.dataQuality,
                        recommendations: data.recommendations
                    });
                    
                    this._oDataProfiler.setModel(oProfileModel, "profile");
                    this._createDataVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    this._oDataProfiler.setBusy(false);
                    MessageBox.error("Failed to load data profile: " + xhr.responseText);
                }.bind(this)
            });
        },

        /**
         * Creates interactive data visualizations using SAP VizFrame for data profiling.
         * Generates charts for feature statistics, data quality metrics, and distribution analysis.
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {object} data - Profiling data containing statistics and quality metrics
         * @since 1.0.0
         */
        _createDataVisualizations: function(data) {
            // Create charts for data distribution, quality metrics, etc.
            var oVizFrame = this._oDataProfiler.byId("statisticsChart");
            if (!oVizFrame) return;
            
            // Prepare data for visualization
            var aChartData = [];
            if (data.statistics && data.statistics.features) {
                aChartData = data.statistics.features.map(function(feature) {
                    return {
                        Feature: feature.name,
                        Missing: feature.missing_percent || 0,
                        Unique: feature.cardinality || 0,
                        Mean: feature.mean || 0
                    };
                });
            }
            
            // Create JSON model for chart
            var oChartModel = new sap.ui.model.json.JSONModel({
                chartData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            // Configure chart
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Features" }
                },
                valueAxis: {
                    title: { text: "Percentage / Count" }
                },
                title: {
                    text: "Feature Statistics Overview"
                },
                legend: {
                    visible: true
                }
            });
            
            // Set feeds
            var oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Missing", "Unique"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis",
                type: "Dimension",
                values: ["Feature"]
            });
            
            oVizFrame.removeAllFeeds();
            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);
            
            // Create additional distribution charts in the grid
            this._createDistributionCharts(data);
        },
        
        /**
         * Creates distribution charts for individual features with lazy loading support.
         * Limits the number of charts rendered initially for better performance.
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {object} data - Feature statistics data
         * @since 1.0.0
         */
        _createDistributionCharts: function(data) {
            var oGrid = this._oDataProfiler.byId("distributionGrid");
            if (!oGrid || !data.statistics || !data.statistics.features) return;
            
            // Clear existing content
            oGrid.removeAllContent();
            
            // Create charts for each feature
            data.statistics.features.forEach(function(feature, index) {
                if (index >= 6) return; // Limit to first 6 features for performance
                
                var oPanel = new sap.m.Panel({
                    headerText: feature.name + " Distribution",
                    class: "sapUiMediumMargin"
                });
                
                // Create simple chart based on feature type
                if (feature.type === "NUMERICAL") {
                    var oMicroChart = new sap.suite.ui.microchart.ColumnMicroChart({
                        height: "150px",
                        width: "100%"
                    });
                    
                    // Generate sample distribution data
                    var aDistributionData = this._generateDistributionData(feature);
                    aDistributionData.forEach(function(point) {
                        oMicroChart.addColumn(new sap.suite.ui.microchart.ColumnMicroChartData({
                            value: point.value,
                            color: point.value > feature.mean ? "Good" : "Neutral"
                        }));
                    });
                    
                    oPanel.addContent(oMicroChart);
                } else {
                    // For categorical features, show a simple text summary
                    var oText = new sap.m.Text({
                        text: "Type: " + feature.type + 
                              "\nUnique Values: " + (feature.cardinality || "N/A") +
                              "\nMissing: " + (feature.missing_percent || 0) + "%"
                    });
                    oPanel.addContent(oText);
                }
                
                oGrid.addContent(oPanel);
            }.bind(this));
        },
        
        /**
         * Generates sample distribution data for feature visualization.
         * Creates realistic data points based on feature statistics for chart rendering.
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {object} feature - Feature object containing statistical information
         * @returns {Array<object>} Array of data points for visualization
         * @since 1.0.0
         */
        _generateDistributionData: function(feature) {
            // Generate sample distribution data for visualization
            var aData = [];
            var mean = feature.mean || 0;
            var stdDev = feature.std_dev || 1;
            
            for (var i = 0; i < 10; i++) {
                var value = mean + (Math.random() - 0.5) * stdDev * 4;
                aData.push({ value: Math.max(0, value) });
            }
            
            return aData;
        },

        /**
         * Initializes CSRF token for secure API calls
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        /**
         * Initialize the create model with default values and validation states
         * @private
         * @since 1.0.0
         */
        _initializeCreateModel: function() {
            var oCreateModel = new JSONModel({
                taskName: "",
                description: "",
                datasetName: "",
                modelType: "",
                dataType: "",
                framework: "AUTO",
                splitRatio: 80,
                validationStrategy: "KFOLD",
                featureSelection: true,
                autoFeatureEngineering: true,
                optimizationMetric: "AUTO",
                useGPU: false,
                distributed: false,
                memoryOptimized: true,
                cacheResults: true,
                isValid: false,
                taskNameState: "None",
                taskNameStateText: "",
                datasetNameState: "None",
                datasetNameStateText: "",
                modelTypeState: "None",
                modelTypeStateText: "",
                dataTypeState: "None",
                dataTypeStateText: ""
            });
            this.base.getView().setModel(oCreateModel, "create");
        },

        _initializeCSRFToken: function() {
            jQuery.ajax({
                url: "/a2a/agent2/v1/csrf-token",
                type: "GET",
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data, textStatus, xhr) {
                    this._csrfToken = xhr.getResponseHeader("X-CSRF-Token");
                }.bind(this),
                error: function() {
                    // Fallback to generate token if service not available
                    this._csrfToken = "fetch";
                }.bind(this)
            });
        },

        /**
         * Gets secure headers for AJAX requests including CSRF token
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @returns {object} Security headers object
         * @since 1.0.0
         */
        _getSecureHeaders: function() {
            return {
                "X-CSRF-Token": this._csrfToken || "Fetch",
                "X-Requested-With": "XMLHttpRequest",
                "Content-Security-Policy": "default-src 'self'"
            };
        },

        /**
         * Opens cached dialog fragments with optimized loading for AI workflows
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {string} sDialogKey - Dialog cache key
         * @param {string} sFragmentName - Fragment name to load
         * @param {function} [fnCallback] - Optional callback after opening
         * @since 1.0.0
         */
        _openCachedDialog: function(sDialogKey, sFragmentName, fnCallback) {
            var oView = this.base.getView();
            
            if (!this._dialogCache[sDialogKey]) {
                // Show loading indicator for complex AI dialogs
                oView.setBusy(true);
                
                Fragment.load({
                    id: oView.getId(),
                    name: sFragmentName,
                    controller: this
                }).then(function(oDialog) {
                    this._dialogCache[sDialogKey] = oDialog;
                    oView.addDependent(oDialog);
                    oView.setBusy(false);
                    
                    // Initialize lazy loading for complex visualizations
                    this._initializeLazyLoading(oDialog, sDialogKey);
                    
                    oDialog.open();
                    if (fnCallback) {
                        fnCallback(oDialog);
                    }
                }.bind(this));
            } else {
                this._dialogCache[sDialogKey].open();
                if (fnCallback) {
                    fnCallback(this._dialogCache[sDialogKey]);
                }
            }
        },

        /**
         * Initializes lazy loading for data-intensive visualizations
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.ui.core.Control} oDialog - Dialog instance
         * @param {string} sDialogKey - Dialog identifier
         * @since 1.0.0
         */
        _initializeLazyLoading: function(oDialog, sDialogKey) {
            if (sDialogKey === "dataProfiler") {
                // Initialize intersection observer for lazy chart loading
                var oTabBar = oDialog.byId("profilerTabBar");
                if (oTabBar) {
                    oTabBar.attachSelect(this._onTabSelect.bind(this));
                }
            } else if (sDialogKey === "featureAnalysis") {
                // Initialize progressive loading for feature charts
                this._initializeProgressiveLoading(oDialog);
            }
        },

        /**
         * Handles tab selection for lazy loading of chart content
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.ui.base.Event} oEvent - Tab selection event
         * @since 1.0.0
         */
        _onTabSelect: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("key");
            var oTabBar = oEvent.getSource();
            
            // Load tab content only when selected
            setTimeout(function() {
                if (sSelectedKey === "statistics" && !this._statisticsLoaded) {
                    this._loadStatisticsTab();
                    this._statisticsLoaded = true;
                } else if (sSelectedKey === "distribution" && !this._distributionLoaded) {
                    this._loadDistributionTab();
                    this._distributionLoaded = true;
                } else if (sSelectedKey === "correlations" && !this._correlationsLoaded) {
                    this._loadCorrelationsTab();
                    this._correlationsLoaded = true;
                }
            }.bind(this), 100);
        },

        /**
         * Loads statistics tab content with lazy loading
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        _loadStatisticsTab: function() {
            // Implementation for loading statistics visualizations
            var oDialog = this._dialogCache.dataProfiler;
            if (oDialog) {
                var oVizFrame = oDialog.byId("statisticsChart");
                if (oVizFrame && !oVizFrame.getModel()) {
                    // Load chart data only when tab is active
                    this._createStatisticsVisualization(oVizFrame);
                }
            }
        },

        /**
         * Creates statistics visualization with performance optimization
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.viz.ui5.controls.VizFrame} oVizFrame - Visualization frame
         * @since 1.0.0
         */
        _createStatisticsVisualization: function(oVizFrame) {
            // Enhanced visualization creation with caching
            var aCachedData = this._statisticsData || [];
            
            if (aCachedData.length === 0) {
                // Show loading indicator
                oVizFrame.setBusy(true);
                
                jQuery.ajax({
                    url: "/a2a/agent2/v1/statistics",
                    type: "GET",
                    headers: this._getSecureHeaders(),
                    success: function(data) {
                        this._statisticsData = data.statistics;
                        this._renderStatisticsChart(oVizFrame, data.statistics);
                        oVizFrame.setBusy(false);
                    }.bind(this),
                    error: function() {
                        oVizFrame.setBusy(false);
                        MessageBox.error("Failed to load statistics data");
                    }
                });
            } else {
                this._renderStatisticsChart(oVizFrame, aCachedData);
            }
        },

        /**
         * Renders statistics chart with optimized data binding
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.viz.ui5.controls.VizFrame} oVizFrame - Chart control
         * @param {Array} aStatistics - Statistics data array
         * @since 1.0.0
         */
        _renderStatisticsChart: function(oVizFrame, aStatistics) {
            var oChartModel = new sap.ui.model.json.JSONModel({
                data: aStatistics.slice(0, 50) // Limit for performance
            });
            oVizFrame.setModel(oChartModel);
            
            // Configure chart with accessibility support
            oVizFrame.setVizProperties({
                title: {
                    text: "Feature Statistics Overview",
                    visible: true
                },
                legend: {
                    visible: true,
                    title: {
                        text: "Metrics"
                    }
                },
                categoryAxis: {
                    title: {
                        text: "Features"
                    }
                },
                valueAxis: {
                    title: {
                        text: "Values"
                    }
                }
            });
        },

        /**
         * Opens the AutoML wizard dialog for automated machine learning configuration
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onAutoML: function() {
            this._openCachedDialog("autoMLWizard", "a2a.network.agent2.ext.fragment.AutoMLWizard", function(oDialog) {
                this._initializeAutoMLModel(oDialog);
            }.bind(this));
        },

        /**
         * Opens the feature analysis dialog with correlation and importance analysis
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onFeatureAnalysis: function() {
            this._openCachedDialog("featureAnalysis", "a2a.network.agent2.ext.fragment.FeatureAnalysis", function(oDialog) {
                this._loadFeatureAnalysisData(oDialog);
            }.bind(this));
        },

        /**
         * Initializes AutoML model with default configuration
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.m.Dialog} oDialog - AutoML dialog instance
         * @since 1.0.0
         */
        _initializeAutoMLModel: function(oDialog) {
            var oAutoMLModel = new JSONModel({
                dataset: "",
                problemType: "",
                targetColumn: "",
                trainingRatio: 80,
                validationRatio: 15,
                testRatio: 5,
                maxIterations: 100,
                evaluationMetric: "AUTO",
                hyperparameterTuning: true,
                featureEngineering: true,
                ensembleMethods: false,
                explainability: true
            });
            oDialog.setModel(oAutoMLModel, "automl");
        },

        /**
         * Loads feature analysis data from backend service
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.m.Dialog} oDialog - Feature analysis dialog instance
         * @since 1.0.0
         */
        _loadFeatureAnalysisData: function(oDialog) {
            oDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/feature-analysis",
                type: "GET",
                headers: this._getSecureHeaders(),
                success: function(data) {
                    var oAnalysisModel = new JSONModel({
                        features: data.features || [],
                        correlationMatrix: data.correlationMatrix || [],
                        importanceScores: data.importanceScores || [],
                        recommendations: data.recommendations || []
                    });
                    oDialog.setModel(oAnalysisModel, "analysis");
                    this._createFeatureVisualization(oDialog, data);
                    oDialog.setBusy(false);
                }.bind(this),
                error: function() {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load feature analysis data");
                }
            });
        },

        /**
         * Creates feature visualization charts with performance optimization
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.m.Dialog} oDialog - Dialog containing visualization
         * @param {object} data - Feature analysis data
         * @since 1.0.0
         */
        _createFeatureVisualization: function(oDialog, data) {
            var oFeatureChart = oDialog.byId("featureImportanceChart");
            if (!oFeatureChart || !data.importanceScores) return;

            // Create chart model with limited data for performance
            var aChartData = data.importanceScores.slice(0, 20).map(function(item) {
                return {
                    Feature: item.name,
                    Importance: item.score * 100,
                    Type: item.type
                };
            });

            var oChartModel = new JSONModel({
                data: aChartData
            });
            oFeatureChart.setModel(oChartModel);

            // Configure chart properties
            this._configureFeatureChart(oFeatureChart);
        },

        /**
         * Configures feature importance chart with accessibility support
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.viz.ui5.controls.VizFrame} oChart - Chart control
         * @since 1.0.0
         */
        _configureFeatureChart: function(oChart) {
            oChart.setVizProperties({
                title: {
                    text: "Feature Importance Analysis",
                    visible: true
                },
                legend: {
                    visible: true,
                    title: {
                        text: "Feature Types"
                    }
                },
                categoryAxis: {
                    title: {
                        text: "Features"
                    }
                },
                valueAxis: {
                    title: {
                        text: "Importance Score (%)"
                    }
                },
                interaction: {
                    selectability: {
                        mode: "single"
                    }
                }
            });

            // Set up chart feeds
            var oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Importance"]
            });
            var oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "categoryAxis",
                type: "Dimension",
                values: ["Feature"]
            });

            oChart.removeAllFeeds();
            oChart.addFeed(oFeedValueAxis);
            oChart.addFeed(oFeedCategoryAxis);
        },

        /**
         * Wizard step activation handler with accessibility announcements
         * @param {sap.ui.base.Event} oEvent - Step activation event
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onWizardStepActivate: function(oEvent) {
            var oStep = oEvent.getParameter("step");
            var sStepTitle = oStep.getTitle();
            var iStepNumber = oEvent.getParameter("index") + 1;
            
            // Announce step change to screen readers
            sap.ui.getCore().announceForAccessibility(
                "Now on step " + iStepNumber + " of 5: " + sStepTitle
            );
        },

        /**
         * Profiler tab selection handler with lazy loading
         * @param {sap.ui.base.Event} oEvent - Tab selection event
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onProfilerTabSelect: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedKey");
            
            // Announce tab change to screen readers
            sap.ui.getCore().announceForAccessibility("Selected tab: " + sSelectedKey);
            
            // Trigger lazy loading based on tab selection
            this._onTabSelect(oEvent);
        },

        /**
         * Utility function for secure AJAX requests with error handling
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {object} oOptions - AJAX request options
         * @returns {Promise} jQuery AJAX promise
         * @since 1.0.0
         */
        _makeSecureRequest: function(oOptions) {
            var oRequestOptions = jQuery.extend({
                headers: this._getSecureHeaders(),
                timeout: 30000
            }, oOptions);

            return jQuery.ajax(oRequestOptions)
                .fail(function(xhr, status, error) {
                    var sErrorMessage = "Request failed";
                    try {
                        var oErrorData = JSON.parse(xhr.responseText);
                        sErrorMessage = oErrorData.message || sErrorMessage;
                    } catch (e) {
                        sErrorMessage = xhr.responseText || error || sErrorMessage;
                    }
                    MessageBox.error(sErrorMessage);
                });
        }

    });
});
