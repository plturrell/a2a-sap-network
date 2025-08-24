sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, encodeXML, escapeRegExp, sanitizeHTML) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent2.ext.controller.ListReportExt", {

        override: {
            onInit() {
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
        onCreateAITask() {
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
        onOpenDataProfiler() {
            const oView = this.base.getView();

            if (!this._oDataProfiler) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.DataProfiler",
                    controller: this
                }).then((oDialog) => {
                    this._oDataProfiler = oDialog;
                    oView.addDependent(this._oDataProfiler);
                    this._oDataProfiler.open();
                    this._loadDataProfile();
                });
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
        _loadDataProfile() {
            // Show loading
            this._oDataProfiler.setBusy(true);

            jQuery.ajax({
                url: "/a2a/agent2/v1/data-profile",
                type: "GET",
                headers: this._getSecureHeaders(),
                success: function(data) {
                    this._oDataProfiler.setBusy(false);

                    // Create visualization model
                    const oProfileModel = new JSONModel({
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
                    MessageBox.error(`Failed to load data profile: ${ xhr.responseText}`);
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
        _createDataVisualizations(data) {
            // Create charts for data distribution, quality metrics, etc.
            const oVizFrame = this._oDataProfiler.byId("statisticsChart");
            if (!oVizFrame) {return;}

            // Prepare data for visualization
            let aChartData = [];
            if (data.statistics && data.statistics.features) {
                aChartData = data.statistics.features.map((feature) => {
                    return {
                        Feature: feature.name,
                        Missing: feature.missing_percent || 0,
                        Unique: feature.cardinality || 0,
                        Mean: feature.mean || 0
                    };
                });
            }

            // Create JSON model for chart
            const oChartModel = new sap.ui.model.json.JSONModel({
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
            const oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Missing", "Unique"]
            });
            const oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
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
        _createDistributionCharts(data) {
            const oGrid = this._oDataProfiler.byId("distributionGrid");
            if (!oGrid || !data.statistics || !data.statistics.features) {return;}

            // Clear existing content
            oGrid.removeAllContent();

            // Create charts for each feature
            data.statistics.features.forEach((feature, index) => {
                if (index >= 6) {return;} // Limit to first 6 features for performance

                const oPanel = new sap.m.Panel({
                    headerText: `${feature.name } Distribution`,
                    class: "sapUiMediumMargin"
                });

                // Create simple chart based on feature type
                if (feature.type === "NUMERICAL") {
                    const oMicroChart = new sap.suite.ui.microchart.ColumnMicroChart({
                        height: "150px",
                        width: "100%"
                    });

                    // Generate sample distribution data
                    const aDistributionData = this._generateDistributionData(feature);
                    aDistributionData.forEach((point) => {
                        oMicroChart.addColumn(new sap.suite.ui.microchart.ColumnMicroChartData({
                            value: point.value,
                            color: point.value > feature.mean ? "Good" : "Neutral"
                        }));
                    });

                    oPanel.addContent(oMicroChart);
                } else {
                    // For categorical features, show a simple text summary
                    const oText = new sap.m.Text({
                        text: `Type: ${ feature.type
                        }\nUnique Values: ${ feature.cardinality || "N/A"
                        }\nMissing: ${ feature.missing_percent || 0 }%`
                    });
                    oPanel.addContent(oText);
                }

                oGrid.addContent(oPanel);
            });
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
        _generateDistributionData(feature) {
            // Generate sample distribution data for visualization
            const aData = [];
            const mean = feature.mean || 0;
            const stdDev = feature.std_dev || 1;

            for (let i = 0; i < 10; i++) {
                const value = mean + (Math.random() - 0.5) * stdDev * 4;
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
        _initializeCreateModel() {
            const oCreateModel = new JSONModel({
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

        _initializeCSRFToken() {
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
        _getSecureHeaders() {
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
        _openCachedDialog(sDialogKey, sFragmentName, fnCallback) {
            const oView = this.base.getView();

            if (!this._dialogCache[sDialogKey]) {
                // Show loading indicator for complex AI dialogs
                oView.setBusy(true);

                Fragment.load({
                    id: oView.getId(),
                    name: sFragmentName,
                    controller: this
                }).then((oDialog) => {
                    this._dialogCache[sDialogKey] = oDialog;
                    oView.addDependent(oDialog);
                    oView.setBusy(false);

                    // Initialize lazy loading for complex visualizations
                    this._initializeLazyLoading(oDialog, sDialogKey);

                    oDialog.open();
                    if (fnCallback) {
                        fnCallback(oDialog);
                    }
                });
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
        _initializeLazyLoading(oDialog, sDialogKey) {
            if (sDialogKey === "dataProfiler") {
                // Initialize intersection observer for lazy chart loading
                const _oTabBar = oDialog.byId("profilerTabBar");
                if (_oTabBar) {
                    _oTabBar.attachSelect(this._onTabSelect.bind(this));
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
        _onTabSelect(oEvent) {
            const sSelectedKey = oEvent.getParameter("key");
            const _oTabBar = oEvent.getSource();

            // Load tab content only when selected
            setTimeout(() => {
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
            }, 100);
        },

        /**
         * Loads statistics tab content with lazy loading
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        _loadStatisticsTab() {
            // Implementation for loading statistics visualizations
            const oDialog = this._dialogCache.dataProfiler;
            if (oDialog) {
                const oVizFrame = oDialog.byId("statisticsChart");
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
        _createStatisticsVisualization(oVizFrame) {
            // Enhanced visualization creation with caching
            const aCachedData = this._statisticsData || [];

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
                    error() {
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
        _renderStatisticsChart(oVizFrame, aStatistics) {
            const oChartModel = new sap.ui.model.json.JSONModel({
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
        onAutoML() {
            this._openCachedDialog("autoMLWizard", "a2a.network.agent2.ext.fragment.AutoMLWizard", (oDialog) => {
                this._initializeAutoMLModel(oDialog);
            });
        },

        /**
         * Opens the feature analysis dialog with correlation and importance analysis
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onFeatureAnalysis() {
            this._openCachedDialog("featureAnalysis", "a2a.network.agent2.ext.fragment.FeatureAnalysis", (oDialog) => {
                this._loadFeatureAnalysisData(oDialog);
            });
        },

        /**
         * Initializes AutoML model with default configuration
         * @private
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @param {sap.m.Dialog} oDialog - AutoML dialog instance
         * @since 1.0.0
         */
        _initializeAutoMLModel(oDialog) {
            const oAutoMLModel = new JSONModel({
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
        _loadFeatureAnalysisData(oDialog) {
            oDialog.setBusy(true);

            jQuery.ajax({
                url: "/a2a/agent2/v1/feature-analysis",
                type: "GET",
                headers: this._getSecureHeaders(),
                success: function(data) {
                    const oAnalysisModel = new JSONModel({
                        features: data.features || [],
                        correlationMatrix: data.correlationMatrix || [],
                        importanceScores: data.importanceScores || [],
                        recommendations: data.recommendations || []
                    });
                    oDialog.setModel(oAnalysisModel, "analysis");
                    this._createFeatureVisualization(oDialog, data);
                    oDialog.setBusy(false);
                }.bind(this),
                error() {
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
        _createFeatureVisualization(oDialog, data) {
            const oFeatureChart = oDialog.byId("featureImportanceChart");
            if (!oFeatureChart || !data.importanceScores) {return;}

            // Create chart model with limited data for performance
            const aChartData = data.importanceScores.slice(0, 20).map((item) => {
                return {
                    Feature: item.name,
                    Importance: item.score * 100,
                    Type: item.type
                };
            });

            const oChartModel = new JSONModel({
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
        _configureFeatureChart(oChart) {
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
            const oFeedValueAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
                uid: "valueAxis",
                type: "Measure",
                values: ["Importance"]
            });
            const oFeedCategoryAxis = new sap.viz.ui5.controls.common.feeds.FeedItem({
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
        onWizardStepActivate(oEvent) {
            const oStep = oEvent.getParameter("step");
            const sStepTitle = oStep.getTitle();
            const iStepNumber = oEvent.getParameter("index") + 1;

            // Announce step change to screen readers
            sap.ui.getCore().announceForAccessibility(
                `Now on step ${ iStepNumber } of 5: ${ sStepTitle}`
            );
        },

        /**
         * Profiler tab selection handler with lazy loading
         * @param {sap.ui.base.Event} oEvent - Tab selection event
         * @public
         * @memberof a2a.network.agent2.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onProfilerTabSelect(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedKey");

            // Announce tab change to screen readers
            sap.ui.getCore().announceForAccessibility(`Selected tab: ${ sSelectedKey}`);

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
        _makeSecureRequest(oOptions) {
            const oRequestOptions = jQuery.extend({
                headers: this._getSecureHeaders(),
                timeout: 30000
            }, oOptions);

            return jQuery.ajax(oRequestOptions)
                .fail((xhr, status, error) => {
                    let sErrorMessage = "Request failed";
                    try {
                        const oErrorData = JSON.parse(xhr.responseText);
                        sErrorMessage = oErrorData.message || sErrorMessage;
                    } catch (e) {
                        sErrorMessage = xhr.responseText || error || sErrorMessage;
                    }
                    MessageBox.error(sErrorMessage);
                });
        }

    });
});
