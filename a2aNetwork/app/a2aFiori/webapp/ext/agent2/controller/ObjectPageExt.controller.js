sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp"
], (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent2.ext.controller.ObjectPageExt", {

        override: {
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();

                // Initialize device model for responsive behavior
                const oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");

                // Initialize dialog cache for better performance
                this._dialogCache = {};

                // Initialize create model
                this._initializeCreateModel();
            }
        },

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

        onStartPreparation() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            MessageBox.confirm(`Start AI data preparation for '${ sTaskName }'?`, {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startPreparationProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startPreparationProcess(sTaskId) {
            this._extensionAPI.getView().setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent2/v1/tasks/${ sTaskId }/prepare`,
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("AI data preparation started");
                    this._extensionAPI.refresh();

                    // Start real-time monitoring
                    this._startRealtimeMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error(`Failed to start preparation: ${ xhr.responseText}`);
                }.bind(this)
            });
        },

        _startRealtimeMonitoring(sTaskId) {
            // Use EventSource for real-time updates
            // Create secure EventSource with authentication
            const sToken = this._getAuthToken();
            this._eventSource = this._createSecureEventSource(`/a2a/agent2/v1/tasks/${ sTaskId }/stream`, sToken);

            this._eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);

                // Update progress in UI
                if (data.type === "progress") {
                    this._updateProgress(data);
                } else if (data.type === "complete") {
                    this._eventSource.close();
                    this._extensionAPI.refresh();
                    MessageBox.success("AI data preparation completed successfully!");
                } else if (data.type === "error") {
                    this._eventSource.close();
                    MessageBox.error(`Preparation failed: ${ data.error}`);
                }
            }.bind(this);

            this._eventSource.onerror = function() {
                this._eventSource.close();
                MessageBox.error("Lost connection to preparation process");
            }.bind(this);
        },

        _updateProgress(data) {
            // Update progress indicators in the UI
            MessageToast.show(`${data.stage }: ${ data.progress }%`);
        },

        onAnalyzeFeatures() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

            this._extensionAPI.getView().setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent2/v1/tasks/${ sTaskId }/analyze-features`,
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showFeatureAnalysis(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error(`Feature analysis failed: ${ xhr.responseText}`);
                }.bind(this)
            });
        },

        _showFeatureAnalysis(analysisData) {
            const oView = this.base.getView();

            if (!this._oFeatureAnalysisDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.FeatureAnalysis",
                    controller: this
                }).then((oDialog) => {
                    this._oFeatureAnalysisDialog = oDialog;
                    oView.addDependent(this._oFeatureAnalysisDialog);

                    const oModel = new JSONModel(analysisData);
                    this._oFeatureAnalysisDialog.setModel(oModel, "analysis");
                    this._oFeatureAnalysisDialog.open();

                    // Create visualizations
                    this._createFeatureVisualizations(analysisData);
                });
            } else {
                const oModel = new JSONModel(analysisData);
                this._oFeatureAnalysisDialog.setModel(oModel, "analysis");
                this._oFeatureAnalysisDialog.open();
                this._createFeatureVisualizations(analysisData);
            }
        },

        _createFeatureVisualizations(data) {
            // Create feature importance chart, correlation matrix, etc.
            if (!data.features || !this._oFeatureAnalysisDialog) {return;}

            // Create feature importance chart
            this._createFeatureImportanceChart(data.features);

            // Create correlation matrix visualization
            if (data.correlation_matrix) {
                this._createCorrelationMatrix(data.correlation_matrix);
            }

            // Create feature distribution charts
            this._createFeatureDistributionCharts(data.features);
        },

        _createFeatureImportanceChart(features) {
            // Sort features by importance
            const aSortedFeatures = features.slice().sort((a, b) => {
                return (b.importance || 0) - (a.importance || 0);
            }).slice(0, 10); // Top 10 features

            // Prepare chart data
            const aChartData = aSortedFeatures.map((feature) => {
                return {
                    Feature: feature.name,
                    Importance: (feature.importance || 0) * 100,
                    Type: feature.type || "Unknown"
                };
            });

            // Create chart model
            const oChartModel = new sap.ui.model.json.JSONModel({
                importanceData: aChartData
            });

            // Find or create chart container
            const oDialog = this._oFeatureAnalysisDialog;
            const oChartContainer = new sap.m.Panel({
                headerText: "Feature Importance (Top 10)",
                class: "sapUiMediumMargin"
            });

            // Create horizontal bar chart using VizFrame
            const oVizFrame = new sap.viz.ui5.controls.VizFrame({
                height: "400px",
                width: "100%",
                vizType: "horizontal_bar"
            });

            oVizFrame.setModel(oChartModel);

            // Configure chart properties
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Features" }
                },
                valueAxis: {
                    title: { text: "Importance %" }
                },
                title: {
                    text: "Feature Importance Analysis"
                },
                legend: {
                    visible: false
                }
            });

            // Set data binding
            const oDataset = new sap.viz.ui5.data.FlattenedDataset({
                dimensions: [{
                    name: "Feature",
                    value: "{Feature}"
                }],
                measures: [{
                    name: "Importance",
                    value: "{Importance}"
                }],
                data: {
                    path: "/importanceData"
                }
            });
            oVizFrame.setDataset(oDataset);

            // Set feeds
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

            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);

            oChartContainer.addContent(oVizFrame);

            // Add to dialog content
            const oContent = oDialog.getContent()[0]; // Should be the container
            if (oContent && oContent.addContent) {
                oContent.addContent(oChartContainer);
            }
        },

        _createCorrelationMatrix(correlationMatrix) {
            if (!correlationMatrix || !Array.isArray(correlationMatrix)) {return;}

            const oPanel = new sap.m.Panel({
                headerText: "Feature Correlation Matrix",
                class: "sapUiMediumMargin"
            });

            // Create a simple grid representation of correlation matrix
            const oGrid = new sap.ui.layout.Grid({
                defaultSpan: "XL2 L2 M3 S6"
            });

            correlationMatrix.forEach((row, i) => {
                if (i >= 6) {return;} // Limit for display
                row.forEach((correlation, j) => {
                    if (j >= 6) {return;} // Limit for display

                    let sState = "None";
                    if (Math.abs(correlation) > 0.7) {sState = "Success";}
                    else if (Math.abs(correlation) > 0.5) {sState = "Warning";}
                    else if (Math.abs(correlation) > 0.3) {sState = "Information";}

                    const oTile = new sap.m.GenericTile({
                        class: "sapUiTinyMargin",
                        header: correlation.toFixed(2),
                        subheader: `(${ i },${ j })`,
                        state: sState,
                        press() {
                            sap.m.MessageToast.show(`Correlation: ${ correlation.toFixed(3)}`);
                        }
                    });

                    oGrid.addContent(oTile);
                });
            });

            oPanel.addContent(oGrid);

            // Add to dialog
            const oDialog = this._oFeatureAnalysisDialog;
            const oContent = oDialog.getContent()[0];
            if (oContent && oContent.addContent) {
                oContent.addContent(oPanel);
            }
        },

        _createFeatureDistributionCharts(features) {
            const oPanel = new sap.m.Panel({
                headerText: "Feature Distributions",
                class: "sapUiMediumMargin"
            });

            const oGrid = new sap.ui.layout.Grid({
                defaultSpan: "XL4 L4 M6 S12"
            });

            // Create distribution charts for numerical features
            features.slice(0, 8).forEach((feature) => {
                if (feature.type !== "NUMERICAL" && feature.type !== "numerical") {return;}

                const oFeaturePanel = new sap.m.Panel({
                    headerText: feature.name,
                    class: "sapUiTinyMargin"
                });

                // Create micro chart for distribution
                const oMicroChart = new sap.suite.ui.microchart.AreaMicroChart({
                    height: "100px",
                    width: "100%"
                });

                // Generate sample distribution data
                const aPoints = [];
                const mean = feature.mean_value || 0;
                const stdDev = feature.std_dev || 1;

                for (let i = 0; i < 20; i++) {
                    const x = (i - 10) / 2;
                    const y = Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
                    aPoints.push(new sap.suite.ui.microchart.AreaMicroChartPoint({
                        x: i,
                        y: y * 100
                    }));
                }

                aPoints.forEach((point) => {
                    oMicroChart.addPoint(point);
                });

                // Add statistics text
                const oStatsText = new sap.m.Text({
                    text: `Mean: ${ (feature.mean_value || 0).toFixed(2)
                    }, Std: ${ (feature.std_dev || 0).toFixed(2)
                    }, Missing: ${ (feature.missing_percent || 0).toFixed(1) }%`
                });

                oFeaturePanel.addContent(oMicroChart);
                oFeaturePanel.addContent(oStatsText);
                oGrid.addContent(oFeaturePanel);
            });

            oPanel.addContent(oGrid);

            // Add to dialog
            const oDialog = this._oFeatureAnalysisDialog;
            const oContent = oDialog.getContent()[0];
            if (oContent && oContent.addContent) {
                oContent.addContent(oPanel);
            }
        },

        onGenerateEmbeddings() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sModelType = oContext.getProperty("modelType");

            if (!this._oEmbeddingDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent2.ext.fragment.EmbeddingConfiguration",
                    controller: this
                }).then((oDialog) => {
                    this._oEmbeddingDialog = oDialog;
                    this.base.getView().addDependent(this._oEmbeddingDialog);

                    const oModel = new JSONModel({
                        taskId: sTaskId,
                        embeddingModel: sModelType === "LLM" ? "text-embedding-ada-002" : "custom",
                        dimensions: 768,
                        normalization: true,
                        batchSize: 32,
                        useGPU: true
                    });
                    this._oEmbeddingDialog.setModel(oModel, "embedding");
                    this._oEmbeddingDialog.open();
                });
            } else {
                this._oEmbeddingDialog.open();
            }
        },

        onConfirmGenerateEmbeddings() {
            const oModel = this._oEmbeddingDialog.getModel("embedding");
            const oData = oModel.getData();

            this._oEmbeddingDialog.setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent2/v1/tasks/${ oData.taskId }/generate-embeddings`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    model: oData.embeddingModel,
                    dimensions: oData.dimensions,
                    normalization: oData.normalization,
                    batchSize: oData.batchSize,
                    useGPU: oData.useGPU
                }),
                success: function(data) {
                    this._oEmbeddingDialog.setBusy(false);
                    this._oEmbeddingDialog.close();

                    MessageBox.success(
                        "Embeddings generated successfully!\n" +
                        `Vectors: ${ data.vectorCount }\n` +
                        `Dimensions: ${ data.dimensions }\n` +
                        `Processing time: ${ data.processingTime }s`
                    );

                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oEmbeddingDialog.setBusy(false);
                    MessageBox.error(`Failed to generate embeddings: ${ xhr.responseText}`);
                }.bind(this)
            });
        },

        onExportPreparedData() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            if (!this._oExportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent2.ext.fragment.ExportPreparedData",
                    controller: this
                }).then((oDialog) => {
                    this._oExportDialog = oDialog;
                    this.base.getView().addDependent(this._oExportDialog);

                    const oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        format: "TENSORFLOW",
                        includeMetadata: true,
                        splitData: true,
                        compression: "GZIP"
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                });
            } else {
                this._oExportDialog.open();
            }
        },

        onExecuteExport() {
            const oModel = this._oExportDialog.getModel("export");
            const oData = oModel.getData();

            this._oExportDialog.setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent2/v1/tasks/${ oData.taskId }/export`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    format: oData.format,
                    includeMetadata: oData.includeMetadata,
                    splitData: oData.splitData,
                    compression: oData.compression
                }),
                success: function(data) {
                    this._oExportDialog.setBusy(false);
                    this._oExportDialog.close();

                    // Provide download links
                    this._showDownloadLinks(data.files);
                }.bind(this),
                error: function(xhr) {
                    this._oExportDialog.setBusy(false);
                    MessageBox.error(`Export failed: ${ xhr.responseText}`);
                }.bind(this)
            });
        },

        _showDownloadLinks(files) {
            let sMessage = "Export completed! Download files:\n\n";
            files.forEach((file) => {
                sMessage += `${file.name } (${ file.size })\n`;
            });

            MessageBox.information(sMessage, {
                actions: ["Download All", MessageBox.Action.CLOSE],
                onClose(oAction) {
                    if (oAction === "Download All") {
                        files.forEach((file) => {
                            window.open(file.url, "_blank");
                        });
                    }
                }
            });
        },

        onOptimizeHyperparameters() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

            MessageBox.confirm(
                "Start hyperparameter optimization? This may take several hours.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startHyperparameterOptimization(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _startHyperparameterOptimization(sTaskId) {
            jQuery.ajax({
                url: `/a2a/agent2/v1/tasks/${ sTaskId }/optimize`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    method: "BAYESIAN",
                    trials: 100,
                    timeout: 3600,
                    earlyStop: true
                }),
                success: function(data) {
                    MessageBox.success(
                        "Optimization started!\n" +
                        `Job ID: ${ data.jobId }\n` +
                        "You will be notified when complete."
                    );
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Failed to start optimization: ${ errorMsg}`);
                }.bind(this)
            });
        },

        /**
         * Get authentication token for secure API calls
         * @returns {string} Authentication token
         */
        _getAuthToken() {
            // Implementation to get auth token from session or user context
            const sUserId = sap.ushell?.Container?.getUser?.()?.getId?.();
            return sUserId || this._generateSessionToken();
        },

        /**
         * Generate a session-based token
         * @returns {string} Session token
         */
        _generateSessionToken() {
            return `session_${ Date.now() }_${ Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Validate user input for security and format compliance
         * @param {string} sInput - Input to validate
         * @param {string} sType - Type of validation
         * @returns {object} Validation result
         */
        _validateInput(sInput, sType) {
            if (!sInput || typeof sInput !== "string") {
                return { isValid: false, message: "Input is required" };
            }

            const sSanitized = sInput.trim();

            // Check for XSS patterns
            const aXSSPatterns = [
                /<script/i,
                /javascript:/i,
                /on\w+\s*=/i,
                /<iframe/i,
                /<object/i,
                /<embed/i,
                /eval\s*\(/i,
                /Function\s*\(/i
            ];

            for (let i = 0; i < aXSSPatterns.length; i++) {
                if (aXSSPatterns[i].test(sSanitized)) {
                    return { isValid: false, message: "Invalid characters detected" };
                }
            }

            // Type-specific validation
            switch (sType) {
            case "modelConfig":
                if (sSanitized.length > 5000) {
                    return { isValid: false, message: "Configuration too large" };
                }
                break;

            case "featureName":
                if (!/^[a-zA-Z0-9_\-\.]+$/.test(sSanitized)) {
                    return { isValid: false, message: "Invalid feature name format" };
                }
                break;

            default:
                if (sSanitized.length > 10000) {
                    return { isValid: false, message: "Input too long" };
                }
                break;
            }

            return { isValid: true, sanitized: encodeXML(sSanitized) };
        },

        /**
         * Sanitize error messages for user display
         * @param {string} sMessage - Error message
         * @returns {string} Sanitized message
         */
        _sanitizeErrorMessage(sMessage) {
            if (!sMessage) {
                return "An error occurred";
            }

            // Remove sensitive information
            const sSanitized = sMessage
                .replace(/\b(?:token|key|secret|password|auth)\b[:\s]*[^\s]+/gi, "[REDACTED]")
                .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[IP_ADDRESS]")
                .replace(/file:\/\/[^\s]+/g, "[FILE_PATH]")
                .replace(/https?:\/\/[^\s]+/g, "[URL]");

            return encodeXML(sSanitized);
        },

        /**
         * Secure EventSource creation with authentication
         * @param {string} sUrl - EventSource URL
         * @param {string} sToken - Authentication token
         * @returns {EventSource} Secure EventSource
         */
        _createSecureEventSource(sUrl, sToken) {
            if (!sUrl || !sToken) {
                throw new Error("URL and token required for secure EventSource");
            }

            // Validate URL
            try {
                const oUrl = new URL(sUrl, window.location.origin);
                if (!["http:", "https:"].includes(oUrl.protocol)) {
                    throw new Error("Invalid protocol for EventSource");
                }
            } catch (e) {
                throw new Error("Invalid URL for EventSource");
            }

            // Add authentication to URL
            const sAuthParam = "auth_token";
            const sSecureUrl = `${sUrl + (sUrl.includes("?") ? "&" : "?") + sAuthParam }=${ encodeURIComponent(sToken)}`;

            return new EventSource(sSecureUrl);
        },

        /**
         * Secure text formatter to prevent XSS
         * @param {string} sText - Text to format
         * @returns {string} Encoded text
         */
        formatSecureText(sText) {
            if (!sText) {
                return "";
            }
            return encodeXML(sText.toString());
        },

        /**
         * Enhanced task name change handler with real-time validation
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onTaskNameChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oCreateModel = this.base.getView().getModel("create");

            if (!sValue || sValue.trim().length < 3) {
                oCreateModel.setProperty("/taskNameState", "Error");
                oCreateModel.setProperty("/taskNameStateText", "Task name must be at least 3 characters");
            } else if (sValue.length > 100) {
                oCreateModel.setProperty("/taskNameState", "Error");
                oCreateModel.setProperty("/taskNameStateText", "Task name must not exceed 100 characters");
            } else if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sValue)) {
                oCreateModel.setProperty("/taskNameState", "Error");
                oCreateModel.setProperty("/taskNameStateText", "Task name contains invalid characters");
            } else {
                oCreateModel.setProperty("/taskNameState", "Success");
                oCreateModel.setProperty("/taskNameStateText", "Valid task name");
            }

            this._validateForm();
        },

        /**
         * Event handler for model type selection changes
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onModelTypeChange(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const oCreateModel = this.base.getView().getModel("create");

            if (sSelectedKey) {
                oCreateModel.setProperty("/modelTypeState", "None");
                oCreateModel.setProperty("/modelTypeStateText", "");

                // Auto-suggest optimization metric based on model type
                this._suggestOptimizationMetric(sSelectedKey, oCreateModel);
            } else {
                oCreateModel.setProperty("/modelTypeState", "Error");
                oCreateModel.setProperty("/modelTypeStateText", "Please select a model type");
            }

            this._validateForm();
        },

        /**
         * Event handler for data type selection changes
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onDataTypeChange(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const oCreateModel = this.base.getView().getModel("create");

            if (sSelectedKey) {
                oCreateModel.setProperty("/dataTypeState", "None");
                oCreateModel.setProperty("/dataTypeStateText", "");

                // Auto-suggest framework based on data type
                this._suggestFramework(sSelectedKey, oCreateModel);
            } else {
                oCreateModel.setProperty("/dataTypeState", "Error");
                oCreateModel.setProperty("/dataTypeStateText", "Please select a data type");
            }

            this._validateForm();
        },

        /**
         * Suggests optimization metric based on model type
         * @param {string} sModelType - Selected model type
         * @param {sap.ui.model.json.JSONModel} oModel - Create model
         * @private
         * @since 1.0.0
         */
        _suggestOptimizationMetric(sModelType, oModel) {
            const mMetricSuggestions = {
                "CLASSIFICATION": "F1",
                "REGRESSION": "MAE",
                "CLUSTERING": "SILHOUETTE",
                "EMBEDDING": "COSINE_SIMILARITY",
                "LLM": "PERPLEXITY",
                "TIME_SERIES": "MAE",
                "RECOMMENDATION": "AUC",
                "ANOMALY": "AUC"
            };

            const sSuggestedMetric = mMetricSuggestions[sModelType] || "AUTO";
            oModel.setProperty("/optimizationMetric", sSuggestedMetric);
        },

        /**
         * Suggests framework based on data type
         * @param {string} sDataType - Selected data type
         * @param {sap.ui.model.json.JSONModel} oModel - Create model
         * @private
         * @since 1.0.0
         */
        _suggestFramework(sDataType, oModel) {
            const mFrameworkSuggestions = {
                "TABULAR": "SCIKIT_LEARN",
                "TEXT": "HUGGINGFACE",
                "IMAGE": "PYTORCH",
                "AUDIO": "PYTORCH",
                "VIDEO": "PYTORCH",
                "TIME_SERIES": "TENSORFLOW",
                "GRAPH": "PYTORCH"
            };

            const sSuggestedFramework = mFrameworkSuggestions[sDataType] || "AUTO";
            oModel.setProperty("/framework", sSuggestedFramework);
        },

        /**
         * Dataset selection value help handler
         * @public
         * @since 1.0.0
         */
        onSelectDataset() {
            MessageToast.show("Opening dataset browser...");
            // Implementation for dataset selection dialog
            this._openDatasetBrowser();
        },

        /**
         * Opens dataset browser dialog
         * @private
         * @since 1.0.0
         */
        _openDatasetBrowser() {
            if (!this._oDatasetDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent2.ext.fragment.DatasetBrowser",
                    controller: this
                }).then((oDialog) => {
                    this._oDatasetDialog = oDialog;
                    this.base.getView().addDependent(this._oDatasetDialog);
                    this._loadAvailableDatasets();
                    this._oDatasetDialog.open();
                }).catch(() => {
                    // Fallback if fragment doesn't exist
                    MessageBox.information("Dataset browser not yet implemented. Please enter dataset name manually.");
                });
            } else {
                this._oDatasetDialog.open();
            }
        },

        /**
         * Loads available datasets for selection
         * @private
         * @since 1.0.0
         */
        _loadAvailableDatasets() {
            // Implementation for loading datasets
            MessageToast.show("Loading available datasets...");
        },

        /**
         * Dialog after open event handler
         * @public
         * @since 1.0.0
         */
        onDialogAfterOpen() {
            // Focus on first input field for accessibility
            const oDialog = this.base.getView().byId("createAITaskDialog");
            if (oDialog) {
                const oFirstInput = oDialog.byId("aiTaskNameInput");
                if (oFirstInput) {
                    setTimeout(() => {
                        oFirstInput.focus();
                    }, 100);
                }
            }
        },

        /**
         * Dialog after close event handler
         * @public
         * @since 1.0.0
         */
        onDialogAfterClose() {
            // Reset form when dialog closes
            this._initializeCreateModel();
        },

        /**
         * Validates the entire form and updates the isValid flag
         * @private
         * @since 1.0.0
         */
        _validateForm() {
            const oCreateModel = this.base.getView().getModel("create");
            const oData = oCreateModel.getData();

            const bIsValid = oData.taskName &&
                          oData.taskName.trim().length >= 3 &&
                          oData.datasetName &&
                          oData.modelType &&
                          oData.dataType &&
                          oData.taskNameState !== "Error" &&
                          oData.datasetNameState !== "Error" &&
                          oData.modelTypeState !== "Error" &&
                          oData.dataTypeState !== "Error";

            oCreateModel.setProperty("/isValid", bIsValid);
        },

        /**
         * Cancel create dialog handler
         * @public
         * @since 1.0.0
         */
        onCancelCreate() {
            this._getCreateDialog().close();
        },

        /**
         * Get or create the create dialog
         * @private
         * @returns {sap.m.Dialog} The create dialog
         * @since 1.0.0
         */
        _getCreateDialog() {
            return this.base.getView().byId("createAITaskDialog");
        }
    });
});