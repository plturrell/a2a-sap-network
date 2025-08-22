sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent2.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartPreparation: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start AI data preparation for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startPreparationProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startPreparationProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks/" + sTaskId + "/prepare",
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
                    MessageBox.error("Failed to start preparation: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startRealtimeMonitoring: function(sTaskId) {
            // Use EventSource for real-time updates
            // Create secure EventSource with authentication
            var sToken = this._getAuthToken();
            this._eventSource = this._createSecureEventSource("/a2a/agent2/v1/tasks/" + sTaskId + "/stream", sToken);
            
            this._eventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                // Update progress in UI
                if (data.type === "progress") {
                    this._updateProgress(data);
                } else if (data.type === "complete") {
                    this._eventSource.close();
                    this._extensionAPI.refresh();
                    MessageBox.success("AI data preparation completed successfully!");
                } else if (data.type === "error") {
                    this._eventSource.close();
                    MessageBox.error("Preparation failed: " + data.error);
                }
            }.bind(this);
            
            this._eventSource.onerror = function() {
                this._eventSource.close();
                MessageBox.error("Lost connection to preparation process");
            }.bind(this);
        },

        _updateProgress: function(data) {
            // Update progress indicators in the UI
            MessageToast.show(data.stage + ": " + data.progress + "%");
        },

        onAnalyzeFeatures: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks/" + sTaskId + "/analyze-features",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showFeatureAnalysis(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Feature analysis failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showFeatureAnalysis: function(analysisData) {
            var oView = this.base.getView();
            
            if (!this._oFeatureAnalysisDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.FeatureAnalysis",
                    controller: this
                }).then(function(oDialog) {
                    this._oFeatureAnalysisDialog = oDialog;
                    oView.addDependent(this._oFeatureAnalysisDialog);
                    
                    var oModel = new JSONModel(analysisData);
                    this._oFeatureAnalysisDialog.setModel(oModel, "analysis");
                    this._oFeatureAnalysisDialog.open();
                    
                    // Create visualizations
                    this._createFeatureVisualizations(analysisData);
                }.bind(this));
            } else {
                var oModel = new JSONModel(analysisData);
                this._oFeatureAnalysisDialog.setModel(oModel, "analysis");
                this._oFeatureAnalysisDialog.open();
                this._createFeatureVisualizations(analysisData);
            }
        },

        _createFeatureVisualizations: function(data) {
            // Create feature importance chart, correlation matrix, etc.
            if (!data.features || !this._oFeatureAnalysisDialog) return;
            
            // Create feature importance chart
            this._createFeatureImportanceChart(data.features);
            
            // Create correlation matrix visualization
            if (data.correlation_matrix) {
                this._createCorrelationMatrix(data.correlation_matrix);
            }
            
            // Create feature distribution charts
            this._createFeatureDistributionCharts(data.features);
        },
        
        _createFeatureImportanceChart: function(features) {
            // Sort features by importance
            var aSortedFeatures = features.slice().sort(function(a, b) {
                return (b.importance || 0) - (a.importance || 0);
            }).slice(0, 10); // Top 10 features
            
            // Prepare chart data
            var aChartData = aSortedFeatures.map(function(feature) {
                return {
                    Feature: feature.name,
                    Importance: (feature.importance || 0) * 100,
                    Type: feature.type || "Unknown"
                };
            });
            
            // Create chart model
            var oChartModel = new sap.ui.model.json.JSONModel({
                importanceData: aChartData
            });
            
            // Find or create chart container
            var oDialog = this._oFeatureAnalysisDialog;
            var oChartContainer = new sap.m.Panel({
                headerText: "Feature Importance (Top 10)",
                class: "sapUiMediumMargin"
            });
            
            // Create horizontal bar chart using VizFrame
            var oVizFrame = new sap.viz.ui5.controls.VizFrame({
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
            var oDataset = new sap.viz.ui5.data.FlattenedDataset({
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
            
            oVizFrame.addFeed(oFeedValueAxis);
            oVizFrame.addFeed(oFeedCategoryAxis);
            
            oChartContainer.addContent(oVizFrame);
            
            // Add to dialog content
            var oContent = oDialog.getContent()[0]; // Should be the container
            if (oContent && oContent.addContent) {
                oContent.addContent(oChartContainer);
            }
        },
        
        _createCorrelationMatrix: function(correlationMatrix) {
            if (!correlationMatrix || !Array.isArray(correlationMatrix)) return;
            
            var oPanel = new sap.m.Panel({
                headerText: "Feature Correlation Matrix",
                class: "sapUiMediumMargin"
            });
            
            // Create a simple grid representation of correlation matrix
            var oGrid = new sap.ui.layout.Grid({
                defaultSpan: "XL2 L2 M3 S6"
            });
            
            correlationMatrix.forEach(function(row, i) {
                if (i >= 6) return; // Limit for display
                row.forEach(function(correlation, j) {
                    if (j >= 6) return; // Limit for display
                    
                    var sState = "None";
                    if (Math.abs(correlation) > 0.7) sState = "Success";
                    else if (Math.abs(correlation) > 0.5) sState = "Warning";
                    else if (Math.abs(correlation) > 0.3) sState = "Information";
                    
                    var oTile = new sap.m.GenericTile({
                        class: "sapUiTinyMargin",
                        header: correlation.toFixed(2),
                        subheader: "(" + i + "," + j + ")",
                        state: sState,
                        press: function() {
                            sap.m.MessageToast.show("Correlation: " + correlation.toFixed(3));
                        }
                    });
                    
                    oGrid.addContent(oTile);
                });
            });
            
            oPanel.addContent(oGrid);
            
            // Add to dialog
            var oDialog = this._oFeatureAnalysisDialog;
            var oContent = oDialog.getContent()[0];
            if (oContent && oContent.addContent) {
                oContent.addContent(oPanel);
            }
        },
        
        _createFeatureDistributionCharts: function(features) {
            var oPanel = new sap.m.Panel({
                headerText: "Feature Distributions",
                class: "sapUiMediumMargin"
            });
            
            var oGrid = new sap.ui.layout.Grid({
                defaultSpan: "XL4 L4 M6 S12"
            });
            
            // Create distribution charts for numerical features
            features.slice(0, 8).forEach(function(feature) {
                if (feature.type !== "NUMERICAL" && feature.type !== "numerical") return;
                
                var oFeaturePanel = new sap.m.Panel({
                    headerText: feature.name,
                    class: "sapUiTinyMargin"
                });
                
                // Create micro chart for distribution
                var oMicroChart = new sap.suite.ui.microchart.AreaMicroChart({
                    height: "100px",
                    width: "100%"
                });
                
                // Generate sample distribution data
                var aPoints = [];
                var mean = feature.mean_value || 0;
                var stdDev = feature.std_dev || 1;
                
                for (var i = 0; i < 20; i++) {
                    var x = (i - 10) / 2;
                    var y = Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2));
                    aPoints.push(new sap.suite.ui.microchart.AreaMicroChartPoint({
                        x: i,
                        y: y * 100
                    }));
                }
                
                aPoints.forEach(function(point) {
                    oMicroChart.addPoint(point);
                });
                
                // Add statistics text
                var oStatsText = new sap.m.Text({
                    text: "Mean: " + (feature.mean_value || 0).toFixed(2) + 
                          ", Std: " + (feature.std_dev || 0).toFixed(2) +
                          ", Missing: " + (feature.missing_percent || 0).toFixed(1) + "%"
                });
                
                oFeaturePanel.addContent(oMicroChart);
                oFeaturePanel.addContent(oStatsText);
                oGrid.addContent(oFeaturePanel);
            });
            
            oPanel.addContent(oGrid);
            
            // Add to dialog
            var oDialog = this._oFeatureAnalysisDialog;
            var oContent = oDialog.getContent()[0];
            if (oContent && oContent.addContent) {
                oContent.addContent(oPanel);
            }
        },

        onGenerateEmbeddings: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sModelType = oContext.getProperty("modelType");
            
            if (!this._oEmbeddingDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent2.ext.fragment.EmbeddingConfiguration",
                    controller: this
                }).then(function(oDialog) {
                    this._oEmbeddingDialog = oDialog;
                    this.base.getView().addDependent(this._oEmbeddingDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        embeddingModel: sModelType === "LLM" ? "text-embedding-ada-002" : "custom",
                        dimensions: 768,
                        normalization: true,
                        batchSize: 32,
                        useGPU: true
                    });
                    this._oEmbeddingDialog.setModel(oModel, "embedding");
                    this._oEmbeddingDialog.open();
                }.bind(this));
            } else {
                this._oEmbeddingDialog.open();
            }
        },

        onConfirmGenerateEmbeddings: function() {
            var oModel = this._oEmbeddingDialog.getModel("embedding");
            var oData = oModel.getData();
            
            this._oEmbeddingDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks/" + oData.taskId + "/generate-embeddings",
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
                        "Vectors: " + data.vectorCount + "\n" +
                        "Dimensions: " + data.dimensions + "\n" +
                        "Processing time: " + data.processingTime + "s"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oEmbeddingDialog.setBusy(false);
                    MessageBox.error("Failed to generate embeddings: " + xhr.responseText);
                }.bind(this)
            });
        },

        onExportPreparedData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oExportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent2.ext.fragment.ExportPreparedData",
                    controller: this
                }).then(function(oDialog) {
                    this._oExportDialog = oDialog;
                    this.base.getView().addDependent(this._oExportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        format: "TENSORFLOW",
                        includeMetadata: true,
                        splitData: true,
                        compression: "GZIP"
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                }.bind(this));
            } else {
                this._oExportDialog.open();
            }
        },

        onExecuteExport: function() {
            var oModel = this._oExportDialog.getModel("export");
            var oData = oModel.getData();
            
            this._oExportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks/" + oData.taskId + "/export",
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
                    MessageBox.error("Export failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showDownloadLinks: function(files) {
            var sMessage = "Export completed! Download files:\n\n";
            files.forEach(function(file) {
                sMessage += file.name + " (" + file.size + ")\n";
            });
            
            MessageBox.information(sMessage, {
                actions: ["Download All", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Download All") {
                        files.forEach(function(file) {
                            window.open(file.url, "_blank");
                        });
                    }
                }
            });
        },

        onOptimizeHyperparameters: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
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

        _startHyperparameterOptimization: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks/" + sTaskId + "/optimize",
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
                        "Job ID: " + data.jobId + "\n" +
                        "You will be notified when complete."
                    );
                }.bind(this),
                error: function(xhr) {
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to start optimization: " + errorMsg);
                }.bind(this)
            });
        },

        /**
         * Get authentication token for secure API calls
         * @returns {string} Authentication token
         */
        _getAuthToken: function() {
            // Implementation to get auth token from session or user context
            return sap.ushell?.Container?.getUser?.()?.getId?.() || 'default-token';
        },

        /**
         * Validate user input for security and format compliance
         * @param {string} sInput - Input to validate
         * @param {string} sType - Type of validation
         * @returns {object} Validation result
         */
        _validateInput: function(sInput, sType) {
            if (!sInput || typeof sInput !== 'string') {
                return { isValid: false, message: "Input is required" };
            }

            var sSanitized = sInput.trim();
            
            // Check for XSS patterns
            var aXSSPatterns = [
                /<script/i,
                /javascript:/i,
                /on\w+\s*=/i,
                /<iframe/i,
                /<object/i,
                /<embed/i,
                /eval\s*\(/i,
                /Function\s*\(/i
            ];

            for (var i = 0; i < aXSSPatterns.length; i++) {
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
        _sanitizeErrorMessage: function(sMessage) {
            if (!sMessage) {
                return "An error occurred";
            }

            // Remove sensitive information
            var sSanitized = sMessage
                .replace(/\b(?:token|key|secret|password|auth)\b[:\s]*[^\s]+/gi, '[REDACTED]')
                .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, '[IP_ADDRESS]')
                .replace(/file:\/\/[^\s]+/g, '[FILE_PATH]')
                .replace(/https?:\/\/[^\s]+/g, '[URL]');

            return encodeXML(sSanitized);
        },

        /**
         * Secure EventSource creation with authentication
         * @param {string} sUrl - EventSource URL
         * @param {string} sToken - Authentication token
         * @returns {EventSource} Secure EventSource
         */
        _createSecureEventSource: function(sUrl, sToken) {
            if (!sUrl || !sToken) {
                throw new Error("URL and token required for secure EventSource");
            }

            // Validate URL
            try {
                var oUrl = new URL(sUrl, window.location.origin);
                if (!['http:', 'https:'].includes(oUrl.protocol)) {
                    throw new Error("Invalid protocol for EventSource");
                }
            } catch (e) {
                throw new Error("Invalid URL for EventSource");
            }

            // Add authentication to URL
            var sSecureUrl = sUrl + (sUrl.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(sToken);
            
            return new EventSource(sSecureUrl);
        },

        /**
         * Secure text formatter to prevent XSS
         * @param {string} sText - Text to format
         * @returns {string} Encoded text
         */
        formatSecureText: function(sText) {
            if (!sText) {
                return "";
            }
            return encodeXML(sText.toString());
        }
    });
});