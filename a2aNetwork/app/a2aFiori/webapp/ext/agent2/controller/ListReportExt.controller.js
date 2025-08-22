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
            }
        },

        onCreateAITask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.CreateAIPreparationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model for the dialog
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        datasetName: "",
                        modelType: "",
                        dataType: "",
                        framework: "TENSORFLOW",
                        splitRatio: 80,
                        validationStrategy: "KFOLD",
                        featureSelection: true,
                        autoFeatureEngineering: true
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

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

        _loadDataProfile: function() {
            // Show loading
            this._oDataProfiler.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/data-profile",
                type: "GET",
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

        onAutoML: function() {
            var oView = this.base.getView();
            
            if (!this._oAutoMLWizard) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent2.ext.fragment.AutoMLWizard",
                    controller: this
                }).then(function(oWizard) {
                    this._oAutoMLWizard = oWizard;
                    oView.addDependent(this._oAutoMLWizard);
                    
                    // Initialize AutoML model
                    var oAutoMLModel = new JSONModel({
                        step: 1,
                        dataset: null,
                        problemType: "",
                        targetColumn: "",
                        evaluationMetric: "",
                        timeLimit: 60,
                        maxModels: 10,
                        includeEnsemble: true,
                        crossValidation: 5
                    });
                    this._oAutoMLWizard.setModel(oAutoMLModel, "automl");
                    this._oAutoMLWizard.open();
                }.bind(this));
            } else {
                this._oAutoMLWizard.open();
            }
        },

        onModelTemplates: function() {
            // Navigate to model templates
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("ModelTemplates");
        },

        onBatchPrepare: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one task for batch preparation.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch preparation for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchPreparation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchPreparation: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/batch-prepare",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    gpuAcceleration: true
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch preparation started!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Estimated time: " + data.estimatedTime + " minutes"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch preparation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validation
            if (!oData.taskName || !oData.datasetName || !oData.modelType || !oData.dataType) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Validate data before sending
            var oValidation = this._validateTaskData(oData);
            if (!oValidation.isValid) {
                this._oCreateDialog.setBusy(false);
                MessageBox.error("Validation Error: " + oValidation.message);
                return;
            }

            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oValidation.sanitizedData),
                success: function(data) {
                    if (this._validateApiResponse(data)) {
                        this._oCreateDialog.setBusy(false);
                        this._oCreateDialog.close();
                        MessageToast.show("AI preparation task created successfully");
                        this._extensionAPI.refresh();
                    } else {
                        this._oCreateDialog.setBusy(false);
                        MessageBox.error("Invalid response from server");
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to create task: " + errorMsg);
                }.bind(this)
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },

        /**
         * Validate AI task data for security and correctness
         * @param {object} oData - Task data to validate
         * @returns {object} Validation result with sanitized data
         */
        _validateTaskData: function(oData) {
            if (!oData || typeof oData !== 'object') {
                return { isValid: false, message: "Invalid task data" };
            }

            var oSanitized = {};
            
            // Validate task name
            var sTaskName = oData.taskName;
            if (!sTaskName || typeof sTaskName !== 'string') {
                return { isValid: false, message: "Task name is required" };
            }
            
            var oNameValidation = this._validateInput(sTaskName, "taskName");
            if (!oNameValidation.isValid) {
                return { isValid: false, message: "Task name: " + oNameValidation.message };
            }
            oSanitized.taskName = oNameValidation.sanitized;

            // Validate dataset path
            if (oData.datasetPath) {
                var oPathValidation = this._validateInput(oData.datasetPath, "path");
                if (!oPathValidation.isValid) {
                    return { isValid: false, message: "Dataset path: " + oPathValidation.message };
                }
                oSanitized.datasetPath = oPathValidation.sanitized;
            }

            // Validate model type
            var aValidModelTypes = ['Classification', 'Regression', 'Clustering', 'NLP', 'Computer Vision', 'Time Series'];
            if (oData.modelType && !aValidModelTypes.includes(oData.modelType)) {
                return { isValid: false, message: "Invalid model type" };
            }
            oSanitized.modelType = oData.modelType;

            // Validate data type
            var aValidDataTypes = ['Tabular', 'Text', 'Image', 'Audio', 'Time Series', 'Graph'];
            if (oData.dataType && !aValidDataTypes.includes(oData.dataType)) {
                return { isValid: false, message: "Invalid data type" };
            }
            oSanitized.dataType = oData.dataType;

            // Validate numeric fields
            if (oData.trainSplit !== undefined) {
                var nTrainSplit = parseFloat(oData.trainSplit);
                if (isNaN(nTrainSplit) || nTrainSplit < 0.1 || nTrainSplit > 0.9) {
                    return { isValid: false, message: "Train split must be between 0.1 and 0.9" };
                }
                oSanitized.trainSplit = nTrainSplit;
            }

            if (oData.validationSplit !== undefined) {
                var nValidationSplit = parseFloat(oData.validationSplit);
                if (isNaN(nValidationSplit) || nValidationSplit < 0.05 || nValidationSplit > 0.5) {
                    return { isValid: false, message: "Validation split must be between 0.05 and 0.5" };
                }
                oSanitized.validationSplit = nValidationSplit;
            }

            // Copy other safe fields
            var aSafeFields = ['description', 'targetColumn', 'featureColumns', 'validationStrategy', 'optimizationMetric'];
            aSafeFields.forEach(function(sField) {
                if (oData[sField] !== undefined) {
                    if (typeof oData[sField] === 'string') {
                        var oFieldValidation = this._validateInput(oData[sField], "text");
                        if (oFieldValidation.isValid) {
                            oSanitized[sField] = oFieldValidation.sanitized;
                        }
                    } else {
                        oSanitized[sField] = oData[sField];
                    }
                }
            }.bind(this));

            return { isValid: true, sanitizedData: oSanitized };
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
                case "taskName":
                    if (sSanitized.length < 3 || sSanitized.length > 100) {
                        return { isValid: false, message: "Must be 3-100 characters" };
                    }
                    if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Contains invalid characters" };
                    }
                    break;
                
                case "path":
                    if (sSanitized.length > 500) {
                        return { isValid: false, message: "Path too long" };
                    }
                    // Basic path validation
                    if (/[<>:"|?*]/.test(sSanitized)) {
                        return { isValid: false, message: "Contains invalid path characters" };
                    }
                    break;
                
                case "url":
                    try {
                        var oUrl = new URL(sSanitized);
                        if (!['http:', 'https:'].includes(oUrl.protocol)) {
                            return { isValid: false, message: "Only HTTP/HTTPS URLs allowed" };
                        }
                    } catch (e) {
                        return { isValid: false, message: "Invalid URL format" };
                    }
                    break;
                
                case "text":
                default:
                    if (sSanitized.length > 10000) {
                        return { isValid: false, message: "Text too long" };
                    }
                    break;
            }

            return { isValid: true, sanitized: encodeXML(sSanitized) };
        },

        /**
         * Validate API response data
         * @param {object} oData - Response data
         * @returns {boolean} Whether data is valid
         */
        _validateApiResponse: function(oData) {
            if (!oData || typeof oData !== 'object') {
                return false;
            }

            // Check for prototype pollution
            var aSuspiciousKeys = ['__proto__', 'constructor', 'prototype'];
            for (var sKey in oData) {
                if (aSuspiciousKeys.indexOf(sKey) !== -1) {
                    return false;
                }
            }

            return true;
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
         * Secure text formatter to prevent XSS
         * @param {string} sText - Text to format
         * @returns {string} Encoded text
         */
        formatSecureText: function(sText) {
            if (!sText) {
                return "";
            }
            return encodeXML(sText.toString());
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

            // Add authentication to URL (since EventSource doesn't support headers)
            var sSecureUrl = sUrl + (sUrl.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(sToken);
            
            return new EventSource(sSecureUrl);
        },

        /**
         * Event handler for input changes with validation
         */
        onTaskNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oInput = oEvent.getSource();
            var oValidation = this._validateInput(sValue, "taskName");
            
            if (!oValidation.isValid) {
                oInput.setValueState("Error");
                oInput.setValueStateText(oValidation.message);
            } else {
                oInput.setValueState("None");
                oInput.setValueStateText("");
            }
        }
    });
});