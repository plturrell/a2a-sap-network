sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
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
            
            jQuery.ajax({
                url: "/a2a/agent2/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("AI preparation task created successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    MessageBox.error("Failed to create task: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        }
    });
});