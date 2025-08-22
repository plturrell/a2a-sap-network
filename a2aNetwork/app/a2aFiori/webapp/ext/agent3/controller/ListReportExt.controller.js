sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent3.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateVectorTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent3.ext.fragment.CreateVectorTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        dataSource: "",
                        dataType: "TEXT",
                        embeddingModel: "text-embedding-ada-002",
                        modelProvider: "OPENAI",
                        vectorDatabase: "PINECONE",
                        indexType: "HNSW",
                        distanceMetric: "COSINE",
                        dimensions: 1536,
                        chunkSize: 512,
                        chunkOverlap: 50,
                        normalization: true
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onVectorSearch: function() {
            var oView = this.base.getView();
            
            if (!this._oVectorSearchDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent3.ext.fragment.VectorSearch",
                    controller: this
                }).then(function(oDialog) {
                    this._oVectorSearchDialog = oDialog;
                    oView.addDependent(this._oVectorSearchDialog);
                    
                    // Initialize search model
                    var oSearchModel = new JSONModel({
                        query: "",
                        collection: "",
                        topK: 10,
                        similarityThreshold: 0.7,
                        filters: {},
                        searchResults: []
                    });
                    this._oVectorSearchDialog.setModel(oSearchModel, "search");
                    this._oVectorSearchDialog.open();
                    
                    // Load available collections
                    this._loadVectorCollections();
                }.bind(this));
            } else {
                this._oVectorSearchDialog.open();
                this._loadVectorCollections();
            }
        },

        _loadVectorCollections: function() {
            jQuery.ajax({
                url: "/a2a/agent3/v1/collections",
                type: "GET",
                success: function(data) {
                    var oSearchModel = this._oVectorSearchDialog.getModel("search");
                    oSearchModel.setProperty("/collections", data.collections);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load vector collections");
                }
            });
        },

        onExecuteVectorSearch: function() {
            var oSearchModel = this._oVectorSearchDialog.getModel("search");
            var oSearchData = oSearchModel.getData();
            
            if (!oSearchData.query || !oSearchData.collection) {
                MessageBox.error("Please enter a query and select a collection");
                return;
            }
            
            this._oVectorSearchDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/search",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    query: oSearchData.query,
                    collection: oSearchData.collection,
                    topK: oSearchData.topK,
                    threshold: oSearchData.similarityThreshold,
                    filters: oSearchData.filters
                }),
                success: function(data) {
                    this._oVectorSearchDialog.setBusy(false);
                    oSearchModel.setProperty("/searchResults", data.results);
                    
                    if (data.results.length === 0) {
                        MessageToast.show("No similar vectors found");
                    } else {
                        MessageToast.show("Found " + data.results.length + " similar vectors");
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oVectorSearchDialog.setBusy(false);
                    MessageBox.error("Search failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onManageCollections: function() {
            // Navigate to vector collections management
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("VectorCollections");
        },

        onVectorVisualization: function() {
            var oView = this.base.getView();
            
            if (!this._oVisualizationDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent3.ext.fragment.VectorVisualization3D",
                    controller: this
                }).then(function(oDialog) {
                    this._oVisualizationDialog = oDialog;
                    oView.addDependent(this._oVisualizationDialog);
                    this._oVisualizationDialog.open();
                    
                    // Initialize 3D visualization
                    this._init3DVisualization();
                }.bind(this));
            } else {
                this._oVisualizationDialog.open();
                this._refresh3DVisualization();
            }
        },

        _init3DVisualization: function() {
            // Initialize Three.js or similar 3D library for vector visualization
            // This would render vectors in 3D space using t-SNE or UMAP reduction
            setTimeout(function() {
                var oContainer = this.byId("visualization3DContainer");
                if (oContainer) {
                    // Create 3D scene
                    this._create3DScene(oContainer.getDomRef());
                }
            }.bind(this), 100);
        },

        _create3DScene: function(container) {
            // Placeholder for 3D visualization implementation
            // Would use Three.js to create interactive 3D scatter plot
            MessageToast.show("3D visualization loading...");
        },

        onBatchProcessing: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch processing.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch vector processing for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchProcessing(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchProcessing: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/batch-process",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    useGPU: true,
                    priority: "HIGH"
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch processing started!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Estimated vectors: " + data.estimatedVectors
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch processing failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onModelComparison: function() {
            var oView = this.base.getView();
            
            if (!this._oModelComparisonDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent3.ext.fragment.ModelComparison",
                    controller: this
                }).then(function(oDialog) {
                    this._oModelComparisonDialog = oDialog;
                    oView.addDependent(this._oModelComparisonDialog);
                    this._oModelComparisonDialog.open();
                    
                    // Load model comparison data
                    this._loadModelComparison();
                }.bind(this));
            } else {
                this._oModelComparisonDialog.open();
                this._loadModelComparison();
            }
        },

        _loadModelComparison: function() {
            jQuery.ajax({
                url: "/a2a/agent3/v1/model-comparison",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel(data);
                    this._oModelComparisonDialog.setModel(oModel, "comparison");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load model comparison data");
                }
            });
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validation
            if (!oData.taskName || !oData.dataSource) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Vector processing task created successfully");
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