sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent3.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartProcessing: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start vector processing for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startVectorProcessing(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startVectorProcessing: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + sTaskId + "/process",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Vector processing started");
                    this._extensionAPI.refresh();
                    
                    // Start monitoring with WebSocket
                    this._startWebSocketMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start processing: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startWebSocketMonitoring: function(sTaskId) {
            // WebSocket for real-time progress updates
            this._ws = new WebSocket("wss://" + window.location.host + "/a2a/agent3/v1/tasks/" + sTaskId + "/ws");
            
            this._ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                switch(data.type) {
                    case "progress":
                        this._updateProgress(data);
                        break;
                    case "chunk_processed":
                        MessageToast.show("Processed chunk " + data.chunk + "/" + data.totalChunks);
                        break;
                    case "completed":
                        this._ws.close();
                        this._extensionAPI.refresh();
                        MessageBox.success(
                            "Vector processing completed!\n" +
                            "Vectors generated: " + data.vectorCount + "\n" +
                            "Processing time: " + data.duration + "s"
                        );
                        break;
                    case "error":
                        this._ws.close();
                        MessageBox.error("Processing error: " + data.error);
                        break;
                }
            }.bind(this);
            
            this._ws.onerror = function() {
                MessageBox.error("Lost connection to processing server");
            };
        },

        _updateProgress: function(data) {
            // Update progress in UI
            var sMessage = "Processing: " + data.progress + "% " +
                          "(Vectors: " + data.vectorsProcessed + "/" + data.totalVectors + ")";
            MessageToast.show(sMessage);
        },

        onRunSimilaritySearch: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sCollectionName = oContext.getProperty("collectionName");
            
            if (!this._oSimilaritySearchDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.SimilaritySearch",
                    controller: this
                }).then(function(oDialog) {
                    this._oSimilaritySearchDialog = oDialog;
                    this.base.getView().addDependent(this._oSimilaritySearchDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        collection: sCollectionName,
                        queryType: "TEXT",
                        query: "",
                        vectorQuery: [],
                        topK: 10,
                        includeMetadata: true,
                        includeDistance: true,
                        filters: {}
                    });
                    this._oSimilaritySearchDialog.setModel(oModel, "similarity");
                    this._oSimilaritySearchDialog.open();
                }.bind(this));
            } else {
                this._oSimilaritySearchDialog.open();
            }
        },

        onExecuteSimilaritySearch: function() {
            var oModel = this._oSimilaritySearchDialog.getModel("similarity");
            var oData = oModel.getData();
            
            if (!oData.query && oData.queryType === "TEXT") {
                MessageBox.error("Please enter a search query");
                return;
            }
            
            this._oSimilaritySearchDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + oData.taskId + "/similarity-search",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    queryType: oData.queryType,
                    query: oData.query,
                    vectorQuery: oData.vectorQuery,
                    topK: oData.topK,
                    includeMetadata: oData.includeMetadata,
                    includeDistance: oData.includeDistance,
                    filters: oData.filters
                }),
                success: function(data) {
                    this._oSimilaritySearchDialog.setBusy(false);
                    
                    // Show results
                    this._showSimilarityResults(data.results);
                }.bind(this),
                error: function(xhr) {
                    this._oSimilaritySearchDialog.setBusy(false);
                    MessageBox.error("Search failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showSimilarityResults: function(results) {
            if (!this._oResultsDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.SimilarityResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oResultsDialog = oDialog;
                    this.base.getView().addDependent(this._oResultsDialog);
                    
                    var oModel = new JSONModel({ results: results });
                    this._oResultsDialog.setModel(oModel, "results");
                    this._oResultsDialog.open();
                }.bind(this));
            } else {
                var oModel = new JSONModel({ results: results });
                this._oResultsDialog.setModel(oModel, "results");
                this._oResultsDialog.open();
            }
        },

        onOptimizeIndex: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sIndexType = oContext.getProperty("indexType");
            
            MessageBox.confirm(
                "Optimize vector index? This may take several minutes.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeVectorIndex(sTaskId, sIndexType);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeVectorIndex: function(sTaskId, sIndexType) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + sTaskId + "/optimize-index",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    indexType: sIndexType,
                    parameters: {
                        efConstruction: 200,
                        M: 16,
                        compression: true
                    }
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.success(
                        "Index optimization completed!\n" +
                        "Query speed improvement: " + data.speedImprovement + "%\n" +
                        "Memory saved: " + data.memorySaved + " MB"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Optimization failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onExportVectors: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oExportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.ExportVectors",
                    controller: this
                }).then(function(oDialog) {
                    this._oExportDialog = oDialog;
                    this.base.getView().addDependent(this._oExportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        format: "NUMPY",
                        includeMetadata: true,
                        compression: "GZIP",
                        chunkSize: 10000
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
                url: "/a2a/agent3/v1/tasks/" + oData.taskId + "/export",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    format: oData.format,
                    includeMetadata: oData.includeMetadata,
                    compression: oData.compression,
                    chunkSize: oData.chunkSize
                }),
                success: function(data) {
                    this._oExportDialog.setBusy(false);
                    this._oExportDialog.close();
                    
                    MessageBox.success(
                        "Export completed!\n" +
                        "Files: " + data.files.length + "\n" +
                        "Total size: " + data.totalSize + " MB",
                        {
                            actions: ["Download", MessageBox.Action.CLOSE],
                            onClose: function(oAction) {
                                if (oAction === "Download") {
                                    window.open(data.downloadUrl, "_blank");
                                }
                            }
                        }
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oExportDialog.setBusy(false);
                    MessageBox.error("Export failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onVisualizeEmbeddings: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oVisualizationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.EmbeddingVisualization",
                    controller: this
                }).then(function(oDialog) {
                    this._oVisualizationDialog = oDialog;
                    this.base.getView().addDependent(this._oVisualizationDialog);
                    this._oVisualizationDialog.open();
                    
                    // Load embedding visualization data
                    this._loadEmbeddingVisualization(sTaskId);
                }.bind(this));
            } else {
                this._oVisualizationDialog.open();
                this._loadEmbeddingVisualization(sTaskId);
            }
        },

        _loadEmbeddingVisualization: function(sTaskId) {
            this._oVisualizationDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + sTaskId + "/visualization-data",
                type: "GET",
                data: {
                    method: "TSNE",
                    perplexity: 30,
                    dimensions: 3,
                    sampleSize: 1000
                },
                success: function(data) {
                    this._oVisualizationDialog.setBusy(false);
                    
                    // Create 3D visualization
                    this._render3DEmbeddings(data);
                }.bind(this),
                error: function(xhr) {
                    this._oVisualizationDialog.setBusy(false);
                    MessageBox.error("Failed to load visualization data");
                }.bind(this)
            });
        },

        _render3DEmbeddings: function(data) {
            // This would use Three.js or similar to render embeddings in 3D
            var oContainer = this.byId("embeddingVisualizationContainer");
            if (oContainer) {
                // Render 3D scatter plot with embedding points
                MessageToast.show("Rendering " + data.points.length + " embeddings in 3D");
            }
        },

        onClusterAnalysis: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm(
                "Run cluster analysis on embeddings?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runClusterAnalysis(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _runClusterAnalysis: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + sTaskId + "/cluster-analysis",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    algorithm: "KMEANS",
                    numClusters: "auto",
                    minClusterSize: 5
                }),
                success: function(data) {
                    MessageBox.success(
                        "Cluster analysis completed!\n" +
                        "Clusters found: " + data.numClusters + "\n" +
                        "Silhouette score: " + data.silhouetteScore.toFixed(3)
                    );
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Cluster analysis failed: " + xhr.responseText);
                }
            });
        }
    });
});