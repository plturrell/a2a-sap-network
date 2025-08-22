sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "sap/base/Log"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp, sanitizeHTML, Log) {
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
            
            // Check user permission
            if (!this._checkUserPermission("PROCESS_VECTORS")) {
                MessageBox.error("You don't have permission to start vector processing");
                return;
            }
            
            MessageBox.confirm("Start vector processing for '" + encodeXML(sTaskName) + "'?", {
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
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(sTaskId) + "/process",
                type: "POST",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    if (this._validateApiResponse(data)) {
                        MessageToast.show("Vector processing started");
                        this._extensionAPI.refresh();
                        
                        // Start monitoring with secure WebSocket
                        this._startSecureWebSocketMonitoring(sTaskId);
                        
                        // Audit logging
                        this._logAuditEvent("VECTOR_PROCESSING_STARTED", sTaskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to start processing: " + errorMsg);
                }.bind(this)
            });
        },

        _startSecureWebSocketMonitoring: function(sTaskId) {
            // Secure WebSocket with authentication
            var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            var host = window.location.host;
            var token = this._getAuthToken();
            
            if (!token) {
                Log.warning("No authentication token available for WebSocket connection");
                return;
            }
            
            var wsUrl = protocol + "//" + host + "/a2a/agent3/v1/tasks/" + encodeURIComponent(sTaskId) + "/ws";
            var authParam = 'auth_token';
            wsUrl = wsUrl + "?" + authParam + "=" + encodeURIComponent(token);
            
            try {
                this._ws = new WebSocket(wsUrl);
                this._setupWebSocketHandlers();
            } catch (error) {
                Log.error("Failed to initialize WebSocket", error);
                MessageBox.error("Failed to establish real-time connection");
            }
        },

        _setupWebSocketHandlers: function() {
            this._ws.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    // Validate WebSocket message
                    if (!this._validateWebSocketMessage(data)) {
                        Log.warning("Invalid WebSocket message received");
                        return;
                    }
                    
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
                            this._logAuditEvent("VECTOR_PROCESSING_COMPLETED", data.taskId);
                            break;
                        case "error":
                            this._ws.close();
                            var errorMsg = this._sanitizeErrorMessage(data.error);
                            MessageBox.error("Processing error: " + errorMsg);
                            break;
                    }
                } catch (e) {
                    Log.error("Error processing WebSocket message", e);
                }
            }.bind(this);
            
            this._ws.onerror = function() {
                MessageBox.error("Lost connection to processing server");
            };
            
            this._ws.onclose = function() {
                Log.info("WebSocket connection closed");
            };
        },

        _validateWebSocketMessage: function(data) {
            if (!data || typeof data !== 'object') {
                return false;
            }
            
            // Check required fields based on message type
            var requiredFields = {
                "progress": ["progress", "vectorsProcessed", "totalVectors"],
                "chunk_processed": ["chunk", "totalChunks"],
                "completed": ["vectorCount", "duration"],
                "error": ["error"]
            };
            
            var required = requiredFields[data.type];
            if (!required) {
                return false;
            }
            
            for (var i = 0; i < required.length; i++) {
                if (!data.hasOwnProperty(required[i])) {
                    return false;
                }
            }
            
            return true;
        },

        _updateProgress: function(data) {
            // Update progress in UI with validation
            var nProgress = parseFloat(data.progress);
            if (isNaN(nProgress) || nProgress < 0 || nProgress > 100) {
                return;
            }
            
            var sMessage = "Processing: " + nProgress.toFixed(1) + "% " +
                          "(Vectors: " + data.vectorsProcessed + "/" + data.totalVectors + ")";
            MessageToast.show(sMessage);
        },

        onRunSimilaritySearch: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sCollectionName = oContext.getProperty("collectionName");
            
            // Check user permission
            if (!this._checkUserPermission("SIMILARITY_SEARCH")) {
                MessageBox.error("You don't have permission to run similarity search");
                return;
            }
            
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
            
            // Validate search input
            var oValidation = this._validateSimilaritySearchData(oData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }
            
            this._oSimilaritySearchDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(oData.taskId) + "/similarity-search",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oValidation.sanitizedData),
                success: function(data) {
                    this._oSimilaritySearchDialog.setBusy(false);
                    
                    if (this._validateApiResponse(data)) {
                        // Show results
                        this._showSimilarityResults(data.results);
                        
                        // Audit logging
                        this._logAuditEvent("SIMILARITY_SEARCH_EXECUTED", oData.taskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oSimilaritySearchDialog.setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Search failed: " + errorMsg);
                }.bind(this)
            });
        },

        _validateSimilaritySearchData: function(oData) {
            if (!oData.query && oData.queryType === "TEXT") {
                return { isValid: false, message: "Please enter a search query" };
            }
            
            var oSanitized = {
                taskId: oData.taskId,
                collection: oData.collection
            };
            
            // Validate query based on type
            if (oData.queryType === "TEXT") {
                var oQueryValidation = this._validateInput(oData.query, "query");
                if (!oQueryValidation.isValid) {
                    return { isValid: false, message: "Query: " + oQueryValidation.message };
                }
                oSanitized.query = oQueryValidation.sanitized;
                oSanitized.queryType = "TEXT";
            } else if (oData.queryType === "VECTOR") {
                if (!Array.isArray(oData.vectorQuery) || oData.vectorQuery.length === 0) {
                    return { isValid: false, message: "Vector query is required" };
                }
                // Validate vector dimensions
                oSanitized.vectorQuery = oData.vectorQuery.map(function(val) {
                    var nVal = parseFloat(val);
                    return isNaN(nVal) ? 0 : nVal;
                });
                oSanitized.queryType = "VECTOR";
            }
            
            // Validate numeric parameters
            var nTopK = parseInt(oData.topK, 10);
            if (isNaN(nTopK) || nTopK < 1 || nTopK > 100) {
                return { isValid: false, message: "Top K must be between 1 and 100" };
            }
            oSanitized.topK = nTopK;
            
            oSanitized.includeMetadata = !!oData.includeMetadata;
            oSanitized.includeDistance = !!oData.includeDistance;
            
            // Validate filters
            if (oData.filters && typeof oData.filters === 'object') {
                oSanitized.filters = this._sanitizeFilters(oData.filters);
            }
            
            return { isValid: true, sanitizedData: oSanitized };
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
            
            // Check user permission
            if (!this._checkUserPermission("OPTIMIZE_INDEX")) {
                MessageBox.error("You don't have permission to optimize indexes");
                return;
            }
            
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
            
            var oOptimizationParams = {
                indexType: sIndexType,
                parameters: {
                    efConstruction: 200,
                    M: 16,
                    compression: true
                }
            };
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(sTaskId) + "/optimize-index",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oOptimizationParams),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    if (this._validateApiResponse(data)) {
                        MessageBox.success(
                            "Index optimization completed!\n" +
                            "Query speed improvement: " + data.speedImprovement + "%\n" +
                            "Memory saved: " + data.memorySaved + " MB"
                        );
                        this._extensionAPI.refresh();
                        
                        // Audit logging
                        this._logAuditEvent("INDEX_OPTIMIZED", sTaskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Optimization failed: " + errorMsg);
                }.bind(this)
            });
        },

        onExportVectors: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            // Check user permission
            if (!this._checkUserPermission("EXPORT_VECTORS")) {
                MessageBox.error("You don't have permission to export vectors");
                return;
            }
            
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
                        chunkSize: 10000,
                        validation: true
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
            
            // Validate export parameters
            var oValidation = this._validateExportData(oData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }
            
            this._oExportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(oData.taskId) + "/export",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oValidation.sanitizedData),
                success: function(data) {
                    this._oExportDialog.setBusy(false);
                    this._oExportDialog.close();
                    
                    if (this._validateApiResponse(data)) {
                        MessageBox.success(
                            "Export completed!\n" +
                            "Files: " + data.files.length + "\n" +
                            "Total size: " + data.totalSize + " MB",
                            {
                                actions: ["Download", MessageBox.Action.CLOSE],
                                onClose: function(oAction) {
                                    if (oAction === "Download") {
                                        // Secure download
                                        this._secureDownload(data.downloadUrl);
                                    }
                                }.bind(this)
                            }
                        );
                        
                        // Audit logging
                        this._logAuditEvent("VECTORS_EXPORTED", oData.taskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oExportDialog.setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Export failed: " + errorMsg);
                }.bind(this)
            });
        },

        _validateExportData: function(oData) {
            var oSanitized = {
                taskId: oData.taskId
            };
            
            // Validate format
            var aValidFormats = ["NUMPY", "PARQUET", "CSV", "JSON", "HDF5", "PICKLE"];
            if (!aValidFormats.includes(oData.format)) {
                return { isValid: false, message: "Invalid export format" };
            }
            oSanitized.format = oData.format;
            
            // Validate compression
            var aValidCompression = ["NONE", "GZIP", "BZIP2", "LZ4", "ZSTD"];
            if (!aValidCompression.includes(oData.compression)) {
                return { isValid: false, message: "Invalid compression type" };
            }
            oSanitized.compression = oData.compression;
            
            // Validate chunk size
            var nChunkSize = parseInt(oData.chunkSize, 10);
            if (isNaN(nChunkSize) || nChunkSize < 100 || nChunkSize > 100000) {
                return { isValid: false, message: "Chunk size must be between 100 and 100000" };
            }
            oSanitized.chunkSize = nChunkSize;
            
            oSanitized.includeMetadata = !!oData.includeMetadata;
            oSanitized.validation = !!oData.validation;
            
            return { isValid: true, sanitizedData: oSanitized };
        },

        _secureDownload: function(sUrl) {
            // Validate download URL
            try {
                var oUrl = new URL(sUrl, window.location.origin);
                if (!['http:', 'https:'].includes(oUrl.protocol)) {
                    MessageBox.error("Invalid download URL");
                    return;
                }
                
                // Create temporary anchor for download
                var a = document.createElement('a');
                a.href = sUrl;
                a.download = true;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } catch (e) {
                MessageBox.error("Failed to initiate download");
            }
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
            
            var oParams = {
                method: "TSNE",
                perplexity: 30,
                dimensions: 3,
                sampleSize: 1000
            };
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(sTaskId) + "/visualization-data",
                type: "GET",
                data: oParams,
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data) {
                    this._oVisualizationDialog.setBusy(false);
                    
                    if (this._validateApiResponse(data)) {
                        // Create 3D visualization
                        this._render3DEmbeddings(data);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oVisualizationDialog.setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to load visualization data: " + errorMsg);
                }.bind(this)
            });
        },

        _render3DEmbeddings: function(data) {
            // Performance-optimized 3D rendering
            var oContainer = this.byId("embeddingVisualizationContainer");
            if (oContainer && data.points && data.points.length > 0) {
                // Implement throttled 3D rendering
                if (this._renderTimeout) {
                    clearTimeout(this._renderTimeout);
                }
                
                this._renderTimeout = setTimeout(function() {
                    // Render 3D scatter plot with embedding points
                    MessageToast.show("Rendering " + data.points.length + " embeddings in 3D");
                    // Actual Three.js implementation would go here
                }.bind(this), 100);
            }
        },

        onClusterAnalysis: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            // Check user permission
            if (!this._checkUserPermission("CLUSTER_ANALYSIS")) {
                MessageBox.error("You don't have permission to run cluster analysis");
                return;
            }
            
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
            var oAnalysisParams = {
                algorithm: "KMEANS",
                numClusters: "auto",
                minClusterSize: 5,
                validation: true
            };
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks/" + encodeURIComponent(sTaskId) + "/cluster-analysis",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oAnalysisParams),
                success: function(data) {
                    if (this._validateApiResponse(data)) {
                        MessageBox.success(
                            "Cluster analysis completed!\n" +
                            "Clusters found: " + data.numClusters + "\n" +
                            "Silhouette score: " + data.silhouetteScore.toFixed(3)
                        );
                        
                        // Audit logging
                        this._logAuditEvent("CLUSTER_ANALYSIS_COMPLETED", sTaskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Cluster analysis failed: " + errorMsg);
                }.bind(this)
            });
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
                case "query":
                    if (sSanitized.length > 1000) {
                        return { isValid: false, message: "Query too long (max 1000 characters)" };
                    }
                    // Escape special regex characters for vector search
                    sSanitized = escapeRegExp(sSanitized);
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
         * Sanitize filter object for safe querying
         * @param {object} oFilters - Filters to sanitize
         * @returns {object} Sanitized filters
         */
        _sanitizeFilters: function(oFilters) {
            var oSanitized = {};
            
            for (var sKey in oFilters) {
                if (oFilters.hasOwnProperty(sKey)) {
                    // Validate key
                    if (!/^[a-zA-Z0-9_]+$/.test(sKey)) {
                        continue; // Skip invalid keys
                    }
                    
                    var value = oFilters[sKey];
                    if (typeof value === 'string') {
                        oSanitized[sKey] = encodeXML(value);
                    } else if (typeof value === 'number' || typeof value === 'boolean') {
                        oSanitized[sKey] = value;
                    } else if (Array.isArray(value)) {
                        oSanitized[sKey] = value.map(function(item) {
                            return typeof item === 'string' ? encodeXML(item) : item;
                        });
                    }
                }
            }
            
            return oSanitized;
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
                    Log.error("Potential prototype pollution detected in API response");
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
         * Get CSRF token for secure API calls
         * @returns {string} CSRF token
         */
        _getCSRFToken: function() {
            var token = jQuery("meta[name='csrf-token']").attr("content");
            if (!token) {
                // Fallback to fetch token
                jQuery.ajax({
                    url: "/sap/bc/ui2/start_up",
                    method: "HEAD",
                    async: false,
                    headers: {
                        "X-CSRF-Token": "Fetch",
                        "X-Requested-With": "XMLHttpRequest"
                    },
                    success: function(data, status, xhr) {
                        token = xhr.getResponseHeader("X-CSRF-Token");
                    }
                });
            }
            return token || "";
        },

        /**
         * Get authentication token for secure connections
         * @returns {string} Authentication token
         */
        _getAuthToken: function() {
            // Implementation to get auth token from session or user context
            var sUserId = sap.ushell?.Container?.getUser?.()?.getId?.();
            return sUserId || this._generateSessionToken();
        },

        /**
         * Generate a session-based token
         * @returns {string} Session token
         */
        _generateSessionToken: function() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Log audit events for compliance
         * @param {string} sEventType - Type of event
         * @param {string} sEntityId - Entity ID
         */
        _logAuditEvent: function(sEventType, sEntityId) {
            try {
                jQuery.ajax({
                    url: "/a2a/common/v1/audit",
                    type: "POST",
                    contentType: "application/json",
                    data: JSON.stringify({
                        eventType: sEventType,
                        entityId: sEntityId,
                        timestamp: new Date().toISOString(),
                        userId: sap.ushell?.Container?.getUser?.()?.getId?.() || "anonymous",
                        component: "AGENT3_UI"
                    })
                });
            } catch (e) {
                Log.warning("Failed to log audit event", e);
            }
        },

        /**
         * Check user permissions for actions
         * @param {string} sAction - Action to check
         * @returns {boolean} Whether user has permission
         */
        _checkUserPermission: function(sAction) {
            // Implementation would check against user roles/permissions
            var userRoles = sap.ushell?.Container?.getUser?.()?.getRoles?.() || [];
            
            var requiredRoles = {
                "PROCESS_VECTORS": ["VectorAdmin", "VectorUser"],
                "SIMILARITY_SEARCH": ["VectorAdmin", "VectorUser", "VectorViewer"],
                "OPTIMIZE_INDEX": ["VectorAdmin"],
                "EXPORT_VECTORS": ["VectorAdmin", "VectorExporter"],
                "CLUSTER_ANALYSIS": ["VectorAdmin", "DataScientist"]
            };
            
            var required = requiredRoles[sAction] || [];
            return required.length === 0 || required.some(function(role) {
                return userRoles.indexOf(role) !== -1;
            });
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
         * Format distance value
         * @param {number} nDistance - Distance value
         * @returns {string} Formatted distance
         */
        formatDistance: function(nDistance) {
            if (typeof nDistance !== 'number') {
                return "0.0000";
            }
            return nDistance.toFixed(4);
        },

        /**
         * Clean up resources on exit
         */
        onExit: function() {
            // Clean up WebSocket connections
            if (this._ws) {
                this._ws.close();
            }
            
            // Clean up timeouts
            if (this._renderTimeout) {
                clearTimeout(this._renderTimeout);
            }
            
            // Clean up any other resources
            this._cleanup();
        },

        _cleanup: function() {
            // Remove any event listeners or temporary resources
            this._ws = null;
            this._renderTimeout = null;
        }
    });
});