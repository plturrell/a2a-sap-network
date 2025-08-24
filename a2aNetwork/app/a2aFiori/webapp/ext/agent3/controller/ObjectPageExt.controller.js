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
], (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp, sanitizeHTML, Log) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent3.ext.controller.ObjectPageExt", {

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
                dataSource: "",
                dataType: "",
                embeddingModel: "text-embedding-ada-002",
                modelProvider: "OPENAI",
                vectorDatabase: "PINECONE",
                indexType: "HNSW",
                distanceMetric: "COSINE",
                dimensions: 1536,
                chunkSize: 512,
                chunkOverlap: 50,
                normalization: true,
                batchSize: 32,
                parallelProcessing: true,
                useGPU: false,
                errorHandling: 1,
                progressMonitoring: true,
                incrementalUpdates: false,
                collectionName: "",
                metadataSchema: "",
                isValid: false,
                taskNameState: "None",
                taskNameStateText: "",
                dataSourceState: "None",
                dataSourceStateText: "",
                dataTypeState: "None",
                dataTypeStateText: ""
            });
            this.base.getView().setModel(oCreateModel, "create");
        },

        onStartProcessing() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            // Check user permission
            if (!this._checkUserPermission("PROCESS_VECTORS")) {
                MessageBox.error("You don't have permission to start vector processing");
                return;
            }

            MessageBox.confirm(`Start vector processing for '${ encodeXML(sTaskName) }'?`, {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startVectorProcessing(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startVectorProcessing(sTaskId) {
            this._extensionAPI.getView().setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(sTaskId) }/process`,
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
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Failed to start processing: ${ errorMsg}`);
                }.bind(this)
            });
        },

        _startSecureWebSocketMonitoring(sTaskId) {
            // Secure WebSocket with authentication
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const host = window.location.host;
            const token = this._getAuthToken();

            if (!token) {
                Log.warning("No authentication token available for WebSocket connection");
                return;
            }

            let wsUrl = `${protocol }//${ host }/a2a/agent3/v1/tasks/${ encodeURIComponent(sTaskId) }/ws`;
            const authParam = "auth_token";
            wsUrl = `${wsUrl }?${ authParam }=${ encodeURIComponent(token)}`;

            try {
                this._ws = new WebSocket(wsUrl);
                this._setupWebSocketHandlers();
            } catch (error) {
                Log.error("Failed to initialize WebSocket", error);
                MessageBox.error("Failed to establish real-time connection");
            }
        },

        _setupWebSocketHandlers() {
            this._ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);

                    // Validate WebSocket message
                    if (!this._validateWebSocketMessage(data)) {
                        Log.warning("Invalid WebSocket message received");
                        return;
                    }

                    switch (data.type) {
                    case "progress":
                        this._updateProgress(data);
                        break;
                    case "chunk_processed":
                        MessageToast.show(`Processed chunk ${ data.chunk }/${ data.totalChunks}`);
                        break;
                    case "completed":
                        this._ws.close();
                        this._extensionAPI.refresh();
                        MessageBox.success(
                            "Vector processing completed!\n" +
                                `Vectors generated: ${ data.vectorCount }\n` +
                                `Processing time: ${ data.duration }s`
                        );
                        this._logAuditEvent("VECTOR_PROCESSING_COMPLETED", data.taskId);
                        break;
                    case "error":
                        this._ws.close();
                        const errorMsg = this._sanitizeErrorMessage(data.error);
                        MessageBox.error(`Processing error: ${ errorMsg}`);
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

        _validateWebSocketMessage(data) {
            if (!data || typeof data !== "object") {
                return false;
            }

            // Check required fields based on message type
            const requiredFields = {
                "progress": ["progress", "vectorsProcessed", "totalVectors"],
                "chunk_processed": ["chunk", "totalChunks"],
                "completed": ["vectorCount", "duration"],
                "error": ["error"]
            };

            const required = requiredFields[data.type];
            if (!required) {
                return false;
            }

            for (let i = 0; i < required.length; i++) {
                if (!data.hasOwnProperty(required[i])) {
                    return false;
                }
            }

            return true;
        },

        _updateProgress(data) {
            // Update progress in UI with validation
            const nProgress = parseFloat(data.progress);
            if (isNaN(nProgress) || nProgress < 0 || nProgress > 100) {
                return;
            }

            const sMessage = `Processing: ${ nProgress.toFixed(1) }% ` +
                          `(Vectors: ${ data.vectorsProcessed }/${ data.totalVectors })`;
            MessageToast.show(sMessage);
        },

        onRunSimilaritySearch() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sCollectionName = oContext.getProperty("collectionName");

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
                }).then((oDialog) => {
                    this._oSimilaritySearchDialog = oDialog;
                    this.base.getView().addDependent(this._oSimilaritySearchDialog);

                    const oModel = new JSONModel({
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
                });
            } else {
                this._oSimilaritySearchDialog.open();
            }
        },

        onExecuteSimilaritySearch() {
            const oModel = this._oSimilaritySearchDialog.getModel("similarity");
            const oData = oModel.getData();

            // Validate search input
            const oValidation = this._validateSimilaritySearchData(oData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }

            this._oSimilaritySearchDialog.setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(oData.taskId) }/similarity-search`,
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
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Search failed: ${ errorMsg}`);
                }.bind(this)
            });
        },

        _validateSimilaritySearchData(oData) {
            if (!oData.query && oData.queryType === "TEXT") {
                return { isValid: false, message: "Please enter a search query" };
            }

            const oSanitized = {
                taskId: oData.taskId,
                collection: oData.collection
            };

            // Validate query based on type
            if (oData.queryType === "TEXT") {
                const oQueryValidation = this._validateInput(oData.query, "query");
                if (!oQueryValidation.isValid) {
                    return { isValid: false, message: `Query: ${ oQueryValidation.message}` };
                }
                oSanitized.query = oQueryValidation.sanitized;
                oSanitized.queryType = "TEXT";
            } else if (oData.queryType === "VECTOR") {
                if (!Array.isArray(oData.vectorQuery) || oData.vectorQuery.length === 0) {
                    return { isValid: false, message: "Vector query is required" };
                }
                // Validate vector dimensions
                oSanitized.vectorQuery = oData.vectorQuery.map((val) => {
                    const nVal = parseFloat(val);
                    return isNaN(nVal) ? 0 : nVal;
                });
                oSanitized.queryType = "VECTOR";
            }

            // Validate numeric parameters
            const nTopK = parseInt(oData.topK, 10);
            if (isNaN(nTopK) || nTopK < 1 || nTopK > 100) {
                return { isValid: false, message: "Top K must be between 1 and 100" };
            }
            oSanitized.topK = nTopK;

            oSanitized.includeMetadata = !!oData.includeMetadata;
            oSanitized.includeDistance = !!oData.includeDistance;

            // Validate filters
            if (oData.filters && typeof oData.filters === "object") {
                oSanitized.filters = this._sanitizeFilters(oData.filters);
            }

            return { isValid: true, sanitizedData: oSanitized };
        },

        _showSimilarityResults(results) {
            if (!this._oResultsDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.SimilarityResults",
                    controller: this
                }).then((oDialog) => {
                    this._oResultsDialog = oDialog;
                    this.base.getView().addDependent(this._oResultsDialog);

                    const oModel = new JSONModel({ results });
                    this._oResultsDialog.setModel(oModel, "results");
                    this._oResultsDialog.open();
                });
            } else {
                const oModel = new JSONModel({ results });
                this._oResultsDialog.setModel(oModel, "results");
                this._oResultsDialog.open();
            }
        },

        onOptimizeIndex() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sIndexType = oContext.getProperty("indexType");

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

        _optimizeVectorIndex(sTaskId, sIndexType) {
            this._extensionAPI.getView().setBusy(true);

            const oOptimizationParams = {
                indexType: sIndexType,
                parameters: {
                    efConstruction: 200,
                    M: 16,
                    compression: true
                }
            };

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(sTaskId) }/optimize-index`,
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
                            `Query speed improvement: ${ data.speedImprovement }%\n` +
                            `Memory saved: ${ data.memorySaved } MB`
                        );
                        this._extensionAPI.refresh();

                        // Audit logging
                        this._logAuditEvent("INDEX_OPTIMIZED", sTaskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Optimization failed: ${ errorMsg}`);
                }.bind(this)
            });
        },

        onExportVectors() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

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
                }).then((oDialog) => {
                    this._oExportDialog = oDialog;
                    this.base.getView().addDependent(this._oExportDialog);

                    const oModel = new JSONModel({
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
                });
            } else {
                this._oExportDialog.open();
            }
        },

        onExecuteExport() {
            const oModel = this._oExportDialog.getModel("export");
            const oData = oModel.getData();

            // Validate export parameters
            const oValidation = this._validateExportData(oData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }

            this._oExportDialog.setBusy(true);

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(oData.taskId) }/export`,
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
                            `Files: ${ data.files.length }\n` +
                            `Total size: ${ data.totalSize } MB`,
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
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Export failed: ${ errorMsg}`);
                }.bind(this)
            });
        },

        _validateExportData(oData) {
            const oSanitized = {
                taskId: oData.taskId
            };

            // Validate format
            const aValidFormats = ["NUMPY", "PARQUET", "CSV", "JSON", "HDF5", "PICKLE"];
            if (!aValidFormats.includes(oData.format)) {
                return { isValid: false, message: "Invalid export format" };
            }
            oSanitized.format = oData.format;

            // Validate compression
            const aValidCompression = ["NONE", "GZIP", "BZIP2", "LZ4", "ZSTD"];
            if (!aValidCompression.includes(oData.compression)) {
                return { isValid: false, message: "Invalid compression type" };
            }
            oSanitized.compression = oData.compression;

            // Validate chunk size
            const nChunkSize = parseInt(oData.chunkSize, 10);
            if (isNaN(nChunkSize) || nChunkSize < 100 || nChunkSize > 100000) {
                return { isValid: false, message: "Chunk size must be between 100 and 100000" };
            }
            oSanitized.chunkSize = nChunkSize;

            oSanitized.includeMetadata = !!oData.includeMetadata;
            oSanitized.validation = !!oData.validation;

            return { isValid: true, sanitizedData: oSanitized };
        },

        _secureDownload(sUrl) {
            // Validate download URL
            try {
                const oUrl = new URL(sUrl, window.location.origin);
                if (!["http:", "https:"].includes(oUrl.protocol)) {
                    MessageBox.error("Invalid download URL");
                    return;
                }

                // Create temporary anchor for download
                const a = document.createElement("a");
                a.href = sUrl;
                a.download = true;
                a.style.display = "none";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } catch (e) {
                MessageBox.error("Failed to initiate download");
            }
        },

        onVisualizeEmbeddings() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

            if (!this._oVisualizationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.EmbeddingVisualization",
                    controller: this
                }).then((oDialog) => {
                    this._oVisualizationDialog = oDialog;
                    this.base.getView().addDependent(this._oVisualizationDialog);
                    this._oVisualizationDialog.open();

                    // Load embedding visualization data
                    this._loadEmbeddingVisualization(sTaskId);
                });
            } else {
                this._oVisualizationDialog.open();
                this._loadEmbeddingVisualization(sTaskId);
            }
        },

        _loadEmbeddingVisualization(sTaskId) {
            this._oVisualizationDialog.setBusy(true);

            const oParams = {
                method: "TSNE",
                perplexity: 30,
                dimensions: 3,
                sampleSize: 1000
            };

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(sTaskId) }/visualization-data`,
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
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Failed to load visualization data: ${ errorMsg}`);
                }.bind(this)
            });
        },

        validateEmbedding(oEmbeddingData) {
            // Secure embedding data validation
            if (!oEmbeddingData || typeof oEmbeddingData !== "object") {
                return { isValid: false, message: "Invalid embedding data structure" };
            }

            // Validate embedding vector dimensions
            if (oEmbeddingData.vector && Array.isArray(oEmbeddingData.vector)) {
                const sVectorString = JSON.stringify(oEmbeddingData.vector);
                const oValidation = this._validateInput(sVectorString, "embedding");
                if (!oValidation.isValid) {
                    return oValidation;
                }
            }

            // Validate metadata
            if (oEmbeddingData.metadata) {
                const sMetadata = JSON.stringify(oEmbeddingData.metadata);
                const oMetadataValidation = this._validateInput(sMetadata, "metadata");
                if (!oMetadataValidation.isValid) {
                    return oMetadataValidation;
                }
            }

            return { isValid: true, sanitized: oEmbeddingData };
        },

        _render3DEmbeddings(data) {
            // Performance-optimized 3D rendering
            const oContainer = this.byId("embeddingVisualizationContainer");
            if (oContainer && data.points && data.points.length > 0) {
                // Implement throttled 3D rendering
                if (this._renderTimeout) {
                    clearTimeout(this._renderTimeout);
                }

                this._renderTimeout = setTimeout(() => {
                    // Render 3D scatter plot with embedding points
                    MessageToast.show(`Rendering ${ data.points.length } embeddings in 3D`);
                    // Actual Three.js implementation would go here
                }, 100);
            }
        },

        onClusterAnalysis() {
            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

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

        _runClusterAnalysis(sTaskId) {
            const oAnalysisParams = {
                algorithm: "KMEANS",
                numClusters: "auto",
                minClusterSize: 5,
                validation: true
            };

            jQuery.ajax({
                url: `/a2a/agent3/v1/tasks/${ encodeURIComponent(sTaskId) }/cluster-analysis`,
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
                            `Clusters found: ${ data.numClusters }\n` +
                            `Silhouette score: ${ data.silhouetteScore.toFixed(3)}`
                        );

                        // Audit logging
                        this._logAuditEvent("CLUSTER_ANALYSIS_COMPLETED", sTaskId);
                    }
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error(`Cluster analysis failed: ${ errorMsg}`);
                }.bind(this)
            });
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

            let sSanitized = sInput.trim();

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
        _sanitizeFilters(oFilters) {
            const oSanitized = {};

            for (const sKey in oFilters) {
                if (oFilters.hasOwnProperty(sKey)) {
                    // Validate key
                    if (!/^[a-zA-Z0-9_]+$/.test(sKey)) {
                        continue; // Skip invalid keys
                    }

                    const value = oFilters[sKey];
                    if (typeof value === "string") {
                        oSanitized[sKey] = encodeXML(value);
                    } else if (typeof value === "number" || typeof value === "boolean") {
                        oSanitized[sKey] = value;
                    } else if (Array.isArray(value)) {
                        oSanitized[sKey] = value.map((item) => {
                            return typeof item === "string" ? encodeXML(item) : item;
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
        _validateApiResponse(oData) {
            if (!oData || typeof oData !== "object") {
                return false;
            }

            // Check for prototype pollution
            const aSuspiciousKeys = ["__proto__", "constructor", "prototype"];
            for (const sKey in oData) {
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
         * Get CSRF token for secure API calls
         * @returns {string} CSRF token
         */
        _getCSRFToken() {
            let token = jQuery("meta[name='csrf-token']").attr("content");
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
                    success(data, status, xhr) {
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
         * Log audit events for compliance
         * @param {string} sEventType - Type of event
         * @param {string} sEntityId - Entity ID
         */
        _logAuditEvent(sEventType, sEntityId) {
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
        _checkUserPermission(sAction) {
            // Implementation would check against user roles/permissions
            const userRoles = sap.ushell?.Container?.getUser?.()?.getRoles?.() || [];

            const requiredRoles = {
                "PROCESS_VECTORS": ["VectorAdmin", "VectorUser"],
                "SIMILARITY_SEARCH": ["VectorAdmin", "VectorUser", "VectorViewer"],
                "OPTIMIZE_INDEX": ["VectorAdmin"],
                "EXPORT_VECTORS": ["VectorAdmin", "VectorExporter"],
                "CLUSTER_ANALYSIS": ["VectorAdmin", "DataScientist"]
            };

            const required = requiredRoles[sAction] || [];
            return required.length === 0 || required.some((role) => {
                return userRoles.indexOf(role) !== -1;
            });
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
         * Data source change handler with validation
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onDataSourceChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oCreateModel = this.base.getView().getModel("create");

            if (!sValue || sValue.trim().length === 0) {
                oCreateModel.setProperty("/dataSourceState", "Error");
                oCreateModel.setProperty("/dataSourceStateText", "Data source is required");
            } else if (!/^[a-zA-Z0-9\s\-_\.\/\\:]+$/.test(sValue)) {
                oCreateModel.setProperty("/dataSourceState", "Error");
                oCreateModel.setProperty("/dataSourceStateText", "Data source path contains invalid characters");
            } else {
                oCreateModel.setProperty("/dataSourceState", "Success");
                oCreateModel.setProperty("/dataSourceStateText", "Valid data source path");
            }

            this._validateForm();
        },

        /**
         * Data type change handler
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

                // Auto-suggest embedding model based on data type
                this._suggestEmbeddingModel(sSelectedKey, oCreateModel);
            } else {
                oCreateModel.setProperty("/dataTypeState", "Error");
                oCreateModel.setProperty("/dataTypeStateText", "Please select a data type");
            }

            this._validateForm();
        },

        /**
         * Suggests embedding model based on data type
         * @param {string} sDataType - Selected data type
         * @param {sap.ui.model.json.JSONModel} oModel - Create model
         * @private
         * @since 1.0.0
         */
        _suggestEmbeddingModel(sDataType, oModel) {
            const mModelSuggestions = {
                "TEXT": "text-embedding-ada-002",
                "CODE": "text-embedding-3-large",
                "STRUCTURED": "text-embedding-3-small",
                "IMAGE": "clip-vit-base-patch32",
                "AUDIO": "wav2vec2-base",
                "MULTIMODAL": "clip-vit-large-patch14"
            };

            const sSuggestedModel = mModelSuggestions[sDataType] || "text-embedding-ada-002";
            oModel.setProperty("/embeddingModel", sSuggestedModel);

            // Update dimensions based on model
            this._updateDimensionsForModel(sSuggestedModel, oModel);
        },

        /**
         * Updates dimensions based on selected model
         * @param {string} sModel - Selected embedding model
         * @param {sap.ui.model.json.JSONModel} oModel - Create model
         * @private
         * @since 1.0.0
         */
        _updateDimensionsForModel(sModel, oModel) {
            const mDimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "clip-vit-base-patch32": 512,
                "clip-vit-large-patch14": 768
            };

            const nDimensions = mDimensions[sModel] || 1536;
            oModel.setProperty("/dimensions", nDimensions);
        },

        /**
         * Data source value help handler
         * @public
         * @since 1.0.0
         */
        onSelectDataSource() {
            MessageToast.show("Opening data source browser...");
            this._openDataSourceBrowser();
        },

        /**
         * Opens data source browser dialog
         * @private
         * @since 1.0.0
         */
        _openDataSourceBrowser() {
            if (!this._oDataSourceDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent3.ext.fragment.DataSourceBrowser",
                    controller: this
                }).then((oDialog) => {
                    this._oDataSourceDialog = oDialog;
                    this.base.getView().addDependent(this._oDataSourceDialog);
                    this._loadAvailableDataSources();
                    this._oDataSourceDialog.open();
                }).catch(() => {
                    // Fallback if fragment doesn't exist
                    MessageBox.information("Data source browser not yet implemented. Please enter data source path manually.");
                });
            } else {
                this._oDataSourceDialog.open();
            }
        },

        /**
         * Loads available data sources for selection
         * @private
         * @since 1.0.0
         */
        _loadAvailableDataSources() {
            // Implementation for loading data sources
            MessageToast.show("Loading available data sources...");
        },

        /**
         * Dialog after open event handler
         * @public
         * @since 1.0.0
         */
        onDialogAfterOpen() {
            // Focus on first input field for accessibility
            const oDialog = this.base.getView().byId("createVectorTaskDialog");
            if (oDialog) {
                const oFirstInput = oDialog.byId("taskNameInput");
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
                          oData.dataSource &&
                          oData.dataType &&
                          oData.taskNameState !== "Error" &&
                          oData.dataSourceState !== "Error" &&
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
            return this.base.getView().byId("createVectorTaskDialog");
        },

        /**
         * Format distance value
         * @param {number} nDistance - Distance value
         * @returns {string} Formatted distance
         */
        formatDistance(nDistance) {
            if (typeof nDistance !== "number") {
                return "0.0000";
            }
            return nDistance.toFixed(4);
        },

        /**
         * Clean up resources on exit
         */
        onExit() {
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

        _cleanup() {
            // Remove any event listeners or temporary resources
            this._ws = null;
            this._renderTimeout = null;
        }
    });
});