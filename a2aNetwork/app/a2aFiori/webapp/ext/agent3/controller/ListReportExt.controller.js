sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "sap/base/Log"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, encodeXML, escapeRegExp, sanitizeHTML, Log) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent3.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize dialog cache for better performance
                this._dialogCache = {};
                
                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
            }
        },

        onCreateVectorTask: function() {
            // Initialize create model before opening dialog
            this._initializeCreateModel();
            this._openCachedDialog("createVectorTask", "a2a.network.agent3.ext.fragment.CreateVectorTask");
        },

        /**
         * Initialize the create model with default values and validation states
         * @private
         * @since 1.0.0
         */
        _initializeCreateModel: function() {
            var oCreateModel = new JSONModel({
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

        /**
         * Opens cached dialog fragments with optimized loading
         * @private
         * @param {string} sDialogKey - Dialog cache key
         * @param {string} sFragmentName - Fragment name to load
         * @param {function} [fnCallback] - Optional callback after opening
         * @since 1.0.0
         */
        _openCachedDialog: function(sDialogKey, sFragmentName, fnCallback) {
            var oView = this.base.getView();
            
            if (!this._dialogCache[sDialogKey]) {
                Fragment.load({
                    id: oView.getId(),
                    name: sFragmentName,
                    controller: this
                }).then(function(oDialog) {
                    this._dialogCache[sDialogKey] = oDialog;
                    oView.addDependent(oDialog);
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
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data) {
                    if (this._validateApiResponse(data)) {
                        var oSearchModel = this._oVectorSearchDialog.getModel("search");
                        oSearchModel.setProperty("/collections", data.collections);
                    }
                }.bind(this),
                error: function(xhr) {
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to load vector collections: " + errorMsg);
                }.bind(this)
            });
        },

        onExecuteVectorSearch: function() {
            var oSearchModel = this._oVectorSearchDialog.getModel("search");
            var oSearchData = oSearchModel.getData();
            
            // Validate input
            var oValidation = this._validateSearchQuery(oSearchData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }
            
            this._oVectorSearchDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/search",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oValidation.sanitizedData),
                success: function(data) {
                    this._oVectorSearchDialog.setBusy(false);
                    if (this._validateApiResponse(data)) {
                        oSearchModel.setProperty("/searchResults", data.results);
                        
                        if (data.results.length === 0) {
                            MessageToast.show("No similar vectors found");
                        } else {
                            MessageToast.show("Found " + data.results.length + " similar vectors");
                        }
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oVectorSearchDialog.setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Search failed: " + errorMsg);
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
                    
                    // Initialize 3D visualization with throttling
                    this._init3DVisualization();
                }.bind(this));
            } else {
                this._oVisualizationDialog.open();
                this._refresh3DVisualization();
            }
        },

        _init3DVisualization: function() {
            // Initialize Three.js with performance optimization
            setTimeout(function() {
                var oContainer = this.byId("visualization3DContainer");
                if (oContainer) {
                    // Create 3D scene with requestAnimationFrame throttling
                    this._create3DScene(oContainer.getDomRef());
                }
            }.bind(this), 100);
        },

        _create3DScene: function(container) {
            // Performance-optimized 3D visualization
            if (this._animationFrameId) {
                cancelAnimationFrame(this._animationFrameId);
            }
            
            // Throttled render function
            var lastRenderTime = 0;
            var renderThrottle = 16; // ~60fps
            
            var render = function(currentTime) {
                if (currentTime - lastRenderTime > renderThrottle) {
                    // Render 3D scene here
                    lastRenderTime = currentTime;
                }
                this._animationFrameId = requestAnimationFrame(render);
            }.bind(this);
            
            MessageToast.show("3D visualization loading...");
            this._animationFrameId = requestAnimationFrame(render);
        },

        _refresh3DVisualization: function() {
            // Cleanup and reinitialize 3D scene
            if (this._animationFrameId) {
                cancelAnimationFrame(this._animationFrameId);
            }
            this._init3DVisualization();
        },

        onBatchProcessing: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch processing.");
                return;
            }
            
            if (aSelectedContexts.length > 100) {
                MessageBox.warning("Maximum 100 tasks can be processed in a single batch.");
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
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    useGPU: true,
                    priority: "HIGH",
                    chunkSize: 10 // Process in chunks for large datasets
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    if (this._validateApiResponse(data)) {
                        MessageBox.success(
                            "Batch processing started!\n" +
                            "Job ID: " + data.jobId + "\n" +
                            "Estimated vectors: " + data.estimatedVectors
                        );
                        this._extensionAPI.refresh();
                    }
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Batch processing failed: " + errorMsg);
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
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data) {
                    if (this._validateApiResponse(data)) {
                        var oModel = new JSONModel(data);
                        this._oModelComparisonDialog.setModel(oModel, "comparison");
                    }
                }.bind(this),
                error: function(xhr) {
                    var errorMsg = this._sanitizeErrorMessage(xhr.responseText || xhr.statusText);
                    MessageBox.error("Failed to load model comparison data: " + errorMsg);
                }.bind(this)
            });
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Comprehensive validation
            var oValidation = this._validateVectorTaskData(oData);
            if (!oValidation.isValid) {
                MessageBox.error(oValidation.message);
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent3/v1/tasks",
                type: "POST",
                contentType: "application/json",
                headers: {
                    "X-CSRF-Token": this._getCSRFToken(),
                    "X-Requested-With": "XMLHttpRequest"
                },
                data: JSON.stringify(oValidation.sanitizedData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    if (this._validateApiResponse(data)) {
                        this._oCreateDialog.close();
                        MessageToast.show("Vector processing task created successfully");
                        this._extensionAPI.refresh();
                        
                        // Audit logging
                        this._logAuditEvent("VECTOR_TASK_CREATED", data.taskId);
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
         * Validate vector task data for security and correctness
         * @param {object} oData - Task data to validate
         * @returns {object} Validation result with sanitized data
         */
        _validateVectorTaskData: function(oData) {
            if (!oData || typeof oData !== 'object') {
                return { isValid: false, message: "Invalid task data" };
            }

            var oSanitized = {};
            
            // Validate task name
            var oNameValidation = this._validateInput(oData.taskName, "taskName");
            if (!oNameValidation.isValid) {
                return { isValid: false, message: "Task name: " + oNameValidation.message };
            }
            oSanitized.taskName = oNameValidation.sanitized;

            // Validate data source path
            var oSourceValidation = this._validateInput(oData.dataSource, "path");
            if (!oSourceValidation.isValid) {
                return { isValid: false, message: "Data source: " + oSourceValidation.message };
            }
            oSanitized.dataSource = oSourceValidation.sanitized;

            // Validate collection name
            if (oData.collectionName) {
                var oCollectionValidation = this._validateInput(oData.collectionName, "collectionName");
                if (!oCollectionValidation.isValid) {
                    return { isValid: false, message: "Collection name: " + oCollectionValidation.message };
                }
                oSanitized.collectionName = oCollectionValidation.sanitized;
            }

            // Validate metadata schema JSON
            if (oData.metadataSchema) {
                try {
                    var oSchema = JSON.parse(oData.metadataSchema);
                    oSanitized.metadataSchema = JSON.stringify(oSchema);
                } catch (e) {
                    return { isValid: false, message: "Invalid metadata schema JSON" };
                }
            }

            // Validate numeric fields
            var nDimensions = parseInt(oData.dimensions, 10);
            if (isNaN(nDimensions) || nDimensions < 128 || nDimensions > 4096) {
                return { isValid: false, message: "Dimensions must be between 128 and 4096" };
            }
            oSanitized.dimensions = nDimensions;

            var nChunkSize = parseInt(oData.chunkSize, 10);
            if (isNaN(nChunkSize) || nChunkSize < 128 || nChunkSize > 2048) {
                return { isValid: false, message: "Chunk size must be between 128 and 2048" };
            }
            oSanitized.chunkSize = nChunkSize;

            // Copy other safe fields
            var aSafeFields = ['description', 'dataType', 'embeddingModel', 'modelProvider', 
                              'vectorDatabase', 'indexType', 'distanceMetric', 'chunkOverlap',
                              'normalization', 'batchSize', 'parallelProcessing', 'useGPU',
                              'errorHandling', 'progressMonitoring', 'incrementalUpdates'];
            
            aSafeFields.forEach(function(sField) {
                if (oData[sField] !== undefined) {
                    if (typeof oData[sField] === 'string') {
                        oSanitized[sField] = encodeXML(oData[sField]);
                    } else {
                        oSanitized[sField] = oData[sField];
                    }
                }
            });

            return { isValid: true, sanitizedData: oSanitized };
        },

        /**
         * Validate search query data
         * @param {object} oData - Search data to validate
         * @returns {object} Validation result
         */
        _validateSearchQuery: function(oData) {
            if (!oData.query || !oData.collection) {
                return { isValid: false, message: "Please enter a query and select a collection" };
            }

            var oSanitized = {};
            
            // Validate and sanitize query
            var oQueryValidation = this._validateInput(oData.query, "query");
            if (!oQueryValidation.isValid) {
                return { isValid: false, message: "Query: " + oQueryValidation.message };
            }
            oSanitized.query = oQueryValidation.sanitized;

            // Validate collection name
            var oCollectionValidation = this._validateInput(oData.collection, "collectionName");
            if (!oCollectionValidation.isValid) {
                return { isValid: false, message: "Collection: " + oCollectionValidation.message };
            }
            oSanitized.collection = oCollectionValidation.sanitized;

            // Validate numeric parameters
            var nTopK = parseInt(oData.topK, 10);
            if (isNaN(nTopK) || nTopK < 1 || nTopK > 100) {
                return { isValid: false, message: "Top K must be between 1 and 100" };
            }
            oSanitized.topK = nTopK;

            var nThreshold = parseFloat(oData.similarityThreshold);
            if (isNaN(nThreshold) || nThreshold < 0 || nThreshold > 1) {
                return { isValid: false, message: "Similarity threshold must be between 0 and 1" };
            }
            oSanitized.threshold = nThreshold;

            // Validate filters if present
            if (oData.filters && typeof oData.filters === 'object') {
                oSanitized.filters = this._sanitizeFilters(oData.filters);
            }

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
                
                case "collectionName":
                    if (sSanitized.length < 3 || sSanitized.length > 50) {
                        return { isValid: false, message: "Must be 3-50 characters" };
                    }
                    if (!/^[a-zA-Z0-9\-_]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Only alphanumeric, dash, and underscore allowed" };
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
                    // Prevent path traversal
                    if (/\.\./.test(sSanitized)) {
                        return { isValid: false, message: "Path traversal not allowed" };
                    }
                    break;
                
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
                "CREATE_VECTOR_TASK": ["VectorAdmin", "VectorUser"],
                "BATCH_PROCESSING": ["VectorAdmin"],
                "EXPORT_VECTORS": ["VectorAdmin", "VectorExporter"],
                "MANAGE_COLLECTIONS": ["VectorAdmin"]
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
         * Format similarity score
         * @param {number} nScore - Similarity score
         * @returns {string} Formatted score
         */
        formatSimilarityScore: function(nScore) {
            if (typeof nScore !== 'number') {
                return "0.000";
            }
            return nScore.toFixed(3);
        },

        /**
         * Clean up resources on exit
         */
        onExit: function() {
            // Clean up 3D visualization
            if (this._animationFrameId) {
                cancelAnimationFrame(this._animationFrameId);
            }
            
            // Clean up any WebSocket connections
            if (this._ws) {
                this._ws.close();
            }
            
            // Clean up event listeners
            this._cleanup();
        },

        _cleanup: function() {
            // Remove any global event listeners
            if (this._resizeHandler) {
                window.removeEventListener('resize', this._resizeHandler);
            }
        }
    });
});