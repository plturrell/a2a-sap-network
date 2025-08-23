sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "../utils/SecurityUtils"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, encodeXML, escapeRegExp, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent1.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._resourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                // Initialize dialog cache for better performance
                this._dialogCache = {};
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
            }
        },

        /**
         * Initialize the create model with default values and validation states
         * @private
         * @since 1.0.0
         */
        _initializeCreateModel: function() {
            var oCreateModel = new sap.ui.model.json.JSONModel({
                taskName: "",
                description: "",
                sourceFormat: "",
                targetFormat: "",
                schemaTemplateId: "",
                schemaValidation: true,
                dataTypeValidation: true,
                formatValidation: false,
                processingMode: 0,
                batchSize: 1000,
                isValid: false,
                taskNameState: "None",
                taskNameStateText: "",
                sourceFormatState: "None",
                sourceFormatStateText: "",
                targetFormatState: "None",
                targetFormatStateText: ""
            });
            this.base.getView().setModel(oCreateModel, "create");
        },

        /**
         * Opens the create standardization task dialog.
         * Creates and caches the dialog fragment on first use for better performance.
         * @public
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateStandardizationTask: function() {
            // Initialize create model before opening dialog
            this._initializeCreateModel();
            this._openCachedDialog("CreateStandardizationTask", "a2a.network.agent1.ext.fragment.CreateStandardizationTask");
        },

        /**
         * Opens the import schema dialog for template management.
         * Allows users to import predefined or custom schema templates.
         * @public
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onImportSchema: function() {
            this._openCachedDialog("ImportSchema", "a2a.network.agent1.ext.fragment.ImportSchema");
        },

        /**
         * Initiates batch processing for selected standardization tasks.
         * Validates selection and confirms operation before execution.
         * @public
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @since 1.0.0
         * @throws {Error} When no tasks are selected or batch processing fails
         */
        onBatchProcess: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning(this._getResourceBundle().getText("error.selectTasksForBatch"));
                return;
            }
            
            var aTaskNames = aSelectedContexts.map(function(oContext) {
                return oContext.getProperty("taskName");
            });
            
            MessageBox.confirm(
                this._getResourceBundle().getText("confirm.batchProcessingWithTasks", [
                    aSelectedContexts.length,
                    aTaskNames.join(", ")
                ]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchProcessing(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * Executes batch processing for the provided task contexts.
         * Sends parallel processing request to backend service with high priority.
         * @private
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @param {Array<sap.ui.model.Context>} aContexts - Array of selected task contexts
         * @since 1.0.0
         */
        _startBatchProcessing: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            // Show busy indicator
            this.base.getView().setBusy(true);
            
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/batch-process",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    priority: "HIGH"
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        this._getResourceBundle().getText("success.batchProcessingStarted", [
                            data.jobId,
                            data.taskCount
                        ])
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error(this._getResourceBundle().getText("error.batchProcessingFailed", [xhr.responseText]));
                }.bind(this)
            });
        },

        onSchemaTemplates: function() {
            // Navigate to schema templates
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("SchemaTemplates");
        },

        /**
         * Opens the format analyzer dialog and loads current format statistics.
         * Provides insights into data format distribution and usage patterns.
         * @public
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onAnalyzeFormats: function() {
            this._openCachedDialog("FormatAnalyzer", "a2a.network.agent1.ext.fragment.FormatAnalyzer", function() {
                this._loadFormatStatistics();
            }.bind(this));
        },

        /**
         * Loads format statistics from the backend service.
         * Includes CSRF protection and response validation for security.
         * @private
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @since 1.0.0
         */
        _loadFormatStatistics: function() {
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/format-statistics",
                type: "GET",
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data) {
                    if (this._validateApiResponse(data)) {
                        var oModel = new sap.ui.model.json.JSONModel(data);
                        this._oFormatAnalyzer.setModel(oModel, "stats");
                    }
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load format statistics");
                }
            });
        },

        /**
         * Opens a cached dialog fragment with optimized loading.
         * Creates and caches dialogs on first use to improve performance.
         * @private
         * @memberof a2a.network.agent1.ext.controller.ListReportExt
         * @param {string} sDialogKey - Unique key for caching the dialog
         * @param {string} sFragmentName - Fragment name to load
         * @param {function} [fnCallback] - Optional callback after dialog opens
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
                        fnCallback();
                    }
                }.bind(this));
            } else {
                this._dialogCache[sDialogKey].open();
                if (fnCallback) {
                    fnCallback();
                }
            }
        },

        /**
         * Validate user input for security and format compliance
         * @param {string} sInput - Input to validate
         * @param {string} sType - Type of validation (text, filename, email, etc.)
         * @returns {object} Validation result with isValid flag and message
         */
        _validateInput: function(sInput, sType) {
            if (!sInput || typeof sInput !== 'string') {
                return { isValid: false, message: "Input is required" };
            }

            // Sanitize input
            var sSanitized = sInput.trim();
            
            // Check for XSS patterns
            var aXSSPatterns = [
                /<script/i,
                /javascript:/i,
                /on\w+\s*=/i,
                /<iframe/i,
                /<object/i,
                /<embed/i
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
                        return { isValid: false, message: "Task name must be 3-100 characters" };
                    }
                    if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Task name contains invalid characters" };
                    }
                    break;
                
                case "filename":
                    if (sSanitized.length > 255) {
                        return { isValid: false, message: "Filename too long" };
                    }
                    if (!/^[a-zA-Z0-9\s\-_\.]+\.[a-zA-Z0-9]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Invalid filename format" };
                    }
                    // Check for dangerous extensions
                    var aDangerousExt = ['.exe', '.bat', '.cmd', '.scr', '.vbs', '.js'];
                    var sExt = sSanitized.toLowerCase().substring(sSanitized.lastIndexOf('.'));
                    if (aDangerousExt.indexOf(sExt) !== -1) {
                        return { isValid: false, message: "File type not allowed" };
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
                
                case "jsonSchema":
                    try {
                        JSON.parse(sSanitized);
                    } catch (e) {
                        return { isValid: false, message: "Invalid JSON format" };
                    }
                    break;
                
                default:
                    // General text validation
                    if (sSanitized.length > 10000) {
                        return { isValid: false, message: "Input too long" };
                    }
            }

            return { isValid: true, sanitized: sSanitized };
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

            // Check for suspicious properties
            var aSuspiciousKeys = ['__proto__', 'constructor', 'prototype'];
            for (var sKey in oData) {
                if (aSuspiciousKeys.indexOf(sKey) !== -1) {
                    return false;
                }
            }

            return true;
        },

        /**
         * Validate file upload security
         * @param {File} oFile - File object
         * @returns {object} Validation result
         */
        _validateFileUpload: function(oFile) {
            if (!oFile) {
                return { isValid: false, message: "No file selected" };
            }

            // Check file size (max 50MB)
            var nMaxSize = 50 * 1024 * 1024;
            if (oFile.size > nMaxSize) {
                return { isValid: false, message: "File size exceeds 50MB limit" };
            }

            // Check file type
            var aAllowedTypes = [
                'text/csv',
                'application/json',
                'application/xml',
                'text/xml',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/octet-stream' // for Parquet/Avro
            ];

            if (aAllowedTypes.indexOf(oFile.type) === -1) {
                // Additional check by extension if MIME type is generic
                var sFileName = oFile.name.toLowerCase();
                var aAllowedExt = ['.csv', '.json', '.xml', '.xls', '.xlsx', '.parquet', '.avro'];
                var bValidExt = aAllowedExt.some(function(sExt) {
                    return sFileName.endsWith(sExt);
                });

                if (!bValidExt) {
                    return { isValid: false, message: "File type not supported" };
                }
            }

            // Validate filename
            var oNameValidation = this._validateInput(oFile.name, "filename");
            if (!oNameValidation.isValid) {
                return oNameValidation;
            }

            return { isValid: true };
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
         * Event handler for task name input changes with validation
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
        },

        /**
         * Event handler for template name input changes with validation
         */
        onTemplateNameChange: function(oEvent) {
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
        },

        /**
         * Event handler for schema URL input changes with validation
         */
        onSchemaUrlChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oInput = oEvent.getSource();
            var oValidation = this._validateInput(sValue, "url");
            
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