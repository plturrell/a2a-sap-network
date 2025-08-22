sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, encodeXML, escapeRegExp) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent1.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateStandardizationTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.CreateStandardizationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onImportSchema: function() {
            var oView = this.base.getView();
            
            if (!this._oImportSchemaDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ImportSchema",
                    controller: this
                }).then(function(oDialog) {
                    this._oImportSchemaDialog = oDialog;
                    oView.addDependent(this._oImportSchemaDialog);
                    this._oImportSchemaDialog.open();
                }.bind(this));
            } else {
                this._oImportSchemaDialog.open();
            }
        },

        onBatchProcess: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one task for batch processing.");
                return;
            }
            
            var aTaskNames = aSelectedContexts.map(function(oContext) {
                return oContext.getProperty("taskName");
            });
            
            MessageBox.confirm(
                "Start batch processing for " + aSelectedContexts.length + " tasks?\n\n" +
                "Tasks: " + aTaskNames.join(", "),
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
            
            // Show busy indicator
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
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
                        "Batch processing started successfully!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Processing " + data.taskCount + " tasks"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch processing failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onSchemaTemplates: function() {
            // Navigate to schema templates
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("SchemaTemplates");
        },

        onAnalyzeFormats: function() {
            var oView = this.base.getView();
            
            if (!this._oFormatAnalyzer) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.FormatAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormatAnalyzer = oDialog;
                    oView.addDependent(this._oFormatAnalyzer);
                    this._oFormatAnalyzer.open();
                    this._loadFormatStatistics();
                }.bind(this));
            } else {
                this._oFormatAnalyzer.open();
                this._loadFormatStatistics();
            }
        },

        _loadFormatStatistics: function() {
            jQuery.ajax({
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
        }
    });
});