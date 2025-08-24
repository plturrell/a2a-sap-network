sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "../utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, encodeXML, escapeRegExp, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent1.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._resourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                // Initialize dialog cache and search filters
                this._dialogCache = {};
                this._searchFilters = {
                    sourceFields: [],
                    targetFields: []
                };
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize create model for dialog
                this._initializeCreateModel();
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

        onStartStandardization: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start standardization for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startStandardizationProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startStandardizationProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/tasks/" + encodeURIComponent(sTaskId) + "/start",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Standardization process started");
                    this._extensionAPI.refresh();
                    
                    // Start monitoring progress
                    this._startProgressMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start standardization: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startProgressMonitoring: function(sTaskId) {
            // Poll for progress updates every 2 seconds
            this._progressInterval = setInterval(function() {
                this._securityUtils.secureAjaxRequest({
                    url: "/a2a/agent1/v1/tasks/" + sTaskId + "/progress",
                    type: "GET",
                    success: function(data) {
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(this._progressInterval);
                            this._extensionAPI.refresh();
                            
                            if (data.status === "COMPLETED") {
                                MessageBox.success("Standardization completed successfully!");
                            } else {
                                MessageBox.error("Standardization failed: " + data.error);
                            }
                        }
                    }.bind(this),
                    error: function() {
                        clearInterval(this._progressInterval);
                    }.bind(this)
                });
            }.bind(this), 2000);
        },

        onPauseStandardization: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm("Pause standardization process?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._pauseStandardization(sTaskId);
                    }
                }.bind(this)
            });
        },

        _pauseStandardization: function(sTaskId) {
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/tasks/" + sTaskId + "/pause",
                type: "POST",
                success: function() {
                    MessageToast.show("Standardization process paused");
                    clearInterval(this._progressInterval);
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to pause standardization: " + xhr.responseText);
                }
            });
        },

        onValidateMapping: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var oData = oContext.getObject();
            
            if (!oData.mappingRules || oData.mappingRules.length === 0) {
                MessageBox.warning("No mapping rules defined. Please define mapping rules first.");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/validate-mapping",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    sourceSchema: oData.sourceSchema,
                    targetSchema: oData.targetSchema,
                    mappingRules: oData.mappingRules
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    if (data.isValid) {
                        MessageBox.success(
                            "Schema mapping is valid!\n\n" +
                            "Coverage: " + data.coverage + "%\n" +
                            "Mapped fields: " + data.mappedFields + "/" + data.totalFields
                        );
                    } else {
                        this._showValidationErrors(data.errors);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showValidationErrors: function(aErrors) {
            var oView = this.base.getView();
            
            if (!this._oValidationErrorsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ValidationErrors",
                    controller: this
                }).then(function(oDialog) {
                    this._oValidationErrorsDialog = oDialog;
                    oView.addDependent(this._oValidationErrorsDialog);
                    
                    var oModel = new sap.ui.model.json.JSONModel({ errors: aErrors });
                    this._oValidationErrorsDialog.setModel(oModel, "errors");
                    this._oValidationErrorsDialog.open();
                }.bind(this));
            } else {
                var oModel = new sap.ui.model.json.JSONModel({ errors: aErrors });
                this._oValidationErrorsDialog.setModel(oModel, "errors");
                this._oValidationErrorsDialog.open();
            }
        },

        onExportResults: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            var oExportOptions = {
                formats: ["CSV", "JSON", "PARQUET"],
                includeErrors: true,
                includeMetadata: true
            };
            
            this._showExportDialog(sTaskId, sTaskName, oExportOptions);
        },

        _showExportDialog: function(sTaskId, sTaskName, oOptions) {
            var oView = this.base.getView();
            
            if (!this._oExportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ExportResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oExportDialog = oDialog;
                    oView.addDependent(this._oExportDialog);
                    
                    var oModel = new sap.ui.model.json.JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        options: oOptions,
                        selectedFormat: "CSV"
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                }.bind(this));
            } else {
                var oModel = new sap.ui.model.json.JSONModel({
                    taskId: sTaskId,
                    taskName: sTaskName,
                    options: oOptions,
                    selectedFormat: "CSV"
                });
                this._oExportDialog.setModel(oModel, "export");
                this._oExportDialog.open();
            }
        },

        onExecuteExport: function() {
            var oExportModel = this._oExportDialog.getModel("export");
            var oExportData = oExportModel.getData();
            
            this._oExportDialog.setBusy(true);
            
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/tasks/" + oExportData.taskId + "/export",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    format: oExportData.selectedFormat,
                    includeErrors: oExportData.options.includeErrors,
                    includeMetadata: oExportData.options.includeMetadata
                }),
                success: function(data) {
                    this._oExportDialog.setBusy(false);
                    this._oExportDialog.close();
                    
                    // Download the exported file
                    window.open(data.downloadUrl, "_blank");
                    MessageToast.show("Export completed successfully");
                }.bind(this),
                error: function(xhr) {
                    this._oExportDialog.setBusy(false);
                    MessageBox.error("Export failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onPreviewTransformation: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var oData = oContext.getObject();
            
            if (!oData.mappingRules || oData.mappingRules.length === 0) {
                MessageBox.warning("No mapping rules defined for preview.");
                return;
            }
            
            // Open preview dialog with sample data transformation
            this._showTransformationPreview(oData);
        },

        _showTransformationPreview: function(oTaskData) {
            var oView = this.base.getView();
            
            if (!this._oPreviewDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.TransformationPreview",
                    controller: this
                }).then(function(oDialog) {
                    this._oPreviewDialog = oDialog;
                    oView.addDependent(this._oPreviewDialog);
                    
                    this._loadPreviewData(oTaskData);
                    this._oPreviewDialog.open();
                }.bind(this));
            } else {
                this._loadPreviewData(oTaskData);
                this._oPreviewDialog.open();
            }
        },

        _loadPreviewData: function(oTaskData) {
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/preview-transformation",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    sourceSchema: oTaskData.sourceSchema,
                    targetSchema: oTaskData.targetSchema,
                    mappingRules: oTaskData.mappingRules,
                    sampleSize: 5
                }),
                success: function(data) {
                    var oModel = new sap.ui.model.json.JSONModel(data);
                    this._oPreviewDialog.setModel(oModel, "preview");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to generate preview: " + xhr.responseText);
                }
            });
        },

        /**
         * Search handler for source fields with virtualization support
         * @param {sap.ui.base.Event} oEvent - Search event
         * @public
         * @since 1.0.0
         */
        onSearchSourceFields: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            this._filterSchemaFields("sourceSchemaTree", sQuery, "sourceFields");
        },

        /**
         * Search handler for target fields with virtualization support
         * @param {sap.ui.base.Event} oEvent - Search event
         * @public
         * @since 1.0.0
         */
        onSearchTargetFields: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            this._filterSchemaFields("targetSchemaTree", sQuery, "targetFields");
        },

        /**
         * Generic field filtering method for schema trees
         * @param {string} sTreeId - Tree control ID
         * @param {string} sQuery - Search query
         * @param {string} sFieldType - Field type for caching
         * @private
         * @since 1.0.0
         */
        _filterSchemaFields: function(sTreeId, sQuery, sFieldType) {
            var oTree = sap.ui.getCore().byId(sTreeId);
            if (!oTree) return;

            var oBinding = oTree.getBinding("items");
            var aFilters = [];

            if (sQuery && sQuery.length > 0) {
                var oFilter = new sap.ui.model.Filter([
                    new sap.ui.model.Filter("fieldName", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("dataType", sap.ui.model.FilterOperator.Contains, sQuery)
                ], false);
                aFilters.push(oFilter);
            }

            oBinding.filter(aFilters);
            this._searchFilters[sFieldType] = aFilters;
        },

        /**
         * Data type icon formatter for schema fields
         * @param {string} sDataType - Data type
         * @returns {string} SAP icon name
         * @public
         * @since 1.0.0
         */
        getDataTypeIcon: function(sDataType) {
            switch (sDataType?.toLowerCase()) {
                case "string":
                case "text":
                    return "sap-icon://text";
                case "number":
                case "integer":
                case "decimal":
                    return "sap-icon://number-sign";
                case "date":
                case "datetime":
                    return "sap-icon://calendar";
                case "boolean":
                    return "sap-icon://accept";
                case "array":
                    return "sap-icon://list";
                case "object":
                    return "sap-icon://folder";
                default:
                    return "sap-icon://question-mark";
            }
        },

        /**
         * Source field selection handler for accessibility
         * @param {sap.ui.base.Event} oEvent - Selection event
         * @public
         * @since 1.0.0
         */
        onSourceFieldSelect: function(oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            var oContext = oSelectedItem.getBindingContext("mapping");
            var sFieldName = oContext.getProperty("fieldName");
            
            // Announce selection to screen readers
            sap.ui.getCore().announceForAccessibility(
                "Selected source field: " + sFieldName + 
                ", type: " + oContext.getProperty("dataType")
            );
        },

        /**
         * Target field selection handler for accessibility
         * @param {sap.ui.base.Event} oEvent - Selection event
         * @public
         * @since 1.0.0
         */
        onTargetFieldSelect: function(oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            var oContext = oSelectedItem.getBindingContext("mapping");
            var sFieldName = oContext.getProperty("fieldName");
            
            // Announce selection to screen readers
            sap.ui.getCore().announceForAccessibility(
                "Selected target field: " + sFieldName + 
                ", type: " + oContext.getProperty("dataType")
            );
        },

        /**
         * Mapping rule press handler for keyboard navigation
         * @param {sap.ui.base.Event} oEvent - Press event
         * @public
         * @since 1.0.0
         */
        onMappingRulePress: function(oEvent) {
            var oContext = oEvent.getParameter("listItem").getBindingContext("mapping");
            var sSourceField = oContext.getProperty("sourceField");
            var sTargetField = oContext.getProperty("targetField");
            var sTransformation = oContext.getProperty("transformation");
            
            // Announce mapping details to screen readers
            sap.ui.getCore().announceForAccessibility(
                "Mapping rule: " + sSourceField + " transforms to " + 
                sTargetField + " using " + sTransformation + " method"
            );
        },

        /**
         * Script validation handler for transformation scripts
         * @public
         * @since 1.0.0
         */
        onValidateScript: function() {
            var oScriptArea = sap.ui.getCore().byId("transformationScript");
            if (!oScriptArea) return;

            var sScript = oScriptArea.getValue();
            
            try {
                // Validate script for security
                var validation = this._securityUtils.validateTransformationScript(sScript);
                if (!validation.isValid) {
                    throw new Error(validation.errors.join(", "));
                }
                
                // Announce success to screen readers
                sap.ui.getCore().announceForAccessibility("Script validation successful");
                MessageToast.show("Script syntax is valid");
                
                oScriptArea.setValueState("Success");
                oScriptArea.setValueStateText("Script syntax is valid");
                
            } catch (oError) {
                // Announce error to screen readers
                sap.ui.getCore().announceForAccessibility("Script validation failed: " + oError.message);
                MessageToast.show("Script syntax error: " + oError.message);
                
                oScriptArea.setValueState("Error");
                oScriptArea.setValueStateText("Syntax error: " + oError.message);
            }
        },

        /**
         * Script testing handler with sample data
         * @public
         * @since 1.0.0
         */
        onTestScript: function() {
            var oScriptArea = sap.ui.getCore().byId("transformationScript");
            if (!oScriptArea) return;

            var sScript = oScriptArea.getValue();
            
            // Validate script before execution
            var validation = this._securityUtils.validateTransformationScript(sScript);
            if (!validation.isValid) {
                MessageBox.error("Script validation failed: " + validation.errors.join(", "));
                return;
            }
            
            try {
                // Use secure sandbox evaluation instead of Function constructor
                var result = this._securityUtils.executeSecureTransformation(sScript, {
                    value: "Sample Value",
                    row: { field1: "data1", field2: "data2" },
                    context: { sourceFormat: "CSV", targetFormat: "JSON" }
                });
                
                // Announce test result to screen readers
                sap.ui.getCore().announceForAccessibility("Script test completed successfully. Result: " + result);
                MessageBox.information("Script test successful!\n\nInput: Sample Value\nOutput: " + result);
                
            } catch (oError) {
                // Announce test error to screen readers
                sap.ui.getCore().announceForAccessibility("Script test failed: " + oError.message);
                MessageBox.error("Script test failed: " + oError.message);
            }
        },

        /**
         * Script change handler for live validation
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onScriptChange: function(oEvent) {
            var oScriptArea = oEvent.getSource();
            var sScript = oEvent.getParameter("value");
            
            // Validate input parameter
            if (!this._securityUtils.validateInputParameter(sScript, 'string')) {
                oScriptArea.setValueState("Error");
                oScriptArea.setValueStateText("Invalid input");
                return;
            }
            
            // Reset validation state for live editing
            if (sScript.length === 0) {
                oScriptArea.setValueState("None");
                oScriptArea.setValueStateText("");
            }
        },

        /**
         * Event handler for source format selection changes
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onSourceFormatChange: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.base.getView().getModel("create");
            
            if (sSelectedKey) {
                oCreateModel.setProperty("/sourceFormatState", "None");
                oCreateModel.setProperty("/sourceFormatStateText", "");
            } else {
                oCreateModel.setProperty("/sourceFormatState", "Error");
                oCreateModel.setProperty("/sourceFormatStateText", "Please select a source format");
            }
            
            this._validateForm();
        },

        /**
         * Event handler for target format selection changes
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onTargetFormatChange: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.base.getView().getModel("create");
            
            if (sSelectedKey) {
                oCreateModel.setProperty("/targetFormatState", "None");
                oCreateModel.setProperty("/targetFormatStateText", "");
            } else {
                oCreateModel.setProperty("/targetFormatState", "Error");
                oCreateModel.setProperty("/targetFormatStateText", "Please select a target format");
            }
            
            this._validateForm();
        },

        /**
         * Enhanced task name change handler with real-time validation
         * @param {sap.ui.base.Event} oEvent - Change event
         * @public
         * @since 1.0.0
         */
        onTaskNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.base.getView().getModel("create");
            
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
         * Validates the entire form and updates the isValid flag
         * @private
         * @since 1.0.0
         */
        _validateForm: function() {
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            var bIsValid = oData.taskName && 
                          oData.taskName.trim().length >= 3 &&
                          oData.sourceFormat &&
                          oData.targetFormat &&
                          oData.taskNameState !== "Error" &&
                          oData.sourceFormatState !== "Error" &&
                          oData.targetFormatState !== "Error";
            
            oCreateModel.setProperty("/isValid", bIsValid);
        },

        /**
         * Cancel create dialog handler
         * @public
         * @since 1.0.0
         */
        onCancelCreate: function() {
            this._getCreateDialog().close();
            this._initializeCreateModel(); // Reset form
        },

        /**
         * Confirm create dialog handler
         * @public
         * @since 1.0.0
         */
        onConfirmCreate: function() {
            var oCreateModel = this.base.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!oData.isValid) {
                MessageBox.error("Please complete all required fields correctly.");
                return;
            }
            
            this.base.getView().setBusy(true);
            
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent1/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskName: oData.taskName.trim(),
                    description: oData.description?.trim() || "",
                    sourceFormat: oData.sourceFormat,
                    targetFormat: oData.targetFormat,
                    schemaTemplateId: oData.schemaTemplateId || null,
                    validationSettings: {
                        schemaValidation: oData.schemaValidation,
                        dataTypeValidation: oData.dataTypeValidation,
                        formatValidation: oData.formatValidation
                    },
                    processingOptions: {
                        mode: oData.processingMode === 0 ? "ENTIRE_FILE" : "BATCH",
                        batchSize: oData.processingMode === 1 ? oData.batchSize : null
                    }
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    this._getCreateDialog().close();
                    this._initializeCreateModel();
                    MessageToast.show("Standardization task created successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Failed to create task: " + xhr.responseText);
                }.bind(this)
            });
        },

        /**
         * Get or create the create dialog
         * @private
         * @returns {sap.m.Dialog} The create dialog
         * @since 1.0.0
         */
        _getCreateDialog: function() {
            if (!this._oCreateDialog) {
                this._oCreateDialog = sap.ui.xmlfragment(
                    this.base.getView().getId(),
                    "a2a.network.agent1.ext.fragment.CreateStandardizationTask",
                    this
                );
                this.base.getView().addDependent(this._oCreateDialog);
            }
            return this._oCreateDialog;
        },

        /**
         * Upload template handler
         * @public
         * @since 1.0.0
         */
        onUploadTemplate: function() {
            var oFileUploader = new sap.ui.unified.FileUploader({
                fileType: ["json"],
                maximumFileSize: 5,
                change: function(oEvent) {
                    var oFile = oEvent.getParameter("files")[0];
                    if (oFile) {
                        this._processTemplateUpload(oFile);
                    }
                }.bind(this)
            });
            
            oFileUploader.placeAt("content", "only");
            oFileUploader.$().find('input[type=file]').trigger('click');
        },

        /**
         * Process uploaded template file
         * @param {File} oFile - The uploaded file
         * @private
         * @since 1.0.0
         */
        _processTemplateUpload: function(oFile) {
            var oReader = new FileReader();
            oReader.onload = function(e) {
                try {
                    // Pre-validate content before parsing
                    var rawContent = e.target.result;
                    if (!this._securityUtils.validateRawJSON(rawContent)) {
                        throw new Error("Invalid or potentially malicious JSON content detected");
                    }
                    
                    var oTemplate = JSON.parse(rawContent);
                    // Validate and sanitize the schema
                    if (!this._securityUtils.validateSchema(oTemplate)) {
                        throw new Error("Invalid schema format");
                    }
                    oTemplate = this._securityUtils.sanitizeSchema(oTemplate);
                    this._validateTemplateSchema(oTemplate);
                    MessageToast.show(this._resourceBundle.getText("msg.templateUploadSuccess"));
                } catch (oError) {
                    MessageBox.error("Invalid template file: " + oError.message);
                }
            }.bind(this);
            oReader.readAsText(oFile);
        },

        /**
         * Validate template schema
         * @param {object} oTemplate - Template object
         * @private
         * @since 1.0.0
         */
        _validateTemplateSchema: function(oTemplate) {
            if (!oTemplate.schema || !oTemplate.name) {
                throw new Error("Template must contain 'schema' and 'name' properties");
            }
            // Additional validation logic can be added here
        }
    });
});