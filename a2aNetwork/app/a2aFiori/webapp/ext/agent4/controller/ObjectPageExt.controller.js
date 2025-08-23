sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "sap/base/Log",
    "../utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp, sanitizeHTML, Log, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent4.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize dialog cache for better performance
                this._dialogCache = {};
                
                // Initialize create model
                this._initializeCreateModel();
                
                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
            },

            onExit: function() {
                // Cleanup resources and intervals
                if (this._validationInterval) {
                    clearInterval(this._validationInterval);
                }
                // Clean up cached dialogs
                for (var sKey in this._dialogCache) {
                    if (this._dialogCache.hasOwnProperty(sKey)) {
                        this._dialogCache[sKey].destroy();
                    }
                }
                // Clean up any other dialogs
                if (this._oFormulaValidationDialog) {
                    this._oFormulaValidationDialog.destroy();
                }
                if (this._oReportDialog) {
                    this._oReportDialog.destroy();
                }
                if (this._oOptimizationDialog) {
                    this._oOptimizationDialog.destroy();
                }
            }
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
                calculationType: "",
                dataSource: "",
                priority: "MEDIUM",
                validationMode: "STANDARD",
                precisionThreshold: 0.00001,
                toleranceLevel: 0.001,
                comparisonMethod: 0,
                enableCrossValidation: false,
                enableStatisticalTests: false,
                parallelProcessing: true,
                detailedLogs: false,
                formulas: [],
                customValidationRules: "",
                benchmarkDataset: "",
                maxExecutionTime: 300,
                memoryLimit: 1024,
                errorHandling: 1,
                emailNotification: false,
                progressUpdates: true,
                isValid: false,
                taskNameState: "None",
                taskNameStateText: "",
                calculationTypeState: "None",
                calculationTypeStateText: "",
                dataSourceState: "None",
                dataSourceStateText: "",
                templates: [
                    { name: "Simple Arithmetic", formula: "a + b * c", category: "ARITHMETIC" },
                    { name: "Compound Interest", formula: "P * (1 + r/n)^(n*t)", category: "FINANCIAL" },
                    { name: "Standard Deviation", formula: "sqrt(sum((x - mean)^2) / n)", category: "STATISTICAL" },
                    { name: "Pythagorean Theorem", formula: "sqrt(a^2 + b^2)", category: "MATHEMATICAL" },
                    { name: "Sine Wave", formula: "A * sin(2 * PI * f * t + phase)", category: "TRIGONOMETRIC" }
                ]
            });
            this.base.getView().setModel(oCreateModel, "create");
        },

        /**
         * Validation handler for task name changes
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @public
         * @since 1.0.0
         */
        onTaskNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oModel = this.base.getView().getModel("create");
            
            if (!sValue || sValue.trim().length === 0) {
                oModel.setProperty("/taskNameState", "Error");
                oModel.setProperty("/taskNameStateText", this._oResourceBundle.getText("validation.taskNameRequired"));
            } else if (sValue.length < 3) {
                oModel.setProperty("/taskNameState", "Error");
                oModel.setProperty("/taskNameStateText", this._oResourceBundle.getText("validation.taskNameTooShort"));
            } else if (sValue.length > 100) {
                oModel.setProperty("/taskNameState", "Error");
                oModel.setProperty("/taskNameStateText", this._oResourceBundle.getText("validation.taskNameTooLong"));
            } else if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sValue)) {
                oModel.setProperty("/taskNameState", "Error");
                oModel.setProperty("/taskNameStateText", this._oResourceBundle.getText("validation.taskNameInvalid"));
            } else {
                oModel.setProperty("/taskNameState", "Success");
                oModel.setProperty("/taskNameStateText", "");
            }
            
            this._validateForm();
        },

        /**
         * Handler for calculation type changes
         * @param {sap.ui.base.Event} oEvent - Select change event
         * @public
         * @since 1.0.0
         */
        onCalculationTypeChange: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oModel = this.base.getView().getModel("create");
            
            if (!sSelectedKey) {
                oModel.setProperty("/calculationTypeState", "Error");
                oModel.setProperty("/calculationTypeStateText", this._oResourceBundle.getText("validation.calculationTypeRequired"));
            } else {
                oModel.setProperty("/calculationTypeState", "Success");
                oModel.setProperty("/calculationTypeStateText", "");
                
                // Auto-suggest validation mode based on calculation type
                switch(sSelectedKey) {
                    case "FINANCIAL":
                    case "ACCOUNTING":
                        oModel.setProperty("/validationMode", "STRICT");
                        oModel.setProperty("/precisionThreshold", 0.00001);
                        break;
                    case "ENGINEERING":
                    case "PHYSICS":
                        oModel.setProperty("/validationMode", "STANDARD");
                        oModel.setProperty("/precisionThreshold", 0.0001);
                        break;
                    case "STATISTICAL":
                        oModel.setProperty("/enableStatisticalTests", true);
                        break;
                }
            }
            
            this._validateForm();
        },

        /**
         * Handler for data source changes
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @public
         * @since 1.0.0
         */
        onDataSourceChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oModel = this.base.getView().getModel("create");
            
            if (sValue && sValue.trim().length > 0) {
                // Basic path validation
                if (/[<>:"|?*]/.test(sValue)) {
                    oModel.setProperty("/dataSourceState", "Error");
                    oModel.setProperty("/dataSourceStateText", this._oResourceBundle.getText("validation.dataSourceInvalid"));
                } else if (/\.\./.test(sValue)) {
                    oModel.setProperty("/dataSourceState", "Error");
                    oModel.setProperty("/dataSourceStateText", "Path traversal not allowed");
                } else {
                    oModel.setProperty("/dataSourceState", "Success");
                    oModel.setProperty("/dataSourceStateText", "");
                }
            } else {
                oModel.setProperty("/dataSourceState", "None");
                oModel.setProperty("/dataSourceStateText", "");
            }
            
            this._validateForm();
        },

        /**
         * Handler for validation mode changes
         * @param {sap.ui.base.Event} oEvent - Select change event
         * @public
         * @since 1.0.0
         */
        onValidationModeChange: function(oEvent) {
            var sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            var oModel = this.base.getView().getModel("create");
            
            // Adjust settings based on validation mode
            switch(sSelectedKey) {
                case "STRICT":
                    oModel.setProperty("/precisionThreshold", 0.000001);
                    oModel.setProperty("/toleranceLevel", 0.0001);
                    oModel.setProperty("/enableCrossValidation", true);
                    break;
                case "BENCHMARK":
                    oModel.setProperty("/enableStatisticalTests", true);
                    oModel.setProperty("/detailedLogs", true);
                    break;
                case "CUSTOM":
                    // Show custom rules text area
                    MessageToast.show(this.getResourceBundle().getText("message.configureCustomRules"));
                    break;
            }
        },

        /**
         * Handler for adding new formulas
         * @public
         * @since 1.0.0
         */
        onAddFormula: function() {
            var oModel = this.base.getView().getModel("create");
            var aFormulas = oModel.getProperty("/formulas");
            
            aFormulas.push({
                expression: "",
                expectedResult: 0,
                type: "ARITHMETIC",
                testData: "",
                id: Date.now()
            });
            
            oModel.setProperty("/formulas", aFormulas);
            this._validateForm();
        },

        /**
         * Handler for formula expression changes
         * @param {sap.ui.base.Event} oEvent - TextArea change event
         * @public
         * @since 1.0.0
         */
        onFormulaExpressionChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oSource = oEvent.getSource();
            var oContext = oSource.getBindingContext("create");
            
            if (oContext) {
                var sPath = oContext.getPath();
                var oModel = this.base.getView().getModel("create");
                
                // Basic formula validation
                if (sValue && sValue.trim().length > 0) {
                    // Check for common formula syntax
                    if (/[a-zA-Z0-9\+\-\*\/\^\(\)\s\.]+/.test(sValue)) {
                        oModel.setProperty(sPath + "/isValid", true);
                    } else {
                        oModel.setProperty(sPath + "/isValid", false);
                    }
                }
            }
            
            this._validateForm();
        },

        /**
         * Validate the entire form
         * @private
         * @since 1.0.0
         */
        _validateForm: function() {
            var oModel = this.base.getView().getModel("create");
            var oData = oModel.getData();
            
            var bValid = oData.taskNameState === "Success" &&
                        oData.calculationType &&
                        oData.formulas.length > 0 &&
                        oData.formulas.some(function(formula) {
                            return formula.expression && formula.expression.trim().length > 0;
                        });
            
            oModel.setProperty("/isValid", bValid);
        },

        // Security and validation utilities  
        _validateInput: function(sInput, sType) {
            if (!sInput || typeof sInput !== 'string') {
                return { isValid: false, message: "Invalid input format" };
            }

            var sSanitized = sInput.trim();
            
            // XSS prevention
            var aXSSPatterns = [
                /<script/i, /javascript:/i, /on\w+\s*=/i, /<iframe/i, 
                /<object/i, /<embed/i, /eval\s*\(/i, /Function\s*\(/i
            ];

            for (var i = 0; i < aXSSPatterns.length; i++) {
                if (aXSSPatterns[i].test(sSanitized)) {
                    this._logAuditEvent("XSS_ATTEMPT", "Blocked XSS attempt in " + sType, sInput);
                    return { isValid: false, message: "Invalid characters detected" };
                }
            }

            // Type-specific validation
            switch (sType) {
                case "taskId":
                    if (!/^[a-zA-Z0-9_-]+$/.test(sSanitized) || sSanitized.length > 50) {
                        return { isValid: false, message: "Invalid task ID format" };
                    }
                    break;
                case "formula":
                    if (sSanitized.length > 10000) {
                        return { isValid: false, message: "Formula too long" };
                    }
                    break;
                case "reportType":
                    var aValidTypes = ["SUMMARY", "DETAILED", "PERFORMANCE", "ERRORS"];
                    if (!aValidTypes.includes(sSanitized)) {
                        return { isValid: false, message: "Invalid report type" };
                    }
                    break;
            }

            return { isValid: true, sanitized: sSanitized };
        },

        _getCSRFToken: function() {
            return new Promise(function(resolve, reject) {
                this._securityUtils.secureAjaxRequest({
                    url: "/a2a/agent4/v1/csrf-token",
                    type: "GET",
                    success: function(data) {
                        resolve(data.token);
                    },
                    error: function() {
                        reject("Failed to retrieve CSRF token");
                    }
                });
            });
        },

        _secureAjax: function(oOptions) {
            var that = this;
            return this._getCSRFToken().then(function(sToken) {
                oOptions.headers = oOptions.headers || {};
                oOptions.headers["X-CSRF-Token"] = sToken;
                
                var sAuthToken = that._getAuthToken();
                if (sAuthToken) {
                    oOptions.headers["Authorization"] = "Bearer " + sAuthToken;
                }

                return this._securityUtils.secureAjaxRequest(oOptions);
            });
        },

        _getAuthToken: function() {
            return sessionStorage.getItem("a2a_auth_token") || "";
        },

        _logAuditEvent: function(sEventType, sDescription, sData) {
            // Comprehensive audit trail logging
            var oAuditData = {
                timestamp: new Date().toISOString(),
                eventType: sEventType,
                description: sDescription,
                data: sData ? JSON.stringify(sData).substring(0, 500) : "",
                user: this._getCurrentUser(),
                component: "Agent4.ObjectPage",
                sessionId: this._getSessionId(),
                ipAddress: this._getClientIP(),
                userAgent: navigator.userAgent.substring(0, 200),
                contextPath: window.location.pathname
            };

            // Enhanced audit service logging
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/common/v1/audit",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oAuditData),
                async: true,
                success: function() {
                    // Audit success logging
                    console.log("Audit trail recorded:", sEventType);
                },
                error: function() {
                    // Fallback audit logging to local storage
                    try {
                        var aLocalAudit = JSON.parse(localStorage.getItem("a2a_audit_log") || "[]");
                        aLocalAudit.push(oAuditData);
                        if (aLocalAudit.length > 100) aLocalAudit = aLocalAudit.slice(-100);
                        localStorage.setItem("a2a_audit_log", JSON.stringify(aLocalAudit));
                    } catch(e) {
                        // Silent fail for localStorage issues
                    }
                }
            });
        },

        _getSessionId: function() {
            return sessionStorage.getItem("a2a_session_id") || "unknown";
        },

        _getClientIP: function() {
            // In real implementation, this would come from server-side
            return "client_ip_masked";
        },

        _getCurrentUser: function() {
            return "current_user"; // Placeholder
        },

        _checkPermission: function(sAction) {
            var aUserRoles = this._getUserRoles();
            var mRequiredPermissions = {
                "START_VALIDATION": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "VALIDATE_FORMULAS": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "RUN_BENCHMARK": ["CALCULATION_ADMIN"],
                "GENERATE_REPORT": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "OPTIMIZE_CALCULATIONS": ["CALCULATION_ADMIN"]
            };

            var aRequiredRoles = mRequiredPermissions[sAction] || [];
            return aRequiredRoles.some(function(sRole) {
                return aUserRoles.includes(sRole);
            });
        },

        _getUserRoles: function() {
            return ["VALIDATION_USER"]; // Placeholder
        },

        formatSecureText: function(sText) {
            if (!sText) return "";
            return jQuery.sap.encodeXML(String(sText));
        },

        formatCalculationResult: function(nValue) {
            if (typeof nValue !== 'number' || !isFinite(nValue)) {
                return "Invalid";
            }
            if (Math.abs(nValue) > Number.MAX_SAFE_INTEGER) {
                return "Overflow";
            }
            return nValue.toFixed(6);
        },

        formatPercentage: function(nValue) {
            if (typeof nValue !== 'number' || !isFinite(nValue)) {
                return "0%";
            }
            return Math.round(Math.max(0, Math.min(100, nValue))) + "%";
        },

        _validateCalculationResult: function(oResult) {
            // Validate calculation results for security
            if (!oResult || typeof oResult !== 'object') {
                return { isValid: false, message: "Invalid result object" };
            }

            // Check for suspicious values
            if (oResult.hasOwnProperty('value')) {
                var nValue = parseFloat(oResult.value);
                if (!isFinite(nValue)) {
                    return { isValid: false, message: "Invalid calculation result" };
                }
                
                // Check for potential overflow attacks
                if (Math.abs(nValue) > Number.MAX_SAFE_INTEGER) {
                    this._logAuditEvent("OVERFLOW_ATTEMPT", "Potential overflow attack detected", nValue);
                    return { isValid: false, message: "Result value too large" };
                }
            }

            return { isValid: true, sanitized: oResult };
        },

        onStartValidation: function() {
            if (!this._checkPermission("START_VALIDATION")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.validation"));
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            // Validate task ID
            var oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskName", [oTaskIdValidation.message]));
                return;
            }
            
            MessageBox.confirm(this.getResourceBundle().getText("confirm.startValidation", [this.formatSecureText(sTaskName)]), {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startValidationProcess(oTaskIdValidation.sanitized);
                    }
                }.bind(this)
            });
        },

        _startValidationProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(sTaskId) + "/validate",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    priority: "NORMAL",
                    securityMode: "STRICT",
                    maxExecutionTime: 600000, // 10 minutes max
                    enableSafeguards: true
                })
            }).then(function(data) {
                this._extensionAPI.getView().setBusy(false);
                MessageToast.show(this.getResourceBundle().getText("message.validationStarted"));
                this._extensionAPI.refresh();
                
                // Start secure progress monitoring
                this._startValidationMonitoring(sTaskId);
                this._logAuditEvent("VALIDATION_STARTED", "Validation process started", { taskId: sTaskId });
            }.bind(this)).catch(function(xhr) {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.validationStartFailed", [this.formatSecureText(xhr.responseText)]));
                this._logAuditEvent("VALIDATION_START_ERROR", "Failed to start validation", xhr.responseText);
            }.bind(this));
        },

        _startValidationMonitoring: function(sTaskId) {
            // Secure polling with rate limiting
            var nPollCount = 0;
            var nMaxPolls = 300; // Max 10 minutes of polling (2s intervals)
            
            this._validationInterval = setInterval(function() {
                nPollCount++;
                
                if (nPollCount > nMaxPolls) {
                    clearInterval(this._validationInterval);
                    MessageBox.warning(this.getResourceBundle().getText("warning.validationTimeout"));
                    return;
                }
                
                this._secureAjax({
                    url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(sTaskId) + "/progress",
                    type: "GET"
                }).then(function(data) {
                    // Validate progress data
                    var oValidatedData = this._validateProgressData(data);
                    if (!oValidatedData.isValid) {
                        clearInterval(this._validationInterval);
                        return;
                    }
                    
                    if (data.status === "COMPLETED" || data.status === "FAILED") {
                        clearInterval(this._validationInterval);
                        this._extensionAPI.refresh();
                        
                        if (data.status === "COMPLETED") {
                            MessageBox.success(
                                this.getResourceBundle().getText("success.validationComplete", [
                                    this.formatCalculationResult(data.formulasValidated),
                                    this.formatPercentage(data.accuracy),
                                    this.formatCalculationResult(data.errorCount)
                                ])
                            );
                            this._logAuditEvent("VALIDATION_COMPLETED", "Validation completed successfully", data);
                        } else {
                            MessageBox.error(this.getResourceBundle().getText("message.validationFailed", [this.formatSecureText(data.error)]));
                            this._logAuditEvent("VALIDATION_FAILED", "Validation failed", data.error);
                        }
                    } else {
                        // Update progress with throttling
                        if (nPollCount % 5 === 0) { // Only show progress every 10 seconds
                            MessageToast.show("Validating: " + this.formatPercentage(data.progress));
                        }
                    }
                }.bind(this)).catch(function() {
                    // Fail silently on progress errors to avoid spam
                    if (nPollCount > 10) { // Only stop after some attempts
                        clearInterval(this._validationInterval);
                    }
                }.bind(this));
            }.bind(this), 2000);
        },

        _validateProgressData: function(oData) {
            if (!oData || typeof oData !== 'object') {
                return { isValid: false };
            }
            
            // Validate status
            var aValidStatuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "PAUSED"];
            if (!aValidStatuses.includes(oData.status)) {
                return { isValid: false };
            }
            
            // Validate numeric fields
            if (oData.hasOwnProperty('progress')) {
                var nProgress = parseFloat(oData.progress);
                if (!isFinite(nProgress) || nProgress < 0 || nProgress > 100) {
                    return { isValid: false };
                }
            }
            
            return { isValid: true };
        },

        onValidateFormulas: function() {
            if (!this._checkPermission("VALIDATE_FORMULAS")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.formulaValidation"));
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            var oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskId"));
                return;
            }
            
            if (!this._oFormulaValidationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormulaValidationDialog = oDialog;
                    this.base.getView().addDependent(this._oFormulaValidationDialog);
                    
                    // Load formulas for this task
                    this._loadTaskFormulas(oTaskIdValidation.sanitized);
                    this._oFormulaValidationDialog.open();
                }.bind(this));
            } else {
                this._loadTaskFormulas(oTaskIdValidation.sanitized);
                this._oFormulaValidationDialog.open();
            }
        },

        _loadTaskFormulas: function(sTaskId) {
            this._oFormulaValidationDialog.setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(sTaskId) + "/formulas",
                type: "GET"
            }).then(function(data) {
                this._oFormulaValidationDialog.setBusy(false);
                
                // Sanitize formula data
                var oSanitizedData = this._sanitizeFormulaData(data);
                var oModel = new JSONModel({
                    taskId: sTaskId,
                    formulas: oSanitizedData.formulas || [],
                    validationSettings: {
                        precisionThreshold: 0.001,
                        toleranceLevel: 0.01,
                        testCases: 10,
                        enableSafeguards: true
                    }
                });
                this._oFormulaValidationDialog.setModel(oModel, "validation");
            }.bind(this)).catch(function(xhr) {
                this._oFormulaValidationDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.formulaLoadFailed", [this.formatSecureText(xhr.responseText)]));
            }.bind(this));
        },

        _sanitizeFormulaData: function(oData) {
            if (!oData || !Array.isArray(oData.formulas)) {
                return { formulas: [] };
            }
            
            var aSanitizedFormulas = oData.formulas.map(function(formula) {
                return {
                    id: this.formatSecureText(formula.id),
                    expression: this.formatSecureText(formula.expression),
                    description: this.formatSecureText(formula.description),
                    type: this.formatSecureText(formula.type),
                    selected: Boolean(formula.selected),
                    validationResult: formula.validationResult ? this.formatSecureText(formula.validationResult) : "",
                    actualValue: typeof formula.actualValue === 'number' ? this.formatCalculationResult(formula.actualValue) : "",
                    expectedValue: typeof formula.expectedValue === 'number' ? this.formatCalculationResult(formula.expectedValue) : "",
                    variance: typeof formula.variance === 'number' ? this.formatCalculationResult(formula.variance) : ""
                };
            }.bind(this));
            
            return { formulas: aSanitizedFormulas };
        },

        onValidateSelectedFormulas: function() {
            var oModel = this._oFormulaValidationDialog.getModel("validation");
            var aFormulas = oModel.getProperty("/formulas");
            var aSelectedFormulas = aFormulas.filter(function(formula) {
                return formula.selected;
            });
            
            if (aSelectedFormulas.length === 0) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.selectFormulas"));
                return;
            }

            // Limit batch size for security
            if (aSelectedFormulas.length > 50) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.maxFormulas"));
                return;
            }
            
            this._oFormulaValidationDialog.setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/validate-formulas",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    formulas: aSelectedFormulas.map(function(formula) {
                        return {
                            id: formula.id,
                            expression: formula.expression
                        };
                    }),
                    settings: oModel.getProperty("/validationSettings")
                })
            }).then(function(data) {
                this._oFormulaValidationDialog.setBusy(false);
                
                // Validate and update results
                if (data.results && Array.isArray(data.results)) {
                    this._updateFormulaResults(data.results);
                    
                    MessageBox.success(
                        this.getResourceBundle().getText("success.formulaValidationCompleted", [
                            this.formatCalculationResult(data.validated),
                            aSelectedFormulas.length,
                            this.formatCalculationResult(data.passed),
                            this.formatCalculationResult(data.failed)
                        ])
                    );
                    
                    this._logAuditEvent("FORMULAS_VALIDATED", "Formulas validated", { 
                        count: aSelectedFormulas.length,
                        passed: data.passed,
                        failed: data.failed 
                    });
                }
            }.bind(this)).catch(function(xhr) {
                this._oFormulaValidationDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.formulaValidationFailed", [this.formatSecureText(xhr.responseText)]));
            }.bind(this));
        },

        _updateFormulaResults: function(aResults) {
            var oModel = this._oFormulaValidationDialog.getModel("validation");
            var aFormulas = oModel.getProperty("/formulas");
            
            aResults.forEach(function(result) {
                var oFormula = aFormulas.find(function(f) {
                    return f.id === result.formulaId;
                });
                if (oFormula) {
                    // Validate result data before updating
                    var oValidatedResult = this._validateCalculationResult(result);
                    if (oValidatedResult.isValid) {
                        oFormula.validationResult = this.formatSecureText(result.result);
                        oFormula.actualValue = this.formatCalculationResult(result.actualValue);
                        oFormula.variance = this.formatCalculationResult(result.variance);
                        oFormula.errors = result.errors ? this.formatSecureText(JSON.stringify(result.errors)) : "";
                    }
                }
            }.bind(this));
            
            oModel.setProperty("/formulas", aFormulas);
        },

        onRunBenchmark: function() {
            if (!this._checkPermission("RUN_BENCHMARK")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.benchmarks"));
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            var oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskId"));
                return;
            }
            
            MessageBox.confirm(
                this.getResourceBundle().getText("confirm.runBenchmarkSimple"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runPerformanceBenchmark(oTaskIdValidation.sanitized);
                        }
                    }.bind(this)
                }
            );
        },

        _runPerformanceBenchmark: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(sTaskId) + "/benchmark",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    iterations: Math.min(1000, 100), // Limit iterations for security
                    measureMemory: true,
                    measureCPU: true,
                    compareWith: "reference",
                    maxExecutionTime: 300000, // 5 minutes max
                    enableSafeguards: true
                })
            }).then(function(data) {
                this._extensionAPI.getView().setBusy(false);
                this._showBenchmarkResults(data);
                this._logAuditEvent("BENCHMARK_COMPLETED", "Performance benchmark completed", { taskId: sTaskId });
            }.bind(this)).catch(function(xhr) {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.benchmarkFailed", [this.formatSecureText(xhr.responseText)]));
                this._logAuditEvent("BENCHMARK_ERROR", "Benchmark failed", xhr.responseText);
            }.bind(this));
        },

        _showBenchmarkResults: function(data) {
            // Sanitize benchmark data
            var sMessage = "Benchmark Results:\n\n" +
                          "Average execution time: " + this.formatCalculationResult(data.avgExecutionTime) + " ms\n" +
                          "Memory usage: " + this.formatCalculationResult(data.memoryUsage) + " MB\n" +
                          "CPU efficiency: " + this.formatPercentage(data.cpuEfficiency) + "\n" +
                          "Formulas per second: " + this.formatCalculationResult(data.formulasPerSecond);
            
            MessageBox.information(sMessage, {
                title: "Performance Benchmark"
            });
        },

        onGenerateReport: function() {
            if (!this._checkPermission("GENERATE_REPORT")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.reports"));
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            var oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskId"));
                return;
            }
            
            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationMethodsSelector",
                    controller: this
                }).then(function(oDialog) {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: oTaskIdValidation.sanitized,
                        taskName: this.formatSecureText(sTaskName),
                        reportType: "DETAILED",
                        includeFormulas: true,
                        includeErrors: true,
                        includePerformance: true,
                        format: "PDF",
                        maxFileSize: 10 // MB limit for security
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                }.bind(this));
            } else {
                this._oReportDialog.open();
            }
        },

        onGenerateValidationReport: function() {
            var oModel = this._oReportDialog.getModel("report");
            var oData = oModel.getData();
            
            // Validate report type
            var oReportTypeValidation = this._validateInput(oData.reportType, "reportType");
            if (!oReportTypeValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidReportType"));
                return;
            }
            
            this._oReportDialog.setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(oData.taskId) + "/report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    type: oReportTypeValidation.sanitized,
                    includeFormulas: Boolean(oData.includeFormulas),
                    includeErrors: Boolean(oData.includeErrors),
                    includePerformance: Boolean(oData.includePerformance),
                    format: oData.format === "PDF" ? "PDF" : "HTML", // Whitelist formats
                    maxFileSize: Math.min(oData.maxFileSize || 10, 50) // Limit file size
                })
            }).then(function(data) {
                this._oReportDialog.setBusy(false);
                this._oReportDialog.close();
                
                // Validate download URL
                if (data.downloadUrl && typeof data.downloadUrl === 'string') {
                    MessageBox.success(
                        this.getResourceBundle().getText("success.reportGeneratedSuccess"),
                        {
                            actions: ["Download", MessageBox.Action.CLOSE],
                            onClose: function(oAction) {
                                if (oAction === "Download") {
                                    // Secure download with validation
                                    this._secureDownload(data.downloadUrl);
                                }
                            }.bind(this)
                        }
                    );
                    this._logAuditEvent("REPORT_GENERATED", "Validation report generated", { 
                        taskId: oData.taskId,
                        type: oData.reportType 
                    });
                }
            }.bind(this)).catch(function(xhr) {
                this._oReportDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.reportGenerationFailed", [this.formatSecureText(xhr.responseText)]));
            }.bind(this));
        },

        _secureDownload: function(sUrl) {
            // Validate URL format and origin
            try {
                var oUrl = new URL(sUrl, window.location.origin);
                if (oUrl.origin === window.location.origin && oUrl.pathname.startsWith('/a2a/agent4/')) {
                    window.open(sUrl, "_blank");
                } else {
                    MessageBox.error(this.getResourceBundle().getText("error.validation.invalidDownloadUrl"));
                }
            } catch (e) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidDownloadUrlFormat"));
            }
        },

        onOptimizeCalculations: function() {
            if (!this._checkPermission("OPTIMIZE_CALCULATIONS")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.optimization"));
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            var oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskId"));
                return;
            }
            
            MessageBox.confirm(
                this.getResourceBundle().getText("confirm.optimizeCalculationsSimple"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeCalculations(oTaskIdValidation.sanitized);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeCalculations: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks/" + encodeURIComponent(sTaskId) + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    optimizeFor: "SPEED",
                    preserveAccuracy: true,
                    maxIterations: 100,
                    enableSafeguards: true,
                    maxExecutionTime: 180000 // 3 minutes max
                })
            }).then(function(data) {
                this._extensionAPI.getView().setBusy(false);
                
                if (data.optimizations && data.optimizations.length > 0) {
                    this._showOptimizationSuggestions(data);
                } else {
                    MessageBox.information(this.getResourceBundle().getText("success.optimizationCompleted"));
                }
                this._logAuditEvent("OPTIMIZATION_COMPLETED", "Calculation optimization completed", { taskId: sTaskId });
            }.bind(this)).catch(function(xhr) {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.optimizationFailed", [this.formatSecureText(xhr.responseText)]));
            }.bind(this));
        },

        _showOptimizationSuggestions: function(data) {
            var oView = this.base.getView();
            
            if (!this._oOptimizationDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.ExpressionBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._oOptimizationDialog = oDialog;
                    oView.addDependent(this._oOptimizationDialog);
                    
                    // Sanitize optimization data
                    var oSanitizedData = this._sanitizeOptimizationData(data);
                    var oModel = new JSONModel(oSanitizedData);
                    this._oOptimizationDialog.setModel(oModel, "optimization");
                    this._oOptimizationDialog.open();
                }.bind(this));
            } else {
                var oSanitizedData = this._sanitizeOptimizationData(data);
                var oModel = new JSONModel(oSanitizedData);
                this._oOptimizationDialog.setModel(oModel, "optimization");
                this._oOptimizationDialog.open();
            }
        },

        _sanitizeOptimizationData: function(oData) {
            if (!oData || !Array.isArray(oData.optimizations)) {
                return { optimizations: [] };
            }
            
            var aSanitizedOptimizations = oData.optimizations.slice(0, 20).map(function(opt) {
                return {
                    id: this.formatSecureText(opt.id),
                    description: this.formatSecureText(opt.description),
                    expectedImprovement: this.formatPercentage(opt.expectedImprovement),
                    complexity: this.formatSecureText(opt.complexity),
                    type: this.formatSecureText(opt.type),
                    safe: Boolean(opt.safe)
                };
            }.bind(this));
            
            return { 
                optimizations: aSanitizedOptimizations,
                summary: {
                    totalSuggestions: aSanitizedOptimizations.length,
                    potentialImprovement: this.formatPercentage(oData.summary?.potentialImprovement || 0)
                }
            };
        },

        onApplyOptimization: function(oEvent) {
            var oSource = oEvent.getSource();
            var oBindingContext = oSource.getBindingContext("optimization");
            var oOptimization = oBindingContext.getObject();
            
            // Security check - only apply safe optimizations
            if (!oOptimization.safe) {
                MessageBox.error(this.getResourceBundle().getText("warning.optimizationUnsafe"));
                return;
            }
            
            MessageBox.confirm(
                this.getResourceBundle().getText("confirm.applyOptimizationWithDetails", [
                    oOptimization.description,
                    oOptimization.expectedImprovement
                ]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._applyOptimization(oOptimization);
                        }
                    }.bind(this)
                }
            );
        },

        _applyOptimization: function(oOptimization) {
            this._secureAjax({
                url: "/a2a/agent4/v1/apply-optimization",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    id: oOptimization.id,
                    type: oOptimization.type,
                    confirmSafe: true
                })
            }).then(function(data) {
                MessageToast.show(this.getResourceBundle().getText("success.optimizationApplied"));
                this._extensionAPI.refresh();
                this._logAuditEvent("OPTIMIZATION_APPLIED", "Optimization applied", { 
                    optimizationId: oOptimization.id,
                    type: oOptimization.type 
                });
            }.bind(this)).catch(function(xhr) {
                MessageBox.error(this.getResourceBundle().getText("error.operation.optimizationFailed", [this.formatSecureText(xhr.responseText)]));
            }.bind(this));
        }
    });
});