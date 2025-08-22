sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent4.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                // Initialize debounced validation for performance
                this._debouncedValidation = this._debounce(this._validateFormulaSyntax.bind(this), 300);
            },

            onExit: function() {
                // Cleanup resources to prevent memory leaks
                if (this._oCreateDialog) {
                    this._oCreateDialog.destroy();
                }
                if (this._oFormulaBuilderDialog) {
                    this._oFormulaBuilderDialog.destroy();
                }
                if (this._oAnalyticsDialog) {
                    this._oAnalyticsDialog.destroy();
                }
                if (this._oTestDialog) {
                    this._oTestDialog.destroy();
                }
                // Clear debounced functions
                if (this._debouncedValidation) {
                    this._debouncedValidation.cancel();
                }
            }
        },

        // Performance optimization utilities
        _debounce: function(fn, delay) {
            var timeoutId;
            var debounced = function() {
                var args = arguments;
                var context = this;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
                    fn.apply(context, args);
                }, delay);
            };
            
            debounced.cancel = function() {
                clearTimeout(timeoutId);
            };
            
            return debounced;
        },

        _throttle: function(fn, limit) {
            var inThrottle;
            return function() {
                var args = arguments;
                var context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(function() { inThrottle = false; }, limit);
                }
            };
        },

        // Security and validation utilities
        _validateInput: function(sInput, sType) {
            if (!sInput || typeof sInput !== 'string') {
                return { isValid: false, message: "Invalid input format" };
            }

            // Sanitize input
            var sSanitized = sInput.trim();
            
            // XSS prevention patterns
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
                case "formula":
                    // Formula expression validation
                    if (sSanitized.length > 10000) {
                        return { isValid: false, message: "Formula expression too long" };
                    }
                    // Check for dangerous formula patterns
                    var aDangerousPatterns = [/eval\s*\(/, /exec\s*\(/, /system\s*\(/, /import\s+/i];
                    for (var j = 0; j < aDangerousPatterns.length; j++) {
                        if (aDangerousPatterns[j].test(sSanitized)) {
                            return { isValid: false, message: "Potentially dangerous formula pattern" };
                        }
                    }
                    break;
                case "calculation":
                    // Numerical validation
                    if (!/^[0-9+\-*/().\s,E]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Invalid calculation format" };
                    }
                    break;
                case "taskName":
                    if (sSanitized.length > 200) {
                        return { isValid: false, message: "Task name too long" };
                    }
                    break;
            }

            return { isValid: true, sanitized: sSanitized };
        },

        _getCSRFToken: function() {
            return new Promise(function(resolve, reject) {
                jQuery.ajax({
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
                
                // Add authorization header if available
                var sAuthToken = that._getAuthToken();
                if (sAuthToken) {
                    oOptions.headers["Authorization"] = "Bearer " + sAuthToken;
                }

                return jQuery.ajax(oOptions);
            });
        },

        _getAuthToken: function() {
            // Get authentication token from secure storage or session
            return sessionStorage.getItem("a2a_auth_token") || "";
        },

        _logAuditEvent: function(sEventType, sDescription, sData) {
            // Log security and audit events for comprehensive audit trail
            var oAuditData = {
                timestamp: new Date().toISOString(),
                eventType: sEventType,
                description: sDescription,
                data: sData ? JSON.stringify(sData).substring(0, 500) : "",
                user: this._getCurrentUser(),
                component: "Agent4.ListReport",
                sessionId: this._getSessionId(),
                ipAddress: this._getClientIP(),
                userAgent: navigator.userAgent.substring(0, 200)
            };

            // Send to audit service with enhanced logging
            jQuery.ajax({
                url: "/a2a/common/v1/audit",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oAuditData),
                async: true,
                success: function() {
                    // Optionally log to browser console for debugging
                    console.log("Audit event logged:", sEventType);
                },
                error: function() {
                    // Fallback logging to local storage for offline audit
                    try {
                        var aLocalAudit = JSON.parse(localStorage.getItem("a2a_audit_log") || "[]");
                        aLocalAudit.push(oAuditData);
                        if (aLocalAudit.length > 100) aLocalAudit = aLocalAudit.slice(-100); // Keep last 100
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
            // Get current user from session or model
            return "current_user"; // Placeholder
        },

        _validateFormulaSecurity: function(sFormula) {
            // Enhanced formula security validation
            var oValidation = this._validateInput(sFormula, "formula");
            if (!oValidation.isValid) {
                return oValidation;
            }

            // Check for formula injection patterns
            var aInjectionPatterns = [
                /\$\{.*\}/,  // Expression injection
                /\#\{.*\}/,  // Hash injection
                /@\w+/,      // Annotation injection
                /\.\./       // Path traversal
            ];

            for (var i = 0; i < aInjectionPatterns.length; i++) {
                if (aInjectionPatterns[i].test(sFormula)) {
                    this._logAuditEvent("FORMULA_INJECTION", "Blocked formula injection attempt", sFormula);
                    return { isValid: false, message: "Invalid formula pattern detected" };
                }
            }

            return { isValid: true, sanitized: oValidation.sanitized };
        },

        _checkPermission: function(sAction) {
            // Role-based permission check
            var aUserRoles = this._getUserRoles();
            var mRequiredPermissions = {
                "CREATE_TASK": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "BATCH_VALIDATION": ["CALCULATION_ADMIN"],
                "FORMULA_BUILDER": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "VIEW_ANALYTICS": ["CALCULATION_ADMIN", "VALIDATION_VIEWER"]
            };

            var aRequiredRoles = mRequiredPermissions[sAction] || [];
            return aRequiredRoles.some(function(sRole) {
                return aUserRoles.includes(sRole);
            });
        },

        _getUserRoles: function() {
            // Get user roles from session or backend
            return ["VALIDATION_USER"]; // Placeholder
        },

        formatSecureText: function(sText) {
            // Secure text formatter for UI bindings
            if (!sText) return "";
            
            // Basic XSS protection
            return jQuery.sap.encodeXML(String(sText));
        },

        formatCalculationResult: function(nValue) {
            // Secure number formatter
            if (typeof nValue !== 'number' || !isFinite(nValue)) {
                return "Invalid";
            }
            
            // Prevent display of extremely large numbers that could cause issues
            if (Math.abs(nValue) > Number.MAX_SAFE_INTEGER) {
                return "Overflow";
            }
            
            return nValue.toFixed(6);
        },

        onCreateValidationTask: function() {
            // Permission check
            if (!this._checkPermission("CREATE_TASK")) {
                MessageBox.error("Insufficient permissions to create validation tasks");
                return;
            }

            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.CreateValidationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model with secure defaults
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        calculationType: "MATHEMATICAL",
                        dataSource: "",
                        priority: "MEDIUM",
                        validationMode: "STANDARD",
                        precisionThreshold: 0.001,
                        toleranceLevel: 0.01,
                        enableCrossValidation: true,
                        enableStatisticalTests: false,
                        formulas: []
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                    
                    this._logAuditEvent("DIALOG_OPENED", "Create validation task dialog opened");
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onFormulaBuilder: function() {
            // Permission check
            if (!this._checkPermission("FORMULA_BUILDER")) {
                MessageBox.error("Insufficient permissions to access formula builder");
                return;
            }

            var oView = this.base.getView();
            
            if (!this._oFormulaBuilderDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormulaBuilderDialog = oDialog;
                    oView.addDependent(this._oFormulaBuilderDialog);
                    
                    // Initialize formula builder model with secure functions
                    var oModel = new JSONModel({
                        currentFormula: "",
                        variables: [],
                        functions: [
                            "SUM", "AVERAGE", "MIN", "MAX", "COUNT",
                            "SQRT", "POW", "LOG", "EXP", "ABS",
                            "SIN", "COS", "TAN", "PI", "E"
                        ],
                        operators: ["+", "-", "*", "/", "^", "(", ")"],
                        testResults: [],
                        syntaxValid: false,
                        securityChecked: false
                    });
                    this._oFormulaBuilderDialog.setModel(oModel, "formula");
                    this._oFormulaBuilderDialog.open();
                }.bind(this));
            } else {
                this._oFormulaBuilderDialog.open();
            }
        },

        onBatchValidation: function() {
            // Permission check
            if (!this._checkPermission("BATCH_VALIDATION")) {
                MessageBox.error("Insufficient permissions for batch validation");
                return;
            }

            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch validation.");
                return;
            }

            // Validate selection limit for security
            if (aSelectedContexts.length > 100) {
                MessageBox.error("Maximum 100 tasks can be validated in batch for security reasons.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch validation for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchValidation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchValidation: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return this.formatSecureText(oContext.getProperty("ID"));
            }.bind(this));
            
            this.base.getView().setBusy(true);
            
            var oRequestData = {
                taskIds: aTaskIds,
                parallel: true,
                priority: "HIGH",
                validationSettings: {
                    precisionThreshold: 0.001,
                    enableCrossValidation: true,
                    maxExecutionTime: 300000 // 5 minutes max
                }
            };

            this._secureAjax({
                url: "/a2a/agent4/v1/batch-validate",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oRequestData)
            }).then(function(data) {
                this.base.getView().setBusy(false);
                MessageBox.success(
                    "Batch validation started!\n" +
                    "Job ID: " + this.formatSecureText(data.jobId) + "\n" +
                    "Estimated formulas: " + this.formatCalculationResult(data.totalFormulas)
                );
                this._extensionAPI.refresh();
                this._logAuditEvent("BATCH_VALIDATION", "Batch validation started", { taskCount: aTaskIds.length });
            }.bind(this)).catch(function(xhr) {
                this.base.getView().setBusy(false);
                MessageBox.error("Batch validation failed: " + this.formatSecureText(xhr.responseText));
                this._logAuditEvent("BATCH_VALIDATION_ERROR", "Batch validation failed", xhr.responseText);
            }.bind(this));
        },

        onCalculationTemplates: function() {
            // Navigate to calculation templates with audit logging
            this._logAuditEvent("NAVIGATION", "Navigated to calculation templates");
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("CalculationTemplates");
        },

        onValidationAnalytics: function() {
            // Permission check
            if (!this._checkPermission("VIEW_ANALYTICS")) {
                MessageBox.error("Insufficient permissions to view analytics");
                return;
            }

            var oView = this.base.getView();
            
            if (!this._oAnalyticsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationAnalytics",
                    controller: this
                }).then(function(oDialog) {
                    this._oAnalyticsDialog = oDialog;
                    oView.addDependent(this._oAnalyticsDialog);
                    this._oAnalyticsDialog.open();
                    
                    // Load analytics data securely
                    this._loadValidationAnalytics();
                }.bind(this));
            } else {
                this._oAnalyticsDialog.open();
                this._loadValidationAnalytics();
            }
        },

        _loadValidationAnalytics: function() {
            this._oAnalyticsDialog.setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/analytics",
                type: "GET"
            }).then(function(data) {
                this._oAnalyticsDialog.setBusy(false);
                
                // Sanitize analytics data
                var oSanitizedData = this._sanitizeAnalyticsData(data);
                var oModel = new JSONModel(oSanitizedData);
                this._oAnalyticsDialog.setModel(oModel, "analytics");
                
                // Create charts
                this._createAnalyticsCharts(oSanitizedData);
            }.bind(this)).catch(function(xhr) {
                this._oAnalyticsDialog.setBusy(false);
                MessageBox.error("Failed to load analytics: " + this.formatSecureText(xhr.responseText));
            }.bind(this));
        },

        _sanitizeAnalyticsData: function(oData) {
            // Sanitize analytics data for display
            if (!oData || typeof oData !== 'object') {
                return {};
            }

            var oSanitized = {};
            for (var sKey in oData) {
                if (oData.hasOwnProperty(sKey)) {
                    var vValue = oData[sKey];
                    if (typeof vValue === 'string') {
                        oSanitized[sKey] = this.formatSecureText(vValue);
                    } else if (typeof vValue === 'number' && isFinite(vValue)) {
                        oSanitized[sKey] = vValue;
                    } else if (Array.isArray(vValue)) {
                        oSanitized[sKey] = vValue.slice(0, 1000); // Limit array size
                    } else {
                        oSanitized[sKey] = vValue;
                    }
                }
            }
            return oSanitized;
        },

        _createAnalyticsCharts: function(data) {
            // Create secure analytics charts using SAP Viz framework
            // Implementation would use sanitized data
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Comprehensive validation
            var oTaskNameValidation = this._validateInput(oData.taskName, "taskName");
            if (!oTaskNameValidation.isValid) {
                MessageBox.error("Invalid task name: " + oTaskNameValidation.message);
                return;
            }

            var oDescValidation = this._validateInput(oData.description, "description");
            if (!oDescValidation.isValid) {
                MessageBox.error("Invalid description: " + oDescValidation.message);
                return;
            }
            
            if (!oData.calculationType) {
                MessageBox.error("Please select a calculation type");
                return;
            }
            
            if (oData.formulas.length === 0) {
                MessageBox.error("Please add at least one formula to validate");
                return;
            }

            // Validate all formulas
            for (var i = 0; i < oData.formulas.length; i++) {
                var oFormulaValidation = this._validateFormulaSecurity(oData.formulas[i].expression);
                if (!oFormulaValidation.isValid) {
                    MessageBox.error("Invalid formula at position " + (i + 1) + ": " + oFormulaValidation.message);
                    return;
                }
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Prepare sanitized data
            var oSanitizedData = {
                taskName: oTaskNameValidation.sanitized,
                description: oDescValidation.sanitized,
                calculationType: oData.calculationType,
                dataSource: this.formatSecureText(oData.dataSource),
                priority: oData.priority,
                validationMode: oData.validationMode,
                precisionThreshold: Math.max(0.0001, Math.min(1, oData.precisionThreshold)),
                toleranceLevel: Math.max(0.0001, Math.min(1, oData.toleranceLevel)),
                enableCrossValidation: Boolean(oData.enableCrossValidation),
                enableStatisticalTests: Boolean(oData.enableStatisticalTests),
                formulas: oData.formulas.map(function(formula) {
                    return {
                        expression: this._validateFormulaSecurity(formula.expression).sanitized,
                        description: this.formatSecureText(formula.description)
                    };
                }.bind(this))
            };
            
            this._secureAjax({
                url: "/a2a/agent4/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oSanitizedData)
            }).then(function(data) {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show("Validation task created successfully");
                this._extensionAPI.refresh();
                this._logAuditEvent("TASK_CREATED", "Validation task created", { taskName: oSanitizedData.taskName });
            }.bind(this)).catch(function(xhr) {
                this._oCreateDialog.setBusy(false);
                MessageBox.error("Failed to create task: " + this.formatSecureText(xhr.responseText));
                this._logAuditEvent("TASK_CREATE_ERROR", "Failed to create task", xhr.responseText);
            }.bind(this));
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },

        // Enhanced Formula Builder Methods with Security
        onAddFunction: function(oEvent) {
            var sFunction = this.formatSecureText(oEvent.getSource().getText());
            if (this._validateFormulaFunction(sFunction)) {
                this._insertIntoFormula(sFunction + "()");
            }
        },

        onAddOperator: function(oEvent) {
            var sOperator = this.formatSecureText(oEvent.getSource().getText());
            if (this._validateFormulaOperator(sOperator)) {
                this._insertIntoFormula(sOperator);
            }
        },

        _validateFormulaFunction: function(sFunction) {
            var aAllowedFunctions = [
                "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "SQRT", "POW", 
                "LOG", "EXP", "ABS", "SIN", "COS", "TAN", "PI", "E"
            ];
            return aAllowedFunctions.includes(sFunction);
        },

        _validateFormulaOperator: function(sOperator) {
            var aAllowedOperators = ["+", "-", "*", "/", "^", "(", ")"];
            return aAllowedOperators.includes(sOperator);
        },

        _insertIntoFormula: function(sText) {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sCurrentFormula = oModel.getProperty("/currentFormula");
            
            // Check formula length limit
            if ((sCurrentFormula + sText).length > 5000) {
                MessageBox.error("Formula too long. Maximum 5000 characters allowed.");
                return;
            }
            
            oModel.setProperty("/currentFormula", sCurrentFormula + sText);
            
            // Validate syntax and security with debounced performance optimization
            this._debouncedValidation();
        },

        _validateFormulaSyntax: function() {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sFormula = oModel.getProperty("/currentFormula");
            
            if (!sFormula) {
                oModel.setProperty("/syntaxValid", false);
                oModel.setProperty("/securityChecked", false);
                return;
            }

            // Security validation first
            var oSecurityValidation = this._validateFormulaSecurity(sFormula);
            oModel.setProperty("/securityChecked", oSecurityValidation.isValid);
            
            if (!oSecurityValidation.isValid) {
                oModel.setProperty("/syntaxValid", false);
                oModel.setProperty("/syntaxError", oSecurityValidation.message);
                return;
            }
            
            this._secureAjax({
                url: "/a2a/agent4/v1/validate-syntax",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ formula: oSecurityValidation.sanitized })
            }).then(function(data) {
                oModel.setProperty("/syntaxValid", data.valid);
                if (!data.valid) {
                    oModel.setProperty("/syntaxError", this.formatSecureText(data.error));
                }
            }.bind(this)).catch(function() {
                oModel.setProperty("/syntaxValid", false);
                oModel.setProperty("/syntaxError", "Validation service unavailable");
            });
        },

        onTestFormula: function() {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sFormula = oModel.getProperty("/currentFormula");
            
            if (!sFormula) {
                MessageBox.error("Please enter a formula to test");
                return;
            }

            if (!oModel.getProperty("/securityChecked")) {
                MessageBox.error("Formula security validation required before testing");
                return;
            }
            
            // Show test data input dialog
            this._showFormulaTestDialog(sFormula);
        },

        _showFormulaTestDialog: function(sFormula) {
            var oView = this.base.getView();
            
            if (!this._oTestDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaTest",
                    controller: this
                }).then(function(oDialog) {
                    this._oTestDialog = oDialog;
                    oView.addDependent(this._oTestDialog);
                    
                    var oModel = new JSONModel({
                        formula: sFormula,
                        testData: "{}",
                        expectedResult: "",
                        actualResult: "",
                        testPassed: false,
                        variance: 0
                    });
                    this._oTestDialog.setModel(oModel, "test");
                    this._oTestDialog.open();
                }.bind(this));
            } else {
                var oModel = this._oTestDialog.getModel("test");
                oModel.setProperty("/formula", sFormula);
                this._oTestDialog.open();
            }
        },

        onExecuteFormulaTest: function() {
            var oModel = this._oTestDialog.getModel("test");
            var oData = oModel.getData();
            
            // Validate test data
            var oTestDataValidation = this._validateInput(oData.testData, "calculation");
            if (!oTestDataValidation.isValid) {
                MessageBox.error("Invalid test data format");
                return;
            }

            try {
                var oParsedTestData = JSON.parse(oTestDataValidation.sanitized);
                var nExpectedResult = parseFloat(oData.expectedResult);
                
                if (!isFinite(nExpectedResult)) {
                    MessageBox.error("Invalid expected result format");
                    return;
                }
            } catch (e) {
                MessageBox.error("Invalid test data JSON format");
                return;
            }
            
            this._oTestDialog.setBusy(true);
            
            this._secureAjax({
                url: "/a2a/agent4/v1/test-formula",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    formula: this._validateFormulaSecurity(oData.formula).sanitized,
                    testData: oParsedTestData,
                    expectedResult: nExpectedResult
                })
            }).then(function(data) {
                this._oTestDialog.setBusy(false);
                
                oModel.setProperty("/actualResult", this.formatCalculationResult(data.result));
                oModel.setProperty("/testPassed", Boolean(data.passed));
                oModel.setProperty("/variance", this.formatCalculationResult(data.variance));
                
                if (data.passed) {
                    MessageToast.show("Formula test passed!");
                } else {
                    MessageBox.warning("Formula test failed. Check variance: " + this.formatCalculationResult(data.variance));
                }
                
                this._logAuditEvent("FORMULA_TEST", "Formula test executed", { 
                    passed: data.passed, 
                    variance: data.variance 
                });
            }.bind(this)).catch(function(xhr) {
                this._oTestDialog.setBusy(false);
                MessageBox.error("Formula test failed: " + this.formatSecureText(xhr.responseText));
            }.bind(this));
        }
    });
});