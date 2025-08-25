sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "sap/base/Log",
    "../utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel,
    encodeXML, escapeRegExp, sanitizeHTML, Log, SecurityUtils) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent4.ext.controller.ListReportExt", {

        override: {
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._resourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                // Initialize dialog cache and security settings
                this._dialogCache = {};
                this._csrfToken = null;
                this._initializeCSRFToken();

                // Initialize debounced validation for performance
                this._debouncedValidation = this._debounce(this._validateFormulaSyntax.bind(this), 300);

                // Initialize device model for responsive behavior
                const oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");

                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
            },

            onExit() {
                // Cleanup resources to prevent memory leaks
                for (const sKey in this._dialogCache) {
                    if (this._dialogCache.hasOwnProperty(sKey)) {
                        this._dialogCache[sKey].destroy();
                    }
                }
                // Cleanup legacy dialogs
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
        _debounce(fn, delay) {
            let timeoutId;
            const debounced = function() {
                const args = arguments;
                const context = this;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    fn.apply(context, args);
                }, delay);
            };

            debounced.cancel = function() {
                clearTimeout(timeoutId);
            };

            return debounced;
        },

        _throttle(fn, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => { inThrottle = false; }, limit);
                }
            };
        },

        // Security and validation utilities
        _validateInput(sInput, sType) {
            if (!sInput || typeof sInput !== "string") {
                return { isValid: false, message: "Invalid input format" };
            }

            // Sanitize input
            const sSanitized = sInput.trim();

            // XSS prevention patterns
            const aXSSPatterns = [
                /<script/i, /javascript:/i, /on\w+\s*=/i, /<iframe/i,
                /<object/i, /<embed/i, /eval\s*\(/i, /Function\s*\(/i
            ];

            for (let i = 0; i < aXSSPatterns.length; i++) {
                if (aXSSPatterns[i].test(sSanitized)) {
                    this._logAuditEvent("XSS_ATTEMPT", `Blocked XSS attempt in ${ sType}`, sInput);
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
                const aDangerousPatterns = [/eval\s*\(/, /exec\s*\(/, /system\s*\(/, /import\s+/i];
                for (let j = 0; j < aDangerousPatterns.length; j++) {
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

        _getCSRFToken() {
            return new Promise(function(resolve, reject) {
                this._securityUtils.secureAjaxRequest({
                    url: "/a2a/agent4/v1/csrf-token",
                    type: "GET",
                    success(data) {
                        resolve(data.token);
                    },
                    error() {
                        reject(new Error("Failed to retrieve CSRF token"));
                    }
                });
            });
        },

        /**
         * Initializes CSRF token for secure API calls
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @since 1.0.0
         */
        _initializeCSRFToken() {
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent4/v1/csrf-token",
                type: "GET",
                headers: {
                    "X-CSRF-Token": "Fetch",
                    "X-Requested-With": "XMLHttpRequest"
                },
                success: function(data, textStatus, xhr) {
                    this._csrfToken = xhr.getResponseHeader("X-CSRF-Token");
                }.bind(this),
                error: function() {
                    // Fallback to generate token if service not available
                    this._csrfToken = "fetch";
                }.bind(this)
            });
        },

        /**
         * Gets secure headers for AJAX requests including CSRF token
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @returns {object} Security headers object
         * @since 1.0.0
         */
        _getSecureHeaders() {
            return {
                "X-CSRF-Token": this._csrfToken || "Fetch",
                "X-Requested-With": "XMLHttpRequest",
                "Content-Security-Policy": "default-src 'self'"
            };
        },

        /**
         * Opens cached dialog fragments with optimized loading for calculation workflows
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sDialogKey - Dialog cache key
         * @param {string} sFragmentName - Fragment name to load
         * @param {function} [fnCallback] - Optional callback after opening
         * @since 1.0.0
         */
        _openCachedDialog(sDialogKey, sFragmentName, fnCallback) {
            const oView = this.base.getView();

            if (!this._dialogCache[sDialogKey]) {
                // Show loading indicator for complex calculation dialogs
                oView.setBusy(true);

                Fragment.load({
                    id: oView.getId(),
                    name: sFragmentName,
                    controller: this
                }).then((oDialog) => {
                    this._dialogCache[sDialogKey] = oDialog;
                    oView.addDependent(oDialog);
                    oView.setBusy(false);

                    oDialog.open();
                    if (fnCallback) {
                        fnCallback(oDialog);
                    }
                });
            } else {
                this._dialogCache[sDialogKey].open();
                if (fnCallback) {
                    fnCallback(this._dialogCache[sDialogKey]);
                }
            }
        },

        /**
         * Enhanced formula security validation with comprehensive input sanitization
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sFormula - Formula expression to validate
         * @returns {object} Validation result with security assessment
         * @since 1.0.0
         */
        _validateFormulaExpression(sFormula) {
            if (!sFormula || typeof sFormula !== "string") {
                return { isValid: false, message: "Formula is required", securityLevel: "LOW" };
            }

            // Use SecurityUtils for formula validation
            const oValidation = this._securityUtils.validateFormula(sFormula);

            if (!oValidation.isValid) {
                this._logSecurityEvent("FORMULA_SECURITY_VIOLATION", "CRITICAL", oValidation.errors.join(", "), sFormula);
                return {
                    isValid: false,
                    message: oValidation.errors.join(", "),
                    securityLevel: "CRITICAL"
                };
            }

            return {
                isValid: true,
                sanitized: oValidation.sanitizedFormula,
                securityLevel: "SAFE"
            };
        },

        /**
         * Logs security events for audit purposes
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sEventType - Type of security event
         * @param {string} sLevel - Security level (LOW, MEDIUM, HIGH, CRITICAL)
         * @param {string} sMessage - Event description
         * @param {string} [sData] - Additional event data
         * @since 1.0.0
         */
        _logSecurityEvent(sEventType, sLevel, sMessage, sData) {
            const oLogData = {
                timestamp: new Date().toISOString(),
                eventType: sEventType,
                level: sLevel,
                message: sMessage,
                userId: this._getCurrentUserId(),
                sessionId: this._getSessionId(),
                data: sData ? encodeXML(sData) : null
            };

            // Log to SAP logging system
            Log.error(`Security Event: ${ sEventType}`, sMessage, "Agent4.Security");

            // Send to audit service (if available)
            if (sLevel === "CRITICAL" || sLevel === "HIGH") {
                this._sendAuditEvent(oLogData);
            }
        },

        /**
         * Gets current user ID for audit logging
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @returns {string} Current user ID
         * @since 1.0.0
         */
        _getCurrentUserId() {
            try {
                return sap.ushell?.Container?.getUser?.()?.getId?.() || "unknown";
            } catch (e) {
                return "system";
            }
        },

        /**
         * Gets session ID for audit logging
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @returns {string} Session ID
         * @since 1.0.0
         */
        _getSessionId() {
            return this._sessionId || (this._sessionId = `session_${ Date.now() }_${ Math.random().toString(36).substr(2, 9)}`);
        },

        /**
         * Sends audit events to backend logging service
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {object} oLogData - Audit log data
         * @since 1.0.0
         */
        _sendAuditEvent(oLogData) {
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/agent4/v1/audit",
                type: "POST",
                contentType: "application/json",
                headers: this._getSecureHeaders(),
                data: JSON.stringify(oLogData),
                success() {
                    // Audit event sent successfully
                },
                error() {
                    // Log locally if audit service is unavailable
                    Log.warning("Failed to send audit event", oLogData, "Agent4.Audit");
                }
            });
        },

        _secureAjax(oOptions) {
            const that = this;
            return this._getCSRFToken().then(function(sToken) {
                oOptions.headers = oOptions.headers || {};
                oOptions.headers["X-CSRF-Token"] = sToken;

                // Add authorization header if available
                const sAuthToken = that._getAuthToken();
                if (sAuthToken) {
                    oOptions.headers["Authorization"] = `Bearer ${ sAuthToken}`;
                }

                return this._securityUtils.secureAjaxRequest(oOptions);
            });
        },

        _getAuthToken() {
            // Get authentication token from secure storage or session
            return sessionStorage.getItem("a2a_auth_token") || "";
        },

        _logAuditEvent(sEventType, sDescription, sData) {
            // Log security and audit events for comprehensive audit trail
            const oAuditData = {
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
            this._securityUtils.secureAjaxRequest({
                url: "/a2a/common/v1/audit",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oAuditData),
                async: true,
                success() {
                    // Optionally log to browser console for debugging
                    // console.log("Audit event logged:", sEventType);
                },
                error() {
                    // Fallback logging to local storage for offline audit
                    try {
                        let aLocalAudit = JSON.parse(localStorage.getItem("a2a_audit_log") || "[]");
                        aLocalAudit.push(oAuditData);
                        if (aLocalAudit.length > 100) {aLocalAudit = aLocalAudit.slice(-100);} // Keep last 100
                        localStorage.setItem("a2a_audit_log", JSON.stringify(aLocalAudit));
                    } catch (e) {
                        // Silent fail for localStorage issues
                    }
                }
            });
        },

        _getClientIP() {
            // In real implementation, this would come from server-side
            return "client_ip_masked";
        },

        _getCurrentUser() {
            // Get current user from session or model
            return "current_user"; // Placeholder
        },

        /**
         * Enhanced formula security validation with comprehensive threat detection.
         * Validates mathematical expressions against injection attacks, dangerous functions,
         * and unauthorized operations while maintaining calculation integrity.
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sFormula - Formula expression to validate
         * @returns {object} Validation result with security assessment
         * @since 1.0.0
         */
        _validateFormulaSecurity(sFormula) {
            // Enhanced formula security validation
            const oValidation = this._validateInput(sFormula, "formula");
            if (!oValidation.isValid) {
                return oValidation;
            }

            // Check for formula injection patterns
            const aInjectionPatterns = [
                /\$\{.*\}/, // Expression injection
                /\#\{.*\}/, // Hash injection
                /@\w+/, // Annotation injection
                /\.\./ // Path traversal
            ];

            for (let i = 0; i < aInjectionPatterns.length; i++) {
                if (aInjectionPatterns[i].test(sFormula)) {
                    this._logAuditEvent("FORMULA_INJECTION", "Blocked formula injection attempt", sFormula);
                    return { isValid: false, message: "Invalid formula pattern detected" };
                }
            }

            return { isValid: true, sanitized: oValidation.sanitized };
        },

        /**
         * Role-based permission validation for calculation validation operations.
         * Enforces access control for sensitive calculation functions based on user roles.
         * @private
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sAction - Action requiring permission check
         * @returns {boolean} Whether user has required permissions
         * @since 1.0.0
         */
        _checkPermission(sAction) {
            // Role-based permission check
            const aUserRoles = this._getUserRoles();
            const mRequiredPermissions = {
                "CREATE_TASK": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "BATCH_VALIDATION": ["CALCULATION_ADMIN"],
                "FORMULA_BUILDER": ["CALCULATION_ADMIN", "VALIDATION_USER"],
                "VIEW_ANALYTICS": ["CALCULATION_ADMIN", "VALIDATION_VIEWER"]
            };

            const aRequiredRoles = mRequiredPermissions[sAction] || [];
            return aRequiredRoles.some((sRole) => {
                return aUserRoles.includes(sRole);
            });
        },

        _getUserRoles() {
            // Get user roles from session or backend
            return ["VALIDATION_USER"]; // Placeholder
        },

        /**
         * Secure text formatter with XSS protection for UI data binding.
         * Encodes potentially dangerous characters to prevent script injection.
         * @public
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {string} sText - Text to format securely
         * @returns {string} XSS-safe encoded text
         * @since 1.0.0
         */
        formatSecureText(sText) {
            // Secure text formatter for UI bindings
            if (!sText) {return "";}

            // Basic XSS protection
            return jQuery.sap.encodeXML(String(sText));
        },

        /**
         * Secure number formatter for calculation results with overflow protection.
         * Prevents display of dangerous values and ensures numerical stability.
         * @public
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @param {number} nValue - Numerical result to format
         * @returns {string} Safely formatted number or error indicator
         * @since 1.0.0
         */
        formatCalculationResult(nValue) {
            // Secure number formatter
            if (typeof nValue !== "number" || !isFinite(nValue)) {
                return "Invalid";
            }

            // Prevent display of extremely large numbers that could cause issues
            if (Math.abs(nValue) > Number.MAX_SAFE_INTEGER) {
                return "Overflow";
            }

            return nValue.toFixed(6);
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
         * Opens the create validation task dialog with comprehensive security checks.
         * Supports multiple calculation types including financial, mathematical, statistical,
         * engineering, scientific, actuarial, accounting, and physics calculations.
         * @public
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateValidationTask() {
            // Permission check
            if (!this._checkPermission("CREATE_TASK")) {
                MessageBox.error(this._oResourceBundle.getText("message.permissionDenied"));
                return;
            }

            // Initialize create model before opening dialog
            this._initializeCreateModel();

            // Open cached dialog
            this._openCachedDialog("createValidationTask", "a2a.network.agent4.ext.fragment.CreateValidationTask", (oDialog) => {
                // Initialize formula builder if needed
                this._initializeFormulaBuilder(oDialog);
            });
        },

        /**
         * Initialize formula builder with syntax highlighting and validation
         * @private
         * @param {sap.m.Dialog} oDialog - Dialog containing formula builder
         * @since 1.0.0
         */
        _initializeFormulaBuilder(oDialog) {
            // Initialize formula syntax validation
            const oTable = oDialog.getContent()[0].getItems()[2].getContent()[1];
            if (oTable) {
                oTable.attachUpdateFinished(() => {
                    // Add syntax highlighting to formula text areas
                    const aItems = oTable.getItems();
                    aItems.forEach((oItem) => {
                        const oTextArea = oItem.getCells()[0];
                        if (oTextArea && oTextArea.attachLiveChange) {
                            oTextArea.attachLiveChange(this._debouncedValidation);
                        }
                    });
                });
            }
        },

        /**
         * Opens the interactive formula builder dialog with real-time syntax validation.
         * Provides comprehensive formula editor with function library, variable management,
         * and secure expression testing capabilities.
         * @public
         * @memberof a2a.network.agent4.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onFormulaBuilder() {
            // Permission check
            if (!this._checkPermission("FORMULA_BUILDER")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.formulaBuilder"));
                return;
            }

            const oView = this.base.getView();

            if (!this._oFormulaBuilderDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaBuilder",
                    controller: this
                }).then((oDialog) => {
                    this._oFormulaBuilderDialog = oDialog;
                    oView.addDependent(this._oFormulaBuilderDialog);

                    // Initialize formula builder model with secure functions
                    const oModel = new JSONModel({
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
                });
            } else {
                this._oFormulaBuilderDialog.open();
            }
        },

        onBatchValidation() {
            // Permission check
            if (!this._checkPermission("BATCH_VALIDATION")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.batchValidation"));
                return;
            }

            const oTable = this._extensionAPI.getTable();
            const aSelectedContexts = oTable.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageBox.warning(this.getResourceBundle().getText("error.validation.selectTasks"));
                return;
            }

            // Validate selection limit for security
            if (aSelectedContexts.length > 100) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.maxTasks"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("confirm.batchValidation", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchValidation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchValidation(aContexts) {
            const aTaskIds = aContexts.map((oContext) => {
                return this.formatSecureText(oContext.getProperty("ID"));
            });

            this.base.getView().setBusy(true);

            const oRequestData = {
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
            }).then((data) => {
                this.base.getView().setBusy(false);
                MessageBox.success(
                    this.getResourceBundle().getText("success.batchValidationComplete", [this.formatCalculationResult(data.totalFormulas)])
                );
                this._extensionAPI.refresh();
                this._logAuditEvent("BATCH_VALIDATION", "Batch validation started", { taskCount: aTaskIds.length });
            }).catch((xhr) => {
                this.base.getView().setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.batchValidationFailed", [this.formatSecureText(xhr.responseText)]));
                this._logAuditEvent("BATCH_VALIDATION_ERROR", "Batch validation failed", xhr.responseText);
            });
        },

        onCalculationTemplates() {
            // Navigate to calculation templates with audit logging
            this._logAuditEvent("NAVIGATION", "Navigated to calculation templates");
            const oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("CalculationTemplates");
        },

        onValidationAnalytics() {
            // Permission check
            if (!this._checkPermission("VIEW_ANALYTICS")) {
                MessageBox.error(this.getResourceBundle().getText("error.insufficientPermissions.analytics"));
                return;
            }

            const oView = this.base.getView();

            if (!this._oAnalyticsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationAnalytics",
                    controller: this
                }).then((oDialog) => {
                    this._oAnalyticsDialog = oDialog;
                    oView.addDependent(this._oAnalyticsDialog);
                    this._oAnalyticsDialog.open();

                    // Load analytics data securely
                    this._loadValidationAnalytics();
                });
            } else {
                this._oAnalyticsDialog.open();
                this._loadValidationAnalytics();
            }
        },

        _loadValidationAnalytics() {
            this._oAnalyticsDialog.setBusy(true);

            this._secureAjax({
                url: "/a2a/agent4/v1/analytics",
                type: "GET"
            }).then((data) => {
                this._oAnalyticsDialog.setBusy(false);

                // Sanitize analytics data
                const oSanitizedData = this._sanitizeAnalyticsData(data);
                const oModel = new JSONModel(oSanitizedData);
                this._oAnalyticsDialog.setModel(oModel, "analytics");

                // Create charts
                this._createAnalyticsCharts(oSanitizedData);
            }).catch((xhr) => {
                this._oAnalyticsDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.analyticsLoadFailed", [this.formatSecureText(xhr.responseText)]));
            });
        },

        _sanitizeAnalyticsData(oData) {
            // Sanitize analytics data for display
            if (!oData || typeof oData !== "object") {
                return {};
            }

            const oSanitized = {};
            for (const sKey in oData) {
                if (oData.hasOwnProperty(sKey)) {
                    const vValue = oData[sKey];
                    if (typeof vValue === "string") {
                        oSanitized[sKey] = this.formatSecureText(vValue);
                    } else if (typeof vValue === "number" && isFinite(vValue)) {
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

        _createAnalyticsCharts(data) {
            // Create secure analytics charts using SAP Viz framework
            // Implementation would use sanitized data
        },

        onConfirmCreateTask() {
            const oModel = this._oCreateDialog.getModel("create");
            const oData = oModel.getData();

            // Comprehensive validation
            const oTaskNameValidation = this._validateInput(oData.taskName, "taskName");
            if (!oTaskNameValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTaskName", [oTaskNameValidation.message]));
                return;
            }

            const oDescValidation = this._validateInput(oData.description, "description");
            if (!oDescValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidDescription", [oDescValidation.message]));
                return;
            }

            if (!oData.calculationType) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.selectCalculationType"));
                return;
            }

            if (oData.formulas.length === 0) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.addFormulas"));
                return;
            }

            // Validate all formulas
            for (let i = 0; i < oData.formulas.length; i++) {
                const oFormulaValidation = this._validateFormulaSecurity(oData.formulas[i].expression);
                if (!oFormulaValidation.isValid) {
                    MessageBox.error(this.getResourceBundle().getText("error.validation.invalidFormulaPosition", [(i + 1), oFormulaValidation.message]));
                    return;
                }
            }

            this._oCreateDialog.setBusy(true);

            // Prepare sanitized data
            const oSanitizedData = {
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
                formulas: oData.formulas.map((formula) => {
                    return {
                        expression: this._validateFormulaSecurity(formula.expression).sanitized,
                        description: this.formatSecureText(formula.description)
                    };
                })
            };

            this._secureAjax({
                url: "/a2a/agent4/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oSanitizedData)
            }).then((data) => {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show(this.getResourceBundle().getText("success.taskCreated"));
                this._extensionAPI.refresh();
                this._logAuditEvent("TASK_CREATED", "Validation task created", { taskName: oSanitizedData.taskName });
            }).catch((xhr) => {
                this._oCreateDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.taskCreateFailed", [this.formatSecureText(xhr.responseText)]));
                this._logAuditEvent("TASK_CREATE_ERROR", "Failed to create task", xhr.responseText);
            });
        },

        onCancelCreateTask() {
            this._oCreateDialog.close();
        },

        // Enhanced Formula Builder Methods with Security
        onAddFunction(oEvent) {
            const sFunction = this.formatSecureText(oEvent.getSource().getText());
            if (this._validateFormulaFunction(sFunction)) {
                this._insertIntoFormula(`${sFunction }()`);
            }
        },

        onAddOperator(oEvent) {
            const sOperator = this.formatSecureText(oEvent.getSource().getText());
            if (this._validateFormulaOperator(sOperator)) {
                this._insertIntoFormula(sOperator);
            }
        },

        _validateFormulaFunction(sFunction) {
            const aAllowedFunctions = [
                "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "SQRT", "POW",
                "LOG", "EXP", "ABS", "SIN", "COS", "TAN", "PI", "E"
            ];
            return aAllowedFunctions.includes(sFunction);
        },

        _validateFormulaOperator(sOperator) {
            const aAllowedOperators = ["+", "-", "*", "/", "^", "(", ")"];
            return aAllowedOperators.includes(sOperator);
        },

        _insertIntoFormula(sText) {
            const oModel = this._oFormulaBuilderDialog.getModel("formula");
            const sCurrentFormula = oModel.getProperty("/currentFormula");

            // Check formula length limit
            if ((sCurrentFormula + sText).length > 5000) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.formulaTooLong"));
                return;
            }

            oModel.setProperty("/currentFormula", sCurrentFormula + sText);

            // Validate syntax and security with debounced performance optimization
            this._debouncedValidation();
        },

        _validateFormulaSyntax() {
            const oModel = this._oFormulaBuilderDialog.getModel("formula");
            const sFormula = oModel.getProperty("/currentFormula");

            if (!sFormula) {
                oModel.setProperty("/syntaxValid", false);
                oModel.setProperty("/securityChecked", false);
                return;
            }

            // Security validation first
            const oSecurityValidation = this._validateFormulaSecurity(sFormula);
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
            }).then((data) => {
                oModel.setProperty("/syntaxValid", data.valid);
                if (!data.valid) {
                    oModel.setProperty("/syntaxError", this.formatSecureText(data.error));
                }
            }).catch(() => {
                oModel.setProperty("/syntaxValid", false);
                oModel.setProperty("/syntaxError", "Validation service unavailable");
            });
        },

        onTestFormula() {
            const oModel = this._oFormulaBuilderDialog.getModel("formula");
            const sFormula = oModel.getProperty("/currentFormula");

            if (!sFormula) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.enterFormula"));
                return;
            }

            if (!oModel.getProperty("/securityChecked")) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.formulaSecurityRequired"));
                return;
            }

            // Show test data input dialog
            this._showFormulaTestDialog(sFormula);
        },

        _showFormulaTestDialog(sFormula) {
            const oView = this.base.getView();

            if (!this._oTestDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaTest",
                    controller: this
                }).then((oDialog) => {
                    this._oTestDialog = oDialog;
                    oView.addDependent(this._oTestDialog);

                    const oModel = new JSONModel({
                        formula: sFormula,
                        testData: "{}",
                        expectedResult: "",
                        actualResult: "",
                        testPassed: false,
                        variance: 0
                    });
                    this._oTestDialog.setModel(oModel, "test");
                    this._oTestDialog.open();
                });
            } else {
                const oModel = this._oTestDialog.getModel("test");
                oModel.setProperty("/formula", sFormula);
                this._oTestDialog.open();
            }
        },

        onExecuteFormulaTest() {
            const oModel = this._oTestDialog.getModel("test");
            const oData = oModel.getData();

            // Validate test data
            const oTestDataValidation = this._validateInput(oData.testData, "calculation");
            if (!oTestDataValidation.isValid) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTestData"));
                return;
            }

            let oParsedTestData;
            let nExpectedResult;

            try {
                oParsedTestData = JSON.parse(oTestDataValidation.sanitized);
                nExpectedResult = parseFloat(oData.expectedResult);

                if (!isFinite(nExpectedResult)) {
                    MessageBox.error(this.getResourceBundle().getText("error.validation.invalidExpectedResult"));
                    return;
                }
            } catch (e) {
                MessageBox.error(this.getResourceBundle().getText("error.validation.invalidTestDataJson"));
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
            }).then((data) => {
                this._oTestDialog.setBusy(false);

                oModel.setProperty("/actualResult", this.formatCalculationResult(data.result));
                oModel.setProperty("/testPassed", Boolean(data.passed));
                oModel.setProperty("/variance", this.formatCalculationResult(data.variance));

                if (data.passed) {
                    MessageToast.show(this.getResourceBundle().getText("success.formulaTestPassed"));
                } else {
                    MessageBox.warning(this.getResourceBundle().getText("warning.formulaTestFailed", [this.formatCalculationResult(data.variance)]));
                }

                this._logAuditEvent("FORMULA_TEST", "Formula test executed", {
                    passed: data.passed,
                    variance: data.variance
                });
            }).catch((xhr) => {
                this._oTestDialog.setBusy(false);
                MessageBox.error(this.getResourceBundle().getText("error.operation.formulaTestFailed", [this.formatSecureText(xhr.responseText)]));
            });
        }
    });
});