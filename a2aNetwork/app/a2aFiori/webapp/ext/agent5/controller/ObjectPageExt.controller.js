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
], (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, escapeRegExp, sanitizeHTML, Log, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent5.ext.controller.ObjectPageExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 5 Object Page - Quality Assurance and Testing functionality.
     * Provides comprehensive test execution, compliance validation, reporting, and defect management features.
     */
    return ControllerExtension.extend("a2a.network.agent5.ext.controller.ObjectPageExt", {

        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension, sets up performance optimizations and device model.
             * @override
             */
            onInit() {
                // Initialize device model for responsive design
                const oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
                this.base.getView().setModel(oDeviceModel, "device");
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                // Initialize performance optimizations
                this._throttledProgressUpdate = this._throttle(this._updateProgressDisplay.bind(this), 1000);

                // Initialize create model
                this._initializeCreateModel();

                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
            },

            /**
             * @function onExit
             * @description Cleanup method called when the controller is destroyed.
             * Ensures proper cleanup of dialogs, WebSocket connections, and intervals.
             * @override
             */
            onExit() {
                // Cleanup resources and WebSocket connections
                if (this._oComplianceDialog) {
                    this._oComplianceDialog.destroy();
                }
                if (this._oReportDialog) {
                    this._oReportDialog.destroy();
                }
                if (this._oDefectDialog) {
                    this._oDefectDialog.destroy();
                }
                if (this._oRegressionDialog) {
                    this._oRegressionDialog.destroy();
                }
                if (this._ws) {
                    this._ws.close();
                }
                if (this._progressInterval) {
                    clearInterval(this._progressInterval);
                }
            }
        },

        // Dialog caching for performance
        _dialogCache: {},

        /**
         * Initialize the create model with default values and validation states
         * @private
         * @since 1.0.0
         */
        _initializeCreateModel() {
            const oCreateModel = new JSONModel({
                taskName: "",
                description: "",
                testSuite: "",
                testType: "",
                targetApplication: "",
                severity: "MEDIUM",
                testFramework: "SELENIUM",
                testEnvironment: "TEST",
                testDataSource: "",
                automationLevel: 50,
                parallelExecution: true,
                retryOnFailure: true,
                detailedLogs: true,
                captureScreenshots: true,
                testTimeout: 30,
                testCases: [],
                complianceStandard: "",
                requiresApproval: false,
                auditTrail: true,
                regulatoryCompliance: false,
                securityValidation: false,
                performanceValidation: false,
                accessibilityTesting: false,
                usabilityTesting: false,
                minPassRate: 80,
                maxExecutionTime: 4,
                coverageThreshold: 70,
                isValid: false,
                taskNameState: "None",
                taskNameStateText: "",
                testTypeState: "None",
                testTypeStateText: "",
                targetApplicationState: "None",
                targetApplicationStateText: ""
            });
            this.base.getView().setModel(oCreateModel, "create");
        },

        /**
         * Validation handler for task name changes
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @public
         * @since 1.0.0
         */
        onTaskNameChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oModel = this.base.getView().getModel("create");

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
         * Handler for test type changes
         * @param {sap.ui.base.Event} oEvent - Select change event
         * @public
         * @since 1.0.0
         */
        onTestTypeChange(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const oModel = this.base.getView().getModel("create");

            if (!sSelectedKey) {
                oModel.setProperty("/testTypeState", "Error");
                oModel.setProperty("/testTypeStateText", this._oResourceBundle.getText("validation.testTypeRequired"));
            } else {
                oModel.setProperty("/testTypeState", "Success");
                oModel.setProperty("/testTypeStateText", "");

                // Auto-suggest test framework based on test type
                switch (sSelectedKey) {
                case "UNIT":
                    oModel.setProperty("/testFramework", "JEST");
                    oModel.setProperty("/automationLevel", 90);
                    break;
                case "INTEGRATION":
                case "SYSTEM":
                    oModel.setProperty("/testFramework", "SELENIUM");
                    oModel.setProperty("/automationLevel", 70);
                    break;
                case "PERFORMANCE":
                    oModel.setProperty("/testFramework", "JMETER");
                    oModel.setProperty("/automationLevel", 80);
                    break;
                case "SECURITY":
                    oModel.setProperty("/securityValidation", true);
                    break;
                case "ACCESSIBILITY":
                    oModel.setProperty("/accessibilityTesting", true);
                    oModel.setProperty("/complianceStandard", "WCAG");
                    break;
                }
            }

            this._validateForm();
        },

        /**
         * Handler for target application changes
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @public
         * @since 1.0.0
         */
        onTargetApplicationChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oModel = this.base.getView().getModel("create");

            if (!sValue || sValue.trim().length === 0) {
                oModel.setProperty("/targetApplicationState", "Error");
                oModel.setProperty("/targetApplicationStateText", this._oResourceBundle.getText("validation.targetApplicationRequired"));
            } else if (sValue.length > 200) {
                oModel.setProperty("/targetApplicationState", "Error");
                oModel.setProperty("/targetApplicationStateText", "Target application name too long");
            } else {
                oModel.setProperty("/targetApplicationState", "Success");
                oModel.setProperty("/targetApplicationStateText", "");
            }

            this._validateForm();
        },

        /**
         * Handler for test framework changes
         * @param {sap.ui.base.Event} oEvent - Select change event
         * @public
         * @since 1.0.0
         */
        onTestFrameworkChange(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const oModel = this.base.getView().getModel("create");

            // Adjust timeout based on framework
            switch (sSelectedKey) {
            case "CYPRESS":
            case "PLAYWRIGHT":
                oModel.setProperty("/testTimeout", 20);
                oModel.setProperty("/captureScreenshots", true);
                break;
            case "SELENIUM":
                oModel.setProperty("/testTimeout", 30);
                break;
            case "JEST":
            case "MOCHA":
                oModel.setProperty("/testTimeout", 10);
                break;
            }
        },

        /**
         * Handler for compliance standard changes
         * @param {sap.ui.base.Event} oEvent - Select change event
         * @public
         * @since 1.0.0
         */
        onComplianceStandardChange(oEvent) {
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            const oModel = this.base.getView().getModel("create");

            // Auto-enable relevant compliance requirements
            switch (sSelectedKey) {
            case "GDPR":
            case "HIPAA":
                oModel.setProperty("/requiresApproval", true);
                oModel.setProperty("/auditTrail", true);
                oModel.setProperty("/securityValidation", true);
                break;
            case "SOX":
                oModel.setProperty("/requiresApproval", true);
                oModel.setProperty("/auditTrail", true);
                oModel.setProperty("/regulatoryCompliance", true);
                break;
            case "WCAG":
                oModel.setProperty("/accessibilityTesting", true);
                oModel.setProperty("/usabilityTesting", true);
                break;
            case "PCI":
                oModel.setProperty("/securityValidation", true);
                oModel.setProperty("/performanceValidation", true);
                break;
            }
        },

        /**
         * Handler for adding test cases
         * @public
         * @since 1.0.0
         */
        onAddTestCase() {
            const oModel = this.base.getView().getModel("create");
            const aTestCases = oModel.getProperty("/testCases");

            aTestCases.push({
                testName: "",
                description: "",
                category: "FUNCTIONAL",
                priority: "P3",
                automationLevel: "AUTOMATED",
                id: Date.now()
            });

            oModel.setProperty("/testCases", aTestCases);
            this._validateForm();
        },

        /**
         * Handler for test case name changes
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @public
         * @since 1.0.0
         */
        onTestCaseNameChange(oEvent) {
            this._validateForm();
        },

        /**
         * Validate the entire form
         * @private
         * @since 1.0.0
         */
        _validateForm() {
            const oModel = this.base.getView().getModel("create");
            const oData = oModel.getData();

            const bValid = oData.taskNameState === "Success" &&
                        oData.testTypeState === "Success" &&
                        oData.targetApplicationState === "Success";

            oModel.setProperty("/isValid", bValid);
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one for performance optimization.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name to load
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog instance
         * @private
         */
        _getOrCreateDialog(sDialogId, sFragmentName) {
            const that = this;

            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }

            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then((oDialog) => {
                that._dialogCache[sDialogId] = oDialog;
                that.base.getView().addDependent(oDialog);

                // Add keyboard navigation support
                oDialog.addEventDelegate({
                    onAfterRendering() {
                        that._enableKeyboardNavigation(oDialog);
                    }
                });

                return oDialog;
            });
        },

        /**
         * @function _enableKeyboardNavigation
         * @description Enables keyboard navigation for dialog accessibility.
         * @param {sap.m.Dialog} oDialog - Dialog instance
         * @private
         */
        _enableKeyboardNavigation(oDialog) {
            const $dialog = oDialog.$();

            // Set tabindex for focusable elements
            $dialog.find("input, button, select, textarea").attr("tabindex", "0");

            // Handle escape key to close dialog
            $dialog.on("keydown", (e) => {
                if (e.key === "Escape") {
                    oDialog.close();
                }
            });
        },

        /**
         * @function _withErrorRecovery
         * @description Wraps async operations with error recovery and retry logic.
         * @param {Function} fnOperation - Async operation to execute
         * @param {Object} oOptions - Recovery options
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery(fnOperation, oOptions) {
            const that = this;
            const oDefaults = {
                retries: 3,
                retryDelay: 1000,
                fallback: null,
                errorHandler: null
            };
            const oSettings = Object.assign({}, oDefaults, oOptions);

            function attempt(retriesLeft) {
                return fnOperation().catch((error) => {
                    if (retriesLeft > 0) {
                        that._logAuditEvent("ERROR_RECOVERY_RETRY", "Retrying operation", {
                            retriesLeft,
                            error: error.message
                        });

                        return new Promise((resolve) => {
                            setTimeout(resolve, oSettings.retryDelay);
                        }).then(() => {
                            return attempt(retriesLeft - 1);
                        });
                    }

                    if (oSettings.errorHandler) {
                        oSettings.errorHandler(error);
                    }

                    if (oSettings.fallback) {
                        return oSettings.fallback();
                    }

                    throw error;
                });
            }

            return attempt(oSettings.retries);
        },

        /**
         * @function _getResponsiveValue
         * @description Returns appropriate value based on device type.
         * @param {*} phoneValue - Value for phone
         * @param {*} tabletValue - Value for tablet
         * @param {*} desktopValue - Value for desktop
         * @returns {*} Appropriate value for current device
         * @private
         */
        _getResponsiveValue(phoneValue, tabletValue, desktopValue) {
            if (sap.ui.Device.system.phone) {
                return phoneValue;
            } else if (sap.ui.Device.system.tablet) {
                return tabletValue;
            }
            return desktopValue;

        },

        /**
         * @function _throttle
         * @description Creates a throttled version of a function that limits execution frequency.
         * @param {Function} fn - Function to throttle
         * @param {number} limit - Time limit in milliseconds between executions
         * @returns {Function} Throttled function
         * @private
         */
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

        /**
         * @function _validateInput
         * @description Validates and sanitizes user input to prevent XSS and injection attacks.
         * @param {string} sInput - Input string to validate
         * @param {string} sType - Type of input for specific validation rules
         * @returns {Object} Validation result with isValid flag and sanitized value
         * @private
         */
        _validateInput(sInput, sType) {
            if (!sInput || typeof sInput !== "string") {
                return { isValid: false, message: "Invalid input format" };
            }

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
            case "taskId":
                if (!/^[a-zA-Z0-9_-]+$/.test(sSanitized) || sSanitized.length > 50) {
                    return { isValid: false, message: "Invalid task ID format" };
                }
                break;
            case "reportType":
                var aValidTypes = ["EXECUTIVE", "DETAILED", "SUMMARY", "COMPLIANCE"];
                if (!aValidTypes.includes(sSanitized)) {
                    return { isValid: false, message: "Invalid report type" };
                }
                break;
            case "defectTitle":
                if (sSanitized.length > 200) {
                    return { isValid: false, message: "Defect title too long" };
                }
                break;
            case "complianceStandard":
                var aValidStandards = ["ISO27001", "SOX", "GDPR", "HIPAA", "PCI", "FISMA", "WCAG"];
                if (sSanitized && !aValidStandards.includes(sSanitized)) {
                    return { isValid: false, message: "Invalid compliance standard" };
                }
                break;
            }

            return { isValid: true, sanitized: sSanitized };
        },

        /**
         * @function _getCSRFToken
         * @description Retrieves CSRF token for secure API requests.
         * @returns {Promise<string>} Promise resolving to CSRF token
         * @private
         */
        _getCSRFToken() {
            return new Promise(function(resolve, reject) {
                this._securityUtils.secureAjaxRequest({
                    url: "/a2a/agent5/v1/csrf-token",
                    type: "GET",
                    success(data) {
                        resolve(data.token);
                    },
                    error() {
                        reject("Failed to retrieve CSRF token");
                    }
                });
            });
        },

        /**
         * @function _secureAjax
         * @description Performs secure AJAX requests with CSRF token and authentication.
         * @param {Object} oOptions - jQuery AJAX options
         * @returns {Promise} jQuery promise for the AJAX request
         * @private
         */
        _secureAjax(oOptions) {
            const that = this;
            return this._getCSRFToken().then(function(sToken) {
                oOptions.headers = oOptions.headers || {};
                oOptions.headers["X-CSRF-Token"] = sToken;

                const sAuthToken = that._getAuthToken();
                if (sAuthToken) {
                    oOptions.headers["Authorization"] = `Bearer ${ sAuthToken}`;
                }

                return this._securityUtils.secureAjaxRequest(oOptions);
            });
        },

        _getAuthToken() {
            return sessionStorage.getItem("a2a_auth_token") || "";
        },

        /**
         * @function _logAuditEvent
         * @description Logs security and audit events with comprehensive details.
         * @param {string} sEventType - Type of audit event
         * @param {string} sDescription - Human-readable description
         * @param {*} sData - Additional data to log
         * @private
         */
        _logAuditEvent(sEventType, sDescription, sData) {
            // Comprehensive audit trail logging
            const oAuditData = {
                timestamp: new Date().toISOString(),
                eventType: sEventType,
                description: sDescription,
                data: sData ? JSON.stringify(sData).substring(0, 500) : "",
                user: this._getCurrentUser(),
                component: "Agent5.ObjectPage",
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
                success() {
                    console.log("Audit trail recorded:", sEventType);
                },
                error() {
                    // Fallback audit logging to local storage
                    try {
                        let aLocalAudit = JSON.parse(localStorage.getItem("a2a_audit_log") || "[]");
                        aLocalAudit.push(oAuditData);
                        if (aLocalAudit.length > 100) {aLocalAudit = aLocalAudit.slice(-100);}
                        localStorage.setItem("a2a_audit_log", JSON.stringify(aLocalAudit));
                    } catch (e) {
                        // Silent fail for localStorage issues
                    }
                }
            });
        },

        _getSessionId() {
            return sessionStorage.getItem("a2a_session_id") || "unknown";
        },

        _getClientIP() {
            return "client_ip_masked";
        },

        _getCurrentUser() {
            return "current_user"; // Placeholder
        },

        /**
         * @function _checkPermission
         * @description Checks if current user has permission for specified action.
         * @param {string} sAction - Action to check permission for
         * @returns {boolean} True if user has permission
         * @private
         */
        _checkPermission(sAction) {
            const aUserRoles = this._getUserRoles();
            const mRequiredPermissions = {
                "EXECUTE_TESTS": ["QA_ADMIN", "QA_USER"],
                "VALIDATE_COMPLIANCE": ["QA_ADMIN", "COMPLIANCE_OFFICER"],
                "GENERATE_REPORTS": ["QA_ADMIN", "QA_MANAGER"],
                "CREATE_DEFECTS": ["QA_ADMIN", "QA_USER"],
                "SCHEDULE_REGRESSION": ["QA_ADMIN"]
            };

            const aRequiredRoles = mRequiredPermissions[sAction] || [];
            return aRequiredRoles.some((sRole) => {
                return aUserRoles.includes(sRole);
            });
        },

        _getUserRoles() {
            return ["QA_USER"]; // Placeholder
        },

        /**
         * @function formatSecureText
         * @description Formatter function to encode text for XSS prevention.
         * @param {string} sText - Text to format
         * @returns {string} Encoded text safe for display
         * @public
         */
        formatSecureText(sText) {
            if (!sText) {return "";}
            return jQuery.sap.encodeXML(String(sText));
        },

        /**
         * @function formatTestCount
         * @description Formatter function for test count values with validation.
         * @param {number} nValue - Numeric value to format
         * @returns {string} Formatted count string
         * @public
         */
        formatTestCount(nValue) {
            if (typeof nValue !== "number" || !isFinite(nValue)) {
                return "0";
            }
            return Math.max(0, Math.min(nValue, 999999)).toString();
        },

        /**
         * @function formatDuration
         * @description Formatter function to convert milliseconds to human-readable duration.
         * @param {number} nMilliseconds - Duration in milliseconds
         * @returns {string} Formatted duration string
         * @public
         */
        formatDuration(nMilliseconds) {
            if (typeof nMilliseconds !== "number" || !isFinite(nMilliseconds)) {
                return "0ms";
            }
            const nSeconds = Math.floor(nMilliseconds / 1000);
            const nMinutes = Math.floor(nSeconds / 60);
            return nMinutes > 0 ? `${nMinutes }m ${ nSeconds % 60 }s` : `${nSeconds }s`;
        },

        _validateWebSocketURL(sURL) {
            // Validate WebSocket URL for security
            try {
                const oURL = new URL(sURL);
                return oURL.protocol === "wss:" &&
                       oURL.hostname === window.location.hostname &&
                       oURL.pathname.startsWith("/a2a/agent5/");
            } catch (e) {
                return false;
            }
        },

        /**
         * @function onExecuteTests
         * @description Handles test execution action with permission and validation checks.
         * @public
         */
        onExecuteTests() {
            if (!this._checkPermission("EXECUTE_TESTS")) {
                MessageBox.error("Insufficient permissions to execute tests");
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");
            const iTotalTests = oContext.getProperty("totalTests");

            // Validate task data
            const oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error(`Invalid task ID: ${ oTaskIdValidation.message}`);
                return;
            }

            if (iTotalTests === 0) {
                MessageBox.error("No test cases defined for this task. Please add test cases before execution.");
                return;
            }

            // Validate test count for security
            if (iTotalTests > 10000) {
                MessageBox.error("Too many test cases. Maximum 10,000 tests allowed for security reasons.");
                return;
            }

            MessageBox.confirm(`Execute ${ this.formatTestCount(iTotalTests) } tests for '${ this.formatSecureText(sTaskName) }'?`, {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeTestTask(oTaskIdValidation.sanitized);
                    }
                }.bind(this)
            });
        },

        /**
         * @function _executeTestTask
         * @description Executes test task via secure API call.
         * @param {string} sTaskId - Sanitized task ID
         * @private
         */
        _executeTestTask(sTaskId) {
            this._extensionAPI.getView().setBusy(true);

            this._secureAjax({
                url: `/a2a/agent5/v1/tasks/${ encodeURIComponent(sTaskId) }/execute`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    securityMode: "STRICT",
                    maxExecutionTime: 3600000, // 1 hour max
                    enableSafeguards: true
                })
            }).then((data) => {
                this._extensionAPI.getView().setBusy(false);
                MessageToast.show("Test execution started");
                this._extensionAPI.refresh();

                // Start secure real-time monitoring
                this._startTestExecutionMonitoring(sTaskId, data.executionId);
                this._logAuditEvent("TEST_EXECUTION_STARTED", "Test execution started", { taskId: sTaskId });
            }).catch((xhr) => {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error(`Failed to start test execution: ${ this.formatSecureText(xhr.responseText)}`);
                this._logAuditEvent("TEST_EXECUTION_ERROR", "Failed to start test execution", xhr.responseText);
            });
        },

        /**
         * @function _startTestExecutionMonitoring
         * @description Establishes WebSocket connection for real-time test execution monitoring.
         * @param {string} sTaskId - Task ID
         * @param {string} sExecutionId - Execution ID
         * @private
         */
        _startTestExecutionMonitoring(sTaskId, sExecutionId) {
            // Secure WebSocket for real-time test execution updates
            const sWebSocketURL = `wss://${ window.location.host }/a2a/agent5/v1/tasks/${
                encodeURIComponent(sTaskId) }/execution/${ encodeURIComponent(sExecutionId) }/ws`;

            // Validate WebSocket URL
            if (!this._validateWebSocketURL(sWebSocketURL)) {
                MessageBox.error("Invalid WebSocket URL for monitoring");
                return;
            }

            this._ws = new WebSocket(sWebSocketURL);

            // Set connection timeout
            const connectionTimeout = setTimeout(() => {
                if (this._ws && this._ws.readyState === WebSocket.CONNECTING) {
                    this._ws.close();
                    MessageBox.error("WebSocket connection timeout");
                }
            }, 10000);

            this._ws.onopen = function() {
                clearTimeout(connectionTimeout);
                this._logAuditEvent("WEBSOCKET_CONNECTED", "Test monitoring WebSocket connected");
            }.bind(this);

            this._ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);

                    // Validate message data
                    if (!data || typeof data !== "object") {
                        return;
                    }

                    switch (data.type) {
                    case "test_started":
                        if (data.testName) {
                            MessageToast.show(`Test started: ${ this.formatSecureText(data.testName)}`);
                        }
                        break;
                    case "test_completed":
                        if (data.testName && data.result) {
                            const sStatus = data.result === "PASS" ? "✓" : "✗";
                            const sDuration = this.formatDuration(data.duration);
                            MessageToast.show(`${sStatus } ${ this.formatSecureText(data.testName) } (${ sDuration })`);
                        }
                        break;
                    case "suite_completed":
                        this._ws.close();
                        this._extensionAPI.refresh();
                        this._showExecutionSummary(data);
                        break;
                    case "progress_update":
                        this._throttledProgressUpdate(data);
                        break;
                    case "error":
                        this._ws.close();
                        MessageBox.error(`Test execution error: ${ this.formatSecureText(data.error)}`);
                        this._logAuditEvent("TEST_EXECUTION_WS_ERROR", "WebSocket error", data.error);
                        break;
                    }
                } catch (e) {
                    // Ignore malformed messages
                }
            }.bind(this);

            this._ws.onerror = function() {
                MessageBox.error("Lost connection to test execution");
                this._logAuditEvent("WEBSOCKET_ERROR", "WebSocket connection error");
            }.bind(this);

            this._ws.onclose = function() {
                clearTimeout(connectionTimeout);
                this._logAuditEvent("WEBSOCKET_CLOSED", "Test monitoring WebSocket closed");
            }.bind(this);
        },

        _updateProgressDisplay(data) {
            // Throttled progress update to prevent UI spam
            if (data.progress !== undefined) {
                const _nProgress = Math.max(0, Math.min(100, parseFloat(data.progress) || 0));
                // Update progress display if UI element exists
            }
        },

        _showExecutionSummary(data) {
            const sMessage = "Test Execution Summary:\n\n" +
                          `Total Tests: ${ this.formatTestCount(data.totalTests) }\n` +
                          `Passed: ${ this.formatTestCount(data.passedTests) }\n` +
                          `Failed: ${ this.formatTestCount(data.failedTests) }\n` +
                          `Skipped: ${ this.formatTestCount(data.skippedTests) }\n` +
                          `Success Rate: ${ this.formatTestCount(data.successRate) }%\n` +
                          `Execution Time: ${ this.formatDuration(data.totalDuration)}`;

            MessageBox.success(sMessage, {
                title: "Execution Completed"
            });

            this._logAuditEvent("TEST_EXECUTION_SUMMARY", "Test execution summary displayed", data);
        },

        /**
         * @function onValidateCompliance
         * @description Handles compliance validation action for various standards.
         * @public
         */
        onValidateCompliance() {
            if (!this._checkPermission("VALIDATE_COMPLIANCE")) {
                MessageBox.error("Insufficient permissions to validate compliance");
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sComplianceStandard = oContext.getProperty("complianceStandard");

            const oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error("Invalid task ID");
                return;
            }

            if (!sComplianceStandard) {
                MessageBox.error("No compliance standard specified for this task");
                return;
            }

            const oStandardValidation = this._validateInput(sComplianceStandard, "complianceStandard");
            if (!oStandardValidation.isValid) {
                MessageBox.error(`Invalid compliance standard: ${ oStandardValidation.message}`);
                return;
            }

            this._getOrCreateDialog("compliance", "a2a.network.agent5.ext.fragment.ComplianceValidation")
                .then((oDialog) => {

                    const oModel = new JSONModel({
                        taskId: oTaskIdValidation.sanitized,
                        standard: oStandardValidation.sanitized,
                        validationScope: "FULL",
                        generateCertificate: true,
                        includeEvidence: true,
                        maxValidationTime: 1800000 // 30 minutes max
                    });
                    oDialog.setModel(oModel, "compliance");
                    oDialog.open();
                });
        },

        onExecuteComplianceValidation() {
            const oModel = this._oComplianceDialog.getModel("compliance");
            const oData = oModel.getData();

            this._oComplianceDialog.setBusy(true);

            this._secureAjax({
                url: `/a2a/agent5/v1/tasks/${ encodeURIComponent(oData.taskId) }/validate-compliance`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    standard: oData.standard,
                    scope: oData.validationScope,
                    generateCertificate: Boolean(oData.generateCertificate),
                    includeEvidence: Boolean(oData.includeEvidence),
                    maxValidationTime: Math.min(oData.maxValidationTime, 1800000), // Security limit
                    securityMode: "STRICT"
                })
            }).then((data) => {
                this._oComplianceDialog.setBusy(false);

                // Validate compliance response
                const oValidatedData = this._validateComplianceResponse(data);
                if (!oValidatedData.isValid) {
                    MessageBox.error("Invalid compliance validation response");
                    return;
                }

                if (data.compliant) {
                    MessageBox.success(
                        "Compliance validation passed!\n" +
                        `Standard: ${ this.formatSecureText(data.standard) }\n` +
                        `Score: ${ this.formatTestCount(data.complianceScore) }%\n` +
                        `Certificate ID: ${ this.formatSecureText(data.certificateId)}`
                    );
                    this._logAuditEvent("COMPLIANCE_VALIDATED", "Compliance validation passed", {
                        standard: data.standard,
                        score: data.complianceScore
                    });
                } else {
                    this._showComplianceIssues(data.issues || []);
                }

                this._oComplianceDialog.close();
                this._extensionAPI.refresh();
            }).catch((xhr) => {
                this._oComplianceDialog.setBusy(false);
                MessageBox.error(`Compliance validation failed: ${ this.formatSecureText(xhr.responseText)}`);
                this._logAuditEvent("COMPLIANCE_VALIDATION_ERROR", "Compliance validation failed", xhr.responseText);
            });
        },

        _validateComplianceResponse(oData) {
            if (!oData || typeof oData !== "object") {
                return { isValid: false };
            }

            // Validate required fields
            if (typeof oData.compliant !== "boolean") {
                return { isValid: false };
            }

            return { isValid: true };
        },

        _showComplianceIssues(aIssues) {
            let sMessage = "Compliance Issues Found:\n\n";
            aIssues.slice(0, 10).forEach((issue, index) => { // Limit to 10 issues
                sMessage += `${index + 1 }. ${ this.formatSecureText(issue.description) }\n`;
                sMessage += `   Severity: ${ this.formatSecureText(issue.severity) }\n`;
                sMessage += `   Recommendation: ${ this.formatSecureText(issue.recommendation) }\n\n`;
            });

            MessageBox.warning(sMessage, {
                title: "Compliance Validation Failed"
            });

            this._logAuditEvent("COMPLIANCE_ISSUES", "Compliance issues found", { issueCount: aIssues.length });
        },

        /**
         * @function onGenerateTestReport
         * @description Initiates test report generation dialog.
         * @public
         */
        onGenerateTestReport() {
            if (!this._checkPermission("GENERATE_REPORTS")) {
                MessageBox.error("Insufficient permissions to generate reports");
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const sTaskName = oContext.getProperty("taskName");

            const oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error("Invalid task ID");
                return;
            }

            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.TestReport",
                    controller: this
                }).then((oDialog) => {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);

                    const oModel = new JSONModel({
                        taskId: oTaskIdValidation.sanitized,
                        taskName: this.formatSecureText(sTaskName),
                        reportType: "EXECUTIVE",
                        format: "PDF",
                        includeDetails: true,
                        includeScreenshots: true,
                        includeMetrics: true,
                        includeRecommendations: true,
                        maxFileSize: 25, // MB limit for security
                        distribution: []
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                });
            } else {
                this._oReportDialog.open();
            }
        },

        onGenerateReport() {
            const oDialog = this._dialogCache.report;
            const oModel = oDialog.getModel("report");
            const oData = oModel.getData();

            // Validate report type
            const oReportTypeValidation = this._validateInput(oData.reportType, "reportType");
            if (!oReportTypeValidation.isValid) {
                MessageBox.error("Invalid report type");
                return;
            }

            oDialog.setBusy(true);

            this._secureAjax({
                url: `/a2a/agent5/v1/tasks/${ encodeURIComponent(oData.taskId) }/generate-report`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    type: oReportTypeValidation.sanitized,
                    format: oData.format === "PDF" ? "PDF" : "HTML", // Whitelist formats
                    includeDetails: Boolean(oData.includeDetails),
                    includeScreenshots: Boolean(oData.includeScreenshots),
                    includeMetrics: Boolean(oData.includeMetrics),
                    includeRecommendations: Boolean(oData.includeRecommendations),
                    maxFileSize: Math.min(oData.maxFileSize || 25, 50), // Limit file size
                    distribution: (oData.distribution || []).slice(0, 10) // Limit recipients
                })
            }).then((data) => {
                oDialog.setBusy(false);
                oDialog.close();

                MessageBox.success(
                    "Test report generated successfully!",
                    {
                        actions: ["Download", MessageBox.Action.CLOSE],
                        onClose: function(oAction) {
                            if (oAction === "Download") {
                                this._secureDownload(data.downloadUrl);
                            }
                        }.bind(this)
                    }
                );

                this._logAuditEvent("REPORT_GENERATED", "Test report generated", {
                    taskId: oData.taskId,
                    type: oData.reportType
                });
            }).catch((xhr) => {
                oDialog.setBusy(false);
                MessageBox.error(`Report generation failed: ${ this.formatSecureText(xhr.responseText)}`);
                this._logAuditEvent("REPORT_GENERATION_ERROR", "Report generation failed", xhr.responseText);
            });
        },

        _secureDownload(sUrl) {
            // Validate download URL for security
            try {
                const oUrl = new URL(sUrl, window.location.origin);
                if (oUrl.origin === window.location.origin && oUrl.pathname.startsWith("/a2a/agent5/")) {
                    window.open(sUrl, "_blank");
                    this._logAuditEvent("REPORT_DOWNLOADED", "Report download initiated", sUrl);
                } else {
                    MessageBox.error("Invalid download URL");
                }
            } catch (e) {
                MessageBox.error("Invalid download URL format");
            }
        },

        /**
         * @function onCreateDefect
         * @description Opens dialog to create defects from failed tests.
         * @public
         */
        onCreateDefect() {
            if (!this._checkPermission("CREATE_DEFECTS")) {
                MessageBox.error("Insufficient permissions to create defects");
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");
            const iFailedTests = oContext.getProperty("failedTests");

            const oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error("Invalid task ID");
                return;
            }

            if (iFailedTests === 0) {
                MessageBox.information("No failed tests found to create defects from.");
                return;
            }

            if (!this._oDefectDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.CreateDefect",
                    controller: this
                }).then((oDialog) => {
                    this._oDefectDialog = oDialog;
                    this.base.getView().addDependent(this._oDefectDialog);

                    this._loadFailedTests(oTaskIdValidation.sanitized);
                    this._oDefectDialog.open();
                });
            } else {
                this._loadFailedTests(oTaskIdValidation.sanitized);
                this._oDefectDialog.open();
            }
        },

        /**
         * @function _loadFailedTests
         * @description Loads failed tests with lazy loading and virtualization.
         * @param {string} sTaskId - Task ID
         * @private
         */
        _loadFailedTests(sTaskId) {
            const oDialog = this._dialogCache.defect;
            oDialog.setBusy(true);

            // Initialize lazy loading model
            const oLazyModel = new JSONModel({
                taskId: sTaskId,
                failedTests: [],
                loadedTests: 0,
                totalTests: 0,
                pageSize: 20,
                isLoading: false,
                selectedTests: [],
                defectInfo: {
                    title: "",
                    description: "",
                    severity: "MEDIUM",
                    priority: "MEDIUM",
                    assignee: "",
                    component: "",
                    labels: []
                }
            });

            oDialog.setModel(oLazyModel, "defect");

            // Load initial batch
            this._loadFailedTestsBatch(sTaskId, 0, 20).then((result) => {
                oDialog.setBusy(false);

                const oData = oLazyModel.getData();
                oData.failedTests = result.tests;
                oData.totalTests = result.total;
                oData.loadedTests = result.tests.length;
                oLazyModel.setData(oData);

                // Setup scroll listener for lazy loading
                this._setupLazyLoadingScroll(oDialog, sTaskId);
            }).catch((xhr) => {
                oDialog.setBusy(false);
                MessageBox.error(`Failed to load failed tests: ${ this.formatSecureText(xhr.responseText)}`);
            });
        },

        /**
         * @function _loadFailedTestsBatch
         * @description Loads a batch of failed tests.
         * @param {string} sTaskId - Task ID
         * @param {number} iSkip - Number of items to skip
         * @param {number} iTop - Number of items to load
         * @returns {Promise} Promise resolving to test data
         * @private
         */
        _loadFailedTestsBatch(sTaskId, iSkip, iTop) {
            return this._secureAjax({
                url: `/a2a/agent5/v1/tasks/${ encodeURIComponent(sTaskId) }/failed-tests`,
                type: "GET",
                data: {
                    $skip: iSkip,
                    $top: iTop
                }
            }).then((data) => {
                return {
                    tests: this._sanitizeFailedTestsData(data.tests || []),
                    total: data.total || 0
                };
            });
        },

        /**
         * @function _setupLazyLoadingScroll
         * @description Sets up scroll listener for lazy loading more tests.
         * @param {sap.m.Dialog} oDialog - Dialog instance
         * @param {string} sTaskId - Task ID
         * @private
         */
        _setupLazyLoadingScroll(oDialog, sTaskId) {
            const that = this;
            const oModel = oDialog.getModel("defect");

            // Find scrollable container
            const $scrollContainer = oDialog.$().find(".sapMScrollContainer");

            if ($scrollContainer.length > 0) {
                $scrollContainer.on("scroll", this._throttle(() => {
                    const scrollTop = $scrollContainer.scrollTop();
                    const scrollHeight = $scrollContainer[0].scrollHeight;
                    const containerHeight = $scrollContainer.height();

                    // Load more when near bottom (90%)
                    if (scrollTop + containerHeight >= scrollHeight * 0.9) {
                        that._loadMoreFailedTests(sTaskId, oModel);
                    }
                }, 200));
            }
        },

        /**
         * @function _loadMoreFailedTests
         * @description Loads more failed tests when scrolling near bottom.
         * @param {string} sTaskId - Task ID
         * @param {sap.ui.model.json.JSONModel} oModel - Model instance
         * @private
         */
        _loadMoreFailedTests(sTaskId, oModel) {
            const oData = oModel.getData();

            // Check if already loading or all loaded
            if (oData.isLoading || oData.loadedTests >= oData.totalTests) {
                return;
            }

            oData.isLoading = true;
            oModel.setData(oData);

            this._loadFailedTestsBatch(sTaskId, oData.loadedTests, oData.pageSize)
                .then((result) => {
                    const oUpdatedData = oModel.getData();
                    oUpdatedData.failedTests = oUpdatedData.failedTests.concat(result.tests);
                    oUpdatedData.loadedTests += result.tests.length;
                    oUpdatedData.isLoading = false;
                    oModel.setData(oUpdatedData);
                })
                .catch(() => {
                    const oUpdatedData = oModel.getData();
                    oUpdatedData.isLoading = false;
                    oModel.setData(oUpdatedData);
                    MessageToast.show("Failed to load more tests");
                });
        },

        _sanitizeFailedTestsData(aTests) {
            return aTests.slice(0, 100).map((test) => { // Limit to 100 failed tests
                return {
                    id: this.formatSecureText(test.id),
                    testName: this.formatSecureText(test.testName),
                    errorMessage: this.formatSecureText(test.errorMessage),
                    stackTrace: test.stackTrace ? this.formatSecureText(test.stackTrace.substring(0, 1000)) : "",
                    duration: typeof test.duration === "number" ? Math.min(test.duration, 3600000) : 0
                };
            });
        },

        onSubmitDefect() {
            const oDialog = this._dialogCache.defect;
            const oModel = oDialog.getModel("defect");
            const oData = oModel.getData();

            if (oData.selectedTests.length === 0) {
                MessageBox.error("Please select at least one failed test");
                return;
            }

            // Limit selection for security
            if (oData.selectedTests.length > 20) {
                MessageBox.error("Maximum 20 failed tests can be included in a single defect for security reasons");
                return;
            }

            // Validate defect information
            const oTitleValidation = this._validateInput(oData.defectInfo.title, "defectTitle");
            if (!oTitleValidation.isValid) {
                MessageBox.error(`Invalid defect title: ${ oTitleValidation.message}`);
                return;
            }

            if (!oData.defectInfo.description) {
                MessageBox.error("Please provide defect description");
                return;
            }

            oDialog.setBusy(true);

            const oSanitizedData = {
                taskId: oData.taskId,
                failedTests: oData.selectedTests.slice(0, 20), // Security limit
                defectInfo: {
                    title: oTitleValidation.sanitized,
                    description: this.formatSecureText(oData.defectInfo.description.substring(0, 2000)),
                    severity: oData.defectInfo.severity,
                    priority: oData.defectInfo.priority,
                    assignee: this.formatSecureText(oData.defectInfo.assignee),
                    component: this.formatSecureText(oData.defectInfo.component),
                    labels: (oData.defectInfo.labels || []).slice(0, 10).map((label) => {
                        return this.formatSecureText(label);
                    })
                }
            };

            this._secureAjax({
                url: "/a2a/agent5/v1/defects",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oSanitizedData)
            }).then((data) => {
                oDialog.setBusy(false);
                oDialog.close();

                MessageBox.success(
                    "Defect created successfully!\n" +
                    `Defect ID: ${ this.formatSecureText(data.defectId) }\n` +
                    `Tracking URL: ${ this.formatSecureText(data.trackingUrl)}`
                );

                this._logAuditEvent("DEFECT_CREATED", "Defect created", {
                    defectId: data.defectId,
                    testCount: oData.selectedTests.length
                });
            }).catch((xhr) => {
                oDialog.setBusy(false);
                MessageBox.error(`Defect creation failed: ${ this.formatSecureText(xhr.responseText)}`);
                this._logAuditEvent("DEFECT_CREATION_ERROR", "Defect creation failed", xhr.responseText);
            });
        },

        /**
         * @function onScheduleRegression
         * @description Opens dialog to schedule regression test execution.
         * @public
         */
        onScheduleRegression() {
            if (!this._checkPermission("SCHEDULE_REGRESSION")) {
                MessageBox.error("Insufficient permissions to schedule regression tests");
                return;
            }

            const oContext = this._extensionAPI.getBindingContext();
            const sTaskId = oContext.getProperty("ID");

            const oTaskIdValidation = this._validateInput(sTaskId, "taskId");
            if (!oTaskIdValidation.isValid) {
                MessageBox.error("Invalid task ID");
                return;
            }

            if (!this._oRegressionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.ScheduleRegression",
                    controller: this
                }).then((oDialog) => {
                    this._oRegressionDialog = oDialog;
                    this.base.getView().addDependent(this._oRegressionDialog);

                    const oModel = new JSONModel({
                        taskId: oTaskIdValidation.sanitized,
                        scheduleType: "IMMEDIATE",
                        cronExpression: "",
                        selectedDate: new Date(),
                        selectedTime: "00:00",
                        includeNewTests: true,
                        onlyFailedTests: false,
                        notifications: true,
                        maxExecutionTime: 7200000 // 2 hours max for regression
                    });
                    this._oRegressionDialog.setModel(oModel, "regression");
                    this._oRegressionDialog.open();
                });
            } else {
                this._oRegressionDialog.open();
            }
        },

        onScheduleRegressionExecution() {
            const oDialog = this._dialogCache.regression;
            const oModel = oDialog.getModel("regression");
            const oData = oModel.getData();

            oDialog.setBusy(true);

            const oRequestData = {
                scheduleType: oData.scheduleType,
                cronExpression: oData.scheduleType === "CRON" ? this.formatSecureText(oData.cronExpression) : "",
                scheduledDateTime: oData.scheduleType === "SCHEDULED" ?
                    new Date(`${oData.selectedDate }T${ oData.selectedTime}`).toISOString() : null,
                includeNewTests: Boolean(oData.includeNewTests),
                onlyFailedTests: Boolean(oData.onlyFailedTests),
                notifications: Boolean(oData.notifications),
                maxExecutionTime: Math.min(oData.maxExecutionTime, 7200000), // Security limit
                securityMode: "STRICT"
            };

            this._secureAjax({
                url: `/a2a/agent5/v1/tasks/${ encodeURIComponent(oData.taskId) }/schedule-regression`,
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oRequestData)
            }).then((data) => {
                oDialog.setBusy(false);
                oDialog.close();

                MessageBox.success(
                    "Regression test scheduled successfully!\n" +
                    `Schedule ID: ${ this.formatSecureText(data.scheduleId) }\n` +
                    `Next execution: ${ this.formatSecureText(data.nextExecution)}`
                );

                this._logAuditEvent("REGRESSION_SCHEDULED", "Regression test scheduled", {
                    scheduleId: data.scheduleId,
                    taskId: oData.taskId
                });
            }).catch((xhr) => {
                oDialog.setBusy(false);
                MessageBox.error(`Scheduling failed: ${ this.formatSecureText(xhr.responseText)}`);
                this._logAuditEvent("REGRESSION_SCHEDULE_ERROR", "Regression scheduling failed", xhr.responseText);
            });
        }
    });
});