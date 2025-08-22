sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent5.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                // Initialize debounced validation for performance
                this._debouncedValidation = this._debounce(this._validateTestData.bind(this), 300);
            },

            onExit: function() {
                // Cleanup resources to prevent memory leaks
                if (this._oCreateDialog) {
                    this._oCreateDialog.destroy();
                }
                if (this._oTestRunnerDialog) {
                    this._oTestRunnerDialog.destroy();
                }
                if (this._oDefectTrackingDialog) {
                    this._oDefectTrackingDialog.destroy();
                }
                if (this._monitorInterval) {
                    clearInterval(this._monitorInterval);
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
                case "testSuite":
                    if (sSanitized.length > 100 || !/^[a-zA-Z0-9_\-\s]+$/.test(sSanitized)) {
                        return { isValid: false, message: "Invalid test suite name format" };
                    }
                    break;
                case "testCase":
                    if (sSanitized.length > 500) {
                        return { isValid: false, message: "Test case description too long" };
                    }
                    // Check for dangerous test patterns
                    var aDangerousPatterns = [/system\s*\(/i, /exec\s*\(/i, /import\s+os/i, /subprocess/i];
                    for (var j = 0; j < aDangerousPatterns.length; j++) {
                        if (aDangerousPatterns[j].test(sSanitized)) {
                            return { isValid: false, message: "Potentially dangerous test pattern" };
                        }
                    }
                    break;
                case "taskName":
                    if (sSanitized.length > 200) {
                        return { isValid: false, message: "Task name too long" };
                    }
                    break;
                case "defectDescription":
                    if (sSanitized.length > 2000) {
                        return { isValid: false, message: "Defect description too long" };
                    }
                    break;
            }

            return { isValid: true, sanitized: sSanitized };
        },

        _getCSRFToken: function() {
            return new Promise(function(resolve, reject) {
                jQuery.ajax({
                    url: "/a2a/agent5/v1/csrf-token",
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
                component: "Agent5.ListReport",
                sessionId: this._getSessionId(),
                ipAddress: this._getClientIP(),
                userAgent: navigator.userAgent.substring(0, 200)
            };

            // Enhanced audit service logging
            jQuery.ajax({
                url: "/a2a/common/v1/audit",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oAuditData),
                async: true,
                success: function() {
                    console.log("Audit event logged:", sEventType);
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
            return "client_ip_masked";
        },

        _getCurrentUser: function() {
            return "current_user"; // Placeholder
        },

        _validateTestExecutionSecurity: function(oTestData) {
            // Enhanced test execution security validation
            if (!oTestData || typeof oTestData !== 'object') {
                return { isValid: false, message: "Invalid test data object" };
            }

            // Validate test suite
            if (oTestData.testSuite) {
                var oSuiteValidation = this._validateInput(oTestData.testSuite, "testSuite");
                if (!oSuiteValidation.isValid) {
                    return oSuiteValidation;
                }
            }

            // Check for malicious test patterns
            if (oTestData.testCases && Array.isArray(oTestData.testCases)) {
                for (var i = 0; i < oTestData.testCases.length; i++) {
                    var oTestCase = oTestData.testCases[i];
                    if (oTestCase.testSteps) {
                        var oCaseValidation = this._validateInput(JSON.stringify(oTestCase.testSteps), "testCase");
                        if (!oCaseValidation.isValid) {
                            this._logAuditEvent("MALICIOUS_TEST", "Blocked malicious test case", oTestCase);
                            return { isValid: false, message: "Test case contains invalid patterns" };
                        }
                    }
                }
            }

            return { isValid: true };
        },

        _checkPermission: function(sAction) {
            // Role-based permission check
            var aUserRoles = this._getUserRoles();
            var mRequiredPermissions = {
                "CREATE_QA_TASK": ["QA_ADMIN", "QA_USER"],
                "EXECUTE_TESTS": ["QA_ADMIN", "QA_USER"],
                "GENERATE_REPORTS": ["QA_ADMIN", "QA_MANAGER"],
                "MANAGE_DEFECTS": ["QA_ADMIN", "QA_USER"],
                "BATCH_EXECUTION": ["QA_ADMIN"]
            };

            var aRequiredRoles = mRequiredPermissions[sAction] || [];
            return aRequiredRoles.some(function(sRole) {
                return aUserRoles.includes(sRole);
            });
        },

        _getUserRoles: function() {
            return ["QA_USER"]; // Placeholder
        },

        formatSecureText: function(sText) {
            if (!sText) return "";
            return jQuery.sap.encodeXML(String(sText));
        },

        formatTestResult: function(sResult) {
            if (!sResult) return "Unknown";
            var sCleanResult = this.formatSecureText(sResult);
            return sCleanResult === "PASS" ? "✓ Pass" : 
                   sCleanResult === "FAIL" ? "✗ Fail" : 
                   sCleanResult === "SKIP" ? "⚠ Skip" : sCleanResult;
        },

        onCreateQATask: function() {
            // Permission check
            if (!this._checkPermission("CREATE_QA_TASK")) {
                MessageBox.error("Insufficient permissions to create QA tasks");
                return;
            }

            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent5.ext.fragment.CreateQATask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model with secure defaults
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        testSuite: "",
                        testType: "UNIT",
                        targetApplication: "",
                        testFramework: "SELENIUM",
                        testEnvironment: "TEST",
                        severity: "MEDIUM",
                        automationLevel: 80,
                        parallelExecution: true,
                        retryOnFailure: true,
                        complianceStandard: "",
                        maxExecutionTime: 3600000, // 1 hour max
                        testCases: []
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                    
                    this._logAuditEvent("DIALOG_OPENED", "Create QA task dialog opened");
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onTestRunner: function() {
            // Permission check
            if (!this._checkPermission("EXECUTE_TESTS")) {
                MessageBox.error("Insufficient permissions to run tests");
                return;
            }

            var oView = this.base.getView();
            
            if (!this._oTestRunnerDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent5.ext.fragment.TestRunner",
                    controller: this
                }).then(function(oDialog) {
                    this._oTestRunnerDialog = oDialog;
                    oView.addDependent(this._oTestRunnerDialog);
                    
                    // Initialize test runner model with security limits
                    var oModel = new JSONModel({
                        selectedSuite: "",
                        executionMode: "SEQUENTIAL",
                        environment: "TEST",
                        parallelThreads: Math.min(4, 8), // Limit parallel threads
                        retryAttempts: Math.min(3, 5), // Limit retry attempts
                        generateReport: true,
                        notifyOnCompletion: true,
                        maxExecutionTime: 7200000, // 2 hours max
                        testFilters: {
                            includeSmoke: true,
                            includeRegression: false,
                            priorityFilter: "ALL"
                        },
                        executionResults: []
                    });
                    this._oTestRunnerDialog.setModel(oModel, "runner");
                    this._oTestRunnerDialog.open();
                    
                    // Load available test suites securely
                    this._loadTestSuites();
                }.bind(this));
            } else {
                this._oTestRunnerDialog.open();
                this._loadTestSuites();
            }
        },

        _loadTestSuites: function() {
            this._secureAjax({
                url: "/a2a/agent5/v1/test-suites",
                type: "GET"
            }).then(function(data) {
                var oRunnerModel = this._oTestRunnerDialog.getModel("runner");
                // Sanitize test suite data
                var aSanitizedSuites = this._sanitizeTestSuiteData(data.suites || []);
                oRunnerModel.setProperty("/availableSuites", aSanitizedSuites);
            }.bind(this)).catch(function(xhr) {
                MessageBox.error("Failed to load test suites: " + this.formatSecureText(xhr.responseText));
            }.bind(this));
        },

        _sanitizeTestSuiteData: function(aSuites) {
            return aSuites.slice(0, 50).map(function(suite) { // Limit to 50 suites
                return {
                    id: this.formatSecureText(suite.id),
                    name: this.formatSecureText(suite.name),
                    description: this.formatSecureText(suite.description),
                    testCount: typeof suite.testCount === 'number' ? Math.min(suite.testCount, 10000) : 0,
                    estimatedTime: typeof suite.estimatedTime === 'number' ? Math.min(suite.estimatedTime, 86400) : 0
                };
            }.bind(this));
        },

        onExecuteTestSuite: function() {
            var oModel = this._oTestRunnerDialog.getModel("runner");
            var oData = oModel.getData();
            
            // Comprehensive validation
            if (!oData.selectedSuite) {
                MessageBox.error("Please select a test suite to execute");
                return;
            }

            var oSuiteValidation = this._validateInput(oData.selectedSuite, "testSuite");
            if (!oSuiteValidation.isValid) {
                MessageBox.error("Invalid test suite: " + oSuiteValidation.message);
                return;
            }
            
            this._oTestRunnerDialog.setBusy(true);
            
            var oRequestData = {
                suiteId: oSuiteValidation.sanitized,
                executionMode: oData.executionMode,
                environment: oData.environment,
                parallelThreads: Math.min(oData.parallelThreads, 8), // Security limit
                retryAttempts: Math.min(oData.retryAttempts, 5), // Security limit
                maxExecutionTime: Math.min(oData.maxExecutionTime, 7200000), // 2 hours max
                filters: oData.testFilters,
                securityMode: "STRICT"
            };

            this._secureAjax({
                url: "/a2a/agent5/v1/execute-test-suite",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oRequestData)
            }).then(function(data) {
                this._oTestRunnerDialog.setBusy(false);
                
                MessageBox.success(
                    "Test execution started!\n" +
                    "Execution ID: " + this.formatSecureText(data.executionId) + "\n" +
                    "Estimated time: " + this.formatSecureText(data.estimatedTime) + " minutes"
                );
                
                // Start secure monitoring execution
                this._monitorTestExecution(data.executionId);
                this._logAuditEvent("TEST_EXECUTION_STARTED", "Test suite execution started", {
                    suiteId: oData.selectedSuite,
                    executionId: data.executionId
                });
            }.bind(this)).catch(function(xhr) {
                this._oTestRunnerDialog.setBusy(false);
                MessageBox.error("Test execution failed: " + this.formatSecureText(xhr.responseText));
                this._logAuditEvent("TEST_EXECUTION_ERROR", "Test execution failed", xhr.responseText);
            }.bind(this));
        },

        _monitorTestExecution: function(executionId) {
            // Secure polling with rate limiting
            var nPollCount = 0;
            var nMaxPolls = 360; // Max 30 minutes of polling (5s intervals)
            
            this._monitorInterval = setInterval(function() {
                nPollCount++;
                
                if (nPollCount > nMaxPolls) {
                    clearInterval(this._monitorInterval);
                    MessageBox.warning("Execution monitoring timeout. Please check status manually.");
                    return;
                }
                
                this._secureAjax({
                    url: "/a2a/agent5/v1/execution-status/" + encodeURIComponent(executionId),
                    type: "GET"
                }).then(function(data) {
                    var oModel = this._oTestRunnerDialog.getModel("runner");
                    // Validate and sanitize execution results
                    var oValidatedResults = this._validateExecutionResults(data);
                    if (oValidatedResults.isValid) {
                        oModel.setProperty("/executionResults", oValidatedResults.sanitized.results);
                        
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(this._monitorInterval);
                            
                            if (data.status === "COMPLETED") {
                                MessageBox.success(
                                    "Test execution completed!\n" +
                                    "Passed: " + this.formatSecureText(data.passed) + "\n" +
                                    "Failed: " + this.formatSecureText(data.failed) + "\n" +
                                    "Success Rate: " + this.formatSecureText(data.successRate) + "%"
                                );
                                this._logAuditEvent("TEST_EXECUTION_COMPLETED", "Test execution completed", data);
                            } else {
                                MessageBox.error("Test execution failed: " + this.formatSecureText(data.error));
                                this._logAuditEvent("TEST_EXECUTION_FAILED", "Test execution failed", data.error);
                            }
                            
                            this._extensionAPI.refresh();
                        }
                    }
                }.bind(this)).catch(function() {
                    // Silent fail on poll errors to avoid spam
                    if (nPollCount > 10) {
                        clearInterval(this._monitorInterval);
                    }
                }.bind(this));
            }.bind(this), 5000);
        },

        _validateExecutionResults: function(oData) {
            if (!oData || typeof oData !== 'object') {
                return { isValid: false };
            }
            
            // Validate status
            var aValidStatuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "PAUSED"];
            if (!aValidStatuses.includes(oData.status)) {
                return { isValid: false };
            }
            
            // Sanitize results
            var oSanitized = {
                status: this.formatSecureText(oData.status),
                results: (oData.results || []).slice(0, 1000).map(function(result) {
                    return {
                        testName: this.formatSecureText(result.testName),
                        result: this.formatSecureText(result.result),
                        duration: typeof result.duration === 'number' ? Math.min(result.duration, 3600000) : 0
                    };
                }.bind(this))
            };
            
            return { isValid: true, sanitized: oSanitized };
        },

        onQualityDashboard: function() {
            // Navigate to quality dashboard with audit logging
            this._logAuditEvent("NAVIGATION", "Navigated to quality dashboard");
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("QualityDashboard");
        },

        onBatchExecution: function() {
            // Permission check
            if (!this._checkPermission("BATCH_EXECUTION")) {
                MessageBox.error("Insufficient permissions for batch execution");
                return;
            }

            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch execution.");
                return;
            }

            // Validate selection limit for security
            if (aSelectedContexts.length > 50) {
                MessageBox.error("Maximum 50 tasks can be executed in batch for security reasons.");
                return;
            }
            
            MessageBox.confirm(
                "Execute tests for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchExecution(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchExecution: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return this.formatSecureText(oContext.getProperty("ID"));
            }.bind(this));
            
            this.base.getView().setBusy(true);
            
            var oRequestData = {
                taskIds: aTaskIds,
                executionMode: "PARALLEL",
                priority: "HIGH",
                generateConsolidatedReport: true,
                maxExecutionTime: 14400000, // 4 hours max for batch
                securityMode: "STRICT"
            };

            this._secureAjax({
                url: "/a2a/agent5/v1/batch-execute",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oRequestData)
            }).then(function(data) {
                this.base.getView().setBusy(false);
                MessageBox.success(
                    "Batch execution started!\n" +
                    "Job ID: " + this.formatSecureText(data.jobId) + "\n" +
                    "Total test cases: " + this.formatSecureText(data.totalTestCases)
                );
                this._extensionAPI.refresh();
                this._logAuditEvent("BATCH_EXECUTION", "Batch execution started", { taskCount: aTaskIds.length });
            }.bind(this)).catch(function(xhr) {
                this.base.getView().setBusy(false);
                MessageBox.error("Batch execution failed: " + this.formatSecureText(xhr.responseText));
                this._logAuditEvent("BATCH_EXECUTION_ERROR", "Batch execution failed", xhr.responseText);
            }.bind(this));
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
            
            if (!oData.testType || !oData.targetApplication) {
                MessageBox.error("Please fill all required fields");
                return;
            }

            // Validate test execution security
            var oSecurityValidation = this._validateTestExecutionSecurity(oData);
            if (!oSecurityValidation.isValid) {
                MessageBox.error("Security validation failed: " + oSecurityValidation.message);
                return;
            }
            
            if (oData.testCases.length === 0) {
                MessageBox.warning("No test cases defined. Task will be created but cannot be executed until test cases are added.");
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Prepare sanitized data
            var oSanitizedData = {
                taskName: oTaskNameValidation.sanitized,
                description: oDescValidation.sanitized,
                testSuite: this.formatSecureText(oData.testSuite),
                testType: oData.testType,
                targetApplication: this.formatSecureText(oData.targetApplication),
                testFramework: oData.testFramework,
                testEnvironment: oData.testEnvironment,
                severity: oData.severity,
                automationLevel: Math.min(Math.max(oData.automationLevel, 0), 100),
                parallelExecution: Boolean(oData.parallelExecution),
                retryOnFailure: Boolean(oData.retryOnFailure),
                complianceStandard: this.formatSecureText(oData.complianceStandard),
                maxExecutionTime: Math.min(oData.maxExecutionTime, 3600000),
                testCases: oData.testCases.slice(0, 100).map(function(testCase) { // Limit to 100 test cases
                    return {
                        testName: this.formatSecureText(testCase.testName),
                        description: this.formatSecureText(testCase.description),
                        category: testCase.category,
                        priority: testCase.priority,
                        automationLevel: testCase.automationLevel,
                        expectedResult: this.formatSecureText(testCase.expectedResult)
                    };
                }.bind(this))
            };
            
            this._secureAjax({
                url: "/a2a/agent5/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oSanitizedData)
            }).then(function(data) {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show("QA validation task created successfully");
                this._extensionAPI.refresh();
                this._logAuditEvent("TASK_CREATED", "QA validation task created", { taskName: oSanitizedData.taskName });
            }.bind(this)).catch(function(xhr) {
                this._oCreateDialog.setBusy(false);
                MessageBox.error("Failed to create task: " + this.formatSecureText(xhr.responseText));
                this._logAuditEvent("TASK_CREATE_ERROR", "Failed to create task", xhr.responseText);
            }.bind(this));
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },

        onAddTestCase: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var aTestCases = oModel.getProperty("/testCases");
            
            // Limit number of test cases for security
            if (aTestCases.length >= 100) {
                MessageBox.error("Maximum 100 test cases allowed per task for security reasons");
                return;
            }
            
            aTestCases.push({
                testName: "",
                description: "",
                category: "FUNCTIONAL",
                priority: "MEDIUM",
                automationLevel: "MANUAL",
                expectedResult: "",
                testSteps: []
            });
            
            oModel.setProperty("/testCases", aTestCases);
        },

        _validateTestData: function() {
            // Debounced test data validation for performance
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (oData.testCases && oData.testCases.length > 0) {
                var oSecurityValidation = this._validateTestExecutionSecurity(oData);
                if (!oSecurityValidation.isValid) {
                    MessageToast.show("Test data validation warning: " + oSecurityValidation.message);
                }
            }
        },

        onImportTestCases: function() {
            // Show file upload dialog for importing test cases with security validation
            MessageBox.information("Test case import functionality will be available soon with enhanced security validation.");
        }
    });
});