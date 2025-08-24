/**
 * @fileoverview Frontend Trace Client for UI5 Applications
 * @description Integrates frontend actions with backend trace system
 * @module traceClient
 */

sap.ui.define([
    "sap/base/Log"
], (Log) => {
    "use strict";

    /**
     * TraceClient - Frontend trace integration
     */
    const TraceClient = {

        /**
         * Initialize trace client
         */
        init() {
            this._traceId = null;
            this._sessionId = this._generateSessionId();
            this._actionQueue = [];
            this._isInitialized = true;

            // Set up global error handler
            this._setupGlobalErrorHandler();

            Log.info("TraceClient initialized", { sessionId: this._sessionId });
        },

        /**
         * Set the current trace ID from backend response
         */
        setTraceId(traceId) {
            this._traceId = traceId;
            this._flushActionQueue();
        },

        /**
         * Add trace headers to all requests
         */
        addTraceHeaders(requestConfig) {
            if (!requestConfig.headers) {
                requestConfig.headers = {};
            }

            // Add frontend context headers
            requestConfig.headers["X-UI-Component"] = this._getCurrentComponent();
            requestConfig.headers["X-UI-View"] = this._getCurrentView();
            requestConfig.headers["X-UI-Controller"] = this._getCurrentController();
            requestConfig.headers["X-UI-Action"] = this._getCurrentAction();
            requestConfig.headers["X-UI5-Version"] = sap.ui.version;
            requestConfig.headers["X-FLP-Version"] = this._getFLPVersion();
            requestConfig.headers["X-Session-ID"] = this._sessionId;

            if (this._traceId) {
                requestConfig.headers["X-Trace-ID"] = this._traceId;
            }

            return requestConfig;
        },

        /**
         * Track user action
         */
        trackAction(action, component, data = {}) {
            const actionInfo = {
                timestamp: new Date().toISOString(),
                action,
                component: component || this._getCurrentComponent(),
                view: this._getCurrentView(),
                controller: this._getCurrentController(),
                data,
                sessionId: this._sessionId
            };

            if (this._traceId) {
                this._sendActionToBackend(actionInfo);
            } else {
                this._actionQueue.push(actionInfo);
            }

            Log.debug("Action tracked", actionInfo);
        },

        /**
         * Track navigation
         */
        trackNavigation(fromRoute, toRoute, params = {}) {
            this.trackAction("navigation", "router", {
                from: fromRoute,
                to: toRoute,
                params
            });
        },

        /**
         * Track form submission
         */
        trackFormSubmission(formName, formData, validationResults = {}) {
            this.trackAction("form_submission", "form", {
                formName,
                fieldCount: Object.keys(formData).length,
                validationErrors: Object.keys(validationResults).length,
                hasErrors: Object.keys(validationResults).length > 0
            });
        },

        /**
         * Track business operation
         */
        trackBusinessOperation(operation, entity, operationType = "unknown") {
            this.trackAction("business_operation", "service", {
                operation,
                entity,
                operationType
            });
        },

        /**
         * Track error from frontend
         */
        trackError(error, context = {}) {
            const errorInfo = {
                timestamp: new Date().toISOString(),
                message: error.message || String(error),
                stack: error.stack,
                component: context.component || this._getCurrentComponent(),
                view: this._getCurrentView(),
                controller: this._getCurrentController(),
                action: context.action,
                severity: context.severity || "error",
                userAgent: navigator.userAgent,
                url: window.location.href,
                sessionId: this._sessionId
            };

            // Send to backend if trace is active
            if (this._traceId) {
                this._sendErrorToBackend(errorInfo);
            }

            // Log locally
            Log.error("Frontend error tracked", errorInfo);
        },

        /**
         * Track performance metrics
         */
        trackPerformance(operation, duration, component) {
            this.trackAction("performance_metric", component, {
                operation,
                duration,
                timestamp: new Date().toISOString()
            });
        },

        /**
         * Get current component name
         */
        _getCurrentComponent() {
            try {
                const router = sap.ui.core.UIComponent.getRouterFor(this);
                if (router && router.getHashChanger()) {
                    return router.getHashChanger().getHash().split("/")[0] || "unknown";
                }
                return "a2a-network";
            } catch (e) {
                return "a2a-network";
            }
        },

        /**
         * Get current view name
         */
        _getCurrentView() {
            try {
                const core = sap.ui.getCore();
                const currentView = core.byId(core.getCurrentFocusedControlId())?.getParent();
                if (currentView && currentView.getViewName) {
                    return currentView.getViewName();
                }
                return window.location.hash.split("/").pop() || "unknown";
            } catch (e) {
                return window.location.hash.split("/").pop() || "unknown";
            }
        },

        /**
         * Get current controller name
         */
        _getCurrentController() {
            try {
                const view = this._getCurrentView();
                return view.replace(".view.", ".controller.");
            } catch (e) {
                return "unknown";
            }
        },

        /**
         * Get current action context
         */
        _getCurrentAction() {
            // This would be set by individual controllers when performing actions
            return this._currentAction || "unknown";
        },

        /**
         * Set current action context
         */
        setCurrentAction(action) {
            this._currentAction = action;
        },

        /**
         * Get Fiori Launchpad version
         */
        _getFLPVersion() {
            try {
                if (sap.ushell && sap.ushell.Container) {
                    return sap.ushell.Container.getService("LaunchPage").getGroupsMetadata()?.version || "unknown";
                }
                return "standalone";
            } catch (e) {
                return "unknown";
            }
        },

        /**
         * Generate session ID
         */
        _generateSessionId() {
            return `ui5-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Send action to backend
         */
        _sendActionToBackend(actionInfo) {
            if (!this._traceId) {
                return;
            }

            jQuery.ajax({
                url: "/api/v1/trace/action",
                type: "POST",
                data: JSON.stringify({
                    traceId: this._traceId,
                    action: actionInfo
                }),
                contentType: "application/json",
                headers: {
                    "X-Trace-ID": this._traceId
                }
            }).fail((error) => {
                Log.warning("Failed to send action to backend", { error });
            });
        },

        /**
         * Send error to backend
         */
        _sendErrorToBackend(errorInfo) {
            if (!this._traceId) {
                return;
            }

            jQuery.ajax({
                url: "/api/v1/trace/error",
                type: "POST",
                data: JSON.stringify({
                    traceId: this._traceId,
                    error: errorInfo
                }),
                contentType: "application/json",
                headers: {
                    "X-Trace-ID": this._traceId
                }
            }).fail((error) => {
                Log.warning("Failed to send error to backend", { error });
            });
        },

        /**
         * Flush queued actions when trace ID becomes available
         */
        _flushActionQueue() {
            if (this._actionQueue.length > 0 && this._traceId) {
                this._actionQueue.forEach(action => {
                    this._sendActionToBackend(action);
                });
                this._actionQueue = [];
            }
        },

        /**
         * Set up global error handler
         */
        _setupGlobalErrorHandler() {
            const self = this;

            // Handle uncaught JavaScript errors
            window.addEventListener("error", (event) => {
                self.trackError(event.error || new Error(event.message), {
                    component: "global",
                    action: "uncaught_error",
                    severity: "critical"
                });
            });

            // Handle unhandled promise rejections
            window.addEventListener("unhandledrejection", (event) => {
                self.trackError(event.reason || new Error("Unhandled promise rejection"), {
                    component: "global",
                    action: "unhandled_rejection",
                    severity: "error"
                });
            });

            // Integrate with UI5 error handling
            if (sap.ui.getCore) {
                sap.ui.getCore().attachEvent("parseError", (oEvent) => {
                    self.trackError(new Error(oEvent.getParameter("message")), {
                        component: "ui5",
                        action: "parse_error",
                        severity: "error"
                    });
                });
            }
        },

        /**
         * Create enhanced AJAX wrapper with tracing
         */
        createTracedAjax() {
            const self = this;

            return function(options) {
                // Add trace headers
                options = self.addTraceHeaders(options);

                // Track the request
                const startTime = performance.now();
                self.trackAction("ajax_request", "network", {
                    url: options.url,
                    method: options.type || "GET"
                });

                const originalSuccess = options.success;
                const originalError = options.error;

                options.success = function(data, textStatus, jqXHR) {
                    const duration = performance.now() - startTime;

                    // Extract trace ID from response headers
                    const responseTraceId = jqXHR.getResponseHeader("X-Trace-ID");
                    if (responseTraceId && !self._traceId) {
                        self.setTraceId(responseTraceId);
                    }

                    self.trackPerformance("ajax_success", duration, "network");

                    if (originalSuccess) {
                        originalSuccess.apply(this, arguments);
                    }
                };

                options.error = function(jqXHR, textStatus, errorThrown) {
                    const duration = performance.now() - startTime;

                    self.trackError(new Error(errorThrown || textStatus), {
                        component: "network",
                        action: "ajax_error",
                        severity: "error"
                    });

                    self.trackPerformance("ajax_error", duration, "network");

                    if (originalError) {
                        originalError.apply(this, arguments);
                    }
                };

                return jQuery.ajax(options);
            };
        }
    };

    return TraceClient;
});