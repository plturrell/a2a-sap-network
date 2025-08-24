
// Interval tracking for memory leak prevention
const intervals = [];

function trackInterval(intervalId) {
    intervals.push(intervalId);
    return intervalId;
}

function cleanupIntervals() {
    intervals.forEach(clearInterval);
    intervals.length = 0;
}

// Cleanup on exit
if (typeof process !== "undefined") {
    process.on("exit", cleanupIntervals);
    process.on("SIGTERM", cleanupIntervals);
    process.on("SIGINT", cleanupIntervals);
}

sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/f/FlexibleColumnLayoutSemanticHelper",
    "sap/f/library",
    "sap/ui/core/library",
    "sap/base/Log",
    "sap/base/util/UriParameters"
], (
    UIComponent,
    Device,
    JSONModel,
    MessageBox,
    MessageToast,
    FlexibleColumnLayoutSemanticHelper,
    fioriLibrary,
    coreLibrary,
    Log,
    UriParameters
) => {
    "use strict";

    const LayoutType = fioriLibrary.LayoutType;

    return UIComponent.extend("a2a.network.fiori.Component", {
        metadata: {
            manifest: "json",
            interfaces: ["sap.ui.core.IAsyncContentCreation"]
        },

        _oSocket: null,
        _oFlexibleColumnLayoutSemanticHelper: null,

        /**
         * The component is initialized by UI5 automatically during the startup of the app
         * and calls the init method once.
         * @public
         * @override
         */
        init() {
            // call the base component"s init function
            UIComponent.prototype.init.apply(this, arguments);

            // Initialize global error handling
            this._initializeErrorHandling();

            // create the device model
            this.setModel(this._createDeviceModel(), "device");

            // create the app model
            this.setModel(this._createAppModel(), "app");

            // enable routing
            this.getRouter().initialize();

            // initialize services
            this._initServices();

            // Initialize client-side monitoring
            this._initializeClientMonitoring();

            Log.info("A2A Network Fiori Component initialized");
        },

        /**
         * The component is destroyed by UI5 automatically.
         * @public
         * @override
         */
        destroy() {
            // disconnect WebSocket
            if (this._oSocket) {
                this._oSocket.disconnect();
                this._oSocket = null;
            }

            // call the base component"s destroy function
            UIComponent.prototype.destroy.apply(this, arguments);
        },

        /**
         * This method can be called to determine whether the sapUiSizeCompact or sapUiSizeCozy
         * design mode class should be set, which influences the size appearance of some controls.
         * @public
         * @returns {string} css class, either "sapUiSizeCompact" or "sapUiSizeCozy" -
         * or an empty string if no css class should be set
         */
        getContentDensityClass() {
            if (this._sContentDensityClass === undefined) {
                // check whether FLP has already set the content density class; do nothing in this case
                if (document.body.classList.contains("sapUiSizeCozy") ||
                    document.body.classList.contains("sapUiSizeCompact")) {
                    this._sContentDensityClass = "";
                } else if (!Device.support.touch) { // apply "compact" mode if touch is not supported
                    this._sContentDensityClass = "sapUiSizeCompact";
                } else {
                    // "cozy" in case of touch support; default for most sap.m controls,
                    // but needed for desktop-first controls like sap.ui.table.Table
                    this._sContentDensityClass = "sapUiSizeCozy";
                }
            }
            return this._sContentDensityClass;
        },

        /**
         * Returns the flexible column layout semantic helper.
         * @public
         * @returns {sap.f.FlexibleColumnLayoutSemanticHelper} the semantic helper
         */
        getHelper() {
            const oFCL = this.getRootControl().byId("fcl"),
                oParams = UriParameters.fromQuery(location.search),
                oSettings = {
                    defaultTwoColumnLayoutType: LayoutType.TwoColumnsMidExpanded,
                    defaultThreeColumnLayoutType: LayoutType.ThreeColumnsMidExpanded,
                    mode: oParams.get("mode"),
                    maxColumnsCount: oParams.get("max")
                };

            if (!this._oFlexibleColumnLayoutSemanticHelper) {
                this._oFlexibleColumnLayoutSemanticHelper =
                    FlexibleColumnLayoutSemanticHelper.getInstanceFor(oFCL, oSettings);
            }
            return this._oFlexibleColumnLayoutSemanticHelper;
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Creates the device model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the device model
         */
        _createDeviceModel() {
            const oModel = new JSONModel(Device);
            oModel.setDefaultBindingMode("OneWay");
            return oModel;
        },

        /**
         * Creates the app model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the app model
         */
        _createAppModel() {
            const oModel = new JSONModel({
                busy: false,
                delay: 0,
                layout: LayoutType.OneColumn,
                previousLayout: "",
                actionButtonsInfo: {
                    midColumn: {
                        fullScreen: false
                    }
                },
                networkStatus: "unknown",
                currentUser: null,
                environment: this._getEnvironmentInfo(),
                stats: {
                    totalAgents: 0,
                    activeAgents: 0,
                    totalServices: 0,
                    networkLoad: 0,
                    avgResponseTime: 0,
                    successRate: 0
                }
            });
            oModel.setDefaultBindingMode("TwoWay");
            return oModel;
        },

        /**
         * Initializes services.
         * @private
         */
        _initServices() {
            // Initialize WebSocket connection
            this._initWebSocket();

            // Load initial data
            this._loadNetworkStats();

            // Initialize user authentication
            this._initUserAuthentication();

            // Set up periodic refresh
            trackInterval(setInterval(() => {
                this._loadNetworkStats();
            }, 60000)); // Refresh every minute
        },

        /**
         * Initializes WebSocket connection.
         * @private
         */
        _initWebSocket() {
            // Check if Socket.IO is available
            if (typeof io === "undefined") {
                Log.warning("Socket.IO not available - real-time updates disabled");
                return;
            }

            try {
                this._oSocket = io({
                    reconnection: true,
                    reconnectionDelay: 1000,
                    reconnectionDelayMax: 5000,
                    reconnectionAttempts: 5
                });

                this._oSocket.on("connect", () => {
                    Log.info("WebSocket connected");
                    this.getModel("app").setProperty("/networkStatus", "connected");

                    // Subscribe to relevant events
                    this._oSocket.emit("subscribe", {
                        rooms: ["agents", "services", "workflows", "reputation"]
                    });
                });

                this._oSocket.on("disconnect", () => {
                    Log.warning("WebSocket disconnected");
                    this.getModel("app").setProperty("/networkStatus", "disconnected");
                });

                this._oSocket.on("error", (error) => {
                    Log.error("WebSocket error", error);
                    this.getModel("app").setProperty("/networkStatus", "error");
                });

                // Real-time event handlers
                this._attachSocketEventHandlers();

            } catch (error) {
                Log.error("Failed to initialize WebSocket", error);
            }
        },

        /**
         * Attaches Socket.IO event handlers.
         * @private
         */
        _attachSocketEventHandlers() {
            if (!this._oSocket) {
                return;
            }

            // Agent events
            this._oSocket.on("agent:registered", (data) => {
                MessageToast.show(this._getResourceBundle().getText("agentRegisteredMessage", [data.name]));
                this._loadNetworkStats();
                this.getEventBus().publish("app", "agentRegistered", data);
            });

            this._oSocket.on("agent:updated", (data) => {
                this.getEventBus().publish("app", "agentUpdated", data);
            });

            // Service events
            this._oSocket.on("service:created", (data) => {
                MessageToast.show(this._getResourceBundle().getText("serviceCreatedMessage", [data.name]));
                this.getEventBus().publish("app", "serviceCreated", data);
            });

            // Reputation events
            this._oSocket.on("reputation:updated", (data) => {
                this.getEventBus().publish("app", "reputationUpdated", data);

                // Update UI if viewing the affected agent
                const sCurrentHash = this.getRouter().getHashChanger().getHash();
                if (sCurrentHash.includes(data.agentId)) {
                    this.getModel().refresh();
                }
            });

            // Workflow events
            this._oSocket.on("workflow:completed", (data) => {
                MessageToast.show(this._getResourceBundle().getText("workflowCompletedMessage", [data.executionId]));
                this.getEventBus().publish("app", "workflowCompleted", data);
            });

            this._oSocket.on("workflow:failed", (data) => {
                MessageBox.error(this._getResourceBundle().getText("workflowFailedMessage", [data.executionId]));
                this.getEventBus().publish("app", "workflowFailed", data);
            });
        },

        /**
         * Loads network statistics.
         * @private
         */
        _loadNetworkStats() {
            const oModel = this.getModel();
            const oAppModel = this.getModel("app");

            if (!oModel || !oModel.read) {
                Log.warning("OData model not available");
                return;
            }

            // Load network stats
            oModel.read("/NetworkStats", {
                urlParameters: {
                    "$orderby": "validFrom desc",
                    "$top": 1
                },
                success(oData) {
                    if (oData.results && oData.results.length > 0) {
                        const oStats = oData.results[0];
                        oAppModel.setProperty("/stats", {
                            totalAgents: oStats.totalAgents || 0,
                            activeAgents: oStats.activeAgents || 0,
                            totalServices: oStats.totalServices || 0,
                            networkLoad: oStats.networkLoad || 0,
                            avgResponseTime: oStats.avgTransactionTime || 0,
                            successRate: oStats.successRate || 0
                        });
                        Log.debug("Network stats loaded", oStats);
                    }
                },
                error(oError) {
                    Log.error("Failed to load network stats", oError);
                }
            });
        },

        /**
         * Initializes user authentication.
         * @private
         */
        _initUserAuthentication() {
            const oAppModel = this.getModel("app");
            const oEnvironment = oAppModel.getProperty("/environment");

            if (oEnvironment.isBTP) {
                // In BTP, user info comes from XSUAA
                this._loadBTPUserInfo();
            } else {
                // Local development mode
                oAppModel.setProperty("/currentUser", {
                    id: "local-user",
                    name: "Developer",
                    email: sap.ushell.Container.getUser().getEmail() || "user@a2a.network",
                    roles: ["Admin", "Developer"]
                });
                Log.info("Running in development mode with local user");
            }
        },

        /**
         * Loads user info from BTP.
         * @private
         */
        _loadBTPUserInfo() {
            const oAppModel = this.getModel("app");

            // Call user info endpoint
            jQuery.ajax({
                url: "/user-api/currentUser",
                type: "GET",
                success(oUserInfo) {
                    oAppModel.setProperty("/currentUser", {
                        id: oUserInfo.id,
                        name: oUserInfo.name || oUserInfo.id,
                        email: oUserInfo.email,
                        roles: oUserInfo.scopes || []
                    });
                    Log.info("User authenticated", { userId: oUserInfo.id });
                },
                error: (oError) => {
                    Log.error("Failed to load user info", oError);
                    MessageBox.error(this._getResourceBundle().getText("authenticationError"));
                }
            });
        },

        /**
         * Gets environment information.
         * @private
         * @returns {object} Environment info
         */
        _getEnvironmentInfo() {
            const sHostname = window.location.hostname;
            const bIsBTP = sHostname.includes("cfapps") ||
                         sHostname.includes("hana.ondemand.com") ||
                         sHostname.includes("cloud.sap");

            return {
                isBTP: bIsBTP,
                isLocal: sHostname === "localhost" || sHostname === "127.0.0.1",
                hostname: sHostname,
                protocol: window.location.protocol,
                port: window.location.port
            };
        },

        /**
         * Gets the resource bundle.
         * @private
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         */
        _getResourceBundle() {
            return this.getModel("i18n").getResourceBundle();
        },

        /* =========================================================== */
        /* Error Handling & Monitoring Methods                        */
        /* =========================================================== */

        /**
         * Initialize global error handling for the application
         * @private
         * @since 1.0.0
         */
        _initializeErrorHandling() {
            const that = this;

            // Capture JavaScript errors
            window.addEventListener("error", (event) => {
                that._reportClientError({
                    message: event.message,
                    filename: event.filename,
                    lineno: event.lineno,
                    colno: event.colno,
                    stack: event.error ? event.error.stack : "No stack trace available",
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    type: "javascript-error"
                });
            });

            // Capture unhandled promise rejections
            window.addEventListener("unhandledrejection", (event) => {
                that._reportClientError({
                    message: event.reason ? event.reason.message || event.reason : "Unhandled promise rejection",
                    stack: event.reason ? event.reason.stack : "No stack trace available",
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    timestamp: new Date().toISOString(),
                    type: "unhandled-promise-rejection",
                    additionalInfo: {
                        reason: event.reason
                    }
                });

                // Prevent browser console error
                event.preventDefault();
            });

            // Capture UI5 errors
            Log.addLogListener({
                onLogEntry(oLog) {
                    if (oLog.level >= 4) { // Error level and above
                        that._reportClientError({
                            message: oLog.message,
                            stack: oLog.details || "UI5 Error",
                            userAgent: navigator.userAgent,
                            url: window.location.href,
                            timestamp: new Date().toISOString(),
                            type: "ui5-error",
                            additionalInfo: {
                                component: oLog.component,
                                level: oLog.level,
                                details: oLog.details
                            }
                        });
                    }
                }
            });

            Log.info("Global error handling initialized");
        },

        /**
         * Initialize client-side monitoring and performance tracking
         * @private
         * @since 1.0.0
         */
        _initializeClientMonitoring() {
            const that = this;

            // Performance monitoring
            if (window.performance && window.performance.mark) {
                // Mark application start
                performance.mark("app-start");

                // Monitor page load performance
                window.addEventListener("load", () => {
                    performance.mark("app-loaded");

                    const navigation = performance.getEntriesByType("navigation")[0];
                    if (navigation) {
                        that._reportPerformanceMetric("page-load", {
                            loadTime: navigation.loadEventEnd - navigation.loadEventStart,
                            domContentLoaded: navigation.domContentLoadedEventEnd -
                                navigation.domContentLoadedEventStart,
                            firstPaint: that._getFirstPaint(),
                            timestamp: new Date().toISOString()
                        });
                    }
                });
            }

            // Monitor route changes
            this.getRouter().attachRouteMatched((oEvent) => {
                const sRouteName = oEvent.getParameter("name");
                performance.mark(`route-${ sRouteName }-start`);

                // Report route change
                that._reportUserAction("route-change", {
                    route: sRouteName,
                    arguments: oEvent.getParameter("arguments"),
                    timestamp: new Date().toISOString()
                });
            });

            // Monitor user interactions
            document.addEventListener("click", (event) => {
                const sTarget = event.target.tagName + (event.target.id ? `#${ event.target.id}` : "") +
                              (event.target.className ? `.${ event.target.className.replace(/\s+/g, ".")}` : "");

                that._reportUserAction("click", {
                    target: sTarget,
                    timestamp: new Date().toISOString()
                });
            }, { passive: true, capture: true });

            Log.info("Client-side monitoring initialized");
        },

        /**
         * Report client-side errors to the server
         * @private
         * @param {object} errorData Error information
         * @since 1.0.0
         */
        _reportClientError(errorData) {
            // Add correlation ID and session info
            const oAppModel = this.getModel("app");
            const enhancedErrorData = Object.assign({}, errorData, {
                correlationId: this._generateCorrelationId(),
                sessionId: oAppModel.getProperty("/sessionId"),
                userId: oAppModel.getProperty("/currentUser/id") || "anonymous",
                environment: oAppModel.getProperty("/environment"),
                buildInfo: oAppModel.getProperty("/buildInfo")
            });

            // Send to server error reporting endpoint
            jQuery.ajax({
                url: "/api/v1/errors/report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(enhancedErrorData),
                success(response) {
                    Log.debug("Error reported successfully", { errorId: response.errorId });
                },
                error(xhr, status, error) {
                    Log.error("Failed to report error to server", {
                        status,
                        error,
                        originalError: errorData.message
                    });
                }
            });

            // Also log locally for development
            Log.error(`Client Error: ${ errorData.message}`, errorData);
        },

        /**
         * Report performance metrics
         * @private
         * @param {string} metric Metric name
         * @param {object} data Metric data
         * @since 1.0.0
         */
        _reportPerformanceMetric(metric, data) {
            Log.info(`Performance Metric: ${ metric}`, data);

            // In production, send to monitoring service
            if (window.location.hostname !== "localhost") {
                // Could integrate with SAP Cloud ALM or other APM solutions
            }
        },

        /**
         * Report user actions for analytics
         * @private
         * @param {string} action Action type
         * @param {object} data Action data
         * @since 1.0.0
         */
        _reportUserAction(action, data) {
            // Throttle reporting to avoid too many events
            if (!this._actionThrottle) {
                this._actionThrottle = {};
            }

            const throttleKey = `${action }-${ data.target || data.route || "unknown"}`;
            const now = Date.now();

            if (this._actionThrottle[throttleKey] && (now - this._actionThrottle[throttleKey] < 1000)) {
                return; // Skip if same action within 1 second
            }

            this._actionThrottle[throttleKey] = now;

            Log.debug(`User Action: ${ action}`, data);
        },

        /**
         * Get first paint time
         * @private
         * @returns {number} First paint time in milliseconds
         * @since 1.0.0
         */
        _getFirstPaint() {
            if (window.performance && window.performance.getEntriesByType) {
                const paintEntries = performance.getEntriesByType("paint");
                const firstPaint = paintEntries.find(entry => entry.name === "first-paint");
                return firstPaint ? firstPaint.startTime : 0;
            }
            return 0;
        },

        /**
         * Generate correlation ID for request tracking
         * @private
         * @returns {string} Correlation ID
         * @since 1.0.0
         */
        _generateCorrelationId() {
            return `client-${ Date.now() }-${
                Math.random().toString(36).substr(2, 9)}`;
        }
    });
});