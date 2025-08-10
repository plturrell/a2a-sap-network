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
], function(UIComponent, Device, JSONModel, MessageBox, MessageToast, FlexibleColumnLayoutSemanticHelper, fioriLibrary, coreLibrary, Log, UriParameters) {
    "use strict";

    var LayoutType = fioriLibrary.LayoutType;
    var MessageType = coreLibrary.MessageType;

    return UIComponent.extend("a2a.network.fiori.Component", {
        metadata: {
            manifest: "json",
            interfaces: ["sap.ui.core.IAsyncContentCreation"]
        },

        _oSocket: null,
        _oFlexibleColumnLayoutSemanticHelper: null,

        /**
         * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
         * @public
         * @override
         */
        init: function() {
            // call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);

            // create the device model
            this.setModel(this._createDeviceModel(), "device");

            // create the app model
            this.setModel(this._createAppModel(), "app");

            // enable routing
            this.getRouter().initialize();

            // initialize services
            this._initServices();

            Log.info("A2A Network Fiori Component initialized");
        },

        /**
         * The component is destroyed by UI5 automatically.
         * @public
         * @override
         */
        destroy: function() {
            // disconnect WebSocket
            if (this._oSocket) {
                this._oSocket.disconnect();
                this._oSocket = null;
            }

            // call the base component's destroy function
            UIComponent.prototype.destroy.apply(this, arguments);
        },

        /**
         * This method can be called to determine whether the sapUiSizeCompact or sapUiSizeCozy
         * design mode class should be set, which influences the size appearance of some controls.
         * @public
         * @return {string} css class, either 'sapUiSizeCompact' or 'sapUiSizeCozy' - or an empty string if no css class should be set
         */
        getContentDensityClass: function() {
            if (this._sContentDensityClass === undefined) {
                // check whether FLP has already set the content density class; do nothing in this case
                if (document.body.classList.contains("sapUiSizeCozy") || document.body.classList.contains("sapUiSizeCompact")) {
                    this._sContentDensityClass = "";
                } else if (!Device.support.touch) { // apply "compact" mode if touch is not supported
                    this._sContentDensityClass = "sapUiSizeCompact";
                } else {
                    // "cozy" in case of touch support; default for most sap.m controls, but needed for desktop-first controls like sap.ui.table.Table
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
        getHelper: function() {
            var oFCL = this.getRootControl().byId("fcl"),
                oParams = UriParameters.fromQuery(location.search),
                oSettings = {
                    defaultTwoColumnLayoutType: LayoutType.TwoColumnsMidExpanded,
                    defaultThreeColumnLayoutType: LayoutType.ThreeColumnsMidExpanded,
                    mode: oParams.get("mode"),
                    maxColumnsCount: oParams.get("max")
                };

            if (!this._oFlexibleColumnLayoutSemanticHelper) {
                this._oFlexibleColumnLayoutSemanticHelper = FlexibleColumnLayoutSemanticHelper.getInstanceFor(oFCL, oSettings);
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
        _createDeviceModel: function() {
            var oModel = new JSONModel(Device);
            oModel.setDefaultBindingMode("OneWay");
            return oModel;
        },

        /**
         * Creates the app model.
         * @private
         * @returns {sap.ui.model.json.JSONModel} the app model
         */
        _createAppModel: function() {
            var oModel = new JSONModel({
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
        _initServices: function() {
            // Initialize WebSocket connection
            this._initWebSocket();

            // Load initial data
            this._loadNetworkStats();

            // Initialize user authentication
            this._initUserAuthentication();

            // Set up periodic refresh
            setInterval(function() {
                this._loadNetworkStats();
            }.bind(this), 60000); // Refresh every minute
        },

        /**
         * Initializes WebSocket connection.
         * @private
         */
        _initWebSocket: function() {
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

                this._oSocket.on("connect", function() {
                    Log.info("WebSocket connected");
                    this.getModel("app").setProperty("/networkStatus", "connected");

                    // Subscribe to relevant events
                    this._oSocket.emit("subscribe", {
                        rooms: ["agents", "services", "workflows", "reputation"]
                    });
                }.bind(this));

                this._oSocket.on("disconnect", function() {
                    Log.warning("WebSocket disconnected");
                    this.getModel("app").setProperty("/networkStatus", "disconnected");
                }.bind(this));

                this._oSocket.on("error", function(error) {
                    Log.error("WebSocket error", error);
                    this.getModel("app").setProperty("/networkStatus", "error");
                }.bind(this));

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
        _attachSocketEventHandlers: function() {
            if (!this._oSocket) return;

            // Agent events
            this._oSocket.on("agent:registered", function(data) {
                MessageToast.show(this._getResourceBundle().getText("agentRegisteredMessage", [data.name]));
                this._loadNetworkStats();
                this.getEventBus().publish("app", "agentRegistered", data);
            }.bind(this));

            this._oSocket.on("agent:updated", function(data) {
                this.getEventBus().publish("app", "agentUpdated", data);
            }.bind(this));

            // Service events
            this._oSocket.on("service:created", function(data) {
                MessageToast.show(this._getResourceBundle().getText("serviceCreatedMessage", [data.name]));
                this.getEventBus().publish("app", "serviceCreated", data);
            }.bind(this));

            // Reputation events
            this._oSocket.on("reputation:updated", function(data) {
                this.getEventBus().publish("app", "reputationUpdated", data);
                
                // Update UI if viewing the affected agent
                var sCurrentHash = this.getRouter().getHashChanger().getHash();
                if (sCurrentHash.includes(data.agentId)) {
                    this.getModel().refresh();
                }
            }.bind(this));

            // Workflow events
            this._oSocket.on("workflow:completed", function(data) {
                MessageToast.show(this._getResourceBundle().getText("workflowCompletedMessage", [data.executionId]));
                this.getEventBus().publish("app", "workflowCompleted", data);
            }.bind(this));

            this._oSocket.on("workflow:failed", function(data) {
                MessageBox.error(this._getResourceBundle().getText("workflowFailedMessage", [data.executionId]));
                this.getEventBus().publish("app", "workflowFailed", data);
            }.bind(this));
        },

        /**
         * Loads network statistics.
         * @private
         */
        _loadNetworkStats: function() {
            var oModel = this.getModel();
            var oAppModel = this.getModel("app");

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
                success: function(oData) {
                    if (oData.results && oData.results.length > 0) {
                        var oStats = oData.results[0];
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
                error: function(oError) {
                    Log.error("Failed to load network stats", oError);
                }
            });
        },

        /**
         * Initializes user authentication.
         * @private
         */
        _initUserAuthentication: function() {
            var oAppModel = this.getModel("app");
            var oEnvironment = oAppModel.getProperty("/environment");

            if (oEnvironment.isBTP) {
                // In BTP, user info comes from XSUAA
                this._loadBTPUserInfo();
            } else {
                // Local development mode
                oAppModel.setProperty("/currentUser", {
                    id: "local-user",
                    name: "Developer",
                    email: "developer@a2a.network",
                    roles: ["Admin", "Developer"]
                });
                Log.info("Running in development mode with local user");
            }
        },

        /**
         * Loads user info from BTP.
         * @private
         */
        _loadBTPUserInfo: function() {
            var oAppModel = this.getModel("app");

            // Call user info endpoint
            jQuery.ajax({
                url: "/user-api/currentUser",
                type: "GET",
                success: function(oUserInfo) {
                    oAppModel.setProperty("/currentUser", {
                        id: oUserInfo.id,
                        name: oUserInfo.name || oUserInfo.id,
                        email: oUserInfo.email,
                        roles: oUserInfo.scopes || []
                    });
                    Log.info("User authenticated", { userId: oUserInfo.id });
                },
                error: function(oError) {
                    Log.error("Failed to load user info", oError);
                    MessageBox.error(this._getResourceBundle().getText("authenticationError"));
                }.bind(this)
            });
        },

        /**
         * Gets environment information.
         * @private
         * @returns {object} Environment info
         */
        _getEnvironmentInfo: function() {
            var sHostname = window.location.hostname;
            var bIsBTP = sHostname.includes("cfapps") || 
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
        _getResourceBundle: function() {
            return this.getModel("i18n").getResourceBundle();
        }
    });
});