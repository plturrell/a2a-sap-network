sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/ResponsivePopover",
    "sap/m/Button",
    "sap/m/List",
    "sap/m/StandardListItem",
    "sap/m/NotificationListItem",
    "sap/base/Log",
    "../mixin/PersonalizationMixin",
    "../mixin/OfflineMixin"
], function(BaseController, MessageToast, ResponsivePopover, Button, List, StandardListItem,
    NotificationListItem, Log, PersonalizationMixin, OfflineMixin) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.App", {
        _oNotificationPopover: null,
        _oUserPopover: null,
        onInit() {
            // Mix in PersonalizationMixin and OfflineMixin
            Object.assign(this, PersonalizationMixin, OfflineMixin);

            // Initialize mixins
            this.initPersonalization();
            this.initOfflineCapabilities();

            // Apply content density mode to root view
            this.getView().addStyleClass(this.getOwnerComponent().getContentDensityClass());

            // Initialize side navigation
            const oSideNavigation = this.byId("sideNavigation");
            if (oSideNavigation) {
                oSideNavigation.setSelectedKey("home");
            }

            // Register for cleanup
            this._registerForCleanup(function() {
                if (this._oNotificationPopover) {
                    this._oNotificationPopover.destroy();
                    this._oNotificationPopover = null;
                }
                if (this._oUserPopover) {
                    this._oUserPopover.destroy();
                    this._oUserPopover = null;
                }

                // Cleanup mixins
                this.cleanupPersonalization();
                this.cleanupOfflineCapabilities();
            }.bind(this));

            Log.info("App controller initialized with personalization and offline capabilities");
        },

        onSideNavButtonPress() {
            const oToolPage = this.byId("toolPage");
            if (!oToolPage) {
                Log.error("ToolPage control not found");
                return;
            }

            const bSideExpanded = oToolPage.getSideExpanded();
            oToolPage.setSideExpanded(!bSideExpanded);

            Log.debug("Side navigation toggled", bSideExpanded ? "collapsed" : "expanded");
        },

        onItemSelect(oEvent) {
            const oItem = oEvent.getParameter("item");
            if (!oItem) {
                Log.error("No item found in navigation event");
                return;
            }

            const sKey = oItem.getKey();
            Log.debug("Navigation item selected", sKey);

            // Define route mapping
            const mRoutes = {
                "home": "home",
                "agents": "agents",
                "services": "services",
                "capabilities": "capabilities",
                "workflows": "workflows",
                "analytics": "analytics",
                "blockchain": "blockchain",
                "agentVisualization": "agentVisualization",
                "contracts": "contracts",
                "transactions": "transactions",
                "settings": "settings"
            };

            if (mRoutes[sKey]) {
                this.getRouter().navTo(mRoutes[sKey]);
            } else {
                Log.warning("Unknown navigation key", sKey);
                MessageToast.show(this.getResourceBundle().getText("navigationNotImplemented", [sKey]));
            }
        },

        onNotificationPress(oEvent) {
            const oButton = oEvent.getSource();

            // Create notification popover if not exists
            if (!this._notificationPopover) {
                this._notificationPopover = new ResponsivePopover({
                    title: "Notifications",
                    contentWidth: "400px",
                    contentHeight: "400px",
                    placement: "Bottom",
                    content: [
                        new List({
                            items: [
                                new NotificationListItem({
                                    title: "New Agent Registered",
                                    description: "Agent 'DataProcessor-01' has been registered to the network",
                                    datetime: "5 minutes ago",
                                    priority: "High",
                                    close() {
                                        MessageToast.show("Notification closed");
                                    }
                                }),
                                new NotificationListItem({
                                    title: "Service Listed",
                                    description: "New service 'AI Model Training' is now available",
                                    datetime: "1 hour ago",
                                    priority: "Medium"
                                }),
                                new NotificationListItem({
                                    title: "Workflow Completed",
                                    description: "Workflow 'Data Pipeline #123' completed successfully",
                                    datetime: "2 hours ago",
                                    priority: "Low"
                                })
                            ]
                        })
                    ],
                    footer: [
                        new Button({
                            text: "Clear All",
                            type: "Transparent",
                            press: function() {
                                MessageToast.show("All notifications cleared");
                                this._notificationPopover.close();
                            }.bind(this)
                        })
                    ]
                });

                this.getView().addDependent(this._notificationPopover);
            }

            this._notificationPopover.openBy(oButton);
        },

        onAvatarPress(oEvent) {
            const oButton = oEvent.getSource();

            // Create user menu popover if not exists
            if (!this._userPopover) {
                this._userPopover = new ResponsivePopover({
                    title: "User Menu",
                    placement: "Bottom",
                    content: [
                        new List({
                            items: [
                                new StandardListItem({
                                    title: "Profile",
                                    icon: "sap-icon://person-placeholder",
                                    type: "Navigation",
                                    press: function() {
                                        MessageToast.show("Navigate to profile");
                                        this._userPopover.close();
                                    }.bind(this)
                                }),
                                new StandardListItem({
                                    title: "Settings",
                                    icon: "sap-icon://action-settings",
                                    type: "Navigation",
                                    press: function() {
                                        this.getOwnerComponent().getRouter().navTo("settings");
                                        this._userPopover.close();
                                    }.bind(this)
                                }),
                                new StandardListItem({
                                    title: "Help",
                                    icon: "sap-icon://sys-help",
                                    type: "Navigation",
                                    press: function() {
                                        MessageToast.show("Opening help");
                                        this._userPopover.close();
                                    }.bind(this)
                                }),
                                new StandardListItem({
                                    title: "Logout",
                                    icon: "sap-icon://log",
                                    type: "Navigation",
                                    press: function() {
                                        MessageToast.show("Logging out...");
                                        this._userPopover.close();
                                    }.bind(this)
                                })
                            ]
                        })
                    ]
                });

                this.getView().addDependent(this._userPopover);
            }

            this._userPopover.openBy(oButton);
        }
    });
});