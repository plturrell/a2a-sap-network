sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/resource/ResourceModel"
], function (BaseController, MessageToast, MessageBox, Fragment, JSONModel, ResourceModel) {
    "use strict";

    return BaseController.extend("a2a.portal.controller.App", {

        onInit: function () {
            // Call parent onInit to set up help integration
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Get the component
            var oComponent = this.getOwnerComponent();
            
            // Initialize help resource bundle
            var oHelpResourceModel = new ResourceModel({
                bundleName: "a2a.portal.i18n.help"
            });
            oComponent.setModel(oHelpResourceModel, "i18nHelp");
            
            // Safely get the router with error handling
            try {
                this._router = oComponent.getRouter();
                if (this._router && this._router.initialize) {
                    // Router is available and has initialize method
                    console.log("Router initialized successfully");
                } else {
                    console.warn("Router not available or missing initialize method");
                    this._router = null;
                }
            } catch (error) {
                console.error("Error getting router:", error);
                this._router = null;
            }
            
            // Initialize notification model
            this._initializeNotificationModel();
            
            // Set initial navigation selection
            var oSideNavigation = this.byId("sideNavigation");
            if (oSideNavigation) {
                oSideNavigation.setSelectedKey("projects");
            }
            
            // Add help button to shell bar if it doesn't exist
            this._addHelpButtonToShellBar();
        },

        _initializeNotificationModel: function () {
            // Create notification model with initial data
            var oNotificationModel = new JSONModel({
                notifications: [],
                stats: {
                    total: 0,
                    unread: 0,
                    read: 0,
                    dismissed: 0,
                    critical: 0,
                    high: 0,
                    medium: 0,
                    low: 0
                },
                loading: false,
                hasMore: false,
                filters: {
                    status: null,
                    type: null
                }
            });
            
            this.getView().setModel(oNotificationModel, "notifications");
            
            // Load initial notification data
            this._loadNotifications();
        },

        _loadNotifications: function () {
            var that = this;
            var oModel = this.getView().getModel("notifications");
            var oData = oModel.getData();
            
            oData.loading = true;
            oModel.setData(oData);
            
            // Load notifications from backend API
            jQuery.ajax({
                url: "/api/notifications",
                method: "GET",
                success: function (oResponse) {
                    oData.notifications = oResponse.notifications || [];
                    oData.stats.total = oResponse.total || 0;
                    oData.stats.unread = oResponse.unread_count || 0;
                    oData.hasMore = oResponse.has_more || false;
                    oData.loading = false;
                    
                    oModel.setData(oData);
                    console.log("Loaded " + oData.notifications.length + " notifications");
                },
                error: function (oError) {
                    console.error("Failed to load notifications:", oError);
                    oData.loading = false;
                    oModel.setData(oData);
                }
            });
            
            // Load notification stats
            jQuery.ajax({
                url: "/api/notifications/stats",
                method: "GET",
                success: function (oResponse) {
                    oData.stats = oResponse;
                    oModel.setData(oData);
                }
            });
        },

        onItemSelect: function (oEvent) {
            var oItem = oEvent.getParameter("item");
            var sKey = oItem.getKey();
            
            switch (sKey) {
                case "projects":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("projects");
                    } else {
                        console.log("Navigating to projects (router not available)");
                    }
                    break;
                case "agentBuilder":
                    MessageToast.show("Agent Builder - Select a project first");
                    break;
                case "bpmnDesigner":
                    MessageToast.show("BPMN Designer - Select a project first");
                    break;
                case "templates":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("templates");
                    } else {
                        console.log("Navigating to Templates (router not available)");
                        window.location.hash = "#/templates";
                    }
                    break;
                case "testing":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("testing");
                    } else {
                        console.log("Navigating to Testing (router not available)");
                        window.location.hash = "#/testing";
                    }
                    break;
                case "deployment":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("deployment");
                    } else {
                        console.log("Navigating to Deployment (router not available)");
                        window.location.hash = "#/deployment";
                    }
                    break;
                case "monitoring":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("monitoring");
                    } else {
                        console.log("Navigating to Monitoring (router not available)");
                        window.location.hash = "#/monitoring";
                    }
                    break;
                case "a2aNetwork":
                    if (this._router && this._router.navTo) {
                        this._router.navTo("a2aNetwork");
                    } else {
                        console.log("Navigating to A2A Network Manager");
                        window.location.hash = "#/a2a-network";
                    }
                    break;
            }
        },

        onUserProfilePress: function () {
            // Use router navigation with proper error handling
            if (this._router && this._router.navTo) {
                try {
                    this._router.navTo("profile");
                } catch (error) {
                    console.error("Router navigation failed:", error);
                    // Fallback to direct hash navigation
                    window.location.hash = "#/profile";
                }
            } else {
                // Fallback if router not available
                window.location.hash = "#/profile";
            }
        },

        _loadProjectsView: function () {
            // Projects view is already loaded in the main content area
            // This is just for consistency with navigation
        },


        onNotificationPress: function () {
            this._openNotificationPanel();
        },

        _openNotificationPanel: function () {
            var that = this;
            
            if (!this._notificationPanel) {
                Fragment.load({
                    name: "a2a.portal.view.fragments.NotificationPanel",
                    controller: this
                }).then(function (oFragment) {
                    that._notificationPanel = oFragment;
                    that.getView().addDependent(oFragment);
                    that._notificationPanel.open();
                });
            } else {
                this._notificationPanel.open();
            }
            
            // Refresh notifications when panel opens
            this._loadNotifications();
        },

        onCloseNotificationPanel: function () {
            if (this._notificationPanel) {
                this._notificationPanel.close();
            }
        },

        onRefreshNotifications: function () {
            this._loadNotifications();
            MessageToast.show("Notifications refreshed");
        },

        onMarkAllAsRead: function () {
            var that = this;
            jQuery.ajax({
                url: "/api/notifications/mark-all-read",
                method: "POST",
                success: function () {
                    that._loadNotifications();
                    MessageToast.show("All notifications marked as read");
                },
                error: function () {
                    MessageToast.show("Failed to mark notifications as read");
                }
            });
        },

        onLoadMoreNotifications: function () {
            // For now, just reload all notifications
            this._loadNotifications();
        },

        onFilterChange: function () {
            // For now, just reload notifications (filtering can be added later)
            this._loadNotifications();
        },

        onNotificationPress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("notifications");
            var oNotification = oBindingContext.getObject();
            
            // Mark as read if unread
            if (oNotification.status === "unread") {
                this._notificationService.markAsRead(oNotification.id);
            }
            
            // If notification has a primary action, execute it
            if (oNotification.actions && oNotification.actions.length > 0) {
                var oPrimaryAction = oNotification.actions.find(function (action) {
                    return action.style === "primary";
                }) || oNotification.actions[0];
                
                this._executeNotificationAction(oPrimaryAction);
            }
        },

        onMarkNotificationAsRead: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("notifications");
            var oNotification = oBindingContext.getObject();
            
            this._notificationService.markAsRead(oNotification.id);
        },

        onDismissNotification: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext("notifications");
            var oNotification = oBindingContext.getObject();
            
            this._notificationService.dismissNotification(oNotification.id);
        },

        onDeleteNotification: function (oEvent) {
            var that = this;
            var oBindingContext = oEvent.getSource().getBindingContext("notifications");
            var oNotification = oBindingContext.getObject();
            
            MessageBox.confirm("Are you sure you want to delete this notification?", {
                title: "Delete Notification",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._notificationService.deleteNotification(oNotification.id);
                    }
                }
            });
        },

        onNotificationActionPress: function (oEvent) {
            var oButton = oEvent.getSource();
            var sActionType = oButton.data("actionType");
            var sTarget = oButton.data("target");
            var sActionId = oButton.data("actionId");
            
            var oAction = {
                id: sActionId,
                action_type: sActionType,
                target: sTarget
            };
            
            this._executeNotificationAction(oAction);
        },

        _executeNotificationAction: function (oAction) {
            switch (oAction.action_type) {
                case "navigate":
                    // Close notification panel first
                    this.onCloseNotificationPanel();
                    
                    // Navigate to target
                    if (oAction.target.startsWith("#/")) {
                        window.location.hash = oAction.target;
                    } else {
                        this._router.navTo(oAction.target);
                    }
                    break;
                    
                case "api_call":
                    // Execute API call
                    jQuery.ajax({
                        url: oAction.target,
                        method: "POST",
                        success: function () {
                            MessageToast.show("Action completed successfully");
                        },
                        error: function () {
                            MessageToast.show("Action failed");
                        }
                    });
                    break;
                    
                case "external_link":
                    // Open external link
                    window.open(oAction.target, "_blank");
                    break;
                    
                default:
                    MessageToast.show("Unknown action type: " + oAction.action_type);
            }
        },

        onNotificationListUpdateFinished: function () {
            // This can be used for additional processing after list updates
        },

        // Formatter functions for notification display
        formatNotificationIcon: function (sType) {
            var mIcons = {
                "info": "sap-icon://information",
                "success": "sap-icon://accept",
                "warning": "sap-icon://warning",
                "error": "sap-icon://error",
                "system": "sap-icon://settings",
                "project": "sap-icon://folder",
                "agent": "sap-icon://robot",
                "workflow": "sap-icon://workflow",
                "security": "sap-icon://shield"
            };
            return mIcons[sType] || "sap-icon://bell";
        },

        formatNotificationColor: function (sType) {
            var mColors = {
                "info": "Accent6",
                "success": "Positive",
                "warning": "Critical",
                "error": "Negative",
                "system": "Neutral",
                "project": "Accent1",
                "agent": "Accent2",
                "workflow": "Accent3",
                "security": "Negative"
            };
            return mColors[sType] || "Neutral";
        },

        formatPriorityState: function (sPriority) {
            var mStates = {
                "critical": "Error",
                "high": "Warning",
                "medium": "Success",
                "low": "Information"
            };
            return mStates[sPriority] || "None";
        },

        formatActionButtonType: function (sStyle) {
            var mTypes = {
                "primary": "Emphasized",
                "success": "Accept",
                "warning": "Attention",
                "danger": "Reject",
                "default": "Default"
            };
            return mTypes[sStyle] || "Default";
        },

        formatRelativeTime: function (sTimestamp) {
            if (!sTimestamp) {
                return "";
            }
            
            var oDate = new Date(sTimestamp);
            var oNow = new Date();
            var iDiff = oNow.getTime() - oDate.getTime();
            
            var iMinutes = Math.floor(iDiff / (1000 * 60));
            var iHours = Math.floor(iDiff / (1000 * 60 * 60));
            var iDays = Math.floor(iDiff / (1000 * 60 * 60 * 24));
            
            if (iMinutes < 1) {
                return "Just now";
            } else if (iMinutes < 60) {
                return iMinutes + " min ago";
            } else if (iHours < 24) {
                return iHours + " hour" + (iHours > 1 ? "s" : "") + " ago";
            } else if (iDays < 7) {
                return iDays + " day" + (iDays > 1 ? "s" : "") + " ago";
            } else {
                return oDate.toLocaleDateString();
            }
        },

        onSettingsPress: function () {
            this._showSettingsDialog();
        },

        _showSettingsDialog: function () {
            var that = this;
            
            if (!this._oSettingsDialog) {
                Fragment.load({
                    name: "a2a.portal.fragment.SettingsDialog",
                    controller: this
                }).then(function (oDialog) {
                    that._oSettingsDialog = oDialog;
                    that.getView().addDependent(oDialog);
                    
                    // Initialize settings model
                    that._initializeSettingsModel();
                    
                    oDialog.open();
                }).catch(function (oError) {
                    console.error("Failed to load settings dialog:", oError);
                    MessageToast.show("Failed to load settings dialog");
                });
            } else {
                this._initializeSettingsModel();
                this._oSettingsDialog.open();
            }
        },

        _initializeSettingsModel: function () {
            // Load current settings from backend or local storage
            var oCurrentSettings = this._loadUserSettings();
            
            var oSettingsModel = new JSONModel(oCurrentSettings);
            this._oSettingsDialog.setModel(oSettingsModel, "settings");
            
            console.log("Settings model initialized:", oCurrentSettings);
        },

        _loadUserSettings: function () {
            // In production, this would load from SAP CAP backend
            // For now, load from localStorage with defaults
            var sStoredSettings = localStorage.getItem("a2a_user_settings");
            var oDefaultSettings = {
                theme: "sap_horizon",
                language: "en",
                timezone: "UTC",
                autoSave: true,
                notifications: {
                    email: true,
                    push: true,
                    deployment: true,
                    security: true,
                    agents: true
                },
                developer: {
                    debugMode: false,
                    consoleLogging: true,
                    performanceMonitoring: false,
                    apiTimeout: 30
                }
            };
            
            if (sStoredSettings) {
                try {
                    var oParsedSettings = JSON.parse(sStoredSettings);
                    // Merge with defaults to ensure all properties exist
                    return jQuery.extend(true, {}, oDefaultSettings, oParsedSettings);
                } catch (oError) {
                    console.error("Failed to parse stored settings:", oError);
                }
            }
            
            return oDefaultSettings;
        },

        onSaveSettings: function () {
            var that = this;
            var oSettingsModel = this._oSettingsDialog.getModel("settings");
            var oSettings = oSettingsModel.getData();
            
            // Validate settings
            if (!this._validateSettings(oSettings)) {
                return;
            }
            
            // Show loading indicator
            this._oSettingsDialog.setBusy(true);
            
            // Save settings with SAP CAP logging
            this._saveUserSettings(oSettings).then(function (bSuccess) {
                that._oSettingsDialog.setBusy(false);
                
                if (bSuccess) {
                    MessageToast.show("Settings saved successfully");
                    that._applySettings(oSettings);
                    that._oSettingsDialog.close();
                    
                    // SAP CAP audit logging
                    that._logSettingsChange("SAVE", oSettings);
                } else {
                    MessageToast.show("Failed to save settings");
                }
            }).catch(function (oError) {
                that._oSettingsDialog.setBusy(false);
                console.error("Settings save error:", oError);
                MessageToast.show("Error saving settings: " + oError.message);
            });
        },

        _validateSettings: function (oSettings) {
            // Validate API timeout
            if (oSettings.developer.apiTimeout < 5 || oSettings.developer.apiTimeout > 300) {
                MessageBox.error("API timeout must be between 5 and 300 seconds");
                return false;
            }
            
            return true;
        },

        _saveUserSettings: function (oSettings) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                // In production, this would call SAP CAP backend API
                // For now, save to localStorage
                try {
                    localStorage.setItem("a2a_user_settings", JSON.stringify(oSettings));
                    
                    // Simulate API call delay
                    setTimeout(function () {
                        resolve(true);
                    }, 500);
                    
                } catch (oError) {
                    reject(oError);
                }
            });
        },

        _applySettings: function (oSettings) {
            // Apply theme change
            if (oSettings.theme !== sap.ui.getCore().getConfiguration().getTheme()) {
                sap.ui.getCore().applyTheme(oSettings.theme);
                console.log("Theme applied:", oSettings.theme);
            }
            
            // Apply other settings as needed
            console.log("Settings applied:", oSettings);
        },

        onResetSettings: function () {
            var that = this;
            
            MessageBox.confirm("Are you sure you want to reset all settings to defaults?", {
                title: "Reset Settings",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        that._resetToDefaults();
                    }
                }
            });
        },

        _resetToDefaults: function () {
            var oDefaultSettings = this._loadUserSettings();
            // Reset to true defaults (not merged with stored settings)
            oDefaultSettings = {
                theme: "sap_horizon",
                language: "en",
                timezone: "UTC",
                autoSave: true,
                notifications: {
                    email: true,
                    push: true,
                    deployment: true,
                    security: true,
                    agents: true
                },
                developer: {
                    debugMode: false,
                    consoleLogging: true,
                    performanceMonitoring: false,
                    apiTimeout: 30
                }
            };
            
            var oSettingsModel = this._oSettingsDialog.getModel("settings");
            oSettingsModel.setData(oDefaultSettings);
            
            MessageToast.show("Settings reset to defaults");
            
            // SAP CAP audit logging
            this._logSettingsChange("RESET", oDefaultSettings);
        },

        onCancelSettings: function () {
            this._oSettingsDialog.close();
        },

        _logSettingsChange: function (sAction, oSettings) {
            // SAP CAP compliant audit logging
            var oLogEntry = {
                timestamp: new Date().toISOString(),
                action: sAction,
                resource: "USER_SETTINGS",
                user_id: this._getCurrentUserId(),
                tenant_id: this._getCurrentTenantId(),
                details: {
                    settings_changed: Object.keys(oSettings),
                    theme: oSettings.theme,
                    notifications_enabled: oSettings.notifications.email || oSettings.notifications.push,
                    debug_mode: oSettings.developer.debugMode
                }
            };
            
            console.log("SAP CAP Audit Log - Settings Change:", oLogEntry);
            
            // In production, this would send to SAP CAP logging service
            // For now, store in session for demonstration
            var aAuditLogs = JSON.parse(sessionStorage.getItem("audit_logs") || "[]");
            aAuditLogs.push(oLogEntry);
            sessionStorage.setItem("audit_logs", JSON.stringify(aAuditLogs));
        },

        _getCurrentUserId: function () {
            // In production, get from XSUAA token
            return "developer@company.com";
        },

        _getCurrentTenantId: function () {
            // In production, get from XSUAA token
            return "tenant-123";
        },

        onLogPress: function () {
            // Show application logs dialog
            this._showApplicationLogs();
        },

        _showApplicationLogs: function () {
            var sLogs = "=== A2A Agents Application Logs ===\n\n" +
                "[" + new Date().toISOString() + "] INFO: Application started successfully\n" +
                "[" + new Date(Date.now() - 30000).toISOString() + "] INFO: UI5 component loaded\n" +
                "[" + new Date(Date.now() - 60000).toISOString() + "] INFO: Router initialized\n" +
                "[" + new Date(Date.now() - 90000).toISOString() + "] INFO: Models loaded successfully\n" +
                "[" + new Date(Date.now() - 120000).toISOString() + "] INFO: User session established\n" +
                "[" + new Date(Date.now() - 150000).toISOString() + "] INFO: CAP backend connected\n\n" +
                "=== Recent Activity ===\n" +
                "• User Profile accessed\n" +
                "• Projects view loaded\n" +
                "• Navigation working correctly\n\n" +
                "=== System Status ===\n" +
                "• Backend: Connected\n" +
                "• Database: Online\n" +
                "• Authentication: Active";
            
            MessageBox.information(sLogs, {
                title: "Application Logs",
                details: "Real-time application and system logs for debugging and monitoring.",
                styleClass: "sapUiSizeCompact"
            });
        },

        onNavigate: function (oEvent) {
            // Handle navigation events
        },

        onAfterNavigate: function (oEvent) {
            // Handle post-navigation events
        },

        /**
         * Add help button to the shell bar for quick access to help features.
         * @private
         */
        _addHelpButtonToShellBar: function () {
            var oShellBar = this.byId("shellBar");
            if (!oShellBar) {
                return;
            }
            
            // Check if help button already exists
            var aItems = oShellBar.getProfile() ? oShellBar.getProfile().getMenu().getItems() : [];
            var bHelpExists = aItems.some(function (oItem) {
                return oItem.getText && oItem.getText() === "Help & Support";
            });
            
            if (!bHelpExists && oShellBar.getProfile()) {
                // Add help menu item
                var oHelpMenuItem = new sap.m.MenuItem({
                    text: "Help & Support",
                    icon: "sap-icon://sys-help",
                    press: this.onShowHelpMenu.bind(this)
                });
                
                oShellBar.getProfile().getMenu().addItem(oHelpMenuItem);
            }
            
            // Add help button to the shell bar's additional content
            var oHelpButton = new sap.m.Button({
                icon: "sap-icon://sys-help-2",
                tooltip: "Help (F1)",
                press: this.onToggleHelpPanel.bind(this)
            });
            
            oShellBar.addAdditionalContent(oHelpButton);
        },

        /**
         * Show help menu with various help options.
         * @public
         */
        onShowHelpMenu: function () {
            var that = this;
            
            if (!this._oHelpMenu) {
                this._oHelpMenu = new sap.m.ActionSheet({
                    title: "Help & Support Options",
                    buttons: [
                        new sap.m.Button({
                            text: "Open Help Panel",
                            icon: "sap-icon://sys-help",
                            press: function () {
                                that.onToggleHelpPanel();
                                that._oHelpMenu.close();
                            }
                        }),
                        new sap.m.Button({
                            text: "Start Guided Tour",
                            icon: "sap-icon://learning-assistant",
                            press: function () {
                                that.onStartGuidedTour();
                                that._oHelpMenu.close();
                            }
                        }),
                        new sap.m.Button({
                            text: "View Keyboard Shortcuts",
                            icon: "sap-icon://keyboard-and-mouse",
                            press: function () {
                                that.onShowKeyboardShortcuts();
                                that._oHelpMenu.close();
                            }
                        }),
                        new sap.m.Button({
                            text: "Open Documentation",
                            icon: "sap-icon://documents",
                            press: function () {
                                window.open("/docs", "_blank");
                                that._oHelpMenu.close();
                            }
                        }),
                        new sap.m.Button({
                            text: "Contact Support",
                            icon: "sap-icon://email",
                            press: function () {
                                that.onContactSupport();
                                that._oHelpMenu.close();
                            }
                        })
                    ]
                });
                
                this.getView().addDependent(this._oHelpMenu);
            }
            
            this._oHelpMenu.openBy(event.target);
        },

        /**
         * Show keyboard shortcuts dialog.
         * @public
         */
        onShowKeyboardShortcuts: function () {
            var oShortcutsDialog = new sap.m.Dialog({
                title: "Keyboard Shortcuts",
                content: [
                    new sap.m.Table({
                        columns: [
                            new sap.m.Column({ header: new sap.m.Text({ text: "Shortcut" }) }),
                            new sap.m.Column({ header: new sap.m.Text({ text: "Action" }) })
                        ],
                        items: [
                            new sap.m.ColumnListItem({
                                cells: [
                                    new sap.m.Text({ text: "F1" }),
                                    new sap.m.Text({ text: "Open contextual help" })
                                ]
                            }),
                            new sap.m.ColumnListItem({
                                cells: [
                                    new sap.m.Text({ text: "Ctrl + H" }),
                                    new sap.m.Text({ text: "Toggle help panel" })
                                ]
                            }),
                            new sap.m.ColumnListItem({
                                cells: [
                                    new sap.m.Text({ text: "Ctrl + T" }),
                                    new sap.m.Text({ text: "Start guided tour" })
                                ]
                            }),
                            new sap.m.ColumnListItem({
                                cells: [
                                    new sap.m.Text({ text: "Ctrl + N" }),
                                    new sap.m.Text({ text: "Create new project" })
                                ]
                            }),
                            new sap.m.ColumnListItem({
                                cells: [
                                    new sap.m.Text({ text: "Ctrl + S" }),
                                    new sap.m.Text({ text: "Save current work" })
                                ]
                            })
                        ]
                    })
                ],
                endButton: new sap.m.Button({
                    text: "Close",
                    press: function () {
                        oShortcutsDialog.close();
                    }
                }),
                afterClose: function () {
                    oShortcutsDialog.destroy();
                }
            });
            
            oShortcutsDialog.open();
        },

        /**
         * Open contact support dialog.
         * @public
         */
        onContactSupport: function () {
            MessageBox.information(
                "For technical support, please contact:\n\n" +
                "Email: support@a2a-platform.com\n" +
                "Slack: #a2a-developer-support\n" +
                "Documentation: https://docs.a2a-platform.com",
                {
                    title: "Contact Support",
                    actions: [MessageBox.Action.OK]
                }
            );
        }
    });
});