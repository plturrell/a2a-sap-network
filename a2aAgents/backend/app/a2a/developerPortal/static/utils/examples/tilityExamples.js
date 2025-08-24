/**
 * @fileoverview Utility Examples for A2A Developer Portal
 * @module a2a/portal/utils/examples
 * @description Comprehensive examples demonstrating the usage of all utility functions
 * in real-world scenarios with best practices and error handling.
 */

sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "a2a/developer/portal/utils/GuidedTourManager",
    "a2a/portal/utils/HelpProvider",
    "a2a/portal/services/NotificationService",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], (Controller, GuidedTourManager, HelpProvider, NotificationService, MessageBox, MessageToast) => {
    "use strict";
/* global localStorage, WebSocket */

    return Controller.extend("a2a.portal.utils.examples.UtilityExamples", {

        /**
         * =====================================
         * GUIDED TOUR MANAGER EXAMPLES
         * =====================================
         */

        /**
         * Example 1: Basic onboarding tour for new users
         */
        startOnboardingTour: function() {
            const tourManager = new GuidedTourManager();
            
            const onboardingTour = {
                id: "new-user-onboarding",
                steps: [
                    {
                        target: "shellBar",
                        title: "Welcome to A2A Platform!",
                        content: "Let's take a quick tour to get you started with the platform.",
                        placement: "bottom"
                    },
                    {
                        target: "navMenu",
                        title: "Navigation Menu",
                        content: "Access all platform features from here. You can create agents, manage workflows, and monitor performance.",
                        placement: "right"
                    },
                    {
                        target: "createAgentBtn",
                        title: "Create Your First Agent",
                        content: "Click here to create your first A2A agent. We'll guide you through the process.",
                        placement: "bottom"
                    },
                    {
                        target: "notificationIcon",
                        title: "Stay Updated",
                        content: "Important notifications and alerts will appear here. Click to view all notifications.",
                        placement: "left"
                    },
                    {
                        target: "helpButton",
                        title: "Need Help?",
                        content: "Access documentation, tutorials, and support anytime by clicking here.",
                        placement: "left"
                    }
                ]
            };

            // Start tour with completion tracking
            tourManager.startTour(onboardingTour, this.getView(), () => {
                // Mark user as onboarded
                this._saveUserPreference("onboardingCompleted", true);
                
                // Show completion message
                MessageBox.success("Welcome aboard! You're ready to start using the A2A platform.", {
                    title: "Tour Completed",
                    actions: [MessageBox.Action.OK],
                    onClose: function() {
                        // Navigate to dashboard
                        this.getRouter().navTo("dashboard");
                    }.bind(this)
                });
            });
        },

        /**
         * Example 2: Feature-specific tour with validation
         */
        startAgentCreationTour: function() {
            const tourManager = new GuidedTourManager();
            const that = this;
            
            const agentTour = {
                id: "agent-creation-tutorial",
                steps: [
                    {
                        target: "agentNameInput",
                        title: "Step 1: Name Your Agent",
                        content: "Choose a descriptive name for your agent. Use lowercase letters and hyphens only.",
                        placement: "right",
                        validation: function() {
                            const nameInput = that.byId("agentNameInput");
                            const value = nameInput.getValue();
                            
                            if (!value) {
                                MessageToast.show("Please enter an agent name");
                                return false;
                            }
                            
                            if (!/^[a-z0-9-]+$/.test(value)) {
                                MessageToast.show("Agent name must contain only lowercase letters, numbers, and hyphens");
                                nameInput.setValueState("Error");
                                return false;
                            }
                            
                            nameInput.setValueState("Success");
                            return true;
                        }
                    },
                    {
                        target: "agentTypeSelect",
                        title: "Step 2: Select Agent Type",
                        content: "Choose the type based on your use case:\n• Reactive: Responds to events\n• Proactive: Initiates actions\n• Hybrid: Both reactive and proactive",
                        placement: "bottom",
                        beforeShow: function() {
                            // Populate agent types if not already done
                            that._loadAgentTypes();
                        }
                    },
                    {
                        target: "skillsContainer",
                        title: "Step 3: Configure Skills",
                        content: "Select pre-built skills or create custom ones. Skills define what your agent can do.",
                        placement: "top",
                        interactive: true
                    },
                    {
                        target: "testAgentBtn",
                        title: "Step 4: Test Your Agent",
                        content: "Always test your agent before deployment. Click here to run tests with sample data.",
                        placement: "left"
                    },
                    {
                        target: "deployBtn",
                        title: "Step 5: Deploy",
                        content: "Once tests pass, deploy your agent to the selected environment.",
                        placement: "left"
                    }
                ]
            };

            // Override navigation to include validation
            const originalNextStep = tourManager.nextStep.bind(tourManager);
            tourManager.nextStep = function() {
                const currentStep = this.getCurrentStep();
                const step = agentTour.steps[currentStep];
                
                if (step.validation && !step.validation()) {
                    return; // Don't proceed if validation fails
                }
                
                originalNextStep();
            };

            tourManager.startTour(agentTour, this.getView(), () => {
                // Tour completed successfully
                that._trackEvent("agent_tour_completed", {
                    duration: Date.now() - tourStartTime
                });
            });

            const tourStartTime = Date.now();
        },

        /**
         * Example 3: Progressive tour with saved state
         */
        startComprehensiveTour: function() {
            const tourManager = new GuidedTourManager();
            const tourId = "comprehensive-platform-tour";
            
            // Load saved progress
            const savedProgress = this._loadTourProgress(tourId);
            
            const comprehensiveTour = {
                id: tourId,
                steps: this._generateComprehensiveTourSteps()
            };

            if (savedProgress && savedProgress.stepIndex > 0) {
                MessageBox.confirm(
                    `Resume tour from step ${savedProgress.stepIndex + 1}?`,
                    {
                        title: "Resume Tour",
                        actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                        emphasizedAction: MessageBox.Action.YES,
                        onClose: function(oAction) {
                            if (oAction === MessageBox.Action.YES) {
                                // Resume from saved position
                                tourManager._iCurrentStep = savedProgress.stepIndex;
                            }
                            this._startTourWithProgress(tourManager, comprehensiveTour);
                        }.bind(this)
                    }
                );
            } else {
                this._startTourWithProgress(tourManager, comprehensiveTour);
            }
        },

        _startTourWithProgress: function(tourManager, tour) {
            const that = this;
            
            // Override show step to save progress
            const originalShowStep = tourManager._showStep.bind(tourManager);
            tourManager._showStep = function(stepIndex) {
                originalShowStep(stepIndex);
                
                // Save progress
                that._saveTourProgress(tour.id, {
                    stepIndex: stepIndex,
                    timestamp: new Date().toISOString()
                });
                
                // Update progress indicator
                that._updateTourProgress(stepIndex + 1, tour.steps.length);
            };

            tourManager.startTour(tour, this.getView(), () => {
                // Clear saved progress on completion
                that._clearTourProgress(tour.id);
                
                // Award achievement
                that._awardAchievement("tour_master", {
                    tourId: tour.id,
                    completionTime: new Date().toISOString()
                });
            });
        },

        /**
         * =====================================
         * HELP PROVIDER EXAMPLES
         * =====================================
         */

        /**
         * Example 1: Enable contextual help for a form
         */
        setupFormHelp: function() {
            const helpProvider = HelpProvider;
            
            // Define help content for form fields
            const formFields = [
                {
                    controlId: "projectNameInput",
                    helpKey: "projects.createProject.name",
                    options: {
                        showIcon: true,
                        placement: "Right",
                        trigger: "hover"
                    }
                },
                {
                    controlId: "projectDescTextArea",
                    helpKey: "projects.createProject.description",
                    options: {
                        showIcon: true,
                        placement: "Bottom"
                    }
                },
                {
                    controlId: "projectTypeSelect",
                    helpKey: "projects.createProject.type",
                    options: {
                        showIcon: true,
                        placement: "Bottom",
                        onShow: function() {
                            // Track help usage
                            this._trackEvent("help_viewed", {
                                helpKey: "projects.createProject.type"
                            });
                        }.bind(this)
                    }
                },
                {
                    controlId: "securityLevelSelect",
                    helpKey: "projects.createProject.security",
                    options: {
                        showIcon: true,
                        placement: "Top",
                        trigger: "click" // Sensitive field - require click
                    }
                }
            ];

            // Enable help for all form fields
            formFields.forEach((field) => {
                const control = this.byId(field.controlId);
                if (control) {
                    helpProvider.enableHelp(control, field.helpKey, field.options);
                }
            });

            // Add smart tooltips for the entire form
            helpProvider.enableSmartTooltips(this.byId("createProjectForm"), {
                "sap.m.Input": {
                    "ProjectCode": "Unique project identifier (auto-generated if empty)",
                    "Budget": "Enter amount in USD (format: 00000.00)",
                    "StartDate": "Project start date (cannot be in the past)"
                },
                "sap.m.CheckBox": {
                    "AutoDeploy": "Enable automatic deployment after successful tests",
                    "Notifications": "Receive email notifications for project updates"
                }
            });
        },

        /**
         * Example 2: Dynamic help based on user role and context
         */
        setupContextualHelp: function() {
            const helpProvider = HelpProvider;
            const userRole = this._getUserRole();
            const currentMode = this._getCurrentMode(); // 'edit' or 'view'
            
            // Build context-specific help keys
            const contextPrefix = `${currentMode}.${userRole}`;
            
            // Dynamic help content based on context
            const dynamicHelp = {};
            
            if (userRole === "admin") {
                dynamicHelp[`${contextPrefix}.advancedSettings`] = {
                    title: "Advanced Settings (Admin)",
                    content: "Configure system-wide settings and permissions.",
                    detailedHelp: "As an administrator, you can:\n• Set global configurations\n• Manage user permissions\n• Configure security policies\n• Access system logs",
                    warnings: ["Changes affect all users", "Some settings require restart"],
                    learnMoreUrl: "/docs/admin/advanced-settings"
                };
            } else if (userRole === "developer") {
                dynamicHelp[`${contextPrefix}.advancedSettings`] = {
                    title: "Developer Settings",
                    content: "Configure development environment and tools.",
                    detailedHelp: "Available developer options:\n• Debug mode\n• API access tokens\n• Test data generators\n• Performance profiling",
                    tips: ["Enable debug mode for detailed logs", "Use test data for development"],
                    codeExample: "// Enable debug mode\nconfig.setDebugMode(true);"
                };
            }

            // Merge dynamic help into provider
            Object.assign(helpProvider._helpContent, dynamicHelp);

            // Apply contextual help
            const settingsButton = this.byId("advancedSettingsBtn");
            helpProvider.enableHelp(settingsButton, `${contextPrefix}.advancedSettings`, {
                showIcon: true,
                placement: "Auto"
            });
        },

        /**
         * Example 3: Help search implementation
         */
        implementHelpSearch: function() {
            const _helpProvider = HelpProvider;
            const that = this;
            
            // Create search field
            const searchField = new sap.m.SearchField({
                placeholder: "Search help topics...",
                width: "300px",
                search: function(oEvent) {
                    const query = oEvent.getParameter("query");
                    that._searchHelp(query);
                },
                liveChange: function(oEvent) {
                    const value = oEvent.getParameter("newValue");
                    if (value.length >= 3) {
                        that._showHelpSuggestions(value);
                    }
                }
            });

            // Add to toolbar
            this.byId("helpToolbar").addContent(searchField);
        },

        _searchHelp: function(query) {
            const results = [];
            const helpContent = HelpProvider._helpContent;
            
            // Recursive search function
            const searchInContent = function(obj, path = []) {
                Object.keys(obj).forEach(key => {
                    const value = obj[key];
                    const currentPath = [...path, key];
                    
                    if (typeof value === 'object' && value !== null) {
                        // Check if this is a help item
                        if (value.title || value.content) {
                            const searchText = `${value.title} ${value.content} ${value.detailedHelp || ''}`.toLowerCase();
                            if (searchText.includes(query.toLowerCase())) {
                                results.push({
                                    path: currentPath.join('.'),
                                    title: value.title || key,
                                    content: value.content,
                                    score: this._calculateRelevance(searchText, query)
                                });
                            }
                        } else {
                            searchInContent(value, currentPath);
                        }
                    }
                });
            }.bind(this);

            searchInContent(helpContent);
            
            // Sort by relevance
            results.sort((a, b) => b.score - a.score);
            
            // Show results
            this._showHelpSearchResults(results.slice(0, 10));
        },

        /**
         * =====================================
         * NOTIFICATION SERVICE EXAMPLES
         * =====================================
         */

        /**
         * Example 1: Complete notification system setup
         */
        setupNotificationSystem: function() {
            // Initialize service
            this._notificationService = new NotificationService();
            
            // Bind to view
            this.getView().setModel(
                this._notificationService.getModel(),
                "notifications"
            );

            // Setup real-time updates
            this._setupRealtimeNotifications();
            
            // Setup notification filters
            this._setupNotificationFilters();
            
            // Setup notification actions
            this._setupNotificationActions();
            
            // Load initial notifications
            this._notificationService.loadNotifications(true)
                .then(() => {
                    this._updateNotificationBadge();
                })
                .catch((error) => {
                    MessageToast.show("Failed to load notifications");
                    console.error("Notification load error:", error);
                });
        },

        /**
         * Example 2: Real-time notification handling with WebSocket
         */
        _setupRealtimeNotifications: function() {
            const wsUrl = this._getWebSocketUrl("/api/notifications/ws");
            
            try {
                this._notificationWebSocket = new BlockchainEventClient(wsUrl);
                
                this._notificationWebSocket.onopen = function() {
                    // eslint-disable-next-line no-console
                    console.log("Notification WebSocket connected");
                    
                    // Send authentication token
                    this._notificationWebSocket.send(JSON.stringify({
                        type: "auth",
                        token: this._getAuthToken()
                    }));
                }.bind(this);

                this._notificationWebSocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    switch (data.type) {
                        case "notification":
                            this._handleNewNotification(data.notification);
                            break;
                        case "stats_update":
                            this._updateNotificationStats(data.stats);
                            break;
                        case "bulk_update":
                            this._handleBulkNotificationUpdate(data.notifications);
                            break;
                    }
                }.bind(this);

                this._notificationWebSocket.onerror = function(error) {
                    console.error("WebSocket error:", error);
                    // Fallback to polling
                    this._startNotificationPolling();
                }.bind(this);

                this._notificationWebSocket.onclose = function() {
                    // eslint-disable-next-line no-console
                    console.log("WebSocket connection closed");
                    // Attempt to reconnect after delay
                    setTimeout(() => {
                        this._setupRealtimeNotifications();
                    }, 5000);
                }.bind(this);

            } catch (error) {
                console.error("Failed to setup WebSocket:", error);
                this._startNotificationPolling();
            }
        },

        /**
         * Example 3: Advanced notification filtering
         */
        _setupNotificationFilters: function() {
            const that = this;
            
            // Create filter bar
            const filterBar = new sap.ui.comp.filterbar.FilterBar({
                filterItems: [
                    new sap.ui.comp.filterbar.FilterItem({
                        name: "status",
                        label: "Status",
                        control: new sap.m.Select({
                            items: [
                                new sap.ui.core.Item({ key: "all", text: "All" }),
                                new sap.ui.core.Item({ key: "unread", text: "Unread" }),
                                new sap.ui.core.Item({ key: "read", text: "Read" }),
                                new sap.ui.core.Item({ key: "dismissed", text: "Dismissed" })
                            ]
                        })
                    }),
                    new sap.ui.comp.filterbar.FilterItem({
                        name: "priority",
                        label: "Priority",
                        control: new sap.m.MultiComboBox({
                            items: [
                                new sap.ui.core.Item({ key: "critical", text: "Critical" }),
                                new sap.ui.core.Item({ key: "high", text: "High" }),
                                new sap.ui.core.Item({ key: "medium", text: "Medium" }),
                                new sap.ui.core.Item({ key: "low", text: "Low" })
                            ]
                        })
                    }),
                    new sap.ui.comp.filterbar.FilterItem({
                        name: "dateRange",
                        label: "Date Range",
                        control: new sap.m.DateRangeSelection({
                            displayFormat: "MM/dd/yyyy"
                        })
                    })
                ],
                search: function(_oEvent) {
                    const filters = {};
                    
                    // Get filter values
                    const statusFilter = this.getFilterItems()[0].getControl();
                    const priorityFilter = this.getFilterItems()[1].getControl();
                    const dateFilter = this.getFilterItems()[2].getControl();
                    
                    // Build filter object
                    const status = statusFilter.getSelectedKey();
                    if (status !== "all") {
                        filters.status = status;
                    }
                    
                    const priorities = priorityFilter.getSelectedKeys();
                    if (priorities.length > 0) {
                        filters.priorities = priorities;
                    }
                    
                    const dateRange = dateFilter.getDateValue() && dateFilter.getSecondDateValue();
                    if (dateRange) {
                        filters.dateFrom = dateFilter.getDateValue();
                        filters.dateTo = dateFilter.getSecondDateValue();
                    }
                    
                    // Apply filters
                    that._notificationService.setFilters(filters);
                }
            });

            // Add to page
            this.byId("notificationFilterContainer").addContent(filterBar);
        },

        /**
         * Example 4: Custom notification actions
         */
        _setupNotificationActions: function() {
            const that = this;
            
            // Create notification list with custom factory
            const notificationList = new sap.m.List({
                mode: "None",
                items: {
                    path: "notifications>/notifications",
                    factory: function(sId, oContext) {
                        const notification = oContext.getObject();
                        
                        return that._createNotificationItem(notification);
                    }
                }
            });

            this.byId("notificationContainer").addContent(notificationList);
        },

        _createNotificationItem: function(notification) {
            const that = this;
            
            const item = new sap.m.NotificationListItem({
                title: notification.title,
                description: notification.message,
                datetime: notification.created_at,
                unread: notification.status === "unread",
                priority: this._mapPriority(notification.priority),
                showCloseButton: true,
                authorPicture: notification.icon || "sap-icon://bell",
                close: function() {
                    that._notificationService.dismissNotification(notification.id);
                },
                press: function() {
                    // Mark as read if unread
                    if (notification.status === "unread") {
                        that._notificationService.markAsRead(notification.id);
                    }
                    
                    // Handle notification-specific action
                    that._handleNotificationPress(notification);
                }
            });

            // Add custom actions based on notification type
            this._addNotificationActions(item, notification);
            
            return item;
        },

        _addNotificationActions: function(item, notification) {
            const that = this;
            
            switch (notification.type) {
                case "agent_error":
                    item.addAction(new sap.m.Button({
                        text: "View Logs",
                        type: "Transparent",
                        press: function() {
                            that._showAgentLogs(notification.metadata.agentId);
                        }
                    }));
                    item.addAction(new sap.m.Button({
                        text: "Restart Agent",
                        type: "Emphasized",
                        press: function() {
                            that._restartAgent(notification.metadata.agentId);
                        }
                    }));
                    break;
                    
                case "workflow_approval":
                    item.addAction(new sap.m.Button({
                        text: "Approve",
                        type: "Accept",
                        press: function() {
                            that._approveWorkflow(notification.metadata.workflowId);
                        }
                    }));
                    item.addAction(new sap.m.Button({
                        text: "Reject",
                        type: "Reject",
                        press: function() {
                            that._rejectWorkflow(notification.metadata.workflowId);
                        }
                    }));
                    break;
                    
                case "security_alert":
                    item.addAction(new sap.m.Button({
                        text: "View Details",
                        type: "Critical",
                        press: function() {
                            that._showSecurityAlert(notification);
                        }
                    }));
                    break;
            }
        },

        /**
         * =====================================
         * UTILITY HELPER FUNCTIONS
         * =====================================
         */

        _getUserRole: function() {
            // Get user role from model or service
            return this.getModel("user").getProperty("/role") || "developer";
        },

        _getCurrentMode: function() {
            return this._editMode ? "edit" : "view";
        },

        _trackEvent: function(eventName, data) {
            // Analytics tracking
            if (window.analytics) {
                window.analytics.track(eventName, data);
            }
        },

        _saveUserPreference: function(key, value) {
            localStorage.setItem(`a2a_pref_${key}`, JSON.stringify(value));
        },

        _loadTourProgress: function(tourId) {
            const saved = localStorage.getItem(`tour_progress_${tourId}`);
            return saved ? JSON.parse(saved) : null;
        },

        _saveTourProgress: function(tourId, progress) {
            localStorage.setItem(`tour_progress_${tourId}`, JSON.stringify(progress));
        },

        _clearTourProgress: function(tourId) {
            localStorage.removeItem(`tour_progress_${tourId}`);
        },

        _updateNotificationBadge: function() {
            const unreadCount = this._notificationService.getUnreadCount();
            this.byId("notificationBadge").setText(unreadCount > 0 ? unreadCount : "");
            this.byId("notificationBadge").setVisible(unreadCount > 0);
        },

        _mapPriority: function(priority) {
            const mapping = {
                "critical": sap.ui.core.Priority.High,
                "high": sap.ui.core.Priority.High,
                "medium": sap.ui.core.Priority.Medium,
                "low": sap.ui.core.Priority.Low
            };
            return mapping[priority] || sap.ui.core.Priority.None;
        },

        /**
         * =====================================
         * ERROR HANDLING EXAMPLES
         * =====================================
         */

        /**
         * Comprehensive error handling wrapper
         */
        _safeExecute: async function(operation, fallback) {
            try {
                return await operation();
            } catch (error) {
                console.error("Operation failed:", error);
                
                // Log to monitoring service
                this._logError({
                    operation: operation.name || "unknown",
                    error: error.message,
                    stack: error.stack,
                    timestamp: new Date().toISOString()
                });

                // Show user-friendly message
                MessageToast.show("An error occurred. Please try again.");
                
                // Execute fallback if provided
                if (typeof fallback === 'function') {
                    return fallback(error);
                }
                
                return null;
            }
        },

        /**
         * Example usage of error handling
         */
        loadDataWithErrorHandling: function() {
            this._safeExecute(
                async () => {
                    // Risky operation
                    const data = await this._dataService.loadComplexData();
                    this._processData(data);
                    return data;
                },
                (error) => {
                    // Fallback behavior
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log("Loading from cache due to error:", error);
                    return this._loadFromCache();
                }
            );
        }
    });
});