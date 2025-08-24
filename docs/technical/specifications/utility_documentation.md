# A2A Platform Utility Documentation

This comprehensive guide documents all utility functions in the A2A platform with detailed examples, best practices, and real-world scenarios.

## Table of Contents

1. [GuidedTourManager](#guidedtourmanager)
2. [HelpProvider](#helpprovider)
3. [NotificationService](#notificationservice)
4. [Best Practices](#best-practices)
5. [Error Handling](#error-handling)
6. [Performance Considerations](#performance-considerations)

---

## GuidedTourManager

The `GuidedTourManager` is a UI5-based utility for creating interactive guided tours in the A2A Developer Portal. It helps onboard users and showcase features through step-by-step walkthroughs.

### Location
`app/a2a/developer_portal/static/utils/GuidedTourManager.js`

### Import and Initialization

```javascript
// Import in your controller
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "a2a/developer/portal/utils/GuidedTourManager"
], function (Controller, GuidedTourManager) {
    "use strict";
    
    return Controller.extend("your.namespace.Controller", {
        onInit: function() {
            // Create an instance of GuidedTourManager
            this._tourManager = new GuidedTourManager();
        }
    });
});
```

### Basic Usage Examples

#### Example 1: Simple Feature Tour

```javascript
// Create a basic tour configuration
const featureTour = {
    id: "feature-introduction",
    steps: [
        {
            target: "createProjectButton",
            title: "Create New Project",
            content: "Click here to start a new A2A project. Projects organize your agents and workflows.",
            placement: "bottom"
        },
        {
            target: "projectList",
            title: "Your Projects",
            content: "All your projects appear here. You can filter, sort, and search through them.",
            placement: "right"
        },
        {
            target: "helpButton",
            title: "Need Help?",
            content: "Click the help button anytime to access documentation and support.",
            placement: "left"
        }
    ]
};

// Start the tour
this._tourManager.startTour(featureTour, this.getView(), function() {
    console.log("Tour completed!");
    // Save user preference that tour was completed
    localStorage.setItem("featureTourCompleted", "true");
});
```

#### Example 2: Complex Workflow Tour with Conditional Steps

```javascript
// Advanced tour with dynamic steps based on user role
const workflowTour = {
    id: "workflow-creation",
    steps: this._buildWorkflowSteps()
};

_buildWorkflowSteps: function() {
    const steps = [
        {
            target: "workflowDesigner",
            title: "Workflow Designer",
            content: "This is where you design your A2A workflows using BPMN notation.",
            placement: "auto"
        }
    ];
    
    // Add role-specific steps
    const userRole = this.getUserRole();
    
    if (userRole === "developer") {
        steps.push({
            target: "codeEditorButton",
            title: "Code Editor",
            content: "As a developer, you can switch to code view to edit the workflow directly.",
            placement: "bottom"
        });
    }
    
    if (userRole === "admin") {
        steps.push({
            target: "deploymentSettings",
            title: "Deployment Configuration",
            content: "Configure deployment settings and environment variables here.",
            placement: "top"
        });
    }
    
    steps.push({
        target: "saveButton",
        title: "Save Your Work",
        content: "Don't forget to save your workflow regularly. Auto-save is enabled every 5 minutes.",
        placement: "left"
    });
    
    return steps;
}

// Start tour with completion callback
this._tourManager.startTour(workflowTour, this.getView(), function() {
    // Track tour completion
    this._analyticsService.trackEvent("tour_completed", {
        tourId: "workflow-creation",
        stepsCompleted: this._tourManager.getTotalSteps()
    });
}.bind(this));
```

#### Example 3: Progressive Tour with Validation

```javascript
// Tour that validates user actions before proceeding
const agentCreationTour = {
    id: "agent-creation-guided",
    steps: [
        {
            target: "agentNameInput",
            title: "Name Your Agent",
            content: "Enter a descriptive name for your agent. Use lowercase with hyphens (e.g., data-processor).",
            placement: "bottom",
            validation: function(oControl) {
                const value = oControl.getValue();
                return value && /^[a-z0-9-]+$/.test(value);
            },
            validationMessage: "Please enter a valid agent name (lowercase, hyphens only)"
        },
        {
            target: "agentTypeSelect",
            title: "Select Agent Type",
            content: "Choose the type of agent based on your use case.",
            placement: "bottom",
            beforeShow: function() {
                // Highlight available options
                this.byId("agentTypeSelect").open();
            }.bind(this)
        },
        {
            target: "skillsMultiSelect",
            title: "Configure Skills",
            content: "Select the skills your agent needs. You can add custom skills later.",
            placement: "right",
            minSelections: 1,
            validation: function(oControl) {
                return oControl.getSelectedItems().length > 0;
            }
        }
    ]
};

// Enhanced tour with step validation
const startValidatedTour = function() {
    let currentStep = 0;
    
    const validateAndProceed = function() {
        const step = agentCreationTour.steps[currentStep];
        if (step.validation) {
            const control = this.byId(step.target);
            if (!step.validation(control)) {
                sap.m.MessageToast.show(step.validationMessage || "Please complete this step");
                return false;
            }
        }
        return true;
    }.bind(this);
    
    // Override next step behavior
    const originalNextStep = this._tourManager.nextStep.bind(this._tourManager);
    this._tourManager.nextStep = function() {
        if (validateAndProceed()) {
            originalNextStep();
        }
    };
    
    this._tourManager.startTour(agentCreationTour, this.getView(), function() {
        sap.m.MessageToast.show("Agent creation tour completed!");
    });
}.bind(this);
```

#### Example 4: Resumable Tour with Progress Tracking

```javascript
// Tour that can be resumed from where user left off
const comprehensiveTour = {
    id: "comprehensive-platform-tour",
    resumable: true,
    steps: [
        // ... many steps
    ]
};

// Load tour progress
const loadTourProgress = function(tourId) {
    const savedProgress = localStorage.getItem(`tour_progress_${tourId}`);
    return savedProgress ? JSON.parse(savedProgress) : null;
};

// Save tour progress
const saveTourProgress = function(tourId, stepIndex) {
    localStorage.setItem(`tour_progress_${tourId}`, JSON.stringify({
        stepIndex: stepIndex,
        timestamp: new Date().toISOString()
    }));
};

// Start or resume tour
const startResumableTour = function() {
    const progress = loadTourProgress(comprehensiveTour.id);
    
    if (progress && progress.stepIndex > 0) {
        sap.m.MessageBox.confirm("Resume tour from where you left off?", {
            onClose: function(oAction) {
                if (oAction === sap.m.MessageBox.Action.OK) {
                    this._tourManager._iCurrentStep = progress.stepIndex - 1;
                    this._tourManager.startTour(comprehensiveTour, this.getView(), onTourComplete);
                } else {
                    this._tourManager.startTour(comprehensiveTour, this.getView(), onTourComplete);
                }
            }.bind(this)
        });
    } else {
        this._tourManager.startTour(comprehensiveTour, this.getView(), onTourComplete);
    }
    
    // Track progress
    const originalShowStep = this._tourManager._showStep.bind(this._tourManager);
    this._tourManager._showStep = function(stepIndex) {
        originalShowStep(stepIndex);
        saveTourProgress(comprehensiveTour.id, stepIndex);
    };
}.bind(this);

const onTourComplete = function() {
    localStorage.removeItem(`tour_progress_${comprehensiveTour.id}`);
    this._showTourCompletionReward();
};
```

### Advanced Features

#### Custom Styling and Animation

```javascript
// Add custom CSS for enhanced tour experience
const addTourStyles = function() {
    const styleContent = `
        <style id="customTourStyles">
            .guidedTourHighlight {
                animation: pulse 2s infinite, glow 1.5s ease-in-out infinite alternate;
            }
            
            @keyframes glow {
                from { box-shadow: 0 0 10px -10px #0070f2; }
                to { box-shadow: 0 0 20px 10px #0070f2; }
            }
            
            .guidedTourPopover {
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .tourStepNumber {
                background: #0070f2;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-right: 8px;
            }
        </style>
    `;
    
    jQuery("head").append(styleContent);
};

// Initialize styles when tour starts
this._tourManager.startTour = function(oTourConfig, oView, fnEndCallback) {
    addTourStyles();
    // Call original startTour
    GuidedTourManager.prototype.startTour.call(this, oTourConfig, oView, function() {
        jQuery("#customTourStyles").remove();
        if (fnEndCallback) fnEndCallback();
    });
};
```

#### Interactive Elements During Tour

```javascript
// Tour with interactive elements
const interactiveTour = {
    id: "interactive-feature-tour",
    steps: [
        {
            target: "searchInput",
            title: "Try It Out!",
            content: "Type 'agent' in the search box to see live results.",
            placement: "bottom",
            interactive: true,
            waitForAction: "input",
            nextTrigger: function(oControl) {
                return oControl.getValue().toLowerCase().includes("agent");
            }
        },
        {
            target: "filterButton",
            title: "Filter Results",
            content: "Click the filter button to see filtering options.",
            placement: "left",
            interactive: true,
            beforeShow: function() {
                // Enable the button during tour
                this.byId("filterButton").setEnabled(true);
            }.bind(this),
            afterHide: function() {
                // Restore original state
                this.byId("filterButton").setEnabled(false);
            }.bind(this)
        }
    ]
};

// Handle interactive steps
const handleInteractiveTour = function() {
    this._tourManager.startTour(interactiveTour, this.getView(), function() {
        console.log("Interactive tour completed!");
    });
    
    // Monitor user interactions
    interactiveTour.steps.forEach(function(step, index) {
        if (step.interactive && step.waitForAction) {
            const control = this.byId(step.target);
            
            control.attachEvent(step.waitForAction, function() {
                if (step.nextTrigger && step.nextTrigger(control)) {
                    this._tourManager.nextStep();
                }
            }.bind(this));
        }
    }.bind(this));
}.bind(this);
```

---

## HelpProvider

The `HelpProvider` is a comprehensive help system that provides contextual help, tooltips, and documentation access throughout the A2A platform.

### Location
`app/a2a/developer_portal/static/utils/HelpProvider.js`

### Basic Usage

#### Example 1: Simple Tooltip Help

```javascript
// In your controller
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "a2a/portal/utils/HelpProvider"
], function (Controller, HelpProvider) {
    "use strict";
    
    return Controller.extend("your.namespace.Controller", {
        onInit: function() {
            // HelpProvider is a singleton
            this._helpProvider = HelpProvider;
            
            // Enable help for specific controls
            const createButton = this.byId("createButton");
            this._helpProvider.enableHelp(createButton, "agentBuilder.createAgent");
        }
    });
});
```

#### Example 2: Complex Help Configuration

```javascript
// Configure help for an entire form
const configureFormHelp = function() {
    const formElements = [
        {
            control: "agentNameInput",
            helpKey: "agentBuilder.agentName",
            options: {
                showIcon: true,
                placement: "Right"
            }
        },
        {
            control: "agentTypeSelect",
            helpKey: "agentBuilder.agentType",
            options: {
                showIcon: true,
                placement: "Bottom",
                trigger: "click"
            }
        },
        {
            control: "deploymentEnvSelect",
            helpKey: "agentBuilder.deployment",
            options: {
                showIcon: true,
                placement: "Top"
            }
        }
    ];
    
    formElements.forEach(function(element) {
        const control = this.byId(element.control);
        if (control) {
            this._helpProvider.enableHelp(
                control,
                element.helpKey,
                element.options
            );
        }
    }.bind(this));
}.bind(this);

// Call during view initialization
this.getView().addEventDelegate({
    onAfterRendering: configureFormHelp
});
```

#### Example 3: Dynamic Help Content

```javascript
// Add custom help content dynamically
const addDynamicHelp = function() {
    // Extend help content for specific user scenarios
    const customHelp = {
        "customWorkflow": {
            "dataMapping": {
                title: "Data Mapping Configuration",
                content: "Map data fields between source and target systems.",
                detailedHelp: `Data mapping allows you to:
• Transform data types
• Apply business rules
• Handle null values
• Set default values`,
                tips: [
                    "Use JSONPath for complex mappings",
                    "Test mappings with sample data",
                    "Save mapping templates for reuse"
                ],
                examples: [
                    {
                        title: "Simple Field Mapping",
                        code: `{
    "source": "customer.firstName",
    "target": "user.name.first",
    "transform": "uppercase"
}`
                    },
                    {
                        title: "Conditional Mapping",
                        code: `{
    "source": "order.status",
    "target": "workflow.state",
    "condition": "source === 'PAID'",
    "transform": "map:PAID->COMPLETED"
}`
                    }
                ],
                warnings: [
                    "Validate data types match between source and target",
                    "Handle missing fields gracefully"
                ],
                learnMoreUrl: "/docs/data-mapping-guide"
            }
        }
    };
    
    // Merge with existing help content
    Object.assign(this._helpProvider._helpContent, customHelp);
}.bind(this);

// Use the dynamic help
const showDataMappingHelp = function() {
    const mappingControl = this.byId("dataMappingConfig");
    this._helpProvider.enableHelp(
        mappingControl,
        "customWorkflow.dataMapping",
        {
            showIcon: true,
            placement: "Auto"
        }
    );
}.bind(this);
```

#### Example 4: Context-Sensitive Help

```javascript
// Provide different help based on user context
const contextSensitiveHelp = function() {
    const userLevel = this.getUserExperienceLevel(); // beginner, intermediate, expert
    const currentView = this.getCurrentViewName();
    
    // Build context-specific help key
    const buildHelpKey = function(baseKey) {
        return `${currentView}.${baseKey}.${userLevel}`;
    };
    
    // Add context-specific help content
    const contextHelp = {
        "agentBuilder.createAgent.beginner": {
            title: "Creating Your First Agent",
            content: "Let's create your first A2A agent step by step.",
            detailedHelp: `An agent is an autonomous component that can:
• Process data according to rules
• Communicate with other agents
• React to events
• Execute tasks on schedule

Start with a simple data processing agent to learn the basics.`,
            tips: [
                "Use the 'Basic Data Processor' template",
                "Follow the setup wizard",
                "Test with sample data first"
            ],
            videoUrl: "/videos/beginner/first-agent"
        },
        "agentBuilder.createAgent.expert": {
            title: "Advanced Agent Configuration",
            content: "Configure advanced agent features and optimizations.",
            detailedHelp: `Advanced options include:
• Custom skill development
• Performance tuning
• Complex event processing
• Multi-agent orchestration`,
            tips: [
                "Use environment variables for configuration",
                "Implement circuit breakers for resilience",
                "Monitor agent performance metrics"
            ],
            codeExamples: [
                {
                    title: "Custom Skill Implementation",
                    language: "javascript",
                    code: `class CustomDataProcessor extends BaseSkill {
    async execute(context, data) {
        // Implement parallel processing
        const results = await Promise.all(
            data.items.map(item => this.processItem(item))
        );
        
        return {
            processed: results.length,
            results: results
        };
    }
}`
                }
            ]
        }
    };
    
    // Apply context-sensitive help
    Object.assign(this._helpProvider._helpContent, contextHelp);
    
    // Enable help with context
    const enableContextHelp = function(controlId, baseHelpKey) {
        const control = this.byId(controlId);
        const contextKey = buildHelpKey(baseHelpKey);
        
        this._helpProvider.enableHelp(control, contextKey, {
            showIcon: true,
            placement: "Auto"
        });
    }.bind(this);
    
    // Apply to all relevant controls
    enableContextHelp("createAgentBtn", "createAgent");
}.bind(this);
```

#### Example 5: Help Search and Navigation

```javascript
// Implement help search functionality
const implementHelpSearch = function() {
    const searchHelp = function(query) {
        const results = [];
        const searchInObject = function(obj, path = []) {
            Object.keys(obj).forEach(key => {
                const value = obj[key];
                const currentPath = [...path, key];
                
                if (typeof value === 'object' && !Array.isArray(value)) {
                    searchInObject(value, currentPath);
                } else if (typeof value === 'string') {
                    if (value.toLowerCase().includes(query.toLowerCase())) {
                        results.push({
                            path: currentPath.join('.'),
                            content: value,
                            title: obj.title || key
                        });
                    }
                }
            });
        };
        
        searchInObject(this._helpProvider._helpContent);
        return results;
    }.bind(this);
    
    // Create search UI
    const createHelpSearchDialog = function() {
        const dialog = new sap.m.Dialog({
            title: "Search Help",
            contentWidth: "600px",
            content: [
                new sap.m.SearchField({
                    placeholder: "Search for help topics...",
                    liveChange: function(oEvent) {
                        const query = oEvent.getParameter("newValue");
                        const results = searchHelp(query);
                        updateSearchResults(results);
                    }
                }),
                new sap.m.List({
                    id: "helpSearchResults",
                    mode: "SingleSelectMaster",
                    itemPress: function(oEvent) {
                        const item = oEvent.getParameter("listItem");
                        const helpKey = item.data("helpKey");
                        dialog.close();
                        this._helpProvider.showDetailedHelp(helpKey, this.byId("helpButton"));
                    }.bind(this)
                })
            ],
            buttons: [
                new sap.m.Button({
                    text: "Close",
                    press: function() {
                        dialog.close();
                    }
                })
            ]
        });
        
        return dialog;
    }.bind(this);
    
    const updateSearchResults = function(results) {
        const list = sap.ui.getCore().byId("helpSearchResults");
        list.removeAllItems();
        
        results.slice(0, 10).forEach(result => {
            list.addItem(new sap.m.StandardListItem({
                title: result.title,
                description: result.content.substring(0, 100) + "...",
                type: "Navigation"
            }).data("helpKey", result.path));
        });
    };
    
    // Show search dialog
    this.helpSearchDialog = createHelpSearchDialog();
    this.helpSearchDialog.open();
}.bind(this);
```

### Smart Tooltips

```javascript
// Automatically apply smart tooltips to a view
const applySmartTooltips = function() {
    // Enable for entire view
    this._helpProvider.enableSmartTooltips(this.getView());
    
    // Custom tooltip mappings for specific scenarios
    const customMappings = {
        "sap.m.Button": {
            "idBatchProcessBtn": "Process multiple items at once",
            "idExportBtn": "Export data in various formats (CSV, JSON, XML)",
            "idScheduleBtn": "Schedule this task to run automatically"
        },
        "sap.m.Input": {
            "Email": "Enter a valid email address (e.g., user@example.com)",
            "APIKey": "Your API key (found in Settings > API Access)",
            "CronExpression": "Cron expression (e.g., '0 0 * * *' for daily at midnight)"
        },
        "sap.m.DatePicker": {
            "StartDate": "Select the start date (cannot be in the past)",
            "EndDate": "Select the end date (must be after start date)"
        }
    };
    
    // Apply custom mappings
    const applyCustomTooltips = function(container) {
        const controls = container.findAggregatedObjects(true);
        
        controls.forEach(control => {
            const controlType = control.getMetadata().getName();
            const controlId = control.getId();
            
            if (customMappings[controlType]) {
                Object.keys(customMappings[controlType]).forEach(pattern => {
                    if (controlId.includes(pattern)) {
                        control.setTooltip(customMappings[controlType][pattern]);
                        
                        // Add visual indicator for fields with help
                        control.addStyleClass("hasHelpTooltip");
                    }
                });
            }
        });
    };
    
    applyCustomTooltips(this.getView());
}.bind(this);
```

---

## NotificationService

The `NotificationService` manages notifications, alerts, and system messages in the A2A platform.

### Location
`app/a2a/developer_portal/static/utils/NotificationService.js`

### Basic Usage

#### Example 1: Simple Notification Management

```javascript
// In your controller
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "a2a/portal/services/NotificationService"
], function (Controller, NotificationService) {
    "use strict";
    
    return Controller.extend("your.namespace.Controller", {
        onInit: function() {
            // Create notification service instance
            this._notificationService = new NotificationService();
            
            // Set the model to the view
            this.getView().setModel(
                this._notificationService.getModel(),
                "notifications"
            );
            
            // Load initial notifications
            this._notificationService.loadNotifications(true);
        },
        
        onExit: function() {
            // Clean up
            this._notificationService.destroy();
        }
    });
});
```

#### Example 2: Real-time Notification Handling

```javascript
// Implement real-time notifications with WebSocket
const setupRealtimeNotifications = function() {
    // WebSocket connection for real-time updates
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/notifications/ws`;
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = function(event) {
        const notification = JSON.parse(event.data);
        
        // Add to notification model
        const model = this._notificationService.getModel();
        const notifications = model.getProperty("/notifications");
        
        // Add new notification to the beginning
        notifications.unshift(notification);
        model.setProperty("/notifications", notifications);
        
        // Update stats
        if (notification.status === "unread") {
            const unreadCount = model.getProperty("/stats/unread");
            model.setProperty("/stats/unread", unreadCount + 1);
        }
        
        // Show toast for important notifications
        if (notification.priority === "high" || notification.priority === "critical") {
            sap.m.MessageToast.show(notification.title, {
                duration: 5000,
                width: "300px",
                at: "end top",
                offset: "-10 10"
            });
        }
        
        // Play notification sound for critical alerts
        if (notification.priority === "critical") {
            this._playNotificationSound();
        }
    }.bind(this);
    
    ws.onerror = function(error) {
        console.error("WebSocket error:", error);
        // Fallback to polling
        this._notificationService.startAutoRefresh();
    }.bind(this);
    
    // Store WebSocket reference for cleanup
    this._notificationWebSocket = ws;
}.bind(this);

// Play notification sound
const _playNotificationSound = function() {
    const audio = new Audio("/sounds/notification.mp3");
    audio.volume = 0.5;
    audio.play().catch(e => console.log("Could not play notification sound:", e));
};
```

#### Example 3: Notification Filtering and Grouping

```javascript
// Advanced notification management with filtering
const setupNotificationFilters = function() {
    // Create filter bar
    const filterBar = new sap.m.Bar({
        contentLeft: [
            new sap.m.Select({
                items: [
                    new sap.ui.core.Item({ key: "all", text: "All Notifications" }),
                    new sap.ui.core.Item({ key: "unread", text: "Unread" }),
                    new sap.ui.core.Item({ key: "read", text: "Read" }),
                    new sap.ui.core.Item({ key: "dismissed", text: "Dismissed" })
                ],
                change: function(oEvent) {
                    const status = oEvent.getParameter("selectedItem").getKey();
                    this._applyNotificationFilter("status", status === "all" ? null : status);
                }.bind(this)
            }),
            new sap.m.Select({
                items: [
                    new sap.ui.core.Item({ key: "all", text: "All Types" }),
                    new sap.ui.core.Item({ key: "agent", text: "Agent Alerts" }),
                    new sap.ui.core.Item({ key: "workflow", text: "Workflow Updates" }),
                    new sap.ui.core.Item({ key: "system", text: "System Messages" }),
                    new sap.ui.core.Item({ key: "security", text: "Security Alerts" })
                ],
                change: function(oEvent) {
                    const type = oEvent.getParameter("selectedItem").getKey();
                    this._applyNotificationFilter("type", type === "all" ? null : type);
                }.bind(this)
            })
        ],
        contentRight: [
            new sap.m.Button({
                text: "Mark All Read",
                press: this._markAllAsRead.bind(this)
            }),
            new sap.m.Button({
                text: "Clear All",
                press: this._clearAllNotifications.bind(this)
            })
        ]
    });
    
    return filterBar;
}.bind(this);

// Apply filters
const _applyNotificationFilter = function(filterType, filterValue) {
    const currentFilters = this._notificationService.getModel().getProperty("/filters");
    currentFilters[filterType] = filterValue;
    
    this._notificationService.setFilters(currentFilters);
}.bind(this);

// Group notifications by date
const groupNotificationsByDate = function(notifications) {
    const groups = {
        today: [],
        yesterday: [],
        thisWeek: [],
        older: []
    };
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    
    notifications.forEach(notification => {
        const notifDate = new Date(notification.created_at);
        notifDate.setHours(0, 0, 0, 0);
        
        if (notifDate.getTime() === today.getTime()) {
            groups.today.push(notification);
        } else if (notifDate.getTime() === yesterday.getTime()) {
            groups.yesterday.push(notification);
        } else if (notifDate > weekAgo) {
            groups.thisWeek.push(notification);
        } else {
            groups.older.push(notification);
        }
    });
    
    return groups;
};
```

#### Example 4: Custom Notification Actions

```javascript
// Implement custom actions for notifications
const createNotificationItem = function(notification) {
    const item = new sap.m.NotificationListItem({
        title: notification.title,
        description: notification.message,
        datetime: notification.created_at,
        unread: notification.status === "unread",
        priority: this._mapPriority(notification.priority),
        showCloseButton: true,
        close: function() {
            this._notificationService.dismissNotification(notification.id);
        }.bind(this),
        press: function() {
            // Mark as read
            if (notification.status === "unread") {
                this._notificationService.markAsRead(notification.id);
            }
            
            // Handle notification-specific actions
            this._handleNotificationAction(notification);
        }.bind(this)
    });
    
    // Add custom actions based on notification type
    if (notification.type === "agent") {
        item.addAction(new sap.m.Button({
            text: "View Agent",
            type: "Transparent",
            press: function() {
                this._navigateToAgent(notification.metadata.agentId);
            }.bind(this)
        }));
    }
    
    if (notification.type === "workflow" && notification.metadata.requiresApproval) {
        item.addAction(new sap.m.Button({
            text: "Approve",
            type: "Accept",
            press: function() {
                this._approveWorkflow(notification.metadata.workflowId);
            }.bind(this)
        }));
        
        item.addAction(new sap.m.Button({
            text: "Reject",
            type: "Reject",
            press: function() {
                this._rejectWorkflow(notification.metadata.workflowId);
            }.bind(this)
        }));
    }
    
    if (notification.actionUrl) {
        item.addAction(new sap.m.Button({
            text: notification.actionText || "Take Action",
            type: "Emphasized",
            press: function() {
                window.open(notification.actionUrl, "_blank");
            }
        }));
    }
    
    return item;
}.bind(this);

// Handle notification-specific navigation
const _handleNotificationAction = function(notification) {
    switch (notification.type) {
        case "agent":
            if (notification.subtype === "error") {
                this._showAgentErrorDetails(notification);
            } else {
                this._navigateToAgent(notification.metadata.agentId);
            }
            break;
            
        case "workflow":
            this._navigateToWorkflow(notification.metadata.workflowId);
            break;
            
        case "security":
            this._showSecurityAlert(notification);
            break;
            
        default:
            // Show details in a dialog
            this._showNotificationDetails(notification);
    }
}.bind(this);
```

#### Example 5: Notification Preferences and Settings

```javascript
// User notification preferences
const NotificationPreferences = {
    init: function() {
        this.preferences = {
            channels: {
                inApp: true,
                email: false,
                sms: false,
                push: false
            },
            types: {
                agent: {
                    enabled: true,
                    priority: ["critical", "high"]
                },
                workflow: {
                    enabled: true,
                    priority: ["critical", "high", "medium"]
                },
                system: {
                    enabled: true,
                    priority: ["critical"]
                },
                security: {
                    enabled: true,
                    priority: ["critical", "high", "medium", "low"]
                }
            },
            quietHours: {
                enabled: false,
                start: "22:00",
                end: "08:00",
                timezone: "UTC"
            },
            digest: {
                enabled: true,
                frequency: "daily",
                time: "09:00"
            }
        };
        
        this.loadPreferences();
    },
    
    loadPreferences: function() {
        // Load from backend or local storage
        const saved = localStorage.getItem("notificationPreferences");
        if (saved) {
            this.preferences = JSON.parse(saved);
        }
    },
    
    savePreferences: function() {
        localStorage.setItem("notificationPreferences", JSON.stringify(this.preferences));
        
        // Sync with backend
        jQuery.ajax({
            url: "/api/user/notification-preferences",
            method: "PUT",
            data: JSON.stringify(this.preferences),
            contentType: "application/json"
        });
    },
    
    shouldNotify: function(notification) {
        // Check if notification should be shown based on preferences
        const typePrefs = this.preferences.types[notification.type];
        
        if (!typePrefs || !typePrefs.enabled) {
            return false;
        }
        
        if (!typePrefs.priority.includes(notification.priority)) {
            return false;
        }
        
        // Check quiet hours
        if (this.preferences.quietHours.enabled) {
            const now = new Date();
            const start = this._parseTime(this.preferences.quietHours.start);
            const end = this._parseTime(this.preferences.quietHours.end);
            
            if (this._isInQuietHours(now, start, end)) {
                return false;
            }
        }
        
        return true;
    },
    
    _parseTime: function(timeStr) {
        const [hours, minutes] = timeStr.split(':').map(Number);
        const date = new Date();
        date.setHours(hours, minutes, 0, 0);
        return date;
    },
    
    _isInQuietHours: function(now, start, end) {
        const currentMinutes = now.getHours() * 60 + now.getMinutes();
        const startMinutes = start.getHours() * 60 + start.getMinutes();
        const endMinutes = end.getHours() * 60 + end.getMinutes();
        
        if (startMinutes <= endMinutes) {
            return currentMinutes >= startMinutes && currentMinutes < endMinutes;
        } else {
            return currentMinutes >= startMinutes || currentMinutes < endMinutes;
        }
    }
};

// Create preferences UI
const createPreferencesDialog = function() {
    const dialog = new sap.m.Dialog({
        title: "Notification Preferences",
        contentWidth: "500px",
        content: [
            new sap.m.VBox({
                items: [
                    new sap.m.Title({ text: "Notification Channels" }),
                    new sap.m.CheckBox({
                        text: "In-App Notifications",
                        selected: "{/channels/inApp}"
                    }),
                    new sap.m.CheckBox({
                        text: "Email Notifications",
                        selected: "{/channels/email}"
                    }),
                    new sap.m.CheckBox({
                        text: "SMS Notifications",
                        selected: "{/channels/sms}"
                    }),
                    new sap.m.CheckBox({
                        text: "Push Notifications",
                        selected: "{/channels/push}"
                    }),
                    
                    new sap.m.Title({ 
                        text: "Notification Types",
                        class: "sapUiMediumMarginTop"
                    }),
                    // ... Add controls for each notification type
                    
                    new sap.m.Title({ 
                        text: "Quiet Hours",
                        class: "sapUiMediumMarginTop"
                    }),
                    new sap.m.Switch({
                        state: "{/quietHours/enabled}",
                        customTextOn: "On",
                        customTextOff: "Off"
                    }),
                    new sap.m.TimePicker({
                        value: "{/quietHours/start}",
                        displayFormat: "HH:mm",
                        enabled: "{/quietHours/enabled}"
                    }),
                    new sap.m.TimePicker({
                        value: "{/quietHours/end}",
                        displayFormat: "HH:mm",
                        enabled: "{/quietHours/enabled}"
                    })
                ]
            })
        ],
        buttons: [
            new sap.m.Button({
                text: "Save",
                type: "Emphasized",
                press: function() {
                    NotificationPreferences.savePreferences();
                    dialog.close();
                    sap.m.MessageToast.show("Preferences saved");
                }
            }),
            new sap.m.Button({
                text: "Cancel",
                press: function() {
                    dialog.close();
                }
            })
        ]
    });
    
    // Bind preferences model
    const prefsModel = new sap.ui.model.json.JSONModel(NotificationPreferences.preferences);
    dialog.setModel(prefsModel);
    
    return dialog;
};
```

---

## Best Practices

### 1. Error Handling

```javascript
// Comprehensive error handling for utilities
const safeExecute = function(fn, context, fallback) {
    try {
        return fn.call(context);
    } catch (error) {
        console.error("Utility execution error:", error);
        
        // Log to monitoring service
        if (window.monitoringService) {
            window.monitoringService.logError({
                component: "utilities",
                error: error.message,
                stack: error.stack,
                context: context
            });
        }
        
        // Show user-friendly message
        sap.m.MessageToast.show("An error occurred. Please try again.");
        
        // Return fallback value
        return typeof fallback === 'function' ? fallback() : fallback;
    }
};

// Usage example
const loadData = function() {
    return safeExecute(function() {
        // Potentially error-prone operation
        return this._dataService.loadComplexData();
    }, this, []);  // Return empty array as fallback
};
```

### 2. Performance Optimization

```javascript
// Debounce utility for performance
const debounce = function(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        
        if (callNow) func.apply(context, args);
    };
};

// Throttle utility
const throttle = function(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

// Usage in notification service
this._notificationService.loadNotifications = debounce(
    this._notificationService.loadNotifications.bind(this._notificationService),
    500
);

// Usage in help search
const searchHelp = throttle(function(query) {
    // Search implementation
}, 300);
```

### 3. Memory Management

```javascript
// Proper cleanup to prevent memory leaks
const UtilityCleanup = {
    cleanupHandlers: [],
    
    register: function(cleanup) {
        this.cleanupHandlers.push(cleanup);
    },
    
    executeCleanup: function() {
        this.cleanupHandlers.forEach(handler => {
            try {
                handler();
            } catch (e) {
                console.error("Cleanup error:", e);
            }
        });
        this.cleanupHandlers = [];
    }
};

// Register cleanup in controller
onInit: function() {
    // Setup utilities
    this._tourManager = new GuidedTourManager();
    this._notificationService = new NotificationService();
    
    // Register cleanup
    UtilityCleanup.register(() => {
        this._tourManager.endTour();
        this._notificationService.destroy();
        
        if (this._notificationWebSocket) {
            this._notificationWebSocket.close();
        }
    });
},

onExit: function() {
    UtilityCleanup.executeCleanup();
}
```

### 4. Accessibility

```javascript
// Ensure utilities are accessible
const AccessibilityEnhancer = {
    enhanceTour: function(tourManager) {
        // Add ARIA labels
        const originalShowStep = tourManager._showStep;
        tourManager._showStep = function(stepIndex) {
            originalShowStep.call(this, stepIndex);
            
            const step = this._oCurrentTour.steps[stepIndex];
            const popover = this._oPopover;
            
            if (popover) {
                popover.addAriaLabelledBy(popover.getId() + "-title");
                popover.setInitialFocus(popover.getId() + "-content");
                
                // Announce step to screen readers
                const announcement = `Step ${stepIndex + 1} of ${this._oCurrentTour.steps.length}. ${step.title}`;
                this._announceToScreenReader(announcement);
            }
        };
        
        tourManager._announceToScreenReader = function(text) {
            const announcer = document.getElementById("tourAnnouncer") || 
                             this._createAnnouncer();
            announcer.textContent = text;
        };
        
        tourManager._createAnnouncer = function() {
            const announcer = document.createElement("div");
            announcer.id = "tourAnnouncer";
            announcer.setAttribute("role", "status");
            announcer.setAttribute("aria-live", "polite");
            announcer.style.position = "absolute";
            announcer.style.left = "-10000px";
            document.body.appendChild(announcer);
            return announcer;
        };
    },
    
    enhanceNotifications: function(notificationService) {
        // Add keyboard navigation
        const originalCreateItem = notificationService.createNotificationItem;
        notificationService.createNotificationItem = function(notification) {
            const item = originalCreateItem.call(this, notification);
            
            // Add keyboard handler
            item.addEventDelegate({
                onkeydown: function(oEvent) {
                    if (oEvent.key === "Enter" || oEvent.key === " ") {
                        oEvent.preventDefault();
                        item.firePress();
                    } else if (oEvent.key === "Delete") {
                        item.fireClose();
                    }
                }
            });
            
            // Add ARIA attributes
            item.addCustomData(new sap.ui.core.CustomData({
                key: "aria-label",
                value: `${notification.priority} priority notification: ${notification.title}`
            }));
            
            return item;
        };
    }
};
```

---

## Performance Considerations

### 1. Lazy Loading

```javascript
// Lazy load utilities only when needed
const UtilityLoader = {
    _cache: {},
    
    load: function(utilityName) {
        if (this._cache[utilityName]) {
            return Promise.resolve(this._cache[utilityName]);
        }
        
        return new Promise((resolve, reject) => {
            sap.ui.require([`a2a/portal/utils/${utilityName}`], (Utility) => {
                this._cache[utilityName] = Utility;
                resolve(Utility);
            }, reject);
        });
    },
    
    loadMultiple: function(utilities) {
        return Promise.all(utilities.map(name => this.load(name)));
    }
};

// Usage
UtilityLoader.load("GuidedTourManager").then(TourManager => {
    const tour = new TourManager();
    tour.startTour(config, view, callback);
});
```

### 2. Resource Pooling

```javascript
// Pool frequently used resources
const ResourcePool = {
    _pools: {},
    
    create: function(type, factory, maxSize = 10) {
        this._pools[type] = {
            available: [],
            inUse: new Set(),
            factory: factory,
            maxSize: maxSize
        };
    },
    
    acquire: function(type) {
        const pool = this._pools[type];
        if (!pool) throw new Error(`Pool ${type} not found`);
        
        let resource;
        if (pool.available.length > 0) {
            resource = pool.available.pop();
        } else if (pool.inUse.size < pool.maxSize) {
            resource = pool.factory();
        } else {
            throw new Error(`Pool ${type} exhausted`);
        }
        
        pool.inUse.add(resource);
        return resource;
    },
    
    release: function(type, resource) {
        const pool = this._pools[type];
        if (!pool || !pool.inUse.has(resource)) return;
        
        pool.inUse.delete(resource);
        pool.available.push(resource);
    }
};

// Create popover pool for help system
ResourcePool.create("helpPopover", () => {
    return new sap.m.Popover({
        showCloseButton: true,
        contentWidth: "400px"
    });
});
```

### 3. Event Optimization

```javascript
// Optimize event handling
const EventOptimizer = {
    // Use event delegation instead of individual handlers
    setupDelegation: function(container, eventMap) {
        Object.keys(eventMap).forEach(selector => {
            const events = eventMap[selector];
            
            Object.keys(events).forEach(eventName => {
                container.addEventDelegate({
                    [eventName]: function(oEvent) {
                        const target = oEvent.target;
                        const control = sap.ui.getCore().byId(target.id);
                        
                        if (control && target.matches(selector)) {
                            events[eventName].call(this, oEvent, control);
                        }
                    }
                }, this);
            });
        });
    },
    
    // Batch DOM updates
    batchUpdates: function(updates) {
        requestAnimationFrame(() => {
            updates.forEach(update => update());
        });
    }
};

// Usage
EventOptimizer.setupDelegation(this.getView(), {
    ".helpIcon": {
        onclick: function(oEvent, oControl) {
            this._helpProvider.showDetailedHelp(
                oControl.data("helpKey"),
                oControl
            );
        }
    },
    ".notificationItem": {
        onclick: function(oEvent, oControl) {
            this._handleNotificationClick(oControl.data("notificationId"));
        }
    }
});
```

## Conclusion

This documentation provides comprehensive examples and best practices for using the utility functions in the A2A platform. By following these patterns and examples, developers can effectively implement tours, help systems, and notifications in their applications while maintaining high performance and accessibility standards.

Remember to:
- Always handle errors gracefully
- Clean up resources properly
- Consider accessibility requirements
- Optimize for performance
- Test across different scenarios
- Document custom extensions

For additional support or advanced use cases, refer to the platform documentation or contact the A2A development team.