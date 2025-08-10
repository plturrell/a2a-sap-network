/**
 * @fileoverview Contextual Help Provider for A2A Developer Portal
 * @module a2a/portal/utils/HelpProvider
 * @description Provides contextual help, tooltips, and user guidance throughout the application
 */

sap.ui.define([
    "sap/ui/base/Object",
    "sap/m/Popover",
    "sap/m/Text",
    "sap/m/Link",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Button",
    "sap/ui/core/Icon",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], function (BaseObject, Popover, Text, Link, VBox, HBox, Button, Icon, JSONModel, MessageToast) {
    "use strict";

    /**
     * Help Provider singleton
     * @class
     * @alias a2a.portal.utils.HelpProvider
     * @extends sap.ui.base.Object
     * @description Manages contextual help, tooltips, and user guidance for the A2A platform
     * @singleton
     */
    var HelpProvider = BaseObject.extend("a2a.portal.utils.HelpProvider", {

        /**
         * Constructor
         * @memberof a2a.portal.utils.HelpProvider
         * @constructor
         */
        constructor: function () {
            BaseObject.apply(this, arguments);
            
            /**
             * Help content repository
             * @type {Object}
             * @private
             */
            this._helpContent = {};
            
            /**
             * Active popovers
             * @type {Map<string, sap.m.Popover>}
             * @private
             */
            this._activePopovers = new Map();
            
            /**
             * Help model
             * @type {sap.ui.model.json.JSONModel}
             * @private
             */
            this._helpModel = new JSONModel({
                enabled: true,
                showTooltips: true,
                showGuidedTours: true,
                completedTours: [],
                userLevel: "beginner" // beginner, intermediate, expert
            });
            
            this._initializeHelpContent();
        },

        /**
         * Initialize help content repository
         * @memberof a2a.portal.utils.HelpProvider
         * @private
         */
        _initializeHelpContent: function () {
            // Agent Builder Help Content
            this._helpContent["agentBuilder"] = {
                "createAgent": {
                    title: "Create New Agent",
                    content: "Create a new A2A agent by selecting a template and configuring its properties.",
                    detailedHelp: "Agents are autonomous components that can process data, communicate with other agents, and perform specific tasks. Choose from reactive, proactive, or hybrid agent types.",
                    learnMoreUrl: "/docs/agents/creating-agents",
                    videoUrl: "/videos/agent-creation-tutorial",
                    relatedTopics: ["agent-templates", "agent-configuration", "agent-deployment"]
                },
                "agentTemplate": {
                    title: "Agent Templates",
                    content: "Pre-configured agent templates for common use cases.",
                    detailedHelp: "Templates provide starting configurations for:\n• Data Processing Agents\n• Integration Agents\n• Monitoring Agents\n• Custom Agents",
                    tips: [
                        "Start with a template close to your use case",
                        "Templates can be customized after creation",
                        "Custom templates can be saved for reuse"
                    ]
                },
                "deployAgent": {
                    title: "Deploy Agent",
                    content: "Deploy your agent to the selected environment.",
                    detailedHelp: "Deployment options:\n• Development - For testing\n• Staging - Pre-production validation\n• Production - Live environment",
                    warnings: ["Production deployments require approval", "Ensure all tests pass before deployment"],
                    prerequisites: ["Agent must be saved", "All required fields must be filled", "Validation must pass"]
                }
            };

            // Project Management Help Content
            this._helpContent["projects"] = {
                "createProject": {
                    title: "Create New Project",
                    content: "Create a new A2A project to organize your agents and workflows.",
                    detailedHelp: "Projects are containers for related agents, workflows, and configurations. They provide:\n• Organized workspace\n• Shared configurations\n• Team collaboration features",
                    bestPractices: [
                        "Use descriptive project names",
                        "Group related agents in the same project",
                        "Set up project-level configurations first"
                    ]
                },
                "projectSettings": {
                    title: "Project Settings",
                    content: "Configure project-wide settings and permissions.",
                    sections: {
                        "general": "Project name, description, and metadata",
                        "security": "Access control and permissions",
                        "integration": "External system connections",
                        "notifications": "Alert and notification preferences"
                    }
                }
            };

            // BPMN Designer Help Content
            this._helpContent["bpmnDesigner"] = {
                "canvas": {
                    title: "BPMN Canvas",
                    content: "Design your workflow by dragging and dropping BPMN elements.",
                    shortcuts: {
                        "Space + Drag": "Pan the canvas",
                        "Ctrl + Scroll": "Zoom in/out",
                        "Delete": "Remove selected element",
                        "Ctrl + C/V": "Copy/Paste elements"
                    },
                    tips: ["Use swim lanes to organize by agent", "Connect elements with sequence flows", "Add gateways for conditional logic"]
                },
                "elements": {
                    title: "BPMN Elements",
                    content: "Standard BPMN 2.0 elements for workflow design.",
                    elementTypes: {
                        "events": "Start, intermediate, and end events",
                        "activities": "Tasks, sub-processes, and agent calls",
                        "gateways": "Exclusive, parallel, and inclusive gateways",
                        "flows": "Sequence and message flows"
                    }
                }
            };

            // Navigation Help
            this._helpContent["navigation"] = {
                "shellBar": {
                    title: "Navigation Bar",
                    content: "Main navigation and quick actions.",
                    features: {
                        "search": "Global search across projects and agents",
                        "notifications": "System alerts and updates",
                        "profile": "User settings and preferences",
                        "help": "Access help and documentation"
                    }
                }
            };
        },

        /**
         * Enable contextual help for a control
         * @memberof a2a.portal.utils.HelpProvider
         * @public
         * @param {sap.ui.core.Control} oControl - The control to add help to
         * @param {string} sHelpKey - The help content key (dot-separated path)
         * @param {Object} [oOptions] - Additional options
         * @param {boolean} [oOptions.showIcon=true] - Show help icon next to control
         * @param {string} [oOptions.placement="Auto"] - Popover placement (Auto|Top|Bottom|Left|Right)
         * @param {string} [oOptions.trigger="hover"] - Help trigger (hover|click|focus)
         * @param {function} [oOptions.onShow] - Callback when help is shown
         * @param {function} [oOptions.onHide] - Callback when help is hidden
         * @returns {void}
         * 
         * @example
         * // Basic help tooltip
         * helpProvider.enableHelp(
         *     this.byId("submitButton"),
         *     "forms.submission.button"
         * );
         * 
         * @example
         * // Help with icon and specific placement
         * helpProvider.enableHelp(
         *     this.byId("complexField"),
         *     "advanced.dataMapping",
         *     {
         *         showIcon: true,
         *         placement: "Right",
         *         trigger: "click"
         *     }
         * );
         * 
         * @example
         * // Help with callbacks for analytics
         * helpProvider.enableHelp(
         *     this.byId("criticalAction"),
         *     "security.criticalAction",
         *     {
         *         onShow: function() {
         *             analytics.track("help_viewed", {
         *                 helpKey: "security.criticalAction",
         *                 timestamp: new Date()
         *             });
         *         },
         *         onHide: function() {
         *             console.log("Help closed");
         *         }
         *     }
         * );
         * 
         * @example
         * // Batch enable help for form fields
         * const formFields = [
         *     { id: "nameInput", help: "user.profile.name" },
         *     { id: "emailInput", help: "user.profile.email" },
         *     { id: "roleSelect", help: "user.profile.role" }
         * ];
         * 
         * formFields.forEach(field => {
         *     helpProvider.enableHelp(
         *         this.byId(field.id),
         *         field.help,
         *         { showIcon: true }
         *     );
         * });
         */
        enableHelp: function (oControl, sHelpKey, oOptions) {
            if (!oControl || !sHelpKey) {
                return;
            }

            var oDefaults = {
                showIcon: true,
                placement: "Auto",
                trigger: "hover"
            };

            var oSettings = Object.assign({}, oDefaults, oOptions);

            // Add tooltip if simple help exists
            var helpPath = sHelpKey.split(".");
            var helpContent = this._getHelpContent(helpPath);
            
            if (helpContent && helpContent.content) {
                oControl.setTooltip(helpContent.content);
            }

            // Add help icon if requested
            if (oSettings.showIcon) {
                this._addHelpIcon(oControl, sHelpKey, oSettings);
            }

            // Store help association
            if (!oControl.data("helpKey")) {
                oControl.data("helpKey", sHelpKey);
                oControl.data("helpSettings", oSettings);
            }
        },

        /**
         * Add help icon to control
         * @memberof a2a.portal.utils.HelpProvider
         * @private
         * @param {sap.ui.core.Control} oControl - The control
         * @param {string} sHelpKey - Help content key
         * @param {Object} oSettings - Settings
         */
        _addHelpIcon: function (oControl, sHelpKey, oSettings) {
            var that = this;
            
            // Create help icon
            var oHelpIcon = new Icon({
                src: "sap-icon://sys-help",
                size: "1rem",
                color: "#0070f2",
                cursor: "pointer",
                tooltip: "Click for help",
                press: function (oEvent) {
                    oEvent.stopPropagation();
                    that.showDetailedHelp(sHelpKey, oHelpIcon);
                }
            }).addStyleClass("sapUiTinyMarginBegin");

            // Try to add icon to control
            if (oControl.addEndIcon) {
                oControl.addEndIcon(oHelpIcon);
            } else if (oControl.getParent && oControl.getParent()) {
                // Add to parent container if possible
                var oParent = oControl.getParent();
                if (oParent.addContent) {
                    oParent.addContent(oHelpIcon);
                }
            }
        },

        /**
         * Show detailed help in a popover
         * @memberof a2a.portal.utils.HelpProvider
         * @public
         * @param {string} sHelpKey - Help content key (dot-separated path)
         * @param {sap.ui.core.Control} oOpenBy - Control to open popover by
         * @returns {void}
         * @fires helpOpened
         * @fires helpClosed
         * 
         * @example
         * // Show help programmatically
         * this.byId("helpButton").attachPress(function() {
         *     helpProvider.showDetailedHelp(
         *         "workflows.creation.advanced",
         *         this.byId("helpButton")
         *     );
         * }.bind(this));
         * 
         * @example
         * // Show context-sensitive help
         * showContextHelp: function() {
         *     const currentView = this.getCurrentViewName();
         *     const helpKey = this.getHelpKeyForView(currentView);
         *     
         *     if (helpKey) {
         *         helpProvider.showDetailedHelp(
         *             helpKey,
         *             this.byId("contextHelpButton")
         *         );
         *     } else {
         *         sap.m.MessageToast.show("No help available for this view");
         *     }
         * }
         * 
         * @example
         * // Show help with pre-processing
         * showEnhancedHelp: function(helpKey, control) {
         *     // Add user-specific content
         *     const userRole = this.getUserRole();
         *     const enhancedKey = `${helpKey}.${userRole}`;
         *     
         *     // Check if enhanced help exists
         *     if (helpProvider._getHelpContent(enhancedKey.split('.'))) {
         *         helpProvider.showDetailedHelp(enhancedKey, control);
         *     } else {
         *         // Fallback to generic help
         *         helpProvider.showDetailedHelp(helpKey, control);
         *     }
         * }
         */
        showDetailedHelp: function (sHelpKey, oOpenBy) {
            var helpPath = sHelpKey.split(".");
            var helpContent = this._getHelpContent(helpPath);
            
            if (!helpContent) {
                MessageToast.show("Help content not available");
                return;
            }

            // Check if popover already exists
            var existingPopover = this._activePopovers.get(sHelpKey);
            if (existingPopover && existingPopover.isOpen()) {
                existingPopover.close();
                return;
            }

            // Create popover content
            var oContent = this._createHelpPopoverContent(helpContent, sHelpKey);
            
            // Create and configure popover
            var oPopover = new Popover({
                title: helpContent.title || "Help",
                placement: "Auto",
                contentWidth: "400px",
                contentHeight: "auto",
                showCloseButton: true,
                content: oContent,
                afterClose: function () {
                    this._activePopovers.delete(sHelpKey);
                    oPopover.destroy();
                }.bind(this)
            });

            // Store and open popover
            this._activePopovers.set(sHelpKey, oPopover);
            oPopover.openBy(oOpenBy);
        },

        /**
         * Create help popover content
         * @memberof a2a.portal.utils.HelpProvider
         * @private
         * @param {Object} helpContent - Help content object
         * @param {string} sHelpKey - Help key
         * @returns {sap.m.VBox} Popover content
         */
        _createHelpPopoverContent: function (helpContent, sHelpKey) {
            var aContent = [];

            // Main content
            if (helpContent.content) {
                aContent.push(new Text({
                    text: helpContent.content
                }).addStyleClass("sapUiSmallMarginBottom"));
            }

            // Detailed help
            if (helpContent.detailedHelp) {
                aContent.push(new Text({
                    text: helpContent.detailedHelp
                }).addStyleClass("sapUiSmallMarginBottom"));
            }

            // Tips
            if (helpContent.tips && helpContent.tips.length > 0) {
                aContent.push(new Text({
                    text: "Tips:",
                    wrapping: true
                }).addStyleClass("sapUiTinyMarginTop boldText"));
                
                helpContent.tips.forEach(function (tip) {
                    aContent.push(new Text({
                        text: "• " + tip,
                        wrapping: true
                    }).addStyleClass("sapUiTinyMarginBegin"));
                });
            }

            // Warnings
            if (helpContent.warnings && helpContent.warnings.length > 0) {
                aContent.push(new Text({
                    text: "Important:",
                    wrapping: true
                }).addStyleClass("sapUiTinyMarginTop boldText warningText"));
                
                helpContent.warnings.forEach(function (warning) {
                    aContent.push(new Text({
                        text: "⚠ " + warning,
                        wrapping: true
                    }).addStyleClass("sapUiTinyMarginBegin warningText"));
                });
            }

            // Prerequisites
            if (helpContent.prerequisites && helpContent.prerequisites.length > 0) {
                aContent.push(new Text({
                    text: "Prerequisites:",
                    wrapping: true
                }).addStyleClass("sapUiTinyMarginTop boldText"));
                
                helpContent.prerequisites.forEach(function (prereq) {
                    aContent.push(new Text({
                        text: "✓ " + prereq,
                        wrapping: true
                    }).addStyleClass("sapUiTinyMarginBegin"));
                });
            }

            // Action buttons
            var oButtonBox = new HBox({
                justifyContent: "SpaceBetween",
                width: "100%"
            }).addStyleClass("sapUiSmallMarginTop");

            // Learn more link
            if (helpContent.learnMoreUrl) {
                oButtonBox.addItem(new Link({
                    text: "Learn More",
                    href: helpContent.learnMoreUrl,
                    target: "_blank"
                }));
            }

            // Video tutorial
            if (helpContent.videoUrl) {
                oButtonBox.addItem(new Button({
                    text: "Watch Tutorial",
                    icon: "sap-icon://video",
                    type: "Emphasized",
                    press: function () {
                        window.open(helpContent.videoUrl, "_blank");
                    }
                }));
            }

            // Feedback button
            oButtonBox.addItem(new Button({
                text: "Feedback",
                icon: "sap-icon://feedback",
                press: function () {
                    this._showFeedbackDialog(sHelpKey);
                }.bind(this)
            }));

            if (oButtonBox.getItems().length > 0) {
                aContent.push(oButtonBox);
            }

            return new VBox({
                items: aContent
            }).addStyleClass("sapUiSmallPadding");
        },

        /**
         * Get help content by path
         * @memberof a2a.portal.utils.HelpProvider
         * @private
         * @param {string[]} aPath - Path array
         * @returns {Object|null} Help content
         */
        _getHelpContent: function (aPath) {
            var content = this._helpContent;
            for (var i = 0; i < aPath.length; i++) {
                if (content[aPath[i]]) {
                    content = content[aPath[i]];
                } else {
                    return null;
                }
            }
            return content;
        },

        /**
         * Show feedback dialog
         * @memberof a2a.portal.utils.HelpProvider
         * @private
         * @param {string} sHelpKey - Help key for context
         */
        _showFeedbackDialog: function (sHelpKey) {
            // Implementation for feedback dialog
            MessageToast.show("Thank you for your feedback!");
        },

        /**
         * Enable smart tooltips for all controls in a container
         * @memberof a2a.portal.utils.HelpProvider
         * @public
         * @param {sap.ui.core.Control} oContainer - Container control (View, Page, Panel, etc.)
         * @param {Object} [oCustomMappings] - Custom tooltip mappings to merge with defaults
         * @returns {void}
         * 
         * @example
         * // Enable smart tooltips for entire view
         * onAfterRendering: function() {
         *     helpProvider.enableSmartTooltips(this.getView());
         * }
         * 
         * @example
         * // Enable with custom mappings
         * helpProvider.enableSmartTooltips(this.getView(), {
         *     "sap.m.Input": {
         *         "CustomerId": "Enter the 10-digit customer ID",
         *         "OrderNumber": "Format: ORD-YYYY-NNNNNN"
         *     },
         *     "sap.m.DatePicker": {
         *         "DeliveryDate": "Expected delivery date (2-5 business days)"
         *     }
         * });
         * 
         * @example
         * // Dynamic tooltips based on context
         * const dynamicMappings = {};
         * 
         * // Build mappings based on current state
         * if (this.isEditMode()) {
         *     dynamicMappings["sap.m.Button"] = {
         *         "SaveBtn": "Save changes (Ctrl+S)",
         *         "CancelBtn": "Discard changes (Esc)"
         *     };
         * } else {
         *     dynamicMappings["sap.m.Button"] = {
         *         "EditBtn": "Enter edit mode (E)",
         *         "DeleteBtn": "Delete item (Del)"
         *     };
         * }
         * 
         * helpProvider.enableSmartTooltips(
         *     this.byId("formContainer"),
         *     dynamicMappings
         * );
         * 
         * @example
         * // Apply to specific form with field validation hints
         * const formTooltips = {
         *     "sap.m.Input": {
         *         "Email": "Valid email required (user@domain.com)",
         *         "Phone": "Format: +1 (555) 123-4567",
         *         "ZipCode": "5-digit ZIP code"
         *     }
         * };
         * 
         * helpProvider.enableSmartTooltips(
         *     this.byId("userForm"),
         *     formTooltips
         * );
         */
        enableSmartTooltips: function (oContainer, oCustomMappings) {
            var that = this;
            
            // Define tooltip mappings based on control type and context
            // Merge custom mappings with defaults
            var tooltipMappings = Object.assign({
                "sap.m.Button": {
                    "idCreateBtn": "Create a new item",
                    "idSaveBtn": "Save your changes",
                    "idCancelBtn": "Cancel without saving",
                    "idDeleteBtn": "Delete selected item",
                    "idDeployBtn": "Deploy to environment"
                },
                "sap.m.Input": {
                    "Name": "Enter a descriptive name",
                    "Description": "Provide detailed description",
                    "Id": "Unique identifier (auto-generated if empty)"
                },
                "sap.m.Select": {
                    "Type": "Select the appropriate type",
                    "Environment": "Choose target environment",
                    "Template": "Select a template to start with"
                }
            }, oCustomMappings || {});

            // Recursively process controls
            function processControl(oControl) {
                if (!oControl) return;

                var sControlType = oControl.getMetadata().getName();
                var sId = oControl.getId();
                
                // Check for mapping
                if (tooltipMappings[sControlType]) {
                    // Try to match by ID suffix
                    Object.keys(tooltipMappings[sControlType]).forEach(function (key) {
                        if (sId.endsWith(key)) {
                            oControl.setTooltip(tooltipMappings[sControlType][key]);
                        }
                    });
                    
                    // Try to match by label
                    if (oControl.getLabels && oControl.getLabels().length > 0) {
                        var sLabel = oControl.getLabels()[0].getText();
                        if (tooltipMappings[sControlType][sLabel]) {
                            oControl.setTooltip(tooltipMappings[sControlType][sLabel]);
                        }
                    }
                }

                // Process aggregations
                var aAggregations = oControl.getMetadata().getAllAggregations();
                Object.keys(aAggregations).forEach(function (sAggregation) {
                    var oAggregation = oControl.getAggregation(sAggregation);
                    if (Array.isArray(oAggregation)) {
                        oAggregation.forEach(processControl);
                    } else if (oAggregation) {
                        processControl(oAggregation);
                    }
                });
            }

            processControl(oContainer);
        },

        /**
         * Get instance (singleton pattern)
         * @memberof a2a.portal.utils.HelpProvider
         * @static
         * @returns {a2a.portal.utils.HelpProvider} The help provider instance
         */
        getInstance: function () {
            if (!HelpProvider._instance) {
                HelpProvider._instance = new HelpProvider();
            }
            return HelpProvider._instance;
        }
    });

    return HelpProvider.getInstance();
});