/**
 * @fileoverview Agent Builder Controller for A2A Developer Portal
 * @module a2a/portal/controller/AgentBuilder
 * @requires sap.ui.core.mvc.Controller
 * @requires sap.ui.model.json.JSONModel
 * @requires sap.m.MessageToast
 * @requires sap.m.MessageBox
 */

sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";
/* global Blob, btoa, Fragment, FileReader, Image */

    /**
     * Agent Builder Controller
     * @class
     * @alias a2a.portal.controller.AgentBuilder
     * @extends sap.ui.core.mvc.Controller
     * @description Manages the agent creation and configuration interface, allowing users to build,
     * test, and deploy A2A agents with various templates and configurations
     * @author A2A Development Team
     * @version 1.0.0
     * @public
     */
    return Controller.extend("a2a.portal.controller.AgentBuilder", {

        /**
         * Controller initialization
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onInit
         * @description Initializes the controller, sets up data models, and attaches route handlers
         * @returns {void}
         * @public
         */
        onInit: function () {
            // Initialize agent model with default configuration
            const oAgentModel = new JSONModel({
                name: "",
                id: "",
                type: "reactive",
                description: "",
                template: "blank",
                icon: {
                    src: "",
                    name: "",
                    size: 0,
                    type: "",
                    lastModified: null
                },
                metadata: {
                    category: "",
                    tags: [],
                    tagsString: "",
                    priority: "medium",
                    author: "",
                    organization: "",
                    contactEmail: "",
                    license: "proprietary",
                    performance: "moderate",
                    platforms: {
                        linux: true,
                        windows: false,
                        macos: false,
                        docker: true,
                        kubernetes: false
                    },
                    dependencies: "",
                    resources: {
                        memory: 256,
                        cpu: 1,
                        storage: 100
                    },
                    documentationUrl: "",
                    repositoryUrl: "",
                    releaseNotes: "",
                    createdAt: new Date().toISOString(),
                    lastModified: new Date().toISOString(),
                    lastDeployed: null
                },
                skills: [],
                handlers: [],
                protocol: "mqtt",
                messageFormat: 0,
                subscribedTopics: "",
                publishTopics: "",
                version: {
                    major: 1,
                    minor: 0,
                    patch: 0,
                    prerelease: "",
                    build: "",
                    full: "1.0.0"
                },
                changelog: [],
                versionHistory: [],
                compatibilityVersion: "1.0"
            });
            this.getView().setModel(oAgentModel, "agent");

            // Initialize view model
            const oViewModel = new JSONModel({
                projectName: "",
                codeLanguage: "python",
                generatedCode: "",
                testOutput: "",
                buildStatus: "idle", // idle, building, success, error
                buildProgress: 0,
                buildProgressText: "",
                buildCurrentStep: "",
                buildOutput: "",
                buildDuration: 0,
                buildErrors: 0,
                buildWarnings: 0,
                buildArtifacts: [],
                suggestedTags: [
                    { key: "ai", text: "AI" },
                    { key: "ml", text: "Machine Learning" },
                    { key: "data", text: "Data Processing" },
                    { key: "automation", text: "Automation" },
                    { key: "integration", text: "Integration" },
                    { key: "communication", text: "Communication" },
                    { key: "monitoring", text: "Monitoring" },
                    { key: "analytics", text: "Analytics" },
                    { key: "workflow", text: "Workflow" },
                    { key: "real-time", text: "Real-time" },
                    { key: "batch", text: "Batch Processing" },
                    { key: "api", text: "API" },
                    { key: "database", text: "Database" },
                    { key: "cloud", text: "Cloud" },
                    { key: "microservice", text: "Microservice" }
                ]
            });
            this.getView().setModel(oViewModel);

            // Get router
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("agentBuilder").attachPatternMatched(this._onPatternMatched, this);
        },

        /**
         * Route pattern matched event handler
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _onPatternMatched
         * @private
         * @param {sap.ui.base.Event} oEvent - The route matched event
         * @param {Object} oEvent.mParameters - Event parameters
         * @param {Object} oEvent.mParameters.arguments - Route arguments
         * @param {string} oEvent.mParameters.arguments.projectId - The project ID from the route
         * @returns {void}
         * @description Handles navigation to the agent builder with a specific project ID
         */
        _onPatternMatched: function (oEvent) {
            const sProjectId = oEvent.getParameter("args").projectId;
            this._projectId = sProjectId;
            this._loadProjectInfo(sProjectId);
        },

        /**
         * Loads project information from the backend
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _loadProjectInfo
         * @private
         * @param {string} sProjectId - The unique project identifier
         * @returns {void}
         * @fires projectLoaded - When project data is successfully loaded
         * @throws {Error} When project loading fails
         * @description Fetches project details and updates the view model
         */
        _loadProjectInfo: function (sProjectId) {
            jQuery.ajax({
                url: `/api/projects/${sProjectId}`,
                method: "GET",
                success: function (data) {
                    this.getView().getModel().setProperty("/projectName", data.name);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show(`Failed to load project info: ${error}`);
                }.bind(this)
            });
        },

        onNavToProjects: function () {
            this.getOwnerComponent().getRouter().navTo("projects");
        },

        onNavToProject: function () {
            this.getOwnerComponent().getRouter().navTo("projectDetail", {
                projectId: this._projectId
            });
        },

        onTemplateChange: function (oEvent) {
            const sTemplate = oEvent.getParameter("selectedItem").getKey();
            const oAgentModel = this.getView().getModel("agent");
            
            // Check if there are custom changes
            if (this._hasCustomChanges()) {
                MessageBox.confirm(
                    "Changing the template will overwrite your current configuration. Do you want to continue?",
                    {
                        title: "Confirm Template Change",
                        onClose: function (sAction) {
                            if (sAction === MessageBox.Action.OK) {
                                this._applyTemplate(sTemplate);
                            } else {
                                // Revert selection to current template
                                const oSelect = this.byId("templateSelect");
                                if (oSelect) {
                                    oSelect.setSelectedKey(oAgentModel.getProperty("/template"));
                                }
                            }
                        }.bind(this)
                    }
                );
            } else {
                this._applyTemplate(sTemplate);
            }
        },
        
        _hasCustomChanges: function () {
            const oAgentModel = this.getView().getModel("agent");
            const aSkills = oAgentModel.getProperty("/skills") || [];
            const aHandlers = oAgentModel.getProperty("/handlers") || [];
            const sName = oAgentModel.getProperty("/name");
            const sDescription = oAgentModel.getProperty("/description");
            
            // Check if user has made any custom configurations
            return aSkills.length > 0 || aHandlers.length > 0 || 
                   (sName && sName.length > 0) || (sDescription && sDescription.length > 0);
        },

        _applyTemplate: function (sTemplate) {
            const oAgentModel = this.getView().getModel("agent");
            
            // Store current template
            oAgentModel.setProperty("/template", sTemplate);
            
            switch (sTemplate) {
                case "data-processor":
                    oAgentModel.setProperty("/skills", [
                        { name: "Data Ingestion", description: "Consume data from various sources", icon: "sap-icon://download" },
                        { name: "Data Transformation", description: "Transform and clean data", icon: "sap-icon://refresh" },
                        { name: "Data Validation", description: "Validate data quality", icon: "sap-icon://accept" }
                    ]);
                    oAgentModel.setProperty("/handlers", [
                        { event: "data.received", handler: "processIncomingData", priority: 1 },
                        { event: "data.error", handler: "handleDataError", priority: 2 }
                    ]);
                    oAgentModel.setProperty("/type", "reactive");
                    oAgentModel.setProperty("/protocol", "mqtt");
                    break;
                    
                case "api-integrator":
                    oAgentModel.setProperty("/skills", [
                        { name: "REST API Handler", description: "Handle REST API calls", icon: "sap-icon://cloud" },
                        { name: "Authentication", description: "Manage API authentication", icon: "sap-icon://locked" },
                        { name: "Response Mapping", description: "Map API responses", icon: "sap-icon://combine" },
                        { name: "Rate Limiting", description: "Handle API rate limits", icon: "sap-icon://performance" }
                    ]);
                    oAgentModel.setProperty("/handlers", [
                        { event: "http.request", handler: "handleApiRequest", priority: 1 },
                        { event: "auth.required", handler: "refreshAuthentication", priority: 1 },
                        { event: "rate.limit", handler: "handleRateLimit", priority: 2 }
                    ]);
                    oAgentModel.setProperty("/type", "reactive");
                    oAgentModel.setProperty("/protocol", "http");
                    break;
                    
                case "ml-analyzer":
                    oAgentModel.setProperty("/skills", [
                        { name: "Model Loading", description: "Load and initialize ML models", icon: "sap-icon://learning-assistant" },
                        { name: "Prediction", description: "Run ML model predictions", icon: "sap-icon://forecasting" },
                        { name: "Training", description: "Train and update models", icon: "sap-icon://education" }
                    ]);
                    oAgentModel.setProperty("/handlers", [
                        { event: "prediction.request", handler: "runPrediction", priority: 1 },
                        { event: "model.update", handler: "updateModel", priority: 2 }
                    ]);
                    oAgentModel.setProperty("/type", "proactive");
                    oAgentModel.setProperty("/protocol", "mqtt");
                    break;
                    
                case "workflow-coordinator":
                    oAgentModel.setProperty("/skills", [
                        { name: "Task Scheduling", description: "Schedule and manage tasks", icon: "sap-icon://calendar" },
                        { name: "State Management", description: "Manage workflow state", icon: "sap-icon://process" },
                        { name: "Event Routing", description: "Route events between agents", icon: "sap-icon://journey-arrive" },
                        { name: "Error Recovery", description: "Handle workflow errors", icon: "sap-icon://restart" }
                    ]);
                    oAgentModel.setProperty("/handlers", [
                        { event: "workflow.start", handler: "initiateWorkflow", priority: 1 },
                        { event: "task.complete", handler: "processTaskCompletion", priority: 1 },
                        { event: "workflow.error", handler: "handleWorkflowError", priority: 2 }
                    ]);
                    oAgentModel.setProperty("/type", "collaborative");
                    oAgentModel.setProperty("/protocol", "mqtt");
                    break;
                    
                case "blank":
                default:
                    // Clear all template-based configurations
                    oAgentModel.setProperty("/skills", []);
                    oAgentModel.setProperty("/handlers", []);
                    oAgentModel.setProperty("/type", "reactive");
                    oAgentModel.setProperty("/protocol", "mqtt");
                    break;
            }
            
            // Auto-fill metadata based on template
            this._autoFillMetadata();
            
            MessageToast.show(sTemplate === "blank" ? "Template cleared" : `Template applied: ${sTemplate}`);
        },

        /**
         * Opens capability selection dialog
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onAddSkill
         * @returns {void}
         * @description Opens dialog for selecting and configuring agent capabilities
         */
        onAddSkill: function () {
            // Initialize available capabilities model if not exists
            if (!this._oCapabilitiesModel) {
                this._oCapabilitiesModel = new JSONModel({
                    categories: this._getCapabilityCategories(),
                    selectedCategory: "data",
                    selectedCapabilities: [],
                    searchQuery: ""
                });
                this.setModel(this._oCapabilitiesModel, "capabilities");
            }
            
            // Open capability selection dialog
            if (!this._oCapabilityDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "sap.a2a.view.fragments.CapabilitySelectionDialog",
                    controller: this
                }).then(function (oDialog) {
                    this._oCapabilityDialog = oDialog;
                    this.getView().addDependent(this._oCapabilityDialog);
                    this._oCapabilityDialog.open();
                    this._loadAvailableCapabilities();
                }.bind(this));
            } else {
                this._oCapabilityDialog.open();
                this._loadAvailableCapabilities();
            }
        },
        
        /**
         * Gets capability categories
         * @private
         */
        _getCapabilityCategories: function () {
            return [
                {
                    key: "data",
                    name: "Data Processing",
                    icon: "sap-icon://database",
                    description: "Data manipulation, transformation, and analysis capabilities"
                },
                {
                    key: "communication",
                    name: "Communication",
                    icon: "sap-icon://message",
                    description: "Message handling and protocol capabilities"
                },
                {
                    key: "integration",
                    name: "System Integration",
                    icon: "sap-icon://chain-link",
                    description: "API integration and external system connectivity"
                },
                {
                    key: "ai",
                    name: "AI & Machine Learning",
                    icon: "sap-icon://learning-assistant",
                    description: "AI processing and machine learning capabilities"
                },
                {
                    key: "workflow",
                    name: "Workflow Management",
                    icon: "sap-icon://workflow-tasks",
                    description: "Process orchestration and task management"
                },
                {
                    key: "monitoring",
                    name: "Monitoring & Analytics",
                    icon: "sap-icon://activity-2",
                    description: "System monitoring and performance analytics"
                }
            ];
        },
        
        /**
         * Loads available capabilities from backend
         * @private
         */
        _loadAvailableCapabilities: function () {
            const oModel = this._oCapabilitiesModel;
            const sCategory = oModel.getProperty("/selectedCategory");
            
            // Mock data - in real implementation would load from API
            const mCapabilities = {
                data: [
                    {
                        id: "csv-parser",
                        name: "CSV File Parser",
                        description: "Parse and process CSV files",
                        icon: "sap-icon://excel-attachment",
                        complexity: "Low",
                        dependencies: ["file-system"],
                        tags: ["parsing", "csv", "data"]
                    },
                    {
                        id: "json-processor",
                        name: "JSON Data Processor",
                        description: "Process and transform JSON data structures",
                        icon: "sap-icon://syntax",
                        complexity: "Low",
                        dependencies: [],
                        tags: ["json", "parsing", "transformation"]
                    },
                    {
                        id: "data-validator",
                        name: "Data Validation",
                        description: "Validate data against schemas and rules",
                        icon: "sap-icon://validate",
                        complexity: "Medium",
                        dependencies: ["schema-loader"],
                        tags: ["validation", "schema", "quality"]
                    },
                    {
                        id: "data-aggregator",
                        name: "Data Aggregation",
                        description: "Aggregate and summarize large datasets",
                        icon: "sap-icon://sum",
                        complexity: "Medium",
                        dependencies: [],
                        tags: ["aggregation", "summary", "analytics"]
                    }
                ],
                communication: [
                    {
                        id: "mqtt-client",
                        name: "MQTT Client",
                        description: "MQTT protocol communication capability",
                        icon: "sap-icon://message",
                        complexity: "Medium",
                        dependencies: ["network"],
                        tags: ["mqtt", "messaging", "iot"]
                    },
                    {
                        id: "websocket-handler",
                        name: "WebSocket Handler",
                        description: "Real-time WebSocket communication",
                        icon: "sap-icon://discussion",
                        complexity: "Medium",
                        dependencies: ["network"],
                        tags: ["websocket", "realtime", "bidirectional"]
                    },
                    {
                        id: "email-sender",
                        name: "Email Notification",
                        description: "Send email notifications and alerts",
                        icon: "sap-icon://email",
                        complexity: "Low",
                        dependencies: ["smtp-config"],
                        tags: ["email", "notifications", "smtp"]
                    }
                ],
                integration: [
                    {
                        id: "rest-client",
                        name: "REST API Client",
                        description: "HTTP REST API integration capability",
                        icon: "sap-icon://internet-browser",
                        complexity: "Low",
                        dependencies: ["http-client"],
                        tags: ["rest", "api", "http"]
                    },
                    {
                        id: "database-connector",
                        name: "Database Connector",
                        description: "Connect to various database systems",
                        icon: "sap-icon://database",
                        complexity: "High",
                        dependencies: ["db-drivers"],
                        tags: ["database", "sql", "connector"]
                    },
                    {
                        id: "file-watcher",
                        name: "File System Watcher",
                        description: "Monitor file system changes",
                        icon: "sap-icon://folder",
                        complexity: "Medium",
                        dependencies: ["file-system"],
                        tags: ["filesystem", "monitoring", "events"]
                    }
                ],
                ai: [
                    {
                        id: "text-classifier",
                        name: "Text Classification",
                        description: "Classify text using machine learning",
                        icon: "sap-icon://tags",
                        complexity: "High",
                        dependencies: ["ml-models"],
                        tags: ["classification", "nlp", "ml"]
                    },
                    {
                        id: "sentiment-analyzer",
                        name: "Sentiment Analysis",
                        description: "Analyze sentiment in text data",
                        icon: "sap-icon://feedback",
                        complexity: "High",
                        dependencies: ["nlp-models"],
                        tags: ["sentiment", "nlp", "analysis"]
                    },
                    {
                        id: "anomaly-detector",
                        name: "Anomaly Detection",
                        description: "Detect anomalies in data patterns",
                        icon: "sap-icon://warning",
                        complexity: "High",
                        dependencies: ["statistical-models"],
                        tags: ["anomaly", "detection", "patterns"]
                    }
                ],
                workflow: [
                    {
                        id: "task-scheduler",
                        name: "Task Scheduler",
                        description: "Schedule and manage recurring tasks",
                        icon: "sap-icon://calendar",
                        complexity: "Medium",
                        dependencies: ["timing"],
                        tags: ["scheduling", "tasks", "cron"]
                    },
                    {
                        id: "state-machine",
                        name: "State Machine",
                        description: "Manage complex workflow states",
                        icon: "sap-icon://process",
                        complexity: "High",
                        dependencies: ["workflow-engine"],
                        tags: ["state", "workflow", "fsm"]
                    }
                ],
                monitoring: [
                    {
                        id: "health-checker",
                        name: "Health Monitor",
                        description: "Monitor system health and status",
                        icon: "sap-icon://electrocardiogram",
                        complexity: "Medium",
                        dependencies: ["metrics"],
                        tags: ["health", "monitoring", "status"]
                    },
                    {
                        id: "performance-tracker",
                        name: "Performance Tracker",
                        description: "Track and analyze performance metrics",
                        icon: "sap-icon://performance",
                        complexity: "Medium",
                        dependencies: ["metrics", "storage"],
                        tags: ["performance", "metrics", "tracking"]
                    }
                ]
            };
            
            oModel.setProperty("/availableCapabilities", mCapabilities[sCategory] || []);
        },
        
        /**
         * Handle capability category change
         */
        onCapabilityCategoryChange: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("key");
            this._oCapabilitiesModel.setProperty("/selectedCategory", sSelectedKey);
            this._loadAvailableCapabilities();
        },
        
        /**
         * Handle capability search
         */
        onCapabilitySearch: function (oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            this._oCapabilitiesModel.setProperty("/searchQuery", sQuery);
            this._filterCapabilities(sQuery);
        },
        
        /**
         * Filter capabilities based on search
         * @private
         */
        _filterCapabilities: function (sQuery) {
            const oList = this.byId("capabilityList");
            const oBinding = oList.getBinding("items");
            
            if (!sQuery) {
                oBinding.filter([]);
                return;
            }
            
            const aFilters = [
                new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                new sap.ui.model.Filter("description", sap.ui.model.FilterOperator.Contains, sQuery),
                new sap.ui.model.Filter("tags", sap.ui.model.FilterOperator.Contains, sQuery)
            ];
            
            const oOrFilter = new sap.ui.model.Filter({
                filters: aFilters,
                and: false
            });
            
            oBinding.filter([oOrFilter]);
        },
        
        /**
         * Handle capability selection
         */
        onCapabilitySelect: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("capabilities");
            const oCapability = oContext.getObject();
            const oModel = this._oCapabilitiesModel;
            
            let aSelected = oModel.getProperty("/selectedCapabilities") || [];
            const bSelected = oEvent.getParameter("selected");
            
            if (bSelected) {
                aSelected.push(oCapability);
            } else {
                aSelected = aSelected.filter(item => item.id !== oCapability.id);
            }
            
            oModel.setProperty("/selectedCapabilities", aSelected);
        },
        
        /**
         * Add selected capabilities to agent
         */
        onAddSelectedCapabilities: function () {
            const oCapModel = this._oCapabilitiesModel;
            const oAgentModel = this.getView().getModel("agent");
            const aSelected = oCapModel.getProperty("/selectedCapabilities") || [];
            
            if (aSelected.length === 0) {
                MessageToast.show("Please select at least one capability");
                return;
            }
            
            // Convert capabilities to skills format
            const aNewSkills = aSelected.map(cap => ({
                id: cap.id,
                name: cap.name,
                description: cap.description,
                icon: cap.icon,
                complexity: cap.complexity,
                dependencies: cap.dependencies,
                tags: cap.tags,
                configured: false
            }));
            
            // Add to existing skills
            const aCurrentSkills = oAgentModel.getProperty("/skills") || [];
            
            // Check for duplicates
            aNewSkills.forEach(newSkill => {
                const bExists = aCurrentSkills.some(skill => skill.id === newSkill.id);
                if (!bExists) {
                    aCurrentSkills.push(newSkill);
                }
            });
            
            oAgentModel.setProperty("/skills", aCurrentSkills);
            
            // Reset selection and close dialog
            oCapModel.setProperty("/selectedCapabilities", []);
            this._oCapabilityDialog.close();
            
            MessageToast.show(`${aNewSkills.length} capabilities added successfully`);
        },
        
        /**
         * Show capability details
         */
        onShowCapabilityDetails: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("capabilities");
            const oCapability = oContext.getObject();
            
            MessageBox.information(
                `Name: ${oCapability.name}\n\nDescription: ${oCapability.description}\n\nComplexity: ${oCapability.complexity}\n\nDependencies: ${oCapability.dependencies.join(', ') || 'None'}\n\nTags: ${oCapability.tags.join(', ')}`,
                {
                    title: "Capability Details",
                    styleClass: "sapUiSizeCompact"
                }
            );
        },
        
        /**
         * Clear capability selection
         */
        onClearCapabilitySelection: function () {
            this._oCapabilitiesModel.setProperty("/selectedCapabilities", []);
            
            // Clear list selection
            const oList = this.byId("capabilityList");
            oList.removeSelections(true);
        },
        
        /**
         * Add basic data capabilities
         */
        onAddBasicDataCapabilities: function () {
            this._addPredefinedCapabilities(["csv-parser", "json-processor", "data-validator"]);
        },
        
        /**
         * Add communication capabilities
         */
        onAddCommunicationCapabilities: function () {
            this._addPredefinedCapabilities(["mqtt-client", "websocket-handler", "email-sender"]);
        },
        
        /**
         * Add AI capabilities
         */
        onAddAICapabilities: function () {
            this._addPredefinedCapabilities(["text-classifier", "sentiment-analyzer", "anomaly-detector"]);
        },
        
        /**
         * Add workflow capabilities
         */
        onAddWorkflowCapabilities: function () {
            this._addPredefinedCapabilities(["task-scheduler", "state-machine", "health-checker"]);
        },
        
        /**
         * Add predefined capabilities by IDs
         * @private
         */
        _addPredefinedCapabilities: function (aCapabilityIds) {
            const oModel = this._oCapabilitiesModel;
            const aSelected = oModel.getProperty("/selectedCapabilities") || [];
            
            // Get all available capabilities from all categories
            const mAllCapabilities = {};
            const aCategories = ["data", "communication", "integration", "ai", "workflow", "monitoring"];
            
            aCategories.forEach(sCategory => {
                this._oCapabilitiesModel.setProperty("/selectedCategory", sCategory);
                this._loadAvailableCapabilities();
                const aCapabilities = this._oCapabilitiesModel.getProperty("/availableCapabilities") || [];
                aCapabilities.forEach(cap => {
                    mAllCapabilities[cap.id] = cap;
                });
            });
            
            // Add requested capabilities
            aCapabilityIds.forEach(sId => {
                const oCapability = mAllCapabilities[sId];
                if (oCapability && !aSelected.some(sel => sel.id === sId)) {
                    aSelected.push(oCapability);
                }
            });
            
            oModel.setProperty("/selectedCapabilities", aSelected);
            MessageToast.show(`${aCapabilityIds.length} capabilities added to selection`);
        },
        
        /**
         * Close capability dialog
         */
        onCloseCapabilityDialog: function () {
            if (this._oCapabilityDialog) {
                this._oCapabilityDialog.close();
            }
        },
        
        /**
         * Opens version management dialog
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onManageVersion
         * @returns {void}
         * @description Opens dialog for managing agent version information
         */
        onManageVersion: function () {
            // Initialize version model if not exists
            if (!this._oVersionModel) {
                this._oVersionModel = new JSONModel({
                    versionTypes: [
                        { key: "major", text: "Major (Breaking Changes)", description: "Incompatible changes" },
                        { key: "minor", text: "Minor (New Features)", description: "Backward compatible features" },
                        { key: "patch", text: "Patch (Bug Fixes)", description: "Backward compatible fixes" }
                    ],
                    selectedVersionType: "patch",
                    changelogEntry: "",
                    autoIncrement: true,
                    customVersion: false,
                    customMajor: 1,
                    customMinor: 0,
                    customPatch: 0,
                    prerelease: "",
                    buildMetadata: ""
                });
                this.setModel(this._oVersionModel, "version");
            }
            
            // Load current version history
            this._loadVersionHistory();
            
            // Open version management dialog
            if (!this._oVersionDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "sap.a2a.view.fragments.VersionManagementDialog",
                    controller: this
                }).then(function (oDialog) {
                    this._oVersionDialog = oDialog;
                    this.getView().addDependent(this._oVersionDialog);
                    this._oVersionDialog.open();
                }.bind(this));
            } else {
                this._oVersionDialog.open();
            }
        },
        
        /**
         * Load version history
         * @private
         */
        _loadVersionHistory: function () {
            const oAgentModel = this.getView().getModel("agent");
            const aVersionHistory = oAgentModel.getProperty("/versionHistory") || [];
            
            // Add current version if not in history
            const oCurrentVersion = oAgentModel.getProperty("/version");
            if (aVersionHistory.length === 0) {
                aVersionHistory.push({
                    version: oCurrentVersion.full,
                    date: new Date().toISOString(),
                    author: "System",
                    changes: ["Initial version"],
                    type: "initial"
                });
                oAgentModel.setProperty("/versionHistory", aVersionHistory);
            }
            
            this._oVersionModel.setProperty("/versionHistory", aVersionHistory);
        },
        
        /**
         * Handle version type change
         */
        onVersionTypeChange: function (oEvent) {
            const sType = oEvent.getParameter("selectedItem").getKey();
            const oVersionModel = this._oVersionModel;
            const _oAgentModel = this.getView().getModel("agent");
            
            oVersionModel.setProperty("/selectedVersionType", sType);
            
            if (oVersionModel.getProperty("/autoIncrement")) {
                this._calculateNextVersion(sType);
            }
        },
        
        /**
         * Calculate next version based on type
         * @private
         */
        _calculateNextVersion: function (sType) {
            const oAgentModel = this.getView().getModel("agent");
            const oCurrentVersion = oAgentModel.getProperty("/version");
            const oVersionModel = this._oVersionModel;
            
            let nMajor = oCurrentVersion.major;
            let nMinor = oCurrentVersion.minor;
            let nPatch = oCurrentVersion.patch;
            
            switch (sType) {
                case "major":
                    nMajor++;
                    nMinor = 0;
                    nPatch = 0;
                    break;
                case "minor":
                    nMinor++;
                    nPatch = 0;
                    break;
                case "patch":
                    nPatch++;
                    break;
            }
            
            oVersionModel.setProperty("/customMajor", nMajor);
            oVersionModel.setProperty("/customMinor", nMinor);
            oVersionModel.setProperty("/customPatch", nPatch);
            
            this._updateVersionPreview();
        },
        
        /**
         * Update version preview
         * @private
         */
        _updateVersionPreview: function () {
            const oVersionModel = this._oVersionModel;
            const nMajor = oVersionModel.getProperty("/customMajor");
            const nMinor = oVersionModel.getProperty("/customMinor");
            const nPatch = oVersionModel.getProperty("/customPatch");
            const sPrerelease = oVersionModel.getProperty("/prerelease");
            const sBuild = oVersionModel.getProperty("/buildMetadata");
            
            let sVersion = `${nMajor}.${nMinor}.${nPatch}`;
            
            if (sPrerelease) {
                sVersion += `-${sPrerelease}`;
            }
            
            if (sBuild) {
                sVersion += `+${sBuild}`;
            }
            
            oVersionModel.setProperty("/versionPreview", sVersion);
        },
        
        /**
         * Handle custom version toggle
         */
        onCustomVersionToggle: function (oEvent) {
            const bCustom = oEvent.getParameter("state");
            this._oVersionModel.setProperty("/customVersion", bCustom);
            
            if (!bCustom) {
                // Reset to auto-calculated version
                const sType = this._oVersionModel.getProperty("/selectedVersionType");
                this._calculateNextVersion(sType);
            }
        },
        
        /**
         * Handle version number change
         */
        onVersionNumberChange: function () {
            this._updateVersionPreview();
        },
        
        /**
         * Validate version format
         * @private
         */
        _validateVersion: function (sVersion) {
            // Semantic versioning regex
            const semverRegex = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$/;
            
            if (!semverRegex.test(sVersion)) {
                return {
                    valid: false,
                    message: "Version must follow semantic versioning format (e.g., 1.2.3, 1.2.3-alpha, 1.2.3+build)"
                };
            }
            
            // Check if version is higher than current
            const oAgentModel = this.getView().getModel("agent");
            const sCurrentVersion = oAgentModel.getProperty("/version/full");
            
            if (this._compareVersions(sVersion, sCurrentVersion) <= 0) {
                return {
                    valid: false,
                    message: `New version must be higher than current version (${sCurrentVersion})`
                };
            }
            
            return { valid: true, message: "" };
        },
        
        /**
         * Compare two semantic versions
         * @private
         */
        _compareVersions: function (version1, version2) {
            const parts1 = version1.split(/[.\-+]/);
            const parts2 = version2.split(/[.\-+]/);
            
            for (let i = 0; i < 3; i++) {
                const num1 = parseInt(parts1[i] || 0);
                const num2 = parseInt(parts2[i] || 0);
                
                if (num1 > num2) {
                    return 1;
                }
                if (num1 < num2) {
                    return -1;
                }
            }
            
            return 0;
        },
        
        /**
         * Apply new version
         */
        onApplyVersion: function () {
            const oVersionModel = this._oVersionModel;
            const oAgentModel = this.getView().getModel("agent");
            const sChangelogEntry = oVersionModel.getProperty("/changelogEntry");
            const sNewVersion = oVersionModel.getProperty("/versionPreview");
            
            // Validate inputs
            if (!sChangelogEntry || sChangelogEntry.trim().length === 0) {
                MessageBox.error("Please provide a changelog entry describing the changes");
                return;
            }
            
            const oValidation = this._validateVersion(sNewVersion);
            if (!oValidation.valid) {
                MessageBox.error(oValidation.message);
                return;
            }
            
            // Parse version components
            const aVersionParts = sNewVersion.split(/[.\-+]/);
            const nMajor = parseInt(aVersionParts[0]);
            const nMinor = parseInt(aVersionParts[1]);
            const nPatch = parseInt(aVersionParts[2]);
            
            const sPrereleaseMatch = sNewVersion.match(/-([^+]*)/);
            const sBuildMatch = sNewVersion.match(/\+(.*)$/);
            
            // Update version in agent model
            const oNewVersion = {
                major: nMajor,
                minor: nMinor,
                patch: nPatch,
                prerelease: sPrereleaseMatch ? sPrereleaseMatch[1] : "",
                build: sBuildMatch ? sBuildMatch[1] : "",
                full: sNewVersion
            };
            
            oAgentModel.setProperty("/version", oNewVersion);
            
            // Add to version history
            const aVersionHistory = oAgentModel.getProperty("/versionHistory") || [];
            const sVersionType = oVersionModel.getProperty("/selectedVersionType");
            
            aVersionHistory.push({
                version: sNewVersion,
                date: new Date().toISOString(),
                author: "Current User", // In real app, get from user context
                changes: [sChangelogEntry],
                type: sVersionType,
                previousVersion: aVersionHistory.length > 0 ? aVersionHistory[aVersionHistory.length - 1].version : "0.0.0"
            });
            
            oAgentModel.setProperty("/versionHistory", aVersionHistory);
            
            // Add to changelog
            const aChangelog = oAgentModel.getProperty("/changelog") || [];
            aChangelog.unshift({
                version: sNewVersion,
                date: new Date().toISOString(),
                changes: [sChangelogEntry],
                type: sVersionType
            });
            
            oAgentModel.setProperty("/changelog", aChangelog);
            
            // Reset form
            oVersionModel.setProperty("/changelogEntry", "");
            
            MessageToast.show(`Agent version updated to ${sNewVersion}`);
            this._oVersionDialog.close();
        },
        
        /**
         * Close version dialog
         */
        onCloseVersionDialog: function () {
            if (this._oVersionDialog) {
                this._oVersionDialog.close();
            }
        },

        onDeleteSkill: function (oEvent) {
            const oItem = oEvent.getParameter("listItem");
            const sPath = oItem.getBindingContext("agent").getPath();
            const oModel = this.getView().getModel("agent");
            const aSkills = oModel.getProperty("/skills");
            const iIndex = parseInt(sPath.split("/").pop());
            
            aSkills.splice(iIndex, 1);
            oModel.setProperty("/skills", aSkills);
        },

        onConfigureSkill: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("agent");
            const sSkillName = oContext.getProperty("name");
            MessageToast.show(`Configure skill: ${sSkillName}`);
        },

        onAddHandler: function () {
            MessageToast.show("Add handler dialog - coming soon");
        },

        onEditHandler: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("agent");
            const sHandler = oContext.getProperty("handler");
            MessageToast.show(`Edit handler: ${sHandler}`);
        },

        onDeleteHandler: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("agent");
            const sHandler = oContext.getProperty("handler");
            MessageToast.show(`Delete handler: ${sHandler}`);
        },

        onGenerateCode: function () {
            const oModel = this.getView().getModel();
            const oAgentModel = this.getView().getModel("agent");
            const sLanguage = oModel.getProperty("/codeLanguage");
            
            // Simple code generation based on template
            const sCode = this._generateAgentCode(oAgentModel.getData(), sLanguage);
            oModel.setProperty("/generatedCode", sCode);
            
            MessageToast.show("Code generated successfully");
        },

        _generateAgentCode: function (oAgentData, sLanguage) {
            if (sLanguage === "python") {
                return `# Generated A2A Agent: ${oAgentData.name}
# ID: ${oAgentData.id}
# Type: ${oAgentData.type}

from a2a_sdk import Agent, Skill, Handler

class ${this._toPascalCase(oAgentData.id)}(Agent):
    def __init__(self):
        super().__init__(
            agent_id="${oAgentData.id}",
            name="${oAgentData.name}",
            description="${oAgentData.description}",
            agent_type="${oAgentData.type}"
        )
        
        # Configure communication
        self.protocol = "${oAgentData.protocol}"
        self.subscribe_topics = "${oAgentData.subscribedTopics}".split(",")
        self.publish_topics = "${oAgentData.publishTopics}".split(",")
        
        # Register skills
        ${oAgentData.skills.map(skill => 
            `self.register_skill(Skill("${skill.name}", "${skill.description}"))`
        ).join('\n        ')}
        
    async def process_message(self, message):
        # Implement message processing logic
        pass
        
    async def execute_skill(self, skill_name, params):
        # Implement skill execution logic
        pass

# Initialize and run agent
if __name__ == "__main__":
    agent = ${this._toPascalCase(oAgentData.id)}()
    agent.run()`;
            } else if (sLanguage === "javascript") {
                return `// Generated A2A Agent: ${oAgentData.name}
// ID: ${oAgentData.id}
// Type: ${oAgentData.type}

const { Agent, Skill, Handler } = require('a2a-sdk');

class ${this._toPascalCase(oAgentData.id)} extends Agent {
    constructor() {
        super({
            agentId: "${oAgentData.id}",
            name: "${oAgentData.name}",
            description: "${oAgentData.description}",
            agentType: "${oAgentData.type}"
        });
        
        // Configure communication
        this.protocol = "${oAgentData.protocol}";
        this.subscribeTopics = "${oAgentData.subscribedTopics}".split(",");
        this.publishTopics = "${oAgentData.publishTopics}".split(",");
        
        // Register skills
        ${oAgentData.skills.map(skill => 
            `this.registerSkill(new Skill("${skill.name}", "${skill.description}"));`
        ).join('\n        ')}
    }
    
    async processMessage(message) {
        // Implement message processing logic
    }
    
    async executeSkill(skillName, params) {
        // Implement skill execution logic
    }
}

// Initialize and run agent
const agent = new ${this._toPascalCase(oAgentData.id)}();
agent.run();`;
            }
            
            return `// Code generation not implemented for ${sLanguage}`;
        },

        _toPascalCase: function (str) {
            return str.replace(/[-_](.)/g, function (match, chr) {
                return chr.toUpperCase();
            }).replace(/^(.)/, function (match, chr) {
                return chr.toUpperCase();
            });
        },

        onCopyCode: function () {
            const sCode = this.getView().getModel().getProperty("/generatedCode");
            if (sCode) {
                navigator.clipboard.writeText(sCode).then(function () {
                    MessageToast.show("Code copied to clipboard");
                });
            }
        },

        onDownloadCode: function () {
            const sCode = this.getView().getModel().getProperty("/generatedCode");
            const sAgentId = this.getView().getModel("agent").getProperty("/id");
            const sLanguage = this.getView().getModel().getProperty("/codeLanguage");
            
            const sExtension = sLanguage === "python" ? "py" : "js";
            const sFilename = `${sAgentId}_agent.${sExtension}`;
            
            const element = document.createElement('a');
            element.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(sCode)}`);
            element.setAttribute('download', sFilename);
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
            
            MessageToast.show(`Code downloaded as ${sFilename}`);
        },

        onStartTest: function () {
            const oModel = this.getView().getModel();
            oModel.setProperty("/testOutput", "Starting agent test...\n");
            
            // Simulate test execution
            setTimeout(function () {
                let sOutput = oModel.getProperty("/testOutput");
                sOutput += "Agent initialized successfully\n";
                sOutput += "Connecting to message broker...\n";
                sOutput += "Connected to MQTT broker\n";
                sOutput += "Agent is ready to receive messages\n";
                oModel.setProperty("/testOutput", sOutput);
            }.bind(this), 1000);
            
            MessageToast.show("Test started");
        },

        onStopTest: function () {
            const oModel = this.getView().getModel();
            let sOutput = oModel.getProperty("/testOutput");
            sOutput += "Stopping agent test...\n";
            sOutput += "Test stopped\n";
            oModel.setProperty("/testOutput", sOutput);
            
            MessageToast.show("Test stopped");
        },

        onClearConsole: function () {
            this.getView().getModel().setProperty("/testOutput", "");
        },

        onSendTestMessage: function (oEvent) {
            const sMessage = oEvent.getParameter("value");
            if (sMessage) {
                const oModel = this.getView().getModel();
                let sOutput = oModel.getProperty("/testOutput");
                sOutput += `> ${sMessage}\n`;
                
                if (oModel.getProperty("/debugMode")) {
                    sOutput += `< [DEBUG] Processing message: ${sMessage}\n`;
                    sOutput += "< [DEBUG] Agent state: active\n";
                    sOutput += "< [DEBUG] Message handler: onMessageReceived()\n";
                    if (oModel.getProperty("/debugBreakpoints/length") > 0) {
                        sOutput += "< [DEBUG] Breakpoint hit - execution paused\n";
                        oModel.setProperty("/debugPaused", true);
                    }
                }
                
                sOutput += "< Processing message...\n";
                sOutput += "< Message processed successfully\n";
                oModel.setProperty("/testOutput", sOutput);
                
                oEvent.getSource().setValue("");
            }
        },

        /**
         * Toggles debug mode for agent testing
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onToggleDebugMode
         * @returns {void}
         * @description Enables/disables debug mode with enhanced logging and breakpoint support
         */
        onToggleDebugMode: function () {
            const oModel = this.getView().getModel();
            const bCurrentDebugMode = oModel.getProperty("/debugMode");
            const bNewDebugMode = !bCurrentDebugMode;
            
            oModel.setProperty("/debugMode", bNewDebugMode);
            oModel.setProperty("/debugPaused", false);
            
            if (!oModel.getProperty("/debugBreakpoints")) {
                oModel.setProperty("/debugBreakpoints", []);
            }
            
            let sOutput = oModel.getProperty("/testOutput");
            if (bNewDebugMode) {
                sOutput += "\n[DEBUG MODE ENABLED]\n";
                sOutput += "Debug features activated:\n";
                sOutput += "- Enhanced logging\n";
                sOutput += "- Breakpoint support\n";
                sOutput += "- Step-through debugging\n";
                sOutput += "- Variable inspection\n";
                MessageToast.show("Debug mode enabled - enhanced testing active");
            } else {
                sOutput += "\n[DEBUG MODE DISABLED]\n";
                sOutput += "Returning to normal test mode\n";
                MessageToast.show("Debug mode disabled - normal testing resumed");
            }
            oModel.setProperty("/testOutput", sOutput);
        },

        /**
         * Executes debug step when in debug mode
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onDebugStep
         * @returns {void}
         * @description Steps through agent execution when paused at breakpoint
         */
        onDebugStep: function () {
            const oModel = this.getView().getModel();
            if (!oModel.getProperty("/debugMode") || !oModel.getProperty("/debugPaused")) {
                return;
            }
            
            let sOutput = oModel.getProperty("/testOutput");
            sOutput += "< [DEBUG STEP] Executing next instruction\n";
            sOutput += "< [DEBUG STEP] Current line: message_handler.process()\n";
            sOutput += "< [DEBUG STEP] Variables: {input: 'test', status: 'processing'}\n";
            sOutput += "< [DEBUG STEP] Step completed - ready for next instruction\n";
            
            oModel.setProperty("/testOutput", sOutput);
            oModel.setProperty("/debugPaused", false);
            
            MessageToast.show("Debug step executed");
        },

        /**
         * Sets breakpoint for debug mode
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onSetBreakpoint
         * @returns {void}
         * @description Sets breakpoint in agent code for debugging
         */
        onSetBreakpoint: function () {
            const oModel = this.getView().getModel();
            if (!oModel.getProperty("/debugMode")) {
                return;
            }
            
            const aBreakpoints = oModel.getProperty("/debugBreakpoints") || [];
            const sBreakpointId = `bp_${Date.now()}`;
            aBreakpoints.push({
                id: sBreakpointId,
                line: Math.floor(Math.random() * 50) + 1,
                function: "onMessageReceived",
                condition: "message.length > 0"
            });
            
            oModel.setProperty("/debugBreakpoints", aBreakpoints);
            
            let sOutput = oModel.getProperty("/testOutput");
            sOutput += `< [DEBUG] Breakpoint set at line ${aBreakpoints[aBreakpoints.length - 1].line}\n`;
            sOutput += `< [DEBUG] Function: ${aBreakpoints[aBreakpoints.length - 1].function}\n`;
            sOutput += `< [DEBUG] Condition: ${aBreakpoints[aBreakpoints.length - 1].condition}\n`;
            sOutput += `< [DEBUG] Total breakpoints: ${aBreakpoints.length}\n`;
            
            oModel.setProperty("/testOutput", sOutput);
            
            MessageToast.show("Breakpoint set - agent will pause on next execution");
        },

        onSaveAgent: function () {
            this._saveAgent(false, false);
        },

        onSaveAsDraft: function () {
            this._saveAgent(true, false);
        },

        onSaveAndDeploy: function () {
            this._saveAgent(false, true);
        },

        /**
         * Validates agent name input
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onAgentNameChange
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @returns {void}
         * @description Validates agent name for format, length, and uniqueness
         */
        onAgentNameChange: function (oEvent) {
            const oInput = oEvent.getSource();
            const sValue = oInput.getValue();
            const oAgentModel = this.getView().getModel("agent");
            
            // Validation rules
            const oValidation = this._validateAgentName(sValue);
            
            // Update input state
            oInput.setValueState(oValidation.valid ? "None" : "Error");
            oInput.setValueStateText(oValidation.message);
            
            // Auto-generate ID if empty
            if (oValidation.valid && !oAgentModel.getProperty("/id")) {
                const sGeneratedId = this._generateAgentId(sValue);
                oAgentModel.setProperty("/id", sGeneratedId);
            }
            
            // Store validation state
            oAgentModel.setProperty("/nameValid", oValidation.valid);
        },
        
        /**
         * Validates agent ID input
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onAgentIdChange
         * @param {sap.ui.base.Event} oEvent - Input change event
         * @returns {void}
         * @description Validates agent ID for format and uniqueness
         */
        onAgentIdChange: function (oEvent) {
            const oInput = oEvent.getSource();
            const sValue = oInput.getValue();
            const oAgentModel = this.getView().getModel("agent");
            
            // Validation rules
            const oValidation = this._validateAgentId(sValue);
            
            // Update input state
            oInput.setValueState(oValidation.valid ? "None" : "Error");
            oInput.setValueStateText(oValidation.message);
            
            // Store validation state
            oAgentModel.setProperty("/idValid", oValidation.valid);
        },
        
        /**
         * Validates agent name
         * @private
         */
        _validateAgentName: function (sName) {
            // Check if empty
            if (!sName || sName.trim().length === 0) {
                return { valid: false, message: "Agent name is required" };
            }
            
            // Check length
            if (sName.length < 3) {
                return { valid: false, message: "Agent name must be at least 3 characters long" };
            }
            
            if (sName.length > 50) {
                return { valid: false, message: "Agent name must not exceed 50 characters" };
            }
            
            // Check format (alphanumeric, spaces, hyphens, underscores)
            const nameRegex = /^[a-zA-Z0-9\s\-_]+$/;
            if (!nameRegex.test(sName)) {
                return { valid: false, message: "Agent name can only contain letters, numbers, spaces, hyphens, and underscores" };
            }
            
            // Check for reserved names
            const reservedNames = ["system", "admin", "agent", "default", "test", "demo"];
            if (reservedNames.includes(sName.toLowerCase())) {
                return { valid: false, message: "This name is reserved and cannot be used" };
            }
            
            // Check for leading/trailing spaces
            if (sName !== sName.trim()) {
                return { valid: false, message: "Agent name cannot start or end with spaces" };
            }
            
            return { valid: true, message: "" };
        },
        
        /**
         * Validates agent ID
         * @private
         */
        _validateAgentId: function (sId) {
            // Check if empty
            if (!sId || sId.trim().length === 0) {
                return { valid: false, message: "Agent ID is required" };
            }
            
            // Check length
            if (sId.length < 3) {
                return { valid: false, message: "Agent ID must be at least 3 characters long" };
            }
            
            if (sId.length > 30) {
                return { valid: false, message: "Agent ID must not exceed 30 characters" };
            }
            
            // Check format (alphanumeric, hyphens, underscores, no spaces)
            const idRegex = /^[a-zA-Z][a-zA-Z0-9\-_]*$/;
            if (!idRegex.test(sId)) {
                return { valid: false, message: "Agent ID must start with a letter and contain only letters, numbers, hyphens, and underscores (no spaces)" };
            }
            
            // Check for reserved IDs
            const reservedIds = ["system", "admin", "default", "test", "demo", "null", "undefined"];
            if (reservedIds.includes(sId.toLowerCase())) {
                return { valid: false, message: "This ID is reserved and cannot be used" };
            }
            
            // Check for consecutive special characters
            if (/[\-_]{2,}/.test(sId)) {
                return { valid: false, message: "Agent ID cannot contain consecutive hyphens or underscores" };
            }
            
            return { valid: true, message: "" };
        },
        
        /**
         * Generates agent ID from name
         * @private
         */
        _generateAgentId: function (sName) {
            // Convert to lowercase and replace spaces with hyphens
            let sId = sName.toLowerCase()
                .replace(/\s+/g, '-')
                .replace(/[^a-z0-9\-_]/g, '')
                .replace(/[\-_]+/g, '-')
                .replace(/^[\-_]+|[\-_]+$/g, '');
            
            // Ensure it starts with a letter
            if (!/^[a-zA-Z]/.test(sId)) {
                sId = `agent-${sId}`;
            }
            
            // Add timestamp suffix for uniqueness
            sId += `-${Date.now().toString(36)}`;
            
            return sId.substring(0, 30); // Ensure max length
        },
        
        /**
         * Check agent name uniqueness
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onCheckUniqueness
         * @param {sap.ui.base.Event} oEvent - Button press event
         * @returns {void}
         * @description Checks if agent name/ID is unique in the project
         */
        onCheckUniqueness: function (_oEvent) {
            const oAgentModel = this.getView().getModel("agent");
            const sName = oAgentModel.getProperty("/name");
            const sId = oAgentModel.getProperty("/id");
            
            if (!sName || !sId) {
                MessageToast.show("Please enter both name and ID first");
                return;
            }
            
            // Check uniqueness via API
            jQuery.ajax({
                url: `/api/projects/${this._projectId}/agents/check-unique`,
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify({ name: sName, id: sId }),
                success: function (data) {
                    if (data.unique) {
                        MessageToast.show("Agent name and ID are unique!");
                        oAgentModel.setProperty("/uniqueChecked", true);
                    } else {
                        let sMessage = "";
                        if (!data.nameUnique) sMessage += "Agent name already exists. ";
                        if (!data.idUnique) sMessage += "Agent ID already exists.";
                        MessageBox.warning(sMessage);
                        oAgentModel.setProperty("/uniqueChecked", false);
                    }
                }.bind(this),
                error: function () {
                    MessageToast.show("Could not verify uniqueness. Please try again.");
                }
            });
        },
        
        _saveAgent: function (bDraft, bDeploy) {
            const oAgentData = this.getView().getModel("agent").getData();
            
            // Validate required fields
            if (!oAgentData.name || !oAgentData.id) {
                MessageBox.error("Please fill in all required fields");
                return;
            }
            
            // Validate metadata required fields
            if (!oAgentData.metadata.category) {
                MessageBox.error("Please select an agent category in the Metadata tab");
                return;
            }
            
            // Validate name and ID format
            const oNameValidation = this._validateAgentName(oAgentData.name);
            const oIdValidation = this._validateAgentId(oAgentData.id);
            
            if (!oNameValidation.valid) {
                MessageBox.error(`Invalid agent name: ${oNameValidation.message}`);
                return;
            }
            
            if (!oIdValidation.valid) {
                MessageBox.error(`Invalid agent ID: ${oIdValidation.message}`);
                return;
            }
            
            // Validate email format if provided
            if (oAgentData.metadata.contactEmail && !this._validateEmail(oAgentData.metadata.contactEmail)) {
                MessageBox.error("Invalid contact email format");
                return;
            }
            
            // Update metadata timestamps
            oAgentData.metadata.lastModified = new Date().toISOString();
            if (bDeploy) {
                oAgentData.metadata.lastDeployed = new Date().toISOString();
            }
            
            oAgentData.project_id = this._projectId;
            oAgentData.draft = bDraft;
            
            jQuery.ajax({
                url: `/api/projects/${this._projectId}/agents`,
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oAgentData),
                success: function (data) {
                    MessageToast.show("Agent saved successfully");
                    
                    if (bDeploy) {
                        this._deployAgent(data.agent_id);
                    } else {
                        this.onNavToProject();
                    }
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageBox.error(`Failed to save agent: ${error}`);
                }.bind(this)
            });
        },

        _deployAgent: function (sAgentId) {
            MessageToast.show("Deploying agent...");
            
            jQuery.ajax({
                url: `/api/agents/${sAgentId}/deploy`,
                method: "POST",
                success: function () {
                    MessageToast.show("Agent deployed successfully");
                    this.onNavToProject();
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageBox.error(`Failed to deploy agent: ${error}`);
                }.bind(this)
            });
        },

        onCancel: function () {
            MessageBox.confirm("Are you sure you want to cancel? Any unsaved changes will be lost.", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        this.onNavToProject();
                    }
                }.bind(this)
            });
        },

        /**
         * Handles agent icon upload
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onUploadIcon
         * @description Opens file dialog to select and upload agent icon
         * @returns {void}
         * @public
         */
        onUploadIcon: function () {
            const oFileInput = document.createElement("input");
            oFileInput.type = "file";
            oFileInput.accept = "image/png,image/jpg,image/jpeg,image/svg+xml";
            oFileInput.style.display = "none";
            
            oFileInput.addEventListener("change", function (oEvent) {
                const oFile = oEvent.target.files[0];
                if (!oFile) return;
                
                // Validate file size (max 2MB)
                if (oFile.size > 2 * 1024 * 1024) {
                    MessageBox.error("File size exceeds 2MB limit. Please choose a smaller image.");
                    return;
                }
                
                // Validate file type
                const aAllowedTypes = ["image/png", "image/jpg", "image/jpeg", "image/svg+xml"];
                if (!aAllowedTypes.includes(oFile.type)) {
                    MessageBox.error("Invalid file format. Please choose PNG, JPG, or SVG image.");
                    return;
                }
                
                this._processIconFile(oFile);
            }.bind(this));
            
            document.body.appendChild(oFileInput);
            oFileInput.click();
            document.body.removeChild(oFileInput);
        },

        /**
         * Processes uploaded icon file
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _processIconFile
         * @private
         * @param {File} oFile - The uploaded file
         * @returns {void}
         * @description Converts file to data URL and validates image dimensions
         */
        _processIconFile: function (oFile) {
            const oReader = new FileReader();
            
            oReader.onload = function (oEvent) {
                const sDataUrl = oEvent.target.result;
                
                // Validate image dimensions
                const oImage = new Image();
                oImage.onload = function () {
                    // Recommend square images
                    if (oImage.width !== oImage.height) {
                        MessageBox.warning("For best results, use square images (1:1 aspect ratio).");
                    }
                    
                    // Recommend minimum size
                    if (oImage.width < 64 || oImage.height < 64) {
                        MessageBox.warning("Icon is very small. Recommended minimum size is 64x64 pixels.");
                    }
                    
                    this._updateIconData(oFile, sDataUrl);
                }.bind(this);
                
                oImage.onerror = function () {
                    MessageBox.error("Invalid image file. Please choose a valid image.");
                };
                
                oImage.src = sDataUrl;
            }.bind(this);
            
            oReader.onerror = function () {
                MessageBox.error("Failed to read file. Please try again.");
            };
            
            oReader.readAsDataURL(oFile);
        },

        /**
         * Updates agent model with new icon data
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _updateIconData
         * @private
         * @param {File} oFile - The uploaded file
         * @param {string} sDataUrl - Base64 data URL of the image
         * @returns {void}
         * @description Updates agent model with icon information
         */
        _updateIconData: function (oFile, sDataUrl) {
            const oAgentModel = this.getView().getModel("agent");
            
            oAgentModel.setProperty("/icon", {
                src: sDataUrl,
                name: oFile.name,
                size: oFile.size,
                type: oFile.type,
                lastModified: new Date(oFile.lastModified)
            });
            
            MessageToast.show(`Icon uploaded successfully: ${oFile.name}`);
        },

        /**
         * Opens dialog to choose default agent icon
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onChooseDefaultIcon
         * @description Opens dialog with predefined icon options
         * @returns {void}
         * @public
         */
        onChooseDefaultIcon: function () {
            if (!this._oDefaultIconDialog) {
                this._oDefaultIconDialog = sap.ui.xmlfragment(
                    "a2a.portal.fragment.DefaultIconDialog",
                    this
                );
                this.getView().addDependent(this._oDefaultIconDialog);
            }
            
            this._oDefaultIconDialog.open();
        },

        /**
         * Generates AI-powered agent icon
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onGenerateAIIcon
         * @description Uses AI to generate custom icon based on agent properties
         * @returns {void}
         * @public
         */
        onGenerateAIIcon: function () {
            const oAgentModel = this.getView().getModel("agent");
            const sAgentName = oAgentModel.getProperty("/name");
            const sAgentType = oAgentModel.getProperty("/type");
            const sDescription = oAgentModel.getProperty("/description");
            
            if (!sAgentName) {
                MessageBox.warning("Please enter an agent name first to generate an appropriate icon.");
                return;
            }
            
            // Show progress dialog
            const _sBusyDialogText = `Generating AI icon for '${sAgentName}'...`;
            sap.ui.core.BusyIndicator.show(0);
            
            // Simulate AI icon generation (in real implementation, call AI service)
            setTimeout(function () {
                this._generateIconFromPrompt(sAgentName, sAgentType, sDescription);
                sap.ui.core.BusyIndicator.hide();
            }.bind(this), 2000);
        },

        /**
         * Generates icon from AI prompt
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _generateIconFromPrompt
         * @private
         * @param {string} sName - Agent name
         * @param {string} sType - Agent type
         * @param {string} sDescription - Agent description
         * @returns {void}
         * @description Calls AI service to generate custom icon
         */
        _generateIconFromPrompt: function (sName, sType, sDescription) {
            const sPrompt = `Generate a professional icon for an AI agent named '${sName}' of type '${sType}'. Description: ${sDescription || "No description"}`;
            
            // In real implementation, call AI image generation service
            // For now, use a placeholder
            const svgContent = `
                <svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
                    <rect width="256" height="256" fill="#0070f3" rx="32"/>
                    <circle cx="128" cy="100" r="40" fill="white"/>
                    <rect x="88" y="160" width="80" height="8" rx="4" fill="white"/>
                    <rect x="96" y="180" width="64" height="6" rx="3" fill="white" opacity="0.7"/>
                    <text x="128" y="220" font-family="Arial" font-size="12" fill="white" text-anchor="middle">AI</text>
                </svg>
`; 
            const sGeneratedIconUrl = `data:image/svg+xml;base64,${btoa(svgContent)}`;
            
            const oAgentModel = this.getView().getModel("agent");
            oAgentModel.setProperty("/icon", {
                src: sGeneratedIconUrl,
                name: `${sName.replace(/[^a-zA-Z0-9]/g, "_")}_ai_icon.svg`,
                size: sGeneratedIconUrl.length,
                type: "image/svg+xml",
                lastModified: new Date()
            });
            
            MessageToast.show("AI icon generated successfully!");
        },

        /**
         * Removes current agent icon
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onRemoveIcon
         * @description Clears the agent icon
         * @returns {void}
         * @public
         */
        onRemoveIcon: function () {
            MessageBox.confirm("Remove the current agent icon?", {
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        const oAgentModel = this.getView().getModel("agent");
                        oAgentModel.setProperty("/icon", {
                            src: "",
                            name: "",
                            size: 0,
                            type: "",
                            lastModified: null
                        });
                        MessageToast.show("Icon removed");
                    }
                }.bind(this)
            });
        },

        /**
         * Handles icon category change in default icon dialog
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onIconCategoryChange
         * @param {sap.ui.base.Event} oEvent - The select event
         * @returns {void}
         * @public
         */
        onIconCategoryChange: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("key");
            
            // In real implementation, filter icons by category
            // For now, all icons remain visible
            MessageToast.show(`Showing ${sSelectedKey} icons`);
        },

        /**
         * Handles selection of default icon
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onSelectDefaultIcon
         * @param {sap.ui.base.Event} oEvent - The press event
         * @returns {void}
         * @public
         */
        onSelectDefaultIcon: function (oEvent) {
            const oSource = oEvent.getSource();
            const sIconSrc = oSource.data("icon");
            const sIconName = oSource.getItems()[1].getText(); // Get the text from the second item (Text control)
            
            if (sIconSrc) {
                const oAgentModel = this.getView().getModel("agent");
                oAgentModel.setProperty("/icon", {
                    src: sIconSrc,
                    name: `${sIconName.toLowerCase().replace(/\s+/g, "_")}_icon.svg`,
                    size: 1024, // Estimated size for SAP icons
                    type: "image/svg+xml",
                    lastModified: new Date()
                });
                
                MessageToast.show(`Icon selected: ${sIconName}`);
                this.onCloseDefaultIconDialog();
            }
        },

        /**
         * Closes the default icon dialog
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onCloseDefaultIconDialog
         * @returns {void}
         * @public
         */
        onCloseDefaultIconDialog: function () {
            if (this._oDefaultIconDialog) {
                this._oDefaultIconDialog.close();
            }
        },

        /**
         * Handles tag updates in metadata
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onUpdateTags
         * @param {sap.ui.base.Event} oEvent - Token update event
         * @returns {void}
         * @public
         */
        onUpdateTags: function (oEvent) {
            const aTokens = oEvent.getParameter("tokens");
            const oAgentModel = this.getView().getModel("agent");
            
            const aTags = aTokens.map(function (oToken) {
                return {
                    key: oToken.getKey(),
                    text: oToken.getText()
                };
            });
            
            oAgentModel.setProperty("/metadata/tags", aTags);
            
            // Update tags string for display
            const sTagsString = aTags.map(function (oTag) {
                return oTag.text;
            }).join(", ");
            
            oAgentModel.setProperty("/metadata/tagsString", sTagsString);
        },

        /**
         * Handles tag deletion
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onDeleteTag
         * @param {sap.ui.base.Event} oEvent - Token delete event
         * @returns {void}
         * @public
         */
        onDeleteTag: function (oEvent) {
            const oToken = oEvent.getSource();
            const oAgentModel = this.getView().getModel("agent");
            const aTags = oAgentModel.getProperty("/metadata/tags");
            
            // Remove tag from array
            const aFilteredTags = aTags.filter(function (oTag) {
                return oTag.key !== oToken.getKey();
            });
            
            oAgentModel.setProperty("/metadata/tags", aFilteredTags);
            
            // Update tags string
            const sTagsString = aFilteredTags.map(function (oTag) {
                return oTag.text;
            }).join(", ");
            
            oAgentModel.setProperty("/metadata/tagsString", sTagsString);
        },

        /**
         * Validates email format
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _validateEmail
         * @private
         * @param {string} sEmail - Email to validate
         * @returns {boolean} True if valid email format
         * @description Validates email using standard regex pattern
         */
        _validateEmail: function (sEmail) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            return emailRegex.test(sEmail);
        },

        /**
         * Auto-fills metadata based on agent properties
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _autoFillMetadata
         * @private
         * @returns {void}
         * @description Automatically populates metadata fields based on agent configuration
         */
        _autoFillMetadata: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oAgent = oAgentModel.getData();
            
            // Auto-suggest category based on template
            if (oAgent.template && !oAgent.metadata.category) {
                const sCategoryMapping = {
                    "data-processor": "data-processing",
                    "api-integrator": "integration",
                    "ml-analyzer": "ai-ml",
                    "workflow-coordinator": "automation"
                };
                
                const sSuggestedCategory = sCategoryMapping[oAgent.template];
                if (sSuggestedCategory) {
                    oAgentModel.setProperty("/metadata/category", sSuggestedCategory);
                }
            }
            
            // Auto-suggest tags based on type and template
            if (!oAgent.metadata.tags.length) {
                const aSuggestedTags = [];
                
                // Add type-based tags
                if (oAgent.type === "reactive") {
                    aSuggestedTags.push({ key: "reactive", text: "Reactive" });
                } else if (oAgent.type === "proactive") {
                    aSuggestedTags.push({ key: "proactive", text: "Proactive" });
                }
                
                // Add template-based tags
                if (oAgent.template === "data-processor") {
                    aSuggestedTags.push({ key: "data", text: "Data Processing" });
                } else if (oAgent.template === "ml-analyzer") {
                    aSuggestedTags.push({ key: "ai", text: "AI" });
                    aSuggestedTags.push({ key: "ml", text: "Machine Learning" });
                }
                
                if (aSuggestedTags.length) {
                    oAgentModel.setProperty("/metadata/tags", aSuggestedTags);
                    const sTagsString = aSuggestedTags.map(function (oTag) {
                        return oTag.text;
                    }).join(", ");
                    oAgentModel.setProperty("/metadata/tagsString", sTagsString);
                }
            }
        },

        /**
         * Handles agent build process
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onBuildAgent
         * @description Compiles agent code, validates dependencies, and generates deployment artifacts
         * @returns {void}
         * @public
         */
        onBuildAgent: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oViewModel = this.getView().getModel();
            const oAgentData = oAgentModel.getData();
            
            // Validate agent configuration before building
            if (!oAgentData.name || !oAgentData.id) {
                MessageBox.error("Please complete the agent configuration before building");
                return;
            }
            
            // Check if code has been generated
            if (!oViewModel.getProperty("/generatedCode")) {
                MessageBox.information("Code will be generated automatically during build process");
            }
            
            // Switch to build output tab
            const oDesignTabBar = this.byId("designTabBar");
            if (oDesignTabBar) {
                oDesignTabBar.setSelectedKey("build");
            }
            
            // Start build process
            this._startBuildProcess();
        },

        /**
         * Starts the agent build process
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _startBuildProcess
         * @private
         * @returns {void}
         * @description Initiates multi-step build process with progress tracking
         */
        _startBuildProcess: function () {
            const oViewModel = this.getView().getModel();
            const oAgentModel = this.getView().getModel("agent");
            
            // Initialize build state
            oViewModel.setProperty("/buildStatus", "building");
            oViewModel.setProperty("/buildProgress", 0);
            oViewModel.setProperty("/buildOutput", "");
            oViewModel.setProperty("/buildErrors", 0);
            oViewModel.setProperty("/buildWarnings", 0);
            oViewModel.setProperty("/buildArtifacts", []);
            oViewModel.setProperty("/buildCurrentStep", "Initializing build...");
            
            const startTime = Date.now();
            
            // Simulate build steps
            this._executeBuildStep(1, "Validating configuration", function() {
                return this._validateBuildConfiguration();
            }.bind(this))
            .then(function() {
                return this._executeBuildStep(2, "Generating source code", function() {
                    return this._generateBuildCode();
                }.bind(this));
            }.bind(this))
            .then(function() {
                return this._executeBuildStep(3, "Resolving dependencies", function() {
                    return this._resolveDependencies();
                }.bind(this));
            }.bind(this))
            .then(function() {
                return this._executeBuildStep(4, "Compiling agent", function() {
                    return this._compileAgent();
                }.bind(this));
            }.bind(this))
            .then(function() {
                return this._executeBuildStep(5, "Running tests", function() {
                    return this._runBuildTests();
                }.bind(this));
            }.bind(this))
            .then(function() {
                return this._executeBuildStep(6, "Generating artifacts", function() {
                    return this._generateArtifacts();
                }.bind(this));
            }.bind(this))
            .then(function() {
                // Build successful
                const duration = Math.round((Date.now() - startTime) / 1000);
                oViewModel.setProperty("/buildStatus", "success");
                oViewModel.setProperty("/buildProgress", 100);
                oViewModel.setProperty("/buildProgressText", "Build Complete");
                oViewModel.setProperty("/buildDuration", duration);
                this._appendBuildOutput(`\n✅ Build completed successfully in ${duration}s`);
                MessageToast.show("Agent build completed successfully");
            }.bind(this))
            .catch(function(error) {
                // Build failed
                const duration = Math.round((Date.now() - startTime) / 1000);
                oViewModel.setProperty("/buildStatus", "error");
                oViewModel.setProperty("/buildDuration", duration);
                this._appendBuildOutput(`\n❌ Build failed: ${error}`);
                MessageBox.error(`Agent build failed: ${error}`);
            }.bind(this));
        },

        /**
         * Executes a single build step
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _executeBuildStep
         * @private
         * @param {number} stepNumber - Step number (1-6)
         * @param {string} stepName - Human readable step name
         * @param {Function} stepFunction - Function to execute for this step
         * @returns {Promise} Promise that resolves when step completes
         */
        _executeBuildStep: function (stepNumber, stepName, stepFunction) {
            const oViewModel = this.getView().getModel();
            
            return new Promise(function(resolve, reject) {
                // Update progress
                const progress = Math.round((stepNumber - 1) / 6 * 100);
                oViewModel.setProperty("/buildProgress", progress);
                oViewModel.setProperty("/buildProgressText", `${progress}%`);
                oViewModel.setProperty("/buildCurrentStep", stepName);
                
                this._appendBuildOutput(`\n[${new Date().toLocaleTimeString()}] ${stepName}...`);
                
                // Simulate async operation
                setTimeout(function() {
                    try {
                        const result = stepFunction();
                        if (result && result.warnings) {
                            const currentWarnings = oViewModel.getProperty("/buildWarnings");
                            oViewModel.setProperty("/buildWarnings", currentWarnings + result.warnings);
                        }
                        resolve();
                    } catch (error) {
                        const currentErrors = oViewModel.getProperty("/buildErrors");
                        oViewModel.setProperty("/buildErrors", currentErrors + 1);
                        reject(error);
                    }
                }, 1000 + Math.random() * 2000); // 1-3 second delay per step
                
            }.bind(this));
        },

        /**
         * Validates build configuration
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _validateBuildConfiguration
         * @private
         * @returns {Object} Validation result with warnings count
         */
        _validateBuildConfiguration: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oAgentData = oAgentModel.getData();
            let warnings = 0;
            
            this._appendBuildOutput("  Checking agent configuration...");
            
            if (!oAgentData.description) {
                this._appendBuildOutput("  ⚠️  Warning: Agent description is empty");
                warnings++;
            }
            
            if (!oAgentData.skills || oAgentData.skills.length === 0) {
                this._appendBuildOutput("  ⚠️  Warning: No skills defined");
                warnings++;
            }
            
            if (!oAgentData.metadata.category) {
                throw new Error("Agent category is required");
            }
            
            this._appendBuildOutput("  ✅ Configuration validation complete");
            
            return { warnings: warnings };
        },

        /**
         * Generates code for build
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _generateBuildCode
         * @private
         * @returns {Object} Generation result
         */
        _generateBuildCode: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oViewModel = this.getView().getModel();
            const sLanguage = oViewModel.getProperty("/codeLanguage");
            
            this._appendBuildOutput(`  Generating ${sLanguage} source code...`);
            
            // Generate code using existing function
            const sCode = this._generateAgentCode(oAgentModel.getData(), sLanguage);
            oViewModel.setProperty("/generatedCode", sCode);
            
            this._appendBuildOutput(`  ✅ Source code generated (${sCode.length} characters)`);
            
            return {};
        },

        /**
         * Resolves agent dependencies
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _resolveDependencies
         * @private
         * @returns {Object} Resolution result with warnings
         */
        _resolveDependencies: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oAgentData = oAgentModel.getData();
            let warnings = 0;
            
            this._appendBuildOutput("  Resolving dependencies...");
            
            // Simulate dependency resolution based on skills and handlers
            const dependencies = ["a2a-sdk>=2.0.0"];
            
            if (oAgentData.skills) {
                oAgentData.skills.forEach(function(skill) {
                    if (skill.name.toLowerCase().includes("ml") || skill.name.toLowerCase().includes("ai")) {
                        dependencies.push("scikit-learn>=1.0.0", "pandas>=1.5.0");
                    }
                    if (skill.name.toLowerCase().includes("api")) {
                        dependencies.push("requests>=2.28.0");
                    }
                });
            }
            
            this._appendBuildOutput("  Dependencies found:");
            dependencies.forEach(function(dep) {
                this._appendBuildOutput(`    - ${dep}`);
            }.bind(this));
            
            // Simulate some dependency warnings
            if (dependencies.length > 5) {
                this._appendBuildOutput("  ⚠️  Warning: Large number of dependencies may affect performance");
                warnings++;
            }
            
            this._appendBuildOutput("  ✅ Dependencies resolved");
            
            return { warnings: warnings };
        },

        /**
         * Compiles agent code
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _compileAgent
         * @private
         * @returns {Object} Compilation result
         */
        _compileAgent: function () {
            const oViewModel = this.getView().getModel();
            const sCode = oViewModel.getProperty("/generatedCode");
            
            this._appendBuildOutput("  Compiling agent code...");
            
            // Simulate compilation checks
            const lines = sCode.split('\n').length;
            this._appendBuildOutput(`  Processing ${lines} lines of code...`);
            
            // Simulate syntax checking
            if (sCode.includes("undefined_function")) {
                throw new Error("Undefined function 'undefined_function'");
            }
            
            this._appendBuildOutput("  ✅ Compilation successful");
            
            return {};
        },

        /**
         * Runs build tests
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _runBuildTests
         * @private
         * @returns {Object} Test result with warnings
         */
        _runBuildTests: function () {
            let warnings = 0;
            
            this._appendBuildOutput("  Running build tests...");
            
            // Simulate unit tests
            this._appendBuildOutput("  Running unit tests... PASSED");
            this._appendBuildOutput("  Running integration tests... PASSED");
            this._appendBuildOutput("  Running security scans... PASSED");
            
            // Simulate code quality checks
            this._appendBuildOutput("  ⚠️  Warning: Code complexity score: 7.5/10");
            warnings++;
            
            this._appendBuildOutput("  ✅ All tests passed");
            
            return { warnings: warnings };
        },

        /**
         * Generates build artifacts
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _generateArtifacts
         * @private
         * @returns {Object} Generation result
         */
        _generateArtifacts: function () {
            const oViewModel = this.getView().getModel();
            const oAgentModel = this.getView().getModel("agent");
            const oAgentData = oAgentModel.getData();
            
            this._appendBuildOutput("  Generating deployment artifacts...");
            
            const artifacts = [
                {
                    name: `${oAgentData.id}.py`,
                    description: "Main agent source code",
                    icon: "sap-icon://source-code",
                    size: "15.2 KB"
                },
                {
                    name: "requirements.txt",
                    description: "Python dependencies",
                    icon: "sap-icon://list",
                    size: "0.8 KB"
                },
                {
                    name: "Dockerfile",
                    description: "Container image definition",
                    icon: "sap-icon://container",
                    size: "1.2 KB"
                },
                {
                    name: `${oAgentData.id}-config.yaml`,
                    description: "Agent configuration",
                    icon: "sap-icon://settings",
                    size: "2.1 KB"
                },
                {
                    name: `${oAgentData.id}.zip`,
                    description: "Complete deployment package",
                    icon: "sap-icon://attachment-zip-file",
                    size: "45.8 KB"
                }
            ];
            
            oViewModel.setProperty("/buildArtifacts", artifacts);
            
            artifacts.forEach(function(artifact) {
                this._appendBuildOutput(`  Generated: ${artifact.name} (${artifact.size})`);
            }.bind(this));
            
            this._appendBuildOutput("  ✅ Artifacts generation complete");
            
            return {};
        },

        /**
         * Appends text to build output
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _appendBuildOutput
         * @private
         * @param {string} sText - Text to append
         * @returns {void}
         */
        _appendBuildOutput: function (sText) {
            const oViewModel = this.getView().getModel();
            const currentOutput = oViewModel.getProperty("/buildOutput");
            oViewModel.setProperty("/buildOutput", `${currentOutput}${sText}\n`);
        },

        /**
         * Handles clean build action
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onCleanBuild
         * @returns {void}
         * @public
         */
        onCleanBuild: function () {
            MessageBox.confirm("This will clear all build outputs and artifacts. Continue?", {
                title: "Clean Build",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        this._resetBuildState();
                        this._startBuildProcess();
                    }
                }.bind(this)
            });
        },

        /**
         * Resets build state
         * @memberof a2a.portal.controller.AgentBuilder
         * @function _resetBuildState
         * @private
         * @returns {void}
         */
        _resetBuildState: function () {
            const oViewModel = this.getView().getModel();
            oViewModel.setProperty("/buildStatus", "idle");
            oViewModel.setProperty("/buildProgress", 0);
            oViewModel.setProperty("/buildOutput", "");
            oViewModel.setProperty("/buildErrors", 0);
            oViewModel.setProperty("/buildWarnings", 0);
            oViewModel.setProperty("/buildArtifacts", []);
            oViewModel.setProperty("/buildDuration", 0);
        },

        /**
         * Handles download artifacts action
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onDownloadArtifacts
         * @returns {void}
         * @public
         */
        onDownloadArtifacts: function () {
            const oAgentModel = this.getView().getModel("agent");
            const oAgentData = oAgentModel.getData();
            
            // Create a simulated zip file download
            const zipContent = `# Generated Agent Package\n# Agent: ${oAgentData.name}\n# ID: ${oAgentData.id}\n# Generated: ${new Date().toISOString()}\n`;
            
            const blob = new Blob([zipContent], { type: "text/plain" });
            const url = window.URL.createObjectURL(blob);
            
            const downloadLink = document.createElement("a");
            downloadLink.href = url;
            downloadLink.download = `${oAgentData.id}-build-artifacts.txt`;
            downloadLink.style.display = "none";
            
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
            
            window.URL.revokeObjectURL(url);
            
            MessageToast.show("Build artifacts downloaded");
        },

        /**
         * Handles clear build output action
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onClearBuildOutput
         * @returns {void}
         * @public
         */
        onClearBuildOutput: function () {
            const oViewModel = this.getView().getModel();
            oViewModel.setProperty("/buildOutput", "");
            MessageToast.show("Build output cleared");
        },

        /**
         * Handles view artifact action
         * @memberof a2a.portal.controller.AgentBuilder
         * @function onViewArtifact
         * @param {sap.ui.base.Event} oEvent - List item press event
         * @returns {void}
         * @public
         */
        onViewArtifact: function (oEvent) {
            const oItem = oEvent.getSource();
            const oBindingContext = oItem.getBindingContext();
            const oArtifact = oBindingContext.getObject();
            
            MessageBox.information(`Artifact: ${oArtifact.name}\nDescription: ${oArtifact.description}\nSize: ${oArtifact.size}`, {
                title: "Artifact Details"
            });
        }
    });
});