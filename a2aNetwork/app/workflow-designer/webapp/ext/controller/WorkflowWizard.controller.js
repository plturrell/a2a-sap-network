sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/m/Input",
    "sap/m/TextArea",
    "sap/m/ComboBox",
    "sap/m/CheckBox",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Label",
    "sap/m/Text",
    "sap/m/Button",
    "sap/ui/core/Item",
    "sap/ui/layout/form/SimpleForm"
], function (Controller, JSONModel, MessageToast, MessageBox, Fragment, Input, TextArea, ComboBox, CheckBox, VBox, HBox, Label, Text, Button, Item, SimpleForm) {
    "use strict";

    return Controller.extend("a2a.workflow.designer.ext.controller.WorkflowWizard", {

        onOpenWizard: function () {
            if (!this._oWizardDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.workflow.designer.ext.fragment.WorkflowWizard",
                    controller: this
                }).then(function (oDialog) {
                    this._oWizardDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeWizard();
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeWizard();
                this._oWizardDialog.open();
            }
        },

        _initializeWizard: function () {
            const oWizardModel = new JSONModel({
                useCase: null,
                workflow: {
                    name: "",
                    description: "",
                    strategy: "sequential",
                    timeoutMinutes: 30,
                    tasks: []
                },
                availableAgents: this._getAvailableAgents(),
                selectedAgents: [],
                configuration: {}
            });

            this._oWizardDialog.setModel(oWizardModel, "wizard");
            
            // Reset wizard to first step
            const oWizard = this.byId("workflowWizard");
            oWizard.discardProgress(this.byId("useCaseStep"));
            
            this._updateWizardButtons();
        },

        _getAvailableAgents: function () {
            return [
                {
                    id: 0,
                    name: "Data Product Agent",
                    capabilities: ["data_processing", "document_extraction"],
                    status: "available",
                    port: 8000,
                    required: false
                },
                {
                    id: 1,
                    name: "Data Standardization",
                    capabilities: ["data_standardization", "validation"],
                    status: "available",
                    port: 8001,
                    required: false
                },
                {
                    id: 2,
                    name: "AI Preparation",
                    capabilities: ["semantic_chunking", "embedding_generation"],
                    status: "available",
                    port: 8002,
                    required: false
                },
                {
                    id: 3,
                    name: "Vector Processing",
                    capabilities: ["vector_operations", "similarity_search"],
                    status: "available",
                    port: 8003,
                    required: false
                },
                {
                    id: 4,
                    name: "Calculation Validation",
                    capabilities: ["mathematical_validation", "computation"],
                    status: "available",
                    port: 8004,
                    required: false
                },
                {
                    id: 5,
                    name: "QA Validation",
                    capabilities: ["quality_assurance", "testing"],
                    status: "available",
                    port: 8005,
                    required: false
                },
                {
                    id: 9,
                    name: "Reasoning Agent",
                    capabilities: ["logical_reasoning", "inference"],
                    status: "available",
                    port: 8009,
                    required: false
                }
            ];
        },

        onUseCaseSelect: function (oEvent) {
            const sUseCaseId = oEvent.getSource().getId().replace(this.getView().getId() + "--", "").replace("Panel", "");
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            
            const oUseCase = this._getUseCaseConfig(sUseCaseId);
            oWizardModel.setProperty("/useCase", oUseCase);
            
            // Update workflow with use case defaults
            oWizardModel.setProperty("/workflow/name", oUseCase.defaultName);
            oWizardModel.setProperty("/workflow/description", oUseCase.defaultDescription);
            oWizardModel.setProperty("/workflow/strategy", oUseCase.strategy);
            oWizardModel.setProperty("/workflow/tasks", oUseCase.defaultTasks);
            
            // Pre-select required agents
            const aSelectedAgents = oWizardModel.getProperty("/selectedAgents");
            aSelectedAgents.length = 0; // Clear existing selections
            
            oUseCase.requiredAgents.forEach(function(iAgentId) {
                const aAgents = oWizardModel.getProperty("/availableAgents");
                const oAgent = aAgents.find(function(a) { return a.id === iAgentId; });
                if (oAgent) {
                    oAgent.required = true;
                    aSelectedAgents.push(oAgent);
                }
            });
            
            oWizardModel.refresh();
            
            // Move to next step
            const oWizard = this.byId("workflowWizard");
            oWizard.nextStep();
            this._generateConfigurationContent(oUseCase);
        },

        _getUseCaseConfig: function (sUseCaseId) {
            const oUseCases = {
                docProcessing: {
                    id: "docProcessing",
                    name: "Document Processing",
                    defaultName: "Document Processing Workflow",
                    defaultDescription: "Extract, chunk, and process documents for AI workflows",
                    strategy: "sequential",
                    requiredAgents: [0, 1, 2, 3],
                    defaultTasks: [
                        { id: "extract", name: "Extract Document", agent: 0, action: "extractDocument" },
                        { id: "standardize", name: "Standardize Data", agent: 1, action: "standardizeData", dependencies: ["extract"] },
                        { id: "chunk", name: "Semantic Chunking", agent: 2, action: "performSemanticChunking", dependencies: ["standardize"] },
                        { id: "embed", name: "Generate Embeddings", agent: 3, action: "createEmbeddings", dependencies: ["chunk"] }
                    ],
                    parameters: {
                        documentFormat: { type: "select", options: ["pdf", "docx", "txt", "html"], default: "pdf" },
                        chunkingStrategy: { type: "select", options: ["semantic_similarity", "fixed_size", "sentence"], default: "semantic_similarity" },
                        embeddingModel: { type: "select", options: ["all-mpnet-base-v2", "all-MiniLM-L6-v2"], default: "all-mpnet-base-v2" }
                    }
                },
                rag: {
                    id: "rag",
                    name: "RAG Pipeline",
                    defaultName: "RAG Workflow",
                    defaultDescription: "Retrieval-Augmented Generation for question answering",
                    strategy: "sequential",
                    requiredAgents: [2, 3, 9],
                    defaultTasks: [
                        { id: "search", name: "Vector Search", agent: 3, action: "searchVectors" },
                        { id: "retrieve", name: "Retrieve Context", agent: 2, action: "retrieveContext", dependencies: ["search"] },
                        { id: "generate", name: "Generate Response", agent: 9, action: "generateWithContext", dependencies: ["retrieve"] }
                    ],
                    parameters: {
                        topK: { type: "number", default: 10, min: 1, max: 50 },
                        similarityThreshold: { type: "number", default: 0.7, min: 0, max: 1, step: 0.1 },
                        maxTokens: { type: "number", default: 2000, min: 100, max: 8000 }
                    }
                },
                dataIntegration: {
                    id: "dataIntegration",
                    name: "Data Integration",
                    defaultName: "Data Integration Workflow",
                    defaultDescription: "Integrate and transform data from multiple sources",
                    strategy: "parallel",
                    requiredAgents: [0, 1, 5],
                    defaultTasks: [
                        { id: "extract", name: "Extract Data", agent: 0, action: "extractData" },
                        { id: "transform", name: "Transform Data", agent: 1, action: "transformData", dependencies: ["extract"] },
                        { id: "validate", name: "Validate Quality", agent: 5, action: "validateQuality", dependencies: ["transform"] }
                    ],
                    parameters: {
                        sourceType: { type: "select", options: ["database", "api", "file"], default: "database" },
                        transformationRules: { type: "text", default: "" },
                        validationLevel: { type: "select", options: ["basic", "strict", "comprehensive"], default: "basic" }
                    }
                },
                qualityAssurance: {
                    id: "qualityAssurance",
                    name: "Quality Assurance",
                    defaultName: "Quality Assurance Workflow",
                    defaultDescription: "Validate and improve data quality across systems",
                    strategy: "sequential",
                    requiredAgents: [0, 5, 4],
                    defaultTasks: [
                        { id: "assess", name: "Assess Quality", agent: 5, action: "assessQuality" },
                        { id: "validate", name: "Validate Calculations", agent: 4, action: "validateCalculations", dependencies: ["assess"] },
                        { id: "report", name: "Generate Report", agent: 0, action: "generateReport", dependencies: ["validate"] }
                    ],
                    parameters: {
                        qualityMetrics: { type: "multiselect", options: ["completeness", "accuracy", "consistency", "validity"], default: ["completeness", "accuracy"] },
                        reportFormat: { type: "select", options: ["json", "pdf", "html"], default: "json" }
                    }
                },
                custom: {
                    id: "custom",
                    name: "Custom Workflow",
                    defaultName: "Custom Workflow",
                    defaultDescription: "Create a custom workflow from scratch",
                    strategy: "sequential",
                    requiredAgents: [],
                    defaultTasks: [],
                    parameters: {}
                },
                nl: {
                    id: "nl",
                    name: "Natural Language",
                    defaultName: "AI Generated Workflow",
                    defaultDescription: "Workflow generated from natural language description",
                    strategy: "sequential",
                    requiredAgents: [],
                    defaultTasks: [],
                    parameters: {
                        description: { type: "textarea", placeholder: "Describe your workflow requirements in natural language..." }
                    }
                }
            };

            return oUseCases[sUseCaseId] || oUseCases.custom;
        },

        _generateConfigurationContent: function (oUseCase) {
            const oConfigContent = this.byId("configurationContent");
            oConfigContent.destroyItems();

            if (oUseCase.id === "nl") {
                this._generateNLConfigContent(oConfigContent, oUseCase);
            } else {
                this._generateStandardConfigContent(oConfigContent, oUseCase);
            }
        },

        _generateNLConfigContent: function (oConfigContent, oUseCase) {
            const oVBox = new VBox();
            
            const oLabel = new Label({ text: "Workflow Description:" });
            const oTextArea = new TextArea({
                value: "{wizard>/configuration/description}",
                placeholder: "Describe what you want your workflow to accomplish...",
                rows: 8,
                width: "100%",
                liveChange: this.onNLDescriptionChange.bind(this)
            });
            
            oVBox.addItem(oLabel);
            oVBox.addItem(oTextArea);
            
            const oGenerateButton = new Button({
                text: "Generate Workflow",
                type: "Emphasized",
                icon: "sap-icon://artificial-intelligence",
                press: this.onGenerateFromNL.bind(this),
                class: "sapUiMediumMarginTop"
            });
            
            oVBox.addItem(oGenerateButton);
            oConfigContent.addItem(oVBox);
        },

        _generateStandardConfigContent: function (oConfigContent, oUseCase) {
            const oForm = new SimpleForm({
                layout: "ResponsiveGridLayout",
                columnsM: 2,
                columnsL: 2,
                class: "sapUiResponsivePadding"
            });

            Object.keys(oUseCase.parameters).forEach(function (sParamKey) {
                const oParam = oUseCase.parameters[sParamKey];
                const oLabel = new Label({ text: sParamKey + ":" });
                let oControl;

                switch (oParam.type) {
                    case "select":
                        oControl = new ComboBox({
                            selectedKey: "{wizard>/configuration/" + sParamKey + "}",
                            items: oParam.options.map(function (sOption) {
                                return new Item({ key: sOption, text: sOption });
                            })
                        });
                        break;
                    case "number":
                        oControl = new Input({
                            value: "{wizard>/configuration/" + sParamKey + "}",
                            type: "Number"
                        });
                        break;
                    case "textarea":
                        oControl = new TextArea({
                            value: "{wizard>/configuration/" + sParamKey + "}",
                            placeholder: oParam.placeholder || "",
                            rows: 4
                        });
                        break;
                    default:
                        oControl = new Input({
                            value: "{wizard>/configuration/" + sParamKey + "}"
                        });
                }

                oForm.addContent(oLabel);
                oForm.addContent(oControl);
                
                // Set default value
                const oWizardModel = this._oWizardDialog.getModel("wizard");
                oWizardModel.setProperty("/configuration/" + sParamKey, oParam.default);
            }.bind(this));

            oConfigContent.addItem(oForm);
        },

        onNLDescriptionChange: function (oEvent) {
            const sValue = oEvent.getParameter("value");
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            oWizardModel.setProperty("/configuration/description", sValue);
            
            // Enable/disable generate button based on content
            const bHasContent = sValue && sValue.trim().length > 10;
            this.byId("configurationStep").setValidated(bHasContent);
        },

        onGenerateFromNL: function () {
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            const sDescription = oWizardModel.getProperty("/configuration/description");
            
            if (!sDescription || sDescription.trim().length < 10) {
                MessageBox.warning("Please provide a more detailed description of your workflow.");
                return;
            }

            const oModel = this.getView().getModel();
            
            MessageToast.show("Generating workflow from description...");
            
            oModel.callFunction("/generateTemplateFromNL", {
                urlParameters: {
                    description: sDescription,
                    category: "5" // Custom category
                },
                success: function (oResponse) {
                    const oGeneratedTemplate = oResponse.generateTemplateFromNL;
                    
                    // Update wizard model with generated data
                    const oDefinition = JSON.parse(oGeneratedTemplate.definition);
                    oWizardModel.setProperty("/workflow/name", oGeneratedTemplate.name);
                    oWizardModel.setProperty("/workflow/description", oGeneratedTemplate.description);
                    oWizardModel.setProperty("/workflow/tasks", oDefinition.tasks || []);
                    oWizardModel.setProperty("/workflow/strategy", oDefinition.strategy || "sequential");
                    
                    // Update required agents
                    const aRequiredAgents = oGeneratedTemplate.requiredAgents || [];
                    this._updateRequiredAgents(aRequiredAgents);
                    
                    MessageToast.show("Workflow generated successfully!");
                    this.byId("configurationStep").setValidated(true);
                }.bind(this),
                error: function (oError) {
                    MessageBox.error("Failed to generate workflow: " + oError.message);
                }
            });
        },

        _updateRequiredAgents: function (aRequiredAgentIds) {
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            const aAvailableAgents = oWizardModel.getProperty("/availableAgents");
            const aSelectedAgents = [];
            
            // Reset required flags
            aAvailableAgents.forEach(function (oAgent) {
                oAgent.required = false;
            });
            
            // Set required agents
            aRequiredAgentIds.forEach(function (iAgentId) {
                const oAgent = aAvailableAgents.find(function (a) { return a.id === iAgentId; });
                if (oAgent) {
                    oAgent.required = true;
                    aSelectedAgents.push(oAgent);
                }
            });
            
            oWizardModel.setProperty("/selectedAgents", aSelectedAgents);
            oWizardModel.refresh();
        },

        onAgentSelectionChange: function (oEvent) {
            const oTable = oEvent.getSource();
            const aSelectedItems = oTable.getSelectedItems();
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            
            const aSelectedAgents = aSelectedItems.map(function (oItem) {
                return oItem.getBindingContext("wizard").getObject();
            });
            
            oWizardModel.setProperty("/selectedAgents", aSelectedAgents);
            
            // Validate step - need at least one agent
            const bValid = aSelectedAgents.length > 0;
            this.byId("agentSelectionStep").setValidated(bValid);
        },

        onWorkflowNameChange: function (oEvent) {
            const sValue = oEvent.getParameter("value");
            const bValid = sValue && sValue.trim().length > 0;
            this.byId("reviewStep").setValidated(bValid);
        },

        onStepActivate: function (oEvent) {
            const oStep = oEvent.getParameter("step");
            const sStepId = oStep.getId();
            
            if (sStepId.includes("agentSelectionStep")) {
                this._populateAgentTable();
            }
            
            this._updateWizardButtons();
        },

        _populateAgentTable: function () {
            const oTable = this.byId("agentSelectionTable");
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            const aAgents = oWizardModel.getProperty("/availableAgents");
            const aSelectedAgents = oWizardModel.getProperty("/selectedAgents");
            
            oTable.destroyItems();
            
            aAgents.forEach(function (oAgent, iIndex) {
                const oItem = new sap.m.ColumnListItem({
                    cells: [
                        new sap.m.Text({ text: oAgent.name }),
                        new sap.m.Text({ text: oAgent.capabilities.join(", ") }),
                        new sap.m.ObjectStatus({ 
                            text: oAgent.status,
                            state: oAgent.status === "available" ? "Success" : "Error"
                        }),
                        new sap.m.Text({ text: oAgent.required ? "Yes" : "No" })
                    ]
                });
                
                oItem.setBindingContext(new sap.ui.model.Context(oWizardModel, "/availableAgents/" + iIndex), "wizard");
                oTable.addItem(oItem);
                
                // Pre-select required agents and previously selected agents
                if (oAgent.required || aSelectedAgents.some(function (a) { return a.id === oAgent.id; })) {
                    oTable.setSelectedItem(oItem);
                }
            });
        },

        _updateWizardButtons: function () {
            const oWizard = this.byId("workflowWizard");
            const oCurrentStep = oWizard.getCurrentStep();
            const sCurrentStepId = oCurrentStep ? oCurrentStep.getId() : "";
            
            const oNextButton = this.byId("wizardNextButton");
            const oCreateButton = this.byId("wizardCreateButton");
            
            if (sCurrentStepId.includes("reviewStep")) {
                oNextButton.setVisible(false);
                oCreateButton.setVisible(true);
            } else {
                oNextButton.setVisible(true);
                oCreateButton.setVisible(false);
            }
        },

        onWizardNext: function () {
            const oWizard = this.byId("workflowWizard");
            oWizard.nextStep();
            this._updateWizardButtons();
        },

        onWizardCreate: function () {
            const oWizardModel = this._oWizardDialog.getModel("wizard");
            const oWorkflow = oWizardModel.getProperty("/workflow");
            const aSelectedAgents = oWizardModel.getProperty("/selectedAgents");
            const oConfiguration = oWizardModel.getProperty("/configuration");
            
            // Validate required fields
            if (!oWorkflow.name || !oWorkflow.name.trim()) {
                MessageBox.error("Please provide a workflow name.");
                return;
            }
            
            if (aSelectedAgents.length === 0) {
                MessageBox.error("Please select at least one agent.");
                return;
            }
            
            // Create template definition
            const oTemplateDefinition = {
                name: oWorkflow.name,
                tasks: oWorkflow.tasks,
                strategy: oWorkflow.strategy
            };
            
            const oModel = this.getView().getModel();
            
            // Create the template
            oModel.create("/Templates", {
                name: oWorkflow.name,
                description: oWorkflow.description,
                category_ID: "5", // Custom category
                definition: JSON.stringify(oTemplateDefinition),
                parameters: JSON.stringify(oConfiguration),
                requiredAgents: aSelectedAgents.map(function (a) { return a.id; }),
                estimatedDuration: oWorkflow.timeoutMinutes,
                author: "Current User",
                tags: ["custom"],
                isPublic: false
            }, {
                success: function (oResponse) {
                    MessageToast.show("Workflow template created successfully!");
                    this._oWizardDialog.close();
                    
                    // Navigate to the new template
                    const oRouter = this.getOwnerComponent().getRouter();
                    oRouter.navTo("TemplateDetail", {
                        templateId: oResponse.ID
                    });
                }.bind(this),
                error: function (oError) {
                    MessageBox.error("Failed to create workflow template: " + oError.message);
                }
            });
        },

        onWizardCancel: function () {
            this._oWizardDialog.close();
        },

        onWizardClose: function () {
            // Reset wizard state if needed
        }
    });
});