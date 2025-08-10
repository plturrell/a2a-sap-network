sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("a2a.portal.controller.AgentBuilder", {

        onInit: function () {
            // Initialize agent model
            var oAgentModel = new JSONModel({
                name: "",
                id: "",
                type: "reactive",
                description: "",
                template: "blank",
                skills: [],
                handlers: [],
                protocol: "mqtt",
                messageFormat: 0,
                subscribedTopics: "",
                publishTopics: ""
            });
            this.getView().setModel(oAgentModel, "agent");

            // Initialize view model
            var oViewModel = new JSONModel({
                projectName: "",
                codeLanguage: "python",
                generatedCode: "",
                testOutput: ""
            });
            this.getView().setModel(oViewModel);

            // Get router
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("agentBuilder").attachPatternMatched(this._onPatternMatched, this);
        },

        _onPatternMatched: function (oEvent) {
            var sProjectId = oEvent.getParameter("arguments").projectId;
            this._projectId = sProjectId;
            this._loadProjectInfo(sProjectId);
        },

        _loadProjectInfo: function (sProjectId) {
            jQuery.ajax({
                url: "/api/projects/" + sProjectId,
                method: "GET",
                success: function (data) {
                    this.getView().getModel().setProperty("/projectName", data.name);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to load project info: " + error);
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
            var sTemplate = oEvent.getParameter("selectedItem").getKey();
            this._applyTemplate(sTemplate);
        },

        _applyTemplate: function (sTemplate) {
            var oAgentModel = this.getView().getModel("agent");
            
            switch (sTemplate) {
                case "data-processor":
                    oAgentModel.setProperty("/skills", [
                        { name: "Data Ingestion", description: "Consume data from various sources", icon: "sap-icon://download" },
                        { name: "Data Transformation", description: "Transform and clean data", icon: "sap-icon://refresh" },
                        { name: "Data Validation", description: "Validate data quality", icon: "sap-icon://accept" }
                    ]);
                    oAgentModel.setProperty("/type", "reactive");
                    break;
                case "api-integrator":
                    oAgentModel.setProperty("/skills", [
                        { name: "REST API Handler", description: "Handle REST API calls", icon: "sap-icon://cloud" },
                        { name: "Authentication", description: "Manage API authentication", icon: "sap-icon://locked" },
                        { name: "Rate Limiting", description: "Handle API rate limits", icon: "sap-icon://performance" }
                    ]);
                    oAgentModel.setProperty("/protocol", "http");
                    break;
                case "ml-analyzer":
                    oAgentModel.setProperty("/skills", [
                        { name: "Model Inference", description: "Run ML model predictions", icon: "sap-icon://learning-assistant" },
                        { name: "Feature Engineering", description: "Extract and transform features", icon: "sap-icon://settings" },
                        { name: "Result Processing", description: "Process ML results", icon: "sap-icon://chart" }
                    ]);
                    oAgentModel.setProperty("/type", "proactive");
                    break;
                case "workflow-coordinator":
                    oAgentModel.setProperty("/skills", [
                        { name: "Task Orchestration", description: "Coordinate multiple agents", icon: "sap-icon://org-chart" },
                        { name: "State Management", description: "Manage workflow state", icon: "sap-icon://process" },
                        { name: "Error Handling", description: "Handle workflow errors", icon: "sap-icon://alert" }
                    ]);
                    oAgentModel.setProperty("/type", "collaborative");
                    break;
            }
            
            MessageToast.show("Template applied: " + sTemplate);
        },

        onAddSkill: function () {
            var that = this;
            if (!this._oAddSkillDialog) {
                sap.ui.core.Fragment.load({
                    name: "a2a.portal.view.fragments.AddSkillDialog",
                    controller: this
                }).then(function (oDialog) {
                    that._oAddSkillDialog = oDialog;
                    that.getView().addDependent(that._oAddSkillDialog);
                    that._oAddSkillDialog.open();
                });
            } else {
                this._oAddSkillDialog.open();
            }
        },

        onAddSkillConfirm: function () {
            var sName = sap.ui.getCore().byId("skillName").getValue();
            var sDescription = sap.ui.getCore().byId("skillDescription").getValue();
            var sType = sap.ui.getCore().byId("skillType").getSelectedKey();
            var sIcon = sap.ui.getCore().byId("skillIcon").getSelectedKey() || "sap-icon://process";
            var bRequired = sap.ui.getCore().byId("skillRequired").getSelected();
            var bAsync = sap.ui.getCore().byId("skillAsync").getSelected();
            
            if (!sName) {
                MessageToast.show("Please enter a skill name");
                return;
            }
            
            var oNewSkill = {
                name: sName,
                description: sDescription || "Custom skill",
                type: sType,
                icon: sIcon,
                required: bRequired,
                async: bAsync
            };
            
            var oModel = this.getView().getModel("agent");
            var aSkills = oModel.getProperty("/skills") || [];
            aSkills.push(oNewSkill);
            oModel.setProperty("/skills", aSkills);
            
            this._oAddSkillDialog.close();
            
            // Clear form
            sap.ui.getCore().byId("skillName").setValue("");
            sap.ui.getCore().byId("skillDescription").setValue("");
            sap.ui.getCore().byId("skillType").setSelectedKey("processing");
            sap.ui.getCore().byId("skillIcon").setSelectedKey("");
            sap.ui.getCore().byId("skillRequired").setSelected(false);
            sap.ui.getCore().byId("skillAsync").setSelected(false);
            
            MessageToast.show("Skill '" + sName + "' added successfully");
        },

        onAddSkillCancel: function () {
            this._oAddSkillDialog.close();
        },

        onDeleteSkill: function (oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var sPath = oItem.getBindingContext("agent").getPath();
            var oModel = this.getView().getModel("agent");
            var aSkills = oModel.getProperty("/skills");
            var iIndex = parseInt(sPath.split("/").pop());
            
            aSkills.splice(iIndex, 1);
            oModel.setProperty("/skills", aSkills);
        },

        onConfigureSkill: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("agent");
            var sSkillName = oContext.getProperty("name");
            MessageToast.show("Configure skill: " + sSkillName);
        },

        onAddHandler: function () {
            var that = this;
            if (!this._oAddHandlerDialog) {
                sap.ui.core.Fragment.load({
                    name: "a2a.portal.view.fragments.AddHandlerDialog",
                    controller: this
                }).then(function (oDialog) {
                    that._oAddHandlerDialog = oDialog;
                    that.getView().addDependent(that._oAddHandlerDialog);
                    that._oAddHandlerDialog.open();
                });
            } else {
                this._oAddHandlerDialog.open();
            }
        },

        onAddHandlerConfirm: function () {
            var sName = sap.ui.getCore().byId("handlerName").getValue();
            var sType = sap.ui.getCore().byId("messageType").getSelectedKey();
            var sPattern = sap.ui.getCore().byId("messagePattern").getValue();
            var iPriority = sap.ui.getCore().byId("handlerPriority").getValue();
            var sCode = sap.ui.getCore().byId("handlerCode").getValue();
            var bActive = sap.ui.getCore().byId("handlerActive").getSelected();
            var bRetry = sap.ui.getCore().byId("handlerRetryOnError").getSelected();
            
            if (!sName || !sPattern) {
                MessageToast.show("Please enter handler name and message pattern");
                return;
            }
            
            var oNewHandler = {
                handler: sName,
                type: sType,
                pattern: sPattern,
                priority: iPriority,
                code: sCode,
                active: bActive,
                retryOnError: bRetry,
                status: bActive ? "Active" : "Inactive"
            };
            
            var oModel = this.getView().getModel("agent");
            var aHandlers = oModel.getProperty("/handlers") || [];
            aHandlers.push(oNewHandler);
            oModel.setProperty("/handlers", aHandlers);
            
            this._oAddHandlerDialog.close();
            
            // Clear form
            sap.ui.getCore().byId("handlerName").setValue("");
            sap.ui.getCore().byId("messageType").setSelectedKey("request");
            sap.ui.getCore().byId("messagePattern").setValue("");
            sap.ui.getCore().byId("handlerPriority").setValue(5);
            sap.ui.getCore().byId("handlerActive").setSelected(true);
            sap.ui.getCore().byId("handlerRetryOnError").setSelected(false);
            
            MessageToast.show("Handler '" + sName + "' added successfully");
        },

        onAddHandlerCancel: function () {
            this._oAddHandlerDialog.close();
        },

        onEditHandler: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("agent");
            var sHandler = oContext.getProperty("handler");
            MessageToast.show("Edit handler: " + sHandler);
        },

        onDeleteHandler: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext("agent");
            var sHandler = oContext.getProperty("handler");
            MessageToast.show("Delete handler: " + sHandler);
        },

        onGenerateCode: function () {
            var oModel = this.getView().getModel();
            var oAgentModel = this.getView().getModel("agent");
            var sLanguage = oModel.getProperty("/codeLanguage");
            
            // Simple code generation based on template
            var sCode = this._generateAgentCode(oAgentModel.getData(), sLanguage);
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
            
            return "// Code generation not implemented for " + sLanguage;
        },

        _toPascalCase: function (str) {
            return str.replace(/[-_](.)/g, function (match, chr) {
                return chr.toUpperCase();
            }).replace(/^(.)/, function (match, chr) {
                return chr.toUpperCase();
            });
        },

        onCopyCode: function () {
            var sCode = this.getView().getModel().getProperty("/generatedCode");
            if (sCode) {
                navigator.clipboard.writeText(sCode).then(function () {
                    MessageToast.show("Code copied to clipboard");
                });
            }
        },

        onDownloadCode: function () {
            var sCode = this.getView().getModel().getProperty("/generatedCode");
            var sAgentId = this.getView().getModel("agent").getProperty("/id");
            var sLanguage = this.getView().getModel().getProperty("/codeLanguage");
            
            var sExtension = sLanguage === "python" ? "py" : "js";
            var sFilename = sAgentId + "_agent." + sExtension;
            
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(sCode));
            element.setAttribute('download', sFilename);
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
            
            MessageToast.show("Code downloaded as " + sFilename);
        },

        onStartTest: function () {
            var oModel = this.getView().getModel();
            oModel.setProperty("/testOutput", "Starting agent test...\n");
            
            // Simulate test execution
            setTimeout(function () {
                var sOutput = oModel.getProperty("/testOutput");
                sOutput += "Agent initialized successfully\n";
                sOutput += "Connecting to message broker...\n";
                sOutput += "Connected to MQTT broker\n";
                sOutput += "Agent is ready to receive messages\n";
                oModel.setProperty("/testOutput", sOutput);
            }.bind(this), 1000);
            
            MessageToast.show("Test started");
        },

        onStopTest: function () {
            var oModel = this.getView().getModel();
            var sOutput = oModel.getProperty("/testOutput");
            sOutput += "Stopping agent test...\n";
            sOutput += "Test stopped\n";
            oModel.setProperty("/testOutput", sOutput);
            
            MessageToast.show("Test stopped");
        },

        onClearConsole: function () {
            this.getView().getModel().setProperty("/testOutput", "");
        },

        onSendTestMessage: function (oEvent) {
            var sMessage = oEvent.getParameter("value");
            if (sMessage) {
                var oModel = this.getView().getModel();
                var sOutput = oModel.getProperty("/testOutput");
                sOutput += "> " + sMessage + "\n";
                sOutput += "< Processing message...\n";
                sOutput += "< Message processed successfully\n";
                oModel.setProperty("/testOutput", sOutput);
                
                oEvent.getSource().setValue("");
            }
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

        _saveAgent: function (bDraft, bDeploy) {
            var oAgentData = this.getView().getModel("agent").getData();
            
            // Validate required fields
            if (!oAgentData.name || !oAgentData.id) {
                MessageBox.error("Please fill in all required fields");
                return;
            }
            
            oAgentData.project_id = this._projectId;
            oAgentData.draft = bDraft;
            
            jQuery.ajax({
                url: "/api/projects/" + this._projectId + "/agents",
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
                    MessageBox.error("Failed to save agent: " + error);
                }.bind(this)
            });
        },

        _deployAgent: function (sAgentId) {
            MessageToast.show("Deploying agent...");
            
            jQuery.ajax({
                url: "/api/agents/" + sAgentId + "/deploy",
                method: "POST",
                success: function () {
                    MessageToast.show("Agent deployed successfully");
                    this.onNavToProject();
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageBox.error("Failed to deploy agent: " + error);
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
        }
    });
});