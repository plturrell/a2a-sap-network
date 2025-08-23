sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (Controller, JSONModel, MessageToast, MessageBox) => {
    "use strict";
/* No global declarations needed - jQuery is a built-in global */

    return Controller.extend("a2a.portal.controller.CodeEditor", {

        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                projectName: "",
                fileTree: [],
                openTabs: [],
                activeTab: null,
                currentFile: null,
                editorContent: "",
                editorLanguage: "python",
                isDirty: false,
                cursorPosition: { line: 1, column: 1 },
                outline: [],
                problems: [],
                problemCount: 0,
                gitBranch: "main",
                gitChanges: []
            });
            this.getView().setModel(oViewModel);

            // Get router
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("codeEditor").attachPatternMatched(this._onPatternMatched, this);
        },

        _onPatternMatched: function (oEvent) {
            const oArgs = oEvent.getParameter("args");
            this._projectId = oArgs.projectId;
            const sFilePath = oArgs.filePath;
            
            this._loadProjectInfo(this._projectId);
            this._loadFileTree();
            
            if (sFilePath) {
                this._openFile(sFilePath);
            }
        },

        _loadProjectInfo: function (sProjectId) {
            jQuery.ajax({
                url: `/api/projects/${  sProjectId}`,
                method: "GET",
                success: function (data) {
                    this.getView().getModel().setProperty("/projectName", data.name);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show(`Failed to load project info: ${  error}`);
                }.bind(this)
            });
        },

        _loadFileTree: function () {
            // Simulate file tree
            const aFileTree = [
                {
                    name: "src",
                    icon: "sap-icon://folder",
                    nodes: [
                        {
                            name: "agents",
                            icon: "sap-icon://folder",
                            nodes: [
                                { name: "data_processor.py", icon: "sap-icon://document-text", path: "src/agents/data_processor.py" },
                                { name: "api_integrator.py", icon: "sap-icon://document-text", path: "src/agents/api_integrator.py" }
                            ]
                        },
                        {
                            name: "workflows",
                            icon: "sap-icon://folder",
                            nodes: [
                                { name: "main_workflow.bpmn", icon: "sap-icon://process", path: "src/workflows/main_workflow.bpmn" }
                            ]
                        },
                        { name: "__init__.py", icon: "sap-icon://document-text", path: "src/__init__.py" }
                    ]
                },
                {
                    name: "tests",
                    icon: "sap-icon://folder",
                    nodes: [
                        { name: "test_agents.py", icon: "sap-icon://document-text", path: "tests/test_agents.py" }
                    ]
                },
                { name: "README.md", icon: "sap-icon://document", path: "README.md" },
                { name: "requirements.txt", icon: "sap-icon://document-text", path: "requirements.txt" },
                { name: "config.yaml", icon: "sap-icon://action-settings", path: "config.yaml" }
            ];
            
            this.getView().getModel().setProperty("/fileTree", aFileTree);
        },

        _openFile: function (sPath) {
            const oModel = this.getView().getModel();
            const aOpenTabs = oModel.getProperty("/openTabs");
            
            // Check if already open
            const oExistingTab = aOpenTabs.find((tab) => {
                return tab.path === sPath;
            });
            
            if (oExistingTab) {
                oModel.setProperty("/activeTab", oExistingTab.id);
                this._loadFileContent(sPath);
                return;
            }
            
            // Create new tab
            const oNewTab = {
                id: `tab_${  Date.now()}`,
                name: sPath.split("/").pop(),
                path: sPath,
                modified: false
            };
            
            aOpenTabs.push(oNewTab);
            oModel.setProperty("/openTabs", aOpenTabs);
            oModel.setProperty("/activeTab", oNewTab.id);
            oModel.setProperty("/currentFile", oNewTab);
            
            this._loadFileContent(sPath);
        },

        _loadFileContent: function (sPath) {
            // Simulate loading file content
            let sContent = "";
            let sLanguage = "python";
            
            if (sPath.endsWith(".py")) {
                sContent = `# ${sPath}
from a2a_sdk import Agent, Skill

class DataProcessor(Agent):
    def __init__(self):
        super().__init__(
            agent_id="data_processor_001",
            name="Data Processor Agent"
        )
        
    async def process_message(self, message):
        # Process incoming data
        data = message.get("data")
        
        # Transform data
        processed_data = self.transform_data(data)
        
        # Send to next agent
        await self.publish("processed_data", processed_data)
        
    def transform_data(self, data):
        # Data transformation logic
        return data`;
                sLanguage = "python";
            } else if (sPath.endsWith(".yaml")) {
                sContent = `# Configuration
agent:
  id: data_processor_001
  name: Data Processor Agent
  type: reactive
  
communication:
  protocol: mqtt
  broker: localhost:1883
  
skills:
  - name: data_ingestion
    enabled: true
  - name: data_transformation
    enabled: true`;
                sLanguage = "yaml";
            } else if (sPath.endsWith(".md")) {
                sContent = `# ${this.getView().getModel().getProperty("/projectName")}

## Overview
This project contains A2A agents for data processing and API integration.

## Agents
- **Data Processor**: Handles data ingestion and transformation
- **API Integrator**: Manages external API connections

## Getting Started
1. Install dependencies: \`pip install -r requirements.txt\`
2. Configure agents in \`config.yaml\`
3. Run: \`python -m src.main\``;
                sLanguage = "markdown";
            }
            
            this.getView().getModel().setProperty("/editorContent", sContent);
            this.getView().getModel().setProperty("/editorLanguage", sLanguage);
            this.getView().getModel().setProperty("/isDirty", false);
            
            this._updateOutline(sLanguage, sContent);
        },

        _updateOutline: function (sLanguage, sContent) {
            const aOutline = [];
            
            if (sLanguage === "python") {
                // Simple Python outline parsing
                const aLines = sContent.split("\n");
                aLines.forEach((line, index) => {
                    if (line.trim().startsWith("class ")) {
                        aOutline.push({
                            name: line.trim(),
                            icon: "sap-icon://group",
                            line: index + 1
                        });
                    } else if (line.trim().startsWith("def ")) {
                        aOutline.push({
                            name: `  ${  line.trim()}`,
                            icon: "sap-icon://function",
                            line: index + 1
                        });
                    }
                });
            }
            
            this.getView().getModel().setProperty("/outline", aOutline);
        },

        onNavToProjects: function () {
            this._checkUnsavedChanges(() => {
                this.getOwnerComponent().getRouter().navTo("projects");
            });
        },

        onNavToProject: function () {
            this._checkUnsavedChanges(() => {
                this.getOwnerComponent().getRouter().navTo("projectDetail", {
                    projectId: this._projectId
                });
            });
        },

        onFileSelect: function (oEvent) {
            const oItem = oEvent.getParameter("listItem");
            const oContext = oItem.getBindingContext();
            const sPath = oContext.getProperty("path");
            
            if (sPath) {
                this._openFile(sPath);
            }
        },

        onTabSelect: function (oEvent) {
            const sTabId = oEvent.getSource().data("tabId");
            const oModel = this.getView().getModel();
            const aOpenTabs = oModel.getProperty("/openTabs");
            
            const oTab = aOpenTabs.find((tab) => {
                return tab.id === sTabId;
            });
            
            if (oTab) {
                oModel.setProperty("/activeTab", sTabId);
                oModel.setProperty("/currentFile", oTab);
                this._loadFileContent(oTab.path);
            }
        },

        onCloseTab: function (oEvent) {
            const sTabId = oEvent.getSource().data("tabId");
            const oModel = this.getView().getModel();
            const aOpenTabs = oModel.getProperty("/openTabs");
            
            const iIndex = aOpenTabs.findIndex((tab) => {
                return tab.id === sTabId;
            });
            
            if (iIndex > -1) {
                const oTab = aOpenTabs[iIndex];
                
                const fnClose = function () {
                    aOpenTabs.splice(iIndex, 1);
                    oModel.setProperty("/openTabs", aOpenTabs);
                    
                    // Select another tab if needed
                    if (oModel.getProperty("/activeTab") === sTabId && aOpenTabs.length > 0) {
                        const oNewTab = aOpenTabs[Math.max(0, iIndex - 1)];
                        oModel.setProperty("/activeTab", oNewTab.id);
                        oModel.setProperty("/currentFile", oNewTab);
                        this._loadFileContent(oNewTab.path);
                    }
                }.bind(this);
                
                if (oTab.modified) {
                    MessageBox.confirm(`Save changes to ${  oTab.name  }?`, {
                        onClose: function (sAction) {
                            if (sAction === MessageBox.Action.OK) {
                                this._saveFile(oTab.path, () => {
                                    fnClose();
                                });
                            } else {
                                fnClose();
                            }
                        }.bind(this)
                    });
                } else {
                    fnClose();
                }
            }
        },

        onEditorChange: function (oEvent) {
            const oModel = this.getView().getModel();
            oModel.setProperty("/isDirty", true);
            
            // Update current tab's modified state
            const sActiveTab = oModel.getProperty("/activeTab");
            const aOpenTabs = oModel.getProperty("/openTabs");
            const oTab = aOpenTabs.find((tab) => {
                return tab.id === sActiveTab;
            });
            
            if (oTab) {
                oTab.modified = true;
                oModel.setProperty("/openTabs", aOpenTabs);
            }
            
            // Update outline
            const sContent = oEvent.getParameter("value");
            const sLanguage = oModel.getProperty("/editorLanguage");
            this._updateOutline(sLanguage, sContent);
        },

        onSaveFile: function () {
            const oModel = this.getView().getModel();
            const oCurrentFile = oModel.getProperty("/currentFile");
            
            if (oCurrentFile) {
                this._saveFile(oCurrentFile.path);
            }
        },

        _saveFile: function (sPath, fnCallback) {
            const oModel = this.getView().getModel();
            const _sContent = oModel.getProperty("/editorContent");
            
            // Simulate saving
            setTimeout(() => {
                oModel.setProperty("/isDirty", false);
                
                // Update tab's modified state
                const aOpenTabs = oModel.getProperty("/openTabs");
                const oTab = aOpenTabs.find((tab) => {
                    return tab.path === sPath;
                });
                
                if (oTab) {
                    oTab.modified = false;
                    oModel.setProperty("/openTabs", aOpenTabs);
                }
                
                MessageToast.show(`File saved: ${  sPath}`);
                
                if (fnCallback) {
                    fnCallback();
                }
            }, 500);
        },

        onRunFile: function () {
            const oCurrentFile = this.getView().getModel().getProperty("/currentFile");
            
            if (oCurrentFile && oCurrentFile.path.endsWith(".py")) {
                MessageToast.show(`Running ${  oCurrentFile.name  }...`);
                // In real implementation, this would execute the file
            } else {
                MessageToast.show("Can only run Python files");
            }
        },

        onNewFile: function () {
            MessageToast.show("New file dialog - coming soon");
        },

        onNewFolder: function () {
            MessageToast.show("New folder dialog - coming soon");
        },

        onRefreshFiles: function () {
            this._loadFileTree();
            MessageToast.show("File tree refreshed");
        },

        onCollapseAll: function () {
            const oTree = this.byId("fileTree");
            oTree.collapseAll();
        },

        onFindReplace: function () {
            MessageToast.show("Find/Replace dialog - coming soon");
        },

        onEditorSettings: function () {
            MessageToast.show("Editor settings - coming soon");
        },

        onLanguageChange: function (oEvent) {
            const sLanguage = oEvent.getParameter("selectedItem").getKey();
            MessageToast.show(`Language changed to ${  sLanguage}`);
        },

        onOutlineSelect: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext();
            const iLine = oContext.getProperty("line");
            MessageToast.show(`Go to line ${  iLine}`);
        },

        onProblemSelect: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext();
            const sPath = oContext.getProperty("path");
            const iLine = oContext.getProperty("line");
            
            this._openFile(sPath);
            MessageToast.show(`Go to line ${  iLine}`);
        },

        onCommit: function () {
            MessageToast.show("Git commit dialog - coming soon");
        },

        _checkUnsavedChanges: function (fnCallback) {
            const oModel = this.getView().getModel();
            const aOpenTabs = oModel.getProperty("/openTabs");
            const aModifiedTabs = aOpenTabs.filter((tab) => {
                return tab.modified;
            });
            
            if (aModifiedTabs.length > 0) {
                MessageBox.confirm("You have unsaved changes. Do you want to save them?", {
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            // Save all modified files
                            let iSaved = 0;
                            aModifiedTabs.forEach((tab) => {
                                this._saveFile(tab.path, () => {
                                    iSaved++;
                                    if (iSaved === aModifiedTabs.length) {
                                        fnCallback();
                                    }
                                });
                            });
                        } else {
                            fnCallback();
                        }
                    }.bind(this)
                });
            } else {
                fnCallback();
            }
        }
    });
});