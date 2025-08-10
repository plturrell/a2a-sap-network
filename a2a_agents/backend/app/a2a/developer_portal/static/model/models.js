sap.ui.define([
    "sap/ui/model/json/JSONModel",
    "sap/ui/Device"
], function (JSONModel, Device) {
    "use strict";

    return {
        
        createDeviceModel: function () {
            var oModel = new JSONModel(Device);
            oModel.setDefaultBindingMode("OneWay");
            return oModel;
        },

        createProjectsModel: function () {
            var oModel = new JSONModel();
            
            // Initialize with empty data structure
            oModel.setData({
                projects: [],
                templates: [],
                busy: false,
                viewSettings: {
                    viewMode: "tiles",
                    sortBy: "lastModified",
                    sortDescending: true
                }
            });

            return oModel;
        },

        createAgentBuilderModel: function () {
            var oModel = new JSONModel();
            
            oModel.setData({
                currentProject: null,
                availableTemplates: [],
                selectedTemplate: null,
                agentConfiguration: {
                    name: "",
                    id: "",
                    description: "",
                    skills: [],
                    handlers: [],
                    customSkills: [],
                    customHandlers: []
                },
                bpmnWorkflow: null,
                generationInProgress: false
            });

            return oModel;
        },

        createCodeEditorModel: function () {
            var oModel = new JSONModel();
            
            oModel.setData({
                currentProject: null,
                fileTree: [],
                openTabs: [],
                activeTab: null,
                editorContent: "",
                editorLanguage: "python",
                isDirty: false
            });

            return oModel;
        }
    };
});