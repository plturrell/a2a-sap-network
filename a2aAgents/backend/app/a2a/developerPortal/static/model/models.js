sap.ui.define([
  'sap/ui/model/json/JSONModel',
  'sap/ui/Device'
], (JSONModel, Device) => {
  'use strict';

  return {
        
    createDeviceModel: function () {
      const oModel = new JSONModel(Device);
      oModel.setDefaultBindingMode('OneWay');
      return oModel;
    },

    createProjectsModel: function () {
      const oModel = new JSONModel();
            
      // Initialize with empty data structure
      oModel.setData({
        projects: [],
        templates: [],
        busy: false,
        viewSettings: {
          viewMode: 'tiles',
          sortBy: 'lastModified',
          sortDescending: true
        }
      });

      return oModel;
    },

    createAgentBuilderModel: function () {
      const oModel = new JSONModel();
            
      oModel.setData({
        currentProject: null,
        availableTemplates: [],
        selectedTemplate: null,
        agentConfiguration: {
          name: '',
          id: '',
          description: '',
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
      const oModel = new JSONModel();
            
      oModel.setData({
        currentProject: null,
        fileTree: [],
        openTabs: [],
        activeTab: null,
        editorContent: '',
        editorLanguage: 'python',
        isDirty: false
      });

      return oModel;
    }
  };
});