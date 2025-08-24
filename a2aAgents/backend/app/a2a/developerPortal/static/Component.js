sap.ui.define([
  'sap/ui/core/UIComponent',
  'sap/ui/Device',
  'sap/ui/model/json/JSONModel'
], (UIComponent, Device, JSONModel) => {
  'use strict';

  return UIComponent.extend('a2a.portal.Component', {

    metadata: {
      manifest: 'json'
    },

    /**
         * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
         * @public
         * @override
         */
    init: function(...args) {
      // Call the base component's init function
      UIComponent.prototype.init.apply(this, args);
            
      // Create and set the device model
      const oDeviceModel = new JSONModel(Device);
      oDeviceModel.setDefaultBindingMode('OneWay');
      this.setModel(oDeviceModel, 'device');
            
      // Create and set the projects model with sample data
      const oProjectsModel = new JSONModel({
        projects: [
          { id: 'proj_001', name: 'DataProcessor', status: 'active', description: 'Data processing agent' },
          { id: 'proj_002', name: 'CustomerAnalyzer', status: 'active', description: 'Customer analytics agent' },
          { id: 'proj_003', name: 'ReportGenerator', status: 'inactive', description: 'Report generation agent' }
        ]
      });
      this.setModel(oProjectsModel, 'projects');
            
      // Initialize the router using standard SAP CAP/UI5 pattern
      try {
        this.getRouter().initialize();
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('Router initialized successfully');
      } catch (error) {
        console.error('Router initialization failed:', error);
        // Don't retry automatically to avoid infinite loops
      }
    },

    /**
         * Create the root view content
         * @public
         * @override
         */
    createContent: function(...args) {
      // Call the base component's createContent function
      const oRootView = UIComponent.prototype.createContent.apply(this, args);
            
             
            
      // eslint-disable-next-line no-console
            
             
            
      // eslint-disable-next-line no-console
      console.log('Root view created:', oRootView ? oRootView.getId() : 'null');
            
      return oRootView;
    }
  });
});