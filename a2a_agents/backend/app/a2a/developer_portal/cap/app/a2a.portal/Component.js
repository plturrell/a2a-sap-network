sap.ui.define([
    "sap/ui/core/UIComponent",
    "sap/ui/Device",
    "sap/ui/model/json/JSONModel"
], function (UIComponent, Device, JSONModel) {
    "use strict";

    return UIComponent.extend("a2a.portal.Component", {

        metadata: {
            manifest: "json"
        },

        /**
         * The component is initialized by UI5 automatically during the startup of the app and calls the init method once.
         * @public
         * @override
         */
        init: function() {
            // Call the base component's init function
            UIComponent.prototype.init.apply(this, arguments);
            
            // Create and set the device model
            var oDeviceModel = new JSONModel(Device);
            oDeviceModel.setDefaultBindingMode("OneWay");
            this.setModel(oDeviceModel, "device");
            
            // Create and set the projects model with sample data
            var oProjectsModel = new JSONModel({
                projects: [
                    { id: "proj_001", name: "DataProcessor", status: "active", description: "Data processing agent" },
                    { id: "proj_002", name: "CustomerAnalyzer", status: "active", description: "Customer analytics agent" },
                    { id: "proj_003", name: "ReportGenerator", status: "inactive", description: "Report generation agent" }
                ]
            });
            this.setModel(oProjectsModel, "projects");
            
            console.log("SAP CAP Component initialized successfully");
        },

        /**
         * Create the root view content
         * @public
         * @override
         */
        createContent: function() {
            // Call the base component's createContent function
            var oRootView = UIComponent.prototype.createContent.apply(this, arguments);
            
            console.log("Root view created:", oRootView ? oRootView.getId() : "null");
            
            return oRootView;
        }
    });
});