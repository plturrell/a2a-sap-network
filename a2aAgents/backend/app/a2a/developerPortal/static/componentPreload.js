sap.ui.require.preload({
    "a2a/portal/Component.js": function() {
        sap.ui.define([
            "sap/ui/core/UIComponent",
            "sap/ui/Device",
            "sap/ui/model/json/JSONModel"
        ], (UIComponent, _Device, JSONModel) => {
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
                init: function () {
                    // call the base component's init function
                    UIComponent.prototype.init.apply(this, arguments);

                    // Create and set the device model inline
                    const deviceModel = new JSONModel(Device);
                    deviceModel.setDefaultBindingMode("OneWay");
                    this.setModel(deviceModel, "device");
                    
                    // Create and set the projects model inline
                    const projectsModel = new JSONModel();
                    projectsModel.setData({
                        projects: [],
                        templates: [],
                        busy: false,
                        viewSettings: {
                            viewMode: "tiles",
                            sortBy: "lastModified",
                            sortDescending: true
                        }
                    });
                    this.setModel(projectsModel, "projects");
                    
                    // Create the views
                    this.getRootControl();
                    
                    // Initialize the router
                    this.getRouter().initialize();
                }
            });
        });
    },
    "a2a/portal/manifest.json": '{"_version":"1.58.0","sap.app":{"id":"a2a.portal","type":"application"}}'
}, "a2a/portal/Component-preload");
