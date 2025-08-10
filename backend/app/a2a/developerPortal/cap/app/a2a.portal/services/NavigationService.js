sap.ui.define([
    "sap/ui/base/Object",
    "sap/ushell/services/CrossApplicationNavigation",
    "sap/m/MessageToast"
], function (BaseObject, CrossApplicationNavigation, MessageToast) {
    "use strict";

    /**
     * Navigation Service for A2A Agents
     * Provides semantic navigation between different application areas
     */
    return BaseObject.extend("com.sap.a2a.developerportal.services.NavigationService", {

        constructor: function () {
            BaseObject.prototype.constructor.apply(this, arguments);
            
            // Initialize cross-application navigation service
            this._oCrossAppNav = sap.ushell && sap.ushell.Container && 
                sap.ushell.Container.getService("CrossApplicationNavigation");
            
            // Define semantic objects and actions
            this._mSemanticObjects = {
                "A2APortal": {
                    "overview": {
                        route: "overview",
                        title: "Overview Dashboard"
                    }
                },
                "Project": {
                    "manage": {
                        route: "projects",
                        title: "Projects"
                    },
                    "manageSmart": {
                        route: "projectsSmart",
                        title: "Smart Projects"
                    },
                    "masterDetail": {
                        route: "projectMasterDetail",
                        title: "Project Master-Detail"
                    },
                    "display": {
                        route: "projectDetail",
                        title: "Project Details",
                        parameters: ["projectId"]
                    },
                    "edit": {
                        route: "projectEdit",
                        title: "Edit Project",
                        parameters: ["projectId"]
                    },
                    "create": {
                        route: "projectCreate",
                        title: "Create Project"
                    }
                },
                "Agent": {
                    "build": {
                        route: "agentBuilder",
                        title: "Agent Builder",
                        parameters: ["projectId"]
                    },
                    "display": {
                        route: "agentDetail",
                        title: "Agent Details",
                        parameters: ["agentId"]
                    },
                    "edit": {
                        route: "agentEdit",
                        title: "Edit Agent",
                        parameters: ["agentId"]
                    }
                },
                "Workflow": {
                    "design": {
                        route: "bpmnDesigner",
                        title: "BPMN Designer",
                        parameters: ["projectId"]
                    },
                    "display": {
                        route: "workflowDetail",
                        title: "Workflow Details",
                        parameters: ["workflowId"]
                    }
                },
                "Code": {
                    "edit": {
                        route: "codeEditor",
                        title: "Code Editor",
                        parameters: ["projectId", "fileId"]
                    }
                },
                "Testing": {
                    "manage": {
                        route: "testingFramework",
                        title: "Testing Framework",
                        parameters: ["projectId"]
                    }
                },
                "Deployment": {
                    "manage": {
                        route: "deploymentPipeline",
                        title: "Deployment Pipeline",
                        parameters: ["projectId"]
                    }
                },
                "Monitoring": {
                    "display": {
                        route: "systemHealth",
                        title: "System Monitoring"
                    }
                },
                "User": {
                    "profile": {
                        route: "userProfile",
                        title: "User Profile"
                    }
                }
            };
        },

        /**
         * Navigate to a semantic object and action
         * @param {string} sSemanticObject - The semantic object
         * @param {string} sAction - The action
         * @param {object} mParameters - Navigation parameters
         * @param {object} mInnerAppData - Inner app data
         * @returns {Promise} Navigation promise
         */
        navigateToSemanticObject: function (sSemanticObject, sAction, mParameters, mInnerAppData) {
            var that = this;
            
            return new Promise(function (resolve, reject) {
                try {
                    // Check if semantic object and action are defined
                    var oSemanticObject = that._mSemanticObjects[sSemanticObject];
                    if (!oSemanticObject) {
                        throw new Error("Semantic object '" + sSemanticObject + "' not found");
                    }
                    
                    var oAction = oSemanticObject[sAction];
                    if (!oAction) {
                        throw new Error("Action '" + sAction + "' not found for semantic object '" + sSemanticObject + "'");
                    }
                    
                    // Use Fiori Launchpad navigation if available
                    if (that._oCrossAppNav) {
                        var sHash = that._oCrossAppNav.hrefForExternal({
                            target: {
                                semanticObject: sSemanticObject,
                                action: sAction
                            },
                            params: mParameters
                        });
                        
                        that._oCrossAppNav.toExternal({
                            target: {
                                semanticObject: sSemanticObject,
                                action: sAction
                            },
                            params: mParameters,
                            appData: mInnerAppData
                        });
                        
                        resolve(sHash);
                    } else {
                        // Fallback to internal routing
                        that._navigateInternally(oAction, mParameters);
                        resolve();
                    }
                } catch (oError) {
                    console.error("Navigation error:", oError);
                    MessageToast.show("Navigation failed: " + oError.message);
                    reject(oError);
                }
            });
        },

        /**
         * Internal navigation fallback
         * @param {object} oAction - Action configuration
         * @param {object} mParameters - Navigation parameters
         * @private
         */
        _navigateInternally: function (oAction, mParameters) {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this);
            
            if (!oRouter) {
                // Get router from any view
                var aViews = sap.ui.getCore().byFieldGroupId("view");
                if (aViews.length > 0) {
                    oRouter = aViews[0].getController().getOwnerComponent().getRouter();
                }
            }
            
            if (oRouter) {
                oRouter.navTo(oAction.route, mParameters);
            } else {
                console.error("No router available for internal navigation");
                MessageToast.show("Navigation not available");
            }
        },

        /**
         * Get supported semantic objects
         * @returns {array} Array of semantic objects
         */
        getSupportedSemanticObjects: function () {
            return Object.keys(this._mSemanticObjects);
        },

        /**
         * Get supported actions for a semantic object
         * @param {string} sSemanticObject - The semantic object
         * @returns {array} Array of actions
         */
        getSupportedActions: function (sSemanticObject) {
            var oSemanticObject = this._mSemanticObjects[sSemanticObject];
            return oSemanticObject ? Object.keys(oSemanticObject) : [];
        },

        /**
         * Check if navigation is supported
         * @param {string} sSemanticObject - The semantic object
         * @param {string} sAction - The action
         * @returns {boolean} True if supported
         */
        isNavigationSupported: function (sSemanticObject, sAction) {
            var oSemanticObject = this._mSemanticObjects[sSemanticObject];
            return !!(oSemanticObject && oSemanticObject[sAction]);
        },

        /**
         * Get navigation intent for external use
         * @param {string} sSemanticObject - The semantic object
         * @param {string} sAction - The action
         * @param {object} mParameters - Navigation parameters
         * @returns {string} Navigation intent hash
         */
        getNavigationIntent: function (sSemanticObject, sAction, mParameters) {
            if (this._oCrossAppNav) {
                return this._oCrossAppNav.hrefForExternal({
                    target: {
                        semanticObject: sSemanticObject,
                        action: sAction
                    },
                    params: mParameters
                });
            } else {
                // Generate internal hash
                var oAction = this._mSemanticObjects[sSemanticObject] && 
                             this._mSemanticObjects[sSemanticObject][sAction];
                
                if (oAction) {
                    var sHash = "#" + oAction.route;
                    if (mParameters && Object.keys(mParameters).length > 0) {
                        var aParams = [];
                        for (var sKey in mParameters) {
                            aParams.push(sKey + "=" + encodeURIComponent(mParameters[sKey]));
                        }
                        sHash += "?" + aParams.join("&");
                    }
                    return sHash;
                }
            }
            
            return "";
        },

        /**
         * Navigate back to previous app
         */
        navigateBack: function () {
            if (this._oCrossAppNav) {
                this._oCrossAppNav.backToPreviousApp();
            } else {
                window.history.back();
            }
        },

        /**
         * Get current navigation context
         * @returns {object} Current navigation context
         */
        getCurrentNavigationContext: function () {
            if (this._oCrossAppNav) {
                return this._oCrossAppNav.getAppState();
            } else {
                return {
                    hash: window.location.hash,
                    search: window.location.search
                };
            }
        },

        /**
         * Create external navigation URL
         * @param {string} sSemanticObject - The semantic object
         * @param {string} sAction - The action
         * @param {object} mParameters - Navigation parameters
         * @returns {string} External URL
         */
        createExternalUrl: function (sSemanticObject, sAction, mParameters) {
            var sBaseUrl = window.location.origin + window.location.pathname;
            var sIntent = this.getNavigationIntent(sSemanticObject, sAction, mParameters);
            
            return sBaseUrl + sIntent;
        },

        /**
         * Register navigation interceptor
         * @param {function} fnInterceptor - Interceptor function
         */
        registerNavigationInterceptor: function (fnInterceptor) {
            if (this._oCrossAppNav && this._oCrossAppNav.registerNavigationFilter) {
                this._oCrossAppNav.registerNavigationFilter(fnInterceptor);
            }
        }
    });
});
