sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/core/UIComponent",
    "sap/base/Log"
], function(Controller, History, UIComponent, Log) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.BaseController", {
        /**
         * Convenience method for accessing the router.
         * @public
         * @returns {sap.ui.core.routing.Router} the router for this component
         */
        getRouter: function() {
            return UIComponent.getRouterFor(this);
        },

        /**
         * Convenience method for getting the view model by name.
         * @public
         * @param {string} [sName] the model name
         * @returns {sap.ui.model.Model} the model instance
         */
        getModel: function(sName) {
            return this.getView().getModel(sName);
        },

        /**
         * Convenience method for setting the view model.
         * @public
         * @param {sap.ui.model.Model} oModel the model instance
         * @param {string} sName the model name
         * @returns {sap.ui.mvc.View} the view instance
         */
        setModel: function(oModel, sName) {
            return this.getView().setModel(oModel, sName);
        },

        /**
         * Getter for the resource bundle.
         * @public
         * @returns {sap.ui.model.resource.ResourceModel} the resourceModel of the component
         */
        getResourceBundle: function() {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },

        /**
         * Event handler when the share by E-Mail button has been clicked
         * @public
         */
        onShareEmailPress: function() {
            var oViewModel = (this.getModel("objectView") || this.getModel("worklistView"));
            sap.m.URLHelper.triggerEmail(
                null,
                oViewModel.getProperty("/shareSendEmailSubject"),
                oViewModel.getProperty("/shareSendEmailMessage")
            );
        },

        /**
         * Navigates back in browser history or to the home screen
         * @public
         */
        onNavBack: function() {
            var sPreviousHash = History.getInstance().getPreviousHash();

            if (sPreviousHash !== undefined) {
                // The history contains a previous entry
                history.go(-1);
            } else {
                // Otherwise we go backwards with a forward history
                var bReplace = true;
                this.getRouter().navTo("home", {}, bReplace);
            }
        },

        /**
         * Creates an error message from an OData response
         * @param {object} oError the OData error response
         * @returns {string} the error message
         */
        _createErrorMessage: function(oError) {
            var sMessage = this.getResourceBundle().getText("errorText");
            
            if (oError.responseText) {
                try {
                    var oErrorResponse = JSON.parse(oError.responseText);
                    if (oErrorResponse.error && oErrorResponse.error.message) {
                        sMessage = oErrorResponse.error.message.value || oErrorResponse.error.message;
                    }
                } catch (e) {
                    Log.error("Failed to parse error response", e);
                }
            } else if (oError.message) {
                sMessage = oError.message;
            }
            
            return sMessage;
        },

        /**
         * Shows a message box with the given message
         * @param {string} sMessage the message to show
         * @param {string} sType the type of message (error, warning, success, information)
         */
        _showMessage: function(sMessage, sType) {
            sap.m.MessageBox[sType || "error"](sMessage);
        },

        /**
         * Gets the component's event bus
         * @returns {sap.ui.core.EventBus} the event bus
         */
        getEventBus: function() {
            return this.getOwnerComponent().getEventBus();
        },

        /**
         * Registers a callback for cleanup when the controller is destroyed
         * @param {function} fnCleanup the cleanup function
         */
        _registerForCleanup: function(fnCleanup) {
            if (!this._aCleanups) {
                this._aCleanups = [];
            }
            this._aCleanups.push(fnCleanup);
        },

        /**
         * Lifecycle method - called when controller is destroyed
         */
        onExit: function() {
            if (this._aCleanups) {
                this._aCleanups.forEach(function(fnCleanup) {
                    fnCleanup();
                });
                this._aCleanups = [];
            }
        }
    });
});