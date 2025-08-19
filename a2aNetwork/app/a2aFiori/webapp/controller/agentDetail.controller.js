sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/m/MessageToast",
    "../model/formatter",
    "sap/ui/thirdparty/showdown",
    "sap/base/Log"
], function(Controller, History, MessageToast, formatter, showdown, Log) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.AgentDetail", {
        formatter,

        onInit() {
            this.getRouter().getRoute("agentDetail").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched(oEvent) {
            const sAgentId = oEvent.getParameter("arguments").agentId;
            const sPath = `/Agents('${sAgentId}')`;
            this.getView().bindElement({
                path: sPath,
                parameters: {
                    expand: "capabilities,services,performance"
                },
                events: {
                    dataReceived: () => this._onBindingDataReceived()
                }
            });
        },

        _onBindingDataReceived() {
            const oContext = this.getView().getBindingContext();
            if (!oContext) {
                return;
            }
            const aCapabilities = oContext.getProperty("capabilities");
            // Use the first capability as the agent type for documentation
            const sAgentType = aCapabilities && aCapabilities.length > 0 ? aCapabilities[0].capability.name : "default";
            this._loadDocumentation(sAgentType);
        },

        _loadDocumentation(sAgentType) {
            const oHtmlControl = this.byId("documentationContent");
            oHtmlControl.setContent("<div class=\"sapUiSmallMargin\">Loading documentation...</div>");

            // Fetch documentation based on agent type (capability)
            fetch(`/api/v1/agent-documentation/${sAgentType}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.text();
                })
                .then(markdown => {
                    const oConverter = new showdown.Converter();
                    const sHtml = oConverter.makeHtml(markdown);
                    oHtmlControl.setContent(sHtml);
                })
                .catch(error => {
                    Log.error("Error loading documentation:", error);
                    oHtmlControl.setContent("<div class=\"sapUiSmallMargin\">Failed to load documentation.</div>");
                });
        },

        onEdit() {
            MessageToast.show("Edit mode - Coming Soon");
        },

        onSync() {
            const _oContext = this.getView().getBindingContext();
            if (!oContext) {
                return;
            }

            const sAgentId = oContext.getProperty("ID");
            const oModel = this.getView().getModel();

            oModel.callFunction(`/Agents('${ sAgentId }')/registerOnBlockchain`, {
                method: "POST",
                success() {
                    MessageToast.show("Agent synced with blockchain");
                },
                error() {
                    MessageToast.show("Sync failed");
                }
            });
        },

        onNavBack() {
            const oHistory = History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getRouter().navTo("agents");
            }
        },

        getRouter() {
            return this.getOwnerComponent().getRouter();
        }
    });
});