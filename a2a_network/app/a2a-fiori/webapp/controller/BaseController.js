sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], function (Controller, MessageToast, MessageBox, JSONModel) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.BaseController", {

        onInit: function () {
            // Initialize UI state model for loading states
            this.oUIModel = new JSONModel({
                // Loading states
                isLoadingSkeleton: false,
                isLoadingSpinner: false,
                isLoadingProgress: false,
                isLoadingBlockchain: false,
                loadingMessage: "",
                loadingSubMessage: "",
                progressTitle: "",
                progressValue: 0,
                progressText: "",
                progressState: "None",
                progressDescription: "",
                blockchainStep: "",
                
                // Error states
                hasError: false,
                errorMessage: "",
                
                // Blockchain education states
                showBlockchainAddress: false,
                showTransactionHash: false,
                showContractInfo: false,
                showGasInfo: false,
                showTransactionStatus: false,
                showAddressHelp: false,
                showTransactionHelp: false,
                showContractHelp: false,
                showGasHelp: false,
                showNetworkHelp: false,
                showConfirmationHelp: false,
                
                // Blockchain data
                address: "",
                txHash: "",
                txHashUrl: "",
                contractName: "",
                gasUsed: 0,
                gasState: "None",
                networkStatusColor: "Default",
                networkStatusText: "",
                confirmationProgress: 0,
                confirmationsText: "",
                confirmationState: "None",
                confirmationDescription: "",
                
                // Help content
                helpTitle: "",
                helpContent: "",
                learnMoreUrl: "",
                
                // Button states
                syncEnabled: true,
                deployEnabled: true
            });
            
            this.getView().setModel(this.oUIModel, "ui");
        },

        /**
         * Show skeleton loading state for lists and tables
         */
        showSkeletonLoading: function (message) {
            this.oUIModel.setProperty("/isLoadingSkeleton", true);
            this.oUIModel.setProperty("/isLoadingSpinner", false);
            this.oUIModel.setProperty("/isLoadingProgress", false);
            this.oUIModel.setProperty("/hasError", false);
            if (message) {
                this.oUIModel.setProperty("/loadingMessage", message);
            }
        },

        /**
         * Show spinner loading state for actions
         */
        showSpinnerLoading: function (message, subMessage) {
            this.oUIModel.setProperty("/isLoadingSpinner", true);
            this.oUIModel.setProperty("/isLoadingSkeleton", false);
            this.oUIModel.setProperty("/isLoadingProgress", false);
            this.oUIModel.setProperty("/hasError", false);
            this.oUIModel.setProperty("/loadingMessage", message || this.getResourceBundle().getText("processing"));
            this.oUIModel.setProperty("/loadingSubMessage", subMessage || this.getResourceBundle().getText("pleaseWait"));
        },

        /**
         * Show progress loading for multi-step operations
         */
        showProgressLoading: function (title, value, text, description, state) {
            this.oUIModel.setProperty("/isLoadingProgress", true);
            this.oUIModel.setProperty("/isLoadingSkeleton", false);
            this.oUIModel.setProperty("/isLoadingSpinner", false);
            this.oUIModel.setProperty("/hasError", false);
            this.oUIModel.setProperty("/progressTitle", title);
            this.oUIModel.setProperty("/progressValue", value);
            this.oUIModel.setProperty("/progressText", text);
            this.oUIModel.setProperty("/progressDescription", description);
            this.oUIModel.setProperty("/progressState", state || "None");
        },

        /**
         * Show blockchain-specific loading
         */
        showBlockchainLoading: function (step) {
            this.oUIModel.setProperty("/isLoadingBlockchain", true);
            this.oUIModel.setProperty("/isLoadingSkeleton", false);
            this.oUIModel.setProperty("/isLoadingSpinner", false);
            this.oUIModel.setProperty("/isLoadingProgress", false);
            this.oUIModel.setProperty("/hasError", false);
            this.oUIModel.setProperty("/blockchainStep", step);
            this.oUIModel.setProperty("/syncEnabled", false);
            this.oUIModel.setProperty("/deployEnabled", false);
        },

        /**
         * Hide all loading states
         */
        hideLoading: function () {
            this.oUIModel.setProperty("/isLoadingSkeleton", false);
            this.oUIModel.setProperty("/isLoadingSpinner", false);
            this.oUIModel.setProperty("/isLoadingProgress", false);
            this.oUIModel.setProperty("/isLoadingBlockchain", false);
            this.oUIModel.setProperty("/hasError", false);
            this.oUIModel.setProperty("/syncEnabled", true);
            this.oUIModel.setProperty("/deployEnabled", true);
        },

        /**
         * Show error state
         */
        showError: function (message) {
            this.hideLoading();
            this.oUIModel.setProperty("/hasError", true);
            this.oUIModel.setProperty("/errorMessage", message);
        },

        /**
         * Educational help functions
         */
        onCopyAddress: function () {
            const address = this.oUIModel.getProperty("/address");
            if (address && navigator.clipboard) {
                navigator.clipboard.writeText(address).then(() => {
                    MessageToast.show("Address copied to clipboard");
                }).catch(() => {
                    MessageToast.show("Could not copy address");
                });
            }
        },

        onShowAddressHelp: function () {
            this.oUIModel.setProperty("/showAddressHelp", !this.oUIModel.getProperty("/showAddressHelp"));
        },

        onShowTransactionHelp: function () {
            this.oUIModel.setProperty("/showTransactionHelp", !this.oUIModel.getProperty("/showTransactionHelp"));
        },

        onShowContractHelp: function () {
            this.oUIModel.setProperty("/showContractHelp", !this.oUIModel.getProperty("/showContractHelp"));
        },

        onShowGasHelp: function () {
            this.oUIModel.setProperty("/showGasHelp", !this.oUIModel.getProperty("/showGasHelp"));
        },

        onShowNetworkHelp: function () {
            this.oUIModel.setProperty("/showNetworkHelp", !this.oUIModel.getProperty("/showNetworkHelp"));
        },

        onShowConfirmationHelp: function () {
            this.oUIModel.setProperty("/showConfirmationHelp", !this.oUIModel.getProperty("/showConfirmationHelp"));
        },

        onCloseHelp: function () {
            var oPopover = this.byId("blockchainEducationPopover");
            if (oPopover) {
                oPopover.close();
            }
        },

        onCloseError: function () {
            this.oUIModel.setProperty("/hasError", false);
        },

        onRetry: function () {
            this.oUIModel.setProperty("/hasError", false);
            // Override in subclasses
        },

        /**
         * Utility functions
         */
        getResourceBundle: function () {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },

        getRouter: function () {
            return this.getOwnerComponent().getRouter();
        }
    });
});