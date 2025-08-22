sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/base/Log"
], function(BaseObject, Fragment, JSONModel, MessageToast, MessageBox, Log) {
    "use strict";

    /**
     * Standard Patterns Mixin for SAP Fiori Compliance
     * Provides standardized methods for dialogs, forms, and interactions
     */
    return {
        
        /**
         * Initialize standard patterns
         */
        initializeStandardPatterns: function() {
            // Create standard models for UI state
            this._standardModels = {
                dialogState: new JSONModel({
                    dialogId: "",
                    title: "",
                    formFragment: "",
                    primaryButtonText: "",
                    primaryButtonPress: "",
                    primaryButtonEnabled: true,
                    secondaryButtonText: "Cancel",
                    secondaryButtonPress: "onCloseDialog"
                }),
                
                pageState: new JSONModel({
                    pageId: "",
                    pageTitle: "",
                    headerExpanded: true,
                    showFooter: false,
                    breadcrumbsVisible: false,
                    keyInfoVisible: false,
                    filterBarVisible: false,
                    searchVisible: true,
                    viewControlsVisible: true,
                    viewModeVisible: true,
                    settingsVisible: true,
                    footerVisible: false,
                    statusVisible: false,
                    statusMessage: "",
                    statusType: "Information",
                    searchQuery: "",
                    viewMode: "table"
                }),
                
                actionState: new JSONModel({
                    primaryActionsVisible: true,
                    secondaryActionsVisible: true,
                    destructiveActionsVisible: true,
                    createVisible: true,
                    createEnabled: true,
                    editVisible: true,
                    editEnabled: false,
                    saveVisible: false,
                    saveEnabled: false,
                    cancelVisible: false,
                    cancelEnabled: false,
                    refreshVisible: true,
                    refreshEnabled: true,
                    copyVisible: false,
                    copyEnabled: false,
                    exportVisible: true,
                    exportEnabled: true,
                    deleteVisible: true,
                    deleteEnabled: false
                })
            };
            
            // Set models on view
            Object.keys(this._standardModels).forEach(modelName => {
                this.getView().setModel(this._standardModels[modelName], modelName);
            });
        },
        
        /**
         * Open standard dialog
         * @param {Object} config - Dialog configuration
         */
        openStandardDialog: function(config) {
            // Update dialog state
            this._standardModels.dialogState.setData({
                dialogId: config.id || "standardDialog",
                title: config.title || "",
                formFragment: config.formFragment || "",
                primaryButtonText: config.primaryButtonText || "OK",
                primaryButtonPress: config.primaryButtonPress || "onDialogOK",
                primaryButtonEnabled: config.primaryButtonEnabled !== false,
                secondaryButtonText: config.secondaryButtonText || "Cancel",
                secondaryButtonPress: config.secondaryButtonPress || "onCloseDialog"
            });
            
            // Load and open dialog
            if (!this._standardDialog) {
                Fragment.load({
                    name: "a2a.network.launchpad.fragment.StandardDialog",
                    controller: this
                }).then(dialog => {
                    this._standardDialog = dialog;
                    this.getView().addDependent(dialog);
                    dialog.open();
                });
            } else {
                this._standardDialog.open();
            }
        },
        
        /**
         * Close standard dialog
         */
        onCloseDialog: function() {
            if (this._standardDialog) {
                this._standardDialog.close();
            }
        },
        
        /**
         * Dialog after close cleanup
         */
        onDialogAfterClose: function() {
            // Reset dialog state
            this._standardModels.dialogState.setData({});
        },
        
        /**
         * Update page state
         * @param {Object} state - Page state updates
         */
        updatePageState: function(state) {
            const currentData = this._standardModels.pageState.getData();
            this._standardModels.pageState.setData(Object.assign(currentData, state));
        },
        
        /**
         * Update action state
         * @param {Object} state - Action state updates
         */
        updateActionState: function(state) {
            const currentData = this._standardModels.actionState.getData();
            this._standardModels.actionState.setData(Object.assign(currentData, state));
        },
        
        /**
         * Show status message
         * @param {string} message - Message text
         * @param {string} type - Message type (Success, Warning, Error, Information)
         */
        showStatusMessage: function(message, type = "Information") {
            this.updatePageState({
                statusVisible: true,
                statusMessage: message,
                statusType: type
            });
            
            // Auto-hide after 5 seconds for success/info messages
            if (type === "Success" || type === "Information") {
                setTimeout(() => {
                    this.updatePageState({ statusVisible: false });
                }, 5000);
            }
        },
        
        /**
         * Standard error handler
         * @param {Error} error - Error object
         */
        handleStandardError: function(error) {
            Log.error("Standard error occurred", error);
            
            const message = error.message || "An unexpected error occurred";
            this.showStatusMessage(message, "Error");
            
            // Show error dialog for critical errors
            if (error.critical) {
                MessageBox.error(message, {
                    title: "Error",
                    details: error.details || ""
                });
            }
        },
        
        /**
         * Standard success handler
         * @param {string} message - Success message
         */
        handleStandardSuccess: function(message) {
            this.showStatusMessage(message, "Success");
            MessageToast.show(message);
        },
        
        /**
         * Standard confirmation dialog
         * @param {Object} config - Confirmation configuration
         */
        showStandardConfirmation: function(config) {
            MessageBox.confirm(config.message || "Are you sure?", {
                title: config.title || "Confirm",
                onClose: (action) => {
                    if (action === MessageBox.Action.OK && config.onConfirm) {
                        config.onConfirm();
                    } else if (action === MessageBox.Action.CANCEL && config.onCancel) {
                        config.onCancel();
                    }
                }
            });
        },
        
        /**
         * Standard form validation
         * @param {Array} fields - Array of field configurations
         * @returns {Object} Validation result
         */
        validateStandardForm: function(fields) {
            const errors = [];
            const warnings = [];
            
            fields.forEach(field => {
                const value = field.getValue ? field.getValue() : field.value;
                
                // Required field validation
                if (field.required && (!value || value.trim() === "")) {
                    errors.push({
                        field: field.name || field.id,
                        message: `${field.label || field.name} is required`
                    });
                    
                    // Set error state on field
                    if (field.setValueState) {
                        field.setValueState("Error");
                        field.setValueStateText(`${field.label || field.name} is required`);
                    }
                }
                
                // Custom validation
                if (field.validate && typeof field.validate === "function") {
                    const result = field.validate(value);
                    if (result.error) {
                        errors.push({
                            field: field.name || field.id,
                            message: result.message
                        });
                        
                        if (field.setValueState) {
                            field.setValueState("Error");
                            field.setValueStateText(result.message);
                        }
                    } else if (result.warning) {
                        warnings.push({
                            field: field.name || field.id,
                            message: result.message
                        });
                        
                        if (field.setValueState) {
                            field.setValueState("Warning");
                            field.setValueStateText(result.message);
                        }
                    } else {
                        // Clear error state
                        if (field.setValueState) {
                            field.setValueState("None");
                        }
                    }
                }
            });
            
            return {
                valid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },
        
        /**
         * Clear form validation states
         * @param {Array} fields - Array of fields to clear
         */
        clearFormValidation: function(fields) {
            fields.forEach(field => {
                if (field.setValueState) {
                    field.setValueState("None");
                }
            });
        },
        
        /**
         * Standard search handler
         * @param {Object} event - Search event
         */
        onSearch: function(event) {
            const query = event.getParameter("query") || event.getParameter("newValue");
            this.updatePageState({ searchQuery: query });
            this._performSearch(query);
        },
        
        /**
         * Standard live search handler
         * @param {Object} event - Live search event
         */
        onLiveSearch: function(event) {
            const query = event.getParameter("newValue");
            this.updatePageState({ searchQuery: query });
            
            // Debounce live search
            clearTimeout(this._searchTimeout);
            this._searchTimeout = setTimeout(() => {
                this._performSearch(query);
            }, 300);
        },
        
        /**
         * Perform search (to be implemented by concrete controller)
         * @param {string} query - Search query
         * @private
         */
        _performSearch: function(query) {
            // Override in concrete controller
            Log.info("Search performed", { query: query });
        },
        
        /**
         * Standard view mode change handler
         * @param {Object} event - View mode change event
         */
        onViewModeChange: function(event) {
            const viewMode = event.getParameter("item").getKey();
            this.updatePageState({ viewMode: viewMode });
            this._changeViewMode(viewMode);
        },
        
        /**
         * Change view mode (to be implemented by concrete controller)
         * @param {string} viewMode - New view mode
         * @private
         */
        _changeViewMode: function(viewMode) {
            // Override in concrete controller
            Log.info("View mode changed", { viewMode: viewMode });
        }
    };
});