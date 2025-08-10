/*!
 * Loading State Mixin for Consistent Loading Management
 * Provides standardized loading state management across all controllers
 */

sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/core/Fragment"
], function(
    BaseObject,
    Fragment
) {
    "use strict";

    /**
     * Loading State Mixin
     * Standardizes loading state management across all views
     * @namespace com.sap.a2a.portal.controller.mixins
     */
    return {
        
        /**
         * Initialize loading state management
         * Call this in onInit() of your controller
         */
        initLoadingStates: function() {
            this._loadingStates = new Map();
            this._loadingFragment = null;
            this._loadingPromises = new Map();
            
            // Default loading configurations
            this._loadingConfigs = {
                mainContent: {
                    type: "spinner",
                    size: "large",
                    message: "Loading content...",
                    overlay: false
                },
                data: {
                    type: "skeleton", 
                    size: "medium",
                    message: "Loading data...",
                    overlay: false
                },
                chart: {
                    type: "shimmer",
                    size: "large", 
                    message: "Preparing visualization...",
                    overlay: false
                },
                agentProcessing: {
                    type: "dots",
                    size: "medium",
                    message: "Processing with AI agents...",
                    overlay: false
                },
                search: {
                    type: "pulse",
                    size: "small",
                    message: "Searching...",
                    overlay: false
                },
                upload: {
                    type: "progress",
                    size: "medium", 
                    message: "Uploading file...",
                    overlay: false
                },
                pageTransition: {
                    type: "spinner",
                    size: "medium",
                    message: "Loading page...",
                    overlay: true
                },
                network: {
                    type: "dots",
                    size: "small",
                    message: "Connecting...",
                    overlay: false
                },
                computation: {
                    type: "pulse",
                    size: "large",
                    message: "Processing complex calculations...",
                    overlay: false,
                    speed: "slow"
                }
            };
        },

        /**
         * Load the loading state fragment
         * @private
         */
        _ensureLoadingFragment: async function() {
            if (!this._loadingFragment) {
                this._loadingFragment = await Fragment.load({
                    id: this.getView().getId(),
                    name: "com.sap.a2a.portal.fragment.LoadingStateManager",
                    controller: this
                });
                
                // Add to view but keep invisible initially
                this.getView().addDependent(this._loadingFragment);
            }
            return this._loadingFragment;
        },

        /**
         * Show loading state
         * @param {string} loadingId - Unique identifier for the loading state
         * @param {object} options - Loading configuration options
         * @param {string} options.type - Loading type (spinner, dots, skeleton, etc.)
         * @param {string} options.size - Size (small, medium, large)
         * @param {string} options.message - Loading message
         * @param {boolean} options.overlay - Show as overlay
         * @param {string} options.targetId - Target control ID to attach loading to
         */
        showLoading: async function(loadingId, options = {}) {
            try {
                await this._ensureLoadingFragment();
                
                // Merge with default configuration
                const config = Object.assign(
                    {},
                    this._loadingConfigs[loadingId] || this._loadingConfigs.mainContent,
                    options
                );
                
                // Get or create loading indicator
                let loadingIndicator = this._getLoadingIndicator(loadingId, config);
                
                if (!loadingIndicator) {
                    console.warn(`Loading indicator not found for ID: ${loadingId}`);
                    return;
                }
                
                // Configure the loading indicator
                this._configureLoadingIndicator(loadingIndicator, config);
                
                // Show the loading indicator
                if (config.targetId) {
                    this._attachToTarget(loadingIndicator, config.targetId);
                } else {
                    loadingIndicator.setVisible(true);
                }
                
                // Track loading state
                this._loadingStates.set(loadingId, {
                    active: true,
                    startTime: new Date(),
                    config: config
                });
                
                // Auto-hide after timeout if specified
                if (config.timeout) {
                    setTimeout(() => {
                        this.hideLoading(loadingId);
                    }, config.timeout);
                }
                
            } catch (error) {
                console.error("Error showing loading state:", error);
            }
        },

        /**
         * Hide loading state
         * @param {string} loadingId - Loading identifier to hide
         */
        hideLoading: function(loadingId) {
            try {
                const loadingIndicator = this._getLoadingIndicator(loadingId);
                
                if (loadingIndicator) {
                    loadingIndicator.setVisible(false);
                    
                    // If it was attached to a target, detach it
                    const loadingState = this._loadingStates.get(loadingId);
                    if (loadingState && loadingState.config.targetId) {
                        this._detachFromTarget(loadingIndicator, loadingState.config.targetId);
                    }
                }
                
                // Update loading state
                if (this._loadingStates.has(loadingId)) {
                    const loadingState = this._loadingStates.get(loadingId);
                    loadingState.active = false;
                    loadingState.endTime = new Date();
                    loadingState.duration = loadingState.endTime - loadingState.startTime;
                }
                
            } catch (error) {
                console.error("Error hiding loading state:", error);
            }
        },

        /**
         * Update loading progress (for progress type loaders)
         * @param {string} loadingId - Loading identifier
         * @param {number} progress - Progress percentage (0-100)
         * @param {string} message - Optional message update
         */
        updateLoadingProgress: function(loadingId, progress, message) {
            try {
                const loadingIndicator = this._getLoadingIndicator(loadingId);
                
                if (loadingIndicator && loadingIndicator.getType() === "progress") {
                    loadingIndicator.setProgress(progress);
                    
                    if (message) {
                        loadingIndicator.setMessage(message);
                    }
                }
            } catch (error) {
                console.error("Error updating loading progress:", error);
            }
        },

        /**
         * Show loading for a specific operation with automatic management
         * @param {string} loadingId - Loading identifier
         * @param {Promise|Function} operation - Promise or function to execute
         * @param {object} options - Loading options
         * @returns {Promise} - The operation result
         */
        withLoading: async function(loadingId, operation, options = {}) {
            try {
                // Show loading
                await this.showLoading(loadingId, options);
                
                let result;
                
                // Execute operation
                if (typeof operation === 'function') {
                    result = await operation();
                } else if (operation && typeof operation.then === 'function') {
                    result = await operation;
                } else {
                    throw new Error("Operation must be a Promise or Function");
                }
                
                // Hide loading on success
                this.hideLoading(loadingId);
                
                return result;
                
            } catch (error) {
                // Hide loading on error
                this.hideLoading(loadingId);
                throw error;
            }
        },

        /**
         * Show global page loading overlay
         * @param {string} message - Loading message
         */
        showGlobalLoading: function(message = "Loading...") {
            sap.ui.core.BusyIndicator.show(0, message);
        },

        /**
         * Hide global page loading overlay
         */
        hideGlobalLoading: function() {
            sap.ui.core.BusyIndicator.hide();
        },

        /**
         * Check if a loading state is currently active
         * @param {string} loadingId - Loading identifier
         * @returns {boolean} - True if loading is active
         */
        isLoading: function(loadingId) {
            const loadingState = this._loadingStates.get(loadingId);
            return loadingState && loadingState.active;
        },

        /**
         * Get loading statistics
         * @param {string} loadingId - Loading identifier
         * @returns {object} - Loading statistics
         */
        getLoadingStats: function(loadingId) {
            return this._loadingStates.get(loadingId) || null;
        },

        /**
         * Clear all loading states
         */
        clearAllLoading: function() {
            for (const loadingId of this._loadingStates.keys()) {
                this.hideLoading(loadingId);
            }
        },

        /**
         * Get loading indicator by ID
         * @private
         */
        _getLoadingIndicator: function(loadingId, config) {
            if (!this._loadingFragment) return null;
            
            // Try predefined loaders first
            const predefinedId = loadingId + "Loader";
            let indicator = sap.ui.getCore().byId(this.getView().getId() + "--" + predefinedId);
            
            if (!indicator) {
                // Try direct ID
                indicator = sap.ui.getCore().byId(this.getView().getId() + "--" + loadingId);
            }
            
            return indicator;
        },

        /**
         * Configure loading indicator properties
         * @private
         */
        _configureLoadingIndicator: function(indicator, config) {
            if (config.type) indicator.setType(config.type);
            if (config.size) indicator.setSize(config.size);
            if (config.message) indicator.setMessage(config.message);
            if (config.showText !== undefined) indicator.setShowText(config.showText);
            if (config.overlay !== undefined) indicator.setOverlay(config.overlay);
            if (config.speed) indicator.setSpeed(config.speed);
            if (config.progress !== undefined) indicator.setProgress(config.progress);
        },

        /**
         * Attach loading indicator to target control
         * @private
         */
        _attachToTarget: function(loadingIndicator, targetId) {
            try {
                const targetControl = this.byId(targetId);
                if (targetControl && targetControl.addContent) {
                    targetControl.addContent(loadingIndicator);
                } else if (targetControl && targetControl.addItem) {
                    targetControl.addItem(loadingIndicator);
                }
            } catch (error) {
                console.warn("Could not attach loading indicator to target:", error);
            }
        },

        /**
         * Detach loading indicator from target control
         * @private
         */
        _detachFromTarget: function(loadingIndicator, targetId) {
            try {
                const targetControl = this.byId(targetId);
                if (targetControl && targetControl.removeContent) {
                    targetControl.removeContent(loadingIndicator);
                } else if (targetControl && targetControl.removeItem) {
                    targetControl.removeItem(loadingIndicator);
                }
            } catch (error) {
                console.warn("Could not detach loading indicator from target:", error);
            }
        },

        /**
         * Predefined loading methods for common scenarios
         */
        
        showMainContentLoading: function(message) {
            return this.showLoading("mainContent", { message: message });
        },
        
        showDataLoading: function() {
            return this.showLoading("data");
        },
        
        showChartLoading: function() {
            return this.showLoading("chart");
        },
        
        showAgentProcessingLoading: function(agentName) {
            return this.showLoading("agentProcessing", {
                message: `Processing with ${agentName}...`
            });
        },
        
        showSearchLoading: function() {
            return this.showLoading("search");
        },
        
        showUploadLoading: function() {
            return this.showLoading("upload");
        },
        
        hideMainContentLoading: function() {
            this.hideLoading("mainContent");
        },
        
        hideDataLoading: function() {
            this.hideLoading("data");
        },
        
        hideChartLoading: function() {
            this.hideLoading("chart");
        },
        
        hideAgentProcessingLoading: function() {
            this.hideLoading("agentProcessing");
        },
        
        hideSearchLoading: function() {
            this.hideLoading("search");
        },
        
        hideUploadLoading: function() {
            this.hideLoading("upload");
        }
    };
});