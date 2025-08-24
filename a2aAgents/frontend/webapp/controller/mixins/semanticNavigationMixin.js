/*!
 * Semantic Navigation Mixin for Intent-Based Routing
 * Provides SAP Fiori compliant cross-app navigation
 */

sap.ui.define([
    'sap/ui/base/Object',
    'sap/ushell/services/CrossApplicationNavigation'
], (
    BaseObject,
    CrossApplicationNavigation
) => {
    'use strict';

    /**
     * Semantic Navigation Mixin
     * Provides intent-based navigation for cross-app scenarios
     * @namespace com.sap.a2a.portal.controller.mixins
     */
    return {
        
        /**
         * Initialize semantic navigation
         * Call this in onInit() of your controller
         */
        initSemanticNavigation: function() {
            this._crossAppNavigator = sap.ushell.Container.getService('CrossApplicationNavigation');
            this._urlParsing = sap.ushell.Container.getService('URLParsing');
            
            // Cache for navigation targets
            this._navigationTargets = new Map();
            
            // Initialize supported intents
            this._initializeSupportedIntents();
        },

        /**
         * Initialize supported navigation intents
         * @private
         */
        _initializeSupportedIntents: function() {
            this._supportedIntents = {
                // Project intents
                'Project-display': {
                    semanticObject: 'Project',
                    action: 'display',
                    parameters: ['projectId']
                },
                'Project-manage': {
                    semanticObject: 'Project',
                    action: 'manage',
                    parameters: []
                },
                'Project-create': {
                    semanticObject: 'Project', 
                    action: 'create',
                    parameters: []
                },
                
                // Agent intents
                'Agent-display': {
                    semanticObject: 'Agent',
                    action: 'display', 
                    parameters: ['agentId', 'agentType']
                },
                'Agent-manage': {
                    semanticObject: 'Agent',
                    action: 'manage',
                    parameters: []
                },
                
                // Analytics intents
                'Analytics-display': {
                    semanticObject: 'Analytics',
                    action: 'display',
                    parameters: ['analyticsType', 'timeRange']
                },
                
                // Workflow intents
                'Workflow-manage': {
                    semanticObject: 'Workflow',
                    action: 'manage',
                    parameters: []
                },
                
                // Settings intents
                'Settings-display': {
                    semanticObject: 'Settings',
                    action: 'display',
                    parameters: []
                },
                
                // External SAP intents
                'BusinessPartner-display': {
                    semanticObject: 'BusinessPartner',
                    action: 'display',
                    parameters: ['businessPartnerId'],
                    external: true
                },
                'Customer-display': {
                    semanticObject: 'Customer',
                    action: 'display',
                    parameters: ['customerId'],
                    external: true
                }
            };
        },

        /**
         * Navigate to a specific intent with parameters
         * @param {string} sIntent Intent key (e.g., "Project-display")
         * @param {object} mParameters Navigation parameters
         * @param {boolean} bNewWindow Open in new window
         * @returns {Promise} Navigation promise
         */
        navigateToIntent: function(sIntent, mParameters, bNewWindow) {
            const oIntent = this._supportedIntents[sIntent];
            
            if (!oIntent) {
                return Promise.reject(new Error(`Unsupported intent: ${  sIntent}`));
            }

            const oNavParams = this._buildNavigationParameters(oIntent, mParameters);
            
            return this._performNavigation(oIntent.semanticObject, oIntent.action, oNavParams, bNewWindow);
        },

        /**
         * Navigate to project display
         * @param {string} sProjectId Project ID
         * @param {object} mAdditionalParams Additional parameters
         * @returns {Promise} Navigation promise
         */
        navigateToProject: function(sProjectId, mAdditionalParams) {
            const mParams = Object.assign({
                projectId: sProjectId
            }, mAdditionalParams || {});
            
            return this.navigateToIntent('Project-display', mParams);
        },

        /**
         * Navigate to projects list
         * @param {object} mFilters Filter parameters
         * @returns {Promise} Navigation promise  
         */
        navigateToProjectsList: function(mFilters) {
            return this.navigateToIntent('Project-manage', mFilters);
        },

        /**
         * Navigate to project creation
         * @param {object} mDefaults Default values
         * @returns {Promise} Navigation promise
         */
        navigateToCreateProject: function(mDefaults) {
            return this.navigateToIntent('Project-create', mDefaults);
        },

        /**
         * Navigate to agent display
         * @param {string} sAgentId Agent ID
         * @param {string} sAgentType Agent type (optional)
         * @returns {Promise} Navigation promise
         */
        navigateToAgent: function(sAgentId, sAgentType) {
            const mParams = {
                agentId: sAgentId
            };
            
            if (sAgentType) {
                mParams.agentType = sAgentType;
            }
            
            return this.navigateToIntent('Agent-display', mParams);
        },

        /**
         * Navigate to agents overview
         * @param {object} mFilters Filter parameters
         * @returns {Promise} Navigation promise
         */
        navigateToAgentsOverview: function(mFilters) {
            return this.navigateToIntent('Agent-manage', mFilters);
        },

        /**
         * Navigate to analytics dashboard
         * @param {string} sAnalyticsType Analytics type
         * @param {string} sTimeRange Time range
         * @returns {Promise} Navigation promise
         */
        navigateToAnalytics: function(sAnalyticsType, sTimeRange) {
            const mParams = {};
            
            if (sAnalyticsType) {
                mParams.analyticsType = sAnalyticsType;
            }
            
            if (sTimeRange) {
                mParams.timeRange = sTimeRange;
            }
            
            return this.navigateToIntent('Analytics-display', mParams);
        },

        /**
         * Navigate to SAP Business Partner
         * @param {string} sBusinessPartnerId Business Partner ID
         * @returns {Promise} Navigation promise
         */
        navigateToSAPBusinessPartner: function(sBusinessPartnerId) {
            return this.navigateToIntent('BusinessPartner-display', {
                businessPartnerId: sBusinessPartnerId
            }, true); // Open in new window for external navigation
        },

        /**
         * Navigate to SAP Customer
         * @param {string} sCustomerId Customer ID
         * @returns {Promise} Navigation promise
         */
        navigateToSAPCustomer: function(sCustomerId) {
            return this.navigateToIntent('Customer-display', {
                customerId: sCustomerId
            }, true); // Open in new window for external navigation
        },

        /**
         * Navigate to workflow management
         * @param {object} mFilters Filter parameters
         * @returns {Promise} Navigation promise
         */
        navigateToWorkflowManagement: function(mFilters) {
            return this.navigateToIntent('Workflow-manage', mFilters);
        },

        /**
         * Navigate to settings
         * @param {string} sSettingsTab Settings tab to open
         * @returns {Promise} Navigation promise
         */
        navigateToSettings: function(sSettingsTab) {
            const mParams = {};
            if (sSettingsTab) {
                mParams.tab = sSettingsTab;
            }
            
            return this.navigateToIntent('Settings-display', mParams);
        },

        /**
         * Navigate back to previous app/context
         * @returns {Promise} Navigation promise
         */
        navigateBack: function() {
            if (this._crossAppNavigator) {
                return this._crossAppNavigator.backToPreviousApp();
            }
            
            // Fallback to browser history
            window.history.back();
            return Promise.resolve();
        },

        /**
         * Build navigation parameters
         * @private
         * @param {object} oIntent Intent configuration
         * @param {object} mParameters Raw parameters
         * @returns {object} Formatted navigation parameters
         */
        _buildNavigationParameters: function(oIntent, mParameters) {
            const oNavParams = {};
            
            if (!mParameters) {
                return oNavParams;
            }
            
            // Map known parameters
            if (oIntent.parameters) {
                oIntent.parameters.forEach((sParam) => {
                    if (mParameters[sParam] !== undefined) {
                        oNavParams[sParam] = [mParameters[sParam]];
                    }
                });
            }
            
            // Add additional parameters
            Object.keys(mParameters).forEach((sKey) => {
                if (oIntent.parameters.indexOf(sKey) === -1) {
                    oNavParams[sKey] = [mParameters[sKey]];
                }
            });
            
            return oNavParams;
        },

        /**
         * Perform the actual navigation
         * @private
         * @param {string} sSemanticObject Semantic object
         * @param {string} sAction Action
         * @param {object} mParameters Parameters
         * @param {boolean} bNewWindow Open in new window
         * @returns {Promise} Navigation promise
         */
        _performNavigation: function(sSemanticObject, sAction, mParameters, bNewWindow) {
            if (!this._crossAppNavigator) {
                return Promise.reject(new Error('Cross application navigation service not available'));
            }
            
            const sHref = this._crossAppNavigator.hrefForExternal({
                target: {
                    semanticObject: sSemanticObject,
                    action: sAction
                },
                params: mParameters
            });
            
            if (bNewWindow) {
                // Open in new window/tab
                window.open(sHref, '_blank');
                return Promise.resolve();
            } else {
                // Navigate in current window
                return this._crossAppNavigator.toExternal({
                    target: {
                        semanticObject: sSemanticObject,
                        action: sAction
                    },
                    params: mParameters
                });
            }
        },

        /**
         * Check if an intent is supported
         * @param {string} sSemanticObject Semantic object
         * @param {string} sAction Action
         * @returns {Promise<boolean>} Promise resolving to support status
         */
        isIntentSupported: function(sSemanticObject, sAction) {
            if (!this._crossAppNavigator) {
                return Promise.resolve(false);
            }
            
            const aIntents = [{
                semanticObject: sSemanticObject,
                action: sAction
            }];
            
            return this._crossAppNavigator.isIntentSupported(aIntents)
                .then((mSupported) => {
                    const sIntentKey = `${sSemanticObject  }-${  sAction}`;
                    return !!(mSupported && mSupported[sIntentKey] && mSupported[sIntentKey].supported);
                });
        },

        /**
         * Get available actions for a semantic object
         * @param {string} sSemanticObject Semantic object
         * @returns {Promise<Array>} Promise resolving to array of available actions
         */
        getAvailableActions: function(sSemanticObject) {
            if (!this._crossAppNavigator) {
                return Promise.resolve([]);
            }
            
            return this._crossAppNavigator.getLinks({
                semanticObject: sSemanticObject
            }).then((aLinks) => {
                return aLinks.map((oLink) => {
                    return {
                        action: oLink.action,
                        text: oLink.text,
                        subTitle: oLink.subTitle,
                        icon: oLink.icon
                    };
                });
            });
        },

        /**
         * Parse current URL parameters
         * @returns {object} Parsed URL parameters
         */
        parseCurrentUrlParameters: function() {
            if (!this._urlParsing) {
                return {};
            }
            
            const sCurrentHash = window.location.hash;
            const oParsedShellHash = this._urlParsing.parseShellHash(sCurrentHash);
            
            if (oParsedShellHash && oParsedShellHash.params) {
                const mParams = {};
                Object.keys(oParsedShellHash.params).forEach((sKey) => {
                    const aValues = oParsedShellHash.params[sKey];
                    mParams[sKey] = aValues && aValues.length > 0 ? aValues[0] : '';
                });
                return mParams;
            }
            
            return {};
        },

        /**
         * Generate href for intent
         * @param {string} sIntent Intent key
         * @param {object} mParameters Parameters
         * @returns {string} Generated href
         */
        getHrefForIntent: function(sIntent, mParameters) {
            const oIntent = this._supportedIntents[sIntent];
            
            if (!oIntent || !this._crossAppNavigator) {
                return '#';
            }
            
            const oNavParams = this._buildNavigationParameters(oIntent, mParameters);
            
            return this._crossAppNavigator.hrefForExternal({
                target: {
                    semanticObject: oIntent.semanticObject,
                    action: oIntent.action
                },
                params: oNavParams
            });
        },

        /**
         * Create navigation menu items for related applications
         * @param {string} sCurrentSemanticObject Current semantic object
         * @returns {Promise<Array>} Promise resolving to menu items
         */
        createRelatedAppsMenu: function(sCurrentSemanticObject) {
            const aRelatedObjects = this._getRelatedSemanticObjects(sCurrentSemanticObject);
            const aPromises = [];
            
            aRelatedObjects.forEach((sObject) => {
                aPromises.push(this.getAvailableActions(sObject));
            });
            
            return Promise.all(aPromises).then((aResults) => {
                const aMenuItems = [];
                
                aResults.forEach((aActions, iIndex) => {
                    const sSemanticObject = aRelatedObjects[iIndex];
                    
                    aActions.forEach((oAction) => {
                        aMenuItems.push({
                            text: oAction.text,
                            icon: oAction.icon,
                            semanticObject: sSemanticObject,
                            action: oAction.action,
                            press: function() {
                                this.navigateToIntent(`${sSemanticObject  }-${  oAction.action}`);
                            }.bind(this)
                        });
                    });
                });
                
                return aMenuItems;
            });
        },

        /**
         * Get related semantic objects
         * @private
         * @param {string} sCurrentObject Current semantic object
         * @returns {Array<string>} Array of related semantic objects
         */
        _getRelatedSemanticObjects: function(sCurrentObject) {
            const mRelations = {
                'Project': ['Agent', 'Analytics', 'Workflow'],
                'Agent': ['Project', 'Analytics'],
                'Analytics': ['Project', 'Agent'],
                'Workflow': ['Project', 'Agent'],
                'BusinessPartner': ['Customer', 'Project'],
                'Customer': ['BusinessPartner', 'Project']
            };
            
            return mRelations[sCurrentObject] || [];
        },

        /**
         * Handle intent-based deep linking on app startup
         * @param {object} mStartupParameters Startup parameters
         */
        handleDeepLinking: function(mStartupParameters) {
            if (mStartupParameters && Object.keys(mStartupParameters).length > 0) {
                // Process startup parameters for deep linking
                this._processDeepLinkParameters(mStartupParameters);
            }
        },

        /**
         * Process deep link parameters
         * @private
         * @param {object} mParameters Deep link parameters
         */
        _processDeepLinkParameters: function(mParameters) {
            // Implement specific deep linking logic based on parameters
            if (mParameters.projectId) {
                this.navigateToProject(mParameters.projectId[0]);
            } else if (mParameters.agentId) {
                const sAgentType = mParameters.agentType ? mParameters.agentType[0] : undefined;
                this.navigateToAgent(mParameters.agentId[0], sAgentType);
            } else if (mParameters.analyticsType) {
                const sTimeRange = mParameters.timeRange ? mParameters.timeRange[0] : undefined;
                this.navigateToAnalytics(mParameters.analyticsType[0], sTimeRange);
            }
        }
    };
});