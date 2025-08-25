sap.ui.define([], () => {
    'use strict';

    /**
     * PFCG Role Mapping Configuration for SAP Enterprise
     * Maps backend PFCG roles to Fiori Launchpad content
     */
    return {
        /**
         * Role to Catalog Mapping
         * Defines which catalogs are available for each role
         */
        roleToCatalogMapping: {
            'SAP_UI2_USER_700': [
                'SAP_FIORI_FOUNDATION',
                'SAP_FIORI_EXTENSIBILITY',
                'A2A_NETWORK_STANDARD'
            ],
            'SAP_UI2_USER_750': [
                'SAP_FIORI_FOUNDATION',
                'SAP_FIORI_EXTENSIBILITY',
                'A2A_NETWORK_STANDARD',
                'A2A_NETWORK_ADVANCED'
            ],
            'SAP_UI2_ADMIN_700': [
                'SAP_FIORI_FOUNDATION_ADMIN',
                'SAP_FIORI_EXTENSIBILITY_ADMIN',
                'A2A_NETWORK_ADMIN'
            ],
            'FLP_USER': [
                'A2A_NETWORK_STANDARD'
            ],
            'FLP_ADMIN': [
                'A2A_NETWORK_STANDARD',
                'A2A_NETWORK_ADMIN'
            ],
            'A2A_AGENT_OPERATOR': [
                'A2A_AGENT_CATALOG'
            ],
            'A2A_WORKFLOW_DESIGNER': [
                'A2A_WORKFLOW_CATALOG',
                'A2A_SERVICE_CATALOG'
            ],
            'A2A_SECURITY_AUDITOR': [
                'A2A_SECURITY_CATALOG'
            ]
        },

        /**
         * Role to Group Mapping
         * Defines default tile groups for each role
         */
        roleToGroupMapping: {
            'SAP_UI2_USER_700': [
                'core_processing_agents',
                'analytics_monitoring'
            ],
            'SAP_UI2_USER_750': [
                'core_processing_agents',
                'analytics_monitoring',
                'services_workflow'
            ],
            'SAP_UI2_ADMIN_700': [
                '*' // All groups
            ],
            'FLP_USER': [
                'core_processing_agents',
                'analytics_monitoring',
                'services_workflow'
            ],
            'FLP_ADMIN': [
                '*' // All groups
            ],
            'A2A_AGENT_OPERATOR': [
                'core_processing_agents',
                'specialized_agents'
            ],
            'A2A_WORKFLOW_DESIGNER': [
                'services_workflow',
                'integration_governance'
            ],
            'A2A_SECURITY_AUDITOR': [
                'security_compliance',
                'analytics_monitoring'
            ]
        },

        /**
         * Authorization Object Mapping
         * Maps PFCG authorization objects to Fiori capabilities
         */
        authObjectMapping: {
            'S_SERVICE': {
                'checkFunction': function(authValues, requestedService) {
                    return authValues.SRV_NAME && authValues.SRV_NAME.includes(requestedService);
                }
            },
            '/UI2/CHIP': {
                'checkFunction': function(authValues, requestedChip) {
                    return authValues.CHIP_ID &&
                           (authValues.CHIP_ID.includes('*') || authValues.CHIP_ID.includes(requestedChip));
                }
            },
            'S_RFCACL': {
                'checkFunction': function(authValues, requestedRFC) {
                    if (!authValues.RFC_NAME) return false;
                    return authValues.RFC_NAME.some(pattern => {
                        if (pattern.includes('*')) {
                            const regex = new RegExp(`^${  pattern.replace('*', '.*')  }$`);
                            return regex.test(requestedRFC);
                        }
                        return pattern === requestedRFC;
                    });
                }
            }
        },

        /**
         * Tile Authorization Mapping
         * Maps tiles to required authorization objects
         */
        tileAuthorizationMapping: {
            'agent0_data_product': {
                'authObject': '/UI2/CHIP',
                'authField': 'CHIP_ID',
                'authValue': 'A2A_AGENT_0'
            },
            'agent1_validation': {
                'authObject': '/UI2/CHIP',
                'authField': 'CHIP_ID',
                'authValue': 'A2A_AGENT_1'
            },
            'service_marketplace': {
                'authObject': 'S_SERVICE',
                'authField': 'SRV_NAME',
                'authValue': 'A2A_MARKETPLACE'
            },
            'workflow_designer': {
                'authObject': 'S_SERVICE',
                'authField': 'SRV_NAME',
                'authValue': 'A2A_WORKFLOW'
            },
            'security_audit': {
                'authObject': 'S_SERVICE',
                'authField': 'SRV_NAME',
                'authValue': 'A2A_SECURITY'
            }
        },

        /**
         * Get authorized catalogs for user roles
         * @param {Array} userRoles - Array of user roles
         * @returns {Array} Authorized catalog IDs
         */
        getAuthorizedCatalogs: function(userRoles) {
            const catalogs = new Set();
            userRoles.forEach(role => {
                const roleCatalogs = this.roleToCatalogMapping[role];
                if (roleCatalogs) {
                    roleCatalogs.forEach(catalog => catalogs.add(catalog));
                }
            });
            return Array.from(catalogs);
        },

        /**
         * Get authorized groups for user roles
         * @param {Array} userRoles - Array of user roles
         * @returns {Array} Authorized group IDs
         */
        getAuthorizedGroups: function(userRoles) {
            const groups = new Set();
            userRoles.forEach(role => {
                const roleGroups = this.roleToGroupMapping[role];
                if (roleGroups) {
                    if (roleGroups.includes('*')) {
                        return ['*']; // All groups
                    }
                    roleGroups.forEach(group => groups.add(group));
                }
            });
            return Array.from(groups);
        },

        /**
         * Check tile authorization
         * @param {string} tileId - Tile ID to check
         * @param {Object} userAuthorizations - User's authorization values
         * @returns {boolean} Is authorized
         */
        isTileAuthorized: function(tileId, userAuthorizations) {
            const tileAuth = this.tileAuthorizationMapping[tileId];
            if (!tileAuth) {
                // No specific authorization required
                return true;
            }

            const authObjectDef = this.authObjectMapping[tileAuth.authObject];
            if (!authObjectDef || !authObjectDef.checkFunction) {
                return false;
            }

            const userAuthValues = userAuthorizations[tileAuth.authObject];
            if (!userAuthValues) {
                return false;
            }

            return authObjectDef.checkFunction(userAuthValues, tileAuth.authValue);
        }
    };
});