/**
 * @fileoverview XSUAA Configuration Helper
 * @since 1.0.0
 * @module xsuaaConfig
 *
 * Provides robust XSUAA configuration with multiple fallback mechanisms
 * and proper validation for production environments
 */

const xsenv = require('@sap/xsenv');
const cds = require('@sap/cds');

/**
 * Validates XSUAA configuration for completeness
 * @param {Object} config - XSUAA configuration object
 * @returns {boolean} - true if valid, false otherwise
 */
const validateXSUAAConfig = (config) => {
    if (!config || !config.credentials) return false;

    const required = ['clientid', 'clientsecret', 'url'];
    const credentials = config.credentials;

    for (const field of required) {
        if (!credentials[field]) {
            cds.log('xsuaa').warn(`Missing required XSUAA field: ${field}`);
            return false;
        }
    }

    return true;
};

/**
 * Get XSUAA configuration from various sources with fallbacks
 * @returns {Object|null} XSUAA configuration or null if not found
 */
const getXSUAAConfig = () => {
    const log = cds.log('xsuaa');

    try {
        // 1. Try CDS configuration first
        if (cds.env.requires?.auth?.credentials) {
            const cdsConfig = {
                credentials: cds.env.requires.auth.credentials,
                verificationKey: cds.env.requires.auth.credentials.verificationkey
            };
            if (validateXSUAAConfig(cdsConfig)) {
                log.info('XSUAA configuration loaded from CDS environment');
                return cdsConfig;
            }
        }

        // 2. Try VCAP_SERVICES (Cloud Foundry)
        if (process.env.VCAP_SERVICES) {
            try {
                const vcapServices = JSON.parse(process.env.VCAP_SERVICES);
                const xsuaaService = vcapServices.xsuaa?.[0];

                if (xsuaaService?.credentials) {
                    const vcapConfig = {
                        credentials: xsuaaService.credentials,
                        verificationKey: xsuaaService.credentials.verificationkey
                    };
                    if (validateXSUAAConfig(vcapConfig)) {
                        log.info('XSUAA configuration loaded from VCAP_SERVICES');
                        return vcapConfig;
                    }
                }
            } catch (parseError) {
                log.error('Failed to parse VCAP_SERVICES:', parseError.message);
            }
        }

        // 3. Try environment variables (Kubernetes/Docker)
        if (process.env.XSUAA_CLIENT_ID && process.env.XSUAA_CLIENT_SECRET) {
            const envConfig = {
                credentials: {
                    clientid: process.env.XSUAA_CLIENT_ID,
                    clientsecret: process.env.XSUAA_CLIENT_SECRET,
                    url: process.env.XSUAA_URL || process.env.XSUAA_AUTH_URL,
                    uaadomain: process.env.XSUAA_UAA_DOMAIN,
                    verificationkey: process.env.XSUAA_VERIFICATION_KEY,
                    xsappname: process.env.XSUAA_XS_APP_NAME || 'a2a-network',
                    identityzone: process.env.XSUAA_IDENTITY_ZONE || 'sap-provisioning',
                    identityzoneid: process.env.XSUAA_IDENTITY_ZONE_ID,
                    tenantid: process.env.XSUAA_TENANT_ID,
                    tenantmode: process.env.XSUAA_TENANT_MODE || 'dedicated',
                    sburl: process.env.XSUAA_SB_URL,
                    apiurl: process.env.XSUAA_API_URL,
                    subaccountid: process.env.XSUAA_SUBACCOUNT_ID,
                    zoneid: process.env.XSUAA_ZONE_ID,
                    credential_type: process.env.XSUAA_CREDENTIAL_TYPE
                }
            };

            if (validateXSUAAConfig(envConfig)) {
                log.info('XSUAA configuration loaded from environment variables');
                return envConfig;
            }
        }

        // 4. Try xsenv service lookup
        try {
            xsenv.loadEnv();
            const services = xsenv.getServices({ xsuaa: { tag: 'xsuaa' } });
            if (services.xsuaa) {
                const xsenvConfig = {
                    credentials: services.xsuaa,
                    verificationKey: services.xsuaa.verificationkey
                };
                if (validateXSUAAConfig(xsenvConfig)) {
                    log.info('XSUAA configuration loaded from xsenv');
                    return xsenvConfig;
                }
            }
        } catch (xsenvError) {
            log.debug('xsenv service lookup failed:', xsenvError.message);
        }

        // 5. Development fallback (only in non-production)
        if (process.env.NODE_ENV !== 'production' && process.env.USE_DEVELOPMENT_AUTH === 'true') {
            log.warn('No XSUAA configuration found - using development mode');
            return {
                isDevelopment: true,
                credentials: {
                    clientid: 'dev-client',
                    clientsecret: 'dev-secret',
                    url: 'http://localhost:8080/uaa',
                    xsappname: 'a2a-network-dev'
                }
            };
        }

        // No configuration found
        if (process.env.NODE_ENV === 'production') {
            throw new Error('XSUAA configuration is required in production environment');
        }

        log.warn('No XSUAA configuration found. Authentication will not work properly.');
        return null;

    } catch (error) {
        log.error('Error loading XSUAA configuration:', error.message);
        if (process.env.NODE_ENV === 'production') {
            throw error;
        }
        return null;
    }
};

/**
 * Initialize XSUAA configuration at startup
 * @returns {Object} Configuration status
 */
const initializeXSUAA = () => {
    const log = cds.log('xsuaa');

    try {
        const config = getXSUAAConfig();

        if (!config) {
            return {
                initialized: false,
                error: 'No XSUAA configuration available'
            };
        }

        if (config.isDevelopment) {
            return {
                initialized: true,
                mode: 'development',
                warning: 'Using development authentication mode'
            };
        }

        return {
            initialized: true,
            mode: 'production',
            clientId: config.credentials.clientid,
            xsAppName: config.credentials.xsappname,
            tenantMode: config.credentials.tenantmode
        };

    } catch (error) {
        log.error('Failed to initialize XSUAA:', error.message);
        return {
            initialized: false,
            error: error.message
        };
    }
};

module.exports = {
    getXSUAAConfig,
    validateXSUAAConfig,
    initializeXSUAA
};