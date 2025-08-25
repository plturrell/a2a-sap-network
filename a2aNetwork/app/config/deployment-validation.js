/* global sap */
sap.ui.define([
    'sap/base/Log'
], (Log) => {
    'use strict';

    /**
     * Production Deployment Validation
     * Validates all required services and configurations for enterprise SAP deployment
     */
    return {
        
        /**
         * Validate BTP service bindings
         */
        validateBTPBindings: function() {
            const errors = [];
            
            // Check for VCAP_SERVICES environment variable
            if (typeof process !== 'undefined' && process.env && process.env.VCAP_SERVICES) {
                const vcapServices = JSON.parse(process.env.VCAP_SERVICES);
                
                // Required SAP BTP services
                const requiredServices = [
                    'xsuaa',           // Authentication service
                    'hana',            // Database service
                    'application-logs', // Logging service
                    'autoscaler'       // Auto-scaling service
                ];
                
                requiredServices.forEach(service => {
                    if (!vcapServices[service] || vcapServices[service].length === 0) {
                        errors.push(`Missing required BTP service binding: ${service}`);
                    }
                });
                
                // Validate XSUAA configuration
                if (vcapServices.xsuaa && vcapServices.xsuaa[0]) {
                    const xsuaaConfig = vcapServices.xsuaa[0].credentials;
                    if (!xsuaaConfig.clientid || !xsuaaConfig.clientsecret || !xsuaaConfig.url) {
                        errors.push('Invalid XSUAA service binding configuration');
                    }
                }
                
                // Validate HANA configuration
                if (vcapServices.hana && vcapServices.hana[0]) {
                    const hanaConfig = vcapServices.hana[0].credentials;
                    if (!hanaConfig.host || !hanaConfig.port || !hanaConfig.user) {
                        errors.push('Invalid HANA service binding configuration');
                    }
                }
                
            } else {
                // For non-BTP environments, check for alternative configuration
                Log.warning('VCAP_SERVICES not found - checking alternative configuration');
                
                if (!window.A2A_CONFIG) {
                    errors.push('No configuration found - neither VCAP_SERVICES nor window.A2A_CONFIG');
                }
            }
            
            return {
                valid: errors.length === 0,
                errors: errors
            };
        },
        
        /**
         * Validate environment configuration
         */
        validateEnvironmentConfig: function() {
            const errors = [];
            const warnings = [];
            
            // Check required environment variables
            const requiredEnvVars = [
                'NODE_ENV',
                'SAP_JWT_TRUST_ACL'
            ];
            
            requiredEnvVars.forEach(envVar => {
                if (typeof process !== 'undefined' && process.env && !process.env[envVar]) {
                    errors.push(`Missing required environment variable: ${envVar}`);
                }
            });
            
            // Check production-specific configuration
            if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'production') {
                
                // Production-only required variables
                const prodEnvVars = [
                    'BLOCKCHAIN_RPC_URL',
                    'WS_LOGS_URL',
                    'REDIS_URL'
                ];
                
                prodEnvVars.forEach(envVar => {
                    if (!process.env[envVar]) {
                        warnings.push(`Production environment missing: ${envVar} (will use fallback)`);
                    }
                });
                
                // Validate URLs are not localhost
                if (process.env.BLOCKCHAIN_RPC_URL && process.env.BLOCKCHAIN_RPC_URL.includes('localhost')) {
                    errors.push('BLOCKCHAIN_RPC_URL cannot contain localhost in production');
                }
                
                if (process.env.WS_LOGS_URL && process.env.WS_LOGS_URL.includes('localhost')) {
                    errors.push('WS_LOGS_URL cannot contain localhost in production');
                }
            }
            
            return {
                valid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },
        
        /**
         * Validate security configuration
         */
        validateSecurityConfig: function() {
            const errors = [];
            const warnings = [];
            
            // Check HTTPS enforcement
            if (typeof window !== 'undefined' && window.location.protocol !== 'https:' && 
                window.location.hostname !== 'localhost') {
                errors.push('HTTPS is required for production deployment');
            }
            
            // Check CSP headers
            if (typeof document !== 'undefined') {
                const metaCSP = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
                if (!metaCSP) {
                    warnings.push('Content Security Policy not configured');
                }
            }
            
            // Validate authentication configuration
            if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'production') {
                if (!process.env.SAP_JWT_TRUST_ACL) {
                    errors.push('JWT trust ACL not configured for production');
                }
            }
            
            return {
                valid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },
        
        /**
         * Validate transport configuration
         */
        validateTransportConfig: function() {
            const errors = [];
            const warnings = [];
            
            // Check for transport configuration in package.json
            if (typeof require !== 'undefined') {
                try {
                    const packageJson = require('../../../../package.json');
                    
                    if (!packageJson.sap || !packageJson.sap.transport) {
                        warnings.push('No SAP transport configuration found in package.json');
                    } else {
                        const transport = packageJson.sap.transport;
                        if (!transport.target || !transport.package) {
                            errors.push('Incomplete SAP transport configuration');
                        }
                    }
                } catch (e) {
                    warnings.push('Could not read package.json for transport configuration');
                }
            }
            
            return {
                valid: errors.length === 0,
                errors: errors,
                warnings: warnings
            };
        },
        
        /**
         * Run complete deployment validation
         */
        runFullValidation: function() {
            Log.info('Starting comprehensive deployment validation...');
            
            const results = {
                btpBindings: this.validateBTPBindings(),
                environment: this.validateEnvironmentConfig(),
                security: this.validateSecurityConfig(),
                transport: this.validateTransportConfig()
            };
            
            const allErrors = [
                ...results.btpBindings.errors,
                ...results.environment.errors,
                ...results.security.errors,
                ...results.transport.errors
            ];
            
            const allWarnings = [
                ...(results.environment.warnings || []),
                ...(results.security.warnings || []),
                ...(results.transport.warnings || [])
            ];
            
            const overallValid = allErrors.length === 0;
            
            // Log results
            if (overallValid) {
                Log.info('✅ Deployment validation PASSED');
                if (allWarnings.length > 0) {
                    Log.warning(`⚠️  ${allWarnings.length} warnings found:`, allWarnings);
                }
            } else {
                Log.error(`❌ Deployment validation FAILED with ${allErrors.length} errors:`, allErrors);
                if (allWarnings.length > 0) {
                    Log.warning(`⚠️  ${allWarnings.length} additional warnings:`, allWarnings);
                }
            }
            
            return {
                valid: overallValid,
                errors: allErrors,
                warnings: allWarnings,
                details: results
            };
        },
        
        /**
         * Initialize validation on app startup
         */
        initializeValidation: function() {
            // Run validation when the application starts
            if (typeof window !== 'undefined') {
                window.addEventListener('load', () => {
                    // Delay validation to ensure all resources are loaded
                    setTimeout(() => {
                        const result = this.runFullValidation();
                        
                        // Store validation result globally for debugging
                        window.A2A_DEPLOYMENT_VALIDATION = result;
                        
                        // In development, show validation result in console
                        if (typeof process !== 'undefined' && process.env && 
                            process.env.NODE_ENV !== 'production') {
                            console.group('🔍 A2A Deployment Validation');
                            console.log('Validation Result:', result);
                            console.groupEnd();
                        }
                        
                        // Trigger custom event for validation completion
                        const event = new CustomEvent('a2aValidationComplete', { 
                            detail: result 
                        });
                        window.dispatchEvent(event);
                        
                    }, 1000);
                });
            }
        }
    };
});