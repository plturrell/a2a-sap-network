#!/usr/bin/env node

/**
 * Configuration Validation Script for A2A Network
 * Validates environment configuration before deployment
 */

const fs = require('fs');
const path = require('path');

class ConfigValidator {
    constructor() {
        this.errors = [];
        this.warnings = [];
        this.isProduction = process.env.NODE_ENV === 'production';
    }

    /**
     * Validate all configuration
     */
    validateAll() {
        console.log('🔍 Validating A2A Network Configuration...\n');
        
        this.validateAuthenticationConfig();
        this.validateBlockchainConfig();
        this.validateDatabaseConfig();
        this.validateSecurityConfig();
        this.validateTemplateValues();
        
        this.reportResults();
        
        return this.errors.length === 0;
    }

    /**
     * Validate authentication configuration
     */
    validateAuthenticationConfig() {
        console.log('🔐 Validating Authentication Configuration...');
        
        const btpEnvironment = process.env.BTP_ENVIRONMENT === 'true';
        const allowNonBtpAuth = process.env.ALLOW_NON_BTP_AUTH === 'true';
        
        if (this.isProduction) {
            if (!btpEnvironment && !allowNonBtpAuth) {
                this.errors.push('CRITICAL: Production requires either BTP_ENVIRONMENT=true or ALLOW_NON_BTP_AUTH=true');
            }
            
            if (allowNonBtpAuth && this.isProduction) {
                this.warnings.push('WARNING: Using non-BTP authentication in production');
                
                if (!process.env.JWT_SECRET) {
                    this.errors.push('CRITICAL: JWT_SECRET required when ALLOW_NON_BTP_AUTH=true in production');
                } else if (process.env.JWT_SECRET.length < 32) {
                    this.errors.push('CRITICAL: JWT_SECRET must be at least 32 characters long');
                }
            }
            
            if (!process.env.SESSION_SECRET) {
                this.errors.push('CRITICAL: SESSION_SECRET required in production');
            }
        }
        
        if (btpEnvironment) {
            const xsuaaFields = ['XSUAA_CLIENTID', 'XSUAA_CLIENTSECRET', 'XSUAA_URL'];
            const missing = xsuaaFields.filter(field => !process.env[field]);
            if (missing.length > 0) {
                this.errors.push(`CRITICAL: Missing XSUAA fields: ${missing.join(', ')}`);
            }
        }
        
        // Check for deprecated environment variables
        if (process.env.USE_DEVELOPMENT_AUTH === 'true') {
            this.errors.push('CRITICAL: USE_DEVELOPMENT_AUTH is deprecated and not allowed');
        }
        
        if (process.env.ENABLE_XSUAA_VALIDATION === 'false') {
            this.errors.push('CRITICAL: ENABLE_XSUAA_VALIDATION cannot be disabled');
        }
        
        console.log('  ✅ Authentication validation complete\n');
    }

    /**
     * Validate blockchain configuration
     */
    validateBlockchainConfig() {
        console.log('⛓️  Validating Blockchain Configuration...');
        
        const required = ['BLOCKCHAIN_RPC_URL', 'CHAIN_ID'];
        const missing = required.filter(field => !process.env[field]);
        
        if (missing.length > 0) {
            this.errors.push(`CRITICAL: Missing blockchain configuration: ${missing.join(', ')}`);
        }
        
        // Check for localhost fallbacks
        if (process.env.BLOCKCHAIN_RPC_URL && process.env.BLOCKCHAIN_RPC_URL.includes('localhost')) {
            if (this.isProduction) {
                this.errors.push('CRITICAL: Cannot use localhost blockchain URL in production');
            } else {
                this.warnings.push('WARNING: Using localhost blockchain URL');
            }
        }
        
        // Validate chain ID
        if (process.env.CHAIN_ID) {
            const chainId = parseInt(process.env.CHAIN_ID);
            if (isNaN(chainId)) {
                this.errors.push('CRITICAL: CHAIN_ID must be a valid number');
            }
        }
        
        // Check for zero address fallbacks
        if (process.env.BLOCKCHAIN_CONTRACT_ADDRESS === '0x0000000000000000000000000000000000000000') {
            this.errors.push('CRITICAL: Cannot use zero address for blockchain contract');
        }
        
        console.log('  ✅ Blockchain validation complete\n');
    }

    /**
     * Validate database configuration  
     */
    validateDatabaseConfig() {
        console.log('🗄️  Validating Database Configuration...');
        
        if (this.isProduction) {
            const required = ['HANA_HOST', 'HANA_DATABASE', 'HANA_USER', 'HANA_PASSWORD'];
            const missing = required.filter(field => !process.env[field]);
            
            if (missing.length > 0) {
                this.errors.push(`CRITICAL: Missing database configuration: ${missing.join(', ')}`);
            }
            
            // Security checks
            if (process.env.HANA_ENCRYPT !== 'true') {
                this.errors.push('CRITICAL: HANA_ENCRYPT must be true in production');
            }
            
            if (process.env.HANA_SSL_VALIDATE_CERTIFICATE !== 'true') {
                this.warnings.push('WARNING: HANA_SSL_VALIDATE_CERTIFICATE should be true in production');
            }
        }
        
        console.log('  ✅ Database validation complete\n');
    }

    /**
     * Validate security configuration
     */
    validateSecurityConfig() {
        console.log('🛡️  Validating Security Configuration...');
        
        if (this.isProduction) {
            if (!process.env.REQUEST_SIGNING_SECRET) {
                this.errors.push('CRITICAL: REQUEST_SIGNING_SECRET required in production');
            }
            
            if (!process.env.CORS_ALLOWED_ORIGINS) {
                this.warnings.push('WARNING: CORS_ALLOWED_ORIGINS not set, will default to restricted');
            } else if (process.env.CORS_ALLOWED_ORIGINS.includes('*')) {
                this.errors.push('CRITICAL: Cannot use wildcard CORS origins in production');
            }
        }
        
        console.log('  ✅ Security validation complete\n');
    }

    /**
     * Check for template values that shouldn't be in production
     */
    validateTemplateValues() {
        console.log('📝 Checking for Template Values...');
        
        const templateValues = [
            'YOUR_PRIVATE_KEY_HERE',
            'AGENT_MANAGER_KEY',
            'YOUR_SECRET_HERE',
            'localhost:8545',
            'template_value',
            'placeholder'
        ];
        
        // Check all environment variables for template values
        Object.entries(process.env).forEach(([key, value]) => {
            if (value && templateValues.some(template => value.includes(template))) {
                if (this.isProduction) {
                    this.errors.push(`CRITICAL: Template value found in ${key}: ${value}`);
                } else {
                    this.warnings.push(`WARNING: Template value found in ${key}: ${value}`);
                }
            }
        });
        
        console.log('  ✅ Template validation complete\n');
    }

    /**
     * Report validation results
     */
    reportResults() {
        console.log('📊 Configuration Validation Results:');
        console.log('=' + '='.repeat(50));
        
        if (this.errors.length === 0 && this.warnings.length === 0) {
            console.log('✅ All configuration checks passed!\n');
            return;
        }
        
        if (this.errors.length > 0) {
            console.log('\n❌ ERRORS (Must be fixed):');
            this.errors.forEach(error => console.log(`  • ${error}`));
        }
        
        if (this.warnings.length > 0) {
            console.log('\n⚠️  WARNINGS (Should be reviewed):');
            this.warnings.forEach(warning => console.log(`  • ${warning}`));
        }
        
        console.log('\n' + '='.repeat(51));
        
        if (this.errors.length > 0) {
            console.log('❌ Configuration validation FAILED');
            console.log(`   ${this.errors.length} error(s) must be fixed before deployment\n`);
        } else {
            console.log('✅ Configuration validation PASSED');
            console.log(`   ${this.warnings.length} warning(s) should be reviewed\n`);
        }
    }

    /**
     * Generate secure configuration values
     */
    static generateSecrets() {
        const crypto = require('crypto');
        
        console.log('🔐 Generated Secure Configuration Values:');
        console.log('=' + '='.repeat(40));
        console.log(`JWT_SECRET=${crypto.randomBytes(32).toString('base64')}`);
        console.log(`SESSION_SECRET=${crypto.randomBytes(32).toString('base64')}`);
        console.log(`REQUEST_SIGNING_SECRET=${crypto.randomBytes(32).toString('base64')}`);
        console.log('=' + '='.repeat(41));
        console.log('⚠️  Store these securely and never commit to version control!\n');
    }
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.includes('--generate-secrets')) {
        ConfigValidator.generateSecrets();
        process.exit(0);
    }
    
    const validator = new ConfigValidator();
    const isValid = validator.validateAll();
    
    if (args.includes('--strict') && !isValid) {
        process.exit(1);
    }
    
    console.log('💡 Tip: Run with --generate-secrets to generate secure configuration values');
    console.log('💡 Tip: Run with --strict to exit with error code if validation fails\n');
}

module.exports = ConfigValidator;