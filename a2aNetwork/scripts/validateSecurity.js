#!/usr/bin/env node
/**
 * Security Validation Script for A2A Network
 * Validates that all security requirements are met before deployment
 */

const fs = require('fs');
const path = require('path');

class SecurityValidator {
    constructor() {
        this.errors = [];
        this.warnings = [];
        this.projectRoot = path.join(__dirname, '..');
    }

    validate() {
(async () => {
        console.log('🔒 Starting Security Validation for A2A Network...\n');

        this.validateEnvironmentVariables();
        this.validateNoHardcodedCredentials();
        this.validateNoConsoleStatements();
        this.validateSSLConfiguration();
        this.validateProductionReadiness();

        this.printResults();
        return this.errors.length === 0;
    }

    validateEnvironmentVariables() {
        console.log('📋 Checking required environment variables...');

        const requiredVars = [
            'HANA_HOST', 'HANA_USER', 'HANA_PASSWORD',
            'REQUEST_SIGNING_SECRET', 'A2A_JWT_SECRET',
            'BLOCKCHAIN_RPC_URL', 'BLOCKCHAIN_CONTRACT_ADDRESS',
            'CORS_ALLOWED_ORIGINS'
        ];

        const requiredAgentVars = [];
        for (let i = 0; i <= 5; i++) {
            requiredAgentVars.push(`AGENT${i}_ADDRESS`, `AGENT${i}_ENDPOINT`);
        }
        requiredAgentVars.push(
            'AGENT_MANAGER_ADDRESS', 'AGENT_MANAGER_ENDPOINT',
            'DATA_MANAGER_ADDRESS', 'DATA_MANAGER_ENDPOINT',
            'CATALOG_MANAGER_ADDRESS', 'CATALOG_MANAGER_ENDPOINT',
            'AGENT_BUILDER_ADDRESS', 'AGENT_BUILDER_ENDPOINT'
        );

        [...requiredVars, ...requiredAgentVars].forEach(varName => {
            if (!process.env[varName]) {
                this.errors.push(`Missing required environment variable: ${varName}`);
            }
        });
    }

    validateNoHardcodedCredentials() {
        console.log('🔍 Scanning for hardcoded credentials...');

        const patterns = [
            /password.*[:=].*['"]\w+['"]/i,
            /secret.*[:=].*['"]\w+['"]/i,
            /Initial@1/g,
            /default-secret/g,
            /DBADMIN/g,
            /0xAA0000000000000000000000000000000000000[0-9A-F]/g
        ];

        this.scanFiles(['srv', 'app', 'scripts'], patterns, 'hardcoded credentials');
    }

    validateNoConsoleStatements() {
        console.log('🔇 Checking for console statements...');

        const patterns = [/console\.(log|error|warn|info)/g];
        this.scanFiles(['srv', 'app'], patterns, 'console statements', 'warning');
    }

    validateSSLConfiguration() {
        console.log('🔐 Validating SSL configuration...');

        const patterns = [/sslValidateCertificate.*[:=].*false/g];
        this.scanFiles(['srv', 'scripts'], patterns, 'disabled SSL validation');
    }

    validateProductionReadiness() {
        console.log('🏭 Checking production readiness...');

        // Check for .env file in repository
        if (fs.existsSync(path.join(this.projectRoot, '.env'))) {
            this.errors.push('.env file found in repository - this should not be committed');
        }

        // Check for default_env.json
        if (fs.existsSync(path.join(this.projectRoot, 'default_env.json'))) {
            this.errors.push('default_env.json file found - this contains credentials and should not be committed');
        }

        // Check NODE_ENV
        if (process.env.NODE_ENV !== 'production' && process.env.NODE_ENV !== 'staging') {
            this.warnings.push('NODE_ENV is not set to production or staging');
        }
    }

    scanFiles(directories, patterns, issueType, severity = 'error') {
        directories.forEach(dir => {
            const dirPath = path.join(this.projectRoot, dir);
            if (fs.existsSync(dirPath)) {
                this.scanDirectory(dirPath, patterns, issueType, severity);
            }
        });
    }

    scanDirectory(dirPath, patterns, issueType, severity) {
        const files = fs.readdirSync(dirPath, { withFileTypes: true });

        files.forEach(file => {
            const filePath = path.join(dirPath, file.name);

            if (file.isDirectory() && !file.name.includes('node_modules')) {
                this.scanDirectory(filePath, patterns, issueType, severity);
            } else if (file.isFile() && (file.name.endsWith('.js') || file.name.endsWith('.json'))) {
                this.scanFile(filePath, patterns, issueType, severity);
            }
        });
    }

    scanFile(filePath, patterns, issueType, severity) {
        try {
            const content = await fs.readFile(filePath, 'utf8');
            const lines = content.split('\n');

            patterns.forEach(pattern => {
                lines.forEach((line, index) => {
                    if (pattern.test(line)) {
                        const message = `Found ${issueType} in ${filePath}:${index + 1}`;
                        if (severity === 'error') {
                            this.errors.push(message);
                        } else {
                            this.warnings.push(message);
                        }
                    }
                });
            });
        } catch (error) {
            this.warnings.push(`Could not scan file ${filePath}: ${error.message}`);
        }
    }

    printResults() {
        console.log('\n' + '='.repeat(60));
        console.log('🔒 SECURITY VALIDATION RESULTS');
        console.log('='.repeat(60));

        if (this.errors.length === 0) {
            console.log('✅ NO CRITICAL SECURITY ISSUES FOUND');
        } else {
            console.log(`❌ ${this.errors.length} CRITICAL SECURITY ISSUES FOUND:`);
            this.errors.forEach(error => console.log(`   ❌ ${error}`));
        }

        if (this.warnings.length > 0) {
            console.log(`\n⚠️  ${this.warnings.length} WARNINGS:`);
            this.warnings.forEach(warning => console.log(`   ⚠️  ${warning}`));
        }

        console.log('\n' + '='.repeat(60));
        
        if (this.errors.length === 0) {
            console.log('🎉 SECURITY VALIDATION PASSED - READY FOR DEPLOYMENT');
        } else {
            console.log('🚨 SECURITY VALIDATION FAILED - FIX ISSUES BEFORE DEPLOYMENT');
        }
        
        console.log('='.repeat(60));
    }
}

// Run validation if script is called directly
if (require.main === module) {
    const validator = new SecurityValidator();
    const passed = validator.validate();
    process.exit(passed ? 0 : 1);
}

module.exports = SecurityValidator;
})().catch(console.error);