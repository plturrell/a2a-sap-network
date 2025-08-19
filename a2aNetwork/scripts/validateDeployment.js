#!/usr/bin/env node

/**
 * Production Startup Validation for A2A Network
 * Ensures all production readiness fixes are properly implemented
 */

const ConfigValidator = require('./validateConfig');
const { validateAllKeys } = require('../srv/lib/secureKeyManager');

async function validateProductionReadiness() {
    console.log('🚀 A2A Network Production Readiness Validation\n');
    console.log('=' + '='.repeat(50));
    
    let allValid = true;
    
    // 1. Configuration validation
    console.log('\n📋 Step 1: Configuration Validation');
    const configValidator = new ConfigValidator();
    const configValid = configValidator.validateAll();
    
    if (!configValid) {
        allValid = false;
    }
    
    // 2. Key management validation
    console.log('\n🔐 Step 2: Key Management Validation');
    try {
        const keyValidation = await validateAllKeys();
        
        if (keyValidation.errors.length > 0) {
            console.log('❌ Key validation errors:');
            keyValidation.errors.forEach(error => console.log(`  • ${error}`));
            allValid = false;
        } else {
            console.log('✅ Key management validation passed');
        }
        
        if (keyValidation.warnings.length > 0) {
            console.log('⚠️  Key validation warnings:');
            keyValidation.warnings.forEach(warning => console.log(`  • ${warning}`));
        }
    } catch (error) {
        console.log(`❌ Key validation failed: ${error.message}`);
        allValid = false;
    }
    
    // 3. Environment checks
    console.log('\n🌍 Step 3: Environment Validation');
    const environment = process.env.NODE_ENV;
    
    if (environment === 'production') {
        console.log('✅ Running in production mode');
        
        // Check critical environment variables
        const criticalVars = [
            'BLOCKCHAIN_RPC_URL',
            'CHAIN_ID',
            'DEFAULT_PRIVATE_KEY'
        ];
        
        const missing = criticalVars.filter(v => !process.env[v]);
        if (missing.length > 0) {
            console.log(`❌ Missing critical environment variables: ${missing.join(', ')}`);
            allValid = false;
        }
        
        // Check for localhost URLs in production
        if (process.env.BLOCKCHAIN_RPC_URL && process.env.BLOCKCHAIN_RPC_URL.includes('localhost')) {
            console.log('❌ Cannot use localhost blockchain URL in production');
            allValid = false;
        }
        
    } else {
        console.log(`⚠️  Running in ${environment} mode (not production)`);
    }
    
    // 4. Security checks
    console.log('\n🛡️  Step 4: Security Validation');
    
    // Check for authentication bypasses in production
    if (environment === 'production') {
        const authMiddlewarePath = require('path').join(__dirname, '..', 'srv', 'middleware', 'auth.js');
        try {
            const authCode = require('fs').readFileSync(authMiddlewarePath, 'utf8');
            if (authCode.includes('tileApiEndpoints') && authCode.includes('return next()')) {
                console.log('❌ Authentication bypass found in auth middleware');
                allValid = false;
            }
        } catch (error) {
            console.log('⚠️  Could not verify auth middleware');
        }
    }
    
    // Check for template values in environment
    const templateValues = [
        'YOUR_PRIVATE_KEY_HERE',
        'AGENT_MANAGER_KEY',
        '0x0000000000000000000000000000000000000000',
        'template_value',
        'placeholder'
    ];
    
    let templatesFound = false;
    Object.entries(process.env).forEach(([key, value]) => {
        if (value && templateValues.some(template => value.includes(template))) {
            console.log(`❌ Template value found in ${key}`);
            allValid = false;
            templatesFound = true;
        }
    });
    
    if (!templatesFound) {
        console.log('✅ No template values found in environment');
    }
    
    // Check for zero address usage
    if (environment === 'production') {
        const web3ClientPath = require('path').join(__dirname, '..', 'pythonSdk', 'blockchain', 'web3Client.py');
        try {
            const web3Code = require('fs').readFileSync(web3ClientPath, 'utf8');
            if (web3Code.includes('0x0000000000000000000000000000000000000000')) {
                console.log('❌ Zero address fallback found in blockchain client');
                allValid = false;
            }
        } catch (error) {
            console.log('⚠️  Could not verify blockchain client');
        }
    }
    
    // Final result
    console.log('\n' + '='.repeat(51));
    if (allValid) {
        console.log('🎉 PRODUCTION READINESS: PASSED');
        console.log('   A2A Network is ready for production deployment!');
        process.exit(0);
    } else {
        console.log('❌ PRODUCTION READINESS: FAILED');
        console.log('   Fix the issues above before deploying to production');
        process.exit(1);
    }
}

// Run validation
if (require.main === module) {
    validateProductionReadiness().catch(error => {
        console.error('❌ Production validation failed:', error.message);
        process.exit(1);
    });
}

module.exports = { validateProductionReadiness };