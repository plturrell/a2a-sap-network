#!/usr/bin/env node
/**
 * Integration Validation Script
 * Validates that the A2A system works correctly in both local and BTP environments
 */

const path = require('path');
const fs = require('fs').promises;
const { existsSync } = require('fs');

console.log('🔍 A2A System Integration Validation');
console.log('=====================================');

let validationErrors = [];
let validationWarnings = [];

function validateFile(filePath, description) {
    if (existsSync(filePath)) {
        console.log(`✅ ${description}: ${filePath}`);
        return true;
    } else {
        console.log(`❌ ${description}: MISSING - ${filePath}`);
        validationErrors.push(`Missing file: ${filePath}`);
        return false;
    }
}

async function validateRequire(filePath, requireStatement, description) {
    if (!existsSync(filePath)) {
        return false;
    }
    
    try {
        const content = await fs.readFile(filePath, 'utf8');
        if (content.includes(requireStatement)) {
            console.log(`✅ ${description}: Found require statement`);
            return true;
        } else {
            console.log(`❌ ${description}: Missing require - ${requireStatement}`);
            validationErrors.push(`Missing require in ${filePath}: ${requireStatement}`);
            return false;
        }
    } catch (error) {
        console.log(`❌ ${description}: Error reading file - ${error.message}`);
        validationErrors.push(`Cannot read ${filePath}: ${error.message}`);
        return false;
    }
}

function validateModuleLoad(modulePath, description) {
    try {
        require(modulePath);
        console.log(`✅ ${description}: Module loads successfully`);
        return true;
    } catch (error) {
        console.log(`❌ ${description}: Module load failed - ${error.message}`);
        validationErrors.push(`Module load error ${modulePath}: ${error.message}`);
        return false;
    }
}

function validateEnvironmentDetection() {
    console.log('\n🔍 Testing Environment Detection:');
    
    // Test local environment (no VCAP_SERVICES)
    delete process.env.VCAP_SERVICES;
    delete process.env.VCAP_APPLICATION;
    
    try {
        const { btpAdapter } = require('./config/btpAdapter.js');
        if (!btpAdapter.isBTP) {
            console.log('✅ Local environment detection: Working');
        } else {
            console.log('❌ Local environment detection: Failed - detected BTP when local');
            validationErrors.push('Environment detection failed for local mode');
        }
    } catch (error) {
        console.log(`❌ Local environment test failed: ${error.message}`);
        validationErrors.push(`Local environment test: ${error.message}`);
    }
    
    // Test BTP environment simulation
    process.env.VCAP_SERVICES = JSON.stringify({
        hana: [{
            credentials: {
                host: 'test-host',
                port: 443,
                user: 'test-user',
                password: 'test-password'
            }
        }],
        xsuaa: [{
            credentials: {
                url: 'https://test.authentication.sap.hana.ondemand.com',
                clientid: 'test-client',
                clientsecret: 'test-secret'
            }
        }]
    });
    
    try {
        // Clear require cache to get fresh instance
        delete require.cache[require.resolve('./config/btpAdapter.js')];
        const { BTPAdapter } = require('./config/btpAdapter.js');
        const testAdapter = new BTPAdapter();
        
        if (testAdapter.isBTP) {
            console.log('✅ BTP environment detection: Working');
            
            // Test service configuration
            const dbConfig = testAdapter.getDatabaseConfig();
            if (dbConfig.host === 'test-host') {
                console.log('✅ BTP service binding parsing: Working');
            } else {
                console.log('❌ BTP service binding parsing: Failed');
                validationErrors.push('BTP service binding parsing failed');
            }
        } else {
            console.log('❌ BTP environment detection: Failed - detected local when BTP');
            validationErrors.push('Environment detection failed for BTP mode');
        }
    } catch (error) {
        console.log(`❌ BTP environment test failed: ${error.message}`);
        validationErrors.push(`BTP environment test: ${error.message}`);
    }
    
    // Clean up
    delete process.env.VCAP_SERVICES;
    delete process.env.VCAP_APPLICATION;
}

async function validatePackageStructure() {
    console.log('\n🔍 Validating Package Structure:');
    
    const packagePath = './package.json';
    if (validateFile(packagePath, 'Main package.json')) {
        try {
            const pkg = JSON.parse(await fs.readFile(packagePath, 'utf8'));
            
            // Check essential dependencies
            const requiredDeps = ['express'];
            const missingDeps = requiredDeps.filter(dep => !pkg.dependencies[dep]);
            
            if (missingDeps.length === 0) {
                console.log('✅ Essential dependencies: Present');
            } else {
                console.log(`❌ Missing dependencies: ${missingDeps.join(', ')}`);
                validationErrors.push(`Missing dependencies: ${missingDeps.join(', ')}`);
            }
            
            // Check main entry point
            if (existsSync(pkg.main)) {
                console.log(`✅ Main entry point: ${pkg.main} exists`);
            } else {
                console.log(`❌ Main entry point: ${pkg.main} missing`);
                validationErrors.push(`Main entry point missing: ${pkg.main}`);
            }
            
        } catch (error) {
            console.log(`❌ Package.json parsing failed: ${error.message}`);
            validationErrors.push(`Package.json parsing: ${error.message}`);
        }
    }
}

// Wrap in async function for top-level await
(async () => {
    // Run validations
    console.log('\n🔍 Validating Core Files:');
    validateFile('./config/btpAdapter.js', 'BTP Adapter');
    validateFile('./config/minimalBtpConfig.js', 'Minimal BTP Config (JS)');
    validateFile('./launchA2aSystem.js', 'Main System Launcher');
    validateFile('./srv/server-minimal.js', 'Minimal Server');
    validateFile('./mta-minimal.yaml', 'MTA Deployment Descriptor');
    validateFile('./start.sh', 'Universal Start Script');
    validateFile('./deploy-local.sh', 'Local Deployment Script');
    validateFile('./deploy-btp.sh', 'BTP Deployment Script');

    console.log('\n🔍 Validating Module Dependencies:');
    await validateRequire('./launchA2aSystem.js', "require('./config/btpAdapter')", 'Main launcher imports BTP adapter');
    await validateRequire('./srv/server-minimal.js', "require('../config/minimalBtpConfig.js')", 'Minimal server imports config');

    console.log('\n🔍 Validating Module Loading:');
    validateModuleLoad('./config/btpAdapter.js', 'BTP Adapter module');
    validateModuleLoad('./config/minimalBtpConfig.js', 'Minimal BTP Config module');

    // Test environment detection
    validateEnvironmentDetection();

    // Validate package structure
    await validatePackageStructure();

    // Summary
    console.log('\n📋 Validation Summary:');
    console.log('======================');

    if (validationErrors.length === 0) {
        console.log('✅ ALL VALIDATIONS PASSED');
        console.log('🚀 System is ready for both local and BTP deployment');
    } else {
        console.log(`❌ VALIDATION FAILED - ${validationErrors.length} errors found:`);
        validationErrors.forEach((error, index) => {
            console.log(`   ${index + 1}. ${error}`);
        });
    }

    if (validationWarnings.length > 0) {
        console.log(`⚠️ Warnings (${validationWarnings.length}):`);
        validationWarnings.forEach((warning, index) => {
            console.log(`   ${index + 1}. ${warning}`);
        });
    }

    console.log('\n🔗 Quick Test Commands:');
    console.log('   Local: ./start.sh');
    console.log('   Minimal: ./start.sh minimal');
    console.log('   Health: curl http://localhost:8080/health');
    console.log('   Agents: curl http://localhost:8080/api/agents');

    process.exit(validationErrors.length > 0 ? 1 : 0);
})().catch(error => {
    console.error('Script execution error:', error);
    process.exit(1);
});