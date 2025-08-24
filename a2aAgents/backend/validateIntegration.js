#!/usr/bin/env node
'use strict';
/**
 * Integration Validation Script
 * Validates that the A2A system works correctly in both local and BTP environments
 */

const _path = require('path');
const fs = require('fs').promises;
const { existsSync } = require('fs');

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console
console.log('🔍 A2A System Integration Validation');
// eslint-disable-next-line no-console
// eslint-disable-next-line no-console
console.log('=====================================');

const validationErrors = [];
const validationWarnings = [];

function validateFile(filePath, description) {
  if (existsSync(filePath)) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`✅ ${description}: ${filePath}`);
    return true;
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
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
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`✅ ${description}: Found require statement`);
      return true;
    } else {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`❌ ${description}: Missing require - ${requireStatement}`);
      validationErrors.push(`Missing require in ${filePath}: ${requireStatement}`);
      return false;
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description}: Error reading file - ${error.message}`);
    validationErrors.push(`Cannot read ${filePath}: ${error.message}`);
    return false;
  }
}

function validateModuleLoad(modulePath, description) {
  try {
    require(modulePath);
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`✅ ${description}: Module loads successfully`);
    return true;
  } catch (error) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description}: Module load failed - ${error.message}`);
    validationErrors.push(`Module load error ${modulePath}: ${error.message}`);
    return false;
  }
}

function validateEnvironmentDetection() {
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('\n🔍 Testing Environment Detection:');
    
  // Test local environment (no VCAP_SERVICES)
  delete process.env.VCAP_SERVICES;
  delete process.env.VCAP_APPLICATION;
    
  try {
    const { btpAdapter } = require('./config/btpAdapter.js');
    if (!btpAdapter.isBTP) {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('✅ Local environment detection: Working');
    } else {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('❌ Local environment detection: Failed - detected BTP when local');
      validationErrors.push('Environment detection failed for local mode');
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
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
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('✅ BTP environment detection: Working');
            
      // Test service configuration
      const dbConfig = testAdapter.getDatabaseConfig();
      if (dbConfig.host === 'test-host') {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('✅ BTP service binding parsing: Working');
      } else {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('❌ BTP service binding parsing: Failed');
        validationErrors.push('BTP service binding parsing failed');
      }
    } else {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('❌ BTP environment detection: Failed - detected local when BTP');
      validationErrors.push('Environment detection failed for BTP mode');
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ BTP environment test failed: ${error.message}`);
    validationErrors.push(`BTP environment test: ${error.message}`);
  }
    
  // Clean up
  delete process.env.VCAP_SERVICES;
  delete process.env.VCAP_APPLICATION;
}

async function validatePackageStructure() {
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('\n🔍 Validating Package Structure:');
    
  const packagePath = './package.json';
  if (validateFile(packagePath, 'Main package.json')) {
    try {
      const pkg = JSON.parse(await fs.readFile(packagePath, 'utf8'));
            
      // Check essential dependencies
      const requiredDeps = ['express'];
      const missingDeps = requiredDeps.filter(dep => !pkg.dependencies[dep]);
            
      if (missingDeps.length === 0) {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('✅ Essential dependencies: Present');
      } else {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`❌ Missing dependencies: ${missingDeps.join(', ')}`);
        validationErrors.push(`Missing dependencies: ${missingDeps.join(', ')}`);
      }
            
      // Check main entry point
      if (existsSync(pkg.main)) {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`✅ Main entry point: ${pkg.main} exists`);
      } else {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log(`❌ Main entry point: ${pkg.main} missing`);
        validationErrors.push(`Main entry point missing: ${pkg.main}`);
      }
            
    } catch (error) {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`❌ Package.json parsing failed: ${error.message}`);
      validationErrors.push(`Package.json parsing: ${error.message}`);
    }
  }
}

// Wrap in async function for top-level await
(async () => {
  // Run validations
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('\n🔍 Validating Core Files:');
  validateFile('./config/btpAdapter.js', 'BTP Adapter');
  validateFile('./config/minimalBtpConfig.js', 'Minimal BTP Config (JS)');
  validateFile('./launchA2aSystem.js', 'Main System Launcher');
  validateFile('./srv/server-minimal.js', 'Minimal Server');
  validateFile('./mta-minimal.yaml', 'MTA Deployment Descriptor');
  validateFile('./start.sh', 'Universal Start Script');
  validateFile('./deploy-local.sh', 'Local Deployment Script');
  validateFile('./deploy-btp.sh', 'BTP Deployment Script');

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🔍 Validating Module Dependencies:');
  await validateRequire('./launchA2aSystem.js', "require('./config/btpAdapter')", 'Main launcher imports BTP adapter');
  await validateRequire('./srv/server-minimal.js', "require('../config/minimalBtpConfig.js')", 'Minimal server imports config');

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🔍 Validating Module Loading:');
  validateModuleLoad('./config/btpAdapter.js', 'BTP Adapter module');
  validateModuleLoad('./config/minimalBtpConfig.js', 'Minimal BTP Config module');

  // Test environment detection
  validateEnvironmentDetection();

  // Validate package structure
  await validatePackageStructure();

  // Summary
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('\n📋 Validation Summary:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('======================');

  if (validationErrors.length === 0) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('✅ ALL VALIDATIONS PASSED');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('🚀 System is ready for both local and BTP deployment');
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ VALIDATION FAILED - ${validationErrors.length} errors found:`);
    validationErrors.forEach((error, index) => {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`   ${index + 1}. ${error}`);
    });
  }

  if (validationWarnings.length > 0) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`⚠️ Warnings (${validationWarnings.length}):`);
    validationWarnings.forEach((warning, index) => {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`   ${index + 1}. ${warning}`);
    });
  }

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🔗 Quick Test Commands:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('   Local: ./start.sh');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('   Minimal: ./start.sh minimal');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('   Health: curl http://localhost:8080/health');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('   Agents: curl http://localhost:8080/api/agents');

  process.exit(validationErrors.length > 0 ? 1 : 0);
})().catch(error => {
  console.error('Script execution error:', error);
  process.exit(1);
});