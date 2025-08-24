#!/usr/bin/env node
'use strict';
/**
 * BTP Deployment Readiness Validator
 * Ensures 100% readiness for SAP BTP deployment
 */

const fs = require('fs').promises;
const { existsSync } = require('fs');
const _path = require('path');

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console
console.log('🎯 SAP BTP Deployment Readiness Check');
// eslint-disable-next-line no-console
// eslint-disable-next-line no-console
console.log('=====================================\n');

let readinessScore = 0;
let maxScore = 0;
const issues = [];

function checkFile(filePath, description, critical = true) {
  maxScore += critical ? 2 : 1;
  if (existsSync(filePath)) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`✅ ${description}`);
    readinessScore += critical ? 2 : 1;
    return true;
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description} - MISSING`);
    issues.push(`Missing ${critical ? 'CRITICAL' : 'optional'} file: ${filePath}`);
    return false;
  }
}

async function checkFileContent(filePath, searchPattern, description, critical = true) {
  maxScore += critical ? 2 : 1;
  if (!existsSync(filePath)) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description} - FILE MISSING`);
    issues.push(`Missing file for content check: ${filePath}`);
    return false;
  }
    
  const content = await fs.readFile(filePath, 'utf8');
  if (content.includes(searchPattern)) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`✅ ${description}`);
    readinessScore += critical ? 2 : 1;
    return true;
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description} - PATTERN NOT FOUND`);
    issues.push(`Missing pattern "${searchPattern}" in ${filePath}`);
    return false;
  }
}

async function checkJSON(filePath, keyPath, description) {
  maxScore += 2;
  try {
    const content = JSON.parse(await fs.readFile(filePath, 'utf8'));
    const keys = keyPath.split('.');
    let value = content;
        
    for (const key of keys) {
      value = value[key];
      if (value === undefined) {
        break;
      }
    }
        
    if (value !== undefined) {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`✅ ${description}: ${JSON.stringify(value)}`);
      readinessScore += 2;
      return true;
    } else {
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log(`❌ ${description} - KEY NOT FOUND`);
      issues.push(`Missing JSON key ${keyPath} in ${filePath}`);
      return false;
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log(`❌ ${description} - JSON PARSE ERROR`);
    issues.push(`JSON parse error in ${filePath}: ${error.message}`);
    return false;
  }
}

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console
console.log('📋 Critical Files Check:');
// eslint-disable-next-line no-console
// eslint-disable-next-line no-console
console.log('------------------------');
checkFile('package.json', 'Main package.json');
checkFile('package-lock.json', 'Package lock file', false);
checkFile('mta-minimal.yaml', 'MTA deployment descriptor');
checkFile('manifest.yml', 'Cloud Foundry manifest (alternative)');
checkFile('.cfignore', 'CF ignore file');
checkFile('.env.example', 'Environment template');
checkFile('deploy-btp.sh', 'BTP deployment script');

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console
console.log('\n📁 Required Directories:');
// eslint-disable-next-line no-console
// eslint-disable-next-line no-console
console.log('------------------------');
checkFile('db', 'Database directory');
checkFile('db/src', 'Database source directory');
checkFile('db/src/.hdiconfig', 'HDI configuration');
checkFile('config', 'Configuration directory');
checkFile('srv', 'Service directory');

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console

// eslint-disable-next-line no-console
console.log('\n🔧 Core System Files:');
// eslint-disable-next-line no-console
// eslint-disable-next-line no-console
console.log('---------------------');
checkFile('launchA2aSystem.js', 'Main entry point');
checkFile('config/btpAdapter.js', 'BTP adapter');
checkFile('config/minimalBtpConfig.js', 'Minimal BTP config');
checkFile('srv/server-minimal.js', 'Minimal server');

// Wrap in async function for top-level await
(async () => {
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('\n📦 Package Configuration:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('-------------------------');
  await checkJSON('package.json', 'main', 'Entry point defined');
  await checkJSON('package.json', 'engines.node', 'Node.js version specified');
  await checkJSON('package.json', 'scripts.start', 'Start script defined');
  await checkJSON('package.json', 'dependencies.express', 'Express dependency');

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🚀 MTA Configuration:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('---------------------');
  await checkFileContent('mta-minimal.yaml', 'a2a-agents-srv', 'Application module defined');
  await checkFileContent('mta-minimal.yaml', 'a2a-agents-hana', 'HANA service defined');
  await checkFileContent('mta-minimal.yaml', 'a2a-agents-xsuaa', 'XSUAA service defined');
  await checkFileContent('mta-minimal.yaml', 'buildpack: nodejs_buildpack', 'Node.js buildpack specified');

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🔌 Service Integration:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('-----------------------');
  await checkFileContent('config/btpAdapter.js', 'VCAP_SERVICES', 'VCAP_SERVICES handling');
  await checkFileContent('config/btpAdapter.js', 'process.env.VCAP_APPLICATION', 'VCAP_APPLICATION handling');
  await checkFileContent('config/btpAdapter.js', 'injectEnvironmentVariables', 'Environment variable injection');

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n🛡️ Deployment Scripts:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('----------------------');
  await checkFileContent('deploy-btp.sh', 'cf login', 'CF login check');
  await checkFileContent('deploy-btp.sh', 'mbt build', 'MTA build command');
  await checkFileContent('deploy-btp.sh', 'cf deploy', 'CF deploy command');

  // Calculate readiness percentage
  const readinessPercentage = Math.round((readinessScore / maxScore) * 100);

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n📊 Readiness Summary:');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('====================');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log(`Score: ${readinessScore}/${maxScore} (${readinessPercentage}%)`);

  if (readinessPercentage === 100) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('\n✅ 🎉 100% READY FOR SAP BTP DEPLOYMENT! 🎉');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('\nNext steps for SAP Engineer:');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('1. Run: cf login -a https://api.cf.<region>.hana.ondemand.com');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('2. Run: ./deploy-btp.sh');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('3. Monitor: cf logs a2a-agents-srv');
  } else if (readinessPercentage >= 90) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('\n⚠️ ALMOST READY - Minor issues to fix:');
    // eslint-disable-next-line no-console
    issues.forEach(issue => console.log(`   - ${issue}`));
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('\n❌ NOT READY - Critical issues found:');
    // eslint-disable-next-line no-console
    issues.forEach(issue => console.log(`   - ${issue}`));
  }

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console

  // eslint-disable-next-line no-console
  console.log('\n📚 Deployment Guide: README-SAP-BTP-DEPLOYMENT.md');

  process.exit(readinessPercentage === 100 ? 0 : 1);
})().catch(error => {
  console.error('Script execution error:', error);
  process.exit(1);
});