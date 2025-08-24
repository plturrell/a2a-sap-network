#!/usr/bin/env node

/**
 * Enterprise SAP Deployment Readiness Check
 * Final validation script for production deployment
 */

const fs = require('fs');
const path = require('path');

console.log('🚀 A2A Network - Enterprise SAP Deployment Readiness Check');
console.log('===========================================================\n');

const checks = [];
let allPassed = true;

// Helper functions
function addCheck(name, passed, details = '') {
    checks.push({ name, passed, details });
    if (!passed) allPassed = false;
    
    const status = passed ? '✅' : '❌';
    console.log(`${status} ${name}${details ? `: ${  details}` : ''}`);
}

function checkFileExists(filePath, description) {
    const exists = fs.existsSync(filePath);
    addCheck(`${description} exists`, exists, filePath);
    return exists;
}

function checkNoLocalhostReferences() {
    const jsFiles = [
        'app/a2aFiori/webapp/controller/settings.controller.js',
        'app/a2aFiori/webapp/controller/ogs.controller.js'
    ];
    
    let hasLocalhost = false;
    
    jsFiles.forEach(file => {
        const fullPath = path.join(__dirname, '..', file);
        if (fs.existsSync(fullPath)) {
            const content = fs.readFileSync(fullPath, 'utf8');
            if (content.includes('localhost')) {
                hasLocalhost = true;
                addCheck(`No localhost in ${file}`, false, 'Found localhost references');
            }
        }
    });
    
    if (!hasLocalhost) {
        addCheck('No localhost references in code', true);
    }
}

// Start validation
console.log('📋 Checking Core Files...\n');

// Check standardized SAP files exist
checkFileExists(path.join(__dirname, '../app/fragment/StandardDialog.fragment.xml'), 'Standard Dialog Fragment');
checkFileExists(path.join(__dirname, '../app/fragment/StandardForm.fragment.xml'), 'Standard Form Fragment');
checkFileExists(path.join(__dirname, '../app/fragment/StandardActionButtons.fragment.xml'), 'Standard Action Buttons');
checkFileExists(path.join(__dirname, '../app/controller/mixin/StandardPatternsMixin.js'), 'Standard Patterns Mixin');

// Check CSS compliance files
checkFileExists(path.join(__dirname, '../app/css/sap-standard-compliance.css'), 'SAP Standard Compliance CSS');
checkFileExists(path.join(__dirname, '../app/css/launchpad-sap-compliance.css'), 'Launchpad SAP Compliance CSS');

// Check enterprise configuration
checkFileExists(path.join(__dirname, '../app/config/deployment-validation.js'), 'Deployment Validation Config');
checkFileExists(path.join(__dirname, '../app/config/transport-config.js'), 'Transport Configuration');
checkFileExists(path.join(__dirname, '../app/services/SecurityService.js'), 'Security Service');

// Check production files
checkFileExists(path.join(__dirname, '../app/launchpad-production.html'), 'Production Launchpad HTML');
checkFileExists(path.join(__dirname, '../app/flp-bootstrap.js'), 'Enterprise FLP Bootstrap');

console.log('\n🔍 Checking Code Quality...\n');

// Check for localhost references
checkNoLocalhostReferences();

// Check package.json has SAP transport config
const packageJsonPath = path.join(__dirname, '../app/package.json');
if (fs.existsSync(packageJsonPath)) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    const hasSapConfig = packageJson.sap && packageJson.sap.transport;
    addCheck('SAP transport configuration in package.json', hasSapConfig);
} else {
    addCheck('Package.json exists', false);
}

// Check manifest.yml for BTP deployment
const manifestPath = path.join(__dirname, '../manifest.yml');
const hasManifest = fs.existsSync(manifestPath);
addCheck('BTP manifest.yml exists', hasManifest);

// Check xs-security.json for XSUAA
const xsSecurityPath = path.join(__dirname, '../xs-security.json');
const hasXsSecurity = fs.existsSync(xsSecurityPath);
addCheck('XSUAA security configuration exists', hasXsSecurity);

console.log('\n🔒 Checking Security Implementation...\n');

// Verify security service exists and has required methods
const securityServicePath = path.join(__dirname, '../app/services/SecurityService.js');
if (fs.existsSync(securityServicePath)) {
    const content = fs.readFileSync(securityServicePath, 'utf8');
    const hasCSRF = content.includes('fetchCSRFToken');
    const hasAuth = content.includes('checkAuthorization');
    const hasSecureAjax = content.includes('secureAjax');
    
    addCheck('CSRF token implementation', hasCSRF);
    addCheck('Authorization checking', hasAuth);
    addCheck('Secure AJAX wrapper', hasSecureAjax);
}

console.log('\n🎨 Checking UI Compliance...\n');

// Check if legacy files were removed
const legacyFiles = [
    'app/css/style.css',
    'app/Component.js',
    'app/manifest.json',
    'app/launchpad.html'
];

let legacyRemoved = true;
legacyFiles.forEach(file => {
    const fullPath = path.join(__dirname, '..', file);
    if (fs.existsSync(fullPath)) {
        legacyRemoved = false;
        addCheck(`Legacy file ${file} removed`, false);
    }
});

if (legacyRemoved) {
    addCheck('All legacy files removed', true);
}

// Check Launchpad controller uses standard patterns
const launchpadControllerPath = path.join(__dirname, '../app/controller/Launchpad.controller.js');
if (fs.existsSync(launchpadControllerPath)) {
    const content = fs.readFileSync(launchpadControllerPath, 'utf8');
    const usesStandardPatterns = content.includes('StandardPatternsMixin');
    const usesSecurityService = content.includes('SecurityService');
    
    addCheck('Launchpad uses standard patterns', usesStandardPatterns);
    addCheck('Launchpad uses security service', usesSecurityService);
}

console.log('\n📊 Final Assessment...\n');

const totalChecks = checks.length;
const passedChecks = checks.filter(c => c.passed).length;
const failedChecks = totalChecks - passedChecks;

console.log(`Total Checks: ${totalChecks}`);
console.log(`Passed: ${passedChecks} ✅`);
console.log(`Failed: ${failedChecks} ❌`);
console.log(`Success Rate: ${Math.round((passedChecks / totalChecks) * 100)}%`);

if (allPassed) {
    console.log('\n🎉 DEPLOYMENT READY! 🎉');
    console.log('The A2A Network platform is ready for enterprise SAP deployment.');
    console.log('\nNext steps:');
    console.log('1. Configure BTP service bindings');
    console.log('2. Set up production environment variables');
    console.log('3. Deploy using: npm run deploy:btp');
    console.log('4. Validate deployment with health checks');
} else {
    console.log('\n⚠️  DEPLOYMENT NOT READY');
    console.log('Please fix the failed checks before deploying to production.');
    console.log('\nFailed checks:');
    checks.filter(c => !c.passed).forEach(check => {
        console.log(`  - ${check.name}${check.details ? `: ${  check.details}` : ''}`);
    });
}

console.log('\n===========================================================');
console.log('Enterprise SAP Deployment Check Complete');

// Exit with appropriate code
process.exit(allPassed ? 0 : 1);