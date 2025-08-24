#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('SAP UI5 Standards Compliance Check\n');

const checks = {
    'manifest.json': {
        path: 'app/a2a-fiori/webapp/manifest.json',
        checks: [
            {
                name: 'SAP App ID Format',
                test: (content) => {
                    const manifest = JSON.parse(content);
                    return manifest['sap.app'].id && manifest['sap.app'].id.includes('.');
                }
            },
            {
                name: 'SAP UI5 Version',
                test: (content) => {
                    const manifest = JSON.parse(content);
                    return manifest['sap.ui5']?.dependencies?.minUI5Version;
                }
            },
            {
                name: 'SAP Fiori Registration',
                test: (content) => {
                    const manifest = JSON.parse(content);
                    return manifest['sap.fiori']?.registrationIds;
                }
            },
            {
                name: 'i18n Model Configuration',
                test: (content) => {
                    const manifest = JSON.parse(content);
                    return manifest['sap.app'].i18n && manifest['sap.ui5'].models?.i18n;
                }
            }
        ]
    },
    'Component.js': {
        path: 'app/a2a-fiori/webapp/Component.js',
        checks: [
            {
                name: 'UIComponent Extension',
                test: (content) => content.includes('UIComponent.extend')
            },
            {
                name: 'Manifest Declaration',
                test: (content) => content.includes('manifest: "json"')
            },
            {
                name: 'IAsyncContentCreation Interface',
                test: (content) => content.includes('sap.ui.core.IAsyncContentCreation')
            },
            {
                name: 'Proper Init Method',
                test: (content) => content.includes('UIComponent.prototype.init.apply(this, arguments)')
            }
        ]
    },
    'index.html': {
        path: 'app/a2a-fiori/webapp/index.html',
        checks: [
            {
                name: 'UI5 Bootstrap',
                test: (content) => content.includes('sap-ui-bootstrap')
            },
            {
                name: 'Theme Configuration',
                test: (content) => content.includes('data-sap-ui-theme')
            },
            {
                name: 'Async Loading',
                test: (content) => content.includes('data-sap-ui-async="true"')
            },
            {
                name: 'Resource Roots',
                test: (content) => content.includes('data-sap-ui-resourceroots')
            }
        ]
    },
    'BaseController.js': {
        path: 'app/a2a-fiori/webapp/controller/BaseController.js',
        checks: [
            {
                name: 'Controller Extension',
                test: (content) => content.includes('Controller.extend')
            },
            {
                name: 'Router Helper',
                test: (content) => content.includes('getRouter: function')
            },
            {
                name: 'Model Helper',
                test: (content) => content.includes('getModel: function')
            },
            {
                name: 'Resource Bundle',
                test: (content) => content.includes('getResourceBundle: function')
            }
        ]
    },
    'ui5.yaml': {
        path: 'app/a2a-fiori/ui5.yaml',
        checks: [
            {
                name: 'Spec Version',
                test: (content) => content.includes('specVersion:')
            },
            {
                name: 'Framework Declaration',
                test: (content) => content.includes('framework:') && content.includes('SAPUI5')
            },
            {
                name: 'Libraries Listed',
                test: (content) => content.includes('libraries:')
            }
        ]
    }
};

let passed = 0;
let failed = 0;

Object.entries(checks).forEach(([file, config]) => {
    console.log(`\nChecking ${file}:`);
    
    try {
        const content = fs.readFileSync(path.join(__dirname, '..', config.path), 'utf-8');
        
        config.checks.forEach(check => {
            try {
                if (check.test(content)) {
                    console.log(`  ✓ ${check.name}`);
                    passed++;
                } else {
                    console.log(`  ✗ ${check.name}`);
                    failed++;
                }
            } catch (e) {
                console.log(`  ✗ ${check.name} (Error: ${e.message})`);
                failed++;
            }
        });
    } catch (e) {
        console.log('  ✗ File not found or readable');
        failed += config.checks.length;
    }
});

console.log(`\n\nSummary: ${passed} passed, ${failed} failed`);
console.log(`Compliance: ${Math.round((passed / (passed + failed)) * 100)}%`);

// Additional namespace consistency check
console.log('\n\nNamespace Consistency Check:');
try {
    const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, '../app/a2a-fiori/webapp/manifest.json'), 'utf-8'));
    const componentContent = fs.readFileSync(path.join(__dirname, '../app/a2a-fiori/webapp/Component.js'), 'utf-8');
    
    const manifestId = manifest['sap.app'].id;
    const componentNameMatch = componentContent.match(/extend\("([^"]+)\.Component"/);
    const componentName = componentNameMatch ? componentNameMatch[1] : null;
    
    if (manifestId === componentName) {
        console.log(`  ✓ Namespace consistent: ${manifestId}`);
    } else {
        console.log(`  ✗ Namespace mismatch: manifest="${manifestId}", component="${componentName}"`);
    }
} catch (e) {
    console.log(`  ✗ Could not check namespace consistency: ${e.message}`);
}

process.exit(failed > 0 ? 1 : 0);