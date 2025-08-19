#!/usr/bin/env node

const http = require('http');
const https = require('https');

console.log('Testing A2A Network UI Access...\n');

const tests = [
    {
        name: 'Main UI Index',
        url: 'http://localhost:4004/app/a2a-fiori/index.html',
        expect: 'ComponentContainer'
    },
    {
        name: 'Manifest.json',
        url: 'http://localhost:4004/app/a2a-fiori/manifest.json',
        expect: 'a2a.network.fiori'
    },
    {
        name: 'Component.js',
        url: 'http://localhost:4004/app/a2a-fiori/Component.js',
        expect: 'UIComponent.extend'
    },
    {
        name: 'BaseController.js',
        url: 'http://localhost:4004/app/a2a-fiori/controller/BaseController.js',
        expect: 'Controller.extend'
    },
    {
        name: 'Fiori Launchpad',
        url: 'http://localhost:4004/fiori-launchpad.html',
        expect: 'sap.ushell.Container'
    },
    {
        name: 'UI5 CDN Access',
        url: 'https://ui5.sap.com/1.120.0/resources/sap-ui-version.json',
        expect: 'version'
    }
];

function testUrl(test) {
    return new Promise((resolve) => {
        const protocol = test.url.startsWith('https') ? https : http;
        
        protocol.get(test.url, (res) => {
            let data = '';
            
            res.on('data', (chunk) => {
                data += chunk;
            });
            
            res.on('end', () => {
                const passed = data.includes(test.expect);
                console.log(`${passed ? '✓' : '✗'} ${test.name}`);
                console.log(`  URL: ${test.url}`);
                console.log(`  Status: ${res.statusCode}`);
                console.log(`  Expected content: ${passed ? 'Found' : 'Not found'} "${test.expect}"`);
                console.log('');
                resolve(passed);
            });
        }).on('error', (err) => {
            console.log(`✗ ${test.name}`);
            console.log(`  URL: ${test.url}`);
            console.log(`  Error: ${err.message}`);
            console.log('');
            resolve(false);
        });
    });
}

async function runTests() {
    const results = [];
    
    for (const test of tests) {
        const result = await testUrl(test);
        results.push(result);
    }
    
    const passed = results.filter(r => r).length;
    const failed = results.length - passed;
    
    console.log(`\nSummary: ${passed} passed, ${failed} failed`);
    
    if (failed === 0) {
        console.log('\n✓ All UI access tests passed! The UI is properly configured.');
    } else {
        console.log('\n✗ Some tests failed. Please check the configuration.');
    }
    
    process.exit(failed > 0 ? 1 : 0);
}

runTests();