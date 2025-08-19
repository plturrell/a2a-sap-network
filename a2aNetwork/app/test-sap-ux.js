#!/usr/bin/env node
/**
 * Test Script to Verify SAP Fiori UX Experience
 * Validates that the launchpad renders proper SAP visual experience
 */
const puppeteer = require('puppeteer-core');
const express = require('express');
const path = require('path');

// Start a simple test server
const app = express();
app.use(express.static(__dirname));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'launchpad.html'));
});

const server = app.listen(3005, () => {
    // console.log('🧪 Test server started on http://localhost:3005');
});

async function testSAPFioriUX() {
    let browser;
    try {
        // console.log('🚀 Testing SAP Fiori UX Experience...\n');
        
        // Use system Chrome if available
        const chromePaths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser'
        ];
        
        let executablePath;
        for (const chromePath of chromePaths) {
            try {
                require('fs').accessSync(chromePath);
                executablePath = chromePath;
                break;
            } catch (e) {
                continue;
            }
        }
        
        if (!executablePath) {
            // console.log('❌ Chrome/Chromium not found for visual testing');
            // console.log('✅ Manual Test: Open http://localhost:3005 to verify SAP Fiori UX');
            setTimeout(() => {
                server.close();
                process.exit(0);
            }, 5000);
            return;
        }
        
        browser = await puppeteer.launch({
            executablePath,
            headless: false, // Show browser to see visual experience
            args: ['--no-sandbox', '--disable-setuid-sandbox'],
            defaultViewport: { width: 1920, height: 1080 }
        });
        
        const page = await browser.newPage();
        
        // Navigate to launchpad
        // console.log('📱 Loading SAP Fiori Launchpad...');
        await page.goto('http://localhost:3005', { waitUntil: 'networkidle0', timeout: 30000 });
        
        // Wait for SAP UI5 to load
        // console.log('⏳ Waiting for SAP UI5 shell to render...');
        await page.waitForTimeout(5000);
        
        // Check for SAP elements
        const checks = [
            {
                name: 'SAP UI5 Core Loaded',
                selector: () => typeof window.sap !== 'undefined' && window.sap.ui
            },
            {
                name: 'SAP UShell Bootstrap',
                selector: () => window.sap && window.sap.ushell
            },
            {
                name: 'SAP Fiori Shell',
                selector: '.sapUShellShell'
            },
            {
                name: 'SAP Theme Applied',
                selector: () => document.documentElement.getAttribute('data-sap-ui-theme') === 'sap_horizon'
            },
            {
                name: 'SAP Loading Screen',
                selector: '.sapUShellLoadingContainer'
            }
        ];
        
        // console.log('\n🔍 SAP UX Visual Checks:');
        
        for (const check of checks) {
            try {
                let result;
                if (typeof check.selector === 'function') {
                    result = await page.evaluate(check.selector);
                } else {
                    result = await page.$(check.selector) !== null;
                }
                
                // console.log(`   ${result ? '✅' : '❌'} ${check.name}`);
            } catch (error) {
                // console.log(`   ❌ ${check.name} (Error: ${error.message})`);
            }
        }
        
        // Take screenshot
        // console.log('\n📸 Taking screenshot of SAP Fiori experience...');
        await page.screenshot({ 
            path: './test/sap-fiori-ux-screenshot.png', 
            fullPage: true 
        });
        
        // Check for tile rendering
        await page.waitForTimeout(3000);
        const tiles = await page.$$('.sapUShellTile');
        // console.log(`\n🎯 Found ${tiles.length} rendered tiles`);
        
        // Check console for SAP-specific messages
        const logs = await page.evaluate(() => {
            return window.console._logs || [];
        });
        
        const sapLogs = logs.filter(log => 
            log.includes('SAP') || 
            log.includes('UI5') || 
            log.includes('ushell') ||
            log.includes('Fiori')
        );
        
        if (sapLogs.length > 0) {
            // console.log('\n📋 SAP-related console messages:');
            sapLogs.forEach(log => console.log(`   📝 ${log}`));
        }
        
        // console.log('\n🎉 SAP Fiori UX Test Complete!');
        // console.log('   Check the screenshot at: ./test/sap-fiori-ux-screenshot.png');
        // console.log('   Browser will remain open for manual inspection...');
        
        // Keep browser open for manual inspection
        await page.waitForTimeout(10000);
        
    } catch (error) {
        console.error('❌ SAP UX Test Failed:', error.message);
    } finally {
        if (browser) {
            await browser.close();
        }
        server.close();
    }
}

// Handle cleanup
process.on('SIGINT', () => {
    // console.log('\n🛑 Test interrupted by user');
    server.close();
    process.exit(0);
});

// Run test
testSAPFioriUX().catch(error => {
    console.error('❌ Test execution failed:', error);
    server.close();
    process.exit(1);
});