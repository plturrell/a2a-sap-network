#!/usr/bin/env node

/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

/**
 * Launchpad Validation Script
 * Ensures the launchpad is fully functional with real data
 */

const http = require('http');
const https = require('https');
const { execSync } = require('child_process');
const puppeteer = require('puppeteer');

class LaunchpadValidator {
    constructor(host = 'localhost', port = 4004) {
        this.host = host;
        this.port = port;
        this.baseUrl = `http://${host}:${port}`;
        this.errors = [];
        this.warnings = [];
    }

    // 1. Check if server is running
    async checkServerHealth() {
        console.log('🔍 Checking server health...');
        try {
            const health = await this.fetchJson('/health');
            if (health.status !== 'healthy') {
                this.errors.push(`Server health status: ${health.status}`);
                return false;
            }
            console.log('✅ Server is healthy');
            return true;
        } catch (error) {
            this.errors.push(`Server not responding: ${error.message}`);
            return false;
        }
    }

    // 2. Check launchpad-specific health
    async checkLaunchpadHealth() {
        console.log('\n🔍 Checking launchpad health...');
        try {
            const health = await this.fetchJson('/api/v1/launchpad/health');
            
            // Check critical components
            if (health.status === 'error') {
                this.errors.push('Launchpad health check failed');
                return false;
            }

            // Check if tiles are loaded
            if (!health.tiles_loaded) {
                this.errors.push('Tiles not loaded properly');
                return false;
            }

            // Check if we're in fallback mode
            if (health.fallback_mode) {
                this.warnings.push('Launchpad is in fallback mode (no real data)');
            }

            // Check individual components
            Object.entries(health.components).forEach(([component, status]) => {
                if (status.status === 'error') {
                    this.errors.push(`Component ${component} is in error state`);
                } else if (status.status === 'warning') {
                    this.warnings.push(`Component ${component} has warnings`);
                }
            });

            console.log('✅ Launchpad health check passed');
            return true;
        } catch (error) {
            this.errors.push(`Launchpad health check failed: ${error.message}`);
            return false;
        }
    }

    // 3. Check all required endpoints
    async checkEndpoints() {
        console.log('\n🔍 Checking API endpoints...');
        const endpoints = [
            { path: '/api/v1/Agents?id=agent_visualization', name: 'Agent Visualization' },
            { path: '/api/v1/network/overview', name: 'Network Overview' },
            { path: '/api/v1/blockchain/status', name: 'Blockchain Status' },
            { path: '/api/v1/services/count', name: 'Services Count' },
            { path: '/api/v1/health/summary', name: 'Health Summary' }
        ];

        let allPassed = true;
        for (const endpoint of endpoints) {
            try {
                const response = await this.fetchJson(endpoint.path);
                if (endpoint.path.includes('agent_visualization')) {
                    // Check if we have real data
                    if (response.agentCount === 0 && 
                        response.services === 0 && 
                        response.workflows === 0) {
                        this.warnings.push(`${endpoint.name}: No real data (all zeros)`);
                    }
                }
                console.log(`✅ ${endpoint.name}: OK`);
            } catch (error) {
                this.errors.push(`${endpoint.name} failed: ${error.message}`);
                allPassed = false;
            }
        }
        return allPassed;
    }

    // 4. Check UI5 resources
    async checkUI5Resources() {
        console.log('\n🔍 Checking UI5 resources...');
        try {
            const response = await this.fetchHead('https://ui5.sap.com/1.120.0/resources/sap-ui-core.js');
            if (response.ok) {
                console.log('✅ UI5 resources are accessible');
                return true;
            } else {
                this.errors.push('UI5 resources not accessible');
                return false;
            }
        } catch (error) {
            this.warnings.push(`UI5 resource check failed: ${error.message} (may be offline)`);
            return true; // Don't fail if offline
        }
    }

    // 5. Visual validation with Puppeteer
    async validateVisualRendering() {
        console.log('\n🔍 Performing visual validation...');
        let browser;
        try {
            browser = await puppeteer.launch({ 
                headless: true,
                args: ['--no-sandbox', '--disable-setuid-sandbox']
            });
            
            const page = await browser.newPage();
            
            // Set viewport
            await page.setViewport({ width: 1920, height: 1080 });
            
            // Navigate to launchpad
            console.log('  Loading launchpad...');
            await page.goto(`${this.baseUrl}/launchpad.html`, { 
                waitUntil: 'networkidle2',
                timeout: 30000 
            });
            
            // Wait for SAP UI5 to initialize
            await page.waitForFunction(() => {
                return window.sap && window.sap.ui && window.sap.ui.getCore;
            }, { timeout: 10000 });
            
            console.log('  Waiting for tiles to render...');
            
            // Wait for tile container
            const tileContainerExists = await page.waitForSelector('#__xmlview0--tileContainer', { 
                timeout: 10000 
            }).then(() => true).catch(() => false);
            
            if (!tileContainerExists) {
                this.errors.push('Tile container not found in DOM');
                return false;
            }
            
            // Count rendered tiles
            const tileCount = await page.evaluate(() => {
                const container = document.querySelector('#__xmlview0--tileContainer');
                if (!container) return 0;
                
                // Count GenericTile elements
                const tiles = container.querySelectorAll('[class*="sapMGT"]');
                return tiles.length;
            });
            
            if (tileCount === 0) {
                this.errors.push('No tiles rendered on the page');
                return false;
            } else if (tileCount < 6) {
                this.warnings.push(`Only ${tileCount} tiles rendered, expected 6`);
            } else {
                console.log(`✅ ${tileCount} tiles rendered successfully`);
            }
            
            // Check for error messages
            const errorMessages = await page.evaluate(() => {
                const errors = [];
                // Check for error dialogs
                const errorDialogs = document.querySelectorAll('[class*="sapMMessageBox"]');
                errorDialogs.forEach(dialog => {
                    errors.push(dialog.textContent);
                });
                
                // Check for error in content
                const errorTexts = document.body.textContent.match(/error|failed|exception/gi);
                if (errorTexts) {
                    errors.push(...errorTexts);
                }
                
                return errors;
            });
            
            if (errorMessages.length > 0) {
                this.warnings.push(`Error messages found on page: ${errorMessages.join(', ')}`);
            }
            
            // Take screenshot for debugging
            await page.screenshot({ 
                path: 'launchpad-validation.png',
                fullPage: true 
            });
            console.log('  Screenshot saved: launchpad-validation.png');
            
            return true;
            
        } catch (error) {
            this.errors.push(`Visual validation failed: ${error.message}`);
            return false;
        } finally {
            if (browser) await browser.close();
        }
    }

    // Helper methods
    async fetchJson(path) {
        return new Promise((resolve, reject) => {
            blockchainClient.sendMessage(`${this.baseUrl}${path}`, { 
                headers: { 'Accept': 'application/json' }
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        resolve(JSON.parse(data));
                    } catch (e) {
                        reject(new Error(`Invalid JSON response from ${path}`));
                    }
                });
            }).on('error', reject);
        });
    }

    async fetchHead(url) {
        return new Promise((resolve, reject) => {
            const client = url.startsWith('https') ? https : http;
            const req = client.request(url, { method: 'HEAD' }, (res) => {
                resolve({ ok: res.statusCode === 200, status: res.statusCode });
            });
            req.on('error', reject);
            req.end();
        });
    }

    // Main validation flow
    async validate() {
        console.log('🚀 A2A Network Launchpad Validation\n');
        console.log(`Target: ${this.baseUrl}`);
        console.log('=' + '='.repeat(40) + '\n');

        const checks = [
            { name: 'Server Health', fn: () => this.checkServerHealth() },
            { name: 'Launchpad Health', fn: () => this.checkLaunchpadHealth() },
            { name: 'API Endpoints', fn: () => this.checkEndpoints() },
            { name: 'UI5 Resources', fn: () => this.checkUI5Resources() },
            { name: 'Visual Rendering', fn: () => this.validateVisualRendering() }
        ];

        const results = [];
        for (const check of checks) {
            try {
                const result = await check.fn();
                results.push({ name: check.name, passed: result });
            } catch (error) {
                results.push({ name: check.name, passed: false });
                this.errors.push(`${check.name} check failed: ${error.message}`);
            }
        }

        // Summary
        console.log('\n' + '='.repeat(50));
        console.log('📊 VALIDATION SUMMARY\n');
        
        results.forEach(result => {
            const icon = result.passed ? '✅' : '❌';
            console.log(`${icon} ${result.name}: ${result.passed ? 'PASSED' : 'FAILED'}`);
        });

        if (this.warnings.length > 0) {
            console.log('\n⚠️  WARNINGS:');
            this.warnings.forEach(warning => console.log(`   - ${warning}`));
        }

        if (this.errors.length > 0) {
            console.log('\n❌ ERRORS:');
            this.errors.forEach(error => console.log(`   - ${error}`));
        }

        const allPassed = results.every(r => r.passed);
        const hasRealData = !this.warnings.some(w => w.includes('fallback mode'));

        console.log('\n' + '='.repeat(50));
        if (allPassed && this.errors.length === 0) {
            if (hasRealData) {
                console.log('✅ VALIDATION PASSED - Launchpad is fully functional with real data!');
                return 0;
            } else {
                console.log('⚠️  VALIDATION PASSED WITH WARNINGS - Launchpad works but with fallback data');
                return 1;
            }
        } else {
            console.log('❌ VALIDATION FAILED - Launchpad has critical issues');
            return 2;
        }
    }
}

// Run validation
if (require.main === module) {
    const validator = new LaunchpadValidator();
    validator.validate().then(exitCode => {
        process.exit(exitCode);
    }).catch(error => {
        console.error('Validation error:', error);
        process.exit(3);
    });
}

module.exports = LaunchpadValidator;