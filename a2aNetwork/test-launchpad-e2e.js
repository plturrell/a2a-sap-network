#!/usr/bin/env node

/**
 * End-to-End Launchpad Test
 * 
 * Comprehensive test of the A2A Network Fiori Launchpad
 * Tests functionality, data loading, navigation, and user interactions
 */

const puppeteer = require('puppeteer');
const path = require('path');

const TEST_CONFIG = {
    baseUrl: 'http://localhost:4004',
    timeout: 30000,
    headless: false, // Set to true for CI/CD
    viewport: { width: 1920, height: 1080 }
};

class LaunchpadTester {
    constructor() {
        this.browser = null;
        this.page = null;
        this.results = {
            passed: 0,
            failed: 0,
            errors: [],
            performance: {}
        };
    }

    async start() {
        console.log('🚀 Starting A2A Network Launchpad E2E Test');
        console.log('='.repeat(60));

        try {
            await this.initBrowser();
            await this.runTests();
        } catch (error) {
            console.error('💥 Test suite failed:', error);
            this.results.errors.push({ test: 'suite', error: error.message });
        } finally {
            await this.cleanup();
            this.printResults();
        }
    }

    async initBrowser() {
        console.log('📱 Initializing browser...');
        this.browser = await puppeteer.launch({
            headless: TEST_CONFIG.headless,
            defaultViewport: TEST_CONFIG.viewport,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu'
            ]
        });

        this.page = await this.browser.newPage();
        
        // Set up console logging
        this.page.on('console', msg => {
            if (msg.type() === 'error') {
                console.log('🔴 Console Error:', msg.text());
            }
        });

        // Set up error logging
        this.page.on('pageerror', error => {
            console.log('🔴 Page Error:', error.message);
            this.results.errors.push({ test: 'page-error', error: error.message });
        });
    }

    async runTests() {
        await this.testServerHealth();
        await this.testLaunchpadLoad();
        await this.testTileData();
        await this.testTileNavigation();
        await this.testResponsiveDesign();
        await this.testPerformance();
        await this.testErrorHandling();
    }

    async testServerHealth() {
        console.log('🏥 Testing server health...');
        try {
            const response = await this.page.goto(`${TEST_CONFIG.baseUrl}/health`);
            const status = response.status();
            const text = await response.text();
            
            if (status === 200 && text.includes('UP')) {
                this.pass('Server health check');
            } else {
                this.fail('Server health check', `Status: ${status}, Response: ${text}`);
            }
        } catch (error) {
            this.fail('Server health check', error.message);
        }
    }

    async testLaunchpadLoad() {
        console.log('🎯 Testing launchpad loading...');
        try {
            const startTime = Date.now();
            
            await this.page.goto(`${TEST_CONFIG.baseUrl}/app/fioriLaunchpad.html`, {
                waitUntil: 'networkidle2',
                timeout: TEST_CONFIG.timeout
            });

            const loadTime = Date.now() - startTime;
            this.results.performance.launchpadLoad = loadTime;

            // Wait for UI5 to load
            await this.page.waitForSelector('.sapUShellShell', { timeout: 15000 });
            
            // Check if loading indicator disappears
            await this.page.waitForFunction(() => {
                const loadingDiv = document.getElementById('loadingIndicator');
                return !loadingDiv || loadingDiv.style.display === 'none';
            }, { timeout: 10000 });

            this.pass('Launchpad loads successfully');
            console.log(`   ⏱️  Load time: ${loadTime}ms`);

        } catch (error) {
            this.fail('Launchpad loading', error.message);
        }
    }

    async testTileData() {
        console.log('📊 Testing tile data loading...');
        
        const tileTests = [
            {
                name: 'Overview Dashboard',
                endpoint: '/api/v1/NetworkStats?id=overview_dashboard'
            },
            {
                name: 'Agent Visualization', 
                endpoint: '/api/v1/Agents?id=agent_visualization'
            },
            {
                name: 'Service Marketplace',
                endpoint: '/api/v1/Services?id=service_marketplace'
            },
            {
                name: 'Blockchain Dashboard',
                endpoint: '/api/v1/blockchain/stats?id=blockchain_dashboard'
            }
        ];

        for (const test of tileTests) {
            try {
                const response = await this.page.goto(`${TEST_CONFIG.baseUrl}${test.endpoint}`);
                const data = await response.json();
                
                if (response.status() === 200 && data.id && data.data) {
                    this.pass(`Tile data: ${test.name}`);
                    console.log(`   ✅ ${test.name}: ${Object.keys(data.data).length} data points`);
                } else {
                    this.fail(`Tile data: ${test.name}`, `Status: ${response.status()}`);
                }
            } catch (error) {
                this.fail(`Tile data: ${test.name}`, error.message);
            }
        }
    }

    async testTileNavigation() {
        console.log('🧭 Testing tile navigation...');
        try {
            // Go back to launchpad
            await this.page.goto(`${TEST_CONFIG.baseUrl}/app/fioriLaunchpad.html`, {
                waitUntil: 'networkidle2'
            });

            // Wait for tiles to load
            await this.page.waitForSelector('.sapUShellTile', { timeout: 10000 });

            // Find and click tiles
            const tiles = await this.page.$$('.sapUShellTile');
            console.log(`   📋 Found ${tiles.length} tiles`);

            if (tiles.length > 0) {
                // Test clicking the first tile
                await tiles[0].click();
                await this.page.waitForTimeout(2000); // Wait for navigation

                this.pass('Tile navigation');
            } else {
                this.fail('Tile navigation', 'No tiles found');
            }

        } catch (error) {
            this.fail('Tile navigation', error.message);
        }
    }

    async testResponsiveDesign() {
        console.log('📱 Testing responsive design...');
        
        const viewports = [
            { width: 1920, height: 1080, name: 'Desktop' },
            { width: 768, height: 1024, name: 'Tablet' },
            { width: 375, height: 667, name: 'Mobile' }
        ];

        for (const viewport of viewports) {
            try {
                await this.page.setViewport(viewport);
                await this.page.goto(`${TEST_CONFIG.baseUrl}/app/fioriLaunchpad.html`, {
                    waitUntil: 'networkidle2'
                });

                // Check if page renders without layout issues
                const bodyVisible = await this.page.$eval('body', el => 
                    window.getComputedStyle(el).visibility === 'visible'
                );

                if (bodyVisible) {
                    this.pass(`Responsive design: ${viewport.name}`);
                } else {
                    this.fail(`Responsive design: ${viewport.name}`, 'Layout issues detected');
                }

            } catch (error) {
                this.fail(`Responsive design: ${viewport.name}`, error.message);
            }
        }

        // Reset to desktop viewport
        await this.page.setViewport(TEST_CONFIG.viewport);
    }

    async testPerformance() {
        console.log('⚡ Testing performance metrics...');
        try {
            // Enable request interception to measure network performance
            await this.page.setRequestInterception(true);
            
            const requests = [];
            this.page.on('request', request => {
                requests.push({
                    url: request.url(),
                    method: request.method(),
                    startTime: Date.now()
                });
                request.continue();
            });

            await this.page.goto(`${TEST_CONFIG.baseUrl}/app/fioriLaunchpad.html`, {
                waitUntil: 'networkidle2'
            });

            // Measure performance metrics
            const metrics = await this.page.metrics();
            this.results.performance.metrics = metrics;

            // Check if page loads within acceptable time
            if (metrics.TaskDuration < 5000) { // 5 seconds
                this.pass('Performance: Page load time');
            } else {
                this.fail('Performance: Page load time', `Task duration: ${metrics.TaskDuration}ms`);
            }

            console.log(`   📈 DOM Nodes: ${metrics.Nodes}`);
            console.log(`   🔧 JS Heap: ${Math.round(metrics.JSHeapUsedSize / 1024 / 1024)}MB`);
            console.log(`   📡 Network Requests: ${requests.length}`);

        } catch (error) {
            this.fail('Performance testing', error.message);
        }
    }

    async testErrorHandling() {
        console.log('🚨 Testing error handling...');
        try {
            // Test 404 page
            const response = await this.page.goto(`${TEST_CONFIG.baseUrl}/non-existent-page`);
            
            if (response.status() === 404) {
                this.pass('Error handling: 404 page');
            } else {
                this.fail('Error handling: 404 page', `Expected 404, got ${response.status()}`);
            }

            // Test invalid API endpoint
            const apiResponse = await this.page.goto(`${TEST_CONFIG.baseUrl}/api/v1/invalid-endpoint`);
            
            if (apiResponse.status() >= 400) {
                this.pass('Error handling: Invalid API endpoint');
            } else {
                this.fail('Error handling: Invalid API endpoint', `Expected error status, got ${apiResponse.status()}`);
            }

        } catch (error) {
            this.fail('Error handling tests', error.message);
        }
    }

    pass(test) {
        console.log(`   ✅ ${test}`);
        this.results.passed++;
    }

    fail(test, reason) {
        console.log(`   ❌ ${test}: ${reason}`);
        this.results.failed++;
        this.results.errors.push({ test, reason });
    }

    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
    }

    printResults() {
        console.log('\n' + '='.repeat(60));
        console.log('📊 TEST RESULTS');
        console.log('='.repeat(60));
        console.log(`✅ Passed: ${this.results.passed}`);
        console.log(`❌ Failed: ${this.results.failed}`);
        console.log(`📈 Success Rate: ${Math.round((this.results.passed / (this.results.passed + this.results.failed)) * 100)}%`);

        if (this.results.performance.launchpadLoad) {
            console.log(`⏱️  Launchpad Load Time: ${this.results.performance.launchpadLoad}ms`);
        }

        if (this.results.errors.length > 0) {
            console.log('\n🔍 DETAILED ERRORS:');
            this.results.errors.forEach((error, index) => {
                console.log(`${index + 1}. ${error.test}: ${error.reason || error.error}`);
            });
        }

        // Recommendations
        console.log('\n💡 RECOMMENDATIONS:');
        if (this.results.performance.launchpadLoad > 3000) {
            console.log('- Consider optimizing launchpad load time (currently > 3s)');
        }
        if (this.results.failed > 0) {
            console.log('- Review failed tests and fix underlying issues');
        }
        if (this.results.errors.some(e => e.test.includes('tile'))) {
            console.log('- Check tile data endpoints and ensure real-time data is loading');
        }

        console.log('\n🎯 OVERALL STATUS:', this.results.failed === 0 ? '✅ HEALTHY' : '⚠️  NEEDS ATTENTION');
    }
}

// Run the test if called directly
if (require.main === module) {
    const tester = new LaunchpadTester();
    tester.start();
}

module.exports = LaunchpadTester;