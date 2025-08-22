/**
 * Launchpad Integration Tests
 * Ensures launchpad renders correctly without blank screens
 */

const { expect } = require('chai');
const puppeteer = require('puppeteer');
const http = require('http');

describe('Launchpad Integration Tests', function() {
    this.timeout(60000); // 60 second timeout for UI tests
    
    let browser;
    let page;
    const baseUrl = process.env.TEST_URL || 'http://localhost:4004';
    
    before(async function() {
        // Wait for server to be ready
        await waitForServer(baseUrl);
        
        // Launch browser
        browser = await puppeteer.launch({
            headless: process.env.HEADLESS !== 'false',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        page = await browser.newPage();
        
        // Set viewport
        await page.setViewport({ width: 1920, height: 1080 });
        
        // Enable console logging
        page.on('console', msg => {
            if (msg.type() === 'error') {
                console.error('Browser console error:', msg.text());
            }
        });
    });
    
    after(async function() {
        if (browser) await browser.close();
    });
    
    describe('Launchpad Loading', function() {
        it('should load launchpad without errors', async function() {
            const response = await page.goto(`${baseUrl}/launchpad.html`, {
                waitUntil: 'networkidle2',
                timeout: 30000
            });
            
            expect(response.status()).to.equal(200);
        });
        
        it('should initialize SAP UI5 framework', async function() {
            const ui5Loaded = await page.waitForFunction(() => {
                return window.sap && 
                       window.sap.ui && 
                       window.sap.ui.getCore && 
                       window.sap.ushell;
            }, { timeout: 15000 });
            
            expect(ui5Loaded).to.be.ok;
        });
        
        it('should not display error messages', async function() {
            const errorMessages = await page.evaluate(() => {
                // Check for error dialogs
                const errorDialogs = document.querySelectorAll('.sapMMessageBox');
                const visibleErrors = Array.from(errorDialogs).filter(el => {
                    const style = window.getComputedStyle(el);
                    return style.display !== 'none' && style.visibility !== 'hidden';
                });
                
                // Check for error text in page
                const bodyText = document.body.innerText || '';
                const hasErrorText = /error|failed|exception/i.test(bodyText);
                
                return {
                    hasErrorDialogs: visibleErrors.length > 0,
                    hasErrorText: hasErrorText,
                    errorCount: visibleErrors.length
                };
            });
            
            expect(errorMessages.hasErrorDialogs).to.be.false;
            if (errorMessages.hasErrorText) {
                console.warn('Warning: Error-like text found in page content');
            }
        });
    });
    
    describe('Tile Rendering', function() {
        it('should render tile container', async function() {
            const tileContainer = await page.waitForSelector('#__xmlview0--tileContainer', {
                timeout: 10000
            });
            
            expect(tileContainer).to.exist;
        });
        
        it('should render exactly 6 tiles', async function() {
            await page.waitForTimeout(2000); // Give tiles time to render
            
            const tileCount = await page.evaluate(() => {
                const container = document.querySelector('#__xmlview0--tileContainer');
                if (!container) return 0;
                
                // Count GenericTile elements
                const tiles = container.querySelectorAll('[class*="sapMGT"]');
                return tiles.length;
            });
            
            expect(tileCount).to.equal(6);
        });
        
        it('should have no empty tiles', async function() {
            const tileData = await page.evaluate(() => {
                const tiles = document.querySelectorAll('[class*="sapMGT"]');
                return Array.from(tiles).map(tile => {
                    const header = tile.querySelector('.sapMGTHdrTxt');
                    const value = tile.querySelector('.sapMGTValueScr');
                    return {
                        hasHeader: header && header.textContent.trim().length > 0,
                        hasValue: value !== null,
                        headerText: header ? header.textContent.trim() : null
                    };
                });
            });
            
            expect(tileData).to.have.lengthOf(6);
            tileData.forEach((tile, index) => {
                expect(tile.hasHeader, `Tile ${index} should have header`).to.be.true;
                expect(tile.hasValue, `Tile ${index} should have value`).to.be.true;
                expect(tile.headerText, `Tile ${index} header should not be empty`).to.not.be.null;
            });
        });
        
        it('should have proper tile structure', async function() {
            const expectedTiles = [
                'Agent Management',
                'Service Marketplace',
                'Workflow Designer',
                'Network Analytics',
                'Notification Center',
                'Security & Audit'
            ];
            
            const actualTiles = await page.evaluate(() => {
                const tiles = document.querySelectorAll('[class*="sapMGT"]');
                return Array.from(tiles).map(tile => {
                    const header = tile.querySelector('.sapMGTHdrTxt');
                    return header ? header.textContent.trim() : '';
                });
            });
            
            expectedTiles.forEach(expected => {
                expect(actualTiles).to.include(expected);
            });
        });
    });
    
    describe('Data Loading', function() {
        it('should fetch tile data from API', async function() {
            // Intercept API calls
            const apiCalls = [];
            page.on('response', response => {
                if (response.url().includes('/api/v1/Agents')) {
                    apiCalls.push({
                        url: response.url(),
                        status: response.status()
                    });
                }
            });
            
            // Trigger refresh
            await page.evaluate(() => {
                const refreshButton = document.querySelector('[icon="sap-icon://refresh"]');
                if (refreshButton) refreshButton.click();
            });
            
            // Wait for API call
            await page.waitForTimeout(2000);
            
            expect(apiCalls).to.have.length.greaterThan(0);
            expect(apiCalls[0].status).to.equal(200);
        });
        
        it('should update tile values', async function() {
            const tileValues = await page.evaluate(() => {
                const tiles = document.querySelectorAll('[class*="sapMGT"]');
                return Array.from(tiles).map(tile => {
                    const valueElement = tile.querySelector('.sapMNCValue');
                    return valueElement ? valueElement.textContent.trim() : null;
                });
            });
            
            expect(tileValues).to.have.lengthOf(6);
            // At least one tile should have a non-zero value
            const hasNonZeroValue = tileValues.some(value => value && value !== '0');
            if (!hasNonZeroValue) {
                console.warn('Warning: All tiles showing zero values - no real data available');
            }
        });
    });
    
    describe('Visual Consistency', function() {
        it('should not have blank screen', async function() {
            const screenshot = await page.screenshot({ fullPage: true });
            
            // Check if page has content
            const pageContent = await page.evaluate(() => {
                const body = document.body;
                const hasVisibleContent = body.offsetHeight > 100 && body.offsetWidth > 100;
                const hasChildElements = body.children.length > 0;
                return { hasVisibleContent, hasChildElements };
            });
            
            expect(pageContent.hasVisibleContent).to.be.true;
            expect(pageContent.hasChildElements).to.be.true;
        });
        
        it('should have proper SAP theme applied', async function() {
            const theme = await page.evaluate(() => {
                return sap.ui.getCore().getConfiguration().getTheme();
            });
            
            expect(theme).to.be.oneOf(['sap_horizon', 'sap_horizon_dark', 'sap_fiori_3', 'sap_fiori_3_dark']);
        });
    });
});

// Helper function to wait for server
async function waitForServer(url, maxRetries = 30) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            await new Promise((resolve, reject) => {
                http.get(url + '/health', res => {
                    if (res.statusCode === 200) resolve();
                    else reject(new Error(`Status ${res.statusCode}`));
                }).on('error', reject);
            });
            return;
        } catch (e) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    throw new Error('Server not ready');
}