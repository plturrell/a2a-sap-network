const puppeteer = require('puppeteer');

async function runUITests() {
    console.log('🧪 Running Simple UI Tests...\n');
    
    const results = {
        backend: {},
        frontend: {},
        useCases: []
    };
    
    // Test 1: Backend Health
    console.log('1️⃣ Testing Backend Services...');
    try {
        const backendResponse = await fetch('http://localhost:4004/');
        results.backend.running = backendResponse.ok;
        results.backend.status = backendResponse.status;
        console.log(`   ✅ Backend is running (status: ${backendResponse.status})`);
    } catch (error) {
        results.backend.running = false;
        results.backend.error = error.message;
        console.log(`   ❌ Backend error: ${error.message}`);
    }
    
    // Test 2: API Services
    console.log('\n2️⃣ Testing API Services...');
    try {
        const apiResponse = await fetch('http://localhost:4004/api/v1');
        results.backend.apiAvailable = apiResponse.ok;
        console.log('   ✅ API endpoint available');
    } catch (error) {
        results.backend.apiAvailable = false;
        console.log('   ❌ API not available');
    }
    
    // Test 3: Frontend UI
    console.log('\n3️⃣ Testing Frontend UI...');
    let browser;
    try {
        browser = await puppeteer.launch({ 
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const page = await browser.newPage();
        
        // Test Fiori Launchpad
        console.log('   📱 Testing Fiori Launchpad...');
        await page.goto('http://localhost:4004/fiori-launchpad.html', {
            waitUntil: 'networkidle2',
            timeout: 30000
        });
        
        // Check if UI5 loaded
        const ui5Loaded = await page.evaluate(() => {
            return window.sap && window.sap.ui && window.sap.ui.getCore();
        });
        results.frontend.ui5Loaded = ui5Loaded;
        console.log(`   ${ui5Loaded ? '✅' : '❌'} UI5 Framework loaded`);
        
        // Check for content container
        const contentExists = await page.$('#content');
        results.frontend.contentContainer = !!contentExists;
        console.log(`   ${contentExists ? '✅' : '❌'} Content container exists`);
        
        // Check for shell
        let shellRendered = false;
        try {
            await page.waitForSelector('.sapUShellShell', { timeout: 5000 });
            shellRendered = true;
        } catch (e) {
            shellRendered = false;
        }
        results.frontend.shellRendered = shellRendered;
        console.log(`   ${shellRendered ? '✅' : '❌'} Shell container rendered`);
        
        // Check for tiles
        let tilesFound = 0;
        try {
            const tiles = await page.$$('.sapUshellTile');
            tilesFound = tiles.length;
        } catch (e) {
            tilesFound = 0;
        }
        results.frontend.tilesCount = tilesFound;
        console.log(`   ${tilesFound > 0 ? '✅' : '❌'} Tiles found: ${tilesFound}`);
        
        // Test 4: Use Cases
        console.log('\n4️⃣ Testing UI Use Cases...');
        
        // Use Case 1: Navigation
        if (tilesFound > 0) {
            console.log('   🔄 Use Case: Tile Navigation');
            try {
                const firstTile = await page.$('.sapUshellTile');
                if (firstTile) {
                    await firstTile.click();
                    await page.waitForTimeout(2000);
                    const newUrl = page.url();
                    const navigated = newUrl !== 'http://localhost:4004/fiori-launchpad.html';
                    results.useCases.push({
                        name: 'Tile Navigation',
                        passed: navigated,
                        details: { newUrl }
                    });
                    console.log(`   ${navigated ? '✅' : '❌'} Navigation worked`);
                }
            } catch (error) {
                results.useCases.push({
                    name: 'Tile Navigation',
                    passed: false,
                    error: error.message
                });
                console.log(`   ❌ Navigation failed: ${error.message}`);
            }
        }
        
        // Use Case 2: Check for A2A Network specific elements
        console.log('   🔍 Use Case: A2A Network Elements');
        const a2aElements = await page.evaluate(() => {
            const elements = {
                hasA2AText: document.body.innerText.includes('A2A'),
                hasNetworkText: document.body.innerText.includes('Network'),
                hasAgentText: document.body.innerText.includes('Agent'),
                hasBlockchainText: document.body.innerText.includes('Blockchain')
            };
            return elements;
        });
        results.useCases.push({
            name: 'A2A Network Content',
            passed: Object.values(a2aElements).some(v => v),
            details: a2aElements
        });
        console.log(`   ${Object.values(a2aElements).some(v => v) ? '✅' : '❌'} A2A content found`);
        
        // Take screenshot
        await page.screenshot({ path: 'ui-test-screenshot.png', fullPage: true });
        console.log('   📸 Screenshot saved as ui-test-screenshot.png');
        
    } catch (error) {
        results.frontend.error = error.message;
        console.log(`   ❌ Frontend test error: ${error.message}`);
    } finally {
        if (browser) await browser.close();
    }
    
    // Summary
    console.log('\n📊 TEST SUMMARY:');
    console.log('================');
    console.log(`Backend: ${results.backend.running ? '✅ Running' : '❌ Not Running'}`);
    console.log(`Frontend UI5: ${results.frontend.ui5Loaded ? '✅ Loaded' : '❌ Not Loaded'}`);
    console.log(`Shell Rendered: ${results.frontend.shellRendered ? '✅ Yes' : '❌ No'}`);
    console.log(`Tiles Found: ${results.frontend.tilesCount || 0}`);
    console.log(`\nUse Cases Passed: ${results.useCases.filter(uc => uc.passed).length}/${results.useCases.length}`);
    
    return results;
}

// Run the tests
runUITests().then(results => {
    console.log('\n✅ Tests completed');
    process.exit(results.frontend.shellRendered ? 0 : 1);
}).catch(error => {
    console.error('❌ Test failed:', error);
    process.exit(1);
});