/**
 * Global Jest Teardown
 * Runs once after all test suites complete
 */

module.exports = async () => {
    console.log('🌍 Global Jest Teardown: Cleaning up A2A Test Environment');
    
    // Clean up any global resources
    if (global.gc) {
        global.gc();
    }
    
    // Log test completion summary
    const fs = require('fs');
    const path = require('path');
    
    try {
        // Check if test results exist
        const testResultsPath = path.join(process.cwd(), 'test-results');
        if (fs.existsSync(testResultsPath)) {
            const files = fs.readdirSync(testResultsPath);
            console.log(`📄 Test artifacts created: ${files.length} files`);
        }
        
        // Check coverage results
        const coveragePath = path.join(process.cwd(), 'coverage');
        if (fs.existsSync(coveragePath)) {
            console.log('📊 Coverage reports generated in coverage/');
        }
    } catch (error) {
        console.log('⚠️ Error checking test artifacts:', error.message);
    }
    
    console.log('✅ Global Jest Teardown: Cleanup complete');
};