/**
 * Global Jest Setup
 * Runs once before all test suites
 */

module.exports = async () => {
    console.log('🌍 Global Jest Setup: Initializing A2A Test Environment');
    
    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.LOG_LEVEL = 'error';
    process.env.DISABLE_EXTERNAL_SERVICES = 'true';
    
    // Mock external service endpoints
    process.env.MOCK_SAML_ENDPOINT = 'http://localhost:4004/test/saml';
    process.env.MOCK_OAUTH_ENDPOINT = 'http://localhost:4004/test/oauth';
    process.env.MOCK_REDIS_URL = 'memory://test-cache';
    
    // Increase Node.js memory limit for tests
    if (!process.env.NODE_OPTIONS) {
        process.env.NODE_OPTIONS = '--max-old-space-size=4096';
    }
    
    // Create test directories if needed
    const fs = require('fs');
    const path = require('path');
    
    const testDirs = [
        'test-results',
        'coverage',
        'logs/test'
    ];
    
    testDirs.forEach(dir => {
        const fullPath = path.join(process.cwd(), dir);
        if (!fs.existsSync(fullPath)) {
            fs.mkdirSync(fullPath, { recursive: true });
        }
    });
    
    console.log('✅ Global Jest Setup: Environment configured');
};