/**
 * Jest Test Environment Setup
 * Configures comprehensive testing environment for A2A Launchpad Common Components
 */

const TestEnvironmentSetup = require('./setup/testEnvironment');
const TestConfiguration = require('./config/testConfig');

// Global test environment instance
let globalTestEnv = null;
let globalTestConfig = null;

/**
 * Setup before all tests
 */
beforeAll(() => {
    console.log('🚀 Setting up A2A Launchpad Test Environment');
    
    // Initialize test environment
    globalTestEnv = new TestEnvironmentSetup();
    globalTestEnv.setupBrowserEnvironment();
    globalTestEnv.setupErrorHandling();
    
    // Initialize test configuration
    globalTestConfig = new TestConfiguration();
    
    // Make available globally for tests
    global.testEnv = globalTestEnv;
    global.testConfig = globalTestConfig;
    
    // Increase timeout for integration tests
    jest.setTimeout(30000);
    
    console.log('✅ Test environment initialized');
});

/**
 * Setup before each test
 */
beforeEach(() => {
    // Reset test environment for clean state
    if (globalTestEnv) {
        globalTestEnv.reset();
    }
    
    // Reset test configuration
    if (globalTestConfig) {
        globalTestConfig.reset();
    }
    
    // Clear console captures for fresh test
    console.log('🔄 Test environment reset for new test');
});

/**
 * Cleanup after each test
 */
afterEach(() => {
    // Get mock data for debugging if test failed
    if (globalTestEnv && global.expect && expect.getState().assertionCalls === 0) {
        const mockData = globalTestEnv.getMockData();
        if (mockData.alerts.length > 0 || mockData.consoleOutput.length > 0) {
            console.log('📊 Test mock data:', {
                alerts: mockData.alerts.length,
                consoleOutput: mockData.consoleOutput.length,
                localStorage: Object.keys(mockData.localStorage).length,
                sessionStorage: Object.keys(mockData.sessionStorage).length
            });
        }
    }
});

/**
 * Cleanup after all tests
 */
afterAll(() => {
    console.log('🧹 Cleaning up A2A Launchpad Test Environment');
    
    // Cleanup test environment
    if (globalTestEnv) {
        globalTestEnv.cleanup();
    }
    
    // Clear global references
    delete global.testEnv;
    delete global.testConfig;
    
    console.log('✅ Test environment cleaned up');
});

/**
 * Global error handler for unhandled promise rejections
 */
process.on('unhandledRejection', (reason, promise) => {
    console.error('🚨 Unhandled Rejection at:', promise, 'reason:', reason);
});

/**
 * Global error handler for uncaught exceptions
 */
process.on('uncaughtException', (error) => {
    console.error('🚨 Uncaught Exception:', error);
});

// Export for use in tests
module.exports = {
    TestEnvironmentSetup,
    TestConfiguration
};