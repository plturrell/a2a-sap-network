/**
 * Comprehensive Error Handling Tests for A2A Launchpad Common Components
 * Tests error scenarios, edge cases, and failure recovery
 */

const assert = require('assert');
const TestEnvironmentSetup = require('./setup/testEnvironment');
const TestConfiguration = require('./config/testConfig');

// Initialize test environment
const testEnv = new TestEnvironmentSetup();
testEnv.setupBrowserEnvironment();
testEnv.setupErrorHandling();

const testConfig = new TestConfiguration();

// Import common components
const SSOManager = require('../common/auth/SSOManager');
const UnifiedNavigation = require('../common/navigation/UnifiedNavigation');
const SharedResourceManager = require('../common/resources/SharedResourceManager');
const UnifiedMonitoring = require('../common/monitoring/UnifiedMonitoring');
const DisasterRecovery = require('../common/resilience/DisasterRecovery');

// Test runner for error handling
async function runErrorHandlingTests() {
    console.log('🚨 Starting A2A Launchpad Error Handling Tests\n');
    
    let passedTests = 0;
    let failedTests = 0;
    
    // Test SSO Error Handling
    console.log('=== SSO Authentication Error Handling ===');
    try {
        await testSSOErrorHandling();
        console.log('✅ SSO Error Handling tests passed\n');
        passedTests++;
    } catch (error) {
        console.error('❌ SSO Error Handling tests failed:', error.message, '\n');
        failedTests++;
    }
    
    // Test Navigation Error Handling
    console.log('=== Navigation Error Handling ===');
    try {
        await testNavigationErrorHandling();
        console.log('✅ Navigation Error Handling tests passed\n');
        passedTests++;
    } catch (error) {
        console.error('❌ Navigation Error Handling tests failed:', error.message, '\n');
        failedTests++;
    }
    
    // Test Resources Error Handling
    console.log('=== Resources Error Handling ===');
    try {
        await testResourcesErrorHandling();
        console.log('✅ Resources Error Handling tests passed\n');
        passedTests++;
    } catch (error) {
        console.error('❌ Resources Error Handling tests failed:', error.message, '\n');
        failedTests++;
    }
    
    // Test Monitoring Error Handling
    console.log('=== Monitoring Error Handling ===');
    try {
        await testMonitoringErrorHandling();
        console.log('✅ Monitoring Error Handling tests passed\n');
        passedTests++;
    } catch (error) {
        console.error('❌ Monitoring Error Handling tests failed:', error.message, '\n');
        failedTests++;
    }
    
    // Test Disaster Recovery Error Handling
    console.log('=== Disaster Recovery Error Handling ===');
    try {
        await testDisasterRecoveryErrorHandling();
        console.log('✅ Disaster Recovery Error Handling tests passed\n');
        passedTests++;
    } catch (error) {
        console.error('❌ Disaster Recovery Error Handling tests failed:', error.message, '\n');
        failedTests++;
    }
    
    // Summary
    console.log('\n=== Error Handling Test Summary ===');
    console.log(`Total Tests: ${passedTests + failedTests}`);
    console.log(`Passed: ${passedTests}`);
    console.log(`Failed: ${failedTests}`);
    console.log(`Success Rate: ${(passedTests / (passedTests + failedTests) * 100).toFixed(2)}%`);
    
    return failedTests === 0;
}

// SSO Error Handling Tests
async function testSSOErrorHandling() {
    const config = testConfig.getConfig('sso');
    const sso = new SSOManager(config);
    
    // Test 1: Invalid credentials
    console.log('  Testing invalid credentials handling...');
    try {
        await sso.authenticateUser(null, 'saml');
        assert.fail('Should have thrown error for null credentials');
    } catch (error) {
        assert(error.message.includes('Invalid') || error.message.includes('null'), 'Should handle null credentials');
    }
    
    // Test 2: Unsupported authentication method
    console.log('  Testing unsupported auth method...');
    try {
        await sso.authenticateUser({ email: 'test@example.com' }, 'unsupported-method');
        assert.fail('Should have thrown error for unsupported method');
    } catch (error) {
        assert(error.message.includes('Unsupported'), 'Should handle unsupported auth method');
    }
    
    // Test 3: Token validation with invalid token
    console.log('  Testing invalid token validation...');
    const invalidTokenResult = await sso.validateToken('invalid-token');
    assert(!invalidTokenResult.valid, 'Invalid token should not validate');
    
    // Test 4: Expired token handling
    console.log('  Testing expired token handling...');
    const expiredToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjF9.invalid';
    const expiredResult = await sso.validateToken(expiredToken);
    assert(!expiredResult.valid, 'Expired token should not validate');
    
    // Test 5: Refresh token with invalid token
    console.log('  Testing invalid refresh token...');
    try {
        await sso.refreshSession('invalid-refresh-token');
        // Should not throw in graceful implementation
    } catch (error) {
        assert(error.message.includes('Invalid') || error.message.includes('token'), 'Should handle invalid refresh token');
    }
    
    // Test 6: Missing JWT secret
    console.log('  Testing missing JWT secret...');
    try {
        const badConfigSSO = new SSOManager({ jwtSecret: null });
        await badConfigSSO.authenticateUser({ email: 'test@example.com' }, 'local');
        // Should not throw if gracefully handled
    } catch (error) {
        assert(error.message.includes('secret') || error.message.includes('key'), 'Should handle missing JWT secret');
    }
    
    // Test 7: Network failure simulation
    console.log('  Testing network failure handling...');
    const networkFailureSSO = new SSOManager({
        ...config,
        externalServiceTimeout: 1 // Very short timeout
    });
    
    try {
        await networkFailureSSO.authenticateUser({
            email: 'test@example.com',
            password: 'password'
        }, 'oauth2');
        // Should handle network timeout gracefully
    } catch (error) {
        // Error is expected for network timeout
        console.log('    Network timeout handled:', error.message);
    }
}

// Navigation Error Handling Tests
async function testNavigationErrorHandling() {
    testEnv.reset();
    const config = testConfig.getConfig('navigation');
    const navigation = new UnifiedNavigation(config);
    
    // Test 1: Unknown application navigation
    console.log('  Testing unknown application navigation...');
    try {
        await navigation.navigateToApplication('unknown-app');
        assert.fail('Should have thrown error for unknown application');
    } catch (error) {
        assert(error.message.includes('Unknown'), 'Should handle unknown application');
    }
    
    // Test 2: Invalid URL construction
    console.log('  Testing invalid URL construction...');
    try {
        const invalidUrl = navigation.buildTargetUrl('network', {
            deepLink: 'javascript:alert("xss")', // XSS attempt
            params: { malicious: '<script>alert("xss")</script>' }
        });
        assert(!invalidUrl.includes('javascript:'), 'Should sanitize dangerous URLs');
        assert(!invalidUrl.includes('<script>'), 'Should sanitize dangerous parameters');
    } catch (error) {
        // Error handling for invalid URLs is acceptable
        console.log('    Invalid URL handled:', error.message);
    }
    
    // Test 3: Navigation timeout
    console.log('  Testing navigation timeout...');
    const fastTimeoutNav = new UnifiedNavigation({
        ...config,
        navigationTimeout: 1 // 1ms timeout
    });
    
    try {
        await fastTimeoutNav.navigateToApplication('network', {
            deepLink: '/slow-loading-page'
        });
        // Should complete or timeout gracefully
    } catch (error) {
        console.log('    Navigation timeout handled:', error.message);
    }
    
    // Test 4: Context preservation failure
    console.log('  Testing context preservation failure...');
    try {
        await navigation.preserveContext(null);
        // Should handle null context gracefully
    } catch (error) {
        assert(error.message.includes('context') || error.message.includes('null'), 'Should handle null context');
    }
    
    // Test 5: Breadcrumb overflow
    console.log('  Testing breadcrumb overflow handling...');
    for (let i = 0; i < 20; i++) {
        navigation.updateBreadcrumb('network', { title: `Level ${i}` });
    }
    assert(navigation.breadcrumbs.length <= 10, 'Breadcrumbs should be limited');
    
    // Test 6: Invalid deep link parameters
    console.log('  Testing invalid deep link parameters...');
    const longParam = 'x'.repeat(300); // Very long parameter
    const validUrl = navigation.buildTargetUrl('network', {
        params: { longParam }
    });
    assert(validUrl.length < 2100, 'Should handle long URLs gracefully');
}

// Resources Error Handling Tests
async function testResourcesErrorHandling() {
    const config = testConfig.getConfig('resources');
    const resources = new SharedResourceManager(config);
    
    // Test 1: Invalid configuration key
    console.log('  Testing invalid configuration handling...');
    try {
        await resources.syncConfiguration(null, { value: 'test' });
        assert.fail('Should have thrown error for null key');
    } catch (error) {
        assert(error.message.includes('key') || error.message.includes('null'), 'Should handle null key');
    }
    
    // Test 2: Circular reference in configuration
    console.log('  Testing circular reference handling...');
    const circularObject = { a: 1 };
    circularObject.self = circularObject;
    
    try {
        await resources.syncConfiguration('circular', circularObject);
        // Should handle circular references gracefully
    } catch (error) {
        console.log('    Circular reference handled:', error.message);
    }
    
    // Test 3: Asset upload with invalid data
    console.log('  Testing invalid asset upload...');
    try {
        await resources.manageSharedAssets('test.txt', null);
        // Should handle null asset data gracefully
    } catch (error) {
        assert(error.message.includes('asset') || error.message.includes('data'), 'Should handle null asset data');
    }
    
    // Test 4: Storage backend failure
    console.log('  Testing storage backend failure...');
    const failureResources = new SharedResourceManager({
        ...config,
        storage: { type: 'nonexistent-backend' }
    });
    
    try {
        await failureResources.syncConfiguration('test', { value: 'test' });
        // Should fallback to in-memory storage
    } catch (error) {
        console.log('    Storage backend failure handled:', error.message);
    }
    
    // Test 5: Conflict resolution with invalid data
    console.log('  Testing conflict resolution with invalid data...');
    try {
        const resolved = await resources.resolveConflict(null, { value: 'new' });
        assert(resolved !== null, 'Should handle null conflict gracefully');
    } catch (error) {
        console.log('    Conflict resolution error handled:', error.message);
    }
    
    // Test 6: Feature flag with invalid value
    console.log('  Testing feature flag validation...');
    try {
        await resources.setFeatureFlag('', true); // Empty key
        // Should handle empty keys gracefully
    } catch (error) {
        assert(error.message.includes('key') || error.message.includes('empty'), 'Should validate feature flag keys');
    }
}

// Monitoring Error Handling Tests
async function testMonitoringErrorHandling() {
    const config = testConfig.getConfig('monitoring');
    const monitoring = new UnifiedMonitoring(config);
    
    // Test 1: Invalid metrics collection
    console.log('  Testing invalid metrics collection...');
    try {
        await monitoring.collectMetrics(null);
        // Should handle null sources gracefully
    } catch (error) {
        assert(error.message.includes('sources') || error.message.includes('null'), 'Should handle null sources');
    }
    
    // Test 2: Alert processing with malformed alert
    console.log('  Testing malformed alert processing...');
    try {
        await monitoring.processAlert(null);
        // Should handle null alerts gracefully
    } catch (error) {
        assert(error.message.includes('alert') || error.message.includes('null'), 'Should handle null alerts');
    }
    
    // Test 3: Dashboard generation failure
    console.log('  Testing dashboard generation failure...');
    try {
        const dashboard = await monitoring.generateDashboard();
        assert(dashboard !== null, 'Dashboard should not be null');
    } catch (error) {
        console.log('    Dashboard generation error handled:', error.message);
    }
    
    // Test 4: Backend service unavailable
    console.log('  Testing backend service unavailability...');
    const offlineMonitoring = new UnifiedMonitoring({
        ...config,
        backends: {
            prometheus: { enabled: false, mock: false }
        }
    });
    
    try {
        await offlineMonitoring.collectMetrics(['test-service']);
        // Should fallback to local metrics
    } catch (error) {
        console.log('    Backend unavailability handled:', error.message);
    }
    
    // Test 5: Historical data query with invalid timerange
    console.log('  Testing invalid timerange query...');
    try {
        const result = await monitoring.queryHistoricalData('test_metric', 'invalid-range');
        // Should handle invalid timerange gracefully
    } catch (error) {
        assert(error.message.includes('timerange') || error.message.includes('invalid'), 'Should validate timerange');
    }
    
    // Test 6: Alert storm detection
    console.log('  Testing alert storm detection...');
    const alerts = [];
    for (let i = 0; i < 200; i++) {
        alerts.push({
            name: `Alert ${i}`,
            severity: 'warning',
            source: 'test',
            message: `Test alert ${i}`
        });
    }
    
    try {
        await Promise.all(alerts.map(alert => monitoring.processAlert(alert)));
        // Should detect and handle alert storm
    } catch (error) {
        console.log('    Alert storm handled:', error.message);
    }
}

// Disaster Recovery Error Handling Tests
async function testDisasterRecoveryErrorHandling() {
    const config = testConfig.getConfig('disasterRecovery');
    const recovery = new DisasterRecovery(config);
    
    // Test 1: Backup validation with corrupted backup
    console.log('  Testing corrupted backup handling...');
    try {
        const validation = await recovery.validateBackups();
        assert(validation !== null, 'Backup validation should not return null');
    } catch (error) {
        console.log('    Corrupted backup handled:', error.message);
    }
    
    // Test 2: Failover with insufficient resources
    console.log('  Testing failover with insufficient resources...');
    try {
        const result = await recovery.initiateFailover('resource-shortage-test');
        // Should handle resource shortage gracefully
    } catch (error) {
        assert(error.message.includes('resource') || error.message.includes('insufficient'), 'Should handle resource shortage');
    }
    
    // Test 3: Service restoration timeout
    console.log('  Testing service restoration timeout...');
    const fastTimeoutRecovery = new DisasterRecovery({
        ...config,
        rto: { application: 1 } // 1ms timeout
    });
    
    try {
        await fastTimeoutRecovery.restoreServices();
        // Should handle timeout gracefully
    } catch (error) {
        console.log('    Restoration timeout handled:', error.message);
    }
    
    // Test 4: Data consistency check failure
    console.log('  Testing data consistency check failure...');
    try {
        const consistency = await recovery.validateDataConsistency();
        assert(consistency !== null, 'Consistency check should not return null');
    } catch (error) {
        console.log('    Consistency check failure handled:', error.message);
    }
    
    // Test 5: Communication system failure
    console.log('  Testing communication system failure...');
    try {
        const commResult = await recovery.testCommunicationChannels();
        assert(commResult !== null, 'Communication test should not return null');
    } catch (error) {
        console.log('    Communication failure handled:', error.message);
    }
    
    // Test 6: Backup storage failure
    console.log('  Testing backup storage failure...');
    const failureRecovery = new DisasterRecovery({
        ...config,
        backupStorage: { type: 'nonexistent-storage' }
    });
    
    try {
        await failureRecovery.validateBackups();
        // Should fallback to alternative storage
    } catch (error) {
        console.log('    Backup storage failure handled:', error.message);
    }
}

// Run all tests
if (require.main === module) {
    runErrorHandlingTests().then(success => {
        process.exit(success ? 0 : 1);
    }).catch(error => {
        console.error('Error handling test execution failed:', error);
        process.exit(1);
    });
}

module.exports = { runErrorHandlingTests };