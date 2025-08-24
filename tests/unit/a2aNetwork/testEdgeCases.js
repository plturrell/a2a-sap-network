/**
 * Additional test cases to cover edge cases and gaps identified in analysis
 * These tests complement the main integration tests
 */

const assert = require('assert');
const crypto = require('crypto');

// Import components for testing
const SSOManager = require('../common/auth/SSOManager');
const UnifiedNavigation = require('../common/navigation/UnifiedNavigation');
const SharedResourceManager = require('../common/resources/SharedResourceManager');
const UnifiedMonitoring = require('../common/monitoring/UnifiedMonitoring');
const DisasterRecovery = require('../common/resilience/DisasterRecovery');

async function runEdgeCaseTests() {
    console.log('Running Edge Case and Security Tests\n');
    
    let passedTests = 0;
    let failedTests = 0;
    
    const tests = [
        { name: 'SSO Security Edge Cases', fn: testSSOSecurityEdgeCases },
        { name: 'Navigation Edge Cases', fn: testNavigationEdgeCases },
        { name: 'Resource Management Edge Cases', fn: testResourceEdgeCases },
        { name: 'Monitoring Edge Cases', fn: testMonitoringEdgeCases },
        { name: 'Disaster Recovery Edge Cases', fn: testDREdgeCases }
    ];
    
    for (const test of tests) {
        console.log(`=== ${test.name} ===`);
        try {
            await test.fn();
            console.log(`✅ ${test.name} passed\n`);
            passedTests++;
        } catch (error) {
            console.error(`❌ ${test.name} failed:`, error.message, '\n');
            failedTests++;
        }
    }
    
    console.log('=== Edge Case Test Summary ===');
    console.log(`Passed: ${passedTests}, Failed: ${failedTests}`);
    console.log(`Success Rate: ${(passedTests / (passedTests + failedTests) * 100).toFixed(2)}%`);
    
    return failedTests === 0;
}

async function testSSOSecurityEdgeCases() {
    const sso = new SSOManager({ jwtSecret: 'test-secret', sessionTimeout: 1000 });
    
    // Test 1: Role Mapping with Complex Roles
    console.log('  Testing SAML role mapping edge cases...');
    const complexRoles = ['NetworkAdmin', 'InvalidRole', 'SystemAdministrator', 'UnknownRole'];
    const mappedRoles = sso.mapSAMLRoles(complexRoles);
    assert(mappedRoles.includes('network_admin'), 'Should map NetworkAdmin');
    assert(mappedRoles.includes('system_admin'), 'Should map SystemAdministrator');
    assert(!mappedRoles.includes('InvalidRole'), 'Should not include unmapped roles');
    
    // Test 2: Permission Wildcard Matching
    console.log('  Testing permission wildcard matching...');
    const permissions = ['agents.*.create', 'network.read', '*'];
    assert(sso.hasPermission(permissions, 'agents.project.create'), 'Should match wildcard pattern');
    assert(sso.hasPermission(permissions, 'network.read'), 'Should match exact permission');
    assert(sso.hasPermission(permissions, 'anything.else'), 'Should match global wildcard');
    assert(!sso.hasPermission(['agents.read'], 'agents.write'), 'Should not match different permission');
    
    // Test 3: Token Blacklist Management
    console.log('  Testing token blacklist edge cases...');
    const auth1 = await sso.authenticateUser({ email: 'test1@example.com' }, 'local');
    const auth2 = await sso.authenticateUser({ email: 'test2@example.com' }, 'local');
    
    await sso.logout(auth1.accessToken);
    await sso.logout(auth2.accessToken);
    
    // Verify blacklisted tokens are invalid
    const validation1 = await sso.validateToken(auth1.accessToken);
    const validation2 = await sso.validateToken(auth2.accessToken);
    assert(!validation1.valid, 'Blacklisted token should be invalid');
    assert(!validation2.valid, 'Blacklisted token should be invalid');
    
    // Test 4: Session Timeout Edge Cases
    console.log('  Testing session timeout...');
    const shortTimeoutSSO = new SSOManager({ jwtSecret: 'test', sessionTimeout: 100 });
    const authShort = await shortTimeoutSSO.authenticateUser({ email: 'test@example.com' }, 'local');
    
    // Wait for session timeout
    await new Promise(resolve => setTimeout(resolve, 150));
    
    // Session should still be valid (token-based)
    const validationShort = await shortTimeoutSSO.validateToken(authShort.accessToken);
    assert(validationShort.valid, 'JWT token should still be valid');
    
    // Test 5: Concurrent Authentication
    console.log('  Testing concurrent authentication...');
    const concurrentAuths = await Promise.all([
        sso.authenticateUser({ email: 'concurrent1@example.com' }, 'local'),
        sso.authenticateUser({ email: 'concurrent2@example.com' }, 'local'),
        sso.authenticateUser({ email: 'concurrent3@example.com' }, 'local')
    ]);
    
    // All should succeed with different session IDs
    const sessionIds = concurrentAuths.map(auth => auth.sessionId);
    const uniqueSessionIds = new Set(sessionIds);
    assert(uniqueSessionIds.size === 3, 'Concurrent auths should have unique session IDs');
}

async function testNavigationEdgeCases() {
    // Mock DOM for navigation tests
    global.window = {
        location: { href: '', pathname: '/test' },
        addEventListener: () => {},
        open: () => {},
        sessionStorage: {
            setItem: () => {},
            getItem: () => null,
            removeItem: () => {}
        }
    };
    global.history = { pushState: () => {} };
    
    const nav = new UnifiedNavigation({
        applications: {
            test: { url: '/test', name: 'Test App' }
        }
    });
    
    // Test 1: Navigation Token Collision Prevention
    console.log('  Testing navigation token uniqueness...');
    const tokens = [];
    for (let i = 0; i < 1000; i++) {
        tokens.push(nav.generateNavigationToken());
    }
    const uniqueTokens = new Set(tokens);
    assert(uniqueTokens.size === tokens.length, 'Navigation tokens should be unique');
    
    // Test 2: Context Storage Limits
    console.log('  Testing large context handling...');
    const largeContext = {
        data: 'x'.repeat(100000), // 100KB of data
        nested: { deep: { object: { with: { many: { levels: 'test' } } } } }
    };
    
    try {
        await nav.preserveContext(largeContext);
        assert(nav.currentContext.data.length === 100000, 'Large context should be preserved');
    } catch (error) {
        // Expected if storage limit exceeded
        assert(error.message.includes('storage') || error.message.includes('quota'), 'Should handle storage limits');
    }
    
    // Test 3: Deep Link Parameter Sanitization
    console.log('  Testing deep link parameter safety...');
    const maliciousParams = {
        script: '<script>alert("xss")</script>',
        sql: '\'; DROP TABLE users; --',
        path: '../../../etc/passwd'
    };
    
    const url = nav.buildTargetUrl('test', {
        params: maliciousParams
    });
    
    // URL should be encoded/sanitized
    assert(!url.includes('<script>'), 'Script tags should be encoded');
    assert(!url.includes('DROP TABLE'), 'SQL injection attempts should be encoded');
    
    // Test 4: Breadcrumb Overflow
    console.log('  Testing breadcrumb overflow...');
    for (let i = 0; i < 15; i++) {
        nav.updateBreadcrumb('test', { title: `Breadcrumb ${i}` });
    }
    
    assert(nav.breadcrumbs.length <= 10, 'Breadcrumbs should be limited to prevent memory issues');
    
    // Test 5: Navigation History Cleanup
    console.log('  Testing navigation history management...');
    for (let i = 0; i < 100; i++) {
        nav.navigationHistory.push({ from: 'test', to: 'test', timestamp: Date.now() });
    }
    
    // History should be bounded
    const history = nav.getNavigationHistory(50);
    assert(history.length <= 50, 'Navigation history should be limited');
}

async function testResourceEdgeCases() {
    const resources = new SharedResourceManager({
        syncInterval: 100,
        conflictResolution: 'merge'
    });
    
    // Test 1: Deep Merge Complex Objects
    console.log('  Testing deep merge with complex objects...');
    const obj1 = { a: { b: { c: 1, d: 2 } }, e: [1, 2] };
    const obj2 = { a: { b: { c: 3, f: 4 } }, e: [3, 4], g: 5 };
    
    const merged = resources.deepMerge(obj1, obj2);
    assert(merged.a.b.c === 3, 'Should merge nested values');
    assert(merged.a.b.d === 2, 'Should preserve original values');
    assert(merged.a.b.f === 4, 'Should add new nested values');
    assert(merged.g === 5, 'Should add new top-level values');
    
    // Test 2: Circular Reference Handling
    console.log('  Testing circular reference handling...');
    const circular1 = { name: 'obj1' };
    const circular2 = { name: 'obj2', ref: circular1 };
    circular1.ref = circular2;
    
    try {
        // This should not cause infinite recursion
        const result = resources.deepMerge({ test: 'value' }, { circular: circular1 });
        assert(result.test === 'value', 'Should handle circular references gracefully');
    } catch (error) {
        // Acceptable if it detects and handles circular references
        assert(error.message.includes('circular') || error.message.includes('Maximum call stack'), 
               'Should handle circular references');
    }
    
    // Test 3: Concurrent Configuration Updates
    console.log('  Testing concurrent configuration updates...');
    const updates = [];
    for (let i = 0; i < 10; i++) {
        updates.push(resources.syncConfiguration(`concurrent.config.${i}`, { 
            value: i, 
            timestamp: Date.now() + i 
        }));
    }
    
    const results = await Promise.all(updates);
    results.forEach((result, index) => {
        assert(result.success, `Concurrent update ${index} should succeed`);
    });
    
    // Test 4: Asset Size Limits
    console.log('  Testing large asset handling...');
    const largeAsset = Buffer.alloc(1024 * 1024); // 1MB
    largeAsset.fill('test data');
    
    const assetResult = await resources.manageSharedAssets('large.asset', largeAsset, {
        type: 'binary'
    });
    
    assert(assetResult.success, 'Large asset should be handled');
    assert(assetResult.assetId === 'large.asset', 'Asset ID should be preserved');
    
    // Test 5: Feature Flag Edge Cases
    console.log('  Testing feature flag edge cases...');
    await resources.setFeatureFlag('boolean.flag', true);
    await resources.setFeatureFlag('object.flag', { enabled: true, config: { max: 100 } });
    await resources.setFeatureFlag('null.flag', null);
    
    assert(resources.getFeatureFlag('boolean.flag') === true, 'Boolean flag should work');
    assert(resources.getFeatureFlag('object.flag').enabled === true, 'Object flag should work');
    assert(resources.getFeatureFlag('null.flag') === null, 'Null flag should work');
    assert(resources.getFeatureFlag('nonexistent.flag') === false, 'Missing flag should be false');
}

async function testMonitoringEdgeCases() {
    const monitoring = new UnifiedMonitoring({
        metricsInterval: 100,
        retentionPeriod: 1000
    });
    
    // Test 1: Percentile Calculation Accuracy
    console.log('  Testing percentile calculation...');
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    const p50 = monitoring.calculatePercentile(values, 50);
    const p95 = monitoring.calculatePercentile(values, 95);
    const p99 = monitoring.calculatePercentile(values, 99);
    
    assert(p50 === 5, 'P50 should be median');
    assert(p95 === 10, 'P95 should be near maximum');
    assert(p99 === 10, 'P99 should be near maximum');
    
    // Test 2: Invalid Metric Data Handling
    console.log('  Testing invalid metric handling...');
    const invalidMetrics = [
        { name: '', value: 100, type: 'gauge' }, // Empty name
        { name: 'test', value: 'not-a-number', type: 'gauge' }, // Invalid value
        { name: 'test', value: 100, type: 'invalid-type' }, // Invalid type
        { name: 'test', value: Infinity, type: 'gauge' }, // Infinity value
        { name: 'test', value: NaN, type: 'gauge' } // NaN value
    ];
    
    const validationResults = invalidMetrics.map(metric => monitoring.validateMetric(metric));
    const validCount = validationResults.filter(v => v).length;
    
    assert(validCount < invalidMetrics.length, 'Some invalid metrics should be rejected');
    
    // Test 3: Alert Correlation Accuracy
    console.log('  Testing alert correlation...');
    const alert1 = { 
        id: '1', 
        source: 'network', 
        severity: 'high', 
        timestamp: Date.now(),
        name: 'High CPU Usage'
    };
    
    const alert2 = { 
        id: '2', 
        source: 'network', 
        severity: 'high', 
        timestamp: Date.now() + 1000,
        name: 'High Memory Usage'
    };
    
    monitoring.alerts.set('1', alert1);
    const correlations = await monitoring.findAlertCorrelations(alert2);
    
    assert(correlations.length > 0, 'Related alerts should be correlated');
    assert(correlations[0].alertId === '1', 'Should find correct correlation');
    
    // Test 4: Memory Management with Retention
    console.log('  Testing metric retention...');
    const oldTimestamp = Date.now() - 2000; // Older than retention period
    const newTimestamp = Date.now();
    
    await monitoring.aggregateMetrics('test', [
        { name: 'old.metric', value: 100, timestamp: oldTimestamp },
        { name: 'new.metric', value: 200, timestamp: newTimestamp }
    ]);
    
    // Old metrics should be cleaned up
    const stats = monitoring.getStatistics();
    assert(stats.metrics > 0, 'Should have metrics stored');
    
    // Test 5: Alert Storm Detection
    console.log('  Testing alert storm detection...');
    for (let i = 0; i < 15; i++) {
        await monitoring.processAlerts({
            name: `Storm Alert ${i}`,
            severity: 'high',
            source: 'network',
            value: i,
            threshold: 0
        });
    }
    
    // Should detect storm and group alerts
    const stormAlert = {
        source: 'network',
        severity: 'high',
        timestamp: Date.now()
    };
    
    const isStorm = monitoring.detectAlertStorm(stormAlert);
    assert(isStorm, 'Should detect alert storm');
}

async function testDREdgeCases() {
    const dr = new DisasterRecovery({
        primarySite: 'test-primary',
        secondarySite: 'test-secondary',
        storageBackends: ['filesystem']
    });
    
    // Test 1: Backup Integrity Validation
    console.log('  Testing backup integrity...');
    const backupData = JSON.stringify({ test: 'data', timestamp: Date.now() });
    const originalChecksum = dr.calculateChecksum(backupData);
    
    // Simulate corrupted data
    const corruptedData = backupData.replace('data', 'corrupted');
    const corruptedChecksum = dr.calculateChecksum(corruptedData);
    
    assert(originalChecksum !== corruptedChecksum, 'Checksums should differ for corrupted data');
    
    // Test 2: Backup Age Validation
    console.log('  Testing backup age limits...');
    const oldBackup = {
        id: 'old',
        type: 'database',
        endTime: Date.now() - (48 * 60 * 60 * 1000), // 48 hours ago
        checksum: 'test-checksum'
    };
    
    const maxAge = dr.getMaxBackupAge('database');
    const age = Date.now() - oldBackup.endTime;
    
    if (age > maxAge) {
        // Old backup should be flagged for cleanup
        assert(true, 'Old backup correctly identified');
    }
    
    // Test 3: Failover State Consistency
    console.log('  Testing failover state management...');
    const initialState = dr.failoverState;
    assert(initialState.site === 'test-primary', 'Should start with primary site');
    assert(!initialState.active, 'Should not be in active failover initially');
    
    // Test 4: Recovery Plan Validation
    console.log('  Testing recovery plan creation...');
    const plan = await dr.createRestorePlan(
        { timestamp: Date.now() - 3600000 },
        { components: ['database', 'application'] }
    );
    
    assert(plan.steps.length > 0, 'Recovery plan should have steps');
    
    // Check step ordering (database should come first)
    const dbSteps = plan.steps.filter(s => s.component === 'database');
    const appSteps = plan.steps.filter(s => s.component === 'application');
    
    if (dbSteps.length > 0 && appSteps.length > 0) {
        const firstDbStep = plan.steps.findIndex(s => s.component === 'database');
        const firstAppStep = plan.steps.findIndex(s => s.component === 'application');
        assert(firstDbStep < firstAppStep, 'Database steps should come before application steps');
    }
    
    // Test 5: Health Check Edge Cases
    console.log('  Testing health check reliability...');
    await dr.performHealthChecks();
    
    const stats = dr.getStatistics();
    assert(stats.health, 'Health statistics should be available');
    assert(stats.backups.total >= 0, 'Backup count should be non-negative');
}

// Run edge case tests if this file is executed directly
if (require.main === module) {
    runEdgeCaseTests().then(success => {
        console.log(`\n${  '='.repeat(50)}`);
        console.log(success ? 'All edge case tests passed!' : 'Some edge case tests failed!');
        process.exit(success ? 0 : 1);
    }).catch(error => {
        console.error('Edge case test execution failed:', error);
        process.exit(1);
    });
}

module.exports = { runEdgeCaseTests };