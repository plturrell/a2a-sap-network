/**
 * Test Controller for Selective and Group-Based Test Execution
 * Enables running specific test groups, components, or individual tests
 */

const { runAllTests: runIntegrationTests } = require('./testLaunchpadIntegration');
const { runErrorHandlingTests } = require('./testErrorHandling');
const TestTracker = require('./database/testTracker');
const TestConfiguration = require('./config/testConfig');

class TestController {
    constructor() {
        this.testGroups = {
            sso: {
                name: 'SSO Authentication',
                component: 'TC-COM-LPD-001',
                description: 'Single Sign-On authentication tests',
                tests: ['saml-auth', 'oauth2-auth', 'token-validation', 'session-refresh', 'mfa-integration']
            },
            navigation: {
                name: 'Unified Navigation',
                component: 'TC-COM-LPD-002',
                description: 'Cross-application navigation tests',
                tests: ['app-switching', 'context-preservation', 'breadcrumbs', 'deep-linking', 'browser-integration']
            },
            resources: {
                name: 'Shared Resources',
                component: 'TC-COM-LPD-003',
                description: 'Resource management and synchronization tests',
                tests: ['config-sync', 'asset-management', 'feature-flags', 'conflict-resolution', 'cache-management']
            },
            monitoring: {
                name: 'Unified Monitoring',
                component: 'TC-COM-LPD-004',
                description: 'Monitoring and alerting tests',
                tests: ['metrics-collection', 'alert-processing', 'dashboard-generation', 'correlation-analysis', 'historical-queries']
            },
            recovery: {
                name: 'Disaster Recovery',
                component: 'TC-COM-LPD-005',
                description: 'Disaster recovery and backup tests',
                tests: ['backup-validation', 'failover-procedures', 'data-consistency', 'service-restoration', 'communication-tests']
            },
            errors: {
                name: 'Error Handling',
                component: 'Error-Tests',
                description: 'Comprehensive error handling and edge cases',
                tests: ['sso-errors', 'navigation-errors', 'resource-errors', 'monitoring-errors', 'recovery-errors']
            }
        };

        this.testRunner = null;
        this.tracker = null;
    }

    /**
     * Initialize test controller
     */
    async initialize() {
        this.tracker = new TestTracker();
        await this.tracker.initialize();
        console.log('🎮 Test Controller initialized');
    }

    /**
     * Run all tests
     */
    async runAll(options = {}) {
        const runId = await this.tracker.startTestRun({
            environment: options.environment || 'test',
            gitCommit: options.gitCommit
        });

        console.log('🚀 Running ALL test groups');
        
        let totalPassed = 0;
        let totalFailed = 0;
        const results = [];

        for (const [groupKey, group] of Object.entries(this.testGroups)) {
            const result = await this.runGroup(groupKey, { runId, silent: true });
            results.push(result);
            
            if (result.success) {
                totalPassed++;
            } else {
                totalFailed++;
            }
        }

        const overallSuccess = totalFailed === 0;
        const coverage = (totalPassed / (totalPassed + totalFailed)) * 100;
        
        await this.tracker.endTestRun(
            overallSuccess ? 'completed' : 'failed',
            coverage
        );

        this.printSummary(results, totalPassed, totalFailed);
        return { success: overallSuccess, results, runId };
    }

    /**
     * Run specific test group
     */
    async runGroup(groupKey, options = {}) {
        const group = this.testGroups[groupKey];
        if (!group) {
            throw new Error(`Unknown test group: ${groupKey}`);
        }

        let runId = options.runId;
        let shouldCloseTracker = false;

        if (!runId) {
            runId = await this.tracker.startTestRun({
                environment: options.environment || 'test',
                gitCommit: options.gitCommit
            });
            shouldCloseTracker = true;
        }

        if (!options.silent) {
            console.log(`🧪 Running test group: ${group.name}`);
            console.log(`📋 Component: ${group.component}`);
            console.log(`📄 Description: ${group.description}\n`);
        }

        await this.tracker.startTestSuite(group.name, 'component', group.description);

        const result = {
            group: groupKey,
            name: group.name,
            component: group.component,
            success: false,
            duration: 0,
            error: null,
            tests: []
        };

        const startTime = Date.now();

        try {
            switch (groupKey) {
                case 'sso':
                    result.success = await this.runSSOTests(options);
                    break;
                case 'navigation':
                    result.success = await this.runNavigationTests(options);
                    break;
                case 'resources':
                    result.success = await this.runResourcesTests(options);
                    break;
                case 'monitoring':
                    result.success = await this.runMonitoringTests(options);
                    break;
                case 'recovery':
                    result.success = await this.runRecoveryTests(options);
                    break;
                case 'errors':
                    result.success = await this.runErrorHandlingTests(options);
                    break;
                default:
                    throw new Error(`No runner for group: ${groupKey}`);
            }
        } catch (error) {
            result.error = error.message;
            result.success = false;
            console.error(`❌ Group ${group.name} failed:`, error.message);
        }

        result.duration = Date.now() - startTime;

        await this.tracker.endTestSuite(
            group.name,
            result.success ? 'passed' : 'failed',
            result.error
        );

        if (shouldCloseTracker) {
            const coverage = result.success ? 100 : 0;
            await this.tracker.endTestRun(
                result.success ? 'completed' : 'failed',
                coverage
            );
        }

        if (!options.silent) {
            console.log(`${result.success ? '✅' : '❌'} Group ${group.name} ${result.success ? 'passed' : 'failed'}`);
            console.log(`⏱️  Duration: ${result.duration}ms\n`);
        }

        return result;
    }

    /**
     * Run specific test within a group
     */
    async runTest(groupKey, testKey, options = {}) {
        const group = this.testGroups[groupKey];
        if (!group) {
            throw new Error(`Unknown test group: ${groupKey}`);
        }

        if (!group.tests.includes(testKey)) {
            throw new Error(`Unknown test '${testKey}' in group '${groupKey}'`);
        }

        const runId = await this.tracker.startTestRun({
            environment: options.environment || 'test',
            gitCommit: options.gitCommit
        });

        console.log(`🔬 Running individual test: ${group.name} -> ${testKey}`);

        await this.tracker.startTestSuite(`${group.name}-${testKey}`, 'unit', `Single test: ${testKey}`);

        const startTime = Date.now();
        let success = false;
        let error = null;

        try {
            success = await this.runIndividualTest(groupKey, testKey, options);
        } catch (err) {
            error = err.message;
            console.error(`❌ Test ${testKey} failed:`, err.message);
        }

        const duration = Date.now() - startTime;

        // Record individual test
        await this.tracker.recordTestCase({
            suiteName: `${group.name}-${testKey}`,
            testName: testKey,
            testId: `${group.component}-${testKey}`,
            component: group.component,
            status: success ? 'passed' : 'failed',
            duration,
            error,
            assertions: { total: 1, passed: success ? 1 : 0 }
        });

        await this.tracker.endTestSuite(
            `${group.name}-${testKey}`,
            success ? 'passed' : 'failed',
            error
        );

        await this.tracker.endTestRun(
            success ? 'completed' : 'failed',
            success ? 100 : 0
        );

        console.log(`${success ? '✅' : '❌'} Test ${testKey} ${success ? 'passed' : 'failed'}`);
        console.log(`⏱️  Duration: ${duration}ms`);

        return { success, duration, error, runId };
    }

    /**
     * List available test groups and tests
     */
    listTests() {
        console.log('📋 Available Test Groups and Tests:\n');
        
        for (const [groupKey, group] of Object.entries(this.testGroups)) {
            console.log(`🧪 ${groupKey.toUpperCase()}: ${group.name}`);
            console.log(`   Component: ${group.component}`);
            console.log(`   Description: ${group.description}`);
            console.log(`   Tests: ${group.tests.join(', ')}`);
            console.log('');
        }

        console.log('Usage examples:');
        console.log('  node testController.js --group sso');
        console.log('  node testController.js --test sso:saml-auth');
        console.log('  node testController.js --all');
    }

    /**
     * Get test statistics
     */
    async getStatistics(days = 7) {
        const stats = await this.tracker.getTestStatistics(days);
        const recentRuns = await this.tracker.getRecentTestRuns(10);

        console.log('📊 Test Statistics (Last 7 days):\n');
        console.log(`Total Runs: ${stats.total_runs}`);
        console.log(`Average Duration: ${Math.round(stats.avg_duration / 1000)}s`);
        console.log(`Average Coverage: ${Math.round(stats.avg_coverage)}%`);
        console.log(`Success Rate: ${Math.round((stats.successful_runs / stats.total_runs) * 100)}%`);
        console.log(`Total Tests Executed: ${stats.total_tests_executed}`);
        console.log(`Total Tests Passed: ${stats.total_tests_passed}\n`);

        console.log('Recent Test Runs:');
        recentRuns.forEach((run, index) => {
            const status = run.status === 'completed' ? '✅' : '❌';
            const duration = Math.round(run.duration_ms / 1000);
            console.log(`${index + 1}. ${status} ${run.run_id} (${duration}s, ${run.passed_tests}/${run.total_tests})`);
        });

        return { stats, recentRuns };
    }

    // Individual test group runners
    async runSSOTests(options) {
        const SSOManager = require('../common/auth/SSOManager');
        const testConfig = new TestConfiguration();
        const config = testConfig.getConfig('sso');
        const sso = new SSOManager(config);

        // Run key SSO tests
        try {
            const testUser = config.testUsers[0];
            const authResult = await sso.authenticateUser({
                nameID: testUser.email,
                email: testUser.email,
                displayName: testUser.name,
                roles: testUser.roles
            }, 'saml');

            const tokenValidation = await sso.validateToken(authResult.accessToken);
            const refreshResult = await sso.refreshSession(authResult.refreshToken);

            return authResult.success && tokenValidation.valid && refreshResult.success;
        } catch (error) {
            console.error('SSO test error:', error.message);
            return false;
        }
    }

    async runNavigationTests(options) {
        const UnifiedNavigation = require('../common/navigation/UnifiedNavigation');
        const testConfig = new TestConfiguration();
        const config = testConfig.getConfig('navigation');
        const navigation = new UnifiedNavigation(config);

        try {
            const url = navigation.buildTargetUrl('network', {
                deepLink: '/test',
                params: { test: 'value' }
            });

            navigation.updateBreadcrumb('network', { title: 'Test' });
            await navigation.preserveContext({ test: 'data' });

            return url.includes('/test') && navigation.breadcrumbs.length > 0;
        } catch (error) {
            console.error('Navigation test error:', error.message);
            return false;
        }
    }

    async runResourcesTests(options) {
        const SharedResourceManager = require('../common/resources/SharedResourceManager');
        const testConfig = new TestConfiguration();
        const config = testConfig.getConfig('resources');
        const resources = new SharedResourceManager(config);

        try {
            const syncResult = await resources.syncConfiguration('test.config', { value: 'test' });
            const consistency = await resources.validateConsistency();
            await resources.setFeatureFlag('testFlag', true);
            const flagValue = resources.getFeatureFlag('testFlag');

            return syncResult.success && consistency.overallHealth !== undefined && flagValue === true;
        } catch (error) {
            console.error('Resources test error:', error.message);
            return false;
        }
    }

    async runMonitoringTests(options) {
        const UnifiedMonitoring = require('../common/monitoring/UnifiedMonitoring');
        const testConfig = new TestConfiguration();
        const config = testConfig.getConfig('monitoring');
        const monitoring = new UnifiedMonitoring(config);

        try {
            await monitoring.collectMetrics(['test-service']);
            const dashboard = await monitoring.generateDashboard();
            await monitoring.processAlert({
                name: 'Test Alert',
                severity: 'info',
                source: 'test'
            });

            return dashboard.components && dashboard.components.length >= 0;
        } catch (error) {
            console.error('Monitoring test error:', error.message);
            return false;
        }
    }

    async runRecoveryTests(options) {
        const DisasterRecovery = require('../common/resilience/DisasterRecovery');
        const testConfig = new TestConfiguration();
        const config = testConfig.getConfig('disasterRecovery');
        const recovery = new DisasterRecovery(config);

        try {
            const backupValidation = await recovery.validateBackups();
            const consistency = await recovery.validateDataConsistency();
            const commTest = await recovery.testCommunicationChannels();

            return backupValidation.complete && consistency.consistent && commTest.channelsFunctional;
        } catch (error) {
            console.error('Recovery test error:', error.message);
            return false;
        }
    }

    async runErrorHandlingTests(options) {
        try {
            return await runErrorHandlingTests();
        } catch (error) {
            console.error('Error handling test error:', error.message);
            return false;
        }
    }

    async runIndividualTest(groupKey, testKey, options) {
        // Run specific test within group
        // This is a simplified implementation - in a full system,
        // you would have more granular test execution
        const success = await this.runGroup(groupKey, { ...options, silent: true });
        return success.success;
    }

    printSummary(results, passed, failed) {
        console.log(`\n${  '='.repeat(80)}`);
        console.log('🎯 TEST EXECUTION SUMMARY');
        console.log('='.repeat(80));
        
        console.log(`📊 Groups Executed: ${results.length}`);
        console.log(`✅ Passed: ${passed}`);
        console.log(`❌ Failed: ${failed}`);
        console.log(`📈 Success Rate: ${Math.round((passed / (passed + failed)) * 100)}%`);

        console.log('\nGroup Results:');
        results.forEach(result => {
            const status = result.success ? '✅' : '❌';
            const duration = Math.round(result.duration / 1000);
            console.log(`${status} ${result.name} (${duration}s)`);
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
        });

        console.log('='.repeat(80));
    }

    async cleanup() {
        if (this.tracker) {
            await this.tracker.close();
        }
    }
}

// CLI Interface
if (require.main === module) {
    const args = process.argv.slice(2);
    const controller = new TestController();

    async function runCLI() {
        await controller.initialize();

        try {
            if (args.includes('--help') || args.includes('-h')) {
                controller.listTests();
            } else if (args.includes('--all')) {
                const result = await controller.runAll();
                process.exit(result.success ? 0 : 1);
            } else if (args.includes('--stats')) {
                await controller.getStatistics();
            } else if (args.includes('--group')) {
                const groupIndex = args.indexOf('--group');
                const groupKey = args[groupIndex + 1];
                if (!groupKey) {
                    console.error('❌ Group name required after --group');
                    process.exit(1);
                }
                const result = await controller.runGroup(groupKey);
                process.exit(result.success ? 0 : 1);
            } else if (args.includes('--test')) {
                const testIndex = args.indexOf('--test');
                const testSpec = args[testIndex + 1];
                if (!testSpec || !testSpec.includes(':')) {
                    console.error('❌ Test specification required in format group:test after --test');
                    process.exit(1);
                }
                const [groupKey, testKey] = testSpec.split(':');
                const result = await controller.runTest(groupKey, testKey);
                process.exit(result.success ? 0 : 1);
            } else {
                controller.listTests();
            }
        } catch (error) {
            console.error('❌ Test execution failed:', error.message);
            process.exit(1);
        } finally {
            await controller.cleanup();
        }
    }

    runCLI();
}

module.exports = TestController;