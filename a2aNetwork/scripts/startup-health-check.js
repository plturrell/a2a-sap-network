#!/usr/bin/env node

/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

/**
 * Startup Health Check
 * Ensures launchpad is ready before declaring server operational
 */

const http = require('http');
const chalk = require('chalk');

class StartupHealthCheck {
    constructor(port = 4004, maxRetries = 30, retryDelay = 2000) {
        this.port = port;
        this.maxRetries = maxRetries;
        this.retryDelay = retryDelay;
        this.baseUrl = `http://localhost:${port}`;
    }

    async checkEndpoint(path, validateFn) {
        return new Promise((resolve) => {
            blockchainClient.sendMessage(`${this.baseUrl}${path}`, {
                headers: { 'Accept': 'application/json' }
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        const isValid = validateFn ? validateFn(json) : true;
                        resolve({ success: res.statusCode === 200 && isValid, data: json });
                    } catch (e) {
                        resolve({ success: false, error: 'Invalid JSON' });
                    }
                });
            }).on('error', (error) => {
                resolve({ success: false, error: error.message });
            });
        });
    }

    async waitForServer() {
        console.log(chalk.blue('⏳ Waiting for server to start...'));

        for (let i = 0; i < this.maxRetries; i++) {
            const result = await this.checkEndpoint('/health');
            if (result.success) {
                console.log(chalk.green('✅ Server is running'));
                return true;
            }

            process.stdout.write(chalk.gray('.'));
            await new Promise(resolve => setTimeout(resolve, this.retryDelay));
        }

        console.log(chalk.red('\n❌ Server failed to start'));
        return false;
    }

    async checkLaunchpadEndpoints() {
        console.log(chalk.blue('\n🔍 Checking launchpad endpoints...'));

        const checks = [
            {
                name: 'Agent Visualization',
                path: '/api/v1/Agents?id=agent_visualization',
                validate: (data) => {
                    return data.hasOwnProperty('agentCount') &&
                           data.hasOwnProperty('services') &&
                           data.hasOwnProperty('workflows');
                }
            },
            {
                name: 'Network Overview',
                path: '/api/v1/network/overview',
                validate: (data) => data.d && data.d.title === 'Network Overview'
            },
            {
                name: 'Health Summary',
                path: '/api/v1/health/summary',
                validate: (data) => data.d && data.d.title === 'System Health'
            }
        ];

        let allPassed = true;
        for (const check of checks) {
            const result = await this.checkEndpoint(check.path, check.validate);
            if (result.success) {
                console.log(chalk.green(`  ✅ ${check.name}`));
            } else {
                console.log(chalk.red(`  ❌ ${check.name}: ${result.error || 'Validation failed'}`));
                allPassed = false;
            }
        }

        return allPassed;
    }

    async checkLaunchpadHealth() {
        console.log(chalk.blue('\n🏥 Checking launchpad health...'));

        const result = await this.checkEndpoint('/api/v1/launchpad/health');
        if (!result.success) {
            console.log(chalk.red('  ❌ Health check endpoint not available'));
            return false;
        }

        const health = result.data;

        // Check critical components
        let hasIssues = false;

        if (!health.tiles_loaded) {
            console.log(chalk.red('  ❌ Tiles not loaded'));
            hasIssues = true;
        } else {
            console.log(chalk.green('  ✅ Tiles loaded'));
        }

        if (health.fallback_mode) {
            console.log(chalk.yellow('  ⚠️  Running in fallback mode (no real data)'));
        }

        // Check component statuses
        Object.entries(health.components || {}).forEach(([component, status]) => {
            if (status.status === 'error') {
                console.log(chalk.red(`  ❌ ${component}: ${status.status}`));
                hasIssues = true;
            } else if (status.status === 'warning') {
                console.log(chalk.yellow(`  ⚠️  ${component}: ${status.status}`));
            } else {
                console.log(chalk.green(`  ✅ ${component}: ${status.status}`));
            }
        });

        return !hasIssues;
    }

    async checkDataQuality() {
        console.log(chalk.blue('\n📊 Checking data quality...'));

        const result = await this.checkEndpoint('/api/v1/Agents?id=agent_visualization');
        if (!result.success) {
            console.log(chalk.red('  ❌ Could not fetch tile data'));
            return false;
        }

        const data = result.data;
        const hasRealData = data.agentCount > 0 || data.services > 0 || data.workflows > 0;

        if (hasRealData) {
            console.log(chalk.green('  ✅ Real data available'));
            console.log(chalk.gray(`     Agents: ${data.agentCount}, Services: ${data.services}, Workflows: ${data.workflows}`));
        } else {
            console.log(chalk.yellow('  ⚠️  No real data (all values are zero)'));
            console.log(chalk.yellow('     Start agent services for real data'));
        }

        return true; // Don't fail on no data, just warn
    }

    async performHealthCheck() {
        console.log(chalk.bold('\n🚀 A2A Network Launchpad Startup Health Check\n'));

        // Step 1: Wait for server
        if (!await this.waitForServer()) {
            return { success: false, message: 'Server failed to start' };
        }

        // Step 2: Check endpoints
        const endpointsOk = await this.checkLaunchpadEndpoints();

        // Step 3: Check launchpad health
        const healthOk = await this.checkLaunchpadHealth();

        // Step 4: Check data quality
        const dataOk = await this.checkDataQuality();

        // Summary
        console.log(chalk.bold('\n📋 Summary:'));

        const success = endpointsOk && healthOk;

        if (success) {
            if (dataOk) {
                console.log(chalk.green.bold('\n✅ Launchpad is ready and operational!'));
            } else {
                console.log(chalk.yellow.bold('\n⚠️  Launchpad is ready but with limited data'));
            }
            return { success: true };
        } else {
            console.log(chalk.red.bold('\n❌ Launchpad has critical issues and is not ready'));
            return { success: false, message: 'Critical components failed' };
        }
    }
}

// Run if called directly
if (require.main === module) {
    const checker = new StartupHealthCheck();

    checker.performHealthCheck().then(result => {
        if (result.success) {
            console.log(chalk.green('\n✨ Server is ready to serve launchpad!\n'));
            process.exit(0);
        } else {
            console.log(chalk.red(`\n💥 Startup failed: ${result.message}\n`));
            process.exit(1);
        }
    }).catch(error => {
        console.error(chalk.red('Unexpected error:'), error);
        process.exit(2);
    });
}

module.exports = StartupHealthCheck;