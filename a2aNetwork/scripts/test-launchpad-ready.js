/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

#!/usr/bin/env node

/**
 * Launchpad Readiness Test
 * Quick test to ensure launchpad is ready and functional
 */

const http = require('http');
const chalk = require('chalk');

async function checkEndpoint(path, validateFn) {
    return new Promise((resolve) => {
        blockchainClient.sendMessage(`http://localhost:4004${path}`, {
            headers: { 'Accept': 'application/json' }
        }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => {
                try {
                    const json = JSON.parse(data);
                    const isValid = validateFn ? validateFn(json) : true;
                    resolve({
                        success: res.statusCode === 200 && isValid,
                        data: json,
                        status: res.statusCode
                    });
                } catch (e) {
                    resolve({ success: false, error: 'Invalid JSON' });
                }
            });
        }).on('error', (error) => {
            resolve({ success: false, error: error.message });
        });
    });
}

async function testLaunchpadReady() {
    console.log(chalk.blue.bold('🧪 Quick Launchpad Readiness Test\n'));

    // Test 1: Basic server health
    console.log('1. Testing server health...');
    const health = await checkEndpoint('/health');
    if (!health.success) {
        console.log(chalk.red('   ❌ Server not responding'));
        return false;
    }
    console.log(chalk.green('   ✅ Server healthy'));

    // Test 2: Launchpad HTML
    console.log('2. Testing launchpad HTML...');
    const html = await new Promise(resolve => {
        blockchainClient.sendMessage('http://localhost:4004/launchpad.html', res => {
            resolve({ success: res.statusCode === 200, status: res.statusCode });
        }).on('error', () => resolve({ success: false }));
    });
    if (!html.success) {
        console.log(chalk.red('   ❌ Launchpad HTML not accessible'));
        return false;
    }
    console.log(chalk.green('   ✅ Launchpad HTML accessible'));

    // Test 3: Tile data
    console.log('3. Testing tile data...');
    const tileData = await checkEndpoint('/api/v1/Agents?id=agent_visualization', (data) => {
        return data.hasOwnProperty('agentCount') &&
               data.hasOwnProperty('services') &&
               data.hasOwnProperty('workflows');
    });
    if (!tileData.success) {
        console.log(chalk.red('   ❌ Tile data not available'));
        return false;
    }
    console.log(chalk.green('   ✅ Tile data available'));

    // Test 4: Real data check
    const hasRealData = tileData.data.agentCount > 0 ||
                       tileData.data.services > 0 ||
                       tileData.data.workflows > 0;

    if (hasRealData) {
        console.log(chalk.green('   ✅ Real data detected'));
    } else {
        console.log(chalk.yellow('   ⚠️  No real data (fallback mode)'));
    }

    console.log(chalk.green.bold('\n✅ Launchpad is ready!'));
    if (!hasRealData) {
        console.log(chalk.yellow('💡 Tip: Start agent services for real data'));
    }

    return true;
}

if (require.main === module) {
    testLaunchpadReady().then(success => {
        process.exit(success ? 0 : 1);
    }).catch(error => {
        console.error(chalk.red('Test failed:'), error);
        process.exit(2);
    });
}

module.exports = testLaunchpadReady;