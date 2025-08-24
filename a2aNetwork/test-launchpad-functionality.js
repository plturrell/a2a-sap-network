/**
 * Test script to verify launchpad functionality
 */

const http = require('http');
const { promisify } = require('util');

async function testEndpoint(path, expectedFields = []) {
    return new Promise(function(resolve, reject) {
        const options = {
            hostname: 'localhost',
            port: 4004,
            path: path,
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        };

        const req = http.request(options, function(res) {
            let data = '';
            res.on('data', function(chunk) {
                data += chunk;
            });
            res.on('end', function() {
                try {
                    const result = JSON.parse(data);
                    const status = res.statusCode;
                    
                    // Check if expected fields exist
                    let fieldsCheck = expectedFields.length === 0 ? true : 
                        expectedFields.every(function(field) { return result.hasOwnProperty(field); });
                    
                    resolve({
                        path,
                        status,
                        success: status >= 200 && status < 300,
                        hasExpectedFields: fieldsCheck,
                        data: result,
                        error: null
                    });
                } catch (error) {
                    resolve({
                        path,
                        status: res.statusCode,
                        success: false,
                        hasExpectedFields: false,
                        data: data,
                        error: `JSON parse error: ${error.message}`
                    });
                }
            });
        });

        req.on('error', (error) => {
            resolve({
                path,
                status: 0,
                success: false,
                hasExpectedFields: false,
                data: null,
                error: `Request error: ${error.message}`
            });
        });

        req.setTimeout(5000, () => {
            req.destroy();
            resolve({
                path,
                status: 0,
                success: false,
                hasExpectedFields: false,
                data: null,
                error: 'Request timeout'
            });
        });

        req.end();
    });
}

async function runLaunchpadTests() {
    console.log('🧪 Testing A2A Launchpad Functionality\n');
    
    const tests = [
        // Agent endpoints
        {
            path: '/api/v1/Agents?id=agent_visualization',
            name: 'Agent Visualization Tile',
            expectedFields: ['id', 'title', 'data', 'status']
        },
        
        // Service endpoints  
        {
            path: '/api/v1/Services?id=service_marketplace',
            name: 'Service Marketplace Tile',
            expectedFields: ['id', 'title', 'data', 'status']
        },
        
        // Network health
        {
            path: '/api/v1/network/health',
            name: 'Network Health',
            expectedFields: ['healthScore', 'status', 'components']
        },
        
        // Notification count
        {
            path: '/api/v1/notifications/count',
            name: 'Notification Count',
            expectedFields: ['unreadCount', 'totalCount']
        },
        
        // Settings endpoints
        {
            path: '/api/v1/settings/network',
            name: 'Network Settings',
            expectedFields: ['network', 'rpcUrl', 'chainId']
        },
        
        // CDS Services
        {
            path: '/api/v1/network/Agents',
            name: 'CDS Agents Service',
            expectedFields: ['value']
        }
    ];
    
    const results = [];
    
    for (const test of tests) {
        console.log(`Testing: ${test.name}`);
        const result = await testEndpoint(test.path, test.expectedFields);
        results.push({ ...result, name: test.name });
        
        if (result.success && result.hasExpectedFields) {
            console.log(`✅ ${test.name} - PASSED`);
        } else {
            console.log(`❌ ${test.name} - FAILED`);
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
            if (!result.hasExpectedFields) {
                console.log(`   Missing expected fields: ${test.expectedFields.join(', ')}`);
            }
            console.log(`   Status: ${result.status}`);
        }
        console.log('');
    }
    
    // Summary
    const passed = results.filter(r => r.success && r.hasExpectedFields).length;
    const total = results.length;
    
    console.log('🏁 Test Summary');
    console.log('================');
    console.log(`Passed: ${passed}/${total}`);
    console.log(`Success Rate: ${Math.round((passed/total) * 100)}%`);
    
    if (passed === total) {
        console.log('\n🎉 All tests passed! Launchpad functionality is working correctly.');
    } else {
        console.log('\n⚠️  Some tests failed. Check the details above.');
    }
    
    // Show data samples for successful endpoints
    console.log('\n📊 Sample Data:');
    console.log('================');
    
    results.forEach(result => {
        if (result.success && result.data && typeof result.data === 'object') {
            console.log(`\n${result.name}:`);
            if (result.data.data) {
                // For tile endpoints, show the data section
                console.log(JSON.stringify(result.data.data, null, 2));
            } else {
                // For other endpoints, show first few fields
                const sample = {};
                const keys = Object.keys(result.data).slice(0, 3);
                keys.forEach(key => sample[key] = result.data[key]);
                console.log(JSON.stringify(sample, null, 2));
            }
        }
    });
    
    return { passed, total, results };
}

// Run tests
runLaunchpadTests()
    .then(({ passed, total }) => {
        process.exit(passed === total ? 0 : 1);
    })
    .catch(error => {
        console.error('Test execution failed:', error);
        process.exit(1);
    });