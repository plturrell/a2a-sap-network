/**
 * @fileoverview API Endpoint Testing Script
 * @description Tests all tile data endpoints to verify they work after security fixes
 * @since 1.0.0
 * @author A2A Network Team
 */

const http = require('http');
const https = require('https');
const { URL } = require('url');

/**
 * API Endpoint Tester
 */
class APIEndpointTester {
    constructor(baseUrl = 'http://localhost:4004') {
        this.baseUrl = baseUrl;
        this.results = [];
        this.successCount = 0;
        this.errorCount = 0;
    }

    /**
     * Make HTTP request
     */
    async makeRequest(method, endpoint, headers = {}, body = null) {
        const handleMakeRequest = (resolve, reject) => {
            const url = new URL(endpoint, this.baseUrl);
            const isHttps = url.protocol === 'https:';
            const httpModule = isHttps ? https : http;

            const options = {
                hostname: url.hostname,
                port: url.port || (isHttps ? 443 : 80),
                path: url.pathname + url.search,
                method: method.toUpperCase(),
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'A2A-Network-API-Tester/1.0.0',
                    'Accept': 'application/json',
                    ...headers
                },
                timeout: 10000
            };

            if (body && typeof body === 'object') {
                const bodyString = JSON.stringify(body);
                options.headers['Content-Length'] = Buffer.byteLength(bodyString);
            }

            const handleHttpResponse = (res) => {
                let data = '';

                const handleDataChunk = (chunk) => {
                    data += chunk;
                };
                res.on('data', handleDataChunk);

                const handleResponseEnd = () => {
                    try {
                        const parsedData = data ? JSON.parse(data) : {};
                        resolve({
                            statusCode: res.statusCode,
                            headers: res.headers,
                            data: parsedData,
                            rawData: data
                        });
                    } catch (error) {
                        resolve({
                            statusCode: res.statusCode,
                            headers: res.headers,
                            data: data,
                            rawData: data,
                            parseError: error.message
                        });
                    }
                };
                res.on('end', handleResponseEnd);
            };

            const req = httpModule.request(options, handleHttpResponse);

            const handleRequestError = (error) => {
                reject(error);
            };
            req.on('error', handleRequestError);

            const handleRequestTimeout = () => {
                req.destroy();
                reject(new Error('Request timeout'));
            };
            req.on('timeout', handleRequestTimeout);

            if (body && typeof body === 'object') {
                req.write(JSON.stringify(body));
            }

            req.end();
        };

        return new Promise(handleMakeRequest);
    }

    /**
     * Test single endpoint
     */
    async testEndpoint(name, method, endpoint, expectedStatus = 200, headers = {}) {
        const startTime = Date.now();

        try {
            console.log(`🧪 Testing ${name}: ${method} ${endpoint}`);

            const response = await this.makeRequest(method, endpoint, headers);
            const duration = Date.now() - startTime;

            const success = response.statusCode === expectedStatus;

            const result = {
                name,
                method,
                endpoint,
                expectedStatus,
                actualStatus: response.statusCode,
                success,
                duration,
                response: response.data,
                headers: response.headers,
                error: null
            };

            if (success) {
                console.log(`✅ ${name}: ${response.statusCode} (${duration}ms)`);
                this.successCount++;
            } else {
                console.log(`❌ ${name}: Expected ${expectedStatus}, got ${response.statusCode} (${duration}ms)`);
                if (response.data && response.data.error) {
                    console.log(`   Error: ${response.data.error}`);
                    result.error = response.data.error;
                }
                this.errorCount++;
            }

            this.results.push(result);
            return result;

        } catch (error) {
            const duration = Date.now() - startTime;
            console.log(`💥 ${name}: Request failed - ${error.message} (${duration}ms)`);

            const result = {
                name,
                method,
                endpoint,
                expectedStatus,
                actualStatus: null,
                success: false,
                duration,
                response: null,
                headers: null,
                error: error.message
            };

            this.results.push(result);
            this.errorCount++;
            return result;
        }
    }

    /**
     * Test all tile data endpoints
     */
    async testTileEndpoints() {
        console.log('\n🎯 Testing Tile Data Endpoints\n');

        const tileTests = [
            // Network Stats tile
            {
                name: 'Network Stats (Overview Dashboard)',
                method: 'GET',
                endpoint: '/api/v1/NetworkStats?id=overview_dashboard'
            },
            {
                name: 'Network Stats (Test Dashboard)',
                method: 'GET',
                endpoint: '/api/v1/NetworkStats?id=dashboard_test'
            },

            // Agents tile
            {
                name: 'Agents (Agent Visualization)',
                method: 'GET',
                endpoint: '/api/v1/Agents?id=agent_visualization'
            },
            {
                name: 'Agents (Test Dashboard)',
                method: 'GET',
                endpoint: '/api/v1/Agents?id=dashboard_test'
            },

            // Services tile
            {
                name: 'Services (Service Marketplace)',
                method: 'GET',
                endpoint: '/api/v1/Services?id=service_marketplace'
            },
            {
                name: 'Services (Test Dashboard)',
                method: 'GET',
                endpoint: '/api/v1/Services?id=dashboard_test'
            },

            // Blockchain stats
            {
                name: 'Blockchain Stats',
                method: 'GET',
                endpoint: '/api/v1/blockchain/stats?id=blockchain_dashboard'
            },
            {
                name: 'Blockchain Stats (OData)',
                method: 'GET',
                endpoint: '/odata/v4/blockchain/BlockchainStats?id=blockchain_dashboard'
            },

            // Notifications
            {
                name: 'Notifications',
                method: 'GET',
                endpoint: '/api/v1/Notifications'
            },
            {
                name: 'Notifications Count',
                method: 'GET',
                endpoint: '/api/v1/notifications/count'
            },

            // Network Analytics & Health
            {
                name: 'Network Analytics',
                method: 'GET',
                endpoint: '/api/v1/network/analytics'
            },
            {
                name: 'Network Health',
                method: 'GET',
                endpoint: '/api/v1/network/health'
            }
        ];

        for (const test of tileTests) {
            await this.testEndpoint(test.name, test.method, test.endpoint);
            // Add small delay between requests
            const delayBetweenTileRequests = (resolve) => setTimeout(resolve, 100);
            await new Promise(delayBetweenTileRequests);
        }
    }

    /**
     * Test system endpoints
     */
    async testSystemEndpoints() {
        console.log('\n🔧 Testing System Endpoints\n');

        const systemTests = [
            {
                name: 'Current Metrics',
                method: 'GET',
                endpoint: '/api/v1/metrics/current'
            },
            {
                name: 'Performance Metrics',
                method: 'GET',
                endpoint: '/api/v1/metrics/performance'
            },
            {
                name: 'Operations Status',
                method: 'GET',
                endpoint: '/api/v1/operations/status'
            },
            {
                name: 'Monitoring Status',
                method: 'GET',
                endpoint: '/api/v1/monitoring/status'
            },
            {
                name: 'Blockchain Status',
                method: 'GET',
                endpoint: '/api/v1/blockchain/status'
            },
            {
                name: 'Debug Agents',
                method: 'GET',
                endpoint: '/api/v1/debug/agents'
            }
        ];

        for (const test of systemTests) {
            await this.testEndpoint(test.name, test.method, test.endpoint);
            const delayBetweenSystemRequests = (resolve) => setTimeout(resolve, 100);
            await new Promise(delayBetweenSystemRequests);
        }
    }

    /**
     * Test API versioning endpoints
     */
    async testVersioningEndpoints() {
        console.log('\n📋 Testing API Versioning Endpoints\n');

        const versionTests = [
            {
                name: 'Supported Versions',
                method: 'GET',
                endpoint: '/api/versions'
            },
            {
                name: 'Version Info (2.0.0)',
                method: 'GET',
                endpoint: '/api/version/2.0.0'
            },
            {
                name: 'Migration Guide',
                method: 'GET',
                endpoint: '/api/migration/1.0.0/2.0.0'
            },
            {
                name: 'Deprecation Status',
                method: 'GET',
                endpoint: '/api/deprecation-status'
            }
        ];

        for (const test of versionTests) {
            await this.testEndpoint(test.name, test.method, test.endpoint);
            const delayBetweenVersionRequests = (resolve) => setTimeout(resolve, 100);
            await new Promise(delayBetweenVersionRequests);
        }
    }

    /**
     * Test with different API versions
     */
    async testVersionedRequests() {
        console.log('\n🔄 Testing Versioned Requests\n');

        const versionedTests = [
            {
                name: 'Network Stats (v1.0.0)',
                method: 'GET',
                endpoint: '/api/v1/NetworkStats?id=overview_dashboard',
                headers: { 'API-Version': '1.0.0' }
            },
            {
                name: 'Network Stats (v2.0.0)',
                method: 'GET',
                endpoint: '/api/v1/NetworkStats?id=overview_dashboard',
                headers: { 'API-Version': '2.0.0' }
            },
            {
                name: 'Metrics with Accept-Version',
                method: 'GET',
                endpoint: '/api/v1/metrics/current',
                headers: { 'Accept-Version': '1.1.0' }
            }
        ];

        for (const test of versionedTests) {
            await this.testEndpoint(test.name, test.method, test.endpoint, 200, test.headers);
            const delayBetweenVersionedRequests = (resolve) => setTimeout(resolve, 100);
            await new Promise(delayBetweenVersionedRequests);
        }
    }

    /**
     * Test security fixes
     */
    async testSecurityFixes() {
        console.log('\n🔒 Testing Security Fixes\n');

        // Test that previously blocked legitimate queries now work
        const securityTests = [
            {
                name: 'Dashboard Query (Previously Blocked)',
                method: 'GET',
                endpoint: '/api/v1/NetworkStats?id=overview_dashboard'
            },
            {
                name: 'Agent Query with Special Chars',
                method: 'GET',
                endpoint: '/api/v1/Agents?id=agent_visualization'
            },
            {
                name: 'Service Query',
                method: 'GET',
                endpoint: '/api/v1/Services?id=service_marketplace'
            }
        ];

        for (const test of securityTests) {
            await this.testEndpoint(test.name, test.method, test.endpoint);
            const delayBetweenSecurityRequests = (resolve) => setTimeout(resolve, 100);
            await new Promise(delayBetweenSecurityRequests);
        }
    }

    /**
     * Generate comprehensive test report
     */
    generateReport() {
        console.log(`\n${  '='.repeat(80)}`);
        console.log('📊 API ENDPOINT TEST REPORT');
        console.log('='.repeat(80));

        console.log('\n📈 Summary:');
        console.log(`   Total Tests: ${this.results.length}`);
        console.log(`   ✅ Successful: ${this.successCount}`);
        console.log(`   ❌ Failed: ${this.errorCount}`);
        console.log(`   📊 Success Rate: ${((this.successCount / this.results.length) * 100).toFixed(1)}%`);

        if (this.errorCount > 0) {
            console.log('\n❌ Failed Tests:');
            const filterFailedResults = (r) => !r.success;
            const logFailedResult = (result) => {
                console.log(`   • ${result.name}: ${result.error || `HTTP ${result.actualStatus}`}`);
            };
            this.results
                .filter(filterFailedResults)
                .forEach(logFailedResult);
        }

        console.log('\n⏱️  Performance:');
        const calculateTotalDuration = (sum, r) => sum + r.duration;
        const avgDuration = this.results.reduce(calculateTotalDuration, 0) / this.results.length;
        const getDurationValue = (r) => r.duration;
        const maxDuration = Math.max(...this.results.map(getDurationValue));
        const minDuration = Math.min(...this.results.map(getDurationValue));

        console.log(`   Average Response Time: ${avgDuration.toFixed(1)}ms`);
        console.log(`   Fastest Response: ${minDuration}ms`);
        console.log(`   Slowest Response: ${maxDuration}ms`);

        // Group by endpoint type
        const isTileEndpoint = (r) => r.name.includes('Stats') || r.name.includes('Agents') || r.name.includes('Services') || r.name.includes('Notifications');
        const isSystemEndpoint = (r) => r.name.includes('Metrics') || r.name.includes('Status') || r.name.includes('Operations');
        const isVersionEndpoint = (r) => r.name.includes('Version') || r.name.includes('Migration');
        const tileEndpoints = this.results.filter(isTileEndpoint);
        const systemEndpoints = this.results.filter(isSystemEndpoint);
        const versionEndpoints = this.results.filter(isVersionEndpoint);

        console.log('\n📋 Results by Category:');
        const isSuccessful = (r) => r.success;
        console.log(`   🎯 Tile Endpoints: ${tileEndpoints.filter(isSuccessful).length}/${tileEndpoints.length} successful`);
        console.log(`   🔧 System Endpoints: ${systemEndpoints.filter(isSuccessful).length}/${systemEndpoints.length} successful`);
        console.log(`   📝 Version Endpoints: ${versionEndpoints.filter(isSuccessful).length}/${versionEndpoints.length} successful`);

        console.log(`\n${  '='.repeat(80)}`);

        return {
            total: this.results.length,
            successful: this.successCount,
            failed: this.errorCount,
            successRate: (this.successCount / this.results.length) * 100,
            avgDuration,
            results: this.results
        };
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('🚀 Starting A2A Network API Endpoint Tests\n');

        try {
            await this.testTileEndpoints();
            await this.testSystemEndpoints();
            await this.testVersioningEndpoints();
            await this.testVersionedRequests();
            await this.testSecurityFixes();

            return this.generateReport();

        } catch (error) {
            console.error('💥 Test suite failed:', error);
            throw error;
        }
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new APIEndpointTester();
    const handleTestSuccess = (report) => {
        console.log('\n✅ Test suite completed');
        process.exit(report.failed > 0 ? 1 : 0);
    };

    const handleTestError = (error) => {
        console.error('💥 Test suite failed:', error);
        process.exit(1);
    };

    tester.runAllTests()
        .then(handleTestSuccess)
        .catch(handleTestError);
}

module.exports = { APIEndpointTester };