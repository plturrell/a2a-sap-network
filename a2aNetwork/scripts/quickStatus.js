#!/usr/bin/env node

/**
 * A2A Network - Quick Status Check Tool
 * Simple, fast status check without starting servers or hanging processes
 */

const http = require('http');

class QuickStatusCheck {
    constructor() {
        this.results = {
            timestamp: new Date().toISOString(),
            status: {},
            summary: ''
        };
    }

    log(message, type = 'info') {
        const prefix = {
            'info': '📋',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }[type] || '📋';
        
        console.log(`${prefix} ${message}`);
    }

    async checkPort(port, name) {
        return new Promise((resolve) => {
            const req = http.request(`http://localhost:${port}`, { timeout: 2000 }, (res) => {
                resolve({ port, name, status: 'RUNNING', statusCode: res.statusCode });
            });

            req.on('error', () => {
                resolve({ port, name, status: 'NOT_RUNNING', error: 'Connection refused' });
            });

            req.on('timeout', () => {
                resolve({ port, name, status: 'TIMEOUT', error: 'Request timeout' });
            });

            req.end();
        });
    }

    async checkAPI(url, name) {
        return new Promise((resolve) => {
            const req = http.request(url, { timeout: 2000 }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    resolve({ 
                        name, 
                        status: 'RESPONDING', 
                        statusCode: res.statusCode,
                        hasData: data.length > 0,
                        isJSON: data.startsWith('{') || data.startsWith('[')
                    });
                });
            });

            req.on('error', (error) => {
                resolve({ name, status: 'FAILED', error: error.code || error.message });
            });

            req.on('timeout', () => {
                resolve({ name, status: 'TIMEOUT', error: 'Request timeout' });
            });

            req.end();
        });
    }

    async run() {
        this.log('🚀 A2A Network Quick Status Check', 'info');
        this.log('='.repeat(50), 'info');

        // Check server ports
        const portChecks = await Promise.all([
            this.checkPort(4005, 'Static File Server'),
            this.checkPort(4004, 'SAP CAP Server')
        ]);

        portChecks.forEach(result => {
            this.results.status[result.name] = result;
            if (result.status === 'RUNNING') {
                this.log(`${result.name} (${result.port}): Running - Status ${result.statusCode}`, 'success');
            } else {
                this.log(`${result.name} (${result.port}): ${result.status} - ${result.error || 'Unknown'}`, 'error');
            }
        });

        // Check API endpoints if SAP CAP server is running
        const sapCapRunning = portChecks.find(p => p.name === 'SAP CAP Server' && p.status === 'RUNNING');
        if (sapCapRunning) {
            this.log('\n🔍 Testing API Endpoints:', 'info');
            
            const apiChecks = await Promise.all([
                this.checkAPI('http://localhost:4004/api/v1/Agents?id=agent_visualization', 'Agents API'),
                this.checkAPI('http://localhost:4004/api/v1/Services?id=dashboard_test', 'Services API'),
                this.checkAPI('http://localhost:4004/api/v1/NetworkStats?id=overview_dashboard', 'Network Stats API'),
                this.checkAPI('http://localhost:4004/api/v1/Notifications?id=notification_center', 'Notifications API')
            ]);

            apiChecks.forEach(result => {
                this.results.status[result.name] = result;
                if (result.status === 'RESPONDING') {
                    const dataInfo = result.isJSON ? 'JSON' : 'Text';
                    this.log(`${result.name}: ${result.statusCode} - ${dataInfo} (${result.hasData ? 'Has Data' : 'Empty'})`, 
                             result.statusCode < 400 ? 'success' : 'warning');
                } else {
                    this.log(`${result.name}: ${result.status} - ${result.error}`, 'error');
                }
            });
        }

        // Generate summary
        const runningServers = portChecks.filter(p => p.status === 'RUNNING').length;
        const totalServers = portChecks.length;
        
        this.log('\n📊 SUMMARY:', 'info');
        this.log(`Servers Running: ${runningServers}/${totalServers}`, runningServers === totalServers ? 'success' : 'warning');
        
        if (sapCapRunning) {
            this.log('Backend API: Available', 'success');
            this.results.summary = 'System operational - both servers running with API access';
        } else {
            this.log('Backend API: Not available', 'error');
            this.results.summary = 'System partially operational - SAP CAP server not running';
        }

        return this.results;
    }
}

// Run the status check if called directly
if (require.main === module) {
    const checker = new QuickStatusCheck();
    checker.run().then((results) => {
        process.exit(results.status['SAP CAP Server']?.status === 'RUNNING' ? 0 : 1);
    }).catch((error) => {
        console.error('❌ Status check failed:', error.message);
        process.exit(1);
    });
}

module.exports = QuickStatusCheck;
