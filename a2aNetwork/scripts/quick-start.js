#!/usr/bin/env node

/**
 * @fileoverview Quick Start Script for A2A Network Development
 * @description Starts all services needed for development
 */

const { spawn } = require('child_process');
const chalk = require('chalk');

console.log(chalk.blue.bold('🚀 A2A Network - Quick Start'));
console.log(chalk.blue('============================\n'));

// Check if setup has been run
const fs = require('fs');
if (!fs.existsSync('.env')) {
    console.log(chalk.red('❌ No .env file found!'));
    console.log(chalk.yellow('Please run: npm run setup:dev'));
    process.exit(1);
}

// Start services
const services = [];

// 1. Start local blockchain (if enabled)
if (process.env.ENABLE_BLOCKCHAIN === 'true') {
    console.log(chalk.yellow('🔗 Starting local blockchain...'));
    const blockchain = spawn('npm', ['run', 'blockchain:local'], {
        stdio: 'pipe',
        shell: true
    });
    services.push(blockchain);
}

// 2. Start Redis (optional)
try {
    const redis = spawn('redis-server', [], {
        stdio: 'pipe'
    });
    services.push(redis);
    console.log(chalk.green('✅ Redis started'));
} catch (e) {
    console.log(chalk.yellow('⚠️  Redis not available, using in-memory cache'));
}

// 3. Start CAP server with watch mode
console.log(chalk.yellow('🚀 Starting CAP server with hot reload...'));
const server = spawn('npm', ['run', 'watch'], {
    stdio: 'inherit',
    shell: true
});
services.push(server);

// Handle shutdown
process.on('SIGINT', () => {
    console.log(chalk.red('\n👋 Shutting down services...'));
    services.forEach(service => {
        if (service && !service.killed) {
            service.kill();
        }
    });
    process.exit(0);
});

console.log(chalk.green.bold('\n✅ All services started!'));
console.log(chalk.cyan('📱 UI: http://localhost:4004'));
console.log(chalk.cyan('📡 API: http://localhost:4004/api/v1/network'));
console.log(chalk.cyan('🏥 Health: http://localhost:4004/health\n'));
