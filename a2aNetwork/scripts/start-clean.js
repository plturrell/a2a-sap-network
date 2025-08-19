#!/usr/bin/env node

/**
 * @fileoverview Clean Startup Script for A2A Network
 * @description Clears port conflicts and starts the server cleanly
 */

const { exec, spawn } = require('child_process');
const { portManager } = require('../srv/utils/portManager');

console.log('🧹 A2A Network - Clean Startup');
console.log('==============================\n');

async function cleanStart() {
    try {
        // Step 1: Kill any conflicting processes on common ports
        console.log('🔄 Clearing port conflicts...');
        const portsToCheck = [4004, 4005, 4006, 8545, 6379];
        
        await portManager.killPortProcesses(portsToCheck, false);
        
        // Wait a moment for cleanup
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Step 2: Start the server
        console.log('🚀 Starting A2A Network server...\n');
        
        const serverProcess = spawn('npm', ['start'], {
            stdio: 'inherit',
            shell: true
        });
        
        // Handle shutdown gracefully
        process.on('SIGINT', () => {
            console.log('\n🛑 Shutting down server...');
            serverProcess.kill('SIGINT');
            process.exit(0);
        });
        
        process.on('SIGTERM', () => {
            console.log('\n🛑 Shutting down server...');
            serverProcess.kill('SIGTERM');
            process.exit(0);
        });
        
        serverProcess.on('close', (code) => {
            console.log(`\n📋 Server process exited with code ${code}`);
            process.exit(code);
        });
        
    } catch (error) {
        console.error('❌ Failed to start server:', error.message);
        process.exit(1);
    }
}

// Run the clean start
cleanStart();