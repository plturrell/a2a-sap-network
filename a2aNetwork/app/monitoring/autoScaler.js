/**
 * @fileoverview Auto-scaling Controller for Fly.io Integration
 * @since 1.0.0
 * @module autoScaler
 * 
 * Handles automatic scaling decisions and executes scaling operations
 * Integrates with Fly.io API and monitoring system
 */

const EventEmitter = require('events');
const { spawn } = require('child_process');
const fetch = require('node-fetch');
const fs = require('fs').promises;
const path = require('path');

class AutoScaler extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Fly.io configuration
            flyAppName: options.flyAppName || process.env.FLY_APP_NAME || 'a2a-platform',
            flyApiToken: options.flyApiToken || process.env.FLY_API_TOKEN,
            flyRegion: options.flyRegion || 'iad', // Default to Ashburn
            
            // Scaling parameters
            minInstances: options.minInstances || 1,
            maxInstances: options.maxInstances || 10,
            scaleUpInstances: options.scaleUpInstances || 2,
            scaleDownInstances: options.scaleDownInstances || 1,
            
            // Timing
            cooldownPeriod: options.cooldownPeriod || 300000, // 5 minutes
            healthCheckTimeout: options.healthCheckTimeout || 60000, // 1 minute
            scaleUpDelay: options.scaleUpDelay || 30000, // 30 seconds
            scaleDownDelay: options.scaleDownDelay || 180000, // 3 minutes
            
            // Thresholds
            scaleUpThreshold: {
                cpu: 75,
                memory: 80,
                responseTime: 2000,
                errorRate: 5
            },
            scaleDownThreshold: {
                cpu: 25,
                memory: 30,
                responseTime: 500,
                errorRate: 1
            },
            
            // Safety settings
            enableAutoScaling: options.enableAutoScaling !== false,
            enableScaleDown: options.enableScaleDown !== false,
            maxScaleUpPerHour: options.maxScaleUpPerHour || 5,
            maxScaleDownPerHour: options.maxScaleDownPerHour || 3,
            
            // Logging
            logLevel: options.logLevel || 'info',
            metricsPath: options.metricsPath || '/tmp/autoscaler-metrics'
        };
        
        // State management
        this.currentInstances = 0;
        this.targetInstances = 0;
        this.lastScalingEvent = null;
        this.scalingInProgress = false;
        this.scalingHistory = [];
        this.hourlyScalingCount = { up: 0, down: 0, hour: new Date().getHours() };
        
        // Metrics tracking
        this.metrics = {
            scalingEvents: [],
            performanceHistory: [],
            instanceHistory: []
        };
        
        this.initialize();
    }
    
    async initialize() {
        try {
            console.log('🚀 Initializing Auto-scaler...');
            
            // Ensure metrics directory exists
            await fs.mkdir(this.config.metricsPath, { recursive: true });
            
            // Load historical data
            await this.loadHistoricalData();
            
            // Get current instance count
            await this.updateCurrentInstanceCount();
            
            // Start monitoring
            this.startMonitoring();
            
            console.log(`✅ Auto-scaler initialized. Current instances: ${this.currentInstances}`);
            
        } catch (error) {
            console.error('❌ Failed to initialize Auto-scaler:', error);
            throw error;
        }
    }
    
    async loadHistoricalData() {
        try {
            const historyFile = path.join(this.config.metricsPath, 'scaling_history.json');
            const historyData = await fs.readFile(historyFile, 'utf8');
            this.scalingHistory = JSON.parse(historyData);
            console.log(`📊 Loaded ${this.scalingHistory.length} historical scaling events`);
        } catch (error) {
            console.log('📊 No historical scaling data found, starting fresh');
        }
    }
    
    async updateCurrentInstanceCount() {
        try {
            if (this.config.flyApiToken) {
                this.currentInstances = await this.getFlyInstanceCount();
            } else {
                // Fallback to estimated count based on health checks
                this.currentInstances = await this.estimateInstanceCount();
            }
            
            this.targetInstances = this.currentInstances;
            
            // Record instance count
            this.recordInstanceCount(this.currentInstances);
            
        } catch (error) {
            console.error('Failed to update instance count:', error);
            this.currentInstances = this.config.minInstances; // Safe fallback
        }
    }
    
    async getFlyInstanceCount() {
        try {
            const response = await fetch(`https://api.machines.dev/v1/apps/${this.config.flyAppName}/machines`, {
                headers: {
                    'Authorization': `Bearer ${this.config.flyApiToken}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Fly API error: ${response.status} ${response.statusText}`);
            }
            
            const machines = await response.json();
            const runningMachines = machines.filter(m => m.state === 'started' || m.state === 'starting');
            
            return runningMachines.length;
            
        } catch (error) {
            console.error('Failed to get Fly.io instance count:', error);
            throw error;
        }
    }
    
    async estimateInstanceCount() {
        // Estimate based on healthy agent endpoints
        const healthyAgents = await this.getHealthyAgentCount();
        
        // Assume each instance runs multiple agents
        const agentsPerInstance = 16; // Adjust based on your setup
        return Math.max(1, Math.ceil(healthyAgents / agentsPerInstance));
    }
    
    async getHealthyAgentCount() {
        const agentPorts = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015];
        let healthyCount = 0;
        
        for (const port of agentPorts) {
            try {
                const response = await fetch(`http://localhost:${port}/health`, { timeout: 5000 });
                if (response.ok) {
                    healthyCount++;
                }
            } catch (error) {
                // Agent is not healthy
            }
        }
        
        return healthyCount;
    }
    
    startMonitoring() {
        // Reset hourly counters every hour
        setInterval(() => {
            const currentHour = new Date().getHours();
            if (currentHour !== this.hourlyScalingCount.hour) {
                this.hourlyScalingCount = { up: 0, down: 0, hour: currentHour };
                console.log('📊 Reset hourly scaling counters');
            }
        }, 60000); // Check every minute
        
        // Persist metrics every 5 minutes
        setInterval(() => {
            this.persistMetrics();
        }, 300000);
    }
    
    async evaluateScaling(performanceMetrics) {
        if (!this.config.enableAutoScaling) {
            return { action: 'none', reason: 'Auto-scaling disabled' };
        }
        
        if (this.scalingInProgress) {
            return { action: 'none', reason: 'Scaling already in progress' };
        }
        
        // Check cooldown period
        if (this.lastScalingEvent && (Date.now() - this.lastScalingEvent.timestamp) < this.config.cooldownPeriod) {
            const remainingCooldown = this.config.cooldownPeriod - (Date.now() - this.lastScalingEvent.timestamp);
            return { action: 'none', reason: `Cooldown period (${Math.round(remainingCooldown / 1000)}s remaining)` };
        }
        
        // Evaluate metrics against thresholds
        const scaleUpReasons = [];
        const scaleDownReasons = [];
        
        // CPU evaluation
        if (performanceMetrics.system?.cpu?.usage >= this.config.scaleUpThreshold.cpu) {
            scaleUpReasons.push(`CPU usage: ${performanceMetrics.system.cpu.usage}%`);
        } else if (performanceMetrics.system?.cpu?.usage <= this.config.scaleDownThreshold.cpu) {
            scaleDownReasons.push(`CPU usage: ${performanceMetrics.system.cpu.usage}%`);
        }
        
        // Memory evaluation
        if (performanceMetrics.system?.memory?.usagePercentage >= this.config.scaleUpThreshold.memory) {
            scaleUpReasons.push(`Memory usage: ${performanceMetrics.system.memory.usagePercentage}%`);
        } else if (performanceMetrics.system?.memory?.usagePercentage <= this.config.scaleDownThreshold.memory) {
            scaleDownReasons.push(`Memory usage: ${performanceMetrics.system.memory.usagePercentage}%`);
        }
        
        // Response time evaluation
        if (performanceMetrics.agents?.averages?.responseTime >= this.config.scaleUpThreshold.responseTime) {
            scaleUpReasons.push(`Response time: ${performanceMetrics.agents.averages.responseTime}ms`);
        } else if (performanceMetrics.agents?.averages?.responseTime <= this.config.scaleDownThreshold.responseTime) {
            scaleDownReasons.push(`Response time: ${performanceMetrics.agents.averages.responseTime}ms`);
        }
        
        // Agent health evaluation
        if (performanceMetrics.agents?.healthy < performanceMetrics.agents?.total * 0.8) {
            scaleUpReasons.push(`Agent health: ${Math.round((performanceMetrics.agents.healthy / performanceMetrics.agents.total) * 100)}%`);
        }
        
        // Decide on scaling action
        if (scaleUpReasons.length > 0) {
            // Check hourly limits
            if (this.hourlyScalingCount.up >= this.config.maxScaleUpPerHour) {
                return { action: 'none', reason: 'Hourly scale-up limit reached' };
            }
            
            // Check max instances
            if (this.currentInstances >= this.config.maxInstances) {
                return { action: 'none', reason: 'Maximum instances reached' };
            }
            
            return {
                action: 'scale_up',
                reason: `Scale up triggered by: ${scaleUpReasons.join(', ')}`,
                targetInstances: Math.min(this.currentInstances + this.config.scaleUpInstances, this.config.maxInstances),
                metrics: performanceMetrics
            };
        }
        
        if (scaleDownReasons.length > 0 && this.config.enableScaleDown) {
            // Check hourly limits
            if (this.hourlyScalingCount.down >= this.config.maxScaleDownPerHour) {
                return { action: 'none', reason: 'Hourly scale-down limit reached' };
            }
            
            // Check min instances
            if (this.currentInstances <= this.config.minInstances) {
                return { action: 'none', reason: 'Minimum instances reached' };
            }
            
            // Additional safety check - don't scale down if we recently scaled up
            if (this.lastScalingEvent && this.lastScalingEvent.action === 'scale_up' && 
                (Date.now() - this.lastScalingEvent.timestamp) < this.config.scaleDownDelay * 2) {
                return { action: 'none', reason: 'Recent scale-up, avoiding scale-down' };
            }
            
            return {
                action: 'scale_down',
                reason: `Scale down triggered by: ${scaleDownReasons.join(', ')}`,
                targetInstances: Math.max(this.currentInstances - this.config.scaleDownInstances, this.config.minInstances),
                metrics: performanceMetrics
            };
        }
        
        return { action: 'none', reason: 'Metrics within acceptable range' };
    }
    
    async executeScaling(scalingDecision) {
        if (scalingDecision.action === 'none') {
            return { success: true, message: scalingDecision.reason };
        }
        
        const scalingEvent = {
            id: `scaling_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: Date.now(),
            action: scalingDecision.action,
            reason: scalingDecision.reason,
            fromInstances: this.currentInstances,
            toInstances: scalingDecision.targetInstances,
            status: 'initiated',
            metrics: scalingDecision.metrics
        };
        
        this.scalingInProgress = true;
        this.lastScalingEvent = scalingEvent;
        this.scalingHistory.push(scalingEvent);
        
        console.log(`🔄 Starting scaling operation: ${scalingEvent.action} (${scalingEvent.fromInstances} → ${scalingEvent.toInstances})`);
        this.emit('scalingStarted', scalingEvent);
        
        try {
            // Execute the scaling operation
            const result = await this.performScalingOperation(scalingEvent);
            
            scalingEvent.status = 'completed';
            scalingEvent.completedAt = Date.now();
            scalingEvent.duration = scalingEvent.completedAt - scalingEvent.timestamp;
            scalingEvent.result = result;
            
            // Update counters
            if (scalingEvent.action === 'scale_up') {
                this.hourlyScalingCount.up++;
            } else if (scalingEvent.action === 'scale_down') {
                this.hourlyScalingCount.down++;
            }
            
            // Update current instance count
            this.currentInstances = scalingEvent.toInstances;
            this.recordInstanceCount(this.currentInstances);
            
            console.log(`✅ Scaling operation completed: ${scalingEvent.action} (${Math.round(scalingEvent.duration / 1000)}s)`);
            this.emit('scalingCompleted', scalingEvent);
            
            return { success: true, event: scalingEvent, message: 'Scaling completed successfully' };
            
        } catch (error) {
            scalingEvent.status = 'failed';
            scalingEvent.error = error.message;
            scalingEvent.completedAt = Date.now();
            
            console.error(`❌ Scaling operation failed: ${error.message}`);
            this.emit('scalingFailed', scalingEvent);
            
            return { success: false, event: scalingEvent, message: error.message };
            
        } finally {
            this.scalingInProgress = false;
        }
    }
    
    async performScalingOperation(scalingEvent) {
        const { action, toInstances } = scalingEvent;
        
        if (this.config.flyApiToken) {
            // Use Fly.io API for scaling
            return await this.scaleFlyApp(action, toInstances);
        } else {
            // Use Fly CLI for scaling
            return await this.scaleFlyAppCLI(action, toInstances);
        }
    }
    
    async scaleFlyApp(action, targetInstances) {
        try {
            // For Fly.io Machines API
            const response = await fetch(`https://api.machines.dev/v1/apps/${this.config.flyAppName}/machines`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.config.flyApiToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    config: {
                        image: process.env.DOCKER_IMAGE || 'finsightintelligence/a2a:main',
                        env: process.env,
                        services: [
                            {
                                ports: [
                                    { port: 80, handlers: ['http'] },
                                    { port: 443, handlers: ['tls', 'http'] }
                                ],
                                protocol: 'tcp',
                                internal_port: 8080
                            }
                        ]
                    },
                    region: this.config.flyRegion,
                    count: targetInstances
                })
            });
            
            if (!response.ok) {
                throw new Error(`Fly API scaling failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Wait for health check
            await this.waitForHealthy(targetInstances);
            
            return {
                method: 'api',
                targetInstances,
                machineIds: result.id ? [result.id] : [],
                apiResponse: result
            };
            
        } catch (error) {
            console.error('Fly.io API scaling failed, falling back to CLI:', error);
            return await this.scaleFlyAppCLI(action, targetInstances);
        }
    }
    
    async scaleFlyAppCLI(action, targetInstances) {
        return new Promise((resolve, reject) => {
            const args = ['scale', 'count', targetInstances.toString()];
            
            if (this.config.flyRegion) {
                args.push('--region', this.config.flyRegion);
            }
            
            console.log(`🔧 Executing: flyctl ${args.join(' ')}`);
            
            const flyProcess = spawn('flyctl', args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { 
                    ...process.env, 
                    FLY_API_TOKEN: this.config.flyApiToken,
                    FLY_APP: this.config.flyAppName
                }
            });
            
            let stdout = '';
            let stderr = '';
            
            flyProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            flyProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            flyProcess.on('close', async (code) => {
                if (code === 0) {
                    console.log(`✅ Fly scale command succeeded`);
                    
                    try {
                        // Wait for instances to be healthy
                        await this.waitForHealthy(targetInstances);
                        
                        resolve({
                            method: 'cli',
                            targetInstances,
                            exitCode: code,
                            stdout,
                            stderr
                        });
                    } catch (healthError) {
                        reject(new Error(`Scaling succeeded but health check failed: ${healthError.message}`));
                    }
                } else {
                    reject(new Error(`Fly scale command failed with exit code ${code}: ${stderr}`));
                }
            });
            
            flyProcess.on('error', (error) => {
                reject(new Error(`Failed to spawn flyctl: ${error.message}`));
            });
        });
    }
    
    async waitForHealthy(targetInstances, timeoutMs = 60000) {
        const startTime = Date.now();
        const checkInterval = 5000; // 5 seconds
        
        console.log(`⏳ Waiting for ${targetInstances} instances to be healthy...`);
        
        while (Date.now() - startTime < timeoutMs) {
            try {
                const currentCount = await this.getFlyInstanceCount();
                const healthyAgents = await this.getHealthyAgentCount();
                
                console.log(`📊 Current: ${currentCount} instances, ${healthyAgents} healthy agents`);
                
                // Check if we have the target number of instances and agents are healthy
                if (currentCount === targetInstances && healthyAgents >= targetInstances * 8) { // Assume 8+ agents per instance
                    console.log(`✅ Health check passed: ${currentCount} instances, ${healthyAgents} healthy agents`);
                    return true;
                }
                
                await new Promise(resolve => setTimeout(resolve, checkInterval));
                
            } catch (error) {
                console.warn('Health check error:', error.message);
                await new Promise(resolve => setTimeout(resolve, checkInterval));
            }
        }
        
        throw new Error(`Health check timeout after ${timeoutMs}ms`);
    }
    
    recordInstanceCount(count) {
        this.metrics.instanceHistory.push({
            timestamp: Date.now(),
            instances: count
        });
        
        // Keep only last 1000 records
        if (this.metrics.instanceHistory.length > 1000) {
            this.metrics.instanceHistory = this.metrics.instanceHistory.slice(-1000);
        }
    }
    
    recordPerformanceMetrics(metrics) {
        this.metrics.performanceHistory.push({
            timestamp: Date.now(),
            ...metrics
        });
        
        // Keep only last 1000 records
        if (this.metrics.performanceHistory.length > 1000) {
            this.metrics.performanceHistory = this.metrics.performanceHistory.slice(-1000);
        }
    }
    
    async persistMetrics() {
        try {
            const metricsFile = path.join(this.config.metricsPath, 'scaling_history.json');
            await fs.writeFile(metricsFile, JSON.stringify(this.scalingHistory.slice(-100), null, 2));
            
            const instanceFile = path.join(this.config.metricsPath, 'instance_history.json');
            await fs.writeFile(instanceFile, JSON.stringify(this.metrics.instanceHistory.slice(-500), null, 2));
            
            console.log('📊 Metrics persisted successfully');
        } catch (error) {
            console.error('Failed to persist metrics:', error);
        }
    }
    
    // Manual scaling methods
    async manualScale(targetInstances, reason = 'Manual scaling') {
        if (targetInstances < this.config.minInstances || targetInstances > this.config.maxInstances) {
            throw new Error(`Target instances ${targetInstances} outside allowed range [${this.config.minInstances}, ${this.config.maxInstances}]`);
        }
        
        const scalingDecision = {
            action: targetInstances > this.currentInstances ? 'scale_up' : 'scale_down',
            reason,
            targetInstances,
            metrics: { manual: true }
        };
        
        return await this.executeScaling(scalingDecision);
    }
    
    async emergencyScale(targetInstances) {
        console.log(`🚨 Emergency scaling to ${targetInstances} instances`);
        
        // Bypass cooldowns and limits for emergency scaling
        const originalCooldown = this.lastScalingEvent;
        const originalMaxHourly = this.hourlyScalingCount;
        
        this.lastScalingEvent = null;
        this.hourlyScalingCount = { up: 0, down: 0, hour: new Date().getHours() };
        
        try {
            const result = await this.manualScale(targetInstances, 'Emergency scaling');
            return result;
        } finally {
            // Restore original state
            this.lastScalingEvent = originalCooldown;
            this.hourlyScalingCount = originalMaxHourly;
        }
    }
    
    // Status and reporting methods
    getStatus() {
        return {
            enabled: this.config.enableAutoScaling,
            currentInstances: this.currentInstances,
            targetInstances: this.targetInstances,
            scalingInProgress: this.scalingInProgress,
            lastScalingEvent: this.lastScalingEvent,
            hourlyLimits: {
                scaleUp: {
                    used: this.hourlyScalingCount.up,
                    max: this.config.maxScaleUpPerHour
                },
                scaleDown: {
                    used: this.hourlyScalingCount.down,
                    max: this.config.maxScaleDownPerHour
                }
            },
            cooldownRemaining: this.lastScalingEvent ? 
                Math.max(0, this.config.cooldownPeriod - (Date.now() - this.lastScalingEvent.timestamp)) : 0,
            limits: {
                min: this.config.minInstances,
                max: this.config.maxInstances
            }
        };
    }
    
    getMetrics() {
        return {
            scalingHistory: this.scalingHistory.slice(-50),
            instanceHistory: this.metrics.instanceHistory.slice(-100),
            performanceHistory: this.metrics.performanceHistory.slice(-100)
        };
    }
    
    getScalingStats() {
        const total = this.scalingHistory.length;
        const successful = this.scalingHistory.filter(e => e.status === 'completed').length;
        const failed = this.scalingHistory.filter(e => e.status === 'failed').length;
        const scaleUps = this.scalingHistory.filter(e => e.action === 'scale_up').length;
        const scaleDowns = this.scalingHistory.filter(e => e.action === 'scale_down').length;
        
        const avgDuration = this.scalingHistory
            .filter(e => e.duration)
            .reduce((sum, e) => sum + e.duration, 0) / 
            this.scalingHistory.filter(e => e.duration).length || 0;
        
        return {
            total,
            successful,
            failed,
            successRate: total > 0 ? Math.round((successful / total) * 100) : 0,
            scaleUps,
            scaleDowns,
            avgDuration: Math.round(avgDuration / 1000) // seconds
        };
    }
    
    // Configuration methods
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        console.log('🔧 Auto-scaler configuration updated');
        this.emit('configUpdated', this.config);
    }
    
    enable() {
        this.config.enableAutoScaling = true;
        console.log('✅ Auto-scaling enabled');
    }
    
    disable() {
        this.config.enableAutoScaling = false;
        console.log('❌ Auto-scaling disabled');
    }
    
    // Cleanup method
    async shutdown() {
        console.log('🛑 Shutting down Auto-scaler...');
        
        // Persist final metrics
        await this.persistMetrics();
        
        console.log('✅ Auto-scaler shutdown complete');
    }
}

module.exports = AutoScaler;