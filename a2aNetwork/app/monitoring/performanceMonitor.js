/**
 * @fileoverview SAP Enterprise Performance Monitoring Service
 * @since 1.0.0
 * @module performanceMonitor
 * @author SAP A2A Platform Team
 * @copyright 2024 SAP SE
 * 
 * Enterprise-grade performance monitoring system with auto-scaling capabilities
 * Compliant with SAP Cloud Application Programming Model standards
 * Includes comprehensive error handling, logging, and configuration management
 */

'use strict';

const EventEmitter = require('events');
const os = require('os');
const fs = require('fs').promises;
const path = require('path');
const cds = require('@sap/cds');
const { randomUUID } = require('crypto');
const fetch = require('node-fetch');

// Enterprise logging with structured format
const LOG = cds.log('performance-monitor');

// SAP standard error types
const { 
    ValidationError, 
    ConfigurationError, 
    SystemError,
    TimeoutError 
} = require('../lib/sapErrorTypes');

// Configuration management
const ConfigManager = require('../lib/sapConfigManager');
const MetricsCollector = require('../lib/sapMetricsCollector');
const AlertManager = require('../lib/sapAlertManager');

class PerformanceMonitor extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Monitoring intervals (milliseconds)
            systemInterval: options.systemInterval || 30000,  // 30 seconds
            agentInterval: options.agentInterval || 60000,    // 1 minute
            metricsInterval: options.metricsInterval || 15000, // 15 seconds
            
            // Auto-scaling thresholds
            cpu: {
                scaleUpThreshold: options.cpuScaleUp || 75,
                scaleDownThreshold: options.cpuScaleDown || 25,
                sustainedDuration: options.cpuSustained || 300000, // 5 minutes
            },
            memory: {
                scaleUpThreshold: options.memoryScaleUp || 80,
                scaleDownThreshold: options.memoryScaleDown || 30,
                sustainedDuration: options.memorySustained || 300000,
            },
            responseTime: {
                scaleUpThreshold: options.responseTimeScaleUp || 2000, // 2 seconds
                scaleDownThreshold: options.responseTimeScaleDown || 500,
                sustainedDuration: options.responseTimeSustained || 180000, // 3 minutes
            },
            
            // Scaling limits
            minReplicas: options.minReplicas || 1,
            maxReplicas: options.maxReplicas || 10,
            cooldownPeriod: options.cooldownPeriod || 300000, // 5 minutes between scaling events
            
            // Storage paths
            metricsPath: options.metricsPath || '/tmp/a2a-metrics',
            alertsPath: options.alertsPath || '/tmp/a2a-alerts'
        };
        
        // Current state
        this.metrics = new Map();
        this.alerts = [];
        this.scalingHistory = [];
        this.lastScalingEvent = null;
        this.thresholdBreaches = new Map();
        this.isMonitoring = false;
        this.intervals = new Map();
        
        // Agent metadata
        this.agents = new Map();
        this.agentPorts = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015];
        
        // Initialize SAP enterprise components
        this.configManager = null;
        this.metricsCollector = null;
        this.alertManager = null;
        
        this.initialize();
    }
    
    async initialize() {
        try {
            // Create directories for metrics storage
            await this.ensureDirectories();
            
            // Load historical data
            await this.loadHistoricalData();
            
            // Start monitoring
            this.startMonitoring();
            
            console.log('✅ Performance Monitor initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize Performance Monitor:', error);
        }
    }
    
    async ensureDirectories() {
        try {
            await fs.mkdir(this.config.metricsPath, { recursive: true });
            await fs.mkdir(this.config.alertsPath, { recursive: true });
        } catch (error) {
            console.error('Failed to create monitoring directories:', error);
        }
    }
    
    async loadHistoricalData() {
        try {
            const metricsFile = path.join(this.config.metricsPath, 'historical.json');
            const alertsFile = path.join(this.config.alertsPath, 'alerts.json');
            
            try {
                const metricsData = await fs.readFile(metricsFile, 'utf8');
                const historicalMetrics = JSON.parse(metricsData);
                console.log(`📊 Loaded ${Object.keys(historicalMetrics).length} historical metric series`);
            } catch (e) {
                console.log('📊 No historical metrics found, starting fresh');
            }
            
            try {
                const alertsData = await fs.readFile(alertsFile, 'utf8');
                this.alerts = JSON.parse(alertsData);
                console.log(`🚨 Loaded ${this.alerts.length} historical alerts`);
            } catch (e) {
                console.log('🚨 No historical alerts found, starting fresh');
            }
        } catch (error) {
            console.error('Failed to load historical data:', error);
        }
    }
    
    startMonitoring() {
        if (this.isMonitoring) {
            console.warn('⚠️ Monitoring is already running');
            return;
        }
        
        this.isMonitoring = true;
        
        // System metrics monitoring
        this.intervals.set('system', setInterval(() => {
            this.collectSystemMetrics();
        }, this.config.systemInterval));
        
        // Agent metrics monitoring
        this.intervals.set('agents', setInterval(() => {
            this.collectAgentMetrics();
        }, this.config.agentInterval));
        
        // Performance analysis
        this.intervals.set('analysis', setInterval(() => {
            this.analyzePerformance();
        }, this.config.metricsInterval));
        
        // Metrics persistence
        this.intervals.set('persistence', setInterval(() => {
            this.persistMetrics();
        }, 60000)); // Every minute
        
        // Alert cleanup
        this.intervals.set('cleanup', setInterval(() => {
            this.cleanupOldAlerts();
        }, 300000)); // Every 5 minutes
        
        console.log('📊 Performance monitoring started');
        this.emit('monitoringStarted');
    }
    
    stopMonitoring() {
        if (!this.isMonitoring) {
            return;
        }
        
        this.isMonitoring = false;
        
        // Clear all intervals
        for (const [name, interval] of this.intervals) {
            clearInterval(interval);
            console.log(`⏹️ Stopped ${name} monitoring`);
        }
        this.intervals.clear();
        
        // Final metrics persistence
        this.persistMetrics();
        
        console.log('📊 Performance monitoring stopped');
        this.emit('monitoringStopped');
    }
    
    async collectSystemMetrics() {
        const timestamp = new Date().toISOString();
        
        try {
            // CPU metrics
            const cpuUsage = await this.getCPUUsage();
            
            // Memory metrics
            const memoryInfo = this.getMemoryInfo();
            
            // Disk metrics
            const diskInfo = await this.getDiskInfo();
            
            // Network metrics
            const networkInfo = this.getNetworkInfo();
            
            const systemMetrics = {
                timestamp,
                system: {
                    cpu: {
                        usage: cpuUsage,
                        loadAverage: os.loadavg(),
                        cores: os.cpus().length
                    },
                    memory: memoryInfo,
                    disk: diskInfo,
                    network: networkInfo,
                    uptime: os.uptime(),
                    platform: os.platform(),
                    arch: os.arch()
                }
            };
            
            this.storeMetric('system', systemMetrics);
            
            // Check for threshold breaches
            this.checkSystemThresholds(systemMetrics.system);
            
        } catch (error) {
            console.error('Failed to collect system metrics:', error);
            this.createAlert('system', 'error', 'Failed to collect system metrics', { error: error.message });
        }
    }
    
    async collectAgentMetrics() {
        const timestamp = new Date().toISOString();
        const agentMetrics = [];
        
        for (const port of this.agentPorts) {
            try {
                const agentHealth = await this.getAgentHealth(port);
                if (agentHealth) {
                    agentMetrics.push({
                        port,
                        timestamp,
                        ...agentHealth
                    });
                    
                    // Check agent-specific thresholds
                    this.checkAgentThresholds(port, agentHealth);
                }
            } catch (error) {
                console.warn(`Failed to collect metrics for agent on port ${port}:`, error.message);
            }
        }
        
        const aggregatedMetrics = {
            timestamp,
            agents: {
                total: this.agentPorts.length,
                healthy: agentMetrics.filter(a => a.status === 'healthy').length,
                unhealthy: agentMetrics.filter(a => a.status !== 'healthy').length,
                metrics: agentMetrics,
                averages: this.calculateAgentAverages(agentMetrics)
            }
        };
        
        this.storeMetric('agents', aggregatedMetrics);
        
        // Check overall agent health
        this.checkAgentClusterHealth(aggregatedMetrics.agents);
    }
    
    async getCPUUsage() {
        return new Promise((resolve) => {
            const startMeasure = this.cpuAverage();
            
            setTimeout(() => {
                const endMeasure = this.cpuAverage();
                const idleDifference = endMeasure.idle - startMeasure.idle;
                const totalDifference = endMeasure.total - startMeasure.total;
                const percentageCPU = 100 - Math.round(100 * idleDifference / totalDifference);
                resolve(Math.max(0, Math.min(100, percentageCPU)));
            }, 1000);
        });
    }
    
    cpuAverage() {
        const cpus = os.cpus();
        let user = 0, nice = 0, sys = 0, idle = 0, irq = 0;
        
        for (const cpu of cpus) {
            user += cpu.times.user;
            nice += cpu.times.nice;
            sys += cpu.times.sys;
            irq += cpu.times.irq;
            idle += cpu.times.idle;
        }
        
        return {
            idle: idle,
            total: user + nice + sys + idle + irq
        };
    }
    
    getMemoryInfo() {
        const total = os.totalmem();
        const free = os.freemem();
        const used = total - free;
        
        return {
            total,
            free,
            used,
            usagePercentage: Math.round((used / total) * 100)
        };
    }
    
    async getDiskInfo() {
        try {
            // Try to get disk usage using df command
            const { spawn } = require('child_process');
            
            return new Promise((resolve) => {
                const df = spawn('df', ['-h', '/']);
                let output = '';
                
                df.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                df.on('close', () => {
                    const lines = output.split('\n');
                    if (lines.length > 1) {
                        const parts = lines[1].split(/\s+/);
                        resolve({
                            filesystem: parts[0],
                            size: parts[1],
                            used: parts[2],
                            available: parts[3],
                            usagePercentage: parseInt(parts[4])
                        });
                    } else {
                        resolve({ error: 'Unable to parse disk usage' });
                    }
                });
                
                df.on('error', () => {
                    resolve({ error: 'df command not available' });
                });
            });
        } catch (error) {
            return { error: error.message };
        }
    }
    
    getNetworkInfo() {
        const interfaces = os.networkInterfaces();
        const networkStats = {};
        
        for (const [name, addrs] of Object.entries(interfaces)) {
            if (addrs) {
                networkStats[name] = addrs.filter(addr => !addr.internal);
            }
        }
        
        return networkStats;
    }
    
    async getAgentHealth(port) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`http://localhost:${port}/health`, {
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                return await response.json();
            }
            
            return null;
        } catch (error) {
            return null;
        }
    }
    
    calculateAgentAverages(agentMetrics) {
        if (agentMetrics.length === 0) return {};
        
        const healthyAgents = agentMetrics.filter(a => a.status === 'healthy');
        if (healthyAgents.length === 0) return {};
        
        const averages = {
            cpuUsage: 0,
            memoryUsage: 0,
            responseTime: 0,
            activeTasks: 0,
            successRate: 0
        };
        
        let count = 0;
        
        for (const agent of healthyAgents) {
            if (agent.cpu_usage !== null) {
                averages.cpuUsage += agent.cpu_usage;
                count++;
            }
            if (agent.memory_usage !== null) {
                averages.memoryUsage += agent.memory_usage;
            }
            if (agent.avg_response_time_ms !== null) {
                averages.responseTime += agent.avg_response_time_ms;
            }
            if (agent.active_tasks !== null) {
                averages.activeTasks += agent.active_tasks;
            }
            if (agent.success_rate !== null) {
                averages.successRate += agent.success_rate;
            }
        }
        
        if (count > 0) {
            averages.cpuUsage = Math.round(averages.cpuUsage / count);
            averages.memoryUsage = Math.round(averages.memoryUsage / count);
            averages.responseTime = Math.round(averages.responseTime / count);
            averages.activeTasks = Math.round(averages.activeTasks / count);
            averages.successRate = Math.round(averages.successRate / count);
        }
        
        return averages;
    }
    
    storeMetric(category, data) {
        if (!this.metrics.has(category)) {
            this.metrics.set(category, []);
        }
        
        const categoryMetrics = this.metrics.get(category);
        categoryMetrics.push(data);
        
        // Keep only last 1000 data points per category
        if (categoryMetrics.length > 1000) {
            categoryMetrics.splice(0, categoryMetrics.length - 1000);
        }
    }
    
    checkSystemThresholds(systemMetrics) {
        const { cpu, memory } = systemMetrics;
        
        // CPU threshold checking
        this.checkThreshold('system-cpu', cpu.usage, this.config.cpu);
        
        // Memory threshold checking
        this.checkThreshold('system-memory', memory.usagePercentage, this.config.memory);
        
        // Disk threshold checking (if available)
        if (systemMetrics.disk && systemMetrics.disk.usagePercentage) {
            this.checkThreshold('system-disk', systemMetrics.disk.usagePercentage, {
                scaleUpThreshold: 85,
                scaleDownThreshold: 60,
                sustainedDuration: 600000 // 10 minutes
            });
        }
    }
    
    checkAgentThresholds(port, agentHealth) {
        // Response time threshold
        if (agentHealth.avg_response_time_ms !== null) {
            this.checkThreshold(`agent-${port}-response`, agentHealth.avg_response_time_ms, this.config.responseTime);
        }
        
        // Success rate threshold (inverse - alert when too low)
        if (agentHealth.success_rate !== null) {
            this.checkThreshold(`agent-${port}-success`, 100 - agentHealth.success_rate, {
                scaleUpThreshold: 20, // Alert when success rate drops below 80%
                scaleDownThreshold: 5,  // OK when success rate above 95%
                sustainedDuration: 120000 // 2 minutes
            });
        }
        
        // Error rate threshold
        if (agentHealth.error_rate !== null) {
            this.checkThreshold(`agent-${port}-error`, agentHealth.error_rate, {
                scaleUpThreshold: 10, // Alert when error rate above 10%
                scaleDownThreshold: 2,
                sustainedDuration: 120000
            });
        }
    }
    
    checkAgentClusterHealth(agentStats) {
        const healthPercentage = (agentStats.healthy / agentStats.total) * 100;
        
        if (healthPercentage < 80) {
            this.createAlert('agents', 'critical', 'Agent cluster health degraded', {
                healthyAgents: agentStats.healthy,
                totalAgents: agentStats.total,
                healthPercentage: Math.round(healthPercentage)
            });
            
            // Consider scaling up if we have capacity
            this.considerScaling('agents-health', 'scale_up', {
                reason: 'Agent cluster health degraded',
                healthPercentage,
                healthyAgents: agentStats.healthy
            });
        }
    }
    
    checkThreshold(metricName, value, thresholds) {
        const now = Date.now();
        
        if (!this.thresholdBreaches.has(metricName)) {
            this.thresholdBreaches.set(metricName, {
                breachStart: null,
                breachType: null,
                lastValue: null
            });
        }
        
        const breach = this.thresholdBreaches.get(metricName);
        
        // Check if we're breaching thresholds
        if (value >= thresholds.scaleUpThreshold) {
            if (breach.breachType !== 'scale_up') {
                breach.breachStart = now;
                breach.breachType = 'scale_up';
            }
            
            // Check if breach has been sustained
            if (now - breach.breachStart >= thresholds.sustainedDuration) {
                this.handleSustainedBreach(metricName, 'scale_up', value, thresholds);
                breach.breachStart = now; // Reset to prevent continuous alerts
            }
        } else if (value <= thresholds.scaleDownThreshold) {
            if (breach.breachType !== 'scale_down') {
                breach.breachStart = now;
                breach.breachType = 'scale_down';
            }
            
            if (now - breach.breachStart >= thresholds.sustainedDuration) {
                this.handleSustainedBreach(metricName, 'scale_down', value, thresholds);
                breach.breachStart = now;
            }
        } else {
            // Value is within normal range
            if (breach.breachType) {
                breach.breachStart = null;
                breach.breachType = null;
            }
        }
        
        breach.lastValue = value;
    }
    
    handleSustainedBreach(metricName, breachType, value, thresholds) {
        console.log(`🎯 Sustained threshold breach: ${metricName} = ${value} (${breachType})`);
        
        // Create alert
        this.createAlert('threshold', breachType === 'scale_up' ? 'warning' : 'info', 
            `Sustained ${breachType} threshold breach: ${metricName}`, {
            metric: metricName,
            value,
            threshold: breachType === 'scale_up' ? thresholds.scaleUpThreshold : thresholds.scaleDownThreshold,
            duration: thresholds.sustainedDuration
        });
        
        // Consider scaling action
        this.considerScaling(metricName, breachType, {
            value,
            threshold: breachType === 'scale_up' ? thresholds.scaleUpThreshold : thresholds.scaleDownThreshold
        });
    }
    
    considerScaling(metricName, action, context) {
        const now = Date.now();
        
        // Check cooldown period
        if (this.lastScalingEvent && (now - this.lastScalingEvent) < this.config.cooldownPeriod) {
            console.log(`⏳ Scaling action skipped due to cooldown period (${metricName})`);
            return;
        }
        
        // Determine if scaling is actually needed
        const shouldScale = this.evaluateScalingNeed(action, context);
        
        if (shouldScale) {
            this.executeScaling(metricName, action, context);
        }
    }
    
    evaluateScalingNeed(action, context) {
        // Get current replica count (simulated for now)
        const currentReplicas = this.getCurrentReplicaCount();
        
        if (action === 'scale_up' && currentReplicas >= this.config.maxReplicas) {
            console.log(`⚠️ Cannot scale up: already at maximum replicas (${currentReplicas})`);
            return false;
        }
        
        if (action === 'scale_down' && currentReplicas <= this.config.minReplicas) {
            console.log(`⚠️ Cannot scale down: already at minimum replicas (${currentReplicas})`);
            return false;
        }
        
        return true;
    }
    
    executeScaling(metricName, action, context) {
        const scalingEvent = {
            id: `scaling_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            metric: metricName,
            action,
            context,
            status: 'initiated'
        };
        
        this.scalingHistory.push(scalingEvent);
        this.lastScalingEvent = Date.now();
        
        console.log(`🔄 Initiating scaling action: ${action} (triggered by ${metricName})`);
        
        // Emit scaling event for external systems to handle
        this.emit('scalingTriggered', scalingEvent);
        
        // For demonstration, simulate scaling execution
        this.simulateScalingExecution(scalingEvent);
    }
    
    async simulateScalingExecution(scalingEvent) {
        // Simulate scaling execution time
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        scalingEvent.status = 'completed';
        scalingEvent.completedAt = new Date().toISOString();
        
        this.createAlert('scaling', 'info', `Scaling ${scalingEvent.action} completed`, {
            metric: scalingEvent.metric,
            scalingId: scalingEvent.id
        });
        
        console.log(`✅ Scaling action completed: ${scalingEvent.action}`);
        this.emit('scalingCompleted', scalingEvent);
    }
    
    getCurrentReplicaCount() {
        // In a real implementation, this would query the orchestrator (Kubernetes, Docker Swarm, etc.)
        // For now, simulate based on agent health
        const agentMetrics = this.metrics.get('agents');
        if (agentMetrics && agentMetrics.length > 0) {
            const latest = agentMetrics[agentMetrics.length - 1];
            return latest.agents.healthy;
        }
        return this.config.minReplicas;
    }
    
    analyzePerformance() {
        // Comprehensive performance analysis
        const analysis = {
            timestamp: new Date().toISOString(),
            system: this.analyzeSystemPerformance(),
            agents: this.analyzeAgentPerformance(),
            trends: this.analyzeTrends(),
            recommendations: []
        };
        
        // Generate recommendations
        analysis.recommendations = this.generateRecommendations(analysis);
        
        // Store analysis
        this.storeMetric('analysis', analysis);
        
        // Emit analysis for external consumption
        this.emit('performanceAnalysis', analysis);
    }
    
    analyzeSystemPerformance() {
        const systemMetrics = this.metrics.get('system') || [];
        if (systemMetrics.length === 0) return {};
        
        const latest = systemMetrics[systemMetrics.length - 1].system;
        
        return {
            current: latest,
            status: this.getSystemStatus(latest),
            score: this.calculateSystemScore(latest)
        };
    }
    
    analyzeAgentPerformance() {
        const agentMetrics = this.metrics.get('agents') || [];
        if (agentMetrics.length === 0) return {};
        
        const latest = agentMetrics[agentMetrics.length - 1].agents;
        
        return {
            current: latest,
            status: this.getAgentStatus(latest),
            score: this.calculateAgentScore(latest)
        };
    }
    
    analyzeTrends() {
        // Analyze trends over the last 10 data points
        const trends = {};
        
        ['system', 'agents'].forEach(category => {
            const data = this.metrics.get(category) || [];
            if (data.length >= 2) {
                trends[category] = this.calculateTrend(data.slice(-10));
            }
        });
        
        return trends;
    }
    
    calculateTrend(data) {
        if (data.length < 2) return { trend: 'insufficient_data' };
        
        // Simple trend calculation based on first and last values
        const first = data[0];
        const last = data[data.length - 1];
        
        const trend = {};
        
        if (first.system && last.system) {
            trend.cpu = this.getTrendDirection(first.system.cpu.usage, last.system.cpu.usage);
            trend.memory = this.getTrendDirection(first.system.memory.usagePercentage, last.system.memory.usagePercentage);
        }
        
        if (first.agents && last.agents) {
            trend.agentHealth = this.getTrendDirection(first.agents.healthy, last.agents.healthy);
            if (first.agents.averages && last.agents.averages) {
                trend.responseTime = this.getTrendDirection(last.agents.averages.responseTime, first.agents.averages.responseTime, true);
            }
        }
        
        return trend;
    }
    
    getTrendDirection(oldValue, newValue, inverse = false) {
        if (oldValue === null || newValue === null) return 'unknown';
        
        const change = ((newValue - oldValue) / oldValue) * 100;
        const threshold = 5; // 5% change threshold
        
        if (Math.abs(change) < threshold) return 'stable';
        
        if (inverse) {
            return change > threshold ? 'degrading' : 'improving';
        } else {
            return change > threshold ? 'increasing' : 'decreasing';
        }
    }
    
    getSystemStatus(systemMetrics) {
        const { cpu, memory } = systemMetrics;
        
        if (cpu.usage > 90 || memory.usagePercentage > 95) {
            return 'critical';
        } else if (cpu.usage > 75 || memory.usagePercentage > 80) {
            return 'warning';
        } else if (cpu.usage > 50 || memory.usagePercentage > 60) {
            return 'moderate';
        } else {
            return 'healthy';
        }
    }
    
    getAgentStatus(agentMetrics) {
        const healthPercentage = (agentMetrics.healthy / agentMetrics.total) * 100;
        
        if (healthPercentage < 50) {
            return 'critical';
        } else if (healthPercentage < 80) {
            return 'warning';
        } else if (healthPercentage < 95) {
            return 'moderate';
        } else {
            return 'healthy';
        }
    }
    
    calculateSystemScore(systemMetrics) {
        const { cpu, memory } = systemMetrics;
        
        // Score from 0-100 based on resource utilization
        const cpuScore = Math.max(0, 100 - cpu.usage);
        const memoryScore = Math.max(0, 100 - memory.usagePercentage);
        
        return Math.round((cpuScore + memoryScore) / 2);
    }
    
    calculateAgentScore(agentMetrics) {
        const healthPercentage = (agentMetrics.healthy / agentMetrics.total) * 100;
        const averages = agentMetrics.averages || {};
        
        let score = healthPercentage;
        
        // Adjust score based on performance metrics
        if (averages.successRate !== undefined) {
            score = (score + averages.successRate) / 2;
        }
        
        if (averages.responseTime !== undefined) {
            const responseTimeScore = Math.max(0, 100 - (averages.responseTime / 50)); // 50ms = 0 penalty
            score = (score + responseTimeScore) / 2;
        }
        
        return Math.round(Math.max(0, Math.min(100, score)));
    }
    
    generateRecommendations(analysis) {
        const recommendations = [];
        
        // System recommendations
        if (analysis.system.status === 'critical') {
            recommendations.push({
                type: 'system',
                priority: 'high',
                title: 'Critical System Resources',
                description: 'System resources are critically low. Immediate scaling or optimization required.',
                actions: ['Scale up immediately', 'Review resource usage', 'Check for resource leaks']
            });
        }
        
        // Agent recommendations
        if (analysis.agents.status === 'critical') {
            recommendations.push({
                type: 'agents',
                priority: 'high',
                title: 'Agent Cluster Health Critical',
                description: 'Multiple agents are unhealthy. Check agent logs and consider restart.',
                actions: ['Restart unhealthy agents', 'Review agent logs', 'Check network connectivity']
            });
        }
        
        // Performance recommendations
        if (analysis.trends.system && analysis.trends.system.cpu === 'increasing') {
            recommendations.push({
                type: 'performance',
                priority: 'medium',
                title: 'CPU Usage Trending Up',
                description: 'CPU usage has been increasing. Monitor closely for scaling needs.',
                actions: ['Monitor CPU trends', 'Prepare for scaling', 'Optimize CPU-intensive tasks']
            });
        }
        
        return recommendations;
    }
    
    createAlert(category, severity, message, metadata = {}) {
        const alert = {
            id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            category,
            severity,
            message,
            metadata,
            acknowledged: false,
            resolved: false
        };
        
        this.alerts.unshift(alert);
        
        // Keep only last 500 alerts
        if (this.alerts.length > 500) {
            this.alerts = this.alerts.slice(0, 500);
        }
        
        console.log(`🚨 Alert [${severity.toUpperCase()}]: ${message}`);
        this.emit('alert', alert);
        
        return alert;
    }
    
    async persistMetrics() {
        try {
            const metricsData = {};
            for (const [category, data] of this.metrics) {
                // Keep only last 100 data points for persistence
                metricsData[category] = data.slice(-100);
            }
            
            const metricsFile = path.join(this.config.metricsPath, `metrics_${new Date().toISOString().split('T')[0]}.json`);
            await fs.writeFile(metricsFile, JSON.stringify(metricsData, null, 2));
            
            const alertsFile = path.join(this.config.alertsPath, 'alerts.json');
            await fs.writeFile(alertsFile, JSON.stringify(this.alerts.slice(0, 100), null, 2));
            
        } catch (error) {
            console.error('Failed to persist metrics:', error);
        }
    }
    
    cleanupOldAlerts() {
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
        
        const initialCount = this.alerts.length;
        this.alerts = this.alerts.filter(alert => 
            new Date(alert.timestamp) > oneWeekAgo
        );
        
        const cleanedCount = initialCount - this.alerts.length;
        if (cleanedCount > 0) {
            console.log(`🧹 Cleaned up ${cleanedCount} old alerts`);
        }
    }
    
    // Public API methods
    getMetrics(category, limit = 100) {
        const data = this.metrics.get(category) || [];
        return data.slice(-limit);
    }
    
    getCurrentMetrics() {
        const result = {};
        for (const [category, data] of this.metrics) {
            if (data.length > 0) {
                result[category] = data[data.length - 1];
            }
        }
        return result;
    }
    
    getAlerts(options = {}) {
        let filtered = [...this.alerts];
        
        if (options.severity) {
            filtered = filtered.filter(a => a.severity === options.severity);
        }
        
        if (options.category) {
            filtered = filtered.filter(a => a.category === options.category);
        }
        
        if (options.unacknowledged) {
            filtered = filtered.filter(a => !a.acknowledged);
        }
        
        if (options.limit) {
            filtered = filtered.slice(0, options.limit);
        }
        
        return filtered;
    }
    
    getScalingHistory(limit = 50) {
        return this.scalingHistory.slice(-limit);
    }
    
    acknowledgeAlert(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert) {
            alert.acknowledged = true;
            alert.acknowledgedAt = new Date().toISOString();
            return alert;
        }
        return null;
    }
    
    resolveAlert(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert) {
            alert.resolved = true;
            alert.resolvedAt = new Date().toISOString();
            return alert;
        }
        return null;
    }
    
    getMonitoringStatus() {
        return {
            isMonitoring: this.isMonitoring,
            config: this.config,
            metrics: {
                categories: Array.from(this.metrics.keys()),
                totalDataPoints: Array.from(this.metrics.values()).reduce((sum, data) => sum + data.length, 0)
            },
            alerts: {
                total: this.alerts.length,
                unacknowledged: this.alerts.filter(a => !a.acknowledged).length,
                critical: this.alerts.filter(a => a.severity === 'critical').length
            },
            scaling: {
                totalEvents: this.scalingHistory.length,
                lastEvent: this.scalingHistory.length > 0 ? this.scalingHistory[this.scalingHistory.length - 1] : null,
                cooldownRemaining: this.lastScalingEvent ? 
                    Math.max(0, this.config.cooldownPeriod - (Date.now() - this.lastScalingEvent)) : 0
            }
        };
    }
}

module.exports = PerformanceMonitor;