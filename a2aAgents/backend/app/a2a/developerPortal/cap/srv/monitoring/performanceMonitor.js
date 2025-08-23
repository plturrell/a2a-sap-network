"use strict";

/**
 * Performance Monitoring Service for SAP A2A Developer Portal
 * Real-time performance metrics collection and analysis
 */

const { PerformanceObserver, performance } = require('perf_hooks');
const prometheus = require('prom-client');
const os = require('os');
const v8 = require('v8');

class PerformanceMonitor {
    constructor() {
        // Initialize Prometheus metrics
        this.register = new prometheus.Registry();
        
        // Default metrics (CPU, memory, etc.)
        prometheus.collectDefaultMetrics({ 
            register: this.register,
            prefix: 'a2a_portal_'
        });

        // Custom metrics
        this._initializeCustomMetrics();
        
        // Performance observers
        this._initializeObservers();
        
        // Real-time metrics storage
        this.realtimeMetrics = {
            requests: new Map(),
            operations: new Map(),
            resources: new Map()
        };

        // Start periodic collection
        this._startPeriodicCollection();
    }

    /**
     * Initialize custom Prometheus metrics
     */
    _initializeCustomMetrics() {
        // HTTP request metrics
        this.httpRequestDuration = new prometheus.Histogram({
            name: 'a2a_portal_http_request_duration_milliseconds',
            help: 'Duration of HTTP requests in milliseconds',
            labelNames: ['method', 'route', 'status_code'],
            buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000]
        });
        this.register.registerMetric(this.httpRequestDuration);

        this.httpRequestsTotal = new prometheus.Counter({
            name: 'a2a_portal_http_requests_total',
            help: 'Total number of HTTP requests',
            labelNames: ['method', 'route', 'status_code']
        });
        this.register.registerMetric(this.httpRequestsTotal);

        // Agent execution metrics
        this.agentExecutionDuration = new prometheus.Histogram({
            name: 'a2a_portal_agent_execution_duration_seconds',
            help: 'Duration of agent executions in seconds',
            labelNames: ['agent_type', 'agent_id', 'status'],
            buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60]
        });
        this.register.registerMetric(this.agentExecutionDuration);

        this.agentExecutionsTotal = new prometheus.Counter({
            name: 'a2a_portal_agent_executions_total',
            help: 'Total number of agent executions',
            labelNames: ['agent_type', 'agent_id', 'status']
        });
        this.register.registerMetric(this.agentExecutionsTotal);

        // Workflow metrics
        this.workflowExecutionDuration = new prometheus.Histogram({
            name: 'a2a_portal_workflow_execution_duration_seconds',
            help: 'Duration of workflow executions in seconds',
            labelNames: ['workflow_type', 'workflow_id', 'status'],
            buckets: [1, 5, 10, 30, 60, 120, 300, 600]
        });
        this.register.registerMetric(this.workflowExecutionDuration);

        // Database metrics
        this.dbQueryDuration = new prometheus.Histogram({
            name: 'a2a_portal_db_query_duration_milliseconds',
            help: 'Duration of database queries in milliseconds',
            labelNames: ['operation', 'table', 'status'],
            buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000]
        });
        this.register.registerMetric(this.dbQueryDuration);

        this.dbConnectionsActive = new prometheus.Gauge({
            name: 'a2a_portal_db_connections_active',
            help: 'Number of active database connections'
        });
        this.register.registerMetric(this.dbConnectionsActive);

        // Cache metrics
        this.cacheHits = new prometheus.Counter({
            name: 'a2a_portal_cache_hits_total',
            help: 'Total number of cache hits',
            labelNames: ['cache_type', 'operation']
        });
        this.register.registerMetric(this.cacheHits);

        this.cacheMisses = new prometheus.Counter({
            name: 'a2a_portal_cache_misses_total',
            help: 'Total number of cache misses',
            labelNames: ['cache_type', 'operation']
        });
        this.register.registerMetric(this.cacheMisses);

        // Business metrics
        this.activeProjects = new prometheus.Gauge({
            name: 'a2a_portal_active_projects',
            help: 'Number of active projects',
            labelNames: ['business_unit', 'department']
        });
        this.register.registerMetric(this.activeProjects);

        this.activeAgents = new prometheus.Gauge({
            name: 'a2a_portal_active_agents',
            help: 'Number of active agents',
            labelNames: ['agent_type', 'status']
        });
        this.register.registerMetric(this.activeAgents);

        // Error metrics
        this.errorsTotal = new prometheus.Counter({
            name: 'a2a_portal_errors_total',
            help: 'Total number of errors',
            labelNames: ['error_type', 'component', 'severity']
        });
        this.register.registerMetric(this.errorsTotal);

        // Resource utilization
        this.memoryUsage = new prometheus.Gauge({
            name: 'a2a_portal_memory_usage_bytes',
            help: 'Memory usage in bytes',
            labelNames: ['type']
        });
        this.register.registerMetric(this.memoryUsage);

        this.cpuUsage = new prometheus.Gauge({
            name: 'a2a_portal_cpu_usage_percent',
            help: 'CPU usage percentage'
        });
        this.register.registerMetric(this.cpuUsage);
    }

    /**
     * Initialize performance observers
     */
    _initializeObservers() {
        // HTTP timing observer
        const httpObs = new PerformanceObserver((items) => {
            items.getEntries().forEach((entry) => {
                if (entry.name.startsWith('HTTP')) {
                    this._processHttpEntry(entry);
                }
            });
        });
        httpObs.observe({ entryTypes: ['measure'] });

        // Resource timing observer
        const resourceObs = new PerformanceObserver((items) => {
            items.getEntries().forEach((entry) => {
                this._processResourceEntry(entry);
            });
        });
        resourceObs.observe({ entryTypes: ['resource'] });

        // Function timing observer
        const functionObs = new PerformanceObserver((items) => {
            items.getEntries().forEach((entry) => {
                if (entry.name.startsWith('function')) {
                    this._processFunctionEntry(entry);
                }
            });
        });
        functionObs.observe({ entryTypes: ['function'] });
    }

    /**
     * Start periodic metrics collection
     */
    _startPeriodicCollection() {
        // Collect system metrics every 10 seconds
        setInterval(() => {
            this._collectSystemMetrics();
        }, 10000);

        // Collect business metrics every 30 seconds
        setInterval(() => {
            this._collectBusinessMetrics();
        }, 30000);

        // Clean up old real-time metrics every minute
        setInterval(() => {
            this._cleanupOldMetrics();
        }, 60000);
    }

    /**
     * Collect system metrics
     */
    _collectSystemMetrics() {
        // Memory metrics
        const memUsage = process.memoryUsage();
        this.memoryUsage.set({ type: 'heapUsed' }, memUsage.heapUsed);
        this.memoryUsage.set({ type: 'heapTotal' }, memUsage.heapTotal);
        this.memoryUsage.set({ type: 'rss' }, memUsage.rss);
        this.memoryUsage.set({ type: 'external' }, memUsage.external);

        // V8 heap statistics
        const heapStats = v8.getHeapStatistics();
        this.memoryUsage.set({ type: 'v8HeapUsed' }, heapStats.used_heap_size);
        this.memoryUsage.set({ type: 'v8HeapTotal' }, heapStats.total_heap_size);

        // CPU usage
        const cpuUsage = process.cpuUsage();
        const totalCpuTime = cpuUsage.user + cpuUsage.system;
        const cpuPercent = (totalCpuTime / 1000000) / os.cpus().length;
        this.cpuUsage.set(Math.min(cpuPercent, 100));

        // Event loop lag (indicates if Node.js is overloaded)
        const lagStart = Date.now();
        setImmediate(() => {
            const lag = Date.now() - lagStart;
            this.register.getSingleMetric('a2a_portal_event_loop_lag_milliseconds')?.set(lag);
        });
    }

    /**
     * Collect business metrics
     */
    _collectBusinessMetrics() {
        try {
            // This would query the database for real metrics
            // For now, using placeholder values
            this.activeProjects.set({ business_unit: 'BU_001', department: 'IT' }, 12);
            this.activeProjects.set({ business_unit: 'BU_002', department: 'Finance' }, 8);
            
            this.activeAgents.set({ agent_type: 'reactive', status: 'deployed' }, 25);
            this.activeAgents.set({ agent_type: 'proactive', status: 'deployed' }, 18);
            this.activeAgents.set({ agent_type: 'collaborative', status: 'deployed' }, 7);
        } catch (error) {
            console.error('Failed to collect business metrics:', error);
            this.errorsTotal.inc({ 
                error_type: 'metrics_collection', 
                component: 'performance_monitor',
                severity: 'warning'
            });
        }
    }

    /**
     * Clean up old real-time metrics
     */
    _cleanupOldMetrics() {
        const now = Date.now();
        const maxAge = 5 * 60 * 1000; // 5 minutes

        // Clean up request metrics
        for (const [id, metric] of this.realtimeMetrics.requests) {
            if (now - metric.timestamp > maxAge) {
                this.realtimeMetrics.requests.delete(id);
            }
        }

        // Clean up operation metrics
        for (const [id, metric] of this.realtimeMetrics.operations) {
            if (now - metric.timestamp > maxAge) {
                this.realtimeMetrics.operations.delete(id);
            }
        }
    }

    /**
     * Process HTTP performance entry
     */
    _processHttpEntry(entry) {
        const duration = entry.duration;
        const labels = this._extractHttpLabels(entry);
        
        this.httpRequestDuration.observe(labels, duration);
        this.httpRequestsTotal.inc(labels);
        
        // Store in real-time metrics
        this.realtimeMetrics.requests.set(entry.name, {
            duration,
            timestamp: Date.now(),
            ...labels
        });
    }

    /**
     * Extract labels from HTTP entry
     */
    _extractHttpLabels(entry) {
        const parts = entry.name.split(' ');
        return {
            method: parts[1] || 'UNKNOWN',
            route: parts[2] || '/',
            status_code: parts[3] || '200'
        };
    }

    /**
     * Process resource timing entry
     */
    _processResourceEntry(entry) {
        this.realtimeMetrics.resources.set(entry.name, {
            duration: entry.duration,
            transferSize: entry.transferSize,
            timestamp: Date.now()
        });
    }

    /**
     * Process function timing entry
     */
    _processFunctionEntry(entry) {
        const functionName = entry.name.replace('function:', '');
        this.realtimeMetrics.operations.set(functionName, {
            duration: entry.duration,
            timestamp: Date.now()
        });
    }

    /**
     * Record HTTP request
     */
    recordHttpRequest(method, route, statusCode, duration) {
        const labels = { method, route, status_code: statusCode.toString() };
        this.httpRequestDuration.observe(labels, duration);
        this.httpRequestsTotal.inc(labels);
    }

    /**
     * Record agent execution
     */
    recordAgentExecution(agentType, agentId, status, duration) {
        const labels = { agent_type: agentType, agent_id: agentId, status };
        this.agentExecutionDuration.observe(labels, duration / 1000); // Convert to seconds
        this.agentExecutionsTotal.inc(labels);
    }

    /**
     * Record workflow execution
     */
    recordWorkflowExecution(workflowType, workflowId, status, duration) {
        const labels = { workflow_type: workflowType, workflow_id: workflowId, status };
        this.workflowExecutionDuration.observe(labels, duration / 1000); // Convert to seconds
    }

    /**
     * Record database query
     */
    recordDbQuery(operation, table, status, duration) {
        const labels = { operation, table, status };
        this.dbQueryDuration.observe(labels, duration);
    }

    /**
     * Record cache access
     */
    recordCacheAccess(cacheType, operation, hit) {
        const labels = { cache_type: cacheType, operation };
        if (hit) {
            this.cacheHits.inc(labels);
        } else {
            this.cacheMisses.inc(labels);
        }
    }

    /**
     * Record error
     */
    recordError(errorType, component, severity = 'error') {
        this.errorsTotal.inc({ error_type: errorType, component, severity });
    }

    /**
     * Update database connections
     */
    updateDbConnections(count) {
        this.dbConnectionsActive.set(count);
    }

    /**
     * Get Prometheus metrics
     */
    getMetrics() {
        return this.register.metrics();
    }

    /**
     * Get real-time metrics dashboard data
     */
    getRealtimeMetrics() {
        const now = Date.now();
        const fiveMinutesAgo = now - 5 * 60 * 1000;
        
        // Calculate request rate
        const recentRequests = Array.from(this.realtimeMetrics.requests.values())
            .filter(m => m.timestamp > fiveMinutesAgo);
        const requestRate = recentRequests.length / 5; // requests per minute
        
        // Calculate average response time
        const avgResponseTime = recentRequests.reduce((sum, m) => sum + m.duration, 0) / 
                               (recentRequests.length || 1);
        
        // Get system metrics
        const memUsage = process.memoryUsage();
        const cpuUsage = process.cpuUsage();
        
        return {
            timestamp: new Date().toISOString(),
            system: {
                memory: {
                    heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024), // MB
                    heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024), // MB
                    rss: Math.round(memUsage.rss / 1024 / 1024) // MB
                },
                cpu: {
                    user: Math.round(cpuUsage.user / 1000), // ms
                    system: Math.round(cpuUsage.system / 1000) // ms
                },
                uptime: Math.round(process.uptime()),
                load: os.loadavg()
            },
            requests: {
                rate: Math.round(requestRate * 10) / 10, // requests/min
                avgResponseTime: Math.round(avgResponseTime * 10) / 10, // ms
                total: recentRequests.length,
                byStatus: this._groupByStatus(recentRequests)
            },
            operations: {
                recent: Array.from(this.realtimeMetrics.operations.entries())
                    .filter(([_, m]) => m.timestamp > fiveMinutesAgo)
                    .map(([name, m]) => ({
                        name,
                        duration: Math.round(m.duration * 10) / 10,
                        timestamp: new Date(m.timestamp).toISOString()
                    }))
                    .slice(-10) // Last 10 operations
            }
        };
    }

    /**
     * Group requests by status code
     */
    _groupByStatus(requests) {
        const grouped = {};
        requests.forEach(req => {
            const status = req.status_code || '200';
            grouped[status] = (grouped[status] || 0) + 1;
        });
        return grouped;
    }

    /**
     * Create performance mark
     */
    mark(name) {
        performance.mark(name);
    }

    /**
     * Measure between two marks
     */
    measure(name, startMark, endMark) {
        try {
            performance.measure(name, startMark, endMark);
        } catch (error) {
            console.error('Performance measurement error:', error);
        }
    }

    /**
     * Time an async operation
     */
    async timeOperation(name, operation) {
        const startMark = `${name}-start-${Date.now()}`;
        const endMark = `${name}-end-${Date.now()}`;
        
        this.mark(startMark);
        
        try {
            const result = await operation();
            this.mark(endMark);
            this.measure(`operation:${name}`, startMark, endMark);
            return result;
        } catch (error) {
            this.mark(endMark);
            this.measure(`operation:${name}-error`, startMark, endMark);
            throw error;
        } finally {
            // Clean up marks
            performance.clearMarks(startMark);
            performance.clearMarks(endMark);
        }
    }
}

// Export singleton instance
module.exports = new PerformanceMonitor();