const cds = require('@sap/cds');
const { applySecurityMiddleware } = require('./middleware/security');
const { applyAuthMiddleware, initializeXSUAAStrategy } = require('./middleware/auth');
const monitoring = require('./lib/monitoring');
const cloudALM = require('./lib/cloud-alm');
const { tracing } = require('./lib/distributed-tracing');

// Standard CAP server setup
module.exports = cds.server;

// Apply security middleware and add health endpoint
cds.on('bootstrap', (app) => {
    // Initialize XSUAA authentication strategy
    initializeXSUAAStrategy();
    
    // Apply comprehensive security middleware
    applySecurityMiddleware(app);
    
    // Apply authentication middleware
    applyAuthMiddleware(app);
    
    // Apply distributed tracing middleware
    app.use(tracing.instrumentHTTP());
    
    // Apply monitoring middleware
    app.use(monitoring.middleware());
    
    // User API endpoints for BTP integration
    app.use('/user-api', require('./user-service'));
    
    // Serve UI5 app
    const path = require('path');
    const express = require('express');
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2a-fiori/webapp')));
    
    // Serve launchpad pages
    app.get('/launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/launchpad.html'));
    });
    app.get('/fiori-launchpad.html', (req, res) => {
        res.sendFile(path.join(__dirname, '../app/fiori-launchpad.html'));
    });
    
    // Health endpoint - integrated with monitoring service
    app.get('/health', async (req, res) => {
        const health = monitoring.getHealthStatus();
        res.status(health.status === 'healthy' ? 200 : 503).json(health);
    });
    
    // Metrics endpoint for Prometheus/Cloud ALM scraping
    app.get('/metrics', (req, res) => {
        const metrics = monitoring.getMetrics();
        
        // Convert to Prometheus format
        let output = '';
        for (const [name, metric] of Object.entries(metrics)) {
            const metricName = name.replace(/\./g, '_');
            output += `# TYPE ${metricName} gauge\n`;
            output += `${metricName}`;
            
            // Add tags
            if (metric.tags && Object.keys(metric.tags).length > 0) {
                const labels = Object.entries(metric.tags)
                    .map(([k, v]) => `${k}="${v}"`)
                    .join(',');
                output += `{${labels}}`;
            }
            
            output += ` ${metric.value}\n`;
        }
        
        res.set('Content-Type', 'text/plain');
        res.send(output);
    });
});

// Start periodic tasks after server is listening
cds.on('listening', async () => {
    const log = cds.log('jobs');
    
    // Initialize tracing collector
    try {
        const operationsService = await cds.connect.to('OperationsService');
        const { OperationsServiceCollector } = require('./lib/distributed-tracing');
        tracing.addCollector(new OperationsServiceCollector(operationsService));
        log.info('Distributed tracing collector initialized');
    } catch (error) {
        log.warn('Failed to initialize tracing collector:', error.message);
    }
    
    // Log application startup to monitoring
    monitoring.log('info', 'A2A Network application started', {
        logger: 'server',
        version: process.env.APP_VERSION || '1.0.0',
        nodeVersion: process.version,
        environment: process.env.NODE_ENV || 'development'
    });
    
    // Update network statistics every minute
    setInterval(async () => {
        try {
            const db = await cds.connect.to('db');
            const { NetworkStats, Agents, Services, Messages } = db.entities('a2a.network');
            
            const [totalAgents] = await SELECT.one`count(*) as count`.from(Agents);
            const [activeAgents] = await SELECT.one`count(*) as count`.from(Agents).where({ isActive: true });
            const [totalServices] = await SELECT.one`count(*) as count`.from(Services);
            const [totalMessages] = await SELECT.one`count(*) as count`.from(Messages);
            
            // Record metrics for monitoring
            monitoring.recordMetric('agents.total', totalAgents?.count || 0);
            monitoring.recordMetric('agents.active', activeAgents?.count || 0);
            monitoring.recordMetric('services.total', totalServices?.count || 0);
            monitoring.recordMetric('messages.total', totalMessages?.count || 0);
            
            await INSERT.into(NetworkStats).entries({
                ID: cds.utils.uuid(),
                totalAgents: totalAgents?.count || 0,
                activeAgents: activeAgents?.count || 0,
                totalServices: totalServices?.count || 0,
                totalMessages: totalMessages?.count || 0,
                totalCapabilities: 0,
                totalTransactions: 0,
                averageReputation: 100,
                networkLoad: 0.5,
                gasPrice: 20,
                validFrom: new Date()
            });
            
            log.info('Network stats updated');
        } catch (error) {
            log.error('Network stats update failed:', error);
            monitoring.recordMetric('errors.network_stats', 1);
        }
    }, 60000);
    
    // Flush logs periodically
    setInterval(async () => {
        try {
            await monitoring.flushLogs();
        } catch (error) {
            log.error('Failed to flush logs:', error);
        }
    }, 5000);
    
    // Create Cloud ALM dashboard
    try {
        await cloudALM.createDashboard();
        log.info('Cloud ALM dashboard created');
    } catch (error) {
        log.error('Failed to create Cloud ALM dashboard:', error);
    }
});

// Graceful shutdown
process.on('SIGTERM', () => {
    monitoring.log('info', 'Received SIGTERM signal, shutting down gracefully', {
        logger: 'server'
    });
    
    // Flush any remaining logs
    monitoring.flushLogs().then(() => {
        process.exit(0);
    });
});