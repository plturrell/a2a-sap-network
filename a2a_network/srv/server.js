const cds = require('@sap/cds');
const { applySecurityMiddleware } = require('./middleware/security');
const { applyAuthMiddleware, initializeXSUAAStrategy } = require('./middleware/auth');

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
    
    // User API endpoints for BTP integration
    app.use('/user-api', require('./user-service'));
    
    // Serve UI5 app
    const path = require('path');
    const express = require('express');
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2a-fiori/webapp')));
    
    // Serve launchpad
    app.use('/launchpad.html', express.static(path.join(__dirname, '../app/launchpad.html')));
    
    // Health endpoint
    app.get('/health', async (req, res) => {
        const services = {
            database: false,
            blockchain: false
        };
        
        try {
            await cds.db.run('SELECT 1');
            services.database = true;
        } catch (e) {
            console.error('Database check failed:', e);
        }
        
        const allHealthy = Object.values(services).every(v => v);
        
        res.status(allHealthy ? 200 : 503).json({
            status: allHealthy ? 'healthy' : 'degraded',
            services,
            timestamp: new Date().toISOString(),
            version: cds.version
        });
    });
});

// Start periodic tasks after server is listening
cds.on('listening', async () => {
    const log = cds.log('jobs');
    
    // Update network statistics every minute
    setInterval(async () => {
        try {
            const db = await cds.connect.to('db');
            const { NetworkStats, Agents, Services, Messages } = db.entities('a2a.network');
            
            const [totalAgents] = await SELECT.one`count(*) as count`.from(Agents);
            const [activeAgents] = await SELECT.one`count(*) as count`.from(Agents).where({ isActive: true });
            const [totalServices] = await SELECT.one`count(*) as count`.from(Services);
            const [totalMessages] = await SELECT.one`count(*) as count`.from(Messages);
            
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
        }
    }, 60000);
});