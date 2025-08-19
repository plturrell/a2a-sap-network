/**
 * Minimal SAP CAP Server for OpenTelemetry conflict isolation
 * Based on working standard SAP CAP server pattern
 */

const cds = require('@sap/cds');
const express = require('express');
const path = require('path');

module.exports = cds.server;

cds.on('bootstrap', (app) => {
    const log = cds.log('server-minimal');
    log.info('Bootstrap phase started');
    
    // Only add static file serving (proven to work with minimal Express + SAP CDS)
    app.use('/common', express.static(path.join(__dirname, '../common')));
    app.use('/app/a2a-fiori', express.static(path.join(__dirname, '../app/a2aFiori/webapp')));
    app.use('/app/launchpad', express.static(path.join(__dirname, '../app/launchpad')));
    app.use('/shells', express.static(path.join(__dirname, '../shells')));
    app.use('/a2aAgents', express.static(path.join(__dirname, '../a2aAgents')));
    
    log.info('Static file serving configured');
});

cds.on('listening', ({ server, url }) => {
    const log = cds.log('server-minimal');
    log.info('Server listening', { url });
    log.info('Static files available at /common, /app/a2a-fiori, /app/launchpad, /shells, /a2aAgents');
});
