'use strict';
/* No global declarations needed - URL is not used in this file */

/**
 * Minimal A2A Agents Server
 * Works locally and on BTP with minimal configuration
 */

const express = require('express');
const { MinimalBTPConfig } = require('../config/minimalBtpConfig.js');

// Simple configuration
const config = new MinimalBTPConfig();
const port = process.env.PORT || 8080;

// Create Express app
const app = express();

// Basic middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check endpoint
app.get('/health', (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    environment: config.is_btp ? 'btp' : 'local',
    services: {
      hana: config.is_service_available('hana'),
      auth: config.is_service_available('xsuaa') || config.is_service_available('local_auth'),
      cache: config.is_service_available('redis')
    }
  };
    
  res.json(health);
});

// Basic info endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'A2A Agents',
    version: '1.0.0',
    environment: config.is_btp ? 'BTP' : 'Local Development',
    endpoints: {
      health: '/health',
      agents: '/api/agents',
      docs: '/docs'
    }
  });
});

// Agents API placeholder
app.get('/api/agents', (req, res) => {
  res.json({
    message: 'A2A Agents API',
    agents: [
      'calculation-agent',
      'data-processing-agent', 
      'reasoning-agent',
      'sql-agent'
    ]
  });
});

// Error handling
app.use((err, _req, res, _next) => {
  console.error('Error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path
  });
});

// Start server
app.listen(port, () => {
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log(`🚀 A2A Agents running on port ${port}`);
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log(`🌐 Environment: ${config.is_btp ? 'BTP' : 'Local'}`);
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log(`❤️ Health check: ${config.is_btp ? 'Check BTP application URL' : `http://localhost:${port}`}/health`);
    
  if (config.is_btp) {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('✅ BTP services available');
  } else {
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('🔧 Local development mode');
    // eslint-disable-next-line no-console
    // eslint-disable-next-line no-console
    console.log('💡 Set VCAP_SERVICES to enable BTP mode');
  }
});

module.exports = app;