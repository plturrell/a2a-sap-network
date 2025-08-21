#!/usr/bin/env node
/**
 * Static File Server for A2A Network Development
 * Serves static files and test pages on port 4005
 */

const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.STATIC_SERVER_PORT || 4005;

// Enable CORS for development
app.use(cors({
  origin: ['http://localhost:4004', 'http://localhost:8080'],
  credentials: true
}));

// Serve static files from app directory
app.use('/app', express.static(path.join(__dirname, 'app')));

// Serve test files
app.use('/test', express.static(path.join(__dirname, 'test')));

// Serve main launchpad
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'app', 'launchpad.html'));
});

// Test launchpad removed - use main launchpad for testing

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    server: 'static-file-server',
    port: PORT,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`📁 Static file server running on http://localhost:${PORT}`);
  console.log(`📄 Main launchpad: http://localhost:${PORT}/`);
  console.log(`🧪 Test environment: http://localhost:${PORT}/app/a2aFiori/`);
  console.log(`❤️  Health check: http://localhost:${PORT}/health`);
});

module.exports = app;