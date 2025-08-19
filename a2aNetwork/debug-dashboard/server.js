const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const logger = require('pino')();

const { AgentMonitor } = require('./src/AgentMonitor');
const { NetworkVisualizer } = require('./src/NetworkVisualizer');
const { LogAggregator } = require('./src/LogAggregator');
const { MetricsCollector } = require('./src/MetricsCollector');
const { AlertManager } = require('./src/AlertManager');
const { PerformanceProfiler } = require('./src/PerformanceProfiler');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Initialize services
const agentMonitor = new AgentMonitor(io);
const networkVisualizer = new NetworkVisualizer(io);
const logAggregator = new LogAggregator(io);
const metricsCollector = new MetricsCollector(io);
const alertManager = new AlertManager(io);
const performanceProfiler = new PerformanceProfiler(io);

// Middleware
app.use(helmet({
  contentSecurityPolicy: false // Disable for development
}));
app.use(compression());
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.static('dist'));

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      agentMonitor: agentMonitor.isHealthy(),
      networkVisualizer: networkVisualizer.isHealthy(),
      logAggregator: logAggregator.isHealthy(),
      metricsCollector: metricsCollector.isHealthy(),
      alertManager: alertManager.isHealthy(),
      performanceProfiler: performanceProfiler.isHealthy()
    }
  });
});

// API Routes

// Agent monitoring endpoints
app.get('/api/agents', async (req, res) => {
  try {
    const agents = await agentMonitor.getAllAgents();
    res.json(agents);
  } catch (error) {
    logger.error('Error fetching agents:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/agents/:agentId', async (req, res) => {
  try {
    const agent = await agentMonitor.getAgent(req.params.agentId);
    res.json(agent);
  } catch (error) {
    logger.error('Error fetching agent:', error);
    res.status(404).json({ error: error.message });
  }
});

app.get('/api/agents/:agentId/status', async (req, res) => {
  try {
    const status = await agentMonitor.getAgentStatus(req.params.agentId);
    res.json(status);
  } catch (error) {
    logger.error('Error fetching agent status:', error);
    res.status(404).json({ error: error.message });
  }
});

app.post('/api/agents/:agentId/action', async (req, res) => {
  try {
    const result = await agentMonitor.executeAction(req.params.agentId, req.body);
    res.json(result);
  } catch (error) {
    logger.error('Error executing agent action:', error);
    res.status(400).json({ error: error.message });
  }
});

// Network visualization endpoints
app.get('/api/network/topology', async (req, res) => {
  try {
    const topology = await networkVisualizer.getTopology();
    res.json(topology);
  } catch (error) {
    logger.error('Error fetching network topology:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/network/flows', async (req, res) => {
  try {
    const flows = await networkVisualizer.getMessageFlows(req.query);
    res.json(flows);
  } catch (error) {
    logger.error('Error fetching message flows:', error);
    res.status(500).json({ error: error.message });
  }
});

// Logging endpoints
app.get('/api/logs', async (req, res) => {
  try {
    const logs = await logAggregator.getLogs(req.query);
    res.json(logs);
  } catch (error) {
    logger.error('Error fetching logs:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/logs/agents/:agentId', async (req, res) => {
  try {
    const logs = await logAggregator.getAgentLogs(req.params.agentId, req.query);
    res.json(logs);
  } catch (error) {
    logger.error('Error fetching agent logs:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/logs/search', async (req, res) => {
  try {
    const results = await logAggregator.searchLogs(req.body);
    res.json(results);
  } catch (error) {
    logger.error('Error searching logs:', error);
    res.status(500).json({ error: error.message });
  }
});

// Metrics endpoints
app.get('/api/metrics', async (req, res) => {
  try {
    const metrics = await metricsCollector.getMetrics(req.query);
    res.json(metrics);
  } catch (error) {
    logger.error('Error fetching metrics:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/metrics/agents/:agentId', async (req, res) => {
  try {
    const metrics = await metricsCollector.getAgentMetrics(req.params.agentId, req.query);
    res.json(metrics);
  } catch (error) {
    logger.error('Error fetching agent metrics:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/metrics/system', async (req, res) => {
  try {
    const metrics = await metricsCollector.getSystemMetrics(req.query);
    res.json(metrics);
  } catch (error) {
    logger.error('Error fetching system metrics:', error);
    res.status(500).json({ error: error.message });
  }
});

// Alert endpoints
app.get('/api/alerts', async (req, res) => {
  try {
    const alerts = await alertManager.getAlerts(req.query);
    res.json(alerts);
  } catch (error) {
    logger.error('Error fetching alerts:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/alerts', async (req, res) => {
  try {
    const alert = await alertManager.createAlert(req.body);
    res.json(alert);
  } catch (error) {
    logger.error('Error creating alert:', error);
    res.status(400).json({ error: error.message });
  }
});

app.put('/api/alerts/:alertId', async (req, res) => {
  try {
    const alert = await alertManager.updateAlert(req.params.alertId, req.body);
    res.json(alert);
  } catch (error) {
    logger.error('Error updating alert:', error);
    res.status(400).json({ error: error.message });
  }
});

app.delete('/api/alerts/:alertId', async (req, res) => {
  try {
    await alertManager.deleteAlert(req.params.alertId);
    res.json({ success: true });
  } catch (error) {
    logger.error('Error deleting alert:', error);
    res.status(400).json({ error: error.message });
  }
});

// Performance profiling endpoints
app.get('/api/profiling/agents/:agentId', async (req, res) => {
  try {
    const profile = await performanceProfiler.getAgentProfile(req.params.agentId);
    res.json(profile);
  } catch (error) {
    logger.error('Error fetching agent profile:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/profiling/agents/:agentId/start', async (req, res) => {
  try {
    const result = await performanceProfiler.startProfiling(req.params.agentId, req.body);
    res.json(result);
  } catch (error) {
    logger.error('Error starting profiling:', error);
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/profiling/agents/:agentId/stop', async (req, res) => {
  try {
    const result = await performanceProfiler.stopProfiling(req.params.agentId);
    res.json(result);
  } catch (error) {
    logger.error('Error stopping profiling:', error);
    res.status(400).json({ error: error.message });
  }
});

// WebSocket connections for real-time updates
io.on('connection', (socket) => {
  logger.info('Client connected to debug dashboard', { socketId: socket.id });

  // Subscribe to agent updates
  socket.on('subscribe:agents', () => {
    agentMonitor.subscribeClient(socket);
  });

  // Subscribe to network updates
  socket.on('subscribe:network', () => {
    networkVisualizer.subscribeClient(socket);
  });

  // Subscribe to log streams
  socket.on('subscribe:logs', (filters) => {
    logAggregator.subscribeClient(socket, filters);
  });

  // Subscribe to metrics
  socket.on('subscribe:metrics', (config) => {
    metricsCollector.subscribeClient(socket, config);
  });

  // Subscribe to alerts
  socket.on('subscribe:alerts', () => {
    alertManager.subscribeClient(socket);
  });

  // Agent control commands
  socket.on('agent:restart', async (agentId) => {
    try {
      const result = await agentMonitor.restartAgent(agentId);
      socket.emit('agent:action-result', { action: 'restart', agentId, result });
    } catch (error) {
      socket.emit('agent:action-error', { action: 'restart', agentId, error: error.message });
    }
  });

  socket.on('agent:stop', async (agentId) => {
    try {
      const result = await agentMonitor.stopAgent(agentId);
      socket.emit('agent:action-result', { action: 'stop', agentId, result });
    } catch (error) {
      socket.emit('agent:action-error', { action: 'stop', agentId, error: error.message });
    }
  });

  socket.on('agent:debug', async (agentId) => {
    try {
      const result = await agentMonitor.enableDebugMode(agentId);
      socket.emit('agent:action-result', { action: 'debug', agentId, result });
    } catch (error) {
      socket.emit('agent:action-error', { action: 'debug', agentId, error: error.message });
    }
  });

  // Message injection for testing
  socket.on('network:inject-message', async (messageData) => {
    try {
      const result = await networkVisualizer.injectMessage(messageData);
      socket.emit('network:message-injected', result);
    } catch (error) {
      socket.emit('network:injection-error', { error: error.message });
    }
  });

  // Performance profiling controls
  socket.on('profiling:start', async (agentId, config) => {
    try {
      const result = await performanceProfiler.startProfiling(agentId, config);
      socket.emit('profiling:started', { agentId, result });
    } catch (error) {
      socket.emit('profiling:error', { agentId, error: error.message });
    }
  });

  socket.on('profiling:stop', async (agentId) => {
    try {
      const result = await performanceProfiler.stopProfiling(agentId);
      socket.emit('profiling:stopped', { agentId, result });
    } catch (error) {
      socket.emit('profiling:error', { agentId, error: error.message });
    }
  });

  // Cleanup on disconnect
  socket.on('disconnect', () => {
    logger.info('Client disconnected from debug dashboard', { socketId: socket.id });
    
    agentMonitor.unsubscribeClient(socket);
    networkVisualizer.unsubscribeClient(socket);
    logAggregator.unsubscribeClient(socket);
    metricsCollector.unsubscribeClient(socket);
    alertManager.unsubscribeClient(socket);
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: error.message
  });
});

// Serve the dashboard UI
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Initialize services
async function initializeServices() {
  try {
    await agentMonitor.initialize();
    await networkVisualizer.initialize();
    await logAggregator.initialize();
    await metricsCollector.initialize();
    await alertManager.initialize();
    await performanceProfiler.initialize();
    
    logger.info('All debug dashboard services initialized');
  } catch (error) {
    logger.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  
  await agentMonitor.shutdown();
  await networkVisualizer.shutdown();
  await logAggregator.shutdown();
  await metricsCollector.shutdown();
  await alertManager.shutdown();
  await performanceProfiler.shutdown();
  
  server.close(() => {
    logger.info('Debug dashboard server closed');
    process.exit(0);
  });
});

// Start the server
const PORT = process.env.PORT || 8080;

initializeServices().then(() => {
  server.listen(PORT, () => {
    logger.info(`🐛 A2A Debug Dashboard running on port ${PORT}`);
    logger.info(`🌐 Access at: http://localhost:${PORT}`);
    logger.info('Features available:');
    logger.info('  - Real-time agent monitoring');
    logger.info('  - Network topology visualization');
    logger.info('  - Log aggregation and search');
    logger.info('  - Performance metrics');
    logger.info('  - Alert management');
    logger.info('  - Performance profiling');
  });
});

module.exports = { app, server, io };