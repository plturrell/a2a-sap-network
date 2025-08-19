const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { MockRegistry } = require('./services/MockRegistry');
const { TestOrchestrator } = require('./services/TestOrchestrator');
const { AgentSimulator } = require('./services/AgentSimulator');
const { ServiceMocker } = require('./services/ServiceMocker');
const { DevMonitor } = require('./services/DevMonitor');
const { HotReloader } = require('./services/HotReloader');
const logger = require('./utils/logger');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Initialize services
const mockRegistry = new MockRegistry();
const testOrchestrator = new TestOrchestrator();
const agentSimulator = new AgentSimulator();
const serviceMocker = new ServiceMocker();
const devMonitor = new DevMonitor(io);
const hotReloader = new HotReloader(io);

// Middleware
app.use(express.json());
app.use(express.static('public'));

// CORS for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

// API Routes

// Mock Registry endpoints
app.get('/registry/agents', async (req, res) => {
  try {
    const agents = await mockRegistry.getAllAgents();
    res.json(agents);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/registry/agents/register', async (req, res) => {
  try {
    const result = await mockRegistry.registerAgent(req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/registry/agents/discover/:capability', async (req, res) => {
  try {
    const agents = await mockRegistry.discoverAgents(req.params.capability);
    res.json(agents);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Test orchestration endpoints
app.post('/test/suite/run', async (req, res) => {
  try {
    const result = await testOrchestrator.runTestSuite(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/test/agent/:agentId', async (req, res) => {
  try {
    const result = await testOrchestrator.testAgent(req.params.agentId, req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/test/results/:testId', async (req, res) => {
  try {
    const result = await testOrchestrator.getTestResults(req.params.testId);
    res.json(result);
  } catch (error) {
    res.status(404).json({ error: error.message });
  }
});

// Agent simulation endpoints
app.post('/simulate/scenario', async (req, res) => {
  try {
    const result = await agentSimulator.runScenario(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/simulate/scenarios', async (req, res) => {
  try {
    const scenarios = await agentSimulator.getAvailableScenarios();
    res.json(scenarios);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/simulate/load-test', async (req, res) => {
  try {
    const result = await agentSimulator.runLoadTest(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Service mocking endpoints
app.post('/mock/service', async (req, res) => {
  try {
    const result = await serviceMocker.createMock(req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/mock/services', async (req, res) => {
  try {
    const mocks = await serviceMocker.getAllMocks();
    res.json(mocks);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.put('/mock/service/:mockId', async (req, res) => {
  try {
    const result = await serviceMocker.updateMock(req.params.mockId, req.body);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.delete('/mock/service/:mockId', async (req, res) => {
  try {
    await serviceMocker.deleteMock(req.params.mockId);
    res.json({ success: true });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Development monitoring endpoints
app.get('/monitor/metrics', async (req, res) => {
  try {
    const metrics = await devMonitor.getMetrics();
    res.json(metrics);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/monitor/logs', async (req, res) => {
  try {
    const logs = await devMonitor.getLogs(req.query);
    res.json(logs);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/monitor/health', async (req, res) => {
  try {
    const health = await devMonitor.getHealthStatus();
    res.json(health);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// WebSocket connections for real-time features
io.on('connection', (socket) => {
  logger.info('Client connected to dev services');

  // Test execution with real-time updates
  socket.on('test:start', async (data) => {
    try {
      const testId = await testOrchestrator.startTestWithUpdates(data, (update) => {
        socket.emit('test:update', update);
      });
      socket.emit('test:started', { testId });
    } catch (error) {
      socket.emit('test:error', { error: error.message });
    }
  });

  // Agent monitoring
  socket.on('monitor:subscribe', (agentId) => {
    devMonitor.subscribeToAgent(socket, agentId);
  });

  socket.on('monitor:unsubscribe', (agentId) => {
    devMonitor.unsubscribeFromAgent(socket, agentId);
  });

  // Hot reload notifications
  socket.on('hotreload:enable', (config) => {
    hotReloader.enableForClient(socket, config);
  });

  socket.on('hotreload:disable', () => {
    hotReloader.disableForClient(socket);
  });

  // Simulation control
  socket.on('simulation:start', async (scenario) => {
    try {
      const simulationId = await agentSimulator.startSimulationWithUpdates(scenario, (update) => {
        socket.emit('simulation:update', update);
      });
      socket.emit('simulation:started', { simulationId });
    } catch (error) {
      socket.emit('simulation:error', { error: error.message });
    }
  });

  socket.on('simulation:stop', async (simulationId) => {
    try {
      await agentSimulator.stopSimulation(simulationId);
      socket.emit('simulation:stopped', { simulationId });
    } catch (error) {
      socket.emit('simulation:error', { error: error.message });
    }
  });

  socket.on('disconnect', () => {
    logger.info('Client disconnected from dev services');
    devMonitor.cleanupClient(socket);
    hotReloader.cleanupClient(socket);
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      mockRegistry: mockRegistry.isHealthy(),
      testOrchestrator: testOrchestrator.isHealthy(),
      agentSimulator: agentSimulator.isHealthy(),
      serviceMocker: serviceMocker.isHealthy(),
      devMonitor: devMonitor.isHealthy(),
      hotReloader: hotReloader.isHealthy()
    }
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

// Start services
async function startServices() {
  try {
    // Initialize all services
    await mockRegistry.initialize();
    await testOrchestrator.initialize();
    await agentSimulator.initialize();
    await serviceMocker.initialize();
    await devMonitor.initialize();
    await hotReloader.initialize();

    logger.info('All services initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  
  await mockRegistry.shutdown();
  await testOrchestrator.shutdown();
  await agentSimulator.shutdown();
  await serviceMocker.shutdown();
  await devMonitor.shutdown();
  await hotReloader.shutdown();
  
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

// Start the server
const PORT = process.env.PORT || 3002;

startServices().then(() => {
  server.listen(PORT, () => {
    logger.info(`🛠️  A2A Development Services running on port ${PORT}`);
    logger.info(`🌐 Access at: http://localhost:${PORT}`);
    logger.info('Services available:');
    logger.info('  - Mock Registry: /registry/*');
    logger.info('  - Test Orchestrator: /test/*');
    logger.info('  - Agent Simulator: /simulate/*');
    logger.info('  - Service Mocker: /mock/*');
    logger.info('  - Dev Monitor: /monitor/*');
    logger.info('  - WebSocket: ws://localhost:' + PORT);
  });
});

module.exports = { app, server, io };