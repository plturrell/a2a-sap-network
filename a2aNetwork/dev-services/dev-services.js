const cds = require('@sap/cds');
const { MockRegistry } = require('./services/MockRegistry');
const { TestOrchestrator } = require('./services/TestOrchestrator');
const { AgentSimulator } = require('./services/AgentSimulator');
const { ServiceMocker } = require('./services/ServiceMocker');
const { DevMonitor } = require('./services/DevMonitor');
const { HotReloader } = require('./services/HotReloader');
const logger = require('./utils/logger');

module.exports = cds.service.impl(async function() {
    // Initialize services
    const mockRegistry = new MockRegistry();
    const testOrchestrator = new TestOrchestrator();
    const agentSimulator = new AgentSimulator();
    const serviceMocker = new ServiceMocker();
    const devMonitor = new DevMonitor();
    const hotReloader = new HotReloader();
    
    // Initialize all services on startup
    this.on('serving', async () => {
        await mockRegistry.initialize();
        await testOrchestrator.initialize();
        await agentSimulator.initialize();
        await serviceMocker.initialize();
        await devMonitor.initialize();
        await hotReloader.initialize();
        logger.info('All dev services initialized successfully');
    });
    
    // Mock Registry handlers
    this.on('READ', 'Agents', async () => {
        return await mockRegistry.getAllAgents();
    });
    
    this.on('registerAgent', async (req) => {
        const result = await mockRegistry.registerAgent(JSON.parse(req.data.agentData));
        return JSON.stringify(result);
    });
    
    this.on('discoverAgents', async (req) => {
        return await mockRegistry.discoverAgents(req.data.capability);
    });
    
    // Test Orchestration handlers
    this.on('runTestSuite', async (req) => {
        const result = await testOrchestrator.runTestSuite(JSON.parse(req.data.testData));
        return result;
    });
    
    this.on('testAgent', async (req) => {
        const result = await testOrchestrator.testAgent(req.data.agentId, JSON.parse(req.data.testConfig));
        return JSON.stringify(result);
    });
    
    this.on('getTestResults', async (req) => {
        return await testOrchestrator.getTestResults(req.data.testId);
    });
    
    // Agent Simulation handlers
    this.on('runScenario', async (req) => {
        const result = await agentSimulator.runScenario(JSON.parse(req.data.scenarioData));
        return JSON.stringify(result);
    });
    
    this.on('getAvailableScenarios', async () => {
        return await agentSimulator.getAvailableScenarios();
    });
    
    this.on('runLoadTest', async (req) => {
        const result = await agentSimulator.runLoadTest(JSON.parse(req.data.loadTestConfig));
        return JSON.stringify(result);
    });
    
    // Service Mocking handlers
    this.on('READ', 'MockServices', async () => {
        return await serviceMocker.getAllMocks();
    });
    
    this.on('createMock', async (req) => {
        return await serviceMocker.createMock(JSON.parse(req.data.mockData));
    });
    
    this.on('updateMock', async (req) => {
        return await serviceMocker.updateMock(req.data.mockId, JSON.parse(req.data.mockData));
    });
    
    this.on('deleteMock', async (req) => {
        await serviceMocker.deleteMock(req.data.mockId);
        return true;
    });
    
    // Development Monitoring handlers
    this.on('getMetrics', async () => {
        return await devMonitor.getMetrics();
    });
    
    this.on('getLogs', async (req) => {
        const filters = req.data.filters ? JSON.parse(req.data.filters) : {};
        return await devMonitor.getLogs(filters);
    });
    
    this.on('getHealthStatus', async () => {
        const health = await devMonitor.getHealthStatus();
        return JSON.stringify({
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
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
        logger.info('Received SIGTERM, shutting down dev services gracefully');
        
        await mockRegistry.shutdown();
        await testOrchestrator.shutdown();
        await agentSimulator.shutdown();
        await serviceMocker.shutdown();
        await devMonitor.shutdown();
        await hotReloader.shutdown();
    });
});