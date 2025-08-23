const cds = require('@sap/cds');
const { MockRegistry } = require('./services/MockRegistry');
const { TestOrchestrator } = require('./services/TestOrchestrator');
const { AgentSimulator } = require('./services/AgentSimulator');
const { ServiceMocker } = require('./services/ServiceMocker');
const { DevMonitor } = require('./services/DevMonitor');
const { HotReloader } = require('./services/HotReloader');
const logger = require('./utils/logger');

const handleCdsServiceImplementation = async function() {
    // Initialize services
    const mockRegistry = new MockRegistry();
    const testOrchestrator = new TestOrchestrator();
    const agentSimulator = new AgentSimulator();
    const serviceMocker = new ServiceMocker();
    const devMonitor = new DevMonitor();
    const hotReloader = new HotReloader();
    
    // Initialize all services on startup
    const handleServiceInitialization = async () => {
        await mockRegistry.initialize();
        await testOrchestrator.initialize();
        await agentSimulator.initialize();
        await serviceMocker.initialize();
        await devMonitor.initialize();
        await hotReloader.initialize();
        logger.info('All dev services initialized successfully');
    };
    this.on('serving', handleServiceInitialization);
    
    // Mock Registry handlers
    const handleReadAgents = async () => {
        return await mockRegistry.getAllAgents();
    };
    this.on('READ', 'Agents', handleReadAgents);
    
    const handleRegisterAgent = async (req) => {
        const result = await mockRegistry.registerAgent(JSON.parse(req.data.agentData));
        return JSON.stringify(result);
    };
    this.on('registerAgent', handleRegisterAgent);
    
    const handleDiscoverAgents = async (req) => {
        return await mockRegistry.discoverAgents(req.data.capability);
    };
    this.on('discoverAgents', handleDiscoverAgents);
    
    // Test Orchestration handlers
    const handleRunTestSuite = async (req) => {
        const result = await testOrchestrator.runTestSuite(JSON.parse(req.data.testData));
        return result;
    };
    this.on('runTestSuite', handleRunTestSuite);
    
    const handleTestAgent = async (req) => {
        const result = await testOrchestrator.testAgent(req.data.agentId, JSON.parse(req.data.testConfig));
        return JSON.stringify(result);
    };
    this.on('testAgent', handleTestAgent);
    
    const handleGetTestResults = async (req) => {
        return await testOrchestrator.getTestResults(req.data.testId);
    };
    this.on('getTestResults', handleGetTestResults);
    
    // Agent Simulation handlers
    const handleRunScenario = async (req) => {
        const result = await agentSimulator.runScenario(JSON.parse(req.data.scenarioData));
        return JSON.stringify(result);
    };
    this.on('runScenario', handleRunScenario);
    
    const handleGetAvailableScenarios = async () => {
        return await agentSimulator.getAvailableScenarios();
    };
    this.on('getAvailableScenarios', handleGetAvailableScenarios);
    
    const handleRunLoadTest = async (req) => {
        const result = await agentSimulator.runLoadTest(JSON.parse(req.data.loadTestConfig));
        return JSON.stringify(result);
    };
    this.on('runLoadTest', handleRunLoadTest);
    
    // Service Mocking handlers
    const handleReadMockServices = async () => {
        return await serviceMocker.getAllMocks();
    };
    this.on('READ', 'MockServices', handleReadMockServices);
    
    const handleCreateMock = async (req) => {
        return await serviceMocker.createMock(JSON.parse(req.data.mockData));
    };
    this.on('createMock', handleCreateMock);
    
    const handleUpdateMock = async (req) => {
        return await serviceMocker.updateMock(req.data.mockId, JSON.parse(req.data.mockData));
    };
    this.on('updateMock', handleUpdateMock);
    
    const handleDeleteMock = async (req) => {
        await serviceMocker.deleteMock(req.data.mockId);
        return true;
    };
    this.on('deleteMock', handleDeleteMock);
    
    // Development Monitoring handlers
    const handleGetMetrics = async () => {
        return await devMonitor.getMetrics();
    };
    this.on('getMetrics', handleGetMetrics);
    
    const handleGetLogs = async (req) => {
        const filters = req.data.filters ? JSON.parse(req.data.filters) : {};
        return await devMonitor.getLogs(filters);
    };
    this.on('getLogs', handleGetLogs);
    
    const handleGetHealthStatus = async () => {
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
    };
    this.on('getHealthStatus', handleGetHealthStatus);
    
    // Graceful shutdown
    const handleGracefulShutdown = async () => {
        logger.info('Received SIGTERM, shutting down dev services gracefully');
        
        await mockRegistry.shutdown();
        await testOrchestrator.shutdown();
        await agentSimulator.shutdown();
        await serviceMocker.shutdown();
        await devMonitor.shutdown();
        await hotReloader.shutdown();
    };
    process.on('SIGTERM', handleGracefulShutdown);
};

module.exports = cds.service.impl(handleCdsServiceImplementation);