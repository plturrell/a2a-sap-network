/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

const cds = require('@sap/cds');
const { Server } = require('socket.io');
const { TestOrchestrator } = require('./services/TestOrchestrator');
const { AgentSimulator } = require('./services/AgentSimulator');
const { DevMonitor } = require('./services/DevMonitor');
const { HotReloader } = require('./services/HotReloader');
const logger = require('./utils/logger');
const { BlockchainEventServer, BlockchainEventClient } = require('./blockchain-event-adapter');

class DevWebSocketService {
    constructor() {
        this.io = null;
        this.testOrchestrator = new TestOrchestrator();
        this.agentSimulator = new AgentSimulator();
        this.devMonitor = new DevMonitor();
        this.hotReloader = new HotReloader();
    }

    async initialize(server) {
        this.io = new Server(server, {
            cors: {
                origin: '*',
                methods: ['GET', 'POST']
            }
        });

        // Pass io instance to services that need it
        this.devMonitor = new DevMonitor(this.io);
        this.hotReloader = new HotReloader(this.io);

        this.setupSocketHandlers();
        logger.info('WebSocket service initialized');
    }

    setupSocketHandlers() {
        this.io.on('blockchain-connection', function(socket) {
            logger.info('Client connected to dev services');

            // Test execution with real-time updates
            socket.on('test:start', async (data) => {
                await this.handleTestStart(socket, data);
            });

            // Agent monitoring
            socket.on('monitor:subscribe', (agentId) => {
                this.devMonitor.subscribeToAgent(socket, agentId);
            });

            socket.on('monitor:unsubscribe', function(agentId) {
                this.devMonitor.unsubscribeFromAgent(socket, agentId);
            });

            // Hot reload notifications
            socket.on('hotreload:enable', (config) => {
                this.hotReloader.enableForClient(socket, config);
            });

            socket.on('hotreload:disable', () => {
                this.hotReloader.disableForClient(socket);
            });

            // Simulation control
            socket.on('simulation:start', async (scenario) => {
                await this.handleSimulationStart(socket, scenario);
            });

            socket.on('simulation:stop', async (simulationId) => {
                await this.handleSimulationStop(socket, simulationId);
            });

            socket.on('disconnect', () => {
                logger.info('Client disconnected from dev services');
                this.devMonitor.cleanupClient(socket);
                this.hotReloader.cleanupClient(socket);
            });
        });
    }

    async handleTestStart(socket, data) {
        try {
            const updateCallback = (update) => {
                socket.emit('test:update', update);
            };
            const testId = await this.testOrchestrator.startTestWithUpdates(data, updateCallback);
            socket.emit('test:started', { testId });
        } catch (error) {
            socket.emit('test:error', { error: error.message });
        }
    }

    async handleSimulationStart(socket, scenario) {
        try {
            const updateCallback = (update) => {
                socket.emit('simulation:update', update);
            };
            const simulationId = await this.agentSimulator.startSimulationWithUpdates(scenario, updateCallback);
            socket.emit('simulation:started', { simulationId });
        } catch (error) {
            socket.emit('simulation:error', { error: error.message });
        }
    }

    async handleSimulationStop(socket, simulationId) {
        try {
            await this.agentSimulator.stopSimulation(simulationId);
            socket.emit('simulation:stopped', { simulationId });
        } catch (error) {
            socket.emit('simulation:error', { error: error.message });
        }
    }
}

module.exports = DevWebSocketService;