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
        this.io.on('blockchain-connection', this.handleSocketConnection.bind(this));
    }

    async handleTestStart(socket, data) {
        try {
            const testId = await this.testOrchestrator.startTestWithUpdates(data, this.createTestUpdateCallback(socket));
            socket.emit('test:started', { testId });
        } catch (error) {
            socket.emit('test:error', { error: error.message });
        }
    }

    async handleSimulationStart(socket, scenario) {
        try {
            const simulationId = await this.agentSimulator.startSimulationWithUpdates(scenario, this.createSimulationUpdateCallback(socket));
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

    handleSocketConnection(socket) {
        logger.info('Client connected to dev services');

        // Test execution with real-time updates
        socket.on('test:start', this.handleTestStartEvent.bind(this, socket));

        // Agent monitoring
        socket.on('monitor:subscribe', this.handleMonitorSubscribe.bind(this, socket));
        socket.on('monitor:unsubscribe', this.handleMonitorUnsubscribe.bind(this, socket));

        // Hot reload notifications
        socket.on('hotreload:enable', this.handleHotReloadEnable.bind(this, socket));
        socket.on('hotreload:disable', this.handleHotReloadDisable.bind(this, socket));

        // Simulation control
        socket.on('simulation:start', this.handleSimulationStartEvent.bind(this, socket));
        socket.on('simulation:stop', this.handleSimulationStopEvent.bind(this, socket));

        socket.on('disconnect', this.handleSocketDisconnect.bind(this, socket));
    }

    async handleTestStartEvent(socket, data) {
        await this.handleTestStart(socket, data);
    }

    handleMonitorSubscribe(socket, agentId) {
        this.devMonitor.subscribeToAgent(socket, agentId);
    }

    handleMonitorUnsubscribe(socket, agentId) {
        this.devMonitor.unsubscribeFromAgent(socket, agentId);
    }

    handleHotReloadEnable(socket, config) {
        this.hotReloader.enableForClient(socket, config);
    }

    handleHotReloadDisable(socket) {
        this.hotReloader.disableForClient(socket);
    }

    async handleSimulationStartEvent(socket, scenario) {
        await this.handleSimulationStart(socket, scenario);
    }

    async handleSimulationStopEvent(socket, simulationId) {
        await this.handleSimulationStop(socket, simulationId);
    }

    handleSocketDisconnect(socket) {
        logger.info('Client disconnected from dev services');
        this.devMonitor.cleanupClient(socket);
        this.hotReloader.cleanupClient(socket);
    }

    createTestUpdateCallback(socket) {
        return function(update) {
            socket.emit('test:update', update);
        };
    }

    createSimulationUpdateCallback(socket) {
        return function(update) {
            socket.emit('simulation:update', update);
        };
    }
}

module.exports = DevWebSocketService;