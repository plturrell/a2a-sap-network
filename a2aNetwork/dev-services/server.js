const cds = require('@sap/cds');
const DevWebSocketService = require('./websocket-service');

// Initialize WebSocket service
const wsService = new DevWebSocketService();

// CAP server bootstrap
cds.on('listening', async function({ server }) {
    // Initialize WebSocket service with the HTTP server
    await wsService.initialize(server);
    
    console.log(`🛠️  A2A Development Services (CAP) running on ${server.address().port}`);
    console.log('Services available:');
    console.log('  - DevServices OData: /dev-services');
    console.log('  - WebSocket: ws://localhost:' + server.address().port);
});

// Start the CAP server
module.exports = cds.server;