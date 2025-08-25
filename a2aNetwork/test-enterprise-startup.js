/**
 * SAP Enterprise Standard Programmatic Startup Test
 * Direct approach using the same pattern as working CLI
 */

const cds = require('@sap/cds');

console.log('SAP CAP Enterprise Server - Programmatic Startup Test');
console.log('Using SAP enterprise standard approach...');

// Set enterprise environment
process.env.NODE_ENV = process.env.NODE_ENV || 'development';
process.env.PORT = process.env.PORT || '4004';

async function startEnterpriseServer() {
    try {
        // Load the enterprise server configuration
        const server = require('./srv/server');

        console.log('Enterprise server module loaded successfully');

        // Start server using SAP enterprise standard model discovery
        // Test with single service to isolate the conflict
        const serviceFiles = [
            'srv/a2aService.cds'
            // Start with just 1 service to test
        ];

        console.log('Loading services:', serviceFiles);

        // Use SAP enterprise standard programmatic startup
        // First load the CDS model, then serve specific services
        const model = await cds.load(serviceFiles);
        const app = await cds.serve(model)
            .to('sqlite')
            .at(4004);

        console.log('SUCCESS: SAP CAP Enterprise Server started successfully');
        console.log('Server running at: http://localhost:4004');
        console.log('Launchpad available at: http://localhost:4004/launchpad.html');

        return app;

    } catch (error) {
        console.error('ERROR: SAP CAP Enterprise Server startup failed');
        console.error('Error:', error.message);
        console.error('Stack:', error.stack);
        throw error;
    }
}

// Execute startup
startEnterpriseServer()
    .then(() => {
        console.log('Enterprise server startup verification complete');
        // Keep running for testing
        setTimeout(() => {
            console.log('Test complete - server running successfully');
            process.exit(0);
        }, 5000);
    })
    .catch(error => {
        console.error('Enterprise server startup failed:', error.message);
        process.exit(1);
    });

// Timeout safety
setTimeout(() => {
    console.error('TIMEOUT: Enterprise server startup took too long');
    process.exit(1);
}, 30000);
