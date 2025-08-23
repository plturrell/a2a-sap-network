"use strict";

/**
 * Main entry point for A2A Developer Portal CAP application
 * Initializes distributed tracing before starting the server
 */

// Initialize distributed tracing first
require('./telemetry/tracer').initialize()
    .then(() => {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('✅ Distributed tracing initialized');
        
        // Start the CAP server
        require('./server');
    })
    .catch(error => {
        console.error('❌ Failed to initialize distributed tracing:', error);
        // Continue without tracing in development
        if (process.env.NODE_ENV !== 'production') {
            require('./server');
        } else {
            process.exit(1);
        }
    });