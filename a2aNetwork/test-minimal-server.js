/**
 * Minimal test server to isolate OpenTelemetry conflict
 * No middleware, no imports, just basic Express static file serving
 */

const express = require('express');
const path = require('path');

const app = express();
const port = 4005; // Different port to avoid conflicts

console.log('Starting minimal test server...');

// Only basic static file serving - no middleware at all
app.use('/common', express.static(path.join(__dirname, 'app/common')));

// Basic error handling
app.use(function(err, req, res, next) {
    console.error('Minimal server error:', err);
    res.status(500).send('Minimal server error: ' + err.message);
});

app.listen(port, function() {
    console.log(`Minimal test server running on http://localhost:${port}`);
    console.log('Test URL: http://localhost:4005/common/auth/SSOManager.js');
});
