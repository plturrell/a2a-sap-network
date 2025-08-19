// Test script to verify launchpad configuration
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 8080;
const STATIC_PATH = __dirname;

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.properties': 'text/plain',
    '.xml': 'application/xml'
};

const server = http.createServer((req, res) => {
    const filePath = path.join(STATIC_PATH, req.url === '/' ? 'launchpad.html' : req.url);
    
    // Handle API mock endpoints
    if (req.url.startsWith('/api/v1/')) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        const mockData = {
            value: Math.floor(Math.random() * 100),
            unit: "items",
            state: "Positive"
        };
        res.end(JSON.stringify(mockData));
        return;
    }
    
    // Serve static files
    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.writeHead(404);
                res.end('File not found');
            } else {
                res.writeHead(500);
                res.end('Server error');
            }
        } else {
            const ext = path.extname(filePath);
            const mimeType = mimeTypes[ext] || 'application/octet-stream';
            res.writeHead(200, { 'Content-Type': mimeType });
            res.end(content);
        }
    });
});

server.listen(PORT, () => {
    // console.log(`
╔════════════════════════════════════════════════════════════════════╗
║                   SAP Fiori Launchpad Test Server                  ║
╠════════════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:${PORT}/launchpad.html            ║
║                                                                    ║
║  This is a test server to verify the launchpad configuration.     ║
║  Press Ctrl+C to stop the server.                                 ║
╚════════════════════════════════════════════════════════════════════╝
    `);
});