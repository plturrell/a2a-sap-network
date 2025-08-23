const fs = require('fs');

// Read the server.js file
const content = fs.readFileSync('/Users/apple/projects/a2a/a2aNetwork/srv/server.js', 'utf8');
const lines = content.split('\n');

// Find the start and end of agent proxy routes
let startLine = -1;
let endLine = -1;

for (let i = 0; i < lines.length; i++) {
    // Look for the start of agent proxy routes
    if (lines[i].includes('// Agent proxy routes have been moved') || 
        lines[i].includes('// AGENT 2 API PROXY ROUTES')) {
        startLine = i;
    }
    
    // Look for the end marker (the OData route for Agent 13 deployments)
    if (startLine !== -1 && lines[i].includes('});') && 
        i > startLine + 100 && // Make sure we're far enough from start
        lines[i-1].includes('}') &&
        lines[i-2].includes('});') &&
        (lines[i+1].includes('log.info(`SAP CAP server') || 
         lines[i+2].includes('log.info(`SAP CAP server'))) {
        endLine = i;
        break;
    }
}

// If we didn't find the exact end, look for the log.info line
if (endLine === -1) {
    for (let i = startLine + 1; i < lines.length; i++) {
        if (lines[i].includes('log.info(`SAP CAP server listening on port')) {
            endLine = i - 2; // Go back a couple lines to catch the closing braces
            break;
        }
    }
}

console.log(`Found agent proxy routes from line ${startLine + 1} to ${endLine + 1}`);

if (startLine === -1 || endLine === -1) {
    console.error('Could not find agent proxy routes section');
    process.exit(1);
}

// Create new content without the Express routes
const newLines = [
    ...lines.slice(0, startLine),
    '    // Agent proxy routes have been moved to AgentProxyService (agentProxyService.cds)',
    '    // All Express routes are now handled through CAP services',
    '    ',
    ...lines.slice(endLine + 1)
];

// Write the new content
fs.writeFileSync('/Users/apple/projects/a2a/a2aNetwork/srv/server.js', newLines.join('\n'));

console.log('Express routes removed successfully');
console.log(`Removed ${endLine - startLine + 1} lines`);