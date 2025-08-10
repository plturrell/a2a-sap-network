#!/usr/bin/env node

/**
 * Script to add missing database imports (SELECT, INSERT, UPDATE, DELETE, UPSERT)
 */

const fs = require('fs');
const path = require('path');

// List of files that need database imports
const filesToFix = [
    'srv/i18n/sapI18nMiddleware.js',
    'srv/lib/sapAbapIntegration.js',
    'srv/lib/sapAgentManager.js',
    'srv/lib/sapChangeTracker.js',
    'srv/lib/sapDbInit.js',
    'srv/lib/sapNetworkStats.js',
    'srv/lib/sapTransactionCoordinator.js',
    'srv/lib/sapWorkflowExecutor.js',
    'srv/sapDataInit.js'
];

const cdsImport = 'const { SELECT, INSERT, UPDATE, DELETE, UPSERT } = cds.ql;';

console.log('Adding missing database imports...');

for (const file of filesToFix) {
    const filePath = path.join(process.cwd(), file);
    
    if (!fs.existsSync(filePath)) {
        console.log(`⚠️  File not found: ${file}`);
        continue;
    }
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Check if imports already exist
    if (content.includes('cds.ql') || content.includes('SELECT') && content.includes('const')) {
        console.log(`✓ ${file} already has database imports`);
        continue;
    }
    
    // Find the line with "const cds = require('@sap/cds');"
    const lines = content.split('\n');
    let cdsLineIndex = -1;
    
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes("const cds = require('@sap/cds')")) {
            cdsLineIndex = i;
            break;
        }
    }
    
    if (cdsLineIndex === -1) {
        console.log(`⚠️  Could not find CDS require statement in ${file}`);
        continue;
    }
    
    // Insert the database imports after the CDS require
    lines.splice(cdsLineIndex + 1, 0, cdsImport);
    
    const newContent = lines.join('\n');
    fs.writeFileSync(filePath, newContent);
    
    console.log(`✅ Added database imports to ${file}`);
}

console.log('✨ Database imports fix completed!');