#!/usr/bin/env node
/**
 * Script to replace console.log statements with proper CDS logging
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Define replacements for different types of console statements
const replacements = [
    {
        pattern: /console\.log\(/g,
        replacement: 'cds.log(\'service\').info('
    },
    {
        pattern: /console\.info\(/g,
        replacement: 'cds.log(\'service\').info('
    },
    {
        pattern: /console\.warn\(/g,
        replacement: 'cds.log(\'service\').warn('
    },
    {
        pattern: /console\.error\(/g,
        replacement: 'cds.log(\'service\').error('
    },
    {
        pattern: /console\.debug\(/g,
        replacement: 'cds.log(\'service\').debug('
    }
];

// Get all JavaScript files in srv directory
function getJsFiles(dir) {
    const files = [];
    
    function walk(directory) {
        const items = fs.readdirSync(directory);
        
        for (const item of items) {
            const fullPath = path.join(directory, item);
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory()) {
                walk(fullPath);
            } else if (item.endsWith('.js')) {
                files.push(fullPath);
            }
        }
    }
    
    walk(dir);
    return files;
}

// Add CDS import if not present
function ensureCdsImport(content) {
    if (content.includes('const cds = require(\'@sap/cds\')') || 
        content.includes('const cds=require(\'@sap/cds\')')) {
        return content;
    }
    
    // Find the first require statement and add CDS import after it
    const lines = content.split('\n');
    let insertIndex = 0;
    
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes('require(') && !lines[i].trim().startsWith('//')) {
            insertIndex = i + 1;
            break;
        }
    }
    
    lines.splice(insertIndex, 0, 'const cds = require(\'@sap/cds\');');
    return lines.join('\n');
}

// Process a single file
function processFile(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        let modified = false;
        
        // Check if file has console statements
        if (!/console\.(log|info|warn|error|debug)\s*\(/.test(content)) {
            return false;
        }
        
        console.log(`Processing: ${filePath}`);
        
        // Apply replacements
        for (const { pattern, replacement } of replacements) {
            if (pattern.test(content)) {
                content = content.replace(pattern, replacement);
                modified = true;
            }
        }
        
        if (modified) {
            // Ensure CDS import is present
            content = ensureCdsImport(content);
            
            // Write back to file
            fs.writeFileSync(filePath, content, 'utf8');
            console.log(`  ✓ Fixed console statements`);
            return true;
        }
        
        return false;
    } catch (error) {
        console.error(`Error processing ${filePath}:`, error.message);
        return false;
    }
}

// Main execution
function main() {
    const srvDir = path.join(__dirname, '..', 'srv');
    
    if (!fs.existsSync(srvDir)) {
        console.error('srv directory not found');
        process.exit(1);
    }
    
    console.log('🔧 Fixing console.log statements with proper CDS logging...\n');
    
    const jsFiles = getJsFiles(srvDir);
    let processedCount = 0;
    
    for (const filePath of jsFiles) {
        if (processFile(filePath)) {
            processedCount++;
        }
    }
    
    console.log(`\n✅ Processed ${processedCount} files with console statements`);
    console.log('🎯 All console statements have been replaced with CDS logging');
}

if (require.main === module) {
    main();
}

module.exports = { processFile, getJsFiles };