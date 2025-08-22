#!/usr/bin/env node

/**
 * Remove Debug Logging from Production Code
 * Removes console.log statements that could expose sensitive data
 */

const fs = require('fs');
const path = require('path');

const excludePatterns = [
    /node_modules/,
    /\.git/,
    /coverage/,
    /logs/,
    /scripts/ // Exclude this script itself
];

const debugPatterns = [
    /console\.log\s*\([^)]*['"](SECRET|KEY|PASSWORD|TOKEN|PRIVATE)[^)]*\)/gi,
    /console\.log\s*\([^)]*process\.env[^)]*\)/gi,
    /console\.debug\s*\(/gi,
    /console\.trace\s*\(/gi
];

function shouldExclude(filePath) {
    return excludePatterns.some(pattern => pattern.test(filePath));
}

function removeDebugLogging(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    let modified = false;
    let newContent = content;
    
    debugPatterns.forEach(pattern => {
        if (pattern.test(newContent)) {
            newContent = newContent.replace(pattern, '// Debug logging removed for production');
            modified = true;
        }
    });
    
    if (modified) {
        fs.writeFileSync(filePath, newContent);
        console.log(`✅ Cleaned debug logging in: ${filePath}`);
        return true;
    }
    
    return false;
}

function scanDirectory(dirPath) {
    const files = fs.readdirSync(dirPath);
    let totalCleaned = 0;
    
    files.forEach(file => {
        const fullPath = path.join(dirPath, file);
        
        if (shouldExclude(fullPath)) {
            return;
        }
        
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
            totalCleaned += scanDirectory(fullPath);
        } else if (file.match(/\.(js|ts|json)$/)) {
            if (removeDebugLogging(fullPath)) {
                totalCleaned++;
            }
        }
    });
    
    return totalCleaned;
}

console.log('🧹 Removing debug logging from production code...\n');

const projectRoot = path.join(__dirname, '..');
const totalCleaned = scanDirectory(projectRoot);

console.log(`\n🎉 Cleaned debug logging from ${totalCleaned} files`);

if (totalCleaned === 0) {
    console.log('✅ No sensitive debug logging found - code is clean!');
} else {
    console.log('⚠️  Sensitive debug statements removed. Review changes before deployment.');
}