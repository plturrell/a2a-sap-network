#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function findJSFiles(dir, files = []) {
    try {
        const items = fs.readdirSync(dir);
        
        for (const item of items) {
            const fullPath = path.join(dir, item);
            
            try {
                const stat = fs.statSync(fullPath);
                
                if (stat.isDirectory() && item !== 'node_modules' && item !== '.git') {
                    findJSFiles(fullPath, files);
                } else if (stat.isFile() && item.endsWith('.js') && !item.includes('test') && !item.includes('Test')) {
                    files.push(fullPath);
                }
            } catch (statError) {
                // Skip files that can't be accessed
                console.log(`⚠️  Skipping ${fullPath}: ${statError.message}`);
            }
        }
    } catch (dirError) {
        // Skip directories that can't be read
        console.log(`⚠️  Skipping directory ${dir}: ${dirError.message}`);
    }
    
    return files;
}

function checkJSSyntax(filePath) {
    try {
        // Try to parse the file using Node.js
        const content = fs.readFileSync(filePath, 'utf8');
        
        // Basic syntax check - try to compile it
        const tempFile = path.join(__dirname, 'temp_syntax_check.js');
        fs.writeFileSync(tempFile, content);
        
        try {
            execSync(`node --check "${tempFile}"`, { stdio: 'pipe' });
            fs.unlinkSync(tempFile);
            return { success: true };
        } catch (error) {
            fs.unlinkSync(tempFile);
            return { success: false, error: error.message };
        }
    } catch (error) {
        return { success: false, error: error.message };
    }
}

function main() {
    console.log('🔍 Checking JavaScript syntax errors...\n');
    
    const jsFiles = findJSFiles('.');
    let totalFiles = 0;
    let syntaxErrors = 0;
    let errorsFound = [];
    
    for (const file of jsFiles) {
        totalFiles++;
        const result = checkJSSyntax(file);
        
        if (!result.success) {
            syntaxErrors++;
            errorsFound.push({ file, error: result.error });
            console.log(`❌ ${file}: ${result.error.split('\n')[0]}`);
        }
    }
    
    console.log('\n📊 JAVASCRIPT SYNTAX CHECK RESULTS:');
    console.log(`  Total JS files checked: ${totalFiles}`);
    console.log(`  Syntactically correct: ${totalFiles - syntaxErrors}`);
    console.log(`  Syntax errors: ${syntaxErrors}`);
    console.log(`  Success rate: ${((totalFiles - syntaxErrors) / totalFiles * 100).toFixed(1)}%`);
    
    if (syntaxErrors === 0) {
        console.log('🎉 ALL JAVASCRIPT FILES HAVE CORRECT SYNTAX!');
    } else {
        console.log(`\n⚠️  ${syntaxErrors} files have syntax errors:`);
        errorsFound.slice(0, 10).forEach(({ file, error }) => {
            console.log(`  • ${file}`);
        });
    }
    
    process.exit(syntaxErrors > 0 ? 1 : 0);
}

main();