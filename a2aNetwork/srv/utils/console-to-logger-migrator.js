/**
 * Console.log to Structured Logger Migration Utility
 * Replaces console.log statements with proper structured logging
 */

const fs = require('fs');
const path = require('path');

class ConsoleToLoggerMigrator {
    constructor() {
        this.replacements = 0;
        this.errors = [];
        
        // Mapping of console methods to logger levels
        this.methodMapping = {
            'console.log': 'logger.info',
            'console.info': 'logger.info',
            'console.warn': 'logger.warn',
            'console.error': 'logger.error',
            'console.debug': 'logger.debug'
        };
    }

    /**
     * Analyze file for console.log usage
     */
    analyzeFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');
        const consoleUsages = [];
        
        lines.forEach((line, index) => {
            // Skip commented lines
            if (line.trim().startsWith('//') || line.trim().startsWith('*')) {
                return;
            }
            
            // Check for console usage
            const consoleMatch = line.match(/console\.(log|info|warn|error|debug)\s*\(/);
            if (consoleMatch) {
                consoleUsages.push({
                    lineNumber: index + 1,
                    line: line.trim(),
                    method: `console.${consoleMatch[1]}`
                });
            }
        });
        
        return consoleUsages;
    }

    /**
     * Generate logger import statement
     */
    generateLoggerImport(existingImports) {
        // Check if logger is already imported
        if (existingImports.includes('LoggerFactory') || existingImports.includes('logger')) {
            return null;
        }
        
        return 'const { LoggerFactory } = require(\'../../shared/logging/structured-logger\');';
    }

    /**
     * Generate logger initialization
     */
    generateLoggerInit(fileName) {
        const serviceName = path.basename(fileName, path.extname(fileName));
        return `const logger = LoggerFactory.createLogger('${serviceName}');`;
    }

    /**
     * Convert console statement to logger statement
     */
    convertConsoleToLogger(line, method) {
        const loggerMethod = this.methodMapping[method] || 'logger.info';
        
        // Extract the console.log content
        const match = line.match(/console\.\w+\s*\((.*)\)/);
        if (!match) return line;
        
        const content = match[1];
        
        // Handle different argument patterns
        if (content.includes(',')) {
            // Multiple arguments - convert to structured logging
            const args = this.parseArguments(content);
            return this.convertToStructuredLog(line, loggerMethod, args);
        } else {
            // Single argument - simple replacement
            return line.replace(/console\.\w+/, loggerMethod);
        }
    }

    /**
     * Parse console.log arguments
     */
    parseArguments(argsString) {
        // Simple parser - may need enhancement for complex cases
        const args = [];
        let current = '';
        let inString = false;
        let stringChar = '';
        let depth = 0;
        
        for (let i = 0; i < argsString.length; i++) {
            const char = argsString[i];
            
            if (!inString && (char === '"' || char === '\'' || char === '`')) {
                inString = true;
                stringChar = char;
            } else if (inString && char === stringChar && argsString[i-1] !== '\\') {
                inString = false;
            }
            
            if (!inString) {
                if (char === '(' || char === '{' || char === '[') depth++;
                if (char === ')' || char === '}' || char === ']') depth--;
                
                if (char === ',' && depth === 0) {
                    args.push(current.trim());
                    current = '';
                    continue;
                }
            }
            
            current += char;
        }
        
        if (current.trim()) {
            args.push(current.trim());
        }
        
        return args;
    }

    /**
     * Convert to structured logging format
     */
    convertToStructuredLog(line, loggerMethod, args) {
        if (args.length === 0) return line;
        
        // First argument is the message
        const message = args[0];
        
        // Additional arguments become structured data
        if (args.length === 1) {
            return line.replace(/console\.\w+\s*\(.*\)/, `${loggerMethod}(${message})`);
        }
        
        // Convert additional arguments to structured format
        const structuredData = this.createStructuredData(args.slice(1));
        
        return line.replace(
            /console\.\w+\s*\(.*\)/,
            `${loggerMethod}(${message}, ${structuredData})`
        );
    }

    /**
     * Create structured data object from arguments
     */
    createStructuredData(args) {
        // If single object argument, use it directly
        if (args.length === 1 && args[0].trim().startsWith('{')) {
            return args[0];
        }
        
        // Convert multiple arguments to object
        const pairs = args.map((arg, index) => {
            // Try to detect variable names
            if (arg.match(/^[a-zA-Z_$][a-zA-Z0-9_$]*$/)) {
                return `${arg}: ${arg}`;
            } else {
                return `value${index + 1}: ${arg}`;
            }
        });
        
        return `{ ${pairs.join(', ')} }`;
    }

    /**
     * Process single file
     */
    processFile(filePath) {
        try {
            // Read file
            const content = fs.readFileSync(filePath, 'utf8');
            const lines = content.split('\n');
            
            // Analyze console usage
            const consoleUsages = this.analyzeFile(filePath);
            if (consoleUsages.length === 0) {
                return { status: 'skipped', reason: 'No console usage found' };
            }
            
            // Check if logger is already imported
            const hasLoggerImport = content.includes('LoggerFactory') || 
                                  content.includes('require(\'../logging') ||
                                  content.includes('logger =');
            
            let modifiedContent = content;
            
            // Add logger import if needed
            if (!hasLoggerImport) {
                const importStatement = this.generateLoggerImport(content);
                const loggerInit = this.generateLoggerInit(filePath);
                
                // Find the right place to insert imports
                const firstRequireIndex = lines.findIndex(line => 
                    line.includes('require(') && !line.trim().startsWith('//')
                );
                
                if (firstRequireIndex !== -1) {
                    // Insert after other requires
                    lines.splice(firstRequireIndex + 1, 0, '', importStatement, loggerInit);
                } else {
                    // Insert at the beginning
                    lines.unshift(importStatement, loggerInit, '');
                }
                
                modifiedContent = lines.join('\n');
            }
            
            // Replace console statements
            let replacementCount = 0;
            modifiedContent = modifiedContent.split('\n').map(line => {
                // Skip comments
                if (line.trim().startsWith('//') || line.trim().startsWith('*')) {
                    return line;
                }
                
                // Check for console usage
                const consoleMatch = line.match(/console\.(log|info|warn|error|debug)\s*\(/);
                if (consoleMatch) {
                    replacementCount++;
                    const method = `console.${consoleMatch[1]}`;
                    return this.convertConsoleToLogger(line, method);
                }
                
                return line;
            }).join('\n');
            
            // Write modified content
            fs.writeFileSync(filePath, modifiedContent);
            
            this.replacements += replacementCount;
            
            return {
                status: 'success',
                replacements: replacementCount,
                loggerAdded: !hasLoggerImport
            };
            
        } catch (error) {
            this.errors.push({ file: filePath, error: error.message });
            return { status: 'error', error: error.message };
        }
    }

    /**
     * Process multiple files
     */
    processFiles(filePaths) {
        const results = [];
        
        filePaths.forEach(filePath => {
            console.log(`Processing ${filePath}...`);
            const result = this.processFile(filePath);
            results.push({ file: filePath, ...result });
        });
        
        return {
            totalFiles: filePaths.length,
            totalReplacements: this.replacements,
            results,
            errors: this.errors
        };
    }

    /**
     * Generate migration report
     */
    generateReport(results) {
        console.log('\n=== Console.log to Logger Migration Report ===\n');
        console.log(`Total files processed: ${results.totalFiles}`);
        console.log(`Total replacements: ${results.totalReplacements}`);
        console.log('\nFile-by-file results:');
        
        results.results.forEach(result => {
            if (result.status === 'success') {
                console.log(`✅ ${result.file}: ${result.replacements} replacements${result.loggerAdded ? ' (logger added)' : ''}`);
            } else if (result.status === 'skipped') {
                console.log(`⏭️  ${result.file}: ${result.reason}`);
            } else {
                console.log(`❌ ${result.file}: ${result.error}`);
            }
        });
        
        if (results.errors.length > 0) {
            console.log('\n❌ Errors:');
            results.errors.forEach(error => {
                console.log(`  - ${error.file}: ${error.error}`);
            });
        }
        
        console.log('\n✅ Migration complete!');
    }
}

// Run migration if called directly
if (require.main === module) {
    const migrator = new ConsoleToLoggerMigrator();
    
    // Files to process (from the search results)
    const filesToProcess = [
        path.join(__dirname, '../websocketDataService.js'),
        path.join(__dirname, '../services/agent14-service.js'),
        path.join(__dirname, '../services/agent15-service.js'),
        path.join(__dirname, '../messageQueueService.js'),
        path.join(__dirname, '../authSessionManager.js'),
        path.join(__dirname, '../chatAgentBridge.js'),
        path.join(__dirname, '../pushNotificationService.js'),
        path.join(__dirname, '../utils/apiDocumentationGenerator.js')
    ];
    
    const results = migrator.processFiles(filesToProcess);
    migrator.generateReport(results);
}

module.exports = ConsoleToLoggerMigrator;