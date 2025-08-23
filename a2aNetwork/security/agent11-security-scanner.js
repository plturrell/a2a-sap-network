const fs = require('fs');
const path = require('path');

// Agent 11 (SQL Agent) Security Scanner
// Comprehensive security analysis for SQL and database operation UI components

const AGENT11_PATH = path.join(__dirname, '../app/a2aFiori/webapp/ext/agent11');

const securityPatterns = {
    // XSS Vulnerabilities
    xss: {
        directHtmlInjection: /\.html\s*\(\s*[^)]+\)/g,
        innerHTMLUsage: /\.innerHTML\s*=/g,
        jqueryHtml: /\$\([^)]+\)\.html\s*\(/g,
        unsafeMessageToast: /MessageToast\.show\s*\(\s*[^)]*\[[^\]]*\]\s*\)/g,
        directDOMManipulation: /document\.(write|writeln)\s*\(/g,
        evalUsage: /\beval\s*\(/g,
        functionConstructor: /new\s+Function\s*\(/g,
        setTimeoutString: /setTimeout\s*\(\s*["']/g,
        setIntervalString: /setInterval\s*\(\s*["']/g,
        unsafeSQLDisplay: /sql.*message|error.*sql/gi,
        dynamicQueryDisplay: /\.(query|sql|statement).*\+/g
    },
    
    // SQL Injection Vulnerabilities
    sqlInjection: {
        directSQLInjection: /sql.*\+.*user|query.*\+.*input/gi,
        unsafeSQLConstruction: /["']\s*\+\s*.*\+\s*["']/g,
        dynamicQueryBuilding: /\$\{.*\}.*sql|sql.*\$\{.*\}/gi,
        interpolatedSQL: /`.*\$\{.*\}.*`/g,
        unsafeParameterUsage: /WHERE.*=.*\+|SET.*=.*\+/gi,
        concatenatedSQL: /concat.*sql|sql.*concat/gi
    },
    
    // CSRF Vulnerabilities
    csrf: {
        missingCSRFToken: /callFunction.*\n(?!.*headers.*x-csrf-token)/g,
        unprotectedStateChange: /(Execute|Run|Create|Update|Delete|Drop|Alter).*callFunction/g,
        missingSecurityHeaders: /\$\.ajax\s*\(\s*\{[^}]*(?!.*beforeSend)/g
    },
    
    // Authentication/Authorization Issues
    auth: {
        missingAuthCheck: /on(Execute|Run|Create|Delete|Update|Drop|Alter)[^{]*\{(?![^}]*checkAuth)/g,
        hardcodedCredentials: /(password|token|key|connection.*string)\s*[:=]\s*["'][^"']+["']/gi,
        exposedAPIKeys: /(api[_-]?key|secret|connection)\s*[:=]\s*["'][^"']+["']/gi,
        missingDatabaseAuthCheck: /callFunction.*sql.*(?!.*auth)/gi
    },
    
    // Input Validation Issues
    validation: {
        missingSQLValidation: /sql.*statement.*(?!.*validate)/gi,
        missingParameterValidation: /parameter.*value.*(?!.*validate)/gi,
        unsanitizedQueryInput: /(query|sql).*input.*(?!.*sanitize)/gi,
        noSQLSyntaxCheck: /execute.*sql.*(?!.*syntax.*check)/gi,
        missingSchemaValidation: /schema.*(?!.*validate)/gi
    },
    
    // Insecure Communications
    insecure: {
        httpEndpoints: /http:\/\/(?!localhost|127\.0\.0\.1)/g,
        unencryptedWebSocket: /ws:\/\/(?!localhost|127\.0\.0\.1)/g,
        hardcodedURLs: /(http|ws)s?:\/\/[^/]+/g,
        insecureEventSource: /new\s+EventSource\s*\(\s*["']http:/g,
        plainTextConnections: /connection.*http:|database.*url.*http:/gi
    },
    
    // Error Handling Issues
    errorHandling: {
        exposedErrors: /catch.*\{[^}]*message[^}]*show/g,
        detailedErrorMessages: /error\.(stack|message|details).*MessageToast/g,
        exposedSQLErrors: /sql.*error.*show|database.*error.*display/gi,
        unhandledPromises: /\.then\s*\([^)]*\)\s*(?!\.catch)/g,
        missingSQLErrorHandling: /execute.*sql.*(?!.*catch)/gi
    },
    
    // Data Exposure
    dataExposure: {
        sensitiveDataInLogs: /console\.(log|info|debug).*\(.*(password|token|key|secret|connection|sql)/gi,
        exposedConnectionData: /localStorage\.setItem.*\(.*(connection|database|query)/gi,
        unencryptedStorage: /sessionStorage\.setItem.*(?!.*encrypt)/g,
        exposedQueryData: /return.*\{.*sql.*:|query.*:/gi,
        connectionStringExposure: /connection.*string.*[^encrypt]/gi
    },
    
    // Resource Management
    resources: {
        memoryLeaks: /addEventListener.*(?!.*removeEventListener)/g,
        unclosedConnections: /new\s*(WebSocket|EventSource).*(?!.*close)|connection.*open.*(?!.*close)/g,
        infiniteLoops: /while\s*\(.*true.*\)|for\s*\(.*;\s*;\s*\)/g,
        unboundedQueries: /execute.*sql.*(?!.*limit|timeout)/gi,
        connectionPoolLeaks: /pool.*get.*(?!.*release)/gi
    },
    
    // SAP Fiori Specific
    sapFiori: {
        missingResourceBundle: /getText\s*\([^)]*\)(?!.*getResourceBundle)/g,
        hardcodedTexts: /MessageToast\.show\s*\(\s*["'][^{]/g,
        improperExtensionPoint: /ControllerExtension\.extend.*(?!.*override)/g,
        missingSecurityUtil: /(?!SecurityUtil\.).*escape.*HTML/gi
    }
};

const specificAgent11Checks = {
    // SQL-specific vulnerabilities
    sqlSecurity: {
        unsafeSQLExecution: /executeQuery.*(?!.*validateSQL)/g,
        missingSQLSanitization: /sql.*statement.*(?!.*sanitize)/gi,
        directSQLConstruction: /["']\s*\+\s*.*sql|sql.*\+.*["']/gi,
        unsafeParameterBinding: /parameter.*(?!.*prepared.*statement)/gi,
        noSQLInjectionCheck: /execute.*(?!.*injection.*check)/gi,
        dynamicSQLConstruction: /\$\{.*\}.*sql|sql.*\$\{/gi
    },
    
    databaseSecurity: {
        missingConnectionValidation: /connection.*(?!.*validate)/gi,
        unencryptedConnections: /connection.*string.*(?!.*ssl|encrypt)/gi,
        hardcodedConnectionStrings: /connection.*=.*["'].*:\/\//gi,
        missingConnectionPoolSecurity: /pool.*(?!.*auth|security)/gi,
        exposedDatabaseCredentials: /(user|password|host).*=.*["'][^"']+["']/gi
    },
    
    queryValidation: {
        missingQueryValidation: /sql.*(?!.*validate)/gi,
        noQueryComplexityCheck: /execute.*(?!.*complexity)/gi,
        missingQueryTimeout: /execute.*(?!.*timeout)/gi,
        unsafeQueryExecution: /callFunction.*Execute.*(?!.*security)/gi
    }
};

function scanFile(filePath, fileName) {
    const vulnerabilities = [];
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n');
    
    // Check all security patterns
    Object.entries(securityPatterns).forEach(([category, patterns]) => {
        Object.entries(patterns).forEach(([name, pattern]) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = content.substring(0, match.index).split('\n').length;
                vulnerabilities.push({
                    type: category,
                    subtype: name,
                    severity: getSeverity(category, name),
                    file: fileName,
                    line: lineNumber,
                    code: lines[lineNumber - 1]?.trim() || '',
                    message: getSecurityMessage(category, name),
                    fix: getFixSuggestion(category, name)
                });
            }
        });
    });
    
    // Agent 11 specific checks
    if (fileName.includes('controller') || fileName.includes('utils')) {
        Object.entries(specificAgent11Checks).forEach(([category, patterns]) => {
            Object.entries(patterns).forEach(([name, pattern]) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = content.substring(0, match.index).split('\n').length;
                    vulnerabilities.push({
                        type: category,
                        subtype: name,
                        severity: 'HIGH',
                        file: fileName,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: getAgent11Message(category, name),
                        fix: getAgent11Fix(category, name)
                    });
                }
            });
        });
    }
    
    return vulnerabilities;
}

function getSeverity(category, name) {
    const criticalPatterns = ['evalUsage', 'functionConstructor', 'directHtmlInjection', 'sqlInjection', 'missingCSRFToken', 'directSQLInjection'];
    const highPatterns = ['xss', 'csrf', 'auth', 'validation', 'insecure', 'sqlInjection'];
    
    if (criticalPatterns.includes(name) || category === 'sqlInjection') return 'CRITICAL';
    if (highPatterns.some(p => category.includes(p))) return 'HIGH';
    return 'MEDIUM';
}

function getSecurityMessage(category, name) {
    const messages = {
        xss: {
            unsafeMessageToast: 'MessageToast with dynamic content can lead to XSS',
            evalUsage: 'eval() usage is a critical security risk',
            functionConstructor: 'Function constructor can execute arbitrary code',
            unsafeSQLDisplay: 'SQL content displayed without sanitization',
            dynamicQueryDisplay: 'Dynamic query content may contain user input'
        },
        sqlInjection: {
            directSQLInjection: 'Direct SQL injection vulnerability detected',
            unsafeSQLConstruction: 'SQL query constructed with string concatenation',
            dynamicQueryBuilding: 'Dynamic SQL query building without validation',
            interpolatedSQL: 'Template literal SQL with user input',
            unsafeParameterUsage: 'SQL parameters concatenated directly',
            concatenatedSQL: 'SQL concatenation detected'
        },
        csrf: {
            missingCSRFToken: 'Missing CSRF token in SQL operation',
            unprotectedStateChange: 'SQL state-changing operation without CSRF protection'
        },
        validation: {
            missingSQLValidation: 'SQL statement not validated before execution',
            missingParameterValidation: 'SQL parameters not validated',
            unsanitizedQueryInput: 'SQL query input not sanitized',
            noSQLSyntaxCheck: 'No SQL syntax validation',
            missingSchemaValidation: 'Database schema not validated'
        },
        insecure: {
            httpEndpoints: 'Insecure HTTP endpoint for database operations',
            unencryptedWebSocket: 'WebSocket connection without encryption',
            plainTextConnections: 'Database connection without encryption'
        }
    };
    
    return messages[category]?.[name] || `Security issue: ${category} - ${name}`;
}

function getFixSuggestion(category, name) {
    const fixes = {
        xss: {
            unsafeMessageToast: 'Use SecurityUtils.escapeHTML() for dynamic content',
            evalUsage: 'Use safe alternatives like JSON.parse',
            unsafeSQLDisplay: 'Sanitize SQL content before display'
        },
        sqlInjection: {
            directSQLInjection: 'Use SQLUtils.validateSQL() and parameterized queries',
            unsafeSQLConstruction: 'Use prepared statements with parameter binding',
            dynamicQueryBuilding: 'Validate and sanitize all SQL components',
            interpolatedSQL: 'Use parameterized queries instead of template literals',
            unsafeParameterUsage: 'Use SQLUtils.sanitizeSQL() for all parameters',
            concatenatedSQL: 'Replace concatenation with parameterized queries'
        },
        csrf: {
            missingCSRFToken: 'Add CSRF token to all SQL operation requests',
            unprotectedStateChange: 'Implement CSRF protection for SQL operations'
        },
        validation: {
            missingSQLValidation: 'Add SQLUtils.validateSQL() before execution',
            missingParameterValidation: 'Validate all SQL parameters',
            unsanitizedQueryInput: 'Use SQLUtils.sanitizeSQL() for all inputs',
            noSQLSyntaxCheck: 'Implement SQL syntax validation',
            missingSchemaValidation: 'Validate schema names and permissions'
        },
        insecure: {
            unencryptedWebSocket: 'Use wss:// protocol for SQL operations',
            plainTextConnections: 'Use encrypted database connections (SSL/TLS)'
        }
    };
    
    return fixes[category]?.[name] || 'Review and fix the security issue';
}

function getAgent11Message(category, name) {
    const messages = {
        sqlSecurity: {
            unsafeSQLExecution: 'SQL execution without validation',
            missingSQLSanitization: 'SQL statement not sanitized',
            directSQLConstruction: 'Direct SQL string construction',
            unsafeParameterBinding: 'SQL parameters not properly bound',
            noSQLInjectionCheck: 'Missing SQL injection detection',
            dynamicSQLConstruction: 'Dynamic SQL construction without validation'
        },
        databaseSecurity: {
            missingConnectionValidation: 'Database connection not validated',
            unencryptedConnections: 'Database connection not encrypted',
            hardcodedConnectionStrings: 'Hardcoded database connection strings',
            missingConnectionPoolSecurity: 'Connection pool missing security controls',
            exposedDatabaseCredentials: 'Database credentials exposed in code'
        },
        queryValidation: {
            missingQueryValidation: 'Query validation missing',
            noQueryComplexityCheck: 'No query complexity validation',
            missingQueryTimeout: 'Query execution without timeout',
            unsafeQueryExecution: 'Query execution without security checks'
        }
    };
    
    return messages[category]?.[name] || `Agent 11 specific issue: ${category}`;
}

function getAgent11Fix(category, name) {
    const fixes = {
        sqlSecurity: {
            unsafeSQLExecution: 'Use SQLUtils.validateSQL() before execution',
            missingSQLSanitization: 'Sanitize all SQL with SQLUtils.sanitizeSQL()',
            directSQLConstruction: 'Use parameterized queries with proper binding',
            unsafeParameterBinding: 'Implement prepared statement parameter binding',
            noSQLInjectionCheck: 'Add SQL injection detection and prevention',
            dynamicSQLConstruction: 'Validate all dynamic SQL components'
        },
        databaseSecurity: {
            missingConnectionValidation: 'Validate all database connections',
            unencryptedConnections: 'Use SSL/TLS for database connections',
            hardcodedConnectionStrings: 'Use configuration management for connections',
            missingConnectionPoolSecurity: 'Add authentication to connection pools',
            exposedDatabaseCredentials: 'Store credentials securely, use environment variables'
        },
        queryValidation: {
            missingQueryValidation: 'Implement comprehensive query validation',
            noQueryComplexityCheck: 'Add query complexity analysis',
            missingQueryTimeout: 'Set appropriate query timeouts',
            unsafeQueryExecution: 'Add security checks before query execution'
        }
    };
    
    return fixes[category]?.[name] || 'Implement security best practices';
}

function scanDirectory(dir) {
    const results = [];
    const files = fs.readdirSync(dir, { withFileTypes: true });
    
    files.forEach(file => {
        const fullPath = path.join(dir, file.name);
        if (file.isDirectory() && !file.name.startsWith('.')) {
            results.push(...scanDirectory(fullPath));
        } else if (file.isFile() && 
                  (file.name.endsWith('.js') || 
                   file.name.endsWith('.xml') || 
                   file.name.endsWith('.json'))) {
            const vulnerabilities = scanFile(fullPath, path.relative(AGENT11_PATH, fullPath));
            results.push(...vulnerabilities);
        }
    });
    
    return results;
}

function generateReport(vulnerabilities) {
    const report = {
        summary: {
            total: vulnerabilities.length,
            critical: vulnerabilities.filter(v => v.severity === 'CRITICAL').length,
            high: vulnerabilities.filter(v => v.severity === 'HIGH').length,
            medium: vulnerabilities.filter(v => v.severity === 'MEDIUM').length,
            low: vulnerabilities.filter(v => v.severity === 'LOW').length
        },
        byType: {},
        byFile: {},
        details: vulnerabilities
    };
    
    // Group by type
    vulnerabilities.forEach(vuln => {
        if (!report.byType[vuln.type]) {
            report.byType[vuln.type] = [];
        }
        report.byType[vuln.type].push(vuln);
        
        if (!report.byFile[vuln.file]) {
            report.byFile[vuln.file] = [];
        }
        report.byFile[vuln.file].push(vuln);
    });
    
    return report;
}

// SAP Fiori compliance checks
function checkSAPCompliance(dir) {
    const issues = [];
    const requiredFiles = ['manifest.json', 'i18n/i18n.properties'];
    
    requiredFiles.forEach(file => {
        const filePath = path.join(dir, file);
        if (!fs.existsSync(filePath)) {
            issues.push({
                type: 'compliance',
                message: `Missing required file: ${file}`,
                severity: 'HIGH'
            });
        }
    });
    
    // Check manifest.json structure
    const manifestPath = path.join(dir, 'manifest.json');
    if (fs.existsSync(manifestPath)) {
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        
        // Check for security-related configurations
        if (!manifest['sap.app']?.crossNavigation) {
            issues.push({
                type: 'compliance',
                message: 'Missing crossNavigation in manifest.json',
                severity: 'MEDIUM'
            });
        }
        
        if (!manifest['sap.ui5']?.dependencies?.minUI5Version) {
            issues.push({
                type: 'compliance',
                message: 'Missing minUI5Version in manifest.json',
                severity: 'LOW'
            });
        }
    }
    
    return issues;
}

// Main execution
console.log('Starting Agent 11 (SQL Agent) Security Scan...\n');

try {
    const vulnerabilities = scanDirectory(AGENT11_PATH);
    const complianceIssues = checkSAPCompliance(AGENT11_PATH);
    const allIssues = [...vulnerabilities, ...complianceIssues];
    
    const report = generateReport(allIssues);
    
    // Save detailed report
    fs.writeFileSync(
        path.join(__dirname, 'agent11-security-report.json'),
        JSON.stringify(report, null, 2)
    );
    
    // Print summary
    console.log('Security Scan Summary for Agent 11 (SQL Agent):');
    console.log('=' . repeat(55));
    console.log(`Total Issues Found: ${report.summary.total}`);
    console.log(`  - Critical: ${report.summary.critical}`);
    console.log(`  - High: ${report.summary.high}`);
    console.log(`  - Medium: ${report.summary.medium}`);
    console.log(`  - Low: ${report.summary.low}`);
    console.log('\nIssues by Type:');
    Object.entries(report.byType).forEach(([type, issues]) => {
        console.log(`  - ${type}: ${issues.length}`);
    });
    
    console.log('\nCritical Issues (SQL Injection & Code Injection):');
    allIssues.filter(v => v.severity === 'CRITICAL').forEach(vuln => {
        console.log(`  - ${vuln.file}:${vuln.line} - ${vuln.message}`);
    });
    
    console.log('\nSQL-Specific Security Issues:');
    allIssues.filter(v => ['sqlSecurity', 'databaseSecurity', 'queryValidation', 'sqlInjection'].includes(v.type))
        .forEach(vuln => {
            console.log(`  - ${vuln.file}:${vuln.line} - ${vuln.message}`);
        });
    
    console.log('\nReport saved to: agent11-security-report.json');
    
} catch (error) {
    console.error('Error during security scan:', error);
    process.exit(1);
}

module.exports = { scanDirectory, generateReport };