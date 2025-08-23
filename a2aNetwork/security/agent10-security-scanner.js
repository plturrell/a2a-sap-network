const fs = require('fs');
const path = require('path');

// Agent 10 (Calculation Agent) Security Scanner
// Comprehensive security analysis for calculation and mathematical UI components

const AGENT10_PATH = path.join(__dirname, '../app/a2aFiori/webapp/ext/agent10');

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
        unsafeFormulaExecution: /\.(formula|expression)\s*\)/g,
        mathExpressionInjection: /parse.*Formula|execute.*Expression/gi
    },
    
    // Formula Injection Vulnerabilities
    formulaInjection: {
        directFormulaEval: /eval.*formula|execute.*expression/gi,
        unsafeCalculation: /calculate\s*\(\s*[^)]*user.*input/gi,
        dynamicFunctionCall: /\[.*\]\s*\(/g,
        unsafeMathParsing: /parse.*\(.*formula.*\)/gi,
        scriptInFormula: /<script|javascript:|on\w+=/gi
    },
    
    // CSRF Vulnerabilities
    csrf: {
        missingCSRFToken: /callFunction.*\n(?!.*headers.*x-csrf-token)/g,
        unprotectedStateChange: /(Execute|Run|Start|Create|Update|Delete).*callFunction/g,
        missingSecurityHeaders: /\$\.ajax\s*\(\s*\{[^}]*(?!.*beforeSend)/g
    },
    
    // Authentication/Authorization Issues
    auth: {
        missingAuthCheck: /on(Execute|Run|Optimize|Validate|Create|Delete|Update)[^{]*\{(?![^}]*checkAuth)/g,
        hardcodedCredentials: /(password|token|key)\s*[:=]\s*["'][^"']+["']/gi,
        exposedAPIKeys: /(api[_-]?key|secret)\s*[:=]\s*["'][^"']+["']/gi
    },
    
    // Input Validation Issues
    validation: {
        missingFormulaValidation: /formula.*value.*(?!.*validate)/gi,
        missingParameterValidation: /getParameter.*(?!.*validate)/g,
        unsanitizedCalculationInput: /calculation.*input.*(?!.*sanitize)/gi,
        missingBoundaryChecks: /parse.*number.*(?!.*isFinite)/gi,
        noOverflowProtection: /Math\.(pow|exp).*(?!.*MAX_SAFE_INTEGER)/g
    },
    
    // Insecure Communications
    insecure: {
        httpEndpoints: /http:\/\/(?!localhost|127\.0\.0\.1)/g,
        unencryptedWebSocket: /ws:\/\/(?!localhost|127\.0\.0\.1)/g,
        hardcodedURLs: /(http|ws)s?:\/\/[^/]+/g,
        insecureEventSource: /new\s+EventSource\s*\(\s*["']http:/g
    },
    
    // Error Handling Issues
    errorHandling: {
        exposedErrors: /catch.*\{[^}]*message[^}]*show/g,
        detailedErrorMessages: /error\.(stack|message|details).*MessageToast/g,
        unhandledPromises: /\.then\s*\([^)]*\)\s*(?!\.catch)/g,
        missingErrorBoundaries: /calculation.*(?!.*try.*catch)/gi
    },
    
    // Data Exposure
    dataExposure: {
        sensitiveDataInLogs: /console\.(log|info|debug).*\(.*(password|token|key|secret|formula)/gi,
        exposedCalculationData: /localStorage\.setItem.*\(.*(result|calculation|formula)/gi,
        unencryptedStorage: /sessionStorage\.setItem.*(?!.*encrypt)/g,
        exposedInternalData: /return.*\{.*internal.*:/gi
    },
    
    // Resource Management
    resources: {
        memoryLeaks: /addEventListener.*(?!.*removeEventListener)/g,
        unclosedConnections: /new\s*(WebSocket|EventSource).*(?!.*close)/g,
        infiniteLoops: /while\s*\(.*true.*\)|for\s*\(.*;\s*;\s*\)/g,
        unboundedCalculations: /calculate.*recursive.*(?!.*limit)/gi
    },
    
    // SAP Fiori Specific
    sapFiori: {
        missingResourceBundle: /getText\s*\([^)]*\)(?!.*getResourceBundle)/g,
        hardcodedTexts: /MessageToast\.show\s*\(\s*["'][^{]/g,
        improperExtensionPoint: /ControllerExtension\.extend.*(?!.*override)/g,
        missingSecurityUtil: /(?!SecurityUtil\.).*escape.*HTML/gi
    }
};

const specificAgent10Checks = {
    // Calculation-specific vulnerabilities
    calculationSecurity: {
        unsafeFormulaExecution: /executeCalculation.*(?!.*validateFormula)/g,
        missingPrecisionValidation: /precision.*(?!.*validate)/gi,
        unprotectedMathOperations: /Math\.(pow|exp|sqrt).*(?!.*try.*catch)/g,
        missingOverflowDetection: /calculate.*(?!.*MAX_SAFE_INTEGER|isFinite)/gi,
        insecureStatisticalAnalysis: /statistical.*analysis.*(?!.*sanitize.*input)/gi
    },
    
    webSocketSecurity: {
        unencryptedWebSocket: /new\s+WebSocket\s*\(\s*['"]ws:/g,
        missingWebSocketAuth: /WebSocket.*(?!.*auth.*token)/g,
        noWebSocketValidation: /ws\.onmessage.*(?!.*validate.*data)/g
    },
    
    eventSourceSecurity: {
        unencryptedEventSource: /new\s+EventSource\s*\(\s*['"]http:/g,
        missingEventValidation: /addEventListener.*(?!.*validate.*event\.data)/g
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
    
    // Agent 10 specific checks
    if (fileName.includes('controller') || fileName.includes('utils')) {
        Object.entries(specificAgent10Checks).forEach(([category, patterns]) => {
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
                        message: getAgent10Message(category, name),
                        fix: getAgent10Fix(category, name)
                    });
                }
            });
        });
    }
    
    return vulnerabilities;
}

function getSeverity(category, name) {
    const criticalPatterns = ['evalUsage', 'functionConstructor', 'directHtmlInjection', 'formulaInjection', 'missingCSRFToken'];
    const highPatterns = ['xss', 'csrf', 'auth', 'validation', 'insecure'];
    
    if (criticalPatterns.includes(name)) return 'CRITICAL';
    if (highPatterns.some(p => category.includes(p))) return 'HIGH';
    return 'MEDIUM';
}

function getSecurityMessage(category, name) {
    const messages = {
        xss: {
            unsafeMessageToast: 'MessageToast with dynamic content can lead to XSS',
            evalUsage: 'eval() usage is a critical security risk',
            functionConstructor: 'Function constructor can execute arbitrary code',
            mathExpressionInjection: 'Mathematical expression parsing without validation'
        },
        formulaInjection: {
            directFormulaEval: 'Direct formula evaluation without validation',
            unsafeCalculation: 'Calculation with unvalidated user input',
            unsafeMathParsing: 'Mathematical parsing without sanitization'
        },
        csrf: {
            missingCSRFToken: 'Missing CSRF token in state-changing operation',
            unprotectedStateChange: 'State-changing operation without CSRF protection'
        },
        validation: {
            missingFormulaValidation: 'Formula input not validated before execution',
            missingBoundaryChecks: 'Missing boundary checks for numerical operations',
            noOverflowProtection: 'Mathematical operation without overflow protection'
        },
        insecure: {
            httpEndpoints: 'Insecure HTTP endpoint usage',
            unencryptedWebSocket: 'WebSocket connection without encryption',
            hardcodedURLs: 'Hardcoded URLs should be configurable'
        }
    };
    
    return messages[category]?.[name] || `Security issue: ${category} - ${name}`;
}

function getFixSuggestion(category, name) {
    const fixes = {
        xss: {
            unsafeMessageToast: 'Use SecurityUtil.escapeHTML() for dynamic content',
            evalUsage: 'Use safe alternatives like JSON.parse or specific calculation libraries',
            mathExpressionInjection: 'Validate and sanitize mathematical expressions before parsing'
        },
        formulaInjection: {
            directFormulaEval: 'Use CalculationUtils.validateFormula() before execution',
            unsafeCalculation: 'Implement input validation and sanitization'
        },
        csrf: {
            missingCSRFToken: 'Add CSRF token to request headers',
            unprotectedStateChange: 'Implement CSRF protection for all state changes'
        },
        validation: {
            missingFormulaValidation: 'Add formula validation using CalculationUtils',
            noOverflowProtection: 'Add overflow checks: if (!Number.isFinite(result))'
        },
        insecure: {
            unencryptedWebSocket: 'Use wss:// instead of ws://',
            httpEndpoints: 'Use HTTPS for all external endpoints'
        }
    };
    
    return fixes[category]?.[name] || 'Review and fix the security issue';
}

function getAgent10Message(category, name) {
    const messages = {
        calculationSecurity: {
            unsafeFormulaExecution: 'Formula execution without validation',
            missingPrecisionValidation: 'Precision level not validated',
            unprotectedMathOperations: 'Mathematical operations without error handling',
            missingOverflowDetection: 'Missing overflow detection in calculations'
        },
        webSocketSecurity: {
            unencryptedWebSocket: 'WebSocket connection not encrypted',
            missingWebSocketAuth: 'WebSocket missing authentication'
        },
        eventSourceSecurity: {
            unencryptedEventSource: 'EventSource using insecure HTTP',
            missingEventValidation: 'Server-sent events not validated'
        }
    };
    
    return messages[category]?.[name] || `Agent 10 specific issue: ${category}`;
}

function getAgent10Fix(category, name) {
    const fixes = {
        calculationSecurity: {
            unsafeFormulaExecution: 'Validate formula with CalculationUtils.validateFormula()',
            missingPrecisionValidation: 'Validate precision levels before calculation',
            unprotectedMathOperations: 'Wrap math operations in try-catch blocks',
            missingOverflowDetection: 'Check Number.isFinite() and Number.isSafeInteger()'
        },
        webSocketSecurity: {
            unencryptedWebSocket: 'Use wss:// protocol with proper certificates',
            missingWebSocketAuth: 'Add authentication token to WebSocket connection'
        },
        eventSourceSecurity: {
            unencryptedEventSource: 'Use HTTPS for EventSource connections',
            missingEventValidation: 'Validate all server-sent event data'
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
            const vulnerabilities = scanFile(fullPath, path.relative(AGENT10_PATH, fullPath));
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
console.log('Starting Agent 10 (Calculation Agent) Security Scan...\n');

try {
    const vulnerabilities = scanDirectory(AGENT10_PATH);
    const complianceIssues = checkSAPCompliance(AGENT10_PATH);
    const allIssues = [...vulnerabilities, ...complianceIssues];
    
    const report = generateReport(allIssues);
    
    // Save detailed report
    fs.writeFileSync(
        path.join(__dirname, 'agent10-security-report.json'),
        JSON.stringify(report, null, 2)
    );
    
    // Print summary
    console.log('Security Scan Summary for Agent 10 (Calculation Agent):');
    console.log('=' . repeat(60));
    console.log(`Total Issues Found: ${report.summary.total}`);
    console.log(`  - Critical: ${report.summary.critical}`);
    console.log(`  - High: ${report.summary.high}`);
    console.log(`  - Medium: ${report.summary.medium}`);
    console.log(`  - Low: ${report.summary.low}`);
    console.log('\nIssues by Type:');
    Object.entries(report.byType).forEach(([type, issues]) => {
        console.log(`  - ${type}: ${issues.length}`);
    });
    
    console.log('\nCritical Issues:');
    allIssues.filter(v => v.severity === 'CRITICAL').forEach(vuln => {
        console.log(`  - ${vuln.file}:${vuln.line} - ${vuln.message}`);
    });
    
    console.log('\nAgent 10 Specific Issues:');
    allIssues.filter(v => ['calculationSecurity', 'webSocketSecurity', 'eventSourceSecurity'].includes(v.type))
        .forEach(vuln => {
            console.log(`  - ${vuln.file}:${vuln.line} - ${vuln.message}`);
        });
    
    console.log('\nReport saved to: agent10-security-report.json');
    
} catch (error) {
    console.error('Error during security scan:', error);
    process.exit(1);
}

module.exports = { scanDirectory, generateReport };