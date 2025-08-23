#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 4 (Calculation Validation Agent) Security Scanner
 * Specialized scanner for formula processing, mathematical expression validation, 
 * calculation validation, and numeric computation vulnerabilities
 */
class Agent4SecurityScanner {
    constructor() {
        this.vulnerabilities = [];
        this.severityLevels = {
            CRITICAL: 'CRITICAL',
            HIGH: 'HIGH',
            MEDIUM: 'MEDIUM',
            LOW: 'LOW',
            WARNING: 'WARNING'
        };
        this.scanStartTime = Date.now();
        this.filesScanned = 0;
        
        // Calculation Validation-specific vulnerability patterns
        this.calculationValidationPatterns = {
            // Formula injection attacks
            FORMULA_INJECTION: {
                patterns: [
                    /formulaExpression\s*=.*eval/gi,
                    /\.executeFormula\s*\([^)]*Function/gi,
                    /formulaData\s*=.*\$\{/gi,
                    /\.injectFormula\s*\([^)]*exec/gi,
                    /calculationQuery\s*=.*user.*\+/gi,
                    /\.manipulateFormula\s*\(/gi,
                    /unsafeFormula\s*=\s*true/gi,
                    /formulaInput\s*=.*input.*eval/gi,
                    /mathematicalExpression\s*=.*Function/gi,
                    /\.dynamicFormula\s*\([^)]*\+/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'FORMULA_INJECTION',
                message: 'Potential formula injection vulnerability',
                impact: 'Could allow injection of malicious formulas leading to code execution, data manipulation, or system compromise'
            },
            
            // Mathematical expression injection
            EXPRESSION_INJECTION: {
                patterns: [
                    /mathExpression\s*=.*eval/gi,
                    /\.parseMath\s*\([^)]*Function/gi,
                    /expression\s*=.*\+.*user/gi,
                    /\.evaluateMath\s*\([^)]*\$\{/gi,
                    /customExpression\s*=.*input/gi,
                    /\.processMath\s*\([^)]*exec/gi,
                    /mathValidation\s*=\s*false/gi,
                    /skipMathValidation\s*=\s*true/gi,
                    /unsafeMathEval\s*=\s*true/gi,
                    /\.directMathExec\s*\(/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'EXPRESSION_INJECTION',
                message: 'Mathematical expression injection vulnerability',
                impact: 'Could allow injection of malicious mathematical expressions leading to arbitrary code execution'
            },
            
            // Calculation bypass vulnerabilities
            CALCULATION_BYPASS: {
                patterns: [
                    /calculationValidation\s*=\s*false/gi,
                    /\.bypassCalculation\s*\(/gi,
                    /skipCalculationCheck\s*=\s*true/gi,
                    /\.overrideCalculation\s*\([^)]*\$\{/gi,
                    /forceCalculationResult\s*=.*user/gi,
                    /\.manipulateResult\s*\(/gi,
                    /calculationSecurityCheck\s*=\s*false/gi,
                    /bypassValidation\s*=\s*true/gi,
                    /\.directResultSet\s*\(/gi,
                    /unsafeCalculation\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'CALCULATION_BYPASS',
                message: 'Calculation validation bypass vulnerability',
                impact: 'Could allow bypassing calculation validation leading to incorrect results or data integrity issues'
            },
            
            // Numeric overflow/underflow vulnerabilities
            NUMERIC_OVERFLOW: {
                patterns: [
                    /Number\.MAX_VALUE\s*\*\s*[^0]/gi,
                    /Math\.pow\s*\([^)]*999/gi,
                    /calculation\s*\*=\s*\d{10,}/gi,
                    /value\s*=.*Infinity/gi,
                    /\.toFixed\s*\(\s*\d{3,}\s*\)/gi,
                    /overflowProtection\s*=\s*false/gi,
                    /\.preventOverflow\s*=\s*false/gi,
                    /maxCalculationValue\s*=.*MAX_VALUE/gi,
                    /infinityCheck\s*=\s*false/gi,
                    /\.allowOverflow\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'NUMERIC_OVERFLOW',
                message: 'Numeric overflow/underflow vulnerability',
                impact: 'Could cause numeric overflow/underflow leading to incorrect calculations and potential system instability'
            },
            
            // Validation rule manipulation
            VALIDATION_MANIPULATION: {
                patterns: [
                    /validationRules\s*=.*eval/gi,
                    /\.setValidationRule\s*\([^)]*Function/gi,
                    /customRule\s*=.*\+.*user/gi,
                    /\.overrideValidation\s*\([^)]*\$\{/gi,
                    /validationConfig\s*=.*input/gi,
                    /\.bypassRule\s*\(/gi,
                    /ruleValidation\s*=\s*false/gi,
                    /skipRuleCheck\s*=\s*true/gi,
                    /\.manipulateRule\s*\(/gi,
                    /unsafeValidation\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'VALIDATION_MANIPULATION',
                message: 'Validation rule manipulation vulnerability',
                impact: 'Could allow manipulation of validation rules compromising calculation integrity and accuracy'
            },
            
            // Result tampering vulnerabilities
            RESULT_TAMPERING: {
                patterns: [
                    /calculationResult\s*=.*eval/gi,
                    /\.setResult\s*\([^)]*Function/gi,
                    /resultValue\s*=.*\+.*user/gi,
                    /\.tamperResult\s*\([^)]*\$\{/gi,
                    /finalResult\s*=.*input/gi,
                    /\.overrideResult\s*\(/gi,
                    /resultIntegrity\s*=\s*false/gi,
                    /skipResultValidation\s*=\s*true/gi,
                    /\.manipulateOutput\s*\(/gi,
                    /unsafeResult\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'RESULT_TAMPERING',
                message: 'Calculation result tampering vulnerability',
                impact: 'Could allow tampering with calculation results leading to data integrity compromise'
            },
            
            // Precision manipulation vulnerabilities
            PRECISION_MANIPULATION: {
                patterns: [
                    /precisionThreshold\s*=.*eval/gi,
                    /\.setPrecision\s*\([^)]*Function/gi,
                    /precision\s*=.*\+.*user/gi,
                    /\.manipulatePrecision\s*\([^)]*\$\{/gi,
                    /customPrecision\s*=.*input/gi,
                    /\.overridePrecision\s*\(/gi,
                    /precisionCheck\s*=\s*false/gi,
                    /skipPrecisionValidation\s*=\s*true/gi,
                    /\.alterPrecision\s*\(/gi,
                    /unsafePrecision\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'PRECISION_MANIPULATION',
                message: 'Calculation precision manipulation vulnerability',
                impact: 'Could allow manipulation of calculation precision affecting accuracy and reliability of results'
            },
            
            // Batch processing vulnerabilities
            BATCH_PROCESSING_VULN: {
                patterns: [
                    /batchSize\s*=.*user/gi,
                    /\.processBatch\s*\([^)]*eval/gi,
                    /batchConfig\s*=.*\+/gi,
                    /\.executeBatch\s*\([^)]*\$\{/gi,
                    /batchLimit\s*=\s*-1/gi,
                    /\.overrideBatchSize\s*\(/gi,
                    /maxBatchSize\s*=.*MAX_VALUE/gi,
                    /batchValidation\s*=\s*false/gi,
                    /\.processBatchUnsafe\s*\(/gi,
                    /unlimitedBatch\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'BATCH_PROCESSING_VULNERABILITY',
                message: 'Batch processing security vulnerability',
                impact: 'Could allow abuse of batch processing features leading to resource exhaustion or denial of service'
            },
            
            // Formula builder vulnerabilities
            FORMULA_BUILDER_VULN: {
                patterns: [
                    /formulaBuilder\s*=.*eval/gi,
                    /\.buildFormula\s*\([^)]*Function/gi,
                    /formulaTemplate\s*=.*\+.*user/gi,
                    /\.generateFormula\s*\([^)]*\$\{/gi,
                    /customFormula\s*=.*input/gi,
                    /\.injectIntoFormula\s*\(/gi,
                    /formulaSecurity\s*=\s*false/gi,
                    /skipFormulaValidation\s*=\s*true/gi,
                    /\.buildUnsafeFormula\s*\(/gi,
                    /allowAnyFormula\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'FORMULA_BUILDER_VULNERABILITY',
                message: 'Formula builder security vulnerability',
                impact: 'Could allow construction of malicious formulas through the formula builder interface'
            },
            
            // Benchmark manipulation vulnerabilities
            BENCHMARK_MANIPULATION: {
                patterns: [
                    /benchmarkConfig\s*=.*eval/gi,
                    /\.runBenchmark\s*\([^)]*Function/gi,
                    /benchmarkData\s*=.*\+.*user/gi,
                    /\.manipulateBenchmark\s*\([^)]*\$\{/gi,
                    /benchmarkResults\s*=.*input/gi,
                    /\.overrideBenchmark\s*\(/gi,
                    /benchmarkSecurity\s*=\s*false/gi,
                    /skipBenchmarkValidation\s*=\s*true/gi,
                    /\.benchmarkUnsafe\s*\(/gi,
                    /allowFakeBenchmark\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'BENCHMARK_MANIPULATION',
                message: 'Benchmark manipulation vulnerability',
                impact: 'Could allow manipulation of benchmark results affecting performance analysis and optimization decisions'
            },
            
            // Template injection vulnerabilities
            TEMPLATE_INJECTION: {
                patterns: [
                    /calculationTemplate\s*=.*eval/gi,
                    /\.loadTemplate\s*\([^)]*Function/gi,
                    /templateData\s*=.*\+.*user/gi,
                    /\.processTemplate\s*\([^)]*\$\{/gi,
                    /customTemplate\s*=.*input/gi,
                    /\.injectTemplate\s*\(/gi,
                    /templateValidation\s*=\s*false/gi,
                    /skipTemplateCheck\s*=\s*true/gi,
                    /\.executeTemplate\s*\(/gi,
                    /unsafeTemplate\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'TEMPLATE_INJECTION',
                message: 'Calculation template injection vulnerability',
                impact: 'Could allow injection of malicious content into calculation templates'
            },
            
            // Report generation vulnerabilities
            REPORT_GENERATION_VULN: {
                patterns: [
                    /reportData\s*=.*eval/gi,
                    /\.generateReport\s*\([^)]*Function/gi,
                    /reportTemplate\s*=.*\+.*user/gi,
                    /\.injectReport\s*\([^)]*\$\{/gi,
                    /customReport\s*=.*input/gi,
                    /\.manipulateReport\s*\(/gi,
                    /reportValidation\s*=\s*false/gi,
                    /skipReportCheck\s*=\s*true/gi,
                    /\.unsafeReport\s*\(/gi,
                    /allowScriptInReport\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'REPORT_GENERATION_VULNERABILITY',
                message: 'Report generation security vulnerability',
                impact: 'Could allow injection of malicious content into generated calculation reports'
            },
            
            // Function whitelist bypass vulnerabilities
            FUNCTION_WHITELIST_BYPASS: {
                patterns: [
                    /allowedFunctions\s*=.*eval/gi,
                    /\.addFunction\s*\([^)]*Function/gi,
                    /whitelistFunction\s*=.*\+.*user/gi,
                    /\.bypassWhitelist\s*\([^)]*\$\{/gi,
                    /functionWhitelist\s*=.*input/gi,
                    /\.overrideFunctionList\s*\(/gi,
                    /whitelistCheck\s*=\s*false/gi,
                    /skipFunctionValidation\s*=\s*true/gi,
                    /\.allowAnyFunction\s*\(/gi,
                    /disableFunctionCheck\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'FUNCTION_WHITELIST_BYPASS',
                message: 'Function whitelist bypass vulnerability',
                impact: 'Could allow bypassing function whitelisting enabling execution of unauthorized functions'
            }
        };
    }
    
    scanFile(filePath) {
        console.log(`🔎 Scanning: ${filePath}`);
        this.filesScanned++;
        
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const lines = content.split('\n');
            
            // Check for general OWASP vulnerabilities
            this.checkOWASPVulnerabilities(content, filePath, lines);
            
            // Check for calculation validation-specific vulnerabilities
            this.checkCalculationValidationVulnerabilities(content, filePath, lines);
            
            // Check for SAP Fiori specific issues
            this.checkSAPFioriCompliance(content, filePath, lines);
            
        } catch (error) {
            console.error(`❌ Error scanning ${filePath}: ${error.message}`);
        }
    }
    
    checkOWASPVulnerabilities(content, filePath, lines) {
        // XSS vulnerabilities
        const xssPatterns = [
            { pattern: /innerHTML\s*=\s*[^'"`]+/gi, type: 'XSS', message: 'Potential XSS via innerHTML' },
            { pattern: /document\.write\s*\(/gi, type: 'XSS', message: 'Potential XSS via document.write' },
            { pattern: /\.html\s*\([^)]*\$\{/gi, type: 'XSS', message: 'Potential XSS via jQuery html()' },
            { pattern: /dangerouslySetInnerHTML/gi, type: 'XSS', message: 'Potential XSS via React dangerouslySetInnerHTML' },
            { pattern: /\.setText\s*\([^)]*\+/gi, type: 'XSS', message: 'Potential XSS via dynamic text setting' }
        ];
        
        xssPatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                
                // Skip if properly sanitized
                if (!code.includes('encodeXML') && !code.includes('sanitizeHTML') && !code.includes('escapeRegExp')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could allow execution of malicious scripts in calculation interfaces',
                        fix: 'Use proper output encoding and sanitization (encodeXML, sanitizeHTML)'
                    });
                }
            }
        });
        
        // CSRF vulnerabilities
        const csrfPatterns = [
            /\$\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["']/gi,
            /\.post\s*\(/gi,
            /\.put\s*\(/gi,
            /\.delete\s*\(/gi,
            /fetch\s*\([^,]+,\s*\{[^}]*method\s*:\s*["'](POST|PUT|DELETE)["']/gi
        ];
        
        csrfPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if CSRF token is present nearby
                const surroundingCode = content.substring(Math.max(0, match.index - 200), match.index + 200);
                if (!surroundingCode.includes('X-CSRF-Token') && 
                    !surroundingCode.includes('csrf') &&
                    !surroundingCode.includes('_getCSRFToken') &&
                    !surroundingCode.includes('_csrfToken')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection in calculation validation requests',
                        impact: 'Could allow unauthorized calculation validation operations',
                        fix: 'Add CSRF token using _getCSRFToken() method'
                    });
                }
            }
        });
        
        // Insecure connections
        const insecurePatterns = [
            { pattern: /http:\/\//gi, type: 'INSECURE_CONNECTION', message: 'Insecure HTTP connection' },
            { pattern: /ws:\/\//gi, type: 'INSECURE_WEBSOCKET', message: 'Insecure WebSocket connection' }
        ];
        
        insecurePatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                // Skip comments, examples, and already secured code
                if (!code.includes('//') && !code.includes('example') && 
                    !code.includes('wss://') && !code.includes('https://')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could expose sensitive calculation data and formulas in transit',
                        fix: 'Use HTTPS/WSS for all calculation validation service connections'
                    });
                }
            }
        });
        
        // Input validation vulnerabilities specific to calculation contexts
        const inputValidationPatterns = [
            { pattern: /eval\s*\(/gi, type: 'CODE_INJECTION', message: 'Code injection via eval()' },
            { pattern: /new\s+Function\s*\(/gi, type: 'CODE_INJECTION', message: 'Code injection via Function constructor' },
            { pattern: /setTimeout\s*\([^)]*\+/gi, type: 'CODE_INJECTION', message: 'Potential code injection in setTimeout' },
            { pattern: /setInterval\s*\([^)]*\+/gi, type: 'CODE_INJECTION', message: 'Potential code injection in setInterval' }
        ];
        
        inputValidationPatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                this.addVulnerability({
                    type: type,
                    severity: this.severityLevels.CRITICAL,
                    file: filePath,
                    line: lineNumber,
                    code: lines[lineNumber - 1]?.trim() || '',
                    message: message,
                    impact: 'Could allow arbitrary code execution in calculation validation context',
                    fix: 'Remove eval() and Function constructor usage, use safe alternatives'
                });
            }
        });
    }
    
    checkCalculationValidationVulnerabilities(content, filePath, lines) {
        Object.entries(this.calculationValidationPatterns).forEach(([vulnType, config]) => {
            config.patterns.forEach(pattern => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';
                    
                    // Skip false positives
                    if (this.isFalsePositive(code, vulnType, filePath)) {
                        continue;
                    }
                    
                    this.addVulnerability({
                        type: config.category,
                        severity: config.severity,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: config.message,
                        impact: config.impact,
                        fix: this.getCalculationValidationFix(vulnType)
                    });
                }
            });
        });
    }
    
    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            FORMULA_INJECTION: [
                /formulaExpression\s*=\s*["']/gi,  // String literals
                /_validateFormula\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ],
            EXPRESSION_INJECTION: [
                /mathExpression\s*=\s*["']/gi,  // String assignments
                /_validateExpression\./gi,  // Validation functions
                /validation\./gi  // Validation operations
            ],
            CALCULATION_BYPASS: [
                /calculationValidation\s*=\s*true/gi,  // Secure settings
                /_validateCalculation\./gi,  // Validation functions
                /security/gi  // Security operations
            ],
            NUMERIC_OVERFLOW: [
                /MAX_SAFE_INTEGER/gi,  // Using safe constants
                /_checkOverflow\./gi,  // Overflow checking functions
                /validation\./gi  // Validation operations
            ],
            RESULT_TAMPERING: [
                /calculationResult\s*=\s*["']/gi,  // String assignments
                /_validateResult\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ]
        };
        
        if (falsePositivePatterns[vulnType]) {
            const patterns = falsePositivePatterns[vulnType];
            if (patterns.some(pattern => pattern.test(code))) {
                return true;
            }
        }
        
        // General false positive checks
        if (filePath.includes('SecurityUtils.js') || filePath.includes('security')) {
            // Security files contain security functions that may trigger patterns
            return code.includes('sanitized') || code.includes('validation') || 
                   code.includes('_sanitize') || code.includes('_validate');
        }
        
        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*')) {
            return true;
        }
        
        // Skip console.log and debug statements
        if (code.includes('console.log') || code.includes('console.error')) {
            return true;
        }
        
        return false;
    }
    
    getCalculationValidationFix(vulnType) {
        const fixes = {
            FORMULA_INJECTION: 'Validate and sanitize all formula input data, use parameterized formula processing with whitelist validation',
            EXPRESSION_INJECTION: 'Implement strict mathematical expression validation, use safe expression parsers with function whitelisting',
            CALCULATION_BYPASS: 'Enforce calculation validation checks, implement integrity verification for all calculation operations',
            NUMERIC_OVERFLOW: 'Implement numeric bounds checking, use safe arithmetic operations and validate result ranges',
            VALIDATION_MANIPULATION: 'Protect validation rules from modification, implement immutable validation configurations',
            RESULT_TAMPERING: 'Implement result integrity checks, use cryptographic verification for calculation outputs',
            PRECISION_MANIPULATION: 'Validate precision parameters, implement strict bounds checking for precision settings',
            BATCH_PROCESSING_VULN: 'Implement proper batch size limits, validate batch processing parameters and implement rate limiting',
            FORMULA_BUILDER_VULN: 'Implement secure formula builder with strict validation, sanitize all formula components',
            BENCHMARK_MANIPULATION: 'Validate benchmark configurations, implement integrity checks for benchmark results',
            TEMPLATE_INJECTION: 'Sanitize template data, implement secure template processing with content validation',
            REPORT_GENERATION_VULN: 'Sanitize report data, implement secure report generation with output encoding',
            FUNCTION_WHITELIST_BYPASS: 'Implement strict function whitelisting, validate all function calls against approved list'
        };
        return fixes[vulnType] || 'Implement proper validation and security controls for calculation validation operations';
    }
    
    checkSAPFioriCompliance(content, filePath, lines) {
        // Check for missing i18n
        if (filePath.includes('.controller.js')) {
            const i18nPatterns = [
                { pattern: /MessageToast\.show\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded message in MessageToast' },
                { pattern: /MessageBox\.\w+\s*\(\s*["'][^"']+["']/gi, message: 'Hardcoded message in MessageBox' },
                { pattern: /setText\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded text in UI element' },
                { pattern: /headerText\s*:\s*["'][^"']+["']/gi, message: 'Hardcoded header text' }
            ];
            
            i18nPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';
                    // Skip already internationalized strings
                    if (!code.includes('getResourceBundle()') && !code.includes('i18n')) {
                        this.addVulnerability({
                            type: 'SAP_STANDARDS',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: code,
                            message: message,
                            impact: 'Reduces internationalization support for calculation validation interfaces',
                            fix: 'Use i18n resource bundle: this.getResourceBundle().getText("key")'
                        });
                    }
                }
            });
        }
        
        // Check for missing security headers in manifest
        if (filePath.includes('manifest.json')) {
            const requiredHeaders = [
                'Content-Security-Policy',
                'X-Frame-Options',
                'X-Content-Type-Options'
            ];
            
            requiredHeaders.forEach(header => {
                if (!content.includes(header)) {
                    this.addVulnerability({
                        type: 'SAP_SECURITY',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: 1,
                        code: 'manifest.json',
                        message: `Missing security header: ${header}`,
                        impact: 'Reduces calculation validation application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
                }
            });
        }
        
        // Check for calculation-specific accessibility issues
        if (filePath.includes('.controller.js') || filePath.includes('.fragment.xml')) {
            const accessibilityPatterns = [
                { pattern: /VizFrame\s*\([^)]*\)/gi, message: 'Chart accessibility: Missing ARIA labels for calculation visualizations' },
                { pattern: /MicroChart\s*\([^)]*\)/gi, message: 'MicroChart accessibility: Missing ARIA labels for calculation charts' },
                { pattern: /setBusy\s*\(\s*true\s*\)/gi, message: 'Loading state accessibility: Consider announcing calculation status changes' },
                { pattern: /formula.*result/gi, message: 'Formula result accessibility: Missing screen reader support for calculation results' },
                { pattern: /validation.*error/gi, message: 'Validation error accessibility: Missing ARIA live regions for error announcements' }
            ];
            
            accessibilityPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const surroundingCode = content.substring(Math.max(0, match.index - 100), match.index + 100);
                    if (!surroundingCode.includes('aria') && !surroundingCode.includes('announceForAccessibility')) {
                        this.addVulnerability({
                            type: 'ACCESSIBILITY',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: lines[lineNumber - 1]?.trim() || '',
                            message: message,
                            impact: 'Reduces accessibility for calculation validation interfaces',
                            fix: 'Add appropriate ARIA labels and accessibility announcements'
                        });
                    }
                }
            });
        }
        
        // Check for calculation processing performance issues
        const performancePatterns = [
            { pattern: /setTimeout.*\d{4,}/gi, message: 'Long timeout detected: May affect calculation processing responsiveness' },
            { pattern: /for\s*\([^)]*;\s*[^;]*<\s*\d{4,}/gi, message: 'Large loop detected: May cause calculation processing delays' },
            { pattern: /while\s*\([^)]*length/gi, message: 'Potentially infinite loop: Could cause calculation processing to hang' },
            { pattern: /recursion.*depth/gi, message: 'Deep recursion detected: May cause stack overflow in formula processing' }
        ];
        
        performancePatterns.forEach(({ pattern, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                this.addVulnerability({
                    type: 'PERFORMANCE',
                    severity: this.severityLevels.WARNING,
                    file: filePath,
                    line: lineNumber,
                    code: code,
                    message: message,
                    impact: 'May affect calculation validation performance and user experience',
                    fix: 'Review implementation for performance optimization and proper resource management'
                });
            }
        });
    }
    
    getLineNumber(content, index) {
        const lines = content.substring(0, index).split('\n');
        return lines.length;
    }
    
    addVulnerability(vuln) {
        // Avoid duplicates
        const exists = this.vulnerabilities.some(v => 
            v.file === vuln.file && 
            v.line === vuln.line && 
            v.type === vuln.type
        );
        
        if (!exists) {
            this.vulnerabilities.push(vuln);
        }
    }
    
    scanDirectory(dirPath) {
        const files = fs.readdirSync(dirPath);
        
        files.forEach(file => {
            const fullPath = path.join(dirPath, file);
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
                this.scanDirectory(fullPath);
            } else if (stat.isFile() && this.shouldScanFile(file)) {
                this.scanFile(fullPath);
            }
        });
    }
    
    shouldScanFile(filename) {
        const extensions = ['.js', '.xml', '.json', '.html', '.ts'];
        return extensions.some(ext => filename.endsWith(ext));
    }
    
    generateReport() {
        const scanDuration = (Date.now() - this.scanStartTime) / 1000;
        const criticalCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.CRITICAL).length;
        const highCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.HIGH).length;
        const mediumCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.MEDIUM).length;
        const lowCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.LOW).length;
        const warningCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.WARNING).length;
        
        console.log('\n' + '='.repeat(80));
        console.log('🧮 AGENT 4 CALCULATION VALIDATION SECURITY SCAN REPORT');
        console.log('='.repeat(80));
        
        console.log(`\n📊 SUMMARY:`);
        console.log(`   Files Scanned: ${this.filesScanned}`);
        console.log(`   Scan Duration: ${scanDuration.toFixed(2)}s`);
        console.log(`   Critical Issues: ${criticalCount}`);
        console.log(`   High Issues: ${highCount}`);
        console.log(`   Medium Issues: ${mediumCount}`);
        console.log(`   Low Issues: ${lowCount}`);
        console.log(`   Warnings: ${warningCount}`);
        
        // Calculate security score
        const totalIssues = criticalCount * 10 + highCount * 5 + mediumCount * 2 + lowCount;
        const maxScore = 100;
        const score = Math.max(0, maxScore - totalIssues);
        
        console.log(`\n🎯 CALCULATION VALIDATION SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log(`   Status: ✅ EXCELLENT - Calculation validation system is well secured`);
        } else if (score >= 70) {
            console.log(`   Status: ⚠️  GOOD - Minor calculation security issues to address`);
        } else if (score >= 50) {
            console.log(`   Status: ⚠️  FAIR - Several calculation security issues need attention`);
        } else {
            console.log(`   Status: ❌ POOR - Significant calculation validation security improvements needed`);
        }
        
        // Calculation-specific findings
        const calculationIssues = this.vulnerabilities.filter(v => 
            v.type.includes('FORMULA') || v.type.includes('CALCULATION') || 
            v.type.includes('EXPRESSION') || v.type.includes('VALIDATION') ||
            v.type.includes('RESULT') || v.type.includes('PRECISION') ||
            v.type.includes('BATCH') || v.type.includes('BENCHMARK'));
        if (calculationIssues.length > 0) {
            console.log(`\n🧮 CALCULATION VALIDATION-SPECIFIC SECURITY FINDINGS:`);
            const issueCounts = {};
            calculationIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });
            
            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type}: ${count} issues`);
            });
        }
        
        // List vulnerabilities by severity
        if (this.vulnerabilities.length > 0) {
            console.log(`\n🚨 VULNERABILITIES FOUND:\n`);
            
            const severityOrder = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING'];
            let issueNumber = 1;
            
            severityOrder.forEach(severity => {
                const sevVulns = this.vulnerabilities.filter(v => v.severity === severity);
                if (sevVulns.length > 0) {
                    console.log(`${severity} (${sevVulns.length}):\n`);
                    sevVulns.forEach(vuln => {
                        console.log(`${issueNumber}. ${vuln.type} - ${vuln.file}:${vuln.line}`);
                        console.log(`   Description: ${vuln.message}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code.substring(0, 80)}${vuln.code.length > 80 ? '...' : ''}`);
                        console.log(`   Fix: ${vuln.fix}\n`);
                        issueNumber++;
                    });
                }
            });
        }
        
        // Calculation validation security recommendations
        console.log(`💡 AGENT 4 CALCULATION VALIDATION SECURITY RECOMMENDATIONS:\n`);
        console.log(`1. 🛡️  Secure Formula Processing`);
        console.log(`   - Implement strict formula input validation and sanitization`);
        console.log(`   - Use function whitelisting for allowed mathematical operations`);
        console.log(`   - Validate formula syntax and structure before execution`);
        console.log(`   - Implement secure expression parsing with bounds checking`);
        
        console.log(`\n2. 🔒 Expression Injection Prevention`);
        console.log(`   - Sanitize all mathematical expressions before processing`);
        console.log(`   - Use parameterized calculation operations where possible`);
        console.log(`   - Implement expression complexity limits and validation`);
        console.log(`   - Monitor for suspicious expression patterns and injection attempts`);
        
        console.log(`\n3. 🔐 Calculation Integrity Protection`);
        console.log(`   - Implement calculation result verification and checksums`);
        console.log(`   - Validate calculation inputs and outputs for consistency`);
        console.log(`   - Use secure arithmetic operations to prevent overflow/underflow`);
        console.log(`   - Monitor calculation operations for tampering attempts`);
        
        console.log(`\n4. ⚡ Numeric Safety Controls`);
        console.log(`   - Implement numeric bounds checking for all calculations`);
        console.log(`   - Use safe arithmetic operations and overflow detection`);
        console.log(`   - Validate precision parameters and calculation limits`);
        console.log(`   - Monitor for numeric manipulation and anomalous results`);
        
        console.log(`\n5. 🔍 Validation Rule Security`);
        console.log(`   - Protect validation rules from unauthorized modification`);
        console.log(`   - Implement immutable validation configurations`);
        console.log(`   - Validate rule integrity and authenticity`);
        console.log(`   - Monitor validation rule changes and access attempts`);
        
        console.log(`\n6. 📊 Batch Processing Protection`);
        console.log(`   - Implement proper batch size limits and validation`);
        console.log(`   - Use rate limiting for batch calculation operations`);
        console.log(`   - Monitor batch processing for resource abuse`);
        console.log(`   - Validate batch configurations and processing parameters`);
        
        console.log(`\n7. 🎨 Formula Builder Security`);
        console.log(`   - Implement secure formula building with strict validation`);
        console.log(`   - Sanitize all formula components and templates`);
        console.log(`   - Use function whitelisting in formula builder interface`);
        console.log(`   - Monitor formula construction for malicious patterns`);
        
        console.log(`\n8. 📈 Benchmark Integrity`);
        console.log(`   - Validate benchmark configurations and parameters`);
        console.log(`   - Implement integrity checks for benchmark results`);
        console.log(`   - Monitor benchmark operations for manipulation attempts`);
        console.log(`   - Use secure benchmark data storage and processing`);
        
        console.log(`\n9. 📄 Report Generation Security`);
        console.log(`   - Sanitize all report data and template content`);
        console.log(`   - Implement secure report generation with output encoding`);
        console.log(`   - Validate report configurations and access permissions`);
        console.log(`   - Monitor report generation for injection attempts`);
        
        console.log(`\n10. 🔧 Template Processing Safety`);
        console.log(`    - Sanitize calculation templates and template data`);
        console.log(`    - Implement secure template processing with validation`);
        console.log(`    - Use content security policies for template rendering`);
        console.log(`    - Monitor template operations for injection attempts`);
        
        console.log('\n' + '='.repeat(80));
        console.log('Calculation Validation Security Scan completed. Address critical formula vulnerabilities first.');
        console.log('Focus on formula injection prevention and calculation integrity protection.');
        console.log('='.repeat(80));
        
        // Generate JSON report for further processing
        this.saveJSONReport();
    }
    
    saveJSONReport() {
        const reportData = {
            agent: 'Agent4-CalculationValidation',
            timestamp: new Date().toISOString(),
            summary: {
                filesScanned: this.filesScanned,
                totalVulnerabilities: this.vulnerabilities.length,
                critical: this.vulnerabilities.filter(v => v.severity === 'CRITICAL').length,
                high: this.vulnerabilities.filter(v => v.severity === 'HIGH').length,
                medium: this.vulnerabilities.filter(v => v.severity === 'MEDIUM').length,
                low: this.vulnerabilities.filter(v => v.severity === 'LOW').length,
                warnings: this.vulnerabilities.filter(v => v.severity === 'WARNING').length
            },
            vulnerabilities: this.vulnerabilities,
            recommendations: [
                'Implement secure formula processing with strict validation',
                'Prevent expression injection with sanitization and whitelisting',
                'Protect calculation integrity with verification and checksums',
                'Implement numeric safety controls and bounds checking',
                'Secure validation rules from unauthorized modification',
                'Implement batch processing protection and rate limiting',
                'Secure formula builder with strict component validation',
                'Protect benchmark integrity with configuration validation',
                'Implement secure report generation with output encoding',
                'Secure template processing with content validation'
            ]
        };
        
        try {
            fs.writeFileSync('agent4-security-report.json', JSON.stringify(reportData, null, 2));
            console.log('\n📄 Detailed report saved to: agent4-security-report.json');
        } catch (error) {
            console.error('Failed to save JSON report:', error.message);
        }
    }
    
    run(targetPath) {
        console.log('🔍 Starting Agent 4 Calculation Validation Security Scan...');
        console.log(`📂 Scanning directory: ${targetPath}\n`);
        
        if (fs.existsSync(targetPath)) {
            if (fs.statSync(targetPath).isDirectory()) {
                this.scanDirectory(targetPath);
            } else {
                this.scanFile(targetPath);
            }
            
            this.generateReport();
        } else {
            console.error(`❌ Path not found: ${targetPath}`);
        }
    }
}

// Run the scanner
const scanner = new Agent4SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent4';
scanner.run(targetPath);