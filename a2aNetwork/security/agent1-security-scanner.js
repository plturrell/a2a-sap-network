#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 1 (Data Standardization Agent) Security Scanner
 * Specialized scanner for data standardization, transformation, and ETL pipeline vulnerabilities
 */
class Agent1SecurityScanner {
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
        
        // Data Standardization-specific vulnerability patterns
        this.dataStandardizationPatterns = {
            // Data transformation injection vulnerabilities
            TRANSFORMATION_INJECTION: {
                patterns: [
                    /eval\s*\(/gi,
                    /new\s+Function\s*\(/gi,
                    /transformationScript\s*=.*eval/gi,
                    /\.transform\s*\([^)]*Function/gi,
                    /mapping\s*=.*\$\{/gi,
                    /\.applyTransformation\s*\([^)]*\+/gi,
                    /fnTransform\s*=.*Function/gi,
                    /sScript.*new\s+Function/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'DATA_TRANSFORMATION_INJECTION',
                message: 'Potential data transformation injection vulnerability',
                impact: 'Could allow malicious code execution during data transformation'
            },
            
            // Schema manipulation vulnerabilities
            SCHEMA_MANIPULATION: {
                patterns: [
                    /schema\s*=.*eval/gi,
                    /\.validateSchema\s*\([^)]*Function/gi,
                    /schemaDefinition\s*=.*\+/gi,
                    /\.updateSchema\s*\([^)]*\$\{/gi,
                    /sourceSchema\s*=.*user/gi,
                    /targetSchema\s*=.*input/gi,
                    /JSON\.parse\s*\([^)]*sScript/gi,
                    /oTemplate\s*=.*JSON\.parse/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'DATA_SCHEMA_MANIPULATION',
                message: 'Potential schema manipulation vulnerability',
                impact: 'Could allow unauthorized schema modifications and data corruption'
            },
            
            // Format validation bypass
            FORMAT_VALIDATION_BYPASS: {
                patterns: [
                    /formatValidation\s*=\s*false/gi,
                    /\.skipValidation\s*\(/gi,
                    /schemaValidation\s*=\s*false/gi,
                    /dataTypeValidation\s*=\s*false/gi,
                    /\.bypassValidation\s*\(/gi,
                    /validation\s*=\s*null/gi,
                    /\.disableValidation\s*\(/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_FORMAT_VALIDATION_BYPASS',
                message: 'Format validation bypass vulnerability',
                impact: 'Could allow processing of malicious or invalid data formats'
            },
            
            // ETL pipeline security vulnerabilities
            ETL_PIPELINE_INJECTION: {
                patterns: [
                    /pipeline\s*=.*eval/gi,
                    /\.executePipeline\s*\([^)]*Function/gi,
                    /batchProcess\s*=.*\+/gi,
                    /\.startBatchProcessing\s*\([^)]*\$\{/gi,
                    /jobId\s*=.*user/gi,
                    /processingMode\s*=.*input/gi,
                    /\.executeScript\s*\(/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_ETL_PIPELINE_INJECTION',
                message: 'ETL pipeline injection vulnerability',
                impact: 'Could allow injection of malicious code into data processing pipelines'
            },
            
            // Data mapping vulnerabilities
            MAPPING_MANIPULATION: {
                patterns: [
                    /mappingRules\s*=.*eval/gi,
                    /\.addMapping\s*\([^)]*Function/gi,
                    /fieldMapping\s*=.*\+/gi,
                    /\.validateMapping\s*\([^)]*\$\{/gi,
                    /sourceField\s*=.*user/gi,
                    /targetField\s*=.*input/gi,
                    /transformation\s*=.*eval/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_MAPPING_MANIPULATION',
                message: 'Data mapping manipulation vulnerability',
                impact: 'Could allow unauthorized modification of data field mappings'
            },
            
            // File upload security vulnerabilities
            FILE_UPLOAD_BYPASS: {
                patterns: [
                    /FileUploader.*fileType\s*:\s*\[\]/gi,
                    /maximumFileSize\s*:\s*0/gi,
                    /\.upload\s*\([^)]*skipValidation/gi,
                    /fileType.*\*/gi,
                    /acceptAllTypes\s*=\s*true/gi,
                    /\.processFile\s*\([^)]*raw/gi,
                    /uploadPath.*user/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_FILE_UPLOAD_BYPASS',
                message: 'File upload security bypass',
                impact: 'Could allow upload of malicious files or oversized data'
            },
            
            // Data export vulnerabilities
            DATA_EXPORT_EXPOSURE: {
                patterns: [
                    /\.exportAll\s*\(/gi,
                    /includeErrors\s*=\s*true/gi,
                    /includeMetadata\s*=\s*true/gi,
                    /\.exportSensitive\s*\(/gi,
                    /filterSensitive\s*=\s*false/gi,
                    /exportRaw\s*=\s*true/gi,
                    /\.downloadUrl.*temp/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'DATA_EXPORT_EXPOSURE',
                message: 'Sensitive data export exposure',
                impact: 'Could allow export of sensitive transformation data or error details'
            },
            
            // Standardization process bypasses
            STANDARDIZATION_BYPASS: {
                patterns: [
                    /status\s*=\s*["']COMPLETED["']/gi,
                    /\.forceComplete\s*\(/gi,
                    /approved\s*=\s*true/gi,
                    /\.skipStandardization\s*\(/gi,
                    /\.overrideStatus\s*\(/gi,
                    /bypassApproval\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_STANDARDIZATION_BYPASS',
                message: 'Data standardization process bypass',
                impact: 'Could allow bypassing of data standardization and approval workflows'
            },
            
            // Template injection vulnerabilities
            TEMPLATE_INJECTION: {
                patterns: [
                    /template\s*=.*eval/gi,
                    /\.loadTemplate\s*\([^)]*Function/gi,
                    /templatePath\s*=.*\+/gi,
                    /schemaTemplate\s*=.*\$\{/gi,
                    /\.processTemplate\s*\([^)]*user/gi,
                    /templateData.*JSON\.parse.*user/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DATA_TEMPLATE_INJECTION',
                message: 'Template injection vulnerability',
                impact: 'Could allow injection of malicious code through schema templates'
            },
            
            // Batch processing vulnerabilities
            BATCH_PROCESSING_ABUSE: {
                patterns: [
                    /batchSize\s*=.*user/gi,
                    /parallel\s*=\s*true.*user/gi,
                    /priority\s*=.*["']HIGH["'].*user/gi,
                    /\.setBatchSize\s*\([^)]*\+/gi,
                    /maxBatch\s*=\s*-1/gi,
                    /unlimitedBatch\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'DATA_BATCH_PROCESSING_ABUSE',
                message: 'Batch processing abuse vulnerability',
                impact: 'Could allow resource exhaustion through large batch processing requests'
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
            
            // Check for data standardization-specific vulnerabilities
            this.checkDataStandardizationVulnerabilities(content, filePath, lines);
            
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
            { pattern: /encodeXML\s*\(/gi, type: 'XSS_GOOD', message: 'Good: Using encodeXML for XSS prevention', isGood: true }
        ];
        
        xssPatterns.forEach(({ pattern, type, message, isGood }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                if (isGood) continue; // Skip good practices
                const lineNumber = this.getLineNumber(content, match.index);
                this.addVulnerability({
                    type: type,
                    severity: this.severityLevels.HIGH,
                    file: filePath,
                    line: lineNumber,
                    code: lines[lineNumber - 1]?.trim() || '',
                    message: message,
                    impact: 'Could allow execution of malicious scripts',
                    fix: 'Use proper output encoding and sanitization'
                });
            }
        });
        
        // CSRF vulnerabilities
        const csrfPatterns = [
            /\$\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["']/gi,
            /jQuery\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["']/gi,
            /\.post\s*\(/gi,
            /\.put\s*\(/gi,
            /\.delete\s*\(/gi,
            /fetch\s*\([^,]+,\s*\{[^}]*method\s*:\s*["'](POST|PUT|DELETE)["']/gi
        ];
        
        csrfPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if CSRF token is present nearby
                const surroundingCode = content.substring(Math.max(0, match.index - 300), match.index + 300);
                if (!surroundingCode.includes('X-CSRF-Token') && 
                    !surroundingCode.includes('csrf') &&
                    !surroundingCode.includes('X-Requested-With')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection',
                        impact: 'Could allow unauthorized state changes',
                        fix: 'Add X-CSRF-Token header to all state-changing requests'
                    });
                }
            }
        });
        
        // Insecure connections
        const insecurePatterns = [
            { pattern: /http:\/\/(?!localhost)/gi, type: 'INSECURE_CONNECTION', message: 'Insecure HTTP connection' },
            { pattern: /ws:\/\/(?!localhost)/gi, type: 'INSECURE_WEBSOCKET', message: 'Insecure WebSocket connection' }
        ];
        
        insecurePatterns.forEach(({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                // Skip comments, examples, and already secure code
                if (!code.includes('//') && !code.includes('example') && 
                    !code.includes('wss://') && !code.includes('https://')) {
                    this.addVulnerability({
                        type: type,
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: code,
                        message: message,
                        impact: 'Could expose sensitive data in transit',
                        fix: 'Use HTTPS/WSS for all connections'
                    });
                }
            }
        });
    }
    
    checkDataStandardizationVulnerabilities(content, filePath, lines) {
        Object.entries(this.dataStandardizationPatterns).forEach(([vulnType, config]) => {
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
                        fix: this.getDataStandardizationFix(vulnType)
                    });
                }
            });
        });
    }
    
    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            TRANSFORMATION_INJECTION: [
                /new\s+Function\s*\([^)]*["']/gi,  // String literals
                /validation\./gi,  // Validation operations
                /sanitized/gi,  // Sanitization operations
                /encodeXML/gi,  // XML encoding
                /escapeRegExp/gi  // Regex escaping
            ],
            SCHEMA_MANIPULATION: [
                /validateSchema\s*\([^)]*["']/gi,  // String validation
                /schemaType\s*=\s*["']/gi,  // String assignments
                /JSON\.parse\s*\([^)]*["']/gi,  // String parsing
                /validateSchema.*sanitize/gi,  // Validation with sanitization
                /SecurityUtils\.validateSchema/gi  // SecurityUtils validation
            ],
            FORMAT_VALIDATION_BYPASS: [
                /formatValidation.*default/gi,  // Default settings
                /schemaValidation.*config/gi,  // Configuration
                /dataTypeValidation.*oData\./gi  // Data binding
            ],
            FILE_UPLOAD_BYPASS: [
                /fileType.*\[["']/gi,  // Array with string literals
                /maximumFileSize.*config/gi,  // Configuration
                /acceptAllTypes.*false/gi  // Disabled setting
            ]
        };
        
        if (falsePositivePatterns[vulnType]) {
            const patterns = falsePositivePatterns[vulnType];
            if (patterns.some(pattern => pattern.test(code))) {
                return true;
            }
        }
        
        // General false positive checks
        if (filePath.includes('SecurityUtils.js') || filePath.includes('validation')) {
            // Security and validation files contain functions that may trigger patterns
            return code.includes('sanitized') || code.includes('validation') || 
                   code.includes('encodeXML') || code.includes('_validate');
        }
        
        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*') || code.includes('try {')) {
            return true;
        }
        
        // Skip legitimate validation patterns
        if (code.includes('_validateInput') || code.includes('_validateApiResponse') || 
            code.includes('_validateFileUpload') || code.includes('_validateForm')) {
            return true;
        }
        
        return false;
    }
    
    getDataStandardizationFix(vulnType) {
        const fixes = {
            TRANSFORMATION_INJECTION: 'Validate and sanitize all transformation scripts, avoid dynamic code execution',
            SCHEMA_MANIPULATION: 'Use predefined schema templates, validate all schema changes against approved patterns',
            FORMAT_VALIDATION_BYPASS: 'Always enable format validation, use whitelist-based validation approaches',
            ETL_PIPELINE_INJECTION: 'Sanitize all pipeline parameters, use parameterized execution methods',
            MAPPING_MANIPULATION: 'Validate all field mappings, use predefined transformation functions',
            FILE_UPLOAD_BYPASS: 'Implement strict file type validation, size limits, and content scanning',
            DATA_EXPORT_EXPOSURE: 'Implement access controls, data classification, and audit logging for exports',
            STANDARDIZATION_BYPASS: 'Enforce standardization workflows at the system level with proper authorization',
            TEMPLATE_INJECTION: 'Validate all template inputs, use safe template processing libraries',
            BATCH_PROCESSING_ABUSE: 'Implement rate limiting, resource quotas, and monitoring for batch operations'
        };
        return fixes[vulnType] || 'Implement proper validation and security controls';
    }
    
    checkSAPFioriCompliance(content, filePath, lines) {
        // Check for missing i18n
        if (filePath.includes('.controller.js')) {
            const i18nPatterns = [
                { pattern: /MessageToast\.show\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded message in MessageToast' },
                { pattern: /MessageBox\.\w+\s*\(\s*["'][^"']+["']/gi, message: 'Hardcoded message in MessageBox' },
                { pattern: /setText\s*\(\s*["'][^"']+["']\s*\)/gi, message: 'Hardcoded text in UI element' }
            ];
            
            i18nPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';
                    // Skip validation messages and technical messages
                    if (!code.includes('validation') && !code.includes('error') && 
                        !code.includes('_oResourceBundle')) {
                        this.addVulnerability({
                            type: 'SAP_STANDARDS',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: code,
                            message: message,
                            impact: 'Reduces internationalization support',
                            fix: 'Use i18n resource bundle for all user-facing text'
                        });
                    }
                }
            });
            
            // Check for proper input validation patterns
            const validationPatterns = [
                { pattern: /getValue\(\).*user/gi, message: 'User input without validation' },
                { pattern: /getParameter\(\s*["']value["']\s*\).*[^_validate]/gi, message: 'Parameter value without validation' },
                { pattern: /oEvent\.getParameter\s*\([^)]*\).*[^validate]/gi, message: 'Event parameter without validation' }
            ];
            
            validationPatterns.forEach(({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    const code = lines[lineNumber - 1]?.trim() || '';
                    if (!code.includes('_validateInput') && !code.includes('validation')) {
                        this.addVulnerability({
                            type: 'INPUT_VALIDATION',
                            severity: this.severityLevels.MEDIUM,
                            file: filePath,
                            line: lineNumber,
                            code: code,
                            message: message,
                            impact: 'Could allow processing of invalid or malicious input',
                            fix: 'Validate all user inputs before processing'
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
                'X-Content-Type-Options',
                'Strict-Transport-Security'
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
                        impact: 'Reduces application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
                }
            });
            
            // Check for data standardization-specific manifest issues
            if (content.includes('StandardizationTasks') || content.includes('agent1')) {
                const dataServicePattern = /"uri"\s*:\s*["'][^"']*\/a2a\/agent1[^"']*["']/gi;
                const matches = content.matchAll(dataServicePattern);
                for (const match of matches) {
                    const uri = match[0];
                    if (!uri.includes('https') && !uri.includes('localhost')) {
                        this.addVulnerability({
                            type: 'INSECURE_DATA_SERVICE',
                            severity: this.severityLevels.HIGH,
                            file: filePath,
                            line: this.getLineNumber(content, match.index),
                            code: uri,
                            message: 'Data service URI should use HTTPS',
                            impact: 'Could expose sensitive standardization data in transit',
                            fix: 'Use HTTPS for all data service endpoints'
                        });
                    }
                }
            }
        }
        
        // Check for accessibility and responsive design patterns
        if (filePath.includes('.controller.js')) {
            const accessibilityPatterns = [
                { pattern: /announceForAccessibility/gi, message: 'Good: Using accessibility announcements', isGood: true },
                { pattern: /getModel\s*\(\s*["']device["']\s*\)/gi, message: 'Good: Using device model for responsiveness', isGood: true }
            ];
            
            // Count good patterns for scoring
            accessibilityPatterns.forEach(({ pattern, isGood }) => {
                if (isGood) {
                    const matches = Array.from(content.matchAll(pattern));
                    if (matches.length > 0) {
                        // This is positive - could be used for scoring improvements
                        console.log(`✅ Found ${matches.length} accessibility/responsive design patterns in ${filePath}`);
                    }
                }
            });
        }
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
        const extensions = ['.js', '.xml', '.json', '.html'];
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
        console.log('🔒 AGENT 1 DATA STANDARDIZATION SECURITY SCAN REPORT');
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
        
        console.log(`\n🎯 DATA STANDARDIZATION SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log(`   Status: ✅ EXCELLENT - Well secured`);
        } else if (score >= 70) {
            console.log(`   Status: ⚠️  GOOD - Minor issues to address`);
        } else if (score >= 50) {
            console.log(`   Status: ⚠️  FAIR - Several issues need attention`);
        } else {
            console.log(`   Status: ❌ POOR - Significant security improvements needed`);
        }
        
        // Data standardization-specific findings
        const dataStandardizationIssues = this.vulnerabilities.filter(v => v.type.startsWith('DATA_'));
        if (dataStandardizationIssues.length > 0) {
            console.log(`\n🔄 DATA STANDARDIZATION-SPECIFIC SECURITY FINDINGS:`);
            const issueCounts = {};
            dataStandardizationIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });
            
            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type.replace('DATA_', '')}: ${count} issues`);
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
                        console.log(`   Code: ${vuln.code.substring(0, 60)}${vuln.code.length > 60 ? '...' : ''}`);
                        console.log(`   Fix: ${vuln.fix}\n`);
                        issueNumber++;
                    });
                }
            });
        }
        
        // Data standardization security recommendations
        console.log(`💡 AGENT 1 DATA STANDARDIZATION SECURITY RECOMMENDATIONS:\n`);
        console.log(`1. 🔐 Secure transformation scripts`);
        console.log(`   - Validate all transformation scripts before execution`);
        console.log(`   - Use safe evaluation environments (sandboxing)`);
        console.log(`   - Implement script content filtering and sanitization`);
        console.log(`   - Avoid dynamic code execution (eval, new Function)`);
        
        console.log(`\n2. 🛡️  Protect schema integrity`);
        console.log(`   - Validate all schema templates against approved patterns`);
        console.log(`   - Implement schema versioning and approval workflows`);
        console.log(`   - Use cryptographic signatures for schema validation`);
        console.log(`   - Prevent unauthorized schema modifications`);
        
        console.log(`\n3. 🔒 Secure ETL pipeline operations`);
        console.log(`   - Implement input validation for all pipeline parameters`);
        console.log(`   - Use resource quotas and rate limiting for batch operations`);
        console.log(`   - Monitor and audit all data transformation activities`);
        console.log(`   - Implement secure communication between pipeline components`);
        
        console.log(`\n4. ⚡ Validate data formats and mappings`);
        console.log(`   - Always enable format validation for data inputs`);
        console.log(`   - Use whitelist-based validation for file types and formats`);
        console.log(`   - Validate all field mappings before transformation`);
        console.log(`   - Implement data lineage tracking for audit purposes`);
        
        console.log(`\n5. 🚀 Secure file upload and export`);
        console.log(`   - Implement strict file type and size validation`);
        console.log(`   - Scan uploaded files for malicious content`);
        console.log(`   - Apply data classification controls to exports`);
        console.log(`   - Use secure temporary file handling and cleanup`);
        
        console.log(`\n6. 🎯 Monitor standardization workflows`);
        console.log(`   - Implement comprehensive audit logging`);
        console.log(`   - Monitor for unusual processing patterns`);
        console.log(`   - Set up alerts for security violations`);
        console.log(`   - Regular security assessments of transformation logic`);
        
        console.log('\n' + '='.repeat(80));
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('Focus on transformation injection and schema manipulation vulnerabilities.');
        console.log('='.repeat(80));
    }
    
    run(targetPath) {
        console.log('🔍 Starting Agent 1 Data Standardization Security Scan...');
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
const scanner = new Agent1SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent1';
scanner.run(targetPath);