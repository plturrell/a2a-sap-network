#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 2 (AI Preparation Agent) Security Scanner
 * Specialized scanner for AI data preparation, model training, and ML pipeline vulnerabilities
 */
class Agent2SecurityScanner {
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
        
        // AI Preparation-specific vulnerability patterns
        this.aiPreparationPatterns = {
            // Model training data poisoning vulnerabilities
            DATA_POISONING: {
                patterns: [
                    /trainingData\s*=.*eval/gi,
                    /\.loadDataset\s*\([^)]*Function/gi,
                    /dataLoader\s*=.*\+/gi,
                    /\.addTrainingData\s*\([^)]*\$\{/gi,
                    /features\s*=.*user/gi,
                    /\.injectData\s*\(/gi,
                    /unsafeData\s*=\s*true/gi,
                    /skipValidation\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'AI_DATA_POISONING',
                message: 'Potential training data poisoning vulnerability',
                impact: 'Could allow injection of malicious data into AI training sets, compromising model integrity'
            },
            
            // Feature engineering injection vulnerabilities
            FEATURE_INJECTION: {
                patterns: [
                    /featureEngineering\s*=.*eval/gi,
                    /\.calculateFeature\s*\([^)]*Function/gi,
                    /featureFormula\s*=.*\+/gi,
                    /\.transformFeature\s*\([^)]*\$\{/gi,
                    /customFeature\s*=.*user/gi,
                    /featureCode\s*=.*input/gi,
                    /\.addFeature\s*\([^)]*eval/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'AI_FEATURE_INJECTION',
                message: 'Feature engineering injection vulnerability',
                impact: 'Could allow malicious feature manipulation affecting model training and predictions'
            },
            
            // Data preprocessing vulnerabilities
            PREPROCESSING_BYPASS: {
                patterns: [
                    /preprocessing\s*=.*eval/gi,
                    /\.skipPreprocessing\s*\(/gi,
                    /validateInput\s*=\s*false/gi,
                    /\.rawData\s*\([^)]*user/gi,
                    /sanitizeData\s*=\s*false/gi,
                    /\.bypassValidation\s*\(/gi,
                    /preprocessingConfig\s*=.*\+/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'AI_PREPROCESSING_BYPASS',
                message: 'Data preprocessing bypass vulnerability',
                impact: 'Could allow bypassing of data validation and sanitization in AI pipelines'
            },
            
            // AI pipeline security vulnerabilities
            PIPELINE_MANIPULATION: {
                patterns: [
                    /pipeline\s*=.*eval/gi,
                    /\.setPipelineConfig\s*\([^)]*Function/gi,
                    /pipelineSteps\s*=.*\+/gi,
                    /\.modifyPipeline\s*\([^)]*\$\{/gi,
                    /executionOrder\s*=.*user/gi,
                    /\.overridePipeline\s*\(/gi,
                    /unsafePipeline\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'AI_PIPELINE_MANIPULATION',
                message: 'AI pipeline manipulation vulnerability',
                impact: 'Could allow unauthorized modification of ML pipeline execution and data flow'
            },
            
            // Model parameter manipulation vulnerabilities
            MODEL_TAMPERING: {
                patterns: [
                    /modelParameters\s*=.*eval/gi,
                    /\.setHyperparams\s*\([^)]*Function/gi,
                    /hyperparameters\s*=.*\+/gi,
                    /\.updateModel\s*\([^)]*\$\{/gi,
                    /modelConfig\s*=.*user/gi,
                    /\.tamperModel\s*\(/gi,
                    /bypassModelValidation\s*=\s*true/gi,
                    /weights\s*=.*input/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'AI_MODEL_TAMPERING',
                message: 'Model parameter tampering vulnerability',
                impact: 'Could allow unauthorized modification of model hyperparameters and weights'
            },
            
            // Training dataset validation bypass vulnerabilities
            VALIDATION_BYPASS: {
                patterns: [
                    /datasetValidation\s*=\s*false/gi,
                    /\.skipDataValidation\s*\(/gi,
                    /validateQuality\s*=\s*false/gi,
                    /\.overrideValidation\s*\(/gi,
                    /checkDataIntegrity\s*=\s*false/gi,
                    /\.forceAcceptData\s*\(/gi,
                    /validationThreshold\s*=\s*0/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'AI_VALIDATION_BYPASS',
                message: 'Training dataset validation bypass',
                impact: 'Could allow poor quality or malicious data to enter training pipelines'
            },
            
            // Embedding and vector manipulation vulnerabilities
            EMBEDDING_MANIPULATION: {
                patterns: [
                    /embeddings\s*=.*eval/gi,
                    /\.generateEmbeddings\s*\([^)]*Function/gi,
                    /vectorData\s*=.*\+/gi,
                    /\.modifyEmbeddings\s*\([^)]*\$\{/gi,
                    /embeddingModel\s*=.*user/gi,
                    /\.poisonVectors\s*\(/gi,
                    /unsafeEmbeddings\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'AI_EMBEDDING_MANIPULATION',
                message: 'Embedding manipulation vulnerability',
                impact: 'Could allow manipulation of vector embeddings affecting similarity and clustering'
            },
            
            // Model export and serialization vulnerabilities
            MODEL_EXPORT_EXPOSURE: {
                patterns: [
                    /\.exportModel\s*\([^)]*includeWeights\s*:\s*true/gi,
                    /serializePrivateData\s*=\s*true/gi,
                    /\.exportSensitive\s*\(/gi,
                    /includeTrainingData\s*=\s*true/gi,
                    /exportModelSecrets\s*=\s*true/gi,
                    /\.dumpModel\s*\(/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'AI_MODEL_EXPORT_EXPOSURE',
                message: 'Model export exposure vulnerability',
                impact: 'Could allow exposure of sensitive model parameters or training data during export'
            },
            
            // AutoML security vulnerabilities
            AUTOML_MANIPULATION: {
                patterns: [
                    /autoMLConfig\s*=.*eval/gi,
                    /\.setAutoMLParams\s*\([^)]*Function/gi,
                    /autoMLSearch\s*=.*\+/gi,
                    /\.overrideAutoML\s*\([^)]*\$\{/gi,
                    /trialConfig\s*=.*user/gi,
                    /\.manipulateSearch\s*\(/gi,
                    /unsafeAutoML\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'AI_AUTOML_MANIPULATION',
                message: 'AutoML manipulation vulnerability',
                impact: 'Could allow manipulation of automated ML processes and hyperparameter search'
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
            
            // Check for AI preparation-specific vulnerabilities
            this.checkAIPreparationVulnerabilities(content, filePath, lines);
            
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
                        impact: 'Could allow execution of malicious scripts in AI preparation interfaces',
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
                    !surroundingCode.includes('_getSecureHeaders') &&
                    !surroundingCode.includes('_csrfToken')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection in AI preparation requests',
                        impact: 'Could allow unauthorized AI preparation operations',
                        fix: 'Add CSRF token using _getSecureHeaders() method'
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
                        impact: 'Could expose sensitive AI training data in transit',
                        fix: 'Use HTTPS/WSS for all AI service connections'
                    });
                }
            }
        });
        
        // Input validation vulnerabilities specific to AI contexts
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
                    impact: 'Could allow arbitrary code execution in AI preparation context',
                    fix: 'Remove eval() and Function constructor usage, use safe alternatives'
                });
            }
        });
    }
    
    checkAIPreparationVulnerabilities(content, filePath, lines) {
        Object.entries(this.aiPreparationPatterns).forEach(([vulnType, config]) => {
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
                        fix: this.getAIPreparationFix(vulnType)
                    });
                }
            });
        });
    }
    
    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            DATA_POISONING: [
                /trainingData\s*=\s*["']/gi,  // String literals
                /_validateInput\./gi,  // Validation functions
                /sanitized/gi  // Sanitization operations
            ],
            FEATURE_INJECTION: [
                /featureEngineering\s*=\s*["']/gi,  // String assignments
                /_sanitizeFeature\./gi,  // Sanitization functions
                /validation\./gi  // Validation operations
            ],
            MODEL_TAMPERING: [
                /modelParameters\s*=\s*["']/gi,  // String literals
                /_validateModel\./gi,  // Model validation functions
                /sanitized/gi  // Sanitization operations
            ],
            VALIDATION_BYPASS: [
                /datasetValidation\s*=\s*oData\./gi,  // Data binding
                /validateQuality\s*=\s*this\._/gi,  // Method calls
                /_validate/gi  // Validation methods
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
    
    getAIPreparationFix(vulnType) {
        const fixes = {
            DATA_POISONING: 'Implement strict data validation and sanitization before training data ingestion',
            FEATURE_INJECTION: 'Validate all feature engineering operations and use predefined feature templates',
            PREPROCESSING_BYPASS: 'Enforce mandatory data preprocessing and validation in AI pipelines',
            PIPELINE_MANIPULATION: 'Use immutable pipeline configurations and validate all pipeline modifications',
            MODEL_TAMPERING: 'Implement model integrity checks and parameter validation',
            VALIDATION_BYPASS: 'Enforce mandatory dataset quality validation before training',
            EMBEDDING_MANIPULATION: 'Validate embedding generation processes and implement vector integrity checks',
            MODEL_EXPORT_EXPOSURE: 'Implement access controls and data classification for model exports',
            AUTOML_MANIPULATION: 'Use controlled AutoML configurations and validate hyperparameter ranges'
        };
        return fixes[vulnType] || 'Implement proper validation and security controls for AI operations';
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
                    if (!code.includes('this.getResourceBundle()') && !code.includes('i18n')) {
                        this.addVulnerability({
                            type: 'SAP_STANDARDS',
                            severity: this.severityLevels.LOW,
                            file: filePath,
                            line: lineNumber,
                            code: code,
                            message: message,
                            impact: 'Reduces internationalization support for AI preparation interfaces',
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
                        impact: 'Reduces AI preparation application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
                }
            });
        }
        
        // Check for AI-specific accessibility issues
        if (filePath.includes('.controller.js') || filePath.includes('.fragment.xml')) {
            const accessibilityPatterns = [
                { pattern: /VizFrame\s*\([^)]*\)/gi, message: 'Chart accessibility: Missing ARIA labels for AI visualizations' },
                { pattern: /MicroChart\s*\([^)]*\)/gi, message: 'MicroChart accessibility: Missing ARIA labels' },
                { pattern: /setBusy\s*\(\s*true\s*\)/gi, message: 'Loading state accessibility: Consider announcing status changes' }
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
                            impact: 'Reduces accessibility for AI preparation interfaces',
                            fix: 'Add appropriate ARIA labels and accessibility announcements'
                        });
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
        console.log('🤖 AGENT 2 AI PREPARATION SECURITY SCAN REPORT');
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
        
        console.log(`\n🎯 AI PREPARATION SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log(`   Status: ✅ EXCELLENT - AI preparation pipeline is well secured`);
        } else if (score >= 70) {
            console.log(`   Status: ⚠️  GOOD - Minor AI security issues to address`);
        } else if (score >= 50) {
            console.log(`   Status: ⚠️  FAIR - Several AI security issues need attention`);
        } else {
            console.log(`   Status: ❌ POOR - Significant AI security improvements needed`);
        }
        
        // AI-specific findings
        const aiIssues = this.vulnerabilities.filter(v => v.type.startsWith('AI_'));
        if (aiIssues.length > 0) {
            console.log(`\n🤖 AI PREPARATION-SPECIFIC SECURITY FINDINGS:`);
            const issueCounts = {};
            aiIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });
            
            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type.replace('AI_', '')}: ${count} issues`);
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
        
        // AI preparation security recommendations
        console.log(`💡 AGENT 2 AI PREPARATION SECURITY RECOMMENDATIONS:\n`);
        console.log(`1. 🛡️  Secure AI Data Pipeline`);
        console.log(`   - Implement strict validation for all training data inputs`);
        console.log(`   - Use cryptographic checksums to verify data integrity`);
        console.log(`   - Sanitize and validate all feature engineering operations`);
        console.log(`   - Implement audit logging for all data preparation steps`);
        
        console.log(`\n2. 🔒 Model Training Security`);
        console.log(`   - Validate all hyperparameters and model configurations`);
        console.log(`   - Implement secure parameter storage and access controls`);
        console.log(`   - Use differential privacy techniques for sensitive data`);
        console.log(`   - Monitor for adversarial inputs during training`);
        
        console.log(`\n3. 🔐 Feature Engineering Protection`);
        console.log(`   - Use predefined, validated feature transformation templates`);
        console.log(`   - Implement feature importance validation and monitoring`);
        console.log(`   - Sanitize all custom feature engineering code`);
        console.log(`   - Validate feature distributions for anomalies`);
        
        console.log(`\n4. ⚡ Pipeline Integrity`);
        console.log(`   - Use immutable pipeline configurations`);
        console.log(`   - Implement end-to-end pipeline validation`);
        console.log(`   - Monitor pipeline execution for anomalous behavior`);
        console.log(`   - Use containerization for pipeline isolation`);
        
        console.log(`\n5. 🔍 Model Export Security`);
        console.log(`   - Implement access controls for model exports`);
        console.log(`   - Validate export configurations and data inclusion`);
        console.log(`   - Use encryption for sensitive model data`);
        console.log(`   - Audit all model export operations`);
        
        console.log(`\n6. 🎯 AutoML Security`);
        console.log(`   - Validate AutoML configuration parameters`);
        console.log(`   - Monitor hyperparameter search boundaries`);
        console.log(`   - Implement resource limits for AutoML processes`);
        console.log(`   - Validate AutoML-generated models before deployment`);
        
        console.log('\n' + '='.repeat(80));
        console.log('AI Security Scan completed. Address critical AI vulnerabilities first.');
        console.log('Focus on data poisoning and model tampering prevention.');
        console.log('='.repeat(80));
        
        // Generate JSON report for further processing
        this.saveJSONReport();
    }
    
    saveJSONReport() {
        const reportData = {
            agent: 'Agent2-AIPreparation',
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
                'Implement strict training data validation',
                'Secure feature engineering operations',
                'Protect model parameters from tampering',
                'Enforce pipeline integrity checks',
                'Implement secure model export controls',
                'Validate AutoML configurations'
            ]
        };
        
        try {
            fs.writeFileSync('agent2-security-report.json', JSON.stringify(reportData, null, 2));
            console.log('\n📄 Detailed report saved to: agent2-security-report.json');
        } catch (error) {
            console.error('Failed to save JSON report:', error.message);
        }
    }
    
    run(targetPath) {
        console.log('🔍 Starting Agent 2 AI Preparation Security Scan...');
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
const scanner = new Agent2SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent2';
scanner.run(targetPath);