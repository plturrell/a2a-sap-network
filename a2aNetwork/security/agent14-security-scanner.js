#!/usr/bin/env node

/**
 * Agent 14 (Embedding Fine-Tuner Agent) Security Scanner
 * 
 * Comprehensive security vulnerability scanner specifically designed for
 * Agent 14's embedding and ML model management functionality including
 * fine-tuning, model evaluation, hyperparameter optimization, and deployment.
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

class Agent14SecurityScanner {
    constructor() {
        this.vulnerabilities = [];
        this.warnings = [];
        this.info = [];
        this.scannedFiles = 0;
        this.startTime = performance.now();
        
        // Agent 14 specific security patterns for ML/embedding operations
        this.embeddingSecurityPatterns = {
            // ML model injection and manipulation (more specific patterns)
            modelInjection: [
                /eval\s*\(\s*.*model.*\+.*input/gi,
                /new\s+Function\s*\(\s*.*model.*\+/gi,
                /exec\s*\(\s*.*model.*input/gi,
                /pickle\.loads.*model.*input/gi,
                /torch\.load.*model.*map_location.*input/gi,
                /joblib\.load.*model.*input/gi,
                /numpy\.load.*model.*input/gi,
                /tf\.keras\.models\.load_model.*model.*input/gi,
                /model\.load_state_dict.*unsafe.*input/gi
            ],
            
            // Model path traversal vulnerabilities
            modelPathTraversal: [
                /model.*path.*\.\./gi,
                /file.*path.*model.*\.\./gi,
                /load.*model.*\+.*input/gi,
                /save.*model.*\+.*input/gi,
                /model.*file.*concat.*input/gi,
                /checkpoint.*path.*input/gi,
                /\.\.\/.*model/gi,
                /\.\.\\.*model/gi
            ],
            
            // Insecure WebSocket/EventSource for training monitoring
            insecureConnections: [
                /new\s+WebSocket\s*\(\s*["']ws:\/\//gi,
                /new\s+EventSource\s*\(\s*["']http:\/\//gi,
                /WebSocket\s*\(\s*["']ws:\/\/[^"']*embedding/gi,
                /EventSource\s*\(\s*["']http:\/\/[^"']*embedding/gi,
                /ws:\/\/.*embedding.*updates/gi,
                /http:\/\/.*embedding.*stream/gi
            ],
            
            // Missing CSRF protection in ML operations
            csrfMissing: [
                /callFunction\s*\(\s*["']\/GetEmbeddingStatistics/gi,
                /callFunction\s*\(\s*["']\/GetModelConfiguration/gi,
                /callFunction\s*\(\s*["']\/GetEvaluationMetrics/gi,
                /callFunction\s*\(\s*["']\/GetAvailableBenchmarks/gi,
                /callFunction\s*\(\s*["']\/GetHyperparameterSpace/gi,
                /callFunction\s*\(\s*["']\/GetVectorDatabases/gi,
                /callFunction\s*\(\s*["']\/AnalyzeModelPerformance/gi,
                /callFunction\s*\(\s*["']\/GetFineTuningOptions/gi,
                /callFunction\s*\(\s*["']\/DeployModel/gi
            ],
            
            // Hyperparameter injection risks
            hyperparameterInjection: [
                /hyperparameter.*\+.*input/gi,
                /learning.*rate.*input/gi,
                /batch.*size.*input/gi,
                /config.*\+.*input/gi,
                /parameter.*eval.*input/gi,
                /optimizer.*config.*input/gi,
                /scheduler.*config.*input/gi
            ],
            
            // Training data poisoning risks
            dataPoisoningRisks: [
                /training.*data.*load.*input/gi,
                /dataset.*path.*input/gi,
                /data.*augmentation.*input/gi,
                /samples.*inject.*input/gi,
                /training.*samples.*unsafe/gi,
                /eval.*dataset.*input/gi
            ],
            
            // Model serialization vulnerabilities
            serializationRisks: [
                /pickle\.dumps.*model/gi,
                /pickle\.loads.*unsafe/gi,
                /yaml\.load.*model.*Loader/gi,
                /json\.loads.*model.*input/gi,
                /torch\.save.*model.*input/gi,
                /tf\.saved_model\.save.*input/gi,
                /joblib\.dump.*model.*input/gi
            ],
            
            // Vector database security
            vectorDbSecurity: [
                /vector.*query.*\+.*input/gi,
                /embedding.*search.*input/gi,
                /similarity.*search.*unsafe/gi,
                /vector.*database.*query.*input/gi,
                /index.*query.*concat.*input/gi,
                /embedding.*vector.*eval.*input/gi
            ],
            
            // Model deployment security
            deploymentSecurity: [
                /deploy.*model.*http:/gi,
                /model.*endpoint.*insecure/gi,
                /serve.*model.*http:/gi,
                /model.*api.*no.*auth/gi,
                /inference.*endpoint.*http:/gi
            ],
            
            // Benchmark manipulation risks
            benchmarkSecurity: [
                /benchmark.*result.*eval.*input/gi,
                /evaluation.*metric.*input/gi,
                /score.*calculation.*input/gi,
                /benchmark.*data.*unsafe/gi,
                /metric.*computation.*eval.*input/gi
            ]
        };
        
        this.sensitiveOperations = [
            'GetEmbeddingStatistics', 'GetModelConfiguration', 'GetEvaluationMetrics',
            'GetAvailableBenchmarks', 'GetHyperparameterSpace', 'GetVectorDatabases',
            'AnalyzeModelPerformance', 'GetFineTuningOptions', 'GetEvaluationOptions',
            'GetOptimizationOptions', 'DeployModel', 'GetModelComparisons',
            'GetEmbeddingVisualization'
        ];
        
        this.embeddingPatterns = [
            'modelName', 'modelId', 'modelPath', 'hyperparameters', 'trainingData',
            'datasetPath', 'checkpointPath', 'configData', 'vectorData', 'embeddingData',
            'benchmarkData', 'evaluationData', 'optimizationConfig', 'deploymentTarget'
        ];
    }

    async scanDirectory(dirPath) {
        console.log(`\n🔍 Starting Agent 14 Embedding Fine-Tuner Security Scan...`);
        console.log(`📂 Scanning directory: ${dirPath}\n`);
        
        try {
            await this.scanFiles(dirPath);
            this.generateReport();
        } catch (error) {
            console.error(`❌ Scan failed: ${error.message}`);
            process.exit(1);
        }
    }

    async scanFiles(dirPath) {
        const files = fs.readdirSync(dirPath, { withFileTypes: true });
        
        for (const file of files) {
            const fullPath = path.join(dirPath, file.name);
            
            if (file.isDirectory()) {
                await this.scanFiles(fullPath);
            } else if (this.isJavaScriptFile(file.name)) {
                await this.scanFile(fullPath);
            }
        }
    }

    isJavaScriptFile(filename) {
        return /\.(js|ts)$/.test(filename) && 
               !filename.includes('.min.') && 
               !filename.includes('test') &&
               !filename.includes('spec') &&
               !filename.includes('SecurityUtils'); // Skip SecurityUtils to avoid false positives
    }

    async scanFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf8');
        const relativePath = path.relative(process.cwd(), filePath);
        this.scannedFiles++;

        // Skip if not Agent 14 related
        if (!this.isAgent14File(content, filePath)) {
            return;
        }

        console.log(`🔎 Scanning: ${relativePath}`);

        // Embedding-specific security scans
        this.scanModelInjection(content, relativePath);
        this.scanModelPathTraversal(content, relativePath);
        this.scanInsecureConnections(content, relativePath);
        this.scanCSRFProtection(content, relativePath);
        this.scanHyperparameterInjection(content, relativePath);
        this.scanDataPoisoningRisks(content, relativePath);
        this.scanSerializationRisks(content, relativePath);
        this.scanVectorDbSecurity(content, relativePath);
        this.scanDeploymentSecurity(content, relativePath);
        this.scanBenchmarkSecurity(content, relativePath);
        
        // General security scans
        this.scanGeneralSecurity(content, relativePath);
        this.scanInputValidation(content, relativePath);
        this.scanAuthenticationChecks(content, relativePath);
        this.scanErrorHandling(content, relativePath);
    }

    isAgent14File(content, filePath) {
        const agent14Indicators = [
            'agent14', 'EmbeddingModels', 'FineTuningDashboard', 'CreateEmbeddingModel',
            'StartFineTuning', 'ModelEvaluator', 'BenchmarkRunner', 'HyperparameterTuner',
            'VectorOptimizer', 'PerformanceAnalyzer', 'FineTuneModel', 'EvaluateModel',
            'OptimizeModel', 'DeployModel', 'TestModel', 'CompareModels', 'ExportModel',
            'VisualizeEmbeddings', 'embedding/updates', 'embedding/stream'
        ];
        
        const checkIndicator = function(indicator) {
            return content.toLowerCase().includes(indicator.toLowerCase()) ||
                filePath.toLowerCase().includes(indicator.toLowerCase());
        };
        
        return agent14Indicators.some(checkIndicator);
    }

    scanModelInjection(content, filePath) {
        const checkModelInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const addModelInjectionVulnerability = (match) => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_MODEL_INJECTION',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Potential model injection vulnerability',
                        code: match.trim(),
                        impact: 'Could allow malicious model loading and arbitrary code execution',
                        recommendation: 'Use SecurityUtils.validateModelPath() and secure model loading'
                    });
                };
                matches.forEach(addModelInjectionVulnerability);
            }
        };
        this.embeddingSecurityPatterns.modelInjection.forEach(checkModelInjectionPattern);
    }

    scanModelPathTraversal(content, filePath) {
        const checkPathTraversalPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const addPathTraversalVulnerability = (match) => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_PATH_TRAVERSAL',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Model path traversal vulnerability',
                        code: match.trim(),
                        impact: 'Could allow access to unauthorized model files',
                        recommendation: 'Validate model paths with SecurityUtils.validateModelPath()'
                    });
                };
                matches.forEach(addPathTraversalVulnerability);
            }
        };
        this.embeddingSecurityPatterns.modelPathTraversal.forEach(checkPathTraversalPattern);
    }

    scanInsecureConnections(content, filePath) {
        const checkInsecureConnectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const addInsecureConnectionVulnerability = (match) => {
                    this.vulnerabilities.push({
                        type: 'INSECURE_EMBEDDING_CONNECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure WebSocket/EventSource for embedding monitoring',
                        code: match.trim(),
                        impact: 'Training and embedding communications not encrypted',
                        recommendation: 'Use WSS/HTTPS for secure embedding communications'
                    });
                };
                matches.forEach(addInsecureConnectionVulnerability);
            }
        };
        this.embeddingSecurityPatterns.insecureConnections.forEach(checkInsecureConnectionPattern);
    }

    scanCSRFProtection(content, filePath) {
        this.embeddingSecurityPatterns.csrfMissing.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    // Check if CSRF token is present in the context
                    const functionStart = content.indexOf(match);
                    const functionBlock = content.substring(functionStart, functionStart + 500);
                    
                    if (!functionBlock.includes('securityToken') && 
                        !functionBlock.includes('X-CSRF-Token') &&
                        !functionBlock.includes('SecurityUtils.secureCallFunction')) {
                        this.vulnerabilities.push({
                            type: 'EMBEDDING_CSRF_MISSING',
                            severity: 'HIGH',
                            file: filePath,
                            line: this.getLineNumber(content, match),
                            description: 'Missing CSRF protection for embedding operation',
                            code: match.trim(),
                            impact: 'Embedding operations vulnerable to CSRF attacks',
                            recommendation: 'Use SecurityUtils.secureCallFunction() for embedding operations'
                        });
                    }
                });
            }
        });
    }

    scanHyperparameterInjection(content, filePath) {
        this.embeddingSecurityPatterns.hyperparameterInjection.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_HYPERPARAMETER_INJECTION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Hyperparameter injection vulnerability',
                        code: match.trim(),
                        impact: 'Could allow malicious hyperparameter manipulation',
                        recommendation: 'Validate hyperparameters with SecurityUtils.validateHyperparameters()'
                    });
                });
            }
        });
    }

    scanDataPoisoningRisks(content, filePath) {
        this.embeddingSecurityPatterns.dataPoisoningRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_DATA_POISONING',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Training data poisoning risk',
                        code: match.trim(),
                        impact: 'Could allow injection of malicious training data',
                        recommendation: 'Validate training data with SecurityUtils.validateTrainingData()'
                    });
                });
            }
        });
    }

    scanSerializationRisks(content, filePath) {
        this.embeddingSecurityPatterns.serializationRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_SERIALIZATION_RISK',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe model serialization/deserialization',
                        code: match.trim(),
                        impact: 'Could allow arbitrary code execution via malicious models',
                        recommendation: 'Use secure model serialization with SecurityUtils.secureModelSave()'
                    });
                });
            }
        });
    }

    scanVectorDbSecurity(content, filePath) {
        this.embeddingSecurityPatterns.vectorDbSecurity.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_VECTOR_DB_SECURITY',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Vector database security vulnerability',
                        code: match.trim(),
                        impact: 'Could allow vector database injection attacks',
                        recommendation: 'Validate vector queries with SecurityUtils.validateVectorQuery()'
                    });
                });
            }
        });
    }

    scanDeploymentSecurity(content, filePath) {
        this.embeddingSecurityPatterns.deploymentSecurity.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_DEPLOYMENT_SECURITY',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure model deployment configuration',
                        code: match.trim(),
                        impact: 'Model endpoints not secure',
                        recommendation: 'Implement secure model deployment protocols'
                    });
                });
            }
        });
    }

    scanBenchmarkSecurity(content, filePath) {
        this.embeddingSecurityPatterns.benchmarkSecurity.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'EMBEDDING_BENCHMARK_SECURITY',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Benchmark result manipulation risk',
                        code: match.trim(),
                        impact: 'Could allow manipulation of evaluation metrics',
                        recommendation: 'Secure benchmark calculations with validation'
                    });
                });
            }
        });
    }

    scanGeneralSecurity(content, filePath) {
        // Hardcoded credentials
        const credentialPatterns = [
            /password\s*[=:]\s*["'][^"']+["']/gi,
            /api[_-]?key\s*[=:]\s*["'][^"']+["']/gi,
            /secret\s*[=:]\s*["'][^"']+["']/gi,
            /token\s*[=:]\s*["'][^"']+["']/gi,
            /model.*key.*["'][^"']+["']/gi,
            /embedding.*token.*["'][^"']+["']/gi
        ];

        credentialPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'HARDCODED_CREDENTIALS',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Hardcoded credentials detected',
                        code: match.trim(),
                        impact: 'Credentials exposed in source code',
                        recommendation: 'Use secure credential management system'
                    });
                });
            }
        });

        // Unsafe functions
        const unsafeFunctions = [
            /eval\s*\(/gi,
            /Function\s*\(/gi,
            /setTimeout\s*\(\s*["'][^"']*["']/gi,
            /setInterval\s*\(\s*["'][^"']*["']/gi
        ];

        unsafeFunctions.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    // Skip if it's just the sap.ui.define function declaration
                    if (match.toLowerCase().includes('function(') && 
                        content.includes('sap.ui.define')) {
                        return;
                    }
                    
                    this.vulnerabilities.push({
                        type: 'UNSAFE_FUNCTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Use of potentially unsafe function',
                        code: match.trim(),
                        impact: 'Could allow code injection',
                        recommendation: 'Avoid eval() and similar unsafe functions'
                    });
                });
            }
        });
    }

    scanInputValidation(content, filePath) {
        // Check for embedding-specific input validation
        this.embeddingPatterns.forEach(pattern => {
            const regex = new RegExp(`${pattern}\\s*[=:]\\s*.*input`, 'gi');
            const matches = content.match(regex);
            if (matches && !content.includes('SecurityUtils.validate') && !content.includes('validateModel')) {
                matches.forEach(match => {
                    this.warnings.push({
                        type: 'MISSING_EMBEDDING_VALIDATION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: `Missing validation for embedding field: ${pattern}`,
                        code: match.trim(),
                        recommendation: 'Add SecurityUtils.validateEmbeddingField() validation'
                    });
                });
            }
        });
    }

    scanAuthenticationChecks(content, filePath) {
        const sensitiveOps = this.sensitiveOperations.filter(op => content.includes(op));
        
        sensitiveOps.forEach(op => {
            if (!content.includes('checkAuth') && !content.includes('isAuthenticated') && 
                !content.includes('SecurityUtils.checkEmbeddingAuth')) {
                this.warnings.push({
                    type: 'MISSING_AUTH_CHECK',
                    severity: 'HIGH',
                    file: filePath,
                    description: `Missing authentication check for embedding operation: ${op}`,
                    recommendation: 'Add SecurityUtils.checkEmbeddingAuth() before operations'
                });
            }
        });
    }

    scanErrorHandling(content, filePath) {
        const errorPatterns = [
            /catch\s*\([^)]*\)\s*\{\s*\}/gi,
            /\.catch\s*\(\s*\)/gi,
            /error[^}]*console\.log/gi
        ];

        errorPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.warnings.push({
                        type: 'POOR_ERROR_HANDLING',
                        severity: 'LOW',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Poor error handling detected',
                        code: match.trim(),
                        recommendation: 'Implement proper error handling with SecurityUtils.logSecureError()'
                    });
                });
            }
        });
    }

    getLineNumber(content, searchString) {
        const lines = content.substring(0, content.indexOf(searchString)).split('\n');
        return lines.length;
    }

    generateReport() {
        const endTime = performance.now();
        const duration = ((endTime - this.startTime) / 1000).toFixed(2);

        console.log('\n' + '='.repeat(80));
        console.log('🔒 AGENT 14 EMBEDDING FINE-TUNER SECURITY SCAN REPORT');
        console.log('='.repeat(80));

        // Summary
        const isCritical = function(v) { return v.severity === 'CRITICAL'; };
        const isHigh = function(v) { return v.severity === 'HIGH'; };
        const isMedium = function(v) { return v.severity === 'MEDIUM'; };
        const isLow = function(v) { return v.severity === 'LOW'; };
        
        const critical = this.vulnerabilities.filter(isCritical).length;
        const high = this.vulnerabilities.filter(isHigh).length;
        const medium = this.vulnerabilities.filter(isMedium).length;
        const low = this.vulnerabilities.filter(isLow).length;

        console.log(`\n📊 SUMMARY:`);
        console.log(`   Files Scanned: ${this.scannedFiles}`);
        console.log(`   Scan Duration: ${duration}s`);
        console.log(`   Critical Issues: ${critical}`);
        console.log(`   High Issues: ${high}`);
        console.log(`   Medium Issues: ${medium}`);
        console.log(`   Low Issues: ${low}`);
        console.log(`   Warnings: ${this.warnings.length}`);

        // Risk Score Calculation
        const riskScore = (critical * 10) + (high * 5) + (medium * 2) + (low * 1);
        const maxRisk = this.scannedFiles * 15; // Theoretical max risk per file
        const securityScore = Math.max(0, 100 - Math.min(100, (riskScore / maxRisk) * 100));

        console.log(`\n🎯 EMBEDDING SECURITY SCORE: ${securityScore.toFixed(1)}/100`);

        if (securityScore >= 90) {
            console.log('   Status: ✅ EXCELLENT - Production Ready');
        } else if (securityScore >= 75) {
            console.log('   Status: ⚠️  GOOD - Minor Issues');
        } else if (securityScore >= 50) {
            console.log('   Status: ⚠️  FAIR - Needs Improvement');
        } else {
            console.log('   Status: ❌ POOR - Significant Issues');
        }

        // Embedding-specific findings
        console.log(`\n🤖 EMBEDDING-SPECIFIC SECURITY FINDINGS:`);
        
        const hasModelInjection = function(v) { return v.type.includes('MODEL_INJECTION'); };
        const hasPathTraversal = function(v) { return v.type.includes('PATH_TRAVERSAL'); };
        const hasCSRF = function(v) { return v.type.includes('CSRF'); };
        const hasDataPoisoning = function(v) { return v.type.includes('DATA_POISONING'); };
        const hasSerializationRisk = function(v) { return v.type.includes('SERIALIZATION_RISK'); };
        const hasVectorDbSecurity = function(v) { return v.type.includes('VECTOR_DB_SECURITY'); };
        const hasInsecureConnection = function(v) { return v.type.includes('INSECURE'); };
        
        const embeddingIssues = {
            'MODEL_INJECTION': this.vulnerabilities.filter(hasModelInjection).length,
            'PATH_TRAVERSAL': this.vulnerabilities.filter(hasPathTraversal).length,
            'CSRF': this.vulnerabilities.filter(hasCSRF).length,
            'DATA_POISONING': this.vulnerabilities.filter(hasDataPoisoning).length,
            'SERIALIZATION_RISK': this.vulnerabilities.filter(hasSerializationRisk).length,
            'VECTOR_DB_SECURITY': this.vulnerabilities.filter(hasVectorDbSecurity).length,
            'INSECURE_CONNECTION': this.vulnerabilities.filter(hasInsecureConnection).length
        };

        const logEmbeddingIssue = function([type, count]) {
            if (count > 0) {
                console.log(`   ${type.replace('_', ' ')}: ${count} issues`);
            }
        };
        Object.entries(embeddingIssues).forEach(logEmbeddingIssue);

        // Detailed vulnerabilities
        if (this.vulnerabilities.length > 0) {
            console.log(`\n🚨 VULNERABILITIES FOUND:`);
            
            const logSeverityIssues = (severity) => {
                const filterBySeverity = function(v) { return v.severity === severity; };
                const issues = this.vulnerabilities.filter(filterBySeverity);
                if (issues.length > 0) {
                    console.log(`\n${severity} (${issues.length}):`);
                    const logVulnerability = function(vuln, index) {
                        console.log(`\n${index + 1}. ${vuln.type} - ${vuln.file}:${vuln.line || 'N/A'}`);
                        console.log(`   Description: ${vuln.description}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code}`);
                        console.log(`   Fix: ${vuln.recommendation}`);
                    };
                    issues.slice(0, 5).forEach(logVulnerability);
                    if (issues.length > 5) {
                        console.log(`   ... and ${issues.length - 5} more ${severity.toLowerCase()} issues`);
                    }
                }
            };
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].forEach(logSeverityIssues.bind(this));
        }

        // Recommendations
        console.log(`\n💡 AGENT 14 EMBEDDING SECURITY RECOMMENDATIONS:`);
        console.log(`\n1. 🔐 Implement SecurityUtils for embedding operations`);
        console.log(`   - Add SecurityUtils.validateModelPath() for model loading`);
        console.log(`   - Use SecurityUtils.secureCallFunction() for ML operations`);
        console.log(`   - Implement SecurityUtils.validateTrainingData() for data validation`);
        
        console.log(`\n2. 🛡️  Enhance ML-specific security`);
        console.log(`   - Validate hyperparameters with SecurityUtils.validateHyperparameters()`);
        console.log(`   - Secure model serialization with SecurityUtils.secureModelSave()`);
        console.log(`   - Validate vector queries with SecurityUtils.validateVectorQuery()`);
        
        console.log(`\n3. 🔒 Secure ML communications`);
        console.log(`   - Upgrade WebSocket to WSS for training monitoring`);
        console.log(`   - Use HTTPS for EventSource embedding streams`);
        console.log(`   - Implement secure model deployment protocols`);

        console.log(`\n4. ⚡ Monitor ML security`);
        console.log(`   - Log all model operations for audit`);
        console.log(`   - Monitor training data for poisoning attempts`);
        console.log(`   - Track model deployment and access patterns`);

        console.log('\n' + '='.repeat(80));
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('='.repeat(80));

        // Exit with appropriate code
        process.exit(critical > 0 || high > 0 ? 1 : 0);
    }
}

// Run the scanner if called directly
if (require.main === module) {
    const scanner = new Agent14SecurityScanner();
    const targetPath = process.argv[2] || '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent14';
    scanner.scanDirectory(targetPath);
}

module.exports = Agent14SecurityScanner;