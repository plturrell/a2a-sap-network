#!/usr/bin/env node

/**
 * Agent 13 (Agent Builder Agent) Security Scanner
 * 
 * Comprehensive security vulnerability scanner specifically designed for
 * Agent 13's agent building functionality including code generation,
 * deployment management, template creation, and pipeline orchestration.
 */

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

class Agent13SecurityScanner {
    constructor() {
        this.vulnerabilities = [];
        this.warnings = [];
        this.info = [];
        this.scannedFiles = 0;
        this.startTime = performance.now();
        
        // Agent 13 specific security patterns for agent building
        this.builderSecurityPatterns = {
            // Code injection risks in code generation
            codeInjection: [
                /eval\s*\(\s*.*template/gi,
                /Function\s*\(\s*.*template/gi,
                /new\s+Function\s*\(\s*.*template/gi,
                /exec\s*\(\s*.*generate/gi,
                /child_process.*exec.*template/gi,
                /vm\.runInNewContext.*template/gi,
                /vm\.runInThisContext.*template/gi,
                /require\s*\(\s*.*template/gi,
                /import\s*\(\s*.*template/gi
            ],
            
            // Template injection vulnerabilities
            templateInjection: [
                /innerHTML\s*=\s*.*template/gi,
                /outerHTML\s*=\s*.*template/gi,
                /document\.write\s*\(.*template/gi,
                /\$\{.*template.*\}/gi,
                /template.*\+.*input/gi,
                /template.*\+.*user/gi,
                /template.*concat.*input/gi,
                /replace\s*\(.*input.*template/gi
            ],
            
            // Insecure WebSocket/EventSource for build monitoring
            insecureConnections: [
                /new\s+WebSocket\s*\(\s*["']ws:\/\//gi,
                /new\s+EventSource\s*\(\s*["']http:\/\//gi,
                /WebSocket\s*\(\s*["']ws:\/\/[^"']*builder/gi,
                /EventSource\s*\(\s*["']http:\/\/[^"']*builder/gi,
                /ws:\/\/.*builder.*updates/gi,
                /http:\/\/.*builder.*stream/gi
            ],
            
            // Missing CSRF protection in builder operations
            csrfMissing: [
                /callFunction\s*\(\s*["']\/GenerateAgent/gi,
                /callFunction\s*\(\s*["']\/BuildAgent/gi,
                /callFunction\s*\(\s*["']\/DeployAgent/gi,
                /callFunction\s*\(\s*["']\/ValidateTemplate/gi,
                /callFunction\s*\(\s*["']\/CloneTemplate/gi,
                /callFunction\s*\(\s*["']\/StartBatchBuild/gi,
                /callFunction\s*\(\s*["']\/GetTemplateDetails/gi,
                /callFunction\s*\(\s*["']\/GetDeploymentTargets/gi,
                /callFunction\s*\(\s*["']\/GetBuildPipelines/gi
            ],
            
            // Deployment configuration security risks
            deploymentRisks: [
                /environment\s*=\s*.*input/gi,
                /deploy.*target.*input/gi,
                /container.*image.*input/gi,
                /kubernetes.*config.*input/gi,
                /docker.*command.*input/gi,
                /pipeline.*script.*input/gi,
                /build.*command.*\+/gi,
                /exec.*deploy.*\+/gi
            ],
            
            // File system access vulnerabilities
            fileSystemRisks: [
                /fs\.readFile\s*\(.*input/gi,
                /fs\.writeFile\s*\(.*input/gi,
                /path\.join\s*\(.*input/gi,
                /require\s*\(.*input/gi,
                /import\s*\(.*input/gi,
                /process\.exec.*input/gi,
                /child_process.*input/gi
            ],
            
            // Repository and source code risks
            repositoryRisks: [
                /git\s+clone.*input/gi,
                /git\s+pull.*input/gi,
                /repository.*url.*input/gi,
                /source.*url.*input/gi,
                /branch.*name.*input/gi,
                /commit.*hash.*input/gi,
                /tag.*name.*input/gi
            ],
            
            // Configuration injection risks
            configInjection: [
                /config.*\+.*input/gi,
                /configuration.*concat.*input/gi,
                /YAML\.load.*input/gi,
                /JSON\.parse.*input.*config/gi,
                /environment.*\[.*input/gi,
                /process\.env\[.*input/gi
            ],
            
            // Build pipeline security
            pipelineRisks: [
                /pipeline.*exec.*input/gi,
                /build.*script.*\+.*input/gi,
                /command.*line.*input/gi,
                /shell.*command.*input/gi,
                /bash.*script.*input/gi,
                /powershell.*script.*input/gi
            ],
            
            // Agent deployment security
            agentDeploymentSecurity: [
                /agent.*deploy.*http:/gi,
                /deploy.*target.*insecure/gi,
                /agent.*endpoint.*http:/gi,
                /deployment.*url.*http:/gi,
                /agent.*config.*plaintext/gi
            ]
        };
        
        this.sensitiveOperations = [
            'GenerateAgent', 'BuildAgent', 'DeployAgent', 'ValidateTemplate',
            'CloneTemplate', 'StartBatchBuild', 'GetTemplateDetails', 
            'GetDeploymentTargets', 'GetBuildPipelines', 'GetAgentComponents',
            'GetTestConfiguration', 'GetTestSuite', 'GetDeploymentOptions',
            'GetAgentConfiguration', 'GetBuilderStatistics'
        ];
        
        this.builderPatterns = [
            'templateName', 'templateId', 'agentCode', 'deploymentConfig',
            'buildScript', 'pipelineConfig', 'repositoryUrl', 'branchName',
            'environmentVariables', 'secretsConfig', 'containerImage', 'buildCommand'
        ];
    }

    async scanDirectory(dirPath) {
        console.log(`\n🔍 Starting Agent 13 Agent Builder Security Scan...`);
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
               !filename.includes('spec');
    }

    async scanFile(filePath) {
        const content = fs.readFileSync(filePath, 'utf8');
        const relativePath = path.relative(process.cwd(), filePath);
        this.scannedFiles++;

        // Skip if not Agent 13 related
        if (!this.isAgent13File(content, filePath)) {
            return;
        }

        console.log(`🔎 Scanning: ${relativePath}`);

        // Builder-specific security scans
        this.scanCodeInjection(content, relativePath);
        this.scanTemplateInjection(content, relativePath);
        this.scanInsecureConnections(content, relativePath);
        this.scanCSRFProtection(content, relativePath);
        this.scanDeploymentRisks(content, relativePath);
        this.scanFileSystemRisks(content, relativePath);
        this.scanRepositoryRisks(content, relativePath);
        this.scanConfigInjection(content, relativePath);
        this.scanPipelineRisks(content, relativePath);
        this.scanAgentDeploymentSecurity(content, relativePath);
        
        // General security scans
        this.scanGeneralSecurity(content, relativePath);
        this.scanInputValidation(content, relativePath);
        this.scanAuthenticationChecks(content, relativePath);
        this.scanErrorHandling(content, relativePath);
    }

    isAgent13File(content, filePath) {
        const agent13Indicators = [
            'agent13', 'AgentBuilder', 'GenerateAgent', 'BuildAgent', 'DeployAgent',
            'ValidateTemplate', 'CloneTemplate', 'StartBatchBuild', 'TemplateWizard',
            'CodeGenerator', 'DeploymentManager', 'PipelineManager', 'ComponentBuilder',
            'TestHarness', 'builder/updates', 'builder/stream', 'BuilderDashboard'
        ];
        
        const checkAgent13Indicator = (indicator) => 
            content.toLowerCase().includes(indicator.toLowerCase()) ||
            filePath.toLowerCase().includes(indicator.toLowerCase());
        
        return agent13Indicators.some(checkAgent13Indicator);
    }

    scanCodeInjection(content, filePath) {
        const checkCodeInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCodeInjectionMatch = (match) => {
                    // Skip false positives - check if it's using SecurityUtils
                    const functionStart = content.indexOf(match);
                    const functionBlock = content.substring(functionStart, functionStart + 500);
                    
                    if (!functionBlock.includes('SecurityUtils.secureCallFunction') &&
                        !functionBlock.includes('SecurityUtils.validateAgentCode') &&
                        !filePath.includes('SecurityUtils')) {
                        this.vulnerabilities.push({
                            type: 'BUILDER_CODE_INJECTION',
                            severity: 'CRITICAL',
                            file: filePath,
                            line: this.getLineNumber(content, match),
                            description: 'Potential code injection vulnerability in agent generation',
                            code: match.trim(),
                            impact: 'Could allow arbitrary code execution during agent building',
                            recommendation: 'Use secure code generation with SecurityUtils.validateAgentCode()'
                        });
                    }
                };
                matches.forEach(processCodeInjectionMatch);
            }
        };
        this.builderSecurityPatterns.codeInjection.forEach(checkCodeInjectionPattern);
    }

    scanTemplateInjection(content, filePath) {
        const checkTemplateInjectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processTemplateInjectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_TEMPLATE_INJECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Template injection vulnerability in agent builder',
                        code: match.trim(),
                        impact: 'Could allow script injection through agent templates',
                        recommendation: 'Use SecurityUtils.sanitizeTemplate() for template processing'
                    });
                };
                matches.forEach(processTemplateInjectionMatch);
            }
        };
        this.builderSecurityPatterns.templateInjection.forEach(checkTemplateInjectionPattern);
    }

    scanInsecureConnections(content, filePath) {
        const checkInsecureConnectionPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processInsecureConnectionMatch = (match) => {
                    this.vulnerabilities.push({
                        type: 'INSECURE_BUILDER_CONNECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure WebSocket/EventSource for builder updates',
                        code: match.trim(),
                        impact: 'Builder communications not encrypted',
                        recommendation: 'Use WSS/HTTPS for secure builder communications'
                    });
                };
                matches.forEach(processInsecureConnectionMatch);
            }
        };
        this.builderSecurityPatterns.insecureConnections.forEach(checkInsecureConnectionPattern);
    }

    scanCSRFProtection(content, filePath) {
        const checkCSRFPattern = (pattern) => {
            const matches = content.match(pattern);
            if (matches) {
                const processCSRFMatch = (match) => {
                    // Check if CSRF token is present in the context
                    const functionStart = content.indexOf(match);
                    const functionBlock = content.substring(functionStart, functionStart + 500);
                    
                    if (!functionBlock.includes('securityToken') && 
                        !functionBlock.includes('X-CSRF-Token') &&
                        !functionBlock.includes('SecurityUtils.secureCallFunction')) {
                        this.vulnerabilities.push({
                            type: 'BUILDER_CSRF_MISSING',
                            severity: 'HIGH',
                            file: filePath,
                            line: this.getLineNumber(content, match),
                            description: 'Missing CSRF protection for builder operation',
                            code: match.trim(),
                            impact: 'Builder operations vulnerable to CSRF attacks',
                            recommendation: 'Use SecurityUtils.secureCallFunction() for builder operations'
                        });
                    }
                };
                matches.forEach(processCSRFMatch);
            }
        };
        this.builderSecurityPatterns.csrfMissing.forEach(checkCSRFPattern);
    }

    scanDeploymentRisks(content, filePath) {
        this.builderSecurityPatterns.deploymentRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_DEPLOYMENT_RISK',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe deployment configuration handling',
                        code: match.trim(),
                        impact: 'Could allow malicious deployment configurations',
                        recommendation: 'Implement SecurityUtils.validateDeploymentConfig()'
                    });
                });
            }
        });
    }

    scanFileSystemRisks(content, filePath) {
        this.builderSecurityPatterns.fileSystemRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_FILESYSTEM_RISK',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe file system access in agent builder',
                        code: match.trim(),
                        impact: 'Could allow unauthorized file system access',
                        recommendation: 'Validate file paths with SecurityUtils.validateFilePath()'
                    });
                });
            }
        });
    }

    scanRepositoryRisks(content, filePath) {
        this.builderSecurityPatterns.repositoryRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_REPOSITORY_RISK',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe repository operation',
                        code: match.trim(),
                        impact: 'Could allow access to malicious repositories',
                        recommendation: 'Validate repository URLs with SecurityUtils.validateRepositoryURL()'
                    });
                });
            }
        });
    }

    scanConfigInjection(content, filePath) {
        this.builderSecurityPatterns.configInjection.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_CONFIG_INJECTION',
                        severity: 'HIGH',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Configuration injection vulnerability',
                        code: match.trim(),
                        impact: 'Could allow arbitrary configuration injection',
                        recommendation: 'Sanitize configuration with SecurityUtils.validateConfiguration()'
                    });
                });
            }
        });
    }

    scanPipelineRisks(content, filePath) {
        this.builderSecurityPatterns.pipelineRisks.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_PIPELINE_RISK',
                        severity: 'CRITICAL',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Unsafe pipeline execution vulnerability',
                        code: match.trim(),
                        impact: 'Could allow arbitrary command execution in build pipelines',
                        recommendation: 'Validate pipeline commands with SecurityUtils.validatePipelineCommand()'
                    });
                });
            }
        });
    }

    scanAgentDeploymentSecurity(content, filePath) {
        this.builderSecurityPatterns.agentDeploymentSecurity.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                matches.forEach(match => {
                    this.vulnerabilities.push({
                        type: 'BUILDER_AGENT_DEPLOYMENT_SECURITY',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: 'Insecure agent deployment configuration',
                        code: match.trim(),
                        impact: 'Agent deployments not secure',
                        recommendation: 'Implement secure agent deployment protocols'
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
            /docker.*password.*["'][^"']+["']/gi,
            /registry.*token.*["'][^"']+["']/gi
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
        // Check for builder-specific input validation
        this.builderPatterns.forEach(pattern => {
            const regex = new RegExp(`${pattern}\\s*[=:]\\s*.*input`, 'gi');
            const matches = content.match(regex);
            if (matches && !content.includes('SecurityUtils.validate') && !content.includes('validateTemplate')) {
                matches.forEach(match => {
                    this.warnings.push({
                        type: 'MISSING_BUILDER_VALIDATION',
                        severity: 'MEDIUM',
                        file: filePath,
                        line: this.getLineNumber(content, match),
                        description: `Missing validation for builder field: ${pattern}`,
                        code: match.trim(),
                        recommendation: 'Add SecurityUtils.validateBuilderField() validation'
                    });
                });
            }
        });
    }

    scanAuthenticationChecks(content, filePath) {
        const sensitiveOps = this.sensitiveOperations.filter(op => content.includes(op));
        
        sensitiveOps.forEach(op => {
            if (!content.includes('checkAuth') && !content.includes('isAuthenticated') && 
                !content.includes('SecurityUtils.checkBuilderAuth')) {
                this.warnings.push({
                    type: 'MISSING_AUTH_CHECK',
                    severity: 'HIGH',
                    file: filePath,
                    description: `Missing authentication check for builder operation: ${op}`,
                    recommendation: 'Add SecurityUtils.checkBuilderAuth() before operations'
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
        console.log('🔒 AGENT 13 AGENT BUILDER SECURITY SCAN REPORT');
        console.log('='.repeat(80));

        // Summary
        const critical = this.vulnerabilities.filter(v => v.severity === 'CRITICAL').length;
        const high = this.vulnerabilities.filter(v => v.severity === 'HIGH').length;
        const medium = this.vulnerabilities.filter(v => v.severity === 'MEDIUM').length;
        const low = this.vulnerabilities.filter(v => v.severity === 'LOW').length;

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

        console.log(`\n🎯 AGENT BUILDER SECURITY SCORE: ${securityScore.toFixed(1)}/100`);

        if (securityScore >= 90) {
            console.log('   Status: ✅ EXCELLENT - Production Ready');
        } else if (securityScore >= 75) {
            console.log('   Status: ⚠️  GOOD - Minor Issues');
        } else if (securityScore >= 50) {
            console.log('   Status: ⚠️  FAIR - Needs Improvement');
        } else {
            console.log('   Status: ❌ POOR - Significant Issues');
        }

        // Builder-specific findings
        console.log(`\n🏗️  BUILDER-SPECIFIC SECURITY FINDINGS:`);
        
        const builderIssues = {
            'CODE_INJECTION': this.vulnerabilities.filter(v => v.type.includes('CODE_INJECTION')).length,
            'TEMPLATE_INJECTION': this.vulnerabilities.filter(v => v.type.includes('TEMPLATE_INJECTION')).length,
            'CSRF': this.vulnerabilities.filter(v => v.type.includes('CSRF')).length,
            'DEPLOYMENT_RISK': this.vulnerabilities.filter(v => v.type.includes('DEPLOYMENT_RISK')).length,
            'PIPELINE_RISK': this.vulnerabilities.filter(v => v.type.includes('PIPELINE_RISK')).length,
            'FILESYSTEM_RISK': this.vulnerabilities.filter(v => v.type.includes('FILESYSTEM_RISK')).length,
            'INSECURE_CONNECTION': this.vulnerabilities.filter(v => v.type.includes('INSECURE')).length
        };

        Object.entries(builderIssues).forEach(([type, count]) => {
            if (count > 0) {
                console.log(`   ${type.replace('_', ' ')}: ${count} issues`);
            }
        });

        // Detailed vulnerabilities
        if (this.vulnerabilities.length > 0) {
            console.log(`\n🚨 VULNERABILITIES FOUND:`);
            
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].forEach(severity => {
                const issues = this.vulnerabilities.filter(v => v.severity === severity);
                if (issues.length > 0) {
                    console.log(`\n${severity} (${issues.length}):`);
                    issues.forEach((vuln, index) => {
                        console.log(`\n${index + 1}. ${vuln.type} - ${vuln.file}:${vuln.line || 'N/A'}`);
                        console.log(`   Description: ${vuln.description}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code}`);
                        console.log(`   Fix: ${vuln.recommendation}`);
                    });
                }
            });
        }

        // Recommendations
        console.log(`\n💡 AGENT 13 BUILDER SECURITY RECOMMENDATIONS:`);
        console.log(`\n1. 🔐 Implement SecurityUtils for builder operations`);
        console.log(`   - Add SecurityUtils.validateAgentCode() for code generation`);
        console.log(`   - Use SecurityUtils.secureCallFunction() for builder operations`);
        console.log(`   - Implement SecurityUtils.sanitizeTemplate() for templates`);
        
        console.log(`\n2. 🛡️  Enhance builder-specific security`);
        console.log(`   - Validate deployment configurations with SecurityUtils.validateDeploymentConfig()`);
        console.log(`   - Sanitize pipeline commands with SecurityUtils.validatePipelineCommand()`);
        console.log(`   - Secure file system access with path validation`);
        
        console.log(`\n3. 🔒 Secure builder communications`);
        console.log(`   - Upgrade WebSocket to WSS for builder updates`);
        console.log(`   - Use HTTPS for EventSource builder streams`);
        console.log(`   - Implement secure agent deployment protocols`);

        console.log(`\n4. ⚡ Monitor builder security`);
        console.log(`   - Log all builder operations for audit`);
        console.log(`   - Monitor code generation for malicious patterns`);
        console.log(`   - Track deployment configuration changes`);

        console.log('\n' + '='.repeat(80));
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('='.repeat(80));

        // Exit with appropriate code
        process.exit(critical > 0 || high > 0 ? 1 : 0);
    }
}

// Run the scanner if called directly
if (require.main === module) {
    const scanner = new Agent13SecurityScanner();
    const targetPath = process.argv[2] || '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent13';
    scanner.scanDirectory(targetPath);
}

module.exports = Agent13SecurityScanner;