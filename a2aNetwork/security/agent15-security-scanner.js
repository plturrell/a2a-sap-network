#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 15 (Orchestrator Agent) Security Scanner
 * Specialized scanner for workflow orchestration and agent coordination vulnerabilities
 */
class Agent15SecurityScanner {
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
        
        // Orchestrator-specific vulnerability patterns
        this.orchestratorPatterns = {
            // Workflow execution vulnerabilities
            WORKFLOW_INJECTION: {
                patterns: [
                    /eval\s*\(.*\+/gi,
                    /new\s+Function\s*\(.*\+/gi,
                    /setTimeout\s*\([^,)]*eval/gi,
                    /setTimeout\s*\([^,)]*Function/gi,
                    /setTimeout\s*\([^,)]*\+.*input/gi,
                    /\.execute\s*\([^)]*\$\{/gi,
                    /\.runWorkflow\s*\([^)]*\+/gi,
                    /workflowConfig\s*=.*eval/gi,
                    /workflowSteps\s*=.*Function/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'ORCHESTRATION_WORKFLOW_INJECTION',
                message: 'Potential workflow injection vulnerability',
                impact: 'Could allow malicious workflow execution and system compromise'
            },
            
            // Agent coordination vulnerabilities
            AGENT_HIJACKING: {
                patterns: [
                    /agentId\s*=\s*[^'"`;\s]+\+/gi,
                    /\.assignAgent\s*\([^)]*\$\{/gi,
                    /\.selectAgent\s*\([^)]*eval/gi,
                    /\.routeToAgent\s*\([^)]*Function/gi,
                    /agentSelection\s*=.*eval/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'ORCHESTRATION_AGENT_HIJACKING',
                message: 'Potential agent hijacking vulnerability',
                impact: 'Could allow unauthorized agent control or task redirection'
            },
            
            // Task queue manipulation
            QUEUE_POISONING: {
                patterns: [
                    /\.enqueue\s*\([^)]*\$\{/gi,
                    /\.addTask\s*\([^)]*eval/gi,
                    /taskData\s*=\s*JSON\.parse\s*\(/gi,
                    /queue\.push\s*\([^)]*user/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'ORCHESTRATION_QUEUE_POISONING',
                message: 'Task queue poisoning vulnerability',
                impact: 'Could allow injection of malicious tasks into execution queue'
            },
            
            // Event bus vulnerabilities
            EVENT_INJECTION: {
                patterns: [
                    /\.emit\s*\([^,]+,\s*[^)]*\$\{/gi,
                    /\.broadcast\s*\([^)]*user/gi,
                    /eventData\s*=\s*[^{]+\+/gi,
                    /\.trigger\s*\([^)]*eval/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'ORCHESTRATION_EVENT_INJECTION',
                message: 'Event injection vulnerability',
                impact: 'Could allow broadcasting of malicious events to agent network'
            },
            
            // Pipeline vulnerabilities
            PIPELINE_MANIPULATION: {
                patterns: [
                    /pipeline\.steps\s*=\s*[^[]+\+/gi,
                    /\.addStep\s*\([^)]*Function/gi,
                    /pipelineConfig\s*=\s*eval/gi,
                    /\.executePipeline\s*\([^)]*\$\{/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'ORCHESTRATION_PIPELINE_MANIPULATION',
                message: 'Pipeline manipulation vulnerability',
                impact: 'Could allow injection of malicious steps into execution pipeline'
            },
            
            // Consensus vulnerabilities
            CONSENSUS_BYPASS: {
                patterns: [
                    /consensus\s*=\s*false/gi,
                    /\.skipConsensus\s*\(/gi,
                    /voting\s*=\s*\[\]/gi,
                    /\.forceDecision\s*\(/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'ORCHESTRATION_CONSENSUS_BYPASS',
                message: 'Consensus mechanism bypass',
                impact: 'Could allow bypassing of multi-agent consensus requirements'
            },
            
            // Circuit breaker vulnerabilities
            CIRCUIT_BREAKER_BYPASS: {
                patterns: [
                    /circuitBreaker\s*=\s*false/gi,
                    /\.disableCircuitBreaker\s*\(/gi,
                    /failureThreshold\s*=\s*999/gi,
                    /\.resetCircuit\s*\(\s*\)/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'ORCHESTRATION_CIRCUIT_BREAKER_BYPASS',
                message: 'Circuit breaker bypass vulnerability',
                impact: 'Could allow continuous execution despite failures'
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
            
            // Check for orchestrator-specific vulnerabilities
            this.checkOrchestratorVulnerabilities(content, filePath, lines);
            
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
            { pattern: /dangerouslySetInnerHTML/gi, type: 'XSS', message: 'Potential XSS via React dangerouslySetInnerHTML' }
        ];
        
        const processXSSPattern = ({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
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
        };
        xssPatterns.forEach(processXSSPattern);
        
        // CSRF vulnerabilities
        const csrfPatterns = [
            /\$\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["']/gi,
            /\.post\s*\(/gi,
            /\.put\s*\(/gi,
            /\.delete\s*\(/gi,
            /fetch\s*\([^,]+,\s*\{[^}]*method\s*:\s*["'](POST|PUT|DELETE)["']/gi
        ];
        
        const processCSRFPattern = (pattern) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                // Check if CSRF token is present nearby or using SecurityUtils
                const surroundingCode = content.substring(Math.max(0, match.index - 200), match.index + 200);
                if (!surroundingCode.includes('X-CSRF-Token') && 
                    !surroundingCode.includes('csrf') &&
                    !surroundingCode.includes('SecurityUtils.secureCallFunction') &&
                    !surroundingCode.includes('refreshSecurityToken')) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'CSRF',
                        severity: this.severityLevels.HIGH,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: 'Missing CSRF protection',
                        impact: 'Could allow unauthorized state changes',
                        fix: 'Add CSRF token to all state-changing requests'
                    });
                }
            }
        };
        csrfPatterns.forEach(processCSRFPattern);
        
        // Insecure connections
        const insecurePatterns = [
            { pattern: /http:\/\//gi, type: 'INSECURE_CONNECTION', message: 'Insecure HTTP connection' },
            { pattern: /ws:\/\//gi, type: 'INSECURE_WEBSOCKET', message: 'Insecure WebSocket connection' }
        ];
        
        const processInsecurePattern = ({ pattern, type, message }) => {
            const matches = content.matchAll(pattern);
            for (const match of matches) {
                const lineNumber = this.getLineNumber(content, match.index);
                const code = lines[lineNumber - 1]?.trim() || '';
                // Skip comments, strings, examples, and security-fixed code
                if (!code.includes('//') && !code.includes('example') && 
                    !code.includes('SecurityUtils.createSecure') && 
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
        };
        insecurePatterns.forEach(processInsecurePattern);
    }
    
    checkOrchestratorVulnerabilities(content, filePath, lines) {
        const processVulnerabilityType = ([vulnType, config]) => {
            const processPattern = (pattern) => {
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
                        fix: this.getOrchestratorFix(vulnType)
                    });
                }
            };
            config.patterns.forEach(processPattern);
        };
        Object.entries(this.orchestratorPatterns).forEach(processVulnerabilityType);
    }
    
    isFalsePositive(code, vulnType, filePath) {
        // Skip legitimate uses that are not security vulnerabilities
        const falsePositivePatterns = {
            WORKFLOW_INJECTION: [
                /setInterval\s*\(\s*\(\s*\)\s*=>\s*\{/gi,  // Arrow function polling
                /setInterval\s*\(\s*function\s*\(\s*\)\s*\{/gi,  // Function polling
                /setTimeout\s*\(\s*\(\s*\)\s*=>\s*this\._/gi,  // Arrow function with this
                /setTimeout\s*\(\s*function\s*\(\s*\)\s*\{.*this\._/gi,  // Function with this
                /_pollInterval\s*=\s*setInterval/gi,  // Polling intervals
                /_initializePolling/gi,  // Polling initialization
                /setTimeout\s*\(\s*function\s*\(\s*\)\s*\{.*\.focus\(\)/gi,  // UI focus
                /setTimeout\s*\(\s*function\s*\(\s*\)\s*\{.*MessageToast/gi,  // UI feedback
                /setTimeout\s*\(\s*\(\s*\)\s*=>\s*\{.*\.focus\(\)/gi,  // Arrow function UI focus
                /setTimeout\s*\(\s*\(\s*\)\s*=>\s*\{.*MessageToast/gi  // Arrow function UI feedback
            ],
            AGENT_HIJACKING: [
                /validation\.sanitizedAgentId/gi,  // SecurityUtils sanitization
                /agentId\.replace\s*\(/gi,  // String sanitization
                /SecurityUtils\./gi,  // SecurityUtils functions
                /sanitized/gi  // Sanitization operations
            ]
        };
        
        if (falsePositivePatterns[vulnType]) {
            const testPattern = (pattern) => pattern.test(code);
            return falsePositivePatterns[vulnType].some(testPattern);
        }
        
        // General false positive checks
        if (filePath.includes('SecurityUtils.js')) {
            // SecurityUtils file contains security functions that may trigger patterns
            return code.includes('sanitized') || code.includes('validation') || 
                   code.includes('SecurityUtils') || code.includes('_sanitize');
        }
        
        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*')) {
            return true;
        }
        
        return false;
    }
    
    getOrchestratorFix(vulnType) {
        const fixes = {
            WORKFLOW_INJECTION: 'Use predefined workflow templates and validate all workflow configurations',
            AGENT_HIJACKING: 'Implement agent authentication and validate all agent assignments',
            QUEUE_POISONING: 'Validate and sanitize all task data before queuing',
            EVENT_INJECTION: 'Use event whitelisting and validate event payloads',
            PIPELINE_MANIPULATION: 'Use immutable pipeline configurations and validate steps',
            CONSENSUS_BYPASS: 'Enforce consensus requirements at the system level',
            CIRCUIT_BREAKER_BYPASS: 'Implement system-level circuit breaker enforcement'
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
            
            const processI18nPattern = ({ pattern, message }) => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: 'SAP_STANDARDS',
                        severity: this.severityLevels.LOW,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: message,
                        impact: 'Reduces internationalization support',
                        fix: 'Use i18n resource bundle for all user-facing text'
                    });
                }
            };
            i18nPatterns.forEach(processI18nPattern);
        }
        
        // Check for missing security headers in manifest
        if (filePath.includes('manifest.json')) {
            const requiredHeaders = [
                'Content-Security-Policy',
                'X-Frame-Options',
                'X-Content-Type-Options'
            ];
            
            const checkRequiredHeader = (header) => {
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
            };
            requiredHeaders.forEach(checkRequiredHeader);
        }
    }
    
    getLineNumber(content, index) {
        const lines = content.substring(0, index).split('\n');
        return lines.length;
    }
    
    addVulnerability(vuln) {
        // Avoid duplicates
        const isDuplicate = (v) => 
            v.file === vuln.file && 
            v.line === vuln.line && 
            v.type === vuln.type;
        const exists = this.vulnerabilities.some(isDuplicate);
        
        if (!exists) {
            this.vulnerabilities.push(vuln);
        }
    }
    
    scanDirectory(dirPath) {
        const files = fs.readdirSync(dirPath);
        
        const processFile = (file) => {
            const fullPath = path.join(dirPath, file);
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
                this.scanDirectory(fullPath);
            } else if (stat.isFile() && this.shouldScanFile(file)) {
                this.scanFile(fullPath);
            }
        };
        files.forEach(processFile);
    }
    
    shouldScanFile(filename) {
        const extensions = ['.js', '.xml', '.json', '.html'];
        const matchesExtension = (ext) => filename.endsWith(ext);
        return extensions.some(matchesExtension);
    }
    
    generateReport() {
        const scanDuration = (Date.now() - this.scanStartTime) / 1000;
        const isCritical = (v) => v.severity === this.severityLevels.CRITICAL;
        const isHigh = (v) => v.severity === this.severityLevels.HIGH;
        const isMedium = (v) => v.severity === this.severityLevels.MEDIUM;
        const isLow = (v) => v.severity === this.severityLevels.LOW;
        const isWarning = (v) => v.severity === this.severityLevels.WARNING;
        
        const criticalCount = this.vulnerabilities.filter(isCritical).length;
        const highCount = this.vulnerabilities.filter(isHigh).length;
        const mediumCount = this.vulnerabilities.filter(isMedium).length;
        const lowCount = this.vulnerabilities.filter(isLow).length;
        const warningCount = this.vulnerabilities.filter(isWarning).length;
        
        console.log('\n' + '='.repeat(80));
        console.log('🔒 AGENT 15 ORCHESTRATOR SECURITY SCAN REPORT');
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
        
        console.log(`\n🎯 ORCHESTRATION SECURITY SCORE: ${score}/100`);
        if (score >= 90) {
            console.log(`   Status: ✅ EXCELLENT - Well secured`);
        } else if (score >= 70) {
            console.log(`   Status: ⚠️  GOOD - Minor issues to address`);
        } else if (score >= 50) {
            console.log(`   Status: ⚠️  FAIR - Several issues need attention`);
        } else {
            console.log(`   Status: ❌ POOR - Significant security improvements needed`);
        }
        
        // Orchestrator-specific findings
        const isOrchestratorIssue = (v) => v.type.startsWith('ORCHESTRATION_');
        const orchestratorIssues = this.vulnerabilities.filter(isOrchestratorIssue);
        if (orchestratorIssues.length > 0) {
            console.log(`\n🤖 ORCHESTRATOR-SPECIFIC SECURITY FINDINGS:`);
            const issueCounts = {};
            const countIssueTypes = (issue) => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            };
            orchestratorIssues.forEach(countIssueTypes);
            
            const logIssueCount = ([type, count]) => {
                console.log(`   ${type.replace('ORCHESTRATION_', '')}: ${count} issues`);
            };
            Object.entries(issueCounts).forEach(logIssueCount);
        }
        
        // List vulnerabilities by severity
        if (this.vulnerabilities.length > 0) {
            console.log(`\n🚨 VULNERABILITIES FOUND:\n`);
            
            const severityOrder = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING'];
            let issueNumber = 1;
            
            const processSeverityLevel = (severity) => {
                const matchesSeverity = (v) => v.severity === severity;
                const sevVulns = this.vulnerabilities.filter(matchesSeverity);
                if (sevVulns.length > 0) {
                    console.log(`${severity} (${sevVulns.length}):\n`);
                    const logVulnerability = (vuln) => {
                        console.log(`${issueNumber}. ${vuln.type} - ${vuln.file}:${vuln.line}`);
                        console.log(`   Description: ${vuln.message}`);
                        console.log(`   Impact: ${vuln.impact}`);
                        console.log(`   Code: ${vuln.code.substring(0, 60)}${vuln.code.length > 60 ? '...' : ''}`);
                        console.log(`   Fix: ${vuln.fix}\n`);
                        issueNumber++;
                    };
                    sevVulns.forEach(logVulnerability);
                }
            };
            severityOrder.forEach(processSeverityLevel);
        }
        
        // Orchestrator security recommendations
        console.log(`💡 AGENT 15 ORCHESTRATOR SECURITY RECOMMENDATIONS:\n`);
        console.log(`1. 🔐 Implement workflow validation`);
        console.log(`   - Validate all workflow configurations before execution`);
        console.log(`   - Use predefined workflow templates where possible`);
        console.log(`   - Implement workflow signing for integrity`);
        
        console.log(`\n2. 🛡️  Secure agent coordination`);
        console.log(`   - Authenticate all agent-to-agent communications`);
        console.log(`   - Implement agent capability verification`);
        console.log(`   - Use secure channels for task distribution`);
        
        console.log(`\n3. 🔒 Protect execution pipelines`);
        console.log(`   - Validate pipeline steps before execution`);
        console.log(`   - Implement step isolation and sandboxing`);
        console.log(`   - Monitor pipeline execution for anomalies`);
        
        console.log(`\n4. ⚡ Implement resilience patterns`);
        console.log(`   - Enforce circuit breakers at system level`);
        console.log(`   - Implement proper retry and backoff strategies`);
        console.log(`   - Monitor consensus mechanisms for manipulation`);
        
        console.log('\n' + '='.repeat(80));
        console.log('Scan completed. Address critical and high severity issues first.');
        console.log('='.repeat(80));
    }
    
    run(targetPath) {
        console.log('🔍 Starting Agent 15 Orchestrator Security Scan...');
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
const scanner = new Agent15SecurityScanner();
const targetPath = process.argv[2] || '../app/a2aFiori/webapp/ext/agent15';
scanner.run(targetPath);