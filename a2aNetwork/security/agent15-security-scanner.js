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
                    /eval\s*\(/gi,
                    /new\s+Function\s*\(/gi,
                    /setTimeout\s*\([^,]+,\s*0\s*\)/gi,
                    /setInterval\s*\([^,]+,/gi,
                    /\.execute\s*\([^)]*\$\{/gi,
                    /\.runWorkflow\s*\([^)]*\+/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'ORCHESTRATION_WORKFLOW_INJECTION',
                message: 'Potential workflow injection vulnerability',
                impact: 'Could allow malicious workflow execution and system compromise'
            },
            
            // Agent coordination vulnerabilities
            AGENT_HIJACKING: {
                patterns: [
                    /agentId\s*=\s*[^'"`]+/gi,
                    /\.assignAgent\s*\([^)]*\$\{/gi,
                    /\.selectAgent\s*\([^)]*user/gi,
                    /\.routeToAgent\s*\([^)]*\+/gi
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
        
        xssPatterns.forEach(({ pattern, type, message }) => {
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
                if (!surroundingCode.includes('X-CSRF-Token') && !surroundingCode.includes('csrf')) {
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
                // Skip comments and strings that might be examples
                if (!code.includes('//') && !code.includes('example')) {
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
    
    checkOrchestratorVulnerabilities(content, filePath, lines) {
        Object.entries(this.orchestratorPatterns).forEach(([vulnType, config]) => {
            config.patterns.forEach(pattern => {
                const matches = content.matchAll(pattern);
                for (const match of matches) {
                    const lineNumber = this.getLineNumber(content, match.index);
                    this.addVulnerability({
                        type: config.category,
                        severity: config.severity,
                        file: filePath,
                        line: lineNumber,
                        code: lines[lineNumber - 1]?.trim() || '',
                        message: config.message,
                        impact: config.impact,
                        fix: this.getOrchestratorFix(vulnType)
                    });
                }
            });
        });
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
            
            i18nPatterns.forEach(({ pattern, message }) => {
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
                        impact: 'Reduces application security posture',
                        fix: `Add ${header} to manifest security configuration`
                    });
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
        const orchestratorIssues = this.vulnerabilities.filter(v => v.type.startsWith('ORCHESTRATION_'));
        if (orchestratorIssues.length > 0) {
            console.log(`\n🤖 ORCHESTRATOR-SPECIFIC SECURITY FINDINGS:`);
            const issueCounts = {};
            orchestratorIssues.forEach(issue => {
                issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
            });
            
            Object.entries(issueCounts).forEach(([type, count]) => {
                console.log(`   ${type.replace('ORCHESTRATION_', '')}: ${count} issues`);
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