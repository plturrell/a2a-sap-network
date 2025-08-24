#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 6 (Quality Control Manager) Security Scanner
 * Specialized scanner for quality control vulnerabilities, quality metrics manipulation,
 * control process bypasses, approval workflow vulnerabilities, and audit trail tampering
 */
class Agent6SecurityScanner {
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
        
        // Quality Control Manager-specific vulnerability patterns
        this.qualityControlPatterns = {
            // Quality metrics manipulation
            QUALITY_METRICS_MANIPULATION: {
                patterns: [
                    /qualityScore\s*=\s*[0-9]{2,3}/gi,
                    /\.setQualityScore\s*\([^)]*[0-9]{2,3}/gi,
                    /overallQuality\s*=.*Math\.(max|min)/gi,
                    /\.manipulateQuality\s*\(/gi,
                    /qualityOverride\s*=\s*true/gi,
                    /\.forceQualityScore\s*\(/gi,
                    /bypassQualityCheck\s*=\s*true/gi,
                    /qualityThreshold\s*=\s*0/gi,
                    /minQualityScore\s*=\s*0/gi,
                    /\.alterQualityMetrics\s*\(/gi,
                    /fakeQualityData\s*=\s*true/gi,
                    /manipulatedScore\s*=/gi,
                    /\.overrideQualityGate\s*\(/gi,
                    /qualityValidation\s*=\s*false/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'QUALITY_METRICS_MANIPULATION',
                message: 'Quality metrics manipulation vulnerability detected',
                impact: 'Could allow unauthorized manipulation of quality scores, leading to acceptance of poor quality products or services'
            },
            
            // Control process bypass
            CONTROL_PROCESS_BYPASS: {
                patterns: [
                    /skipQualityControl\s*=\s*true/gi,
                    /\.bypassControlProcess\s*\(/gi,
                    /controlGate\s*=\s*false/gi,
                    /\.overrideControl\s*\(/gi,
                    /bypassApproval\s*=\s*true/gi,
                    /skipInspection\s*=\s*true/gi,
                    /\.forceApproval\s*\(/gi,
                    /controlValidation\s*=\s*false/gi,
                    /\.disableControls\s*\(/gi,
                    /emergencyBypass\s*=\s*true/gi,
                    /\.skipQualityGate\s*\(/gi,
                    /controlProcessDisabled\s*=\s*true/gi,
                    /\.overrideQualityControl\s*\(/gi,
                    /bypassStandardProcess\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'CONTROL_PROCESS_BYPASS',
                message: 'Control process bypass vulnerability',
                impact: 'Could allow circumvention of essential quality control processes, compromising product/service quality assurance'
            },
            
            // Approval workflow vulnerabilities
            APPROVAL_WORKFLOW_BYPASS: {
                patterns: [
                    /approvalStatus\s*=\s*['"]APPROVED['"]/gi,
                    /\.autoApprove\s*\(/gi,
                    /bypassApprovalWorkflow\s*=\s*true/gi,
                    /\.forceApproval\s*\([^)]*user/gi,
                    /approvalRequired\s*=\s*false/gi,
                    /\.skipApprovalStep\s*\(/gi,
                    /manualApproval\s*=\s*false/gi,
                    /\.overrideApprovalProcess\s*\(/gi,
                    /approvalValidation\s*=\s*false/gi,
                    /\.manipulateApproval\s*\(/gi,
                    /fakeApprovalData\s*=\s*true/gi,
                    /\.directApprovalSet\s*\(/gi,
                    /unsafeApproval\s*=\s*true/gi,
                    /requireManualApproval\s*=\s*false/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'APPROVAL_WORKFLOW_BYPASS',
                message: 'Approval workflow bypass vulnerability',
                impact: 'Could allow bypassing approval workflows, enabling unauthorized approvals and compromising governance'
            },
            
            // Quality gate manipulation
            QUALITY_GATE_MANIPULATION: {
                patterns: [
                    /qualityGate\s*=.*user.*input/gi,
                    /\.setQualityGate\s*\([^)]*\$\{/gi,
                    /qualityGateConfig\s*=.*eval/gi,
                    /\.manipulateGate\s*\(/gi,
                    /qualityGateThreshold\s*=\s*0/gi,
                    /\.bypassQualityGate\s*\(/gi,
                    /gateValidation\s*=\s*false/gi,
                    /\.overrideGate\s*\([^)]*user/gi,
                    /qualityGateCriteria\s*=\s*\[\]/gi,
                    /\.disableQualityGate\s*\(/gi,
                    /gateSecurityCheck\s*=\s*false/gi,
                    /\.forceGatePass\s*\(/gi,
                    /qualityGateBypass\s*=\s*true/gi,
                    /unsafeGateConfiguration\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'QUALITY_GATE_MANIPULATION',
                message: 'Quality gate manipulation vulnerability',
                impact: 'Could allow manipulation of quality gates, compromising quality control checkpoints'
            },
            
            // Audit trail tampering
            AUDIT_TRAIL_TAMPERING: {
                patterns: [
                    /auditLog\s*=\s*\[\]/gi,
                    /\.deleteAuditEntry\s*\(/gi,
                    /auditTrail\s*=\s*null/gi,
                    /\.clearAuditLog\s*\(/gi,
                    /modifyAuditEntry\s*=\s*true/gi,
                    /\.tamperAudit\s*\(/gi,
                    /auditDisabled\s*=\s*true/gi,
                    /\.overwriteAudit\s*\(/gi,
                    /skipAuditLog\s*=\s*true/gi,
                    /\.manipulateAuditTrail\s*\(/gi,
                    /auditBypass\s*=\s*true/gi,
                    /\.eraseAuditHistory\s*\(/gi,
                    /fakeAuditEntry\s*=\s*true/gi,
                    /auditSecurityDisabled\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'AUDIT_TRAIL_TAMPERING',
                message: 'Audit trail tampering vulnerability',
                impact: 'Could allow tampering with audit trails, compromising compliance, accountability, and forensic capabilities'
            },
            
            // Quality standards bypasses
            QUALITY_STANDARDS_BYPASS: {
                patterns: [
                    /qualityStandards\s*=\s*\[\]/gi,
                    /\.bypassStandards\s*\(/gi,
                    /standardsCompliance\s*=\s*false/gi,
                    /\.overrideStandards\s*\(/gi,
                    /complianceCheck\s*=\s*false/gi,
                    /\.skipComplianceValidation\s*\(/gi,
                    /standardsValidation\s*=\s*false/gi,
                    /\.disableStandardsCheck\s*\(/gi,
                    /qualityComplianceRequired\s*=\s*false/gi,
                    /\.forceStandardsPass\s*\(/gi,
                    /bypassQualityStandards\s*=\s*true/gi,
                    /\.manipulateStandards\s*\(/gi,
                    /unsafeStandardsMode\s*=\s*true/gi,
                    /standardsSecurityDisabled\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'QUALITY_STANDARDS_BYPASS',
                message: 'Quality standards bypass vulnerability',
                impact: 'Could allow bypassing quality standards compliance, compromising regulatory compliance and quality assurance'
            },
            
            // Inspection process vulnerabilities
            INSPECTION_PROCESS_BYPASS: {
                patterns: [
                    /inspectionRequired\s*=\s*false/gi,
                    /\.skipInspection\s*\(/gi,
                    /bypassInspectionProcess\s*=\s*true/gi,
                    /\.overrideInspection\s*\(/gi,
                    /inspectionValidation\s*=\s*false/gi,
                    /\.forceInspectionPass\s*\(/gi,
                    /inspectionThreshold\s*=\s*0/gi,
                    /\.manipulateInspection\s*\(/gi,
                    /autoInspectionPass\s*=\s*true/gi,
                    /\.disableInspectionControls\s*\(/gi,
                    /inspectionBypass\s*=\s*true/gi,
                    /\.alterInspectionResults\s*\(/gi,
                    /fakeInspectionData\s*=\s*true/gi,
                    /unsafeInspectionMode\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'INSPECTION_PROCESS_BYPASS',
                message: 'Inspection process bypass vulnerability',
                impact: 'Could allow bypassing inspection processes, compromising quality verification and defect detection'
            }
        };
        
        // OWASP vulnerability patterns
        this.owaspPatterns = {
            // XSS vulnerabilities
            XSS_INJECTION: {
                patterns: [
                    /innerHTML\s*=\s*[^'"`]+/gi,
                    /document\.write\s*\(/gi,
                    /\.html\s*\([^)]*\$\{/gi,
                    /dangerouslySetInnerHTML/gi,
                    /\.setText\s*\([^)]*\+/gi,
                    /MessageToast\.show\s*\([^)]*\+/gi,
                    /MessageBox\.\w+\s*\([^)]*\+/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'XSS_INJECTION',
                message: 'Cross-site scripting (XSS) vulnerability',
                impact: 'Could allow execution of malicious scripts in quality control interfaces'
            },
            
            // CSRF vulnerabilities
            CSRF_VULNERABILITY: {
                patterns: [
                    /\$\.ajax\s*\(\s*\{[^}]*type\s*:\s*["']POST["'][^}]*(?!X-CSRF-Token)/gi,
                    /\.post\s*\([^)]*(?!.*X-CSRF-Token)/gi,
                    /\.put\s*\([^)]*(?!.*X-CSRF-Token)/gi,
                    /\.delete\s*\([^)]*(?!.*X-CSRF-Token)/gi,
                    /fetch\s*\([^,]+,\s*\{[^}]*method\s*:\s*["'](POST|PUT|DELETE)["'][^}]*(?!X-CSRF-Token)/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'CSRF_VULNERABILITY',
                message: 'Cross-Site Request Forgery (CSRF) vulnerability',
                impact: 'Could allow unauthorized state-changing operations in quality control system'
            },
            
            // Input validation vulnerabilities
            INPUT_VALIDATION: {
                patterns: [
                    /eval\s*\(.*\+/gi,
                    /new\s+Function\s*\(.*\+/gi,
                    /setTimeout\s*\([^)]*\+/gi,
                    /setInterval\s*\([^)]*\+/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'INPUT_VALIDATION',
                message: 'Input validation vulnerability',
                impact: 'Could allow code injection and arbitrary code execution in quality control context'
            },
            
            // Insecure connection patterns
            INSECURE_CONNECTION: {
                patterns: [
                    /http:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /ws:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /protocol\s*:\s*["']http["']/gi,
                    /\.connect\s*\([^)]*["']http:/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'INSECURE_CONNECTION',
                message: 'Insecure connection vulnerability',
                impact: 'Could allow interception of quality control data during transmission'
            }
        };
        
        // SAP Fiori compliance patterns
        this.fioriCompliancePatterns = {
            // i18n compliance
            I18N_MISSING: {
                patterns: [
                    /MessageToast\.show\s*\(\s*["'][^"']+["']\s*\)/gi,
                    /MessageBox\.\w+\s*\(\s*["'][^"']+["']/gi,
                    /setText\s*\(\s*["'][^"']+["']\s*\)/gi,
                    /headerText\s*:\s*["'][^"']+["']/gi,
                    /title\s*:\s*["'][^"']+["']/gi,
                    /label\s*:\s*["'][^"']+["']/gi
                ],
                severity: this.severityLevels.LOW,
                category: 'I18N_MISSING',
                message: 'Missing internationalization (i18n)',
                impact: 'Reduces internationalization support for quality control interfaces'
            },
            
            // Security headers
            SECURITY_HEADERS_MISSING: {
                patterns: [
                    /(?!.*Content-Security-Policy).*manifest\.json/gi,
                    /(?!.*X-Frame-Options).*manifest\.json/gi,
                    /(?!.*X-Content-Type-Options).*manifest\.json/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'SECURITY_HEADERS_MISSING',
                message: 'Missing security headers',
                impact: 'Reduces security posture of quality control application'
            },
            
            // Accessibility issues
            ACCESSIBILITY_MISSING: {
                patterns: [
                    /VizFrame\s*\([^)]*\)/gi,
                    /MicroChart\s*\([^)]*\)/gi,
                    /setBusy\s*\(\s*true\s*\)/gi,
                    /Table\s*\([^)]*\)/gi,
                    /Button\s*\([^)]*\)/gi
                ],
                severity: this.severityLevels.LOW,
                category: 'ACCESSIBILITY_MISSING',
                message: 'Missing accessibility features',
                impact: 'Reduces accessibility for quality control interfaces'
            }
        };
    }
    
    /**
     * Main scan method
     */
    async scan(targetPath) {
        console.log('🔍 Starting Agent 6 (Quality Control Manager) Security Scan');
        console.log(`📁 Target: ${path.resolve(targetPath)}\n`);
        
        if (!fs.existsSync(targetPath)) {
            console.error(`❌ Target path does not exist: ${targetPath}`);
            process.exit(1);
        }
        
        await this.scanDirectory(targetPath);
        this.generateReport();
    }
    
    /**
     * Recursively scan directory
     */
    async scanDirectory(dirPath) {
        const items = fs.readdirSync(dirPath);
        
        for (const item of items) {
            const fullPath = path.join(dirPath, item);
            const stat = fs.statSync(fullPath);
            
            if (stat.isDirectory()) {
                // Skip common non-source directories
                if (!['node_modules', '.git', 'dist', 'build', 'coverage'].includes(item)) {
                    await this.scanDirectory(fullPath);
                }
            } else if (stat.isFile()) {
                // Scan relevant file types
                const ext = path.extname(item).toLowerCase();
                if (['.js', '.ts', '.json', '.xml', '.html', '.htm', '.properties'].includes(ext)) {
                    this.scanFile(fullPath);
                }
            }
        }
    }
    
    /**
     * Scan individual file
     */
    scanFile(filePath) {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            this.filesScanned++;
            
            // Scan for Quality Control-specific vulnerabilities
            this.scanPatterns(content, filePath, this.qualityControlPatterns);
            
            // Scan for OWASP vulnerabilities
            this.scanPatterns(content, filePath, this.owaspPatterns);
            
            // Scan for SAP Fiori compliance
            this.scanPatterns(content, filePath, this.fioriCompliancePatterns);
            
            // Additional Quality Control-specific checks
            this.scanForQualityControlSpecificIssues(content, filePath);
            
        } catch (error) {
            console.error(`⚠️  Error scanning file ${filePath}: ${error.message}`);
        }
    }
    
    /**
     * Scan for patterns in content
     */
    scanPatterns(content, filePath, patterns) {
        for (const [patternName, config] of Object.entries(patterns)) {
            for (const pattern of config.patterns) {
                const matches = content.match(pattern);
                if (matches) {
                    const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                    const lineNumber = lines.length;
                    
                    // Skip false positives
                    const code = lines[lineNumber - 1]?.trim() || '';
                    if (this.isFalsePositive(code, patternName, filePath)) {
                        continue;
                    }
                    
                    this.vulnerabilities.push({
                        file: filePath,
                        line: lineNumber,
                        severity: config.severity,
                        category: config.category,
                        message: config.message,
                        impact: config.impact,
                        pattern: pattern.toString(),
                        match: matches[0],
                        isPositive: config.isPositive || false,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        }
    }
    
    /**
     * Check for false positives
     */
    isFalsePositive(code, patternName, filePath) {
        // Skip comments and documentation
        if (code.includes('//') || code.includes('/*') || code.includes('*')) {
            return true;
        }
        
        // Skip console.log and debug statements
        if (code.includes('console.log') || code.includes('console.error')) {
            return true;
        }
        
        // Skip legitimate security functions
        if (filePath.includes('SecurityUtils.js') || filePath.includes('security')) {
            if (code.includes('sanitized') || code.includes('validation') || 
                code.includes('_sanitize') || code.includes('_validate')) {
                return true;
            }
            // Skip all INPUT_VALIDATION patterns in SecurityUtils
            if (patternName === 'INPUT_VALIDATION') {
                return true;
            }
        }
        
        // Skip already sanitized content for XSS patterns
        if (patternName === 'XSS_INJECTION') {
            if (code.includes('encodeXML') || code.includes('sanitizeHTML') || 
                code.includes('escapeRegExp') || code.includes('_sanitizeInput')) {
                return true;
            }
        }
        
        // Skip CSRF patterns that already have token validation
        if (patternName === 'CSRF_VULNERABILITY') {
            const surroundingArea = code.substring(Math.max(0, code.indexOf('ajax') - 100));
            if (surroundingArea.includes('X-CSRF-Token') || surroundingArea.includes('_getCsrfToken')) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Scan for Quality Control-specific security issues
     */
    scanForQualityControlSpecificIssues(content, filePath) {
        // Check for hardcoded quality thresholds
        const hardcodedThresholds = [
            /qualityThreshold\s*[:=]\s*[0-9]+/gi,
            /minQualityScore\s*[:=]\s*[0-9]+/gi,
            /passThreshold\s*[:=]\s*[0-9]+/gi,
            /approvalThreshold\s*[:=]\s*[0-9]+/gi
        ];
        
        hardcodedThresholds.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                
                // Check if it's within a configuration function
                const contextStart = Math.max(0, content.indexOf(matches[0]) - 200);
                const contextEnd = Math.min(content.length, content.indexOf(matches[0]) + 200);
                const context = content.substring(contextStart, contextEnd);
                
                // Skip if it's in a configuration getter or default values
                if (context.includes('_getQualityThresholds') || 
                    context.includes('// Return secure defaults') ||
                    context.includes('getProperty("/qualityThresholds")')) {
                    return;
                }
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.MEDIUM,
                    category: 'HARDCODED_THRESHOLDS',
                    message: 'Hardcoded quality thresholds detected',
                    impact: 'Could make quality control system inflexible and harder to maintain',
                    pattern: pattern.toString(),
                    match: `${matches[0].substring(0, 50)  }...`,
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Check for unsafe routing decisions
        const unsafeRoutingPatterns = [
            /targetAgent\s*=.*user.*input/gi,
            /routingDecision\s*=.*eval/gi,
            /\.routeTo\s*\([^)]*\$\{/gi,
            /agentSelection\s*=.*Function/gi
        ];
        
        unsafeRoutingPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.HIGH,
                    category: 'UNSAFE_ROUTING_DECISION',
                    message: 'Unsafe routing decision implementation',
                    impact: 'Could allow manipulation of task routing leading to unauthorized access or workflow bypass',
                    pattern: pattern.toString(),
                    match: matches[0],
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Check for trust verification bypasses
        const trustBypassPatterns = [
            /trustVerification\s*=\s*false/gi,
            /bypassTrustCheck\s*=\s*true/gi,
            /\.skipTrustVerification\s*\(/gi,
            /trustRequired\s*=\s*false/gi,
            /blockchainVerification\s*=\s*false/gi
        ];
        
        trustBypassPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.HIGH,
                    category: 'TRUST_VERIFICATION_BYPASS',
                    message: 'Trust verification bypass detected',
                    impact: 'Could allow bypassing trust verification processes, compromising system integrity',
                    pattern: pattern.toString(),
                    match: matches[0],
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Check for batch operation security issues
        const batchSecurityPatterns = [
            /batchSize\s*>\s*[0-9]{3,}/gi,
            /maxBatchSize\s*=\s*-1/gi,
            /\.processBatch\s*\([^)]*user/gi,
            /batchValidation\s*=\s*false/gi,
            /parallelProcessing\s*=\s*true.*user/gi
        ];
        
        batchSecurityPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.MEDIUM,
                    category: 'BATCH_OPERATION_SECURITY',
                    message: 'Batch operation security issue',
                    impact: 'Could lead to resource exhaustion or unauthorized batch processing',
                    pattern: pattern.toString(),
                    match: matches[0],
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });
    }
    
    /**
     * Generate comprehensive security report
     */
    generateReport() {
        const scanDuration = Date.now() - this.scanStartTime;
        const criticalCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.CRITICAL).length;
        const highCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.HIGH).length;
        const mediumCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.MEDIUM).length;
        const lowCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.LOW).length;
        const warningCount = this.vulnerabilities.filter(v => v.severity === this.severityLevels.WARNING).length;
        
        console.log(`\n${  '='.repeat(80)}`);
        console.log('🛡️  AGENT 6 (QUALITY CONTROL MANAGER) SECURITY SCAN REPORT');
        console.log('='.repeat(80));
        
        console.log('📊 SCAN SUMMARY:');
        console.log(`   📂 Files Scanned: ${this.filesScanned}`);
        console.log(`   ⏱️  Scan Duration: ${(scanDuration / 1000).toFixed(2)}s`);
        console.log(`   🚨 Total Issues: ${this.vulnerabilities.length}`);
        console.log(`   🔴 Critical: ${criticalCount}`);
        console.log(`   🟠 High: ${highCount}`);
        console.log(`   🟡 Medium: ${mediumCount}`);
        console.log(`   🟢 Low: ${lowCount}`);
        console.log(`   ⚪ Warning: ${warningCount}`);
        
        if (this.vulnerabilities.length > 0) {
            console.log('\n📋 VULNERABILITIES BY CATEGORY:');
            const byCategory = {};
            this.vulnerabilities.forEach(vuln => {
                byCategory[vuln.category] = (byCategory[vuln.category] || 0) + 1;
            });
            
            Object.entries(byCategory)
                .sort(([,a], [,b]) => b - a)
                .forEach(([category, count]) => {
                    console.log(`   • ${category}: ${count}`);
                });
            
            console.log('\n🔍 DETAILED FINDINGS:');
            console.log('-'.repeat(80));
            
            // Sort by severity
            const severityOrder = { 'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'WARNING': 4 };
            this.vulnerabilities.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
            
            this.vulnerabilities.forEach((vuln, index) => {
                const icon = this.getSeverityIcon(vuln.severity);
                console.log(`\n${icon} [${vuln.severity}] ${vuln.category}`);
                console.log(`   📁 File: ${vuln.file}:${vuln.line}`);
                console.log(`   📝 Message: ${vuln.message}`);
                console.log(`   💥 Impact: ${vuln.impact}`);
                console.log(`   🎯 Match: ${vuln.match}`);
                if (index < this.vulnerabilities.length - 1) {
                    console.log('-'.repeat(80));
                }
            });
        }
        
        console.log('\n🏥 QUALITY CONTROL SECURITY RECOMMENDATIONS:');
        console.log('   1. 🔒 Implement comprehensive input validation for quality metrics and scores');
        console.log('   2. 🛡️  Add authorization checks for all quality control operations');
        console.log('   3. 🔐 Implement integrity checks for quality assessment data');
        console.log('   4. 🚫 Use CSRF protection for all state-changing quality operations');
        console.log('   5. 🌐 Ensure HTTPS for all quality control communications');
        console.log('   6. 📊 Implement audit logging for all quality control decisions');
        console.log('   7. 🔍 Add validation for routing decisions and agent selections');
        console.log('   8. 🏭 Secure batch operation processing with size limits');
        console.log('   9. 📋 Implement trust verification for critical quality operations');
        console.log('   10. 🧪 Add workflow integrity checks and approval validations');
        console.log('   11. 🔐 Protect quality thresholds and standards configuration');
        console.log('   12. 🛡️  Implement quality gate bypass detection and prevention');
        
        this.saveReport();
        
        console.log('\n✅ Scan completed successfully!');
        console.log('📄 Report saved to: agent6-security-report.json');
        
        if (criticalCount > 0 || highCount > 0) {
            console.log(`\n⚠️  ${criticalCount + highCount} critical/high severity issues found!`);
            console.log('🔧 Please address these issues before deploying to production.');
            process.exit(1);
        }
    }
    
    /**
     * Get severity icon
     */
    getSeverityIcon(severity) {
        const icons = {
            'CRITICAL': '🚨',
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢',
            'WARNING': '⚪'
        };
        return icons[severity] || '❓';
    }
    
    /**
     * Save report to JSON file
     */
    saveReport() {
        const report = {
            scanMetadata: {
                agent: 'Agent 6 - Quality Control Manager',
                scanType: 'Quality Control Security Scan',
                timestamp: new Date().toISOString(),
                duration: Date.now() - this.scanStartTime,
                filesScanned: this.filesScanned,
                totalVulnerabilities: this.vulnerabilities.length
            },
            summary: {
                critical: this.vulnerabilities.filter(v => v.severity === this.severityLevels.CRITICAL).length,
                high: this.vulnerabilities.filter(v => v.severity === this.severityLevels.HIGH).length,
                medium: this.vulnerabilities.filter(v => v.severity === this.severityLevels.MEDIUM).length,
                low: this.vulnerabilities.filter(v => v.severity === this.severityLevels.LOW).length,
                warning: this.vulnerabilities.filter(v => v.severity === this.severityLevels.WARNING).length
            },
            vulnerabilities: this.vulnerabilities,
            recommendations: [
                'Implement comprehensive input validation for quality metrics and control parameters',
                'Add robust authorization checks for all quality control and routing operations',
                'Implement integrity validation for quality assessment and approval data',
                'Use CSRF protection for all state-changing quality control operations',
                'Ensure HTTPS/WSS for all quality control service communications',
                'Implement comprehensive audit logging for quality decisions and control processes',
                'Add validation and sanitization for routing decisions and agent selections',
                'Secure batch operation processing with appropriate size limits and validation',
                'Implement trust verification mechanisms for critical quality operations',
                'Add workflow integrity checks and multi-level approval validations',
                'Protect quality thresholds and standards from unauthorized modification',
                'Implement quality gate bypass detection and prevention mechanisms',
                'Add compliance monitoring for quality standards and regulatory requirements',
                'Implement secure inspection process workflows with proper controls',
                'Add tamper detection for audit trails and quality metrics',
                'Implement role-based access control for quality control operations'
            ]
        };
        
        fs.writeFileSync('agent6-security-report.json', JSON.stringify(report, null, 2));
    }
}

// Main execution
if (require.main === module) {
    const scanner = new Agent6SecurityScanner();
    const targetDir = process.argv[2] || '../app/a2aFiori/webapp/ext/agent6';
    scanner.scan(targetDir);
}

module.exports = Agent6SecurityScanner;