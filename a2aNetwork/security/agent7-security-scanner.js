#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 7 (Agent Manager) Security Scanner
 * Specialized scanner for agent management security, authentication vulnerabilities,
 * authorization bypasses, and administrative operation security
 */
class Agent7SecurityScanner {
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
        
        // Agent Management-specific vulnerability patterns
        this.agentManagementPatterns = {
            // Missing authentication checks
            MISSING_AUTHENTICATION_CHECK: {
                patterns: [
                    /on[A-Z]\w*\s*:\s*function[^{]*\{(?![\s\S]*_hasRole|[\s\S]*_authorizeOperation)/gi,
                    /function\s+on[A-Z]\w*[^{]*\{(?![\s\S]*_hasRole|[\s\S]*_authorizeOperation)/gi,
                    /\.onBulkOperations[^{]*\{(?![\s\S]*_hasRole)/gi,
                    /\.onRegisterAgent[^{]*\{(?![\s\S]*_hasRole)/gi,
                    /\.onUpdateAgent[^{]*\{(?![\s\S]*_hasRole)/gi,
                    /\.onDeleteAgent[^{]*\{(?![\s\S]*_hasRole)/gi,
                    /\.onConfigureAgent[^{]*\{(?![\s\S]*_hasRole)/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'MISSING_AUTHENTICATION_CHECK',
                message: 'Management operation lacks authentication/authorization checks',
                impact: 'Could allow unauthorized access to critical agent management operations'
            },

            // Insecure API endpoints
            UNSECURED_API_ENDPOINT: {
                patterns: [
                    /jQuery\.ajax\s*\(\s*\{[^}]*url\s*:\s*['"]/gi,
                    /\$\.ajax\s*\(\s*\{[^}]*url\s*:\s*['"]/gi,
                    /\.ajax\s*\(\s*\{[^}]*(?!headers|X-CSRF-Token|secureAjaxRequest)/gi,
                    /url\s*:\s*['"][^'"]*agent7[^'"]*['"](?![^}]*X-CSRF-Token)/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'UNSECURED_API_ENDPOINT',
                message: 'API endpoint lacks CSRF protection or secure headers',
                impact: 'Could expose agent management APIs to CSRF attacks and unauthorized access'
            },

            // Agent registration vulnerabilities
            AGENT_REGISTRATION_BYPASS: {
                patterns: [
                    /agentRegistration\s*=\s*true/gi,
                    /\.allowRegistration\s*\(/gi,
                    /skipRegistrationValidation\s*=\s*true/gi,
                    /\.overrideRegistration\s*\([^)]*\$\{/gi,
                    /forceAgentRegistration\s*=.*user/gi,
                    /\.bypassRegistration\s*\(/gi,
                    /registrationSecurity\s*=\s*false/gi,
                    /allowPublicRegistration\s*=\s*true/gi,
                    /\.directAgentRegister\s*\(/gi,
                    /unsafeAgentRegistration\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'AGENT_REGISTRATION_BYPASS',
                message: 'Agent registration bypass vulnerability',
                impact: 'Could allow unauthorized registration of malicious agents'
            },

            // Management privilege escalation
            PRIVILEGE_ESCALATION: {
                patterns: [
                    /adminMode\s*=\s*true/gi,
                    /\.elevatePrivileges\s*\(/gi,
                    /bypassRoleCheck\s*=\s*true/gi,
                    /\.overridePermissions\s*\([^)]*\$\{/gi,
                    /forceAdminAccess\s*=.*user/gi,
                    /\.grantAdmin\s*\(/gi,
                    /roleValidation\s*=\s*false/gi,
                    /allowPrivilegeEscalation\s*=\s*true/gi,
                    /\.directRoleAssign\s*\(/gi,
                    /unsafePrivilegeGrant\s*=\s*true/gi,
                    /skipAuthorizationCheck\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'PRIVILEGE_ESCALATION',
                message: 'Privilege escalation vulnerability',
                impact: 'Could allow users to gain unauthorized administrative privileges'
            },

            // Agent coordination vulnerabilities
            COORDINATION_SECURITY: {
                patterns: [
                    /coordinationSecurity\s*=\s*false/gi,
                    /\.bypassCoordination\s*\(/gi,
                    /skipCoordinationValidation\s*=\s*true/gi,
                    /\.overrideCoordination\s*\([^)]*\$\{/gi,
                    /forceCoordination\s*=.*user/gi,
                    /\.manipulateCoordination\s*\(/gi,
                    /coordinationValidation\s*=\s*false/gi,
                    /allowCoordinationBypass\s*=\s*true/gi,
                    /\.directCoordinationSet\s*\(/gi,
                    /unsafeCoordination\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'COORDINATION_SECURITY',
                message: 'Agent coordination security vulnerability',
                impact: 'Could allow manipulation of agent coordination and orchestration'
            },

            // Bulk operation vulnerabilities
            BULK_OPERATION_ABUSE: {
                patterns: [
                    /bulkOperationLimit\s*=\s*-1/gi,
                    /\.unlimitedBulkOps\s*\(/gi,
                    /maxBulkOperations\s*=\s*\d{4,}/gi,
                    /\.bypassBulkLimits\s*\([^)]*\$\{/gi,
                    /forceBulkOperation\s*=.*user/gi,
                    /\.manipulateBulkOps\s*\(/gi,
                    /bulkValidation\s*=\s*false/gi,
                    /allowUnlimitedBulk\s*=\s*true/gi,
                    /\.directBulkExecute\s*\(/gi,
                    /unsafeBulkOperation\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'BULK_OPERATION_ABUSE',
                message: 'Bulk operation abuse vulnerability',
                impact: 'Could allow resource exhaustion or system overload through bulk operations'
            },

            // Agent health manipulation
            HEALTH_MANIPULATION: {
                patterns: [
                    /healthStatus\s*=.*fake/gi,
                    /\.manipulateHealth\s*\(/gi,
                    /fakeHealthReport\s*=\s*true/gi,
                    /\.alterHealth\s*\([^)]*user/gi,
                    /healthManipulation\s*=\s*true/gi,
                    /\.corruptHealth\s*\(/gi,
                    /unsafeHealthAccess\s*=\s*true/gi,
                    /\.directHealthModify\s*\(/gi,
                    /healthIntegrity\s*=\s*false/gi,
                    /allowHealthEdit\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'HEALTH_MANIPULATION',
                message: 'Agent health data manipulation vulnerability',
                impact: 'Could allow falsification of agent health status leading to system instability'
            },

            // Performance metrics tampering
            METRICS_TAMPERING: {
                patterns: [
                    /metricsData\s*=.*splice/gi,
                    /\.manipulateMetrics\s*\(/gi,
                    /fakeMetricsData\s*=\s*true/gi,
                    /\.alterMetrics\s*\([^)]*user/gi,
                    /metricsManipulation\s*=\s*true/gi,
                    /\.corruptMetrics\s*\(/gi,
                    /unsafeMetricsAccess\s*=\s*true/gi,
                    /\.directMetricsModify\s*\(/gi,
                    /metricsIntegrity\s*=\s*false/gi,
                    /allowMetricsEdit\s*=\s*true/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'METRICS_TAMPERING',
                message: 'Performance metrics tampering vulnerability',
                impact: 'Could allow falsification of performance data leading to incorrect decisions'
            }
        };
        
        // Standard OWASP Top 10 patterns (adapted for Agent Management context)
        this.owaspPatterns = {
            // XSS in management interfaces
            XSS_VULNERABILITY: {
                patterns: [
                    /<script[\s\S]*?>[\s\S]*?<\/script>/gi,
                    /javascript:\s*[^\/]/gi,
                    /\bon(load|click|error|focus|blur|submit|change|keyup|keydown|mouseover|mouseout)\s*=/gi,
                    /\.innerHTML\s*=\s*[^'"]/gi,
                    /document\.write\s*\(/gi,
                    /setTimeout\s*\([^)]*['"].*[<>]/gi,
                    /setInterval\s*\([^)]*['"].*[<>]/gi,
                    /\.outerHTML\s*=/gi,
                    /\.insertAdjacentHTML/gi,
                    /agentDescription.*<script/gi,
                    /managementContent.*javascript:/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'XSS_VULNERABILITY',
                message: 'Cross-Site Scripting (XSS) vulnerability in management interface',
                impact: 'Could allow execution of malicious scripts in agent management interfaces'
            },

            // SQL Injection in agent queries
            SQL_INJECTION: {
                patterns: [
                    /SELECT\s+.*\+.*user/gi,
                    /INSERT\s+INTO.*\+.*input/gi,
                    /UPDATE\s+.*SET.*\+.*user/gi,
                    /DELETE\s+FROM.*\+.*input/gi,
                    /UNION\s+SELECT/gi,
                    /OR\s+1\s*=\s*1/gi,
                    /DROP\s+TABLE/gi,
                    /;\s*--/gi,
                    /agentQuery\s*\+\s*user/gi,
                    /managementQuery\s*\+\s*input/gi,
                    /coordQuery.*\+/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'SQL_INJECTION',
                message: 'SQL Injection vulnerability in agent management queries',
                impact: 'Could allow unauthorized database access, data theft, or data manipulation'
            },

            // CSRF in management operations
            CSRF_VULNERABILITY: {
                patterns: [
                    /method\s*:\s*['"]POST['"](?![^}]*csrf)/gi,
                    /method\s*:\s*['"]PUT['"](?![^}]*csrf)/gi,
                    /method\s*:\s*['"]DELETE['"](?![^}]*csrf)/gi,
                    /\.post\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /\.put\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /\.delete\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /manageAgent.*POST(?![^}]*token)/gi,
                    /registerAgent.*POST(?![^}]*csrf)/gi,
                    /bulkOperation.*POST(?![^}]*token)/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'CSRF_VULNERABILITY',
                message: 'Cross-Site Request Forgery (CSRF) vulnerability in management operations',
                impact: 'Could allow unauthorized execution of agent management operations'
            },

            // Insecure connections
            INSECURE_CONNECTION: {
                patterns: [
                    /http:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /ws:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /\.protocol\s*=\s*['"]http:['"](?![^}]*localhost)/gi,
                    /url\s*:\s*['"]http:\/\/(?!localhost)/gi,
                    /agentServiceUrl.*http:\/\//gi,
                    /managementServiceUrl.*http:\/\//gi,
                    /coordinationUrl.*ws:\/\//gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'INSECURE_CONNECTION',
                message: 'Insecure HTTP/WebSocket connection in agent management',
                impact: 'Could expose agent management data to man-in-the-middle attacks'
            }
        };
        
        // SAP Fiori compliance patterns
        this.fioriCompliancePatterns = {
            // i18n compliance for management interfaces
            I18N_COMPLIANCE: {
                patterns: [
                    /getText\s*\(\s*['"][^'"]*['"]\s*\)/gi,
                    /\.getResourceBundle\(\)\.getText/gi,
                    /i18n\s*>\s*[^{]*\{/gi,
                    /this\._oResourceBundle\.getText/gi
                ],
                severity: this.severityLevels.LOW,
                category: 'I18N_COMPLIANCE',
                message: 'Internationalization compliance check for management texts',
                impact: 'Management interface texts should be externalized for internationalization',
                isPositive: true
            },

            // Security headers in management responses
            SECURITY_HEADERS: {
                patterns: [
                    /X-Frame-Options/gi,
                    /X-Content-Type-Options/gi,
                    /X-XSS-Protection/gi,
                    /Content-Security-Policy/gi,
                    /Strict-Transport-Security/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'SECURITY_HEADERS',
                message: 'Security headers implementation in management responses',
                impact: 'Missing security headers could expose management interface to various attacks',
                isPositive: true
            }
        };
    }
    
    /**
     * Main scan function
     */
    async scan(targetDirectory) {
        console.log(`🔍 Starting Agent 7 (Agent Manager) Security Scan...`);
        console.log(`📂 Target Directory: ${targetDirectory}`);
        console.log(`⏰ Scan Started: ${new Date().toISOString()}\n`);
        
        if (!fs.existsSync(targetDirectory)) {
            console.error(`❌ Target directory does not exist: ${targetDirectory}`);
            return;
        }
        
        await this.scanDirectory(targetDirectory);
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
            
            // Scan for Agent Management-specific vulnerabilities
            this.scanPatterns(content, filePath, this.agentManagementPatterns);
            
            // Scan for OWASP vulnerabilities
            this.scanPatterns(content, filePath, this.owaspPatterns);
            
            // Scan for SAP Fiori compliance
            this.scanPatterns(content, filePath, this.fioriCompliancePatterns);
            
            // Additional management-specific checks
            this.scanForManagementSpecificIssues(content, filePath);
            
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
                    const matchedText = matches[0];
                    const lineContext = lines[lineNumber - 1] || '';
                    
                    // Skip false positives
                    if (this.isFalsePositive(matchedText, lineContext, patternName, filePath)) {
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
                        match: matchedText,
                        isPositive: config.isPositive || false,
                        timestamp: new Date().toISOString()
                    });
                }
            }
        }
    }
    
    /**
     * Check if a pattern match is a false positive
     */
    isFalsePositive(matchedText, lineContext, patternName, filePath) {
        // General false positive checks for SecurityUtils files
        if (filePath.includes('SecurityUtils.js')) {
            // SecurityUtils contains security patterns that may trigger false positives
            return lineContext.includes('pattern') || lineContext.includes('message') || 
                   lineContext.includes('dangerousPatterns') || lineContext.includes('test') ||
                   lineContext.includes('validateAgent') || lineContext.includes('sanitize') ||
                   lineContext.includes('validation') || lineContext.includes('_validate');
        }
        
        // Specific false positives by pattern type
        switch(patternName) {
            case 'MISSING_AUTHENTICATION_CHECK':
                // Skip if already has security checks (our fixes)
                if (lineContext.includes('_hasRole') || lineContext.includes('_authorizeOperation') ||
                    lineContext.includes('SECURITY FIX') || lineContext.includes('authentication check')) {
                    return true;
                }
                break;
                
            case 'UNSECURED_API_ENDPOINT':
                // Skip if already uses secure methods
                if (lineContext.includes('secureAjaxRequest') || lineContext.includes('_securityUtils') ||
                    lineContext.includes('X-CSRF-Token') || lineContext.includes('SECURITY FIX')) {
                    return true;
                }
                break;
                
            case 'XSS_VULNERABILITY':
                // Skip normal XML attributes and property names
                if (matchedText.includes('ontentWidth=') || matchedText.includes('onAPI =') ||
                    matchedText.includes('onTasksTitle=') || matchedText.includes('ontrol =')) {
                    return true;
                }
                // Skip legitimate WebSocket/EventSource event handlers
                if ((matchedText.includes('onerror =') || matchedText.includes('onclick =') || 
                     matchedText.includes('onload =') || matchedText.includes('onblur =')) && 
                    (lineContext.includes('_ws.') || lineContext.includes('socket.') ||
                     lineContext.includes('WebSocket') || lineContext.includes('= function()') ||
                     lineContext.includes('EventSource'))) {
                    return true;
                }
                // Skip if it's in comments or property definitions
                if (lineContext.includes('//') || lineContext.includes('*') || 
                    lineContext.includes('i18n>') || lineContext.includes('title="{i18n>')) {
                    return true;
                }
                break;
                
            case 'I18N_COMPLIANCE':
                // This is actually a positive pattern, so don't skip
                return false;
                
            case 'INSECURE_CONNECTION':
                // Skip if it's just a placeholder or comment
                if (lineContext.includes('placeholder=') || lineContext.includes('//') ||
                    lineContext.includes('example') || lineContext.includes('sample')) {
                    return true;
                }
                break;
        }
        
        // Skip comments and documentation
        if (lineContext.includes('//') || lineContext.includes('/*') || 
            lineContext.includes('*') || lineContext.includes('try {')) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Scan for management-specific security issues
     */
    scanForManagementSpecificIssues(content, filePath) {
        // Check for hardcoded management credentials
        const credentialPatterns = [
            /adminPassword\s*[:=]\s*['"][^'"]*['"]/gi,
            /managerApiKey\s*[:=]\s*['"][^'"]*['"]/gi,
            /agentToken\s*[:=]\s*['"][^'"]*['"]/gi,
            /coordinationSecret\s*[:=]\s*['"][^'"]*['"]/gi,
            /managementCredentials\s*[:=]/gi,
            /adminPassword\s*[:=]/gi
        ];
        
        credentialPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                const lineContext = lines[lineNumber - 1] || '';
                const matchedText = matches[0];
                
                // Skip false positives in SecurityUtils
                if (filePath.includes('SecurityUtils.js') && 
                    (lineContext.includes('TOKEN=') || lineContext.includes('cookie.startsWith') ||
                     lineContext.includes('XSRF-TOKEN') || lineContext.includes('substring'))) {
                    return;
                }
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.HIGH,
                    category: 'HARDCODED_CREDENTIALS',
                    message: 'Hardcoded credentials in agent management files',
                    impact: 'Could expose management credentials leading to unauthorized access',
                    pattern: pattern.toString(),
                    match: matchedText.substring(0, 50) + '...',
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });
        
        // Check for unvalidated EventSource URLs
        const eventSourcePatterns = [
            /new\s+EventSource\s*\(\s*[^)]*\)/gi,
            /EventSource\s*\(\s*[^)]*user/gi,
            /\.createEventSource\s*\(/gi
        ];
        
        eventSourcePatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;
                
                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.HIGH,
                    category: 'UNVALIDATED_EVENTSOURCE_URL',
                    message: 'EventSource URL not validated for security',
                    impact: 'Could allow SSRF attacks or connection to malicious endpoints',
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
        
        console.log('\n' + '='.repeat(80));
        console.log('🛡️  AGENT 7 (AGENT MANAGER) SECURITY SCAN REPORT');
        console.log('='.repeat(80));
        
        console.log(`📊 SCAN SUMMARY:`);
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
        
        console.log('\n🏥 AGENT MANAGER SECURITY RECOMMENDATIONS:');
        console.log('   1. 🔒 Implement authentication checks for all management operations');
        console.log('   2. 🛡️  Add authorization validation for administrative functions');
        console.log('   3. 🔐 Use secure AJAX requests with CSRF protection');
        console.log('   4. 🚫 Validate and sanitize all agent registration data');
        console.log('   5. 🌐 Use HTTPS for all agent management communications');
        console.log('   6. 📊 Implement audit logging for all security events');
        console.log('   7. 🔍 Validate EventSource URLs to prevent SSRF attacks');
        console.log('   8. 🏭 Limit bulk operations to prevent resource exhaustion');
        console.log('   9. 📋 Implement role-based access control for all operations');
        console.log('   10. 🧪 Secure agent coordination and orchestration endpoints');
        
        this.saveReport();
        
        console.log('\n✅ Scan completed successfully!');
        console.log(`📄 Report saved to: agent7-security-report.json`);
        
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
                agent: 'Agent 7 - Agent Manager',
                scanType: 'Agent Management Security Scan',
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
                'Implement comprehensive authentication checks for all management operations',
                'Add authorization validation for administrative functions',
                'Use secure AJAX requests with CSRF protection for all API calls',
                'Validate and sanitize all agent registration and configuration data',
                'Use HTTPS/WSS for all agent management service communications',
                'Implement comprehensive audit logging for all security events',
                'Validate EventSource URLs to prevent SSRF attacks',
                'Limit bulk operations to prevent resource exhaustion attacks',
                'Implement role-based access control for all management operations',
                'Secure agent coordination and orchestration endpoints'
            ]
        };
        
        fs.writeFileSync('agent7-security-report.json', JSON.stringify(report, null, 2));
    }
}

// Main execution
if (require.main === module) {
    const scanner = new Agent7SecurityScanner();
    const targetDir = process.argv[2] || '../app/a2aFiori/webapp/ext/agent7';
    scanner.scan(targetDir);
}

module.exports = Agent7SecurityScanner;