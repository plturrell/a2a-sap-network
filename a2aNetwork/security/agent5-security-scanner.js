#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Agent 5 (QA Validation Agent) Security Scanner
 * Specialized scanner for test case security, QA validation vulnerabilities,
 * test data tampering, testing framework exploits, and quality assurance workflow security
 */
class Agent5SecurityScanner {
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

        // QA Validation-specific vulnerability patterns
        this.qaValidationPatterns = {
            // Test case injection attacks
            TEST_CASE_INJECTION: {
                patterns: [
                    /testSteps\s*=.*eval/gi,
                    /\.executeTest\s*\([^)]*Function/gi,
                    /testData\s*=.*\$\{/gi,
                    /\.injectTestCase\s*\([^)]*exec/gi,
                    /testQuery\s*=.*user.*\+/gi,
                    /\.manipulateTest\s*\(/gi,
                    /unsafeTestExecution\s*=\s*true/gi,
                    /testInput\s*=.*input.*eval/gi,
                    /dynamicTestCase\s*=.*Function/gi,
                    /\.executeUnsafeTest\s*\(/gi,
                    /testCaseData\s*\+\s*user/gi,
                    /system\s*\(/gi,
                    /subprocess/gi,
                    /import\s+os/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'TEST_CASE_INJECTION',
                message: 'Test case injection vulnerability detected',
                impact: 'Could allow injection of malicious test cases leading to code execution, system compromise, or data exfiltration'
            },

            // Validation rule bypass
            VALIDATION_BYPASS: {
                patterns: [
                    /validationRules\s*=\s*\[\]/gi,
                    /\.bypassValidation\s*\(/gi,
                    /skipQAValidation\s*=\s*true/gi,
                    /\.overrideQuality\s*\([^)]*\$\{/gi,
                    /forceQualityPass\s*=.*user/gi,
                    /\.manipulateQuality\s*\(/gi,
                    /qualitySecurityCheck\s*=\s*false/gi,
                    /bypassQACheck\s*=\s*true/gi,
                    /\.directQualitySet\s*\(/gi,
                    /unsafeQAValidation\s*=\s*true/gi,
                    /disableValidation\s*=\s*true/gi,
                    /skipComplianceCheck\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'VALIDATION_BYPASS',
                message: 'QA validation bypass vulnerability',
                impact: 'Could allow bypassing quality validation leading to release of untested or defective software'
            },

            // Quality criteria manipulation
            QUALITY_MANIPULATION: {
                patterns: [
                    /qualityScore\s*=.*Math\.max/gi,
                    /\.manipulateScore\s*\(/gi,
                    /passRate\s*=\s*100/gi,
                    /\.forcePass\s*\(/gi,
                    /testResults\s*=.*\['PASS'/gi,
                    /coverageThreshold\s*=\s*0/gi,
                    /minPassRate\s*=\s*0/gi,
                    /\.overrideCoverage\s*\(/gi,
                    /fakeTestResults\s*=\s*true/gi,
                    /simulateSuccess\s*=\s*true/gi,
                    /\.alterQualityMetrics\s*\(/gi,
                    /manipulatedResults\s*=/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'QUALITY_MANIPULATION',
                message: 'Quality criteria manipulation vulnerability',
                impact: 'Could allow manipulation of quality metrics leading to false quality assessments'
            },

            // Test data tampering
            TEST_DATA_TAMPERING: {
                patterns: [
                    /testData\s*=.*delete/gi,
                    /\.tamperTestData\s*\(/gi,
                    /modifyTestResults\s*=\s*true/gi,
                    /\.alterTestData\s*\([^)]*user/gi,
                    /testDataManipulation\s*=\s*true/gi,
                    /\.corruptTestData\s*\(/gi,
                    /unsafeTestDataAccess\s*=\s*true/gi,
                    /\.directTestDataModify\s*\(/gi,
                    /testDataIntegrity\s*=\s*false/gi,
                    /allowTestDataEdit\s*=\s*true/gi,
                    /\.injectTestData\s*\(/gi,
                    /testDataSource\s*=.*input/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'TEST_DATA_TAMPERING',
                message: 'Test data tampering vulnerability',
                impact: 'Could allow unauthorized modification of test data leading to compromised test results'
            },

            // QA workflow security issues
            WORKFLOW_SECURITY: {
                patterns: [
                    /workflowSecurity\s*=\s*false/gi,
                    /\.bypassWorkflow\s*\(/gi,
                    /skipApproval\s*=\s*true/gi,
                    /\.overrideWorkflow\s*\([^)]*\$\{/gi,
                    /forceWorkflowStep\s*=.*user/gi,
                    /\.manipulateWorkflow\s*\(/gi,
                    /workflowValidation\s*=\s*false/gi,
                    /bypassApprovalFlow\s*=\s*true/gi,
                    /\.directWorkflowJump\s*\(/gi,
                    /unsafeWorkflow\s*=\s*true/gi,
                    /workflowIntegrity\s*=\s*false/gi,
                    /allowWorkflowSkip\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'WORKFLOW_SECURITY',
                message: 'QA workflow security vulnerability',
                impact: 'Could allow bypassing critical workflow steps leading to unauthorized changes or approvals'
            },

            // Report manipulation
            REPORT_MANIPULATION: {
                patterns: [
                    /reportData\s*=.*splice/gi,
                    /\.manipulateReport\s*\(/gi,
                    /fakeReportData\s*=\s*true/gi,
                    /\.alterReport\s*\([^)]*user/gi,
                    /reportManipulation\s*=\s*true/gi,
                    /\.corruptReport\s*\(/gi,
                    /unsafeReportAccess\s*=\s*true/gi,
                    /\.directReportModify\s*\(/gi,
                    /reportIntegrity\s*=\s*false/gi,
                    /allowReportEdit\s*=\s*true/gi,
                    /\.injectReportData\s*\(/gi,
                    /reportSource\s*=.*input/gi,
                    /modifyExecutiveReport\s*=\s*true/gi,
                    /\.hideFailures\s*\(/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'REPORT_MANIPULATION',
                message: 'Report manipulation vulnerability',
                impact: 'Could allow unauthorized modification of QA reports leading to false quality assessments'
            },

            // Testing framework vulnerabilities
            FRAMEWORK_VULNERABILITY: {
                patterns: [
                    /frameworkSecurity\s*=\s*false/gi,
                    /\.exploitFramework\s*\(/gi,
                    /unsafeFrameworkCall\s*=\s*true/gi,
                    /\.bypassFramework\s*\([^)]*\$\{/gi,
                    /frameworkExploit\s*=.*user/gi,
                    /\.manipulateFramework\s*\(/gi,
                    /frameworkValidation\s*=\s*false/gi,
                    /allowFrameworkHack\s*=\s*true/gi,
                    /\.directFrameworkAccess\s*\(/gi,
                    /unsafeFrameworkMode\s*=\s*true/gi,
                    /frameworkIntegrity\s*=\s*false/gi,
                    /selenium.*exec/gi,
                    /cypress.*eval/gi,
                    /playwright.*Function/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'FRAMEWORK_VULNERABILITY',
                message: 'Testing framework vulnerability',
                impact: 'Could allow exploitation of testing frameworks leading to system compromise or data theft'
            },

            // Compliance validation bypass
            COMPLIANCE_BYPASS: {
                patterns: [
                    /complianceCheck\s*=\s*false/gi,
                    /\.bypassCompliance\s*\(/gi,
                    /skipGDPRCheck\s*=\s*true/gi,
                    /\.overrideCompliance\s*\([^)]*\$\{/gi,
                    /forceCompliance\s*=.*user/gi,
                    /\.manipulateCompliance\s*\(/gi,
                    /complianceValidation\s*=\s*false/gi,
                    /bypassHIPAA\s*=\s*true/gi,
                    /\.directComplianceSet\s*\(/gi,
                    /unsafeCompliance\s*=\s*true/gi,
                    /skipSOXValidation\s*=\s*true/gi,
                    /disableWCAG\s*=\s*true/gi,
                    /bypassPCI\s*=\s*true/gi,
                    /complianceScore\s*=\s*100/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'COMPLIANCE_BYPASS',
                message: 'Compliance validation bypass vulnerability',
                impact: 'Could allow bypassing critical compliance checks leading to regulatory violations'
            },

            // Test execution environment vulnerabilities
            EXECUTION_ENVIRONMENT: {
                patterns: [
                    /testEnvironment\s*=.*production/gi,
                    /\.executeInProd\s*\(/gi,
                    /prodTestExecution\s*=\s*true/gi,
                    /\.runInProduction\s*\([^)]*\$\{/gi,
                    /unsafeEnvironment\s*=.*user/gi,
                    /\.manipulateEnvironment\s*\(/gi,
                    /environmentValidation\s*=\s*false/gi,
                    /allowProdTest\s*=\s*true/gi,
                    /\.directEnvironmentSet\s*\(/gi,
                    /testInLive\s*=\s*true/gi,
                    /environmentIntegrity\s*=\s*false/gi,
                    /productionData\s*=\s*true/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'EXECUTION_ENVIRONMENT',
                message: 'Test execution environment vulnerability',
                impact: 'Could allow execution of tests in production environment leading to data corruption or service disruption'
            },

            // Defect tracking manipulation
            DEFECT_MANIPULATION: {
                patterns: [
                    /defectData\s*=.*splice/gi,
                    /\.manipulateDefect\s*\(/gi,
                    /hideDefects\s*=\s*true/gi,
                    /\.alterDefect\s*\([^)]*user/gi,
                    /defectManipulation\s*=\s*true/gi,
                    /\.corruptDefect\s*\(/gi,
                    /unsafeDefectAccess\s*=\s*true/gi,
                    /\.directDefectModify\s*\(/gi,
                    /defectIntegrity\s*=\s*false/gi,
                    /allowDefectEdit\s*=\s*true/gi,
                    /\.injectDefectData\s*\(/gi,
                    /defectSource\s*=.*input/gi,
                    /suppressDefects\s*=\s*true/gi,
                    /fakeDefectResolution\s*=\s*true/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'DEFECT_MANIPULATION',
                message: 'Defect tracking manipulation vulnerability',
                impact: 'Could allow unauthorized modification of defect data leading to hidden bugs and quality issues'
            }
        };

        // Standard OWASP Top 10 patterns (adapted for QA context)
        this.owaspPatterns = {
            // XSS in test data and reports
            XSS_VULNERABILITY: {
                patterns: [
                    /<script[\s\S]*?>[\s\S]*?<\/script>/gi,
                    /javascript:\s*[^\/]/gi,  // More specific - avoid matching "javascript:" in comments
                    /\bon(load|click|error|focus|blur|submit|change|keyup|keydown|mouseover|mouseout)\s*=/gi,  // Specific dangerous event handlers only
                    /\.innerHTML\s*=\s*[^'"]/gi,  // innerHTML with non-string assignment
                    /document\.write\s*\(/gi,
                    /setTimeout\s*\([^)]*['"].*[<>]/gi,
                    /setInterval\s*\([^)]*['"].*[<>]/gi,
                    /\.outerHTML\s*=/gi,
                    /\.insertAdjacentHTML/gi,
                    /testDescription.*<script/gi,
                    /reportContent.*javascript:/gi,
                    /defectDescription.*\bon(load|click|error)\s*=/gi
                ],
                severity: this.severityLevels.HIGH,
                category: 'XSS_VULNERABILITY',
                message: 'Cross-Site Scripting (XSS) vulnerability in QA data',
                impact: 'Could allow execution of malicious scripts in QA interfaces, reports, or test descriptions'
            },

            // SQL Injection in test queries
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
                    /testQuery\s*\+\s*user/gi,
                    /reportQuery\s*\+\s*input/gi,
                    /defectQuery.*\+/gi,
                    /qualityQuery.*user/gi
                ],
                severity: this.severityLevels.CRITICAL,
                category: 'SQL_INJECTION',
                message: 'SQL Injection vulnerability in QA data queries',
                impact: 'Could allow unauthorized database access, data theft, or data manipulation'
            },

            // CSRF in QA operations
            CSRF_VULNERABILITY: {
                patterns: [
                    /method\s*:\s*['"]POST['"](?![^}]*csrf)/gi,
                    /method\s*:\s*['"]PUT['"](?![^}]*csrf)/gi,
                    /method\s*:\s*['"]DELETE['"](?![^}]*csrf)/gi,
                    /\.post\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /\.put\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /\.delete\s*\([^)]*\)(?![^;]*csrf)/gi,
                    /executeTest.*POST(?![^}]*token)/gi,
                    /generateReport.*POST(?![^}]*csrf)/gi,
                    /createDefect.*POST(?![^}]*token)/gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'CSRF_VULNERABILITY',
                message: 'Cross-Site Request Forgery (CSRF) vulnerability in QA operations',
                impact: 'Could allow unauthorized execution of QA operations like test runs, report generation, or defect creation'
            },

            // Insecure connections
            INSECURE_CONNECTION: {
                patterns: [
                    /http:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /ws:\/\/(?!localhost|127\.0\.0\.1)/gi,
                    /\.protocol\s*=\s*['"]http:['"](?![^}]*localhost)/gi,
                    /url\s*:\s*['"]http:\/\/(?!localhost)/gi,
                    /testServiceUrl.*http:\/\//gi,
                    /reportServiceUrl.*http:\/\//gi,
                    /defectServiceUrl.*ws:\/\//gi,
                    /qaWebSocketUrl.*ws:\/\//gi
                ],
                severity: this.severityLevels.MEDIUM,
                category: 'INSECURE_CONNECTION',
                message: 'Insecure HTTP/WebSocket connection in QA services',
                impact: 'Could expose QA data to man-in-the-middle attacks and eavesdropping'
            }
        };

        // SAP Fiori compliance patterns
        this.fioriCompliancePatterns = {
            // i18n compliance for QA
            I18N_COMPLIANCE: {
                patterns: [
                    /getText\s*\(\s*['"][^'"]*['"]\s*\)/gi,
                    /\.getResourceBundle\(\)\.getText/gi,
                    /i18n\s*>\s*[^{]*\{/gi,
                    /this\._oResourceBundle\.getText/gi
                ],
                severity: this.severityLevels.LOW,
                category: 'I18N_COMPLIANCE',
                message: 'Internationalization compliance check for QA texts',
                impact: 'QA interface texts should be externalized for internationalization',
                isPositive: true
            },

            // Security headers in QA responses
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
                message: 'Security headers implementation in QA responses',
                impact: 'Missing security headers could expose QA interface to various attacks',
                isPositive: true
            }
        };
    }

    /**
     * Main scan function
     */
    async scan(targetDirectory) {
        console.log('🔍 Starting Agent 5 (QA Validation) Security Scan...');
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

            // Scan for QA Validation-specific vulnerabilities
            this.scanPatterns(content, filePath, this.qaValidationPatterns);

            // Scan for OWASP vulnerabilities
            this.scanPatterns(content, filePath, this.owaspPatterns);

            // Scan for SAP Fiori compliance
            this.scanPatterns(content, filePath, this.fioriCompliancePatterns);

            // Additional QA-specific checks
            this.scanForQASpecificIssues(content, filePath);

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
                   lineContext.includes('validateTestCase') || lineContext.includes('sanitize') ||
                   lineContext.includes('validation') || lineContext.includes('_validate');
        }

        // Specific false positives by pattern type
        switch(patternName) {
            case 'TEST_CASE_INJECTION':
                // Skip if it's in pattern definitions or validation functions
                if (lineContext.includes('/subprocess/') || lineContext.includes('aDangerousPatterns') ||
                    lineContext.includes('pattern') || lineContext.includes('validateTestCase')) {
                    return true;
                }
                break;

            case 'XSS_VULNERABILITY':
                // Skip normal XML attributes and property names
                if (matchedText.includes('ontentWidth=') || matchedText.includes('onAPI =') ||
                    matchedText.includes('onTasksTitle=') || matchedText.includes('ontrol =') ||
                    lineContext.includes('contentWidth') || lineContext.includes('onAPI') ||
                    lineContext.includes('validationTasksTitle') || lineContext.includes('control')) {
                    return true;
                }
                // Skip legitimate WebSocket event handlers - specifically handle the pattern
                if ((matchedText.includes('onerror =') || matchedText.includes('onclick =') ||
                     matchedText.includes('onload =') || matchedText.includes('onblur =')) &&
                    (lineContext.includes('_ws.') || lineContext.includes('socket.') ||
                     lineContext.includes('WebSocket') || lineContext.includes('= function()'))) {
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
     * Scan for QA-specific security issues
     */
    scanForQASpecificIssues(content, filePath) {
        // Check for hardcoded test credentials
        const credentialPatterns = [
            /password\s*[:=]\s*['"][^'"]*['"]/gi,
            /apikey\s*[:=]\s*['"][^'"]*['"]/gi,
            /token\s*[:=]\s*['"][^'"]*['"]/gi,
            /secret\s*[:=]\s*['"][^'"]*['"]/gi,
            /testCredentials\s*[:=]/gi,
            /qaPassword\s*[:=]/gi
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
                    message: 'Hardcoded credentials in QA test files',
                    impact: 'Could expose test credentials leading to unauthorized access',
                    pattern: pattern.toString(),
                    match: `${matchedText.substring(0, 50)  }...`,
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // Check for production data usage in tests
        const prodDataPatterns = [
            /productionData\s*[:=]\s*true/gi,
            /useProdData\s*[:=]\s*true/gi,
            /testEnvironment\s*[:=]\s*['"]production['"]/gi,
            /liveData\s*[:=]\s*true/gi
        ];

        prodDataPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;

                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.CRITICAL,
                    category: 'PRODUCTION_DATA_USAGE',
                    message: 'Production data usage in QA tests',
                    impact: 'Could lead to data corruption or privacy violations during testing',
                    pattern: pattern.toString(),
                    match: matches[0],
                    isPositive: false,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // Check for unsafe test execution configurations
        const unsafeConfigPatterns = [
            /maxExecutionTime\s*[:=]\s*-1/gi,
            /timeout\s*[:=]\s*0/gi,
            /securityMode\s*[:=]\s*['"]none['"]/gi,
            /disableSafeguards\s*[:=]\s*true/gi,
            /allowUnsafeExecution\s*[:=]\s*true/gi,
            /parallelThreads\s*[:=]\s*[0-9]{3,}/gi
        ];

        unsafeConfigPatterns.forEach(pattern => {
            const matches = content.match(pattern);
            if (matches) {
                const lines = content.substring(0, content.indexOf(matches[0])).split('\n');
                const lineNumber = lines.length;

                this.vulnerabilities.push({
                    file: filePath,
                    line: lineNumber,
                    severity: this.severityLevels.MEDIUM,
                    category: 'UNSAFE_EXECUTION_CONFIG',
                    message: 'Unsafe test execution configuration',
                    impact: 'Could lead to resource exhaustion or system instability during test execution',
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
        console.log('🛡️  AGENT 5 (QA VALIDATION) SECURITY SCAN REPORT');
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

        console.log('\n🏥 QA VALIDATION SECURITY RECOMMENDATIONS:');
        console.log('   1. 🔒 Implement input validation for all test case data');
        console.log('   2. 🛡️  Sanitize test descriptions and defect reports');
        console.log('   3. 🔐 Use parameterized queries for test data operations');
        console.log('   4. 🚫 Implement CSRF protection for QA operations');
        console.log('   5. 🌐 Use HTTPS for all QA service communications');
        console.log('   6. 📊 Validate and sanitize quality metrics and scores');
        console.log('   7. 🔍 Implement audit trails for all QA operations');
        console.log('   8. 🏭 Ensure test isolation from production environments');
        console.log('   9. 📋 Validate compliance check implementations');
        console.log('   10. 🧪 Secure test execution environments and frameworks');

        this.saveReport();

        console.log('\n✅ Scan completed successfully!');
        console.log('📄 Report saved to: agent5-security-report.json');

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
                agent: 'Agent 5 - QA Validation Agent',
                scanType: 'QA Validation Security Scan',
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
                'Implement comprehensive input validation for test case data',
                'Sanitize all user-generated content in test descriptions and defect reports',
                'Use parameterized queries for all test data database operations',
                'Implement CSRF protection for all QA state-changing operations',
                'Use HTTPS/WSS for all QA service communications',
                'Validate and sanitize quality metrics and compliance scores',
                'Implement comprehensive audit trails for all QA operations',
                'Ensure proper test environment isolation from production',
                'Validate compliance check implementations and standards',
                'Secure test execution environments and testing frameworks'
            ]
        };

        fs.writeFileSync('agent5-security-report.json', JSON.stringify(report, null, 2));
    }
}

// Main execution
if (require.main === module) {
    const scanner = new Agent5SecurityScanner();
    const targetDir = process.argv[2] || '../app/a2aFiori/webapp/ext/agent5';
    scanner.scan(targetDir);
}

module.exports = Agent5SecurityScanner;