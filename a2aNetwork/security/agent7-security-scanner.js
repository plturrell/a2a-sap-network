/**
 * Comprehensive Security Scanner for Agent 7 - Agent Manager
 * Checks for management interface vulnerabilities, privilege escalation, and authorization issues
 */

const fs = require('fs');
const path = require('path');

class Agent7SecurityScanner {
    constructor() {
        this.results = {
            criticalIssues: [],
            highIssues: [],
            mediumIssues: [],
            lowIssues: [],
            complianceGaps: [],
            summary: {
                totalIssues: 0,
                criticalCount: 0,
                highCount: 0,
                mediumCount: 0,
                lowCount: 0,
                complianceScore: 0
            }
        };
        
        this.scannedFiles = [];
        this.basePath = '/Users/apple/projects/a2a/a2aNetwork/app/a2aFiori/webapp/ext/agent7';
    }

    async scanAllFiles() {
        console.log('🔍 Starting comprehensive security scan for Agent 7 - Agent Manager...\n');
        
        // Scan management controller files
        await this.scanFile('controller/AgentManager.controller.js');
        await this.scanFile('controller/MainView.controller.js');
        await this.scanFile('controller/BaseController.js');
        
        // Scan views for privilege escalation issues
        await this.scanFile('view/AgentManager.view.xml');
        await this.scanFile('view/AgentList.view.xml');
        await this.scanFile('view/AgentDetails.view.xml');
        
        // Scan fragments
        await this.scanFile('fragment/CreateAgent.fragment.xml');
        await this.scanFile('fragment/EditAgent.fragment.xml');
        
        // Scan configuration and i18n files
        await this.scanFile('manifest.json');
        await this.scanFile('i18n/i18n.properties');
        
        // Generate comprehensive report
        this.generateSecurityReport();
        
        return this.results;
    }

    async scanFile(relativePath) {
        const filePath = path.join(this.basePath, relativePath);
        
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            console.log(`📄 Scanning: ${relativePath}`);
            
            this.scannedFiles.push(relativePath);
            
            // Perform management-specific security checks
            this.checkPrivilegeEscalation(content, relativePath);
            this.checkAuthorizationBypass(content, relativePath);
            this.checkRoleBasedAccess(content, relativePath);
            this.checkManagementAPISecuity(content, relativePath);
            this.checkAdministrativeActions(content, relativePath);
            this.checkAgentCreationSecurity(content, relativePath);
            this.checkAgentDeletionSecurity(content, relativePath);
            this.checkConfigurationSecurity(content, relativePath);
            this.checkAuditLogging(content, relativePath);
            this.checkSessionElevation(content, relativePath);
            
            // Standard security checks
            this.checkXSSVulnerabilities(content, relativePath);
            this.checkCSRFProtection(content, relativePath);
            this.checkInputValidation(content, relativePath);
            this.checkErrorHandling(content, relativePath);
            this.checkSensitiveDataExposure(content, relativePath);
            
        } catch (error) {
            this.addIssue('high', 'FILE_ACCESS_ERROR', `Cannot read file: ${relativePath}`, relativePath, 0);
        }
    }

    checkPrivilegeEscalation(content, filePath) {
        // Check for functions that could allow privilege escalation
        const privilegePatterns = [
            /setRole\([^)]*\+/g,
            /elevatePermission|grantAccess|promoteUser/gi,
            /admin\s*=\s*true/gi,
            /superuser\s*=\s*[^f]/gi
        ];
        
        privilegePatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('critical', 'PRIVILEGE_ESCALATION_RISK', 
                    'Potential privilege escalation vulnerability in management function',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });

        // Check for direct role manipulation without proper validation
        const roleManipulationPattern = /role\s*=\s*[^"'].*getProperty/g;
        const matches = content.matchAll(roleManipulationPattern);
        
        for (let match of matches) {
            this.addIssue('high', 'ROLE_MANIPULATION_INSECURE', 
                'Role assignment based on user input without proper validation',
                filePath, this.getLineNumber(content, match.index), match[0]);
        }
    }

    checkAuthorizationBypass(content, filePath) {
        // Check for authorization bypass patterns
        const bypassPatterns = [
            /if\s*\([^)]*admin[^)]*\)\s*{[^}]*\/\/\s*bypass/gi,
            /checkPermission\([^)]*\)\s*\|\|\s*true/g,
            /authorize\([^)]*\)\s*\?\s*[^:]*:\s*true/g
        ];
        
        bypassPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('critical', 'AUTHORIZATION_BYPASS', 
                    'Potential authorization bypass in management logic',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });

        // Check for missing authorization checks in sensitive functions
        const sensitiveActions = [
            'deleteAgent', 'createAgent', 'modifyAgent', 'resetAgent', 
            'suspendAgent', 'activateAgent', 'configureAgent'
        ];
        
        sensitiveActions.forEach(action => {
            const actionPattern = new RegExp(`${action}\\s*[:=]\\s*function`, 'g');
            const matches = content.matchAll(actionPattern);
            
            for (let match of matches) {
                const functionStart = match.index;
                const functionEnd = this.findFunctionEnd(content, functionStart);
                const functionBody = content.slice(functionStart, functionEnd);
                
                if (!functionBody.includes('authorize') && !functionBody.includes('checkPermission') && 
                    !functionBody.includes('hasRole')) {
                    this.addIssue('high', 'MISSING_AUTHORIZATION_CHECK', 
                        `Sensitive management function '${action}' lacks authorization check`,
                        filePath, this.getLineNumber(content, functionStart));
                }
            }
        });
    }

    checkRoleBasedAccess(content, filePath) {
        // Check for proper role-based access control implementation
        if (content.includes('getModel') && content.includes('role')) {
            // Check if role validation is proper
            const roleValidationPattern = /role\s*===?\s*["'][^"']+["']/g;
            const matches = content.matchAll(roleValidationPattern);
            
            if (matches.length === 0 && content.includes('role')) {
                this.addIssue('medium', 'WEAK_ROLE_VALIDATION', 
                    'Role-based access control may not be properly implemented',
                    filePath, 1);
            }
        }

        // Check for hardcoded role assignments
        const hardcodedRolePattern = /role\s*=\s*["'](admin|manager|super)["']/gi;
        const matches = content.matchAll(hardcodedRolePattern);
        
        for (let match of matches) {
            this.addIssue('high', 'HARDCODED_ROLE_ASSIGNMENT', 
                'Hardcoded role assignment detected - should use proper role management',
                filePath, this.getLineNumber(content, match.index), match[0]);
        }
    }

    checkManagementAPISecuity(content, filePath) {
        // Check for management API endpoint security
        const managementApiPatterns = [
            /\/a2a\/agents?\/[^"']*\/(create|delete|modify|suspend|activate)/g,
            /\/admin\/[^"']*/g,
            /\/manage\/[^"']*/g
        ];
        
        managementApiPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                // Check if the API call has proper authentication
                const apiCallContext = this.extractApiCallContext(content, match.index);
                
                if (!apiCallContext.includes('Authorization') && 
                    !apiCallContext.includes('X-CSRF-Token') &&
                    !apiCallContext.includes('bearer')) {
                    this.addIssue('critical', 'MANAGEMENT_API_UNSECURED', 
                        'Management API call lacks proper authentication/authorization headers',
                        filePath, this.getLineNumber(content, match.index), match[0]);
                }
            }
        });
    }

    checkAdministrativeActions(content, filePath) {
        // Check for proper validation of administrative actions
        const adminActions = [
            'bulkDelete', 'massUpdate', 'systemReset', 'configureAll', 
            'deployAll', 'suspendAll', 'migrateAgents'
        ];
        
        adminActions.forEach(action => {
            if (content.includes(action)) {
                const actionContext = this.extractFunctionContext(content, action);
                
                // Check for confirmation mechanisms
                if (!actionContext.includes('confirm') && !actionContext.includes('MessageBox')) {
                    this.addIssue('high', 'ADMIN_ACTION_NO_CONFIRMATION', 
                        `Administrative action '${action}' lacks user confirmation`,
                        filePath, 1);
                }
                
                // Check for audit logging
                if (!actionContext.includes('log') && !actionContext.includes('audit')) {
                    this.addIssue('medium', 'ADMIN_ACTION_NO_AUDIT', 
                        `Administrative action '${action}' lacks audit logging`,
                        filePath, 1);
                }
            }
        });
    }

    checkAgentCreationSecurity(content, filePath) {
        if (content.includes('createAgent') || content.includes('onCreate')) {
            // Check for proper validation of agent creation parameters
            const createPattern = /(createAgent|onCreate)[^{]*{[^}]*}/gs;
            const matches = content.matchAll(createPattern);
            
            for (let match of matches) {
                const createFunction = match[0];
                
                // Check for input sanitization
                if (!createFunction.includes('validate') && !createFunction.includes('sanitize')) {
                    this.addIssue('high', 'AGENT_CREATION_NO_VALIDATION', 
                        'Agent creation function lacks proper input validation',
                        filePath, this.getLineNumber(content, match.index));
                }
                
                // Check for duplicate prevention
                if (!createFunction.includes('exists') && !createFunction.includes('duplicate')) {
                    this.addIssue('medium', 'AGENT_CREATION_NO_DUPLICATE_CHECK', 
                        'Agent creation lacks duplicate prevention',
                        filePath, this.getLineNumber(content, match.index));
                }
            }
        }
    }

    checkAgentDeletionSecurity(content, filePath) {
        if (content.includes('deleteAgent') || content.includes('onDelete')) {
            // Check for proper cascade deletion handling
            const deletePattern = /(deleteAgent|onDelete)[^{]*{[^}]*}/gs;
            const matches = content.matchAll(deletePattern);
            
            for (let match of matches) {
                const deleteFunction = match[0];
                
                // Check for confirmation dialog
                if (!deleteFunction.includes('MessageBox.confirm') && 
                    !deleteFunction.includes('confirm')) {
                    this.addIssue('high', 'AGENT_DELETION_NO_CONFIRMATION', 
                        'Agent deletion lacks confirmation dialog',
                        filePath, this.getLineNumber(content, match.index));
                }
                
                // Check for cascade relationship handling
                if (!deleteFunction.includes('cascade') && !deleteFunction.includes('dependency')) {
                    this.addIssue('medium', 'AGENT_DELETION_NO_CASCADE_CHECK', 
                        'Agent deletion may not handle cascading relationships properly',
                        filePath, this.getLineNumber(content, match.index));
                }
            }
        }
    }

    checkConfigurationSecurity(content, filePath) {
        if (filePath.includes('manifest.json')) {
            try {
                const config = JSON.parse(content);
                
                // Check for public access on management interface
                if (config['sap.cloud'] && config['sap.cloud'].public === true) {
                    this.addIssue('critical', 'MANAGEMENT_INTERFACE_PUBLIC', 
                        'Management interface is configured as public - should be private',
                        filePath, 1);
                }
                
                // Check for missing OAuth scopes for management
                if (config['sap.platform.cf'] && config['sap.platform.cf'].oAuthScopes) {
                    const scopes = config['sap.platform.cf'].oAuthScopes;
                    if (!scopes.some(scope => scope.includes('Admin') || scope.includes('Manage'))) {
                        this.addIssue('high', 'MANAGEMENT_OAUTH_SCOPES_MISSING', 
                            'Missing administrative OAuth scopes for management interface',
                            filePath, 1);
                    }
                }
                
            } catch (e) {
                this.addIssue('medium', 'CONFIG_PARSE_ERROR', 
                    'Configuration file could not be parsed properly',
                    filePath, 1);
            }
        }
    }

    checkAuditLogging(content, filePath) {
        // Check for proper audit logging in management actions
        const auditableActions = [
            'createAgent', 'deleteAgent', 'modifyAgent', 'suspendAgent', 
            'activateAgent', 'resetAgent', 'configureAgent'
        ];
        
        auditableActions.forEach(action => {
            if (content.includes(action)) {
                const actionContext = this.extractFunctionContext(content, action);
                
                if (!actionContext.includes('audit') && !actionContext.includes('log') && 
                    !actionContext.includes('track')) {
                    this.addIssue('medium', 'AUDIT_LOGGING_MISSING', 
                        `Management action '${action}' lacks audit logging`,
                        filePath, 1);
                }
            }
        });
    }

    checkSessionElevation(content, filePath) {
        // Check for temporary privilege elevation patterns
        const elevationPatterns = [
            /sudo|elevate|impersonate/gi,
            /runAsAdmin|executeAsRoot/gi,
            /temporaryPrivilege|tempElevation/gi
        ];
        
        elevationPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('high', 'SESSION_ELEVATION_RISK', 
                    'Potential session privilege elevation detected',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkXSSVulnerabilities(content, filePath) {
        // XSS checks specific to management interface
        const xssPatterns = [
            /innerHTML\s*=\s*[^"'].*agentName/g,
            /\.html\([^)]*agentId/g,
            /MessageBox\.(success|error|warning|information)\([^)]*\+.*agent/g
        ];
        
        xssPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('critical', 'XSS_MANAGEMENT_INTERFACE', 
                    'Potential XSS vulnerability in management interface',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkCSRFProtection(content, filePath) {
        // Check AJAX calls for CSRF protection in management operations
        const ajaxPattern = /jQuery\.ajax\s*\({[^}]*type:\s*["']?(POST|PUT|DELETE|PATCH)["']?[^}]*}/gs;
        const matches = content.matchAll(ajaxPattern);
        
        for (let match of matches) {
            const ajaxCall = match[0];
            
            if (!ajaxCall.includes('X-CSRF-Token') && !ajaxCall.includes('csrf')) {
                this.addIssue('high', 'CSRF_MISSING_MANAGEMENT_API', 
                    'Management API call missing CSRF protection',
                    filePath, this.getLineNumber(content, match.index));
            }
        }
    }

    checkInputValidation(content, filePath) {
        // Check for input validation in management forms
        const managementInputs = [
            'agentName', 'agentConfig', 'agentType', 'permissions', 
            'roleAssignment', 'accessLevel'
        ];
        
        managementInputs.forEach(input => {
            const inputPattern = new RegExp(`${input}[^;]*getProperty[^;]*`, 'g');
            const matches = content.matchAll(inputPattern);
            
            for (let match of matches) {
                const context = content.slice(match.index, match.index + 300);
                if (!context.includes('validate') && !context.includes('sanitize')) {
                    this.addIssue('high', 'MANAGEMENT_INPUT_NOT_VALIDATED', 
                        `Management input '${input}' processed without validation`,
                        filePath, this.getLineNumber(content, match.index));
                }
            }
        });
    }

    checkErrorHandling(content, filePath) {
        // Check for information disclosure in management error handling
        const errorPatterns = [
            /error:\s*function[^}]*agentId[^}]*responseText/g,
            /catch[^}]*console\.log[^}]*agent/g,
            /MessageBox\.error[^)]*xhr\.responseText/g
        ];
        
        errorPatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('medium', 'MANAGEMENT_ERROR_INFO_DISCLOSURE', 
                    'Management error handling may expose sensitive agent information',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    checkSensitiveDataExposure(content, filePath) {
        // Check for sensitive management data exposure
        const sensitivePatterns = [
            /console\.log.*password|secret|key|token/gi,
            /alert\(.*config.*\)/gi,
            /debugger.*agent/gi
        ];
        
        sensitivePatterns.forEach(pattern => {
            const matches = content.matchAll(pattern);
            for (let match of matches) {
                this.addIssue('high', 'SENSITIVE_MANAGEMENT_DATA_EXPOSURE', 
                    'Potential sensitive management data exposure',
                    filePath, this.getLineNumber(content, match.index), match[0]);
            }
        });
    }

    // Helper methods
    extractApiCallContext(content, startIndex) {
        const contextStart = Math.max(0, startIndex - 200);
        const contextEnd = Math.min(content.length, startIndex + 200);
        return content.slice(contextStart, contextEnd);
    }

    extractFunctionContext(content, functionName) {
        const functionIndex = content.indexOf(functionName);
        if (functionIndex === -1) return '';
        
        const functionStart = functionIndex;
        const functionEnd = this.findFunctionEnd(content, functionStart);
        return content.slice(functionStart, functionEnd);
    }

    getLineNumber(content, index) {
        return content.slice(0, index).split('\n').length;
    }

    findFunctionEnd(content, startIndex) {
        let braceCount = 0;
        let inFunction = false;
        
        for (let i = startIndex; i < content.length; i++) {
            const char = content[i];
            
            if (char === '{') {
                braceCount++;
                inFunction = true;
            } else if (char === '}') {
                braceCount--;
                if (inFunction && braceCount === 0) {
                    return i + 1;
                }
            }
        }
        
        return startIndex + 500; // fallback
    }

    addIssue(severity, type, description, filePath, lineNumber = 0, code = '') {
        const issue = {
            severity,
            type,
            description,
            file: filePath,
            line: lineNumber,
            code: code.slice(0, 100) + (code.length > 100 ? '...' : '')
        };

        switch (severity) {
            case 'critical':
                this.results.criticalIssues.push(issue);
                this.results.summary.criticalCount++;
                break;
            case 'high':
                this.results.highIssues.push(issue);
                this.results.summary.highCount++;
                break;
            case 'medium':
                this.results.mediumIssues.push(issue);
                this.results.summary.mediumCount++;
                break;
            case 'low':
                this.results.lowIssues.push(issue);
                this.results.summary.lowCount++;
                break;
        }
        
        this.results.summary.totalIssues++;
    }

    generateSecurityReport() {
        this.results.summary.complianceScore = this.calculateComplianceScore();
        
        console.log('\n📊 AGENT 7 - AGENT MANAGER SECURITY SCAN RESULTS');
        console.log('================================================');
        console.log(`🔴 Critical Issues: ${this.results.summary.criticalCount}`);
        console.log(`🟠 High Issues: ${this.results.summary.highCount}`);
        console.log(`🟡 Medium Issues: ${this.results.summary.mediumCount}`);
        console.log(`🟢 Low Issues: ${this.results.summary.lowCount}`);
        console.log(`📋 Total Issues: ${this.results.summary.totalIssues}`);
        console.log(`🎯 Compliance Score: ${this.results.summary.complianceScore}%`);
        console.log(`📁 Files Scanned: ${this.scannedFiles.length}`);
        
        // Display critical and high issues
        if (this.results.criticalIssues.length > 0) {
            console.log('\n🔴 CRITICAL SECURITY ISSUES:');
            this.results.criticalIssues.forEach((issue, index) => {
                console.log(`${index + 1}. ${issue.type}`);
                console.log(`   File: ${issue.file}:${issue.line}`);
                console.log(`   Description: ${issue.description}`);
                if (issue.code) console.log(`   Code: ${issue.code}`);
                console.log('');
            });
        }

        if (this.results.highIssues.length > 0) {
            console.log('\n🟠 HIGH PRIORITY ISSUES:');
            this.results.highIssues.slice(0, 5).forEach((issue, index) => {
                console.log(`${index + 1}. ${issue.type}`);
                console.log(`   File: ${issue.file}:${issue.line}`);
                console.log(`   Description: ${issue.description}`);
                console.log('');
            });
            
            if (this.results.highIssues.length > 5) {
                console.log(`   ... and ${this.results.highIssues.length - 5} more high priority issues`);
            }
        }
    }

    calculateComplianceScore() {
        const totalPossiblePoints = 100;
        const criticalPenalty = 25;
        const highPenalty = 15;
        const mediumPenalty = 7;
        const lowPenalty = 2;
        
        let deductions = 
            (this.results.summary.criticalCount * criticalPenalty) +
            (this.results.summary.highCount * highPenalty) +
            (this.results.summary.mediumCount * mediumPenalty) +
            (this.results.summary.lowCount * lowPenalty);
        
        return Math.max(0, totalPossiblePoints - deductions);
    }

    generateDetailedReport() {
        const report = {
            scanTimestamp: new Date().toISOString(),
            scanTarget: 'Agent 7 - Agent Manager',
            filesScanned: this.scannedFiles,
            summary: this.results.summary,
            issues: {
                critical: this.results.criticalIssues,
                high: this.results.highIssues,
                medium: this.results.mediumIssues,
                low: this.results.lowIssues
            },
            recommendations: this.generateRecommendations()
        };
        
        return report;
    }

    generateRecommendations() {
        const recommendations = [];
        
        if (this.results.summary.criticalCount > 0) {
            recommendations.push('🚨 URGENT: Implement proper privilege escalation protection');
            recommendations.push('Add authorization checks to all management functions');
            recommendations.push('Secure management API endpoints with proper authentication');
        }
        
        if (this.results.summary.highCount > 0) {
            recommendations.push('🔧 Implement role-based access control with proper validation');
            recommendations.push('Add CSRF protection to all management operations');
            recommendations.push('Implement confirmation dialogs for destructive actions');
        }
        
        if (this.results.summary.mediumCount > 0) {
            recommendations.push('📝 Add comprehensive audit logging for all management actions');
            recommendations.push('Implement proper error handling without information disclosure');
            recommendations.push('Add input validation for all management forms');
        }
        
        recommendations.push('🔐 Regular security reviews of management interfaces');
        recommendations.push('🎓 Security training focused on privilege management');
        recommendations.push('🏗️ Implement security controls in agent lifecycle management');
        
        return recommendations;
    }
}

// Export for use
module.exports = Agent7SecurityScanner;

// Run if called directly
if (require.main === module) {
    const scanner = new Agent7SecurityScanner();
    scanner.scanAllFiles().then(results => {
        const report = scanner.generateDetailedReport();
        
        // Save detailed report
        const reportPath = '/Users/apple/projects/a2a/a2aNetwork/security/agent7-security-report.json';
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        
        console.log(`\n📄 Detailed report saved to: ${reportPath}`);
        console.log('\n✅ Security scan completed!');
        
        // Exit with appropriate code
        process.exit(results.summary.criticalCount > 0 ? 2 : 
                    results.summary.highCount > 0 ? 1 : 0);
    }).catch(error => {
        console.error('❌ Security scan failed:', error);
        process.exit(3);
    });
}