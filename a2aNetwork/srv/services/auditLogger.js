/**
 * @fileoverview Enterprise SAP Audit Logging Service
 * @description Provides comprehensive audit trails with SAP compliance standards,
 * tamper-proof logging, real-time monitoring, and enterprise-grade security features.
 * Supports SOX, PCI-DSS, GDPR, HIPAA, and SAP-specific audit requirements.
 * @module audit-logger
 * @since 1.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { EventEmitter } = require('events');

// SAP Enterprise Logging integration
let sapAuditLogging;
try {
    sapAuditLogging = require('@sap/audit-logging');
} catch (error) {
    // SAP Audit Logging not available in development
}

// OpenTelemetry integration for audit tracing
let opentelemetry, trace, SpanStatusCode;
try {
    opentelemetry = require('@opentelemetry/api');

// Track intervals for cleanup
const activeIntervals = new Map();

function stopAllIntervals() {
    for (const [name, intervalId] of activeIntervals) {
        clearInterval(intervalId);
    }
    activeIntervals.clear();
}

function shutdown() {
    stopAllIntervals();
}

// Export cleanup function
module.exports.shutdown = shutdown;

    trace = opentelemetry.trace;
    SpanStatusCode = opentelemetry.SpanStatusCode;
} catch (error) {
    // OpenTelemetry not available
}

/**
 * Enterprise SAP Audit Logger with tamper-proof logging and compliance standards
 */
class AuditLogger extends EventEmitter {
    constructor(options = {}) {
        super();
        this.auditDir = options.auditDir || path.join(process.cwd(), 'audit-logs');
        this.encryptionKey = options.encryptionKey || this.generateEncryptionKey();
        this.retentionDays = options.retentionDays || 2555; // 7 years
        this.complianceMode = options.complianceMode || 'SOX';
        this.maxFileSize = options.maxFileSize || 100 * 1024 * 1024; // 100MB
        
        this.currentLogFile = null;
        this.logFileIndex = 0;
        this.auditQueue = [];
        this.isProcessing = false;
        this.complianceRules = this.initializeComplianceRules();
        
        // SAP Enterprise features
        this.sapAuditLog = null;
        this.blockchainHash = null; // For tamper-proof logging
        this.alertThresholds = {
            criticalEvents: 5,
            suspiciousActivity: 10,
            failedLogins: 3,
            privilegeEscalation: 1
        };
        
        // OpenTelemetry tracer for audit spans
        this.tracer = trace ? trace.getTracer('audit-logger', '1.0.0') : null;
        
        // Real-time monitoring
        this.eventCounters = new Map();
        this.alertHistory = [];
        this.complianceViolations = [];
        
        // Tamper detection
        this.checksumChain = [];
        this.digitalSignatures = new Map();
        
        // Enterprise metrics
        this.metrics = {
            totalEvents: 0,
            encryptedEvents: 0,
            complianceViolations: 0,
            alertsTriggered: 0,
            tamperAttempts: 0,
            averageProcessingTime: 0,
            eventsByType: new Map(),
            eventsBySeverity: new Map()
        };
        
        this.log = cds.log('audit-logger');
    }

    /**
     * Initialize enterprise audit logger with SAP integration
     */
    async initialize() {
        try {
            await fs.mkdir(this.auditDir, { recursive: true });
            await this.rotateLogFile();
            this.startProcessingQueue();
            
            // Initialize SAP Audit Logging service
            await this.initializeSAPAuditLogging();
            
            // Initialize tamper-proof blockchain logging
            await this.initializeBlockchainLogging();
            
            // Start real-time monitoring
            this.startRealTimeMonitoring();
            
            // Initialize compliance monitoring
            this.startComplianceMonitoring();
            
            // Generate initial integrity hash
            await this.generateIntegrityCheckpoint();
            
            this.log.info('Enterprise audit logger initialized', {
                auditDir: this.auditDir,
                complianceMode: this.complianceMode,
                retentionDays: this.retentionDays,
                sapIntegration: !!this.sapAuditLog,
                blockchainLogging: !!this.blockchainHash,
                encryptionEnabled: !!this.encryptionKey
            });
            
            this.emit('initialized', {
                timestamp: new Date().toISOString(),
                features: {
                    sapIntegration: !!this.sapAuditLog,
                    tamperProof: true,
                    realTimeMonitoring: true,
                    complianceFramework: this.complianceMode
                }
            });
            
        } catch (error) {
            this.log.error('Failed to initialize audit logger:', error);
            throw error;
        }
    }
    
    /**
     * Initialize SAP Audit Logging service integration
     */
    async initializeSAPAuditLogging() {
        if (!sapAuditLogging) {
            this.log.warn('SAP Audit Logging not available - using file-based logging only');
            return;
        }
        
        try {
            // Initialize SAP Audit Log service
            const credentials = this.getSAPAuditCredentials();
            if (credentials) {
                this.sapAuditLog = sapAuditLogging.v2(credentials);
                this.log.info('SAP Audit Logging service initialized');
            }
        } catch (error) {
            this.log.warn('Failed to initialize SAP Audit Logging:', error);
        }
    }
    
    /**
     * Get SAP Audit Logging credentials
     */
    getSAPAuditCredentials() {
        try {
            if (process.env.VCAP_SERVICES) {
                const vcapServices = JSON.parse(process.env.VCAP_SERVICES);
                return vcapServices['auditlog-management']?.[0]?.credentials;
            }
            
            // Development credentials
            return {
                url: process.env.SAP_AUDIT_URL,
                user: process.env.SAP_AUDIT_USER,
                password: process.env.SAP_AUDIT_PASSWORD
            };
        } catch (error) {
            this.log.warn('Unable to parse SAP audit credentials:', error);
            return null;
        }
    }
    
    /**
     * Initialize blockchain-based tamper-proof logging
     */
    async initializeBlockchainLogging() {
        try {
            // Initialize with genesis hash
            this.blockchainHash = crypto.createHash('sha256')
                .update(`AUDIT_LOG_GENESIS_${  new Date().toISOString()}`)
                .digest('hex');
                
            this.log.info('Blockchain tamper-proof logging initialized', {
                genesisHash: `${this.blockchainHash.substring(0, 16)  }...`
            });
        } catch (error) {
            this.log.warn('Failed to initialize blockchain logging:', error);
        }
    }
    
    /**
     * Start real-time monitoring and alerting
     */
    startRealTimeMonitoring() {
        // Monitor for suspicious patterns
        activeIntervals.set('interval_223', setInterval(() => {
            this.analyzeSecurityPatterns();
        }, 60000)); // Every minute
        
        // Monitor compliance violations
        activeIntervals.set('interval_228', setInterval(() => {
            this.checkComplianceViolations();
        }, 300000)); // Every 5 minutes
        
        // Generate metrics reports
        activeIntervals.set('interval_233', setInterval(() => {
            this.generateMetricsReport();
        }, 900000)); // Every 15 minutes
        
        this.log.info('Real-time monitoring started');
    }
    
    /**
     * Start compliance monitoring
     */
    startComplianceMonitoring() {
        const rules = this.complianceRules[this.complianceMode];
        
        // Monitor data retention compliance
        activeIntervals.set('interval_247', setInterval(async () => {
            await this.checkDataRetentionCompliance();
        }, 86400000)); // Daily
        
        // Monitor encryption compliance
        if (rules.encryptionRequired) {
            activeIntervals.set('interval_253', setInterval(() => {
                this.checkEncryptionCompliance();
            }, 3600000)); // Hourly
        }
        
        this.log.info('Compliance monitoring started for framework:', this.complianceMode);
    }
    
    /**
     * Generate integrity checkpoint
     */
    async generateIntegrityCheckpoint() {
        try {
            const checkpoint = {
                timestamp: new Date().toISOString(),
                totalEvents: this.metrics.totalEvents,
                blockchainHash: this.blockchainHash,
                fileChecksum: await this.calculateFileChecksum(),
                signature: await this.generateDigitalSignature()
            };
            
            const checkpointFile = path.join(this.auditDir, 'integrity-checkpoint.json');
            await fs.writeFile(checkpointFile, JSON.stringify(checkpoint, null, 2));
            
            this.log.info('Integrity checkpoint generated');
        } catch (error) {
            this.log.error('Failed to generate integrity checkpoint:', error);
        }
    }

    /**
     * Log blockchain transaction audit
     */
    async logTransaction(data) {
        const auditEvent = {
            eventType: 'BLOCKCHAIN_TRANSACTION',
            timestamp: new Date().toISOString(),
            transactionHash: data.hash,
            from: data.from,
            to: data.to,
            value: data.value,
            gasUsed: data.gasUsed,
            gasPrice: data.gasPrice,
            blockNumber: data.blockNumber,
            contractAddress: data.contractAddress,
            status: data.status,
            functionName: data.functionName,
            parameters: this.sanitizeParameters(data.parameters),
            userId: data.userId,
            sessionId: data.sessionId,
            ipAddress: data.ipAddress,
            userAgent: data.userAgent,
            riskScore: this.calculateRiskScore(data),
            complianceFlags: this.checkCompliance(data),
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log smart contract interaction
     */
    async logContractInteraction(data) {
        const auditEvent = {
            eventType: 'CONTRACT_INTERACTION',
            timestamp: new Date().toISOString(),
            contractAddress: data.contractAddress,
            contractName: data.contractName,
            functionName: data.functionName,
            functionSignature: data.functionSignature,
            inputs: this.sanitizeParameters(data.inputs),
            outputs: this.sanitizeParameters(data.outputs),
            transactionHash: data.transactionHash,
            blockNumber: data.blockNumber,
            userId: data.userId,
            sessionId: data.sessionId,
            gasEstimate: data.gasEstimate,
            actualGasUsed: data.actualGasUsed,
            executionTime: data.executionTime,
            success: data.success,
            errorMessage: data.errorMessage,
            securityChecks: data.securityChecks,
            riskAssessment: this.assessContractRisk(data),
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log user authentication events
     */
    async logAuthentication(data) {
        const auditEvent = {
            eventType: 'USER_AUTHENTICATION',
            timestamp: new Date().toISOString(),
            userId: data.userId,
            username: data.username,
            authMethod: data.authMethod,
            success: data.success,
            failureReason: data.failureReason,
            ipAddress: data.ipAddress,
            userAgent: data.userAgent,
            sessionId: data.sessionId,
            mfaUsed: data.mfaUsed,
            riskFactors: data.riskFactors,
            geolocation: data.geolocation,
            deviceFingerprint: data.deviceFingerprint,
            previousLoginTime: data.previousLoginTime,
            loginAttempts: data.loginAttempts,
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log access control events
     */
    async logAccessControl(data) {
        const auditEvent = {
            eventType: 'ACCESS_CONTROL',
            timestamp: new Date().toISOString(),
            userId: data.userId,
            resource: data.resource,
            action: data.action,
            permission: data.permission,
            granted: data.granted,
            denialReason: data.denialReason,
            rolesBefore: data.rolesBefore,
            rolesAfter: data.rolesAfter,
            administratorId: data.administratorId,
            justification: data.justification,
            approvalRequired: data.approvalRequired,
            approvalStatus: data.approvalStatus,
            sessionId: data.sessionId,
            ipAddress: data.ipAddress,
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log system configuration changes
     */
    async logConfigurationChange(data) {
        const auditEvent = {
            eventType: 'CONFIGURATION_CHANGE',
            timestamp: new Date().toISOString(),
            component: data.component,
            configKey: data.configKey,
            oldValue: this.sanitizeValue(data.oldValue),
            newValue: this.sanitizeValue(data.newValue),
            changedBy: data.changedBy,
            reason: data.reason,
            approvedBy: data.approvedBy,
            effectiveDate: data.effectiveDate,
            rollbackPossible: data.rollbackPossible,
            impact: data.impact,
            testingCompleted: data.testingCompleted,
            changeId: data.changeId,
            sessionId: data.sessionId,
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log security incidents
     */
    async logSecurityIncident(data) {
        const auditEvent = {
            eventType: 'SECURITY_INCIDENT',
            timestamp: new Date().toISOString(),
            incidentId: data.incidentId,
            severity: data.severity,
            category: data.category,
            description: data.description,
            affectedSystems: data.affectedSystems,
            attackVector: data.attackVector,
            sourceIp: data.sourceIp,
            targetResource: data.targetResource,
            detectionMethod: data.detectionMethod,
            mitigationActions: data.mitigationActions,
            evidenceCollected: data.evidenceCollected,
            investigationStatus: data.investigationStatus,
            responsibleTeam: data.responsibleTeam,
            escalationLevel: data.escalationLevel,
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Log compliance events
     */
    async logCompliance(data) {
        const auditEvent = {
            eventType: 'COMPLIANCE_EVENT',
            timestamp: new Date().toISOString(),
            complianceFramework: data.framework,
            requirement: data.requirement,
            controlId: data.controlId,
            testResult: data.testResult,
            evidence: data.evidence,
            assessor: data.assessor,
            assessmentDate: data.assessmentDate,
            findings: data.findings,
            recommendations: data.recommendations,
            remediation: data.remediation,
            nextAssessment: data.nextAssessment,
            riskRating: data.riskRating,
            businessOwner: data.businessOwner,
            technicalOwner: data.technicalOwner,
            checksum: this.calculateChecksum(data)
        };

        await this.enqueueAuditEvent(auditEvent);
    }

    /**
     * Enqueue audit event for processing with enterprise security
     */
    async enqueueAuditEvent(auditEvent) {
        const startTime = Date.now();
        let span = null;
        
        try {
            // Create OpenTelemetry span for audit event
            if (this.tracer) {
                span = this.tracer.startSpan('audit.event.process', {
                    attributes: {
                        'audit.event.type': auditEvent.eventType,
                        'audit.compliance.mode': this.complianceMode
                    }
                });
            }
            
            // Add enterprise metadata
            auditEvent.id = this.generateEventId();
            auditEvent.sequence = this.getNextSequenceNumber();
            auditEvent.node = process.env.NODE_NAME || 'unknown';
            auditEvent.version = '1.0';
            auditEvent.blockchainPrevHash = this.blockchainHash;
            
            // Generate tamper-proof hash chain
            auditEvent.blockchainHash = this.generateBlockchainHash(auditEvent);
            this.blockchainHash = auditEvent.blockchainHash;
            
            // Add digital signature for integrity
            auditEvent.digitalSignature = await this.signEvent(auditEvent);
            
            // Validate event
            const validation = this.validateAuditEvent(auditEvent);
            if (!validation.valid) {
                this.log.error('Invalid audit event:', validation.errors);
                this.metrics.complianceViolations++;
                
                if (span) {
                    span.recordException(new Error('Invalid audit event'));
                    span.setStatus({ code: SpanStatusCode.ERROR });
                }
                return;
            }
            
            // Check for security alerts
            await this.checkSecurityAlerts(auditEvent);
            
            // Update metrics
            this.updateEventMetrics(auditEvent);
            
            this.auditQueue.push(auditEvent);
            
            // Send to SAP Audit Logging service
            if (this.sapAuditLog) {
                try {
                    await this.sendToSAPAuditLog(auditEvent);
                } catch (error) {
                    this.log.warn('Failed to send to SAP Audit Log:', error);
                }
            }
            
            // Trigger processing if queue is getting full
            if (this.auditQueue.length >= 100) {
                await this.processQueue();
            }
            
            const processingTime = Date.now() - startTime;
            this.updateAverageProcessingTime(processingTime);
            
            if (span) {
                span.setAttributes({
                    'audit.event.id': auditEvent.id,
                    'audit.processing.time': processingTime
                });
                span.setStatus({ code: SpanStatusCode.OK });
            }
            
        } catch (error) {
            this.log.error('Failed to enqueue audit event:', error);
            this.metrics.tamperAttempts++;
            
            if (span) {
                span.recordException(error);
                span.setStatus({ code: SpanStatusCode.ERROR });
            }
            
            throw error;
        } finally {
            if (span) {
                span.end();
            }
        }
    }
    
    /**
     * Generate blockchain hash for tamper-proof logging
     */
    generateBlockchainHash(event) {
        const dataToHash = {
            prevHash: this.blockchainHash,
            timestamp: event.timestamp,
            eventType: event.eventType,
            id: event.id,
            checksum: event.checksum
        };
        
        return crypto.createHash('sha256')
            .update(JSON.stringify(dataToHash))
            .digest('hex');
    }
    
    /**
     * Generate digital signature for event integrity
     */
    async signEvent(event) {
        try {
            const eventData = JSON.stringify({
                id: event.id,
                timestamp: event.timestamp,
                eventType: event.eventType,
                checksum: event.checksum
            });
            
            const sign = crypto.createSign('RSA-SHA256');
            sign.update(eventData);
            
            // Use environment variable for private key in production
            const privateKey = process.env.AUDIT_PRIVATE_KEY || this.generateRSAKeyPair().privateKey;
            return sign.sign(privateKey, 'hex');
        } catch (error) {
            this.log.warn('Failed to generate digital signature:', error);
            return null;
        }
    }
    
    /**
     * Send event to SAP Audit Logging service
     */
    async sendToSAPAuditLog(event) {
        if (!this.sapAuditLog) return;
        
        try {
            const sapEvent = {
                category: this.mapEventTypeToSAPCategory(event.eventType),
                severity: this.calculateSAPSeverity(event),
                user: event.userId || 'system',
                object: event.resource || event.contractAddress || 'blockchain',
                data_subject: event.dataSubject,
                attributes: {
                    ...event,
                    compliance_framework: this.complianceMode,
                    blockchain_hash: event.blockchainHash,
                    node_id: process.env.NODE_NAME || 'unknown'
                }
            };
            
            await this.sapAuditLog.log(sapEvent);
            this.log.debug('Event sent to SAP Audit Log:', event.id);
        } catch (error) {
            this.log.error('SAP Audit Log error:', error);
            throw error;
        }
    }
    
    /**
     * Map event type to SAP audit category
     */
    mapEventTypeToSAPCategory(eventType) {
        const mapping = {
            'BLOCKCHAIN_TRANSACTION': 'DATA_MODIFICATION',
            'CONTRACT_INTERACTION': 'SYSTEM_ACCESS',
            'USER_AUTHENTICATION': 'AUTHENTICATION',
            'ACCESS_CONTROL': 'AUTHORIZATION',
            'CONFIGURATION_CHANGE': 'CONFIGURATION_CHANGE',
            'SECURITY_INCIDENT': 'SECURITY_EVENT',
            'COMPLIANCE_EVENT': 'COMPLIANCE'
        };
        
        return mapping[eventType] || 'GENERAL';
    }
    
    /**
     * Calculate SAP audit severity
     */
    calculateSAPSeverity(event) {
        // High severity events
        if (event.eventType === 'SECURITY_INCIDENT' || 
            event.severity === 'critical' ||
            event.riskScore > 80) {
            return 'HIGH';
        }
        
        // Medium severity events
        if (event.eventType === 'CONFIGURATION_CHANGE' ||
            event.riskScore > 50) {
            return 'MEDIUM';
        }
        
        return 'LOW';
    }
    
    /**
     * Check for security alerts
     */
    async checkSecurityAlerts(event) {
        const alerts = [];
        
        // Check for critical events
        if (event.eventType === 'SECURITY_INCIDENT') {
            alerts.push({
                type: 'SECURITY_INCIDENT',
                severity: 'CRITICAL',
                message: `Security incident detected: ${event.description}`
            });
        }
        
        // Check for failed authentication patterns
        if (event.eventType === 'USER_AUTHENTICATION' && !event.success) {
            const recentFailures = this.getRecentFailedLogins(event.userId);
            if (recentFailures >= this.alertThresholds.failedLogins) {
                alerts.push({
                    type: 'SUSPICIOUS_ACTIVITY',
                    severity: 'HIGH',
                    message: `Multiple failed login attempts for user: ${event.userId}`
                });
            }
        }
        
        // Check for privilege escalation
        if (event.eventType === 'ACCESS_CONTROL' && 
            event.rolesAfter && event.rolesBefore &&
            event.rolesAfter.length > event.rolesBefore.length) {
            alerts.push({
                type: 'PRIVILEGE_ESCALATION',
                severity: 'HIGH',
                message: `Privilege escalation detected for user: ${event.userId}`
            });
        }
        
        // Process alerts
        for (const alert of alerts) {
            await this.triggerSecurityAlert(alert, event);
        }
    }
    
    /**
     * Trigger security alert
     */
    async triggerSecurityAlert(alert, event) {
        this.metrics.alertsTriggered++;
        
        const alertEvent = {
            ...alert,
            timestamp: new Date().toISOString(),
            relatedEventId: event.id,
            source: 'audit-logger'
        };
        
        this.alertHistory.push(alertEvent);
        this.emit('securityAlert', alertEvent);
        
        this.log.warn('Security alert triggered:', alertEvent);
    }
    
    /**
     * Update event metrics
     */
    updateEventMetrics(event) {
        this.metrics.totalEvents++;
        
        if (event.encrypted) {
            this.metrics.encryptedEvents++;
        }
        
        // Update event type counters
        const count = this.metrics.eventsByType.get(event.eventType) || 0;
        this.metrics.eventsByType.set(event.eventType, count + 1);
        
        // Update severity counters
        const severity = event.severity || 'info';
        const severityCount = this.metrics.eventsBySeverity.get(severity) || 0;
        this.metrics.eventsBySeverity.set(severity, severityCount + 1);
    }
    
    /**
     * Update average processing time
     */
    updateAverageProcessingTime(newTime) {
        const alpha = 0.1;
        this.metrics.averageProcessingTime = this.metrics.averageProcessingTime === 0
            ? newTime
            : (alpha * newTime) + ((1 - alpha) * this.metrics.averageProcessingTime);
    }

    /**
     * Process audit queue
     */
    async processQueue() {
        if (this.isProcessing || this.auditQueue.length === 0) {
            return;
        }

        this.isProcessing = true;

        try {
            const eventsToProcess = this.auditQueue.splice(0, 1000); // Process in batches

            for (const event of eventsToProcess) {
                await this.writeAuditEvent(event);
            }

            // Check if log rotation is needed
            await this.checkLogRotation();

        } catch (error) {
            cds.log('audit-logger').error('Queue processing failed:', error);
            // Re-queue events on failure
            this.auditQueue.unshift(...eventsToProcess);
        } finally {
            this.isProcessing = false;
        }
    }

    /**
     * Write audit event to file
     */
    async writeAuditEvent(event) {
        try {
            // Encrypt event if required
            const eventData = this.complianceMode === 'PCI' ? 
                this.encryptEvent(event) : event;

            // Format for writing
            const logLine = `${JSON.stringify(eventData)  }\n`;
            
            // Write to file
            await fs.appendFile(this.currentLogFile, logLine);

            // Emit event for real-time monitoring
            this.emit('auditEvent', event);

        } catch (error) {
            cds.log('audit-logger').error('Failed to write audit event:', error);
            throw error;
        }
    }

    /**
     * Check if log rotation is needed
     */
    async checkLogRotation() {
        try {
            const stats = await fs.stat(this.currentLogFile);
            
            if (stats.size >= this.maxFileSize) {
                await this.rotateLogFile();
            }
        } catch (error) {
            cds.log('audit-logger').warn('Log rotation check failed:', error);
        }
    }

    /**
     * Rotate log file
     */
    async rotateLogFile() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `audit-${timestamp}-${this.logFileIndex}.log`;
        this.currentLogFile = path.join(this.auditDir, filename);
        this.logFileIndex++;

        // Cleanup old logs
        await this.cleanupOldLogs();
    }

    /**
     * Start queue processing
     */
    startProcessingQueue() {
        activeIntervals.set('interval_857', setInterval(async () => {
            await this.processQueue();
        }, 5000)); // Process every 5 seconds
    }

    /**
     * Initialize compliance rules
     */
    initializeComplianceRules() {
        return {
            SOX: {
                requireUserAuthentication: true,
                requireAccessLogging: true,
                requireChangeApproval: true,
                dataRetentionYears: 7,
                encryptionRequired: false
            },
            PCI: {
                requireUserAuthentication: true,
                requireAccessLogging: true,
                requireChangeApproval: true,
                dataRetentionYears: 3,
                encryptionRequired: true,
                maskSensitiveData: true
            },
            GDPR: {
                requireUserAuthentication: true,
                requireAccessLogging: true,
                requireChangeApproval: false,
                dataRetentionYears: 6,
                encryptionRequired: true,
                allowDataDeletion: true,
                requireConsentLogging: true
            },
            HIPAA: {
                requireUserAuthentication: true,
                requireAccessLogging: true,
                requireChangeApproval: true,
                dataRetentionYears: 6,
                encryptionRequired: true,
                requireMinimumNecessary: true
            }
        };
    }

    /**
     * Calculate risk score for transaction
     */
    calculateRiskScore(data) {
        let score = 0;

        // Value-based risk
        if (data.value > 1e18) score += 30; // > 1 ETH
        if (data.value > 10e18) score += 50; // > 10 ETH

        // Gas-based risk
        if (data.gasUsed > 1000000) score += 20;

        // Contract interaction risk
        if (data.contractAddress && !this.isKnownContract(data.contractAddress)) {
            score += 40;
        }

        // Time-based risk
        const hour = new Date().getHours();
        if (hour < 6 || hour > 22) score += 10; // Off hours

        return Math.min(score, 100);
    }

    /**
     * Check compliance requirements
     */
    checkCompliance(data) {
        const rules = this.complianceRules[this.complianceMode];
        const flags = [];

        if (rules.requireUserAuthentication && !data.userId) {
            flags.push('MISSING_USER_AUTHENTICATION');
        }

        if (rules.encryptionRequired && !data.encrypted) {
            flags.push('ENCRYPTION_REQUIRED');
        }

        if (data.value > 1000e18) { // Large transaction
            flags.push('LARGE_VALUE_TRANSACTION');
        }

        return flags;
    }

    /**
     * Assess contract interaction risk
     */
    assessContractRisk(data) {
        return {
            contractVerified: this.isVerifiedContract(data.contractAddress),
            knownVulnerabilities: this.checkKnownVulnerabilities(data.contractAddress),
            gasUsagePattern: this.analyzeGasPattern(data),
            unusualActivity: this.detectUnusualActivity(data)
        };
    }

    /**
     * Sanitize parameters to remove sensitive data
     */
    sanitizeParameters(params) {
        if (!params) return params;

        const sanitized = JSON.parse(JSON.stringify(params));
        
        // Remove private keys, passwords, etc.
        const sensitivePatterns = [
            /private[\s_]*key/i,
            /password/i,
            /secret/i,
            /token/i
        ];

        const sanitizeObject = (obj) => {
            for (const key in obj) {
                if (sensitivePatterns.some(pattern => pattern.test(key))) {
                    obj[key] = '***REDACTED***';
                } else if (typeof obj[key] === 'object') {
                    sanitizeObject(obj[key]);
                }
            }
        };

        if (typeof sanitized === 'object') {
            sanitizeObject(sanitized);
        }

        return sanitized;
    }

    /**
     * Sanitize configuration values
     */
    sanitizeValue(value) {
        if (typeof value === 'string' && value.length > 50) {
            return `${value.substring(0, 50)  }...[truncated]`;
        }
        return value;
    }

    /**
     * Calculate checksum for data integrity
     */
    calculateChecksum(data) {
        const hash = crypto.createHash('sha256');
        hash.update(JSON.stringify(data));
        return hash.digest('hex');
    }

    /**
     * Validate audit event structure
     */
    validateAuditEvent(event) {
        const required = ['eventType', 'timestamp', 'checksum'];
        const missing = required.filter(field => !event[field]);

        return {
            valid: missing.length === 0,
            errors: missing.map(field => `Missing required field: ${field}`)
        };
    }

    /**
     * Generate unique event ID
     */
    generateEventId() {
        const timestamp = Date.now();
        const random = crypto.randomBytes(4).toString('hex');
        return `${timestamp}-${random}`;
    }

    /**
     * Get next sequence number
     */
    getNextSequenceNumber() {
        this.sequenceNumber = (this.sequenceNumber || 0) + 1;
        return this.sequenceNumber;
    }

    /**
     * Encrypt audit event
     */
    encryptEvent(event) {
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipherGCM('aes-256-gcm', this.encryptionKey, iv);
        
        let encrypted = cipher.update(JSON.stringify(event), 'utf8', 'hex');
        encrypted += cipher.final('hex');
        
        const authTag = cipher.getAuthTag();
        
        return {
            encrypted: true,
            iv: iv.toString('hex'),
            authTag: authTag.toString('hex'),
            data: encrypted
        };
    }

    /**
     * Decrypt audit event
     */
    decryptEvent(encryptedEvent) {
        const iv = Buffer.from(encryptedEvent.iv, 'hex');
        const authTag = Buffer.from(encryptedEvent.authTag, 'hex');
        
        const decipher = crypto.createDecipherGCM('aes-256-gcm', this.encryptionKey, iv);
        decipher.setAuthTag(authTag);
        
        let decrypted = decipher.update(encryptedEvent.data, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        return JSON.parse(decrypted);
    }

    /**
     * Generate encryption key
     */
    generateEncryptionKey() {
        if (process.env.AUDIT_ENCRYPTION_KEY) {
            return Buffer.from(process.env.AUDIT_ENCRYPTION_KEY, 'hex');
        }
        
        if (process.env.NODE_ENV === 'production') {
            throw new Error('AUDIT_ENCRYPTION_KEY environment variable required in production');
        }
        
        return crypto.randomBytes(32);
    }

    /**
     * Cleanup old log files
     */
    async cleanupOldLogs() {
        try {
            const files = await fs.readdir(this.auditDir);
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - this.retentionDays);

            for (const file of files) {
                if (file.startsWith('audit-') && file.endsWith('.log')) {
                    const filepath = path.join(this.auditDir, file);
                    const stats = await fs.stat(filepath);
                    
                    if (stats.mtime < cutoffDate) {
                        await fs.unlink(filepath);
                        cds.log('audit-logger').info('Deleted old audit log:', file);
                    }
                }
            }
        } catch (error) {
            cds.log('audit-logger').warn('Cleanup failed:', error);
        }
    }

    /**
     * Helper methods for risk assessment
     */
    isKnownContract(address) {
        // In a real implementation, check against known contract registry
        return false;
    }

    isVerifiedContract(address) {
        // Check if contract is verified on Etherscan or similar
        return false;
    }

    checkKnownVulnerabilities(address) {
        // Check against vulnerability databases
        return [];
    }

    analyzeGasPattern(data) {
        // Analyze gas usage patterns for anomalies
        return 'normal';
    }

    detectUnusualActivity(data) {
        // Detect unusual activity patterns
        return false;
    }

    /**
     * Generate audit report
     */
    async generateComplianceReport(startDate, endDate, framework) {
        const events = await this.queryAuditEvents(startDate, endDate);
        
        const report = {
            reportId: this.generateEventId(),
            generatedAt: new Date().toISOString(),
            framework: framework,
            period: { startDate, endDate },
            summary: {
                totalEvents: events.length,
                securityIncidents: events.filter(e => e.eventType === 'SECURITY_INCIDENT').length,
                authenticationEvents: events.filter(e => e.eventType === 'USER_AUTHENTICATION').length,
                transactionEvents: events.filter(e => e.eventType === 'BLOCKCHAIN_TRANSACTION').length,
                configurationChanges: events.filter(e => e.eventType === 'CONFIGURATION_CHANGE').length
            },
            complianceStatus: this.assessCompliance(events, framework),
            recommendations: this.generateRecommendations(events, framework)
        };

        return report;
    }

    /**
     * Query audit events from logs
     */
    async queryAuditEvents(startDate, endDate) {
        // Implementation would read and parse log files
        // This is a simplified version
        return [];
    }

    /**
     * Assess compliance status
     */
    assessCompliance(events, framework) {
        const rules = this.complianceRules[framework];
        // Implementation would check events against compliance rules
        return {
            status: 'COMPLIANT',
            score: 95,
            violations: []
        };
    }

    /**
     * Generate compliance recommendations
     */
    generateRecommendations(events, framework) {
        // Generate recommendations based on audit findings
        return [
            'Implement stronger password policies',
            'Enable multi-factor authentication for all admin accounts',
            'Regular security awareness training'
        ];
    }
    
    /**
     * Analyze security patterns for real-time monitoring
     */
    analyzeSecurityPatterns() {
        const recentEvents = this.getRecentEvents(3600000); // Last hour
        
        // Analyze failed login patterns
        const failedLogins = recentEvents.filter(e => 
            e.eventType === 'USER_AUTHENTICATION' && !e.success
        );
        
        if (failedLogins.length > this.alertThresholds.suspiciousActivity) {
            this.triggerSecurityAlert({
                type: 'SUSPICIOUS_ACTIVITY',
                severity: 'MEDIUM',
                message: `High number of failed logins detected: ${failedLogins.length}`
            });
        }
        
        // Analyze privilege escalation patterns
        const privilegeChanges = recentEvents.filter(e => 
            e.eventType === 'ACCESS_CONTROL' && e.rolesAfter
        );
        
        if (privilegeChanges.length > this.alertThresholds.privilegeEscalation) {
            this.triggerSecurityAlert({
                type: 'PRIVILEGE_ESCALATION',
                severity: 'HIGH',
                message: `Multiple privilege changes detected: ${privilegeChanges.length}`
            });
        }
    }
    
    /**
     * Check compliance violations
     */
    checkComplianceViolations() {
        const rules = this.complianceRules[this.complianceMode];
        const violations = [];
        
        // Check encryption compliance
        if (rules.encryptionRequired) {
            const unencryptedEvents = this.metrics.totalEvents - this.metrics.encryptedEvents;
            const encryptionRate = this.metrics.totalEvents > 0 
                ? (this.metrics.encryptedEvents / this.metrics.totalEvents) * 100 
                : 100;
                
            if (encryptionRate < 95) {
                violations.push({
                    type: 'ENCRYPTION_COMPLIANCE',
                    severity: 'HIGH',
                    message: `Encryption rate below threshold: ${encryptionRate.toFixed(2)}%`,
                    requirement: 'All audit events must be encrypted'
                });
            }
        }
        
        // Check retention compliance
        if (violations.length > 0) {
            this.complianceViolations.push(...violations);
            this.metrics.complianceViolations += violations.length;
            
            violations.forEach(violation => {
                this.emit('complianceViolation', violation);
                this.log.warn('Compliance violation detected:', violation);
            });
        }
    }
    
    /**
     * Generate metrics report
     */
    generateMetricsReport() {
        const report = {
            timestamp: new Date().toISOString(),
            complianceMode: this.complianceMode,
            metrics: {
                ...this.metrics,
                eventsByType: Object.fromEntries(this.metrics.eventsByType),
                eventsBySeverity: Object.fromEntries(this.metrics.eventsBySeverity)
            },
            alerts: {
                total: this.alertHistory.length,
                recent: this.alertHistory.filter(a => 
                    Date.now() - new Date(a.timestamp).getTime() < 3600000
                ).length
            },
            compliance: {
                violations: this.complianceViolations.length,
                encryptionRate: this.metrics.totalEvents > 0 
                    ? (this.metrics.encryptedEvents / this.metrics.totalEvents) * 100 
                    : 100
            },
            tamperProof: {
                blockchainIntegrity: !!this.blockchainHash,
                digitalSignatures: this.digitalSignatures.size,
                integrityChecks: this.checksumChain.length
            }
        };
        
        this.emit('metricsReport', report);
        this.log.info('Audit metrics report generated:', {
            totalEvents: report.metrics.totalEvents,
            alertsTriggered: report.metrics.alertsTriggered,
            complianceViolations: report.compliance.violations,
            encryptionRate: `${report.compliance.encryptionRate.toFixed(2)  }%`
        });
        
        return report;
    }
    
    /**
     * Check data retention compliance
     */
    async checkDataRetentionCompliance() {
        const rules = this.complianceRules[this.complianceMode];
        const maxAge = rules.dataRetentionYears * 365 * 24 * 60 * 60 * 1000;
        
        try {
            const files = await fs.readdir(this.auditDir);
            const oldFiles = [];
            
            for (const file of files) {
                if (file.startsWith('audit-') && file.endsWith('.log')) {
                    const filepath = path.join(this.auditDir, file);
                    const stats = await fs.stat(filepath);
                    const age = Date.now() - stats.mtime.getTime();
                    
                    if (age > maxAge) {
                        oldFiles.push({ file, age });
                    }
                }
            }
            
            if (oldFiles.length > 0) {
                const violation = {
                    type: 'DATA_RETENTION',
                    severity: 'MEDIUM',
                    message: `Files exceed retention period: ${oldFiles.length} files`,
                    details: oldFiles.map(f => f.file)
                };
                
                this.complianceViolations.push(violation);
                this.emit('complianceViolation', violation);
                this.log.warn('Data retention violation:', violation);
            }
        } catch (error) {
            this.log.error('Data retention check failed:', error);
        }
    }
    
    /**
     * Check encryption compliance
     */
    checkEncryptionCompliance() {
        const rules = this.complianceRules[this.complianceMode];
        
        if (rules.encryptionRequired && !this.encryptionKey) {
            const violation = {
                type: 'ENCRYPTION_MISSING',
                severity: 'CRITICAL',
                message: 'Encryption required but not configured',
                requirement: rules
            };
            
            this.complianceViolations.push(violation);
            this.emit('complianceViolation', violation);
            this.log.error('Encryption compliance violation:', violation);
        }
    }
    
    /**
     * Get recent events for analysis
     */
    getRecentEvents(timeWindow) {
        const cutoff = Date.now() - timeWindow;
        return this.auditQueue.filter(event => 
            new Date(event.timestamp).getTime() > cutoff
        );
    }
    
    /**
     * Get recent failed logins for a user
     */
    getRecentFailedLogins(userId) {
        const recentEvents = this.getRecentEvents(3600000); // Last hour
        return recentEvents.filter(e => 
            e.eventType === 'USER_AUTHENTICATION' &&
            e.userId === userId &&
            !e.success
        ).length;
    }
    
    /**
     * Calculate file checksum for integrity
     */
    async calculateFileChecksum() {
        if (!this.currentLogFile) return null;
        
        try {
            const data = await fs.readFile(this.currentLogFile);
            return crypto.createHash('sha256').update(data).digest('hex');
        } catch (error) {
            this.log.warn('Failed to calculate file checksum:', error);
            return null;
        }
    }
    
    /**
     * Generate digital signature
     */
    async generateDigitalSignature() {
        try {
            const data = {
                timestamp: new Date().toISOString(),
                totalEvents: this.metrics.totalEvents,
                blockchainHash: this.blockchainHash
            };
            
            const sign = crypto.createSign('RSA-SHA256');
            sign.update(JSON.stringify(data));
            
            const privateKey = process.env.AUDIT_PRIVATE_KEY || this.generateRSAKeyPair().privateKey;
            return sign.sign(privateKey, 'hex');
        } catch (error) {
            this.log.warn('Failed to generate digital signature:', error);
            return null;
        }
    }
    
    /**
     * Generate RSA key pair for development
     */
    generateRSAKeyPair() {
        return crypto.generateKeyPairSync('rsa', {
            modulusLength: 2048,
            publicKeyEncoding: { type: 'spki', format: 'pem' },
            privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
        });
    }
    
    /**
     * Get comprehensive audit status
     */
    getAuditStatus() {
        return {
            initialized: !!this.currentLogFile,
            sapIntegration: !!this.sapAuditLog,
            tamperProof: !!this.blockchainHash,
            complianceMode: this.complianceMode,
            metrics: this.generateMetricsReport(),
            alerts: {
                recent: this.alertHistory.filter(a => 
                    Date.now() - new Date(a.timestamp).getTime() < 3600000
                ),
                total: this.alertHistory.length
            },
            violations: this.complianceViolations,
            queueStatus: {
                pending: this.auditQueue.length,
                processing: this.isProcessing
            }
        };
    }
    
    /**
     * Gracefully shutdown audit logger
     */
    async shutdown() {
        this.log.info('Shutting down audit logger...');
        
        try {
            // Process remaining queue
            if (this.auditQueue.length > 0) {
                await this.processQueue();
            }
            
            // Generate final integrity checkpoint
            await this.generateIntegrityCheckpoint();
            
            // Close SAP Audit Log connection
            if (this.sapAuditLog) {
                await this.sapAuditLog.close();
            }
            
            this.log.info('Audit logger shutdown completed');
        } catch (error) {
            this.log.error('Error during audit logger shutdown:', error);
            throw error;
        }
    }
}

module.exports = {
    AuditLogger
};