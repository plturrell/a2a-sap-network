using { managed, cuid } from '@sap/cds/common';

namespace securityAudit;

/**
 * Security Events Log
 * Stores all security-related events for audit and compliance
 */
entity SecurityEvents : managed, cuid {
  eventType        : String(100) not null;    // SQL_INJECTION, XSS_ATTEMPT, etc.
  severity         : String(20) not null;     // critical, high, medium, low
  category         : String(50) not null;     // INJECTION, AUTHENTICATION, etc.
  description      : String(500) not null;
  source           : String(100);             // request-monitor, auth-middleware, etc.
  ipAddress        : String(45);              // IPv4 or IPv6 address
  userId           : String(255);             // User ID if authenticated
  userAgent        : String(500);             // Browser/client user agent
  endpoint         : String(500);             // Requested endpoint/URL
  requestMethod    : String(10);              // GET, POST, PUT, DELETE, etc.
  statusCode       : Integer;                 // HTTP status code
  responseTime     : Integer;                 // Response time in milliseconds
  threatScore      : Integer;                 // Calculated threat score 0-100
  blocked          : Boolean default false;   // Whether the request was blocked
  metadata         : String(2000);           // JSON metadata about the event
  sessionId        : String(255);            // Session identifier
  correlationId    : String(255);            // Request correlation ID
  geolocation      : String(255);            // Geographic location if available
  
  // Additional audit fields
  reportedBy       : String(255);            // System/service that reported the event
  processedAt      : Timestamp;              // When the event was processed
  acknowledged     : Boolean default false;   // Whether the event was acknowledged
  acknowledgedBy   : String(255);            // Who acknowledged the event
  acknowledgedAt   : Timestamp;              // When it was acknowledged
}

/**
 * Security Alerts
 * High-level security alerts generated from event patterns
 */
entity SecurityAlerts : managed, cuid {
  alertType        : String(100) not null;   // COORDINATED_ATTACK, BRUTE_FORCE, etc.
  severity         : String(20) not null;    // critical, high, medium, low
  title            : String(200) not null;
  description      : String(1000) not null;
  status           : String(20) default 'ACTIVE';  // ACTIVE, ACKNOWLEDGED, RESOLVED, EXPIRED
  priority         : Integer default 3;       // 1=highest, 5=lowest
  
  // Event aggregation info
  triggerEventId   : String(36);             // ID of the event that triggered this alert
  relatedEvents    : Integer default 1;      // Number of related events
  firstEventAt     : Timestamp;              // When the first related event occurred
  lastEventAt      : Timestamp;              // When the last related event occurred
  
  // Source information
  sourceIpAddress  : String(45);             // Primary IP address involved
  affectedUsers    : String(1000);           // Comma-separated list of affected users
  affectedEndpoints: String(1000);           // Comma-separated list of affected endpoints
  
  // Alert lifecycle
  detectedAt       : Timestamp not null;     // When the alert was first detected
  acknowledgedBy   : String(255);            // Who acknowledged the alert
  acknowledgedAt   : Timestamp;              // When it was acknowledged
  resolvedBy       : String(255);            // Who resolved the alert
  resolvedAt       : Timestamp;              // When it was resolved
  resolution       : String(1000);           // Resolution description
  
  // Metrics
  riskScore        : Integer;                 // Calculated risk score 0-100
  impactScore      : Integer;                 // Impact assessment score 0-100
  confidenceScore  : Integer;                 // Confidence in the alert 0-100
  
  // Response actions
  actionsCompleted : String(1000);           // JSON array of completed actions
  recommendedActions: String(1000);          // JSON array of recommended actions
  
  // External references
  ticketId         : String(100);            // External ticket system ID
  runbookUrl       : String(500);            // Link to incident response runbook
}

/**
 * Security Metrics Aggregation
 * Periodic aggregation of security metrics for reporting
 */
entity SecurityMetrics : cuid {
  periodStart      : Timestamp not null;     // Start of the measurement period
  periodEnd        : Timestamp not null;     // End of the measurement period  
  periodType       : String(20) not null;    // HOURLY, DAILY, WEEKLY, MONTHLY
  
  // Event counts by severity
  criticalEvents   : Integer default 0;
  highEvents       : Integer default 0;
  mediumEvents     : Integer default 0;
  lowEvents        : Integer default 0;
  totalEvents      : Integer default 0;
  
  // Event counts by category
  injectionAttempts: Integer default 0;
  authFailures     : Integer default 0;
  authzFailures    : Integer default 0;
  rateLimitHits    : Integer default 0;
  systemErrors     : Integer default 0;
  
  // Alert metrics
  totalAlerts      : Integer default 0;
  activeAlerts     : Integer default 0;
  resolvedAlerts   : Integer default 0;
  avgResolutionTime: Integer default 0;      // Average resolution time in minutes
  
  // Security posture metrics
  blockedIPs       : Integer default 0;
  quarantinedUsers : Integer default 0;
  avgThreatScore   : Decimal(5,2) default 0.0;
  avgResponseTime  : Integer default 0;      // Average API response time in ms
  successRate      : Decimal(5,2) default 100.0;  // % of successful requests
  
  // Compliance metrics
  auditLogEntries  : Integer default 0;
  complianceScore  : Decimal(5,2) default 100.0;
  policyViolations : Integer default 0;
  
  // Additional metadata
  dataPoints       : Integer default 0;      // Number of data points in this aggregation
  calculatedAt     : Timestamp not null;     // When these metrics were calculated
  version          : String(10) default '1.0';  // Schema version for backward compatibility
}

/**
 * Security Configuration
 * Stores security policy and configuration settings
 */
entity SecurityConfig : managed, cuid {
  configType       : String(50) not null;    // THRESHOLD, POLICY, RULE, etc.
  configName       : String(100) not null;   // Unique name for this configuration
  configValue      : String(2000) not null;  // JSON configuration value
  description      : String(500);            // Description of what this config does
  isActive         : Boolean default true;   // Whether this config is active
  priority         : Integer default 100;    // Priority order for processing
  
  // Validation
  validFrom        : Timestamp;              // When this config becomes valid
  validUntil       : Timestamp;              // When this config expires
  
  // Audit trail
  lastModifiedBy   : String(255);            // Who last modified this config
  version          : Integer default 1;      // Version number for this config
  previousValue    : String(2000);           // Previous configuration value
  changeReason     : String(500);            // Reason for the change
}

/**
 * Blocked IPs
 * Tracks IP addresses that have been blocked for security reasons
 */
entity BlockedIPs : managed, cuid {
  ipAddress        : String(45) not null;    // The blocked IP address
  reason           : String(200) not null;   // Reason for blocking
  severity         : String(20) not null;    // Severity level that triggered the block
  blockType        : String(20) not null;    // TEMPORARY, PERMANENT, CONDITIONAL
  
  // Block duration
  blockedAt        : Timestamp not null;     // When the IP was blocked
  expiresAt        : Timestamp;              // When the block expires (if temporary)
  
  // Related information
  triggerEventId   : String(36);             // Event that caused this block
  relatedAlertId   : String(36);             // Alert associated with this block
  attemptCount     : Integer default 1;      // Number of malicious attempts
  
  // Geographic and network info
  country          : String(2);              // ISO country code
  organization     : String(200);            // ISP/Organization name
  asn              : String(20);              // Autonomous System Number
  
  // Status tracking
  isActive         : Boolean default true;   // Whether the block is currently active
  unblockRequested : Boolean default false;  // Whether unblock has been requested
  unblockReason    : String(500);            // Reason for unblock request
  reviewedBy       : String(255);            // Who reviewed the block
  reviewedAt       : Timestamp;              // When it was reviewed
}

/**
 * User Security Profile
 * Tracks security-related information for each user
 */
entity UserSecurityProfile : managed, cuid {
  userId           : String(255) not null;   // User identifier
  riskScore        : Integer default 0;      // Current risk score 0-100
  trustScore       : Integer default 100;    // Trust score 0-100 (higher is better)
  
  // Authentication metrics
  loginAttempts    : Integer default 0;      // Recent login attempts
  failedLogins     : Integer default 0;      // Recent failed login attempts
  lastSuccessfulLogin: Timestamp;            // Last successful login
  lastFailedLogin  : Timestamp;              // Last failed login attempt
  
  // Behavioral analysis
  suspiciousActivity: Integer default 0;     // Count of suspicious activities
  anomalyScore     : Decimal(5,2) default 0.0;  // Behavioral anomaly score
  baselineEstablished: Boolean default false;    // Whether baseline behavior is established
  
  // Security events
  securityEvents   : Integer default 0;      // Number of security events associated
  lastSecurityEvent: Timestamp;              // Last security event timestamp
  
  // Account security
  isQuarantined    : Boolean default false;  // Whether user is quarantined
  quarantineReason : String(200);            // Reason for quarantine
  quarantinedAt    : Timestamp;              // When user was quarantined
  quarantineExpires: Timestamp;              // When quarantine expires
  
  // Monitoring flags
  requiresMonitoring: Boolean default false; // Whether user requires enhanced monitoring
  monitoringReason : String(200);            // Reason for enhanced monitoring
  monitoringLevel  : String(20);             // LOW, MEDIUM, HIGH monitoring level
  
  // Metadata
  lastAnalyzed     : Timestamp;              // When profile was last analyzed
  profileVersion   : String(10) default '1.0';  // Profile schema version
}