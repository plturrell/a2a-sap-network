using { securityAudit } from '../db/securityAudit';
using { managed } from '@sap/cds/common';

/**
 * SAP CAP Security Monitoring Service
 * Provides OData V4 endpoints for SAP Fiori Elements security dashboard
 * Follows SAP CAP and Fiori design patterns
 */
service SecurityMonitoringService @(path: '/security') {

    /**
     * Security Events - Read-only analytical view
     * Used for security event analysis and reporting
     */
    @readonly
    entity SecurityEvents as projection on securityAudit.SecurityEvents {
        *,
        case severity
            when 'critical' then 4
            when 'high' then 3  
            when 'medium' then 2
            when 'low' then 1
            else 0
        end as severityLevel : Integer,
        
        case status
            when 'BLOCKED' then 'Error'
            when 'ALLOWED' then 'Success' 
            when 'REVIEWED' then 'Information'
            else 'Warning'
        end as criticality : String(20)
    }
    
    // SAP Fiori annotations for analytical view
    annotate SecurityEvents with @(
        Analytics.AggregatedProperties : [
            {
                Name : 'TotalEvents',
                AggregationMethod : 'countdistinct',
                AggregatableProperty : ID,
                ![@Common.Label] : 'Total Events'
            },
            {
                Name : 'AvgThreatScore', 
                AggregationMethod : 'average',
                AggregatableProperty : threatScore,
                ![@Common.Label] : 'Average Threat Score'
            }
        ],
        Analytics.AggregatedProperty #TotalEvents : {
            ![@Common.Label] : 'Total Events'
        },
        Analytics.AggregatedProperty #AvgThreatScore : {
            ![@Common.Label] : 'Average Threat Score'
        }
    );

    /**
     * Security Alerts - Main entity for alert management
     * Supports full CRUD operations for security personnel
     */
    entity SecurityAlerts as projection on securityAudit.SecurityAlerts {
        *,
        case priority
            when 1 then 'Error'      // Critical
            when 2 then 'Warning'    // High  
            when 3 then 'Success'    // Medium
            when 4 then 'Information' // Low
            else 'None'
        end as priorityCriticality : String(20),
        
        case status
            when 'ACTIVE' then 'Error'
            when 'ACKNOWLEDGED' then 'Warning'
            when 'RESOLVED' then 'Success'
            when 'EXPIRED' then 'Information'
            else 'None'  
        end as statusCriticality : String(20),
        
        // Calculate time to resolution in hours
        case 
            when resolvedAt is not null and detectedAt is not null 
            then cast(seconds_between(detectedAt, resolvedAt) / 3600.0 as Decimal(10,2))
            else null
        end as resolutionTimeHours : Decimal(10,2),
        
        // Days since detection
        cast(seconds_between(detectedAt, $now) / 86400.0 as Integer) as daysSinceDetection : Integer,
        
        // Computed field for SLA breach
        case status
            when 'ACTIVE' then
                case priority
                    when 1 then seconds_between(detectedAt, $now) > 3600   // 1 hour for critical
                    when 2 then seconds_between(detectedAt, $now) > 14400  // 4 hours for high
                    when 3 then seconds_between(detectedAt, $now) > 86400  // 24 hours for medium
                    else false
                end
            else false
        end as slaBreached : Boolean
    }
    
    // Add actions for alert management
    actions {
        @Common.SideEffects.TargetProperties : ['status', 'acknowledgedBy', 'acknowledgedAt']
        action acknowledgeAlert(alertId: UUID, comment: String(500)) returns String;
        
        @Common.SideEffects.TargetProperties : ['status', 'resolvedBy', 'resolvedAt', 'resolution']  
        action resolveAlert(alertId: UUID, resolution: String(1000)) returns String;
        
        @Common.SideEffects.TargetEntities : ['SecurityAlerts']
        action escalateAlert(alertId: UUID, escalationReason: String(500)) returns String;
    }

    /**
     * Security Metrics - KPI dashboard data
     * Analytical view for executive dashboards
     */
    @readonly  
    entity SecurityMetrics as projection on securityAudit.SecurityMetrics {
        *,
        case complianceScore
            when >= 95.0 then 'Success'
            when >= 85.0 then 'Warning'
            else 'Error' 
        end as complianceCriticality : String(20),
        
        totalEvents + criticalEvents + highEvents as totalSecurityEvents : Integer,
        
        case avgThreatScore
            when >= 70.0 then 'Error'      // High threat
            when >= 40.0 then 'Warning'    // Medium threat  
            when >= 20.0 then 'Success'    // Low threat
            else 'Information'             // Minimal threat
        end as threatLevelCriticality : String(20)
    };

    /**
     * Blocked IPs - IP address management
     * For security team to manage blocked addresses
     */
    entity BlockedIPs as projection on securityAudit.BlockedIPs {
        *,
        case blockType
            when 'PERMANENT' then 'Error'
            when 'TEMPORARY' then 'Warning' 
            when 'CONDITIONAL' then 'Information'
            else 'None'
        end as blockTypeCriticality : String(20),
        
        // Time remaining until unblock (in hours)
        case 
            when expiresAt is not null and isActive = true
            then cast(seconds_between($now, expiresAt) / 3600.0 as Decimal(10,2))
            else null
        end as hoursUntilUnblock : Decimal(10,2),
        
        // Days since blocked
        cast(seconds_between(blockedAt, $now) / 86400.0 as Integer) as daysSinceBlocked : Integer
    }
    
    // Actions for IP management
    actions {
        @Common.SideEffects.TargetProperties : ['isActive', 'unblockReason', 'reviewedBy', 'reviewedAt']
        action unblockIP(ipId: UUID, reason: String(500)) returns String;
        
        @Common.SideEffects.TargetProperties : ['expiresAt']
        action extendBlock(ipId: UUID, additionalHours: Integer) returns String;
    }

    /**
     * User Security Profiles - User risk management
     * For monitoring user security status
     */
    @readonly
    entity UserSecurityProfiles as projection on securityAudit.UserSecurityProfile {
        *,
        case riskScore
            when >= 80 then 'Error'      // High risk
            when >= 60 then 'Warning'    // Medium risk
            when >= 40 then 'Success'    // Low risk  
            else 'Information'           // Minimal risk
        end as riskCriticality : String(20),
        
        case trustScore
            when >= 80 then 'Success'    // High trust
            when >= 60 then 'Information' // Medium trust
            when >= 40 then 'Warning'    // Low trust
            else 'Error'                 // Very low trust
        end as trustCriticality : String(20),
        
        case monitoringLevel
            when 'HIGH' then 'Error'
            when 'MEDIUM' then 'Warning'
            when 'LOW' then 'Information'
            else 'None'
        end as monitoringCriticality : String(20)
    };

    /**
     * Security Dashboard - Executive summary view
     * Virtual entity for dashboard widgets
     */
    @readonly
    entity SecurityDashboard {
        key ID: UUID;
        lastUpdated: Timestamp;
        
        // Current status
        totalActiveAlerts: Integer;
        criticalAlerts: Integer;
        highPriorityAlerts: Integer;
        slaBreachCount: Integer;
        
        // Recent activity (last 24 hours)
        newEventsToday: Integer;
        blockedAttacksToday: Integer;
        newAlertsToday: Integer;
        resolvedAlertsToday: Integer;
        
        // Security posture
        overallRiskScore: Decimal(5,2);
        complianceScore: Decimal(5,2);
        systemHealthScore: Decimal(5,2);
        
        // Infrastructure status
        blockedIPsCount: Integer;
        quarantinedUsersCount: Integer;
        monitoredUsersCount: Integer;
        
        // Performance metrics
        avgResponseTime: Integer;
        systemAvailability: Decimal(5,2);
        successRate: Decimal(5,2);
        
        // Trend indicators
        riskTrend: String(10);      // UP, DOWN, STABLE
        alertTrend: String(10);     // UP, DOWN, STABLE
        complianceTrend: String(10); // UP, DOWN, STABLE
    }

    /**
     * Security Event Types - Master data for event classification
     */
    @readonly
    entity SecurityEventTypes {
        key eventType: String(100);
        category: String(50);
        defaultSeverity: String(20);
        description: String(500);
        riskScore: Integer;
        autoBlock: Boolean;
        requiresReview: Boolean;
    }

    // Functions for calculations and analytics
    function getSecurityTrends(
        period: String(20),      // DAILY, WEEKLY, MONTHLY
        metricType: String(50)   // EVENTS, ALERTS, COMPLIANCE
    ) returns {
        period: String(20);
        values: array of {
            timestamp: Timestamp;
            value: Decimal(10,2);
            change: Decimal(5,2);  // Percentage change
        }
    };

    function calculateRiskScore(
        userId: String(255),
        includeHistorical: Boolean
    ) returns {
        userId: String(255);
        currentRiskScore: Integer;
        riskFactors: array of {
            factor: String(100);
            weight: Decimal(5,2);
            contribution: Decimal(5,2);
        }
    };

    function getComplianceReport(
        startDate: Date,
        endDate: Date,
        framework: String(50)    // SOX, GDPR, ISO27001, etc.
    ) returns {
        framework: String(50);
        overallScore: Decimal(5,2);
        requirements: array of {
            requirement: String(200);
            status: String(20);     // COMPLIANT, NON_COMPLIANT, PARTIAL
            evidence: String(500);
            lastVerified: Timestamp;
        }
    };
}