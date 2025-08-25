/**
 * @fileoverview Monitoring Service - CDS Definition
 * @since 1.0.0
 * @module monitoringService
 * 
 * CAP service definition for performance monitoring and auto-scaling
 * Provides comprehensive system monitoring and scaling management
 */

using { cuid, managed } from '@sap/cds/common';

namespace com.sap.a2a.services;

/**
 * MonitoringService - Performance monitoring and auto-scaling management
 * Provides real-time system metrics, alerting, and automatic scaling capabilities
 */
@path: '/api/v1/monitoring'
@requires: 'authenticated-user'
service MonitoringService {
    
    /**
     * Functions for retrieving monitoring data
     */
    
    /**
     * Get current system metrics and status
     */
    function getSystemMetrics() returns {
        timestamp     : Timestamp;
        system        : {
            cpu       : {
                usage     : Decimal(5,2);
                loadAverage : array of Decimal(10,2);
                cores     : Integer;
            };
            memory    : {
                total     : Integer64;
                free      : Integer64;
                used      : Integer64;
                usagePercentage : Integer;
            };
            disk      : {
                size      : String;
                used      : String;
                available : String;
                usagePercentage : Integer;
            };
            uptime    : Integer64;
        };
        agents        : {
            total     : Integer;
            healthy   : Integer;
            unhealthy : Integer;
            averages  : {
                cpuUsage      : Decimal(5,2);
                memoryUsage   : Decimal(5,2);
                responseTime  : Integer;
                successRate   : Decimal(5,2);
            };
        };
        scaling       : {
            enabled           : Boolean;
            currentInstances  : Integer;
            targetInstances   : Integer;
            scalingInProgress : Boolean;
            cooldownRemaining : Integer;
            limits            : {
                min           : Integer;
                max           : Integer;
            };
        };
    };
    
    /**
     * Get performance metrics history
     */
    function getPerformanceHistory(
        category : String,
        limit    : Integer,
        offset   : Integer
    ) returns {
        category : String;
        metrics  : array of {
            timestamp : Timestamp;
            data      : String; // JSON data
        };
        total    : Integer;
        limit    : Integer;
        offset   : Integer;
    };
    
    /**
     * Get system alerts with filtering
     */
    function getAlerts(
        severity       : String,
        category       : String,
        unacknowledged : Boolean,
        limit          : Integer
    ) returns {
        alerts         : array of {
            id          : String;
            timestamp   : Timestamp;
            category    : String;
            severity    : String;
            message     : String;
            metadata    : String; // JSON metadata
            acknowledged : Boolean;
            resolved    : Boolean;
        };
        total          : Integer;
        unacknowledged : Integer;
        critical       : Integer;
    };
    
    /**
     * Get scaling history and statistics
     */
    function getScalingHistory(
        limit : Integer
    ) returns {
        history : array of {
            id            : String;
            timestamp     : Timestamp;
            action        : String;
            reason        : String;
            fromInstances : Integer;
            toInstances   : Integer;
            status        : String;
            duration      : Integer;
        };
        stats   : {
            total       : Integer;
            successful  : Integer;
            failed      : Integer;
            successRate : Integer;
            scaleUps    : Integer;
            scaleDowns  : Integer;
            avgDuration : Integer;
        };
        current : {
            enabled           : Boolean;
            currentInstances  : Integer;
            scalingInProgress : Boolean;
            cooldownRemaining : Integer;
        };
    };
    
    /**
     * Get comprehensive monitoring dashboard data
     */
    function getMonitoringDashboard() returns {
        timestamp : Timestamp;
        overview  : {
            systemHealth  : {
                status    : String;
                score     : Integer;
                issues    : array of String;
            };
            instances     : Integer;
            activeAlerts  : Integer;
            scalingEvents : Integer;
            uptime        : Integer64;
        };
        metrics   : {
            system  : {
                cpu     : {
                    usage : Decimal(5,2);
                    cores : Integer;
                };
                memory  : {
                    usagePercentage : Integer;
                    total : Integer64;
                    free  : Integer64;
                };
            };
            agents  : {
                total   : Integer;
                healthy : Integer;
                averages : {
                    responseTime : Integer;
                    successRate  : Decimal(5,2);
                };
            };
            scaling : {
                enabled           : Boolean;
                currentInstances  : Integer;
                scalingInProgress : Boolean;
            };
        };
        alerts    : array of {
            id        : String;
            severity  : String;
            message   : String;
            timestamp : Timestamp;
        };
        scaling   : {
            status       : {
                enabled           : Boolean;
                currentInstances  : Integer;
                cooldownRemaining : Integer;
            };
            stats        : {
                total       : Integer;
                successRate : Integer;
            };
            recentEvents : array of {
                id      : String;
                action  : String;
                status  : String;
                timestamp : Timestamp;
            };
        };
        performance : {
            trends         : {
                cpu        : String;
                memory     : String;
                agentHealth : String;
            };
            recommendations : array of {
                type        : String;
                priority    : String;
                title       : String;
                description : String;
            };
        };
    };
    
    /**
     * Get real-time metrics stream information
     */
    function getMetricsStream() returns {
        timestamp   : Timestamp;
        metrics     : String; // JSON metrics data
        monitoring  : {
            isMonitoring    : Boolean;
            totalDataPoints : Integer;
        };
        streamInfo  : {
            available : Boolean;
            interval  : String;
            endpoint  : String;
        };
    };
    
    /**
     * Generate comprehensive monitoring report
     */
    function generateReport(
        timeRange      : String,
        includeDetails : Boolean
    ) returns {
        generated : Timestamp;
        timeRange : String;
        summary   : {
            systemHealth  : {
                status : String;
                score  : Integer;
            };
            totalAlerts   : Integer;
            scalingEvents : {
                total       : Integer;
                successRate : Integer;
            };
            uptime        : Integer64;
        };
        details   : {
            alerts          : array of String; // JSON alert data
            scalingHistory  : array of String; // JSON scaling events
            performanceMetrics : array of String; // JSON metrics
        };
    };
    
    /**
     * Actions for managing monitoring and scaling
     */
    
    /**
     * Acknowledge a system alert
     */
    action acknowledgeAlert(
        alertId : String
    ) returns {
        success : Boolean;
        message : String;
        alert   : {
            id             : String;
            message        : String;
            acknowledged   : Boolean;
            acknowledgedAt : Timestamp;
        };
    };
    
    /**
     * Resolve a system alert
     */
    action resolveAlert(
        alertId    : String,
        resolution : String
    ) returns {
        success : Boolean;
        message : String;
        alert   : {
            id         : String;
            message    : String;
            resolved   : Boolean;
            resolvedAt : Timestamp;
            resolution : String;
        };
    };
    
    /**
     * Manually trigger scaling operation
     */
    action manualScale(
        targetInstances : Integer,
        reason          : String
    ) returns {
        success      : Boolean;
        message      : String;
        scalingEvent : {
            id            : String;
            action        : String;
            fromInstances : Integer;
            toInstances   : Integer;
            status        : String;
        };
    };
    
    /**
     * Emergency scaling (bypasses safety limits)
     */
    @requires: 'admin'
    action emergencyScale(
        targetInstances : Integer,
        justification   : String
    ) returns {
        success      : Boolean;
        message      : String;
        scalingEvent : {
            id            : String;
            action        : String;
            fromInstances : Integer;
            toInstances   : Integer;
            status        : String;
        };
        warning      : String;
    };
    
    /**
     * Update auto-scaling configuration
     */
    @requires: 'admin'
    action updateAutoScalingConfig(
        config : {
            minInstances       : Integer;
            maxInstances       : Integer;
            scaleUpThreshold   : {
                cpu            : Decimal(5,2);
                memory         : Decimal(5,2);
                responseTime   : Integer;
            };
            scaleDownThreshold : {
                cpu            : Decimal(5,2);
                memory         : Decimal(5,2);
                responseTime   : Integer;
            };
            cooldownPeriod     : Integer;
            enableScaleDown    : Boolean;
        }
    ) returns {
        success : Boolean;
        message : String;
        config  : {
            enabled           : Boolean;
            currentInstances  : Integer;
            limits            : {
                min           : Integer;
                max           : Integer;
            };
        };
    };
    
    /**
     * Enable or disable auto-scaling
     */
    action toggleAutoScaling(
        enabled : Boolean
    ) returns {
        success : Boolean;
        message : String;
        enabled : Boolean;
    };
    
    /**
     * Bulk alert operations
     */
    action bulkAlertOperation(
        alertIds  : array of String,
        operation : String, // 'acknowledge', 'resolve', 'delete'
        data      : String  // JSON data for operation (e.g., resolution text)
    ) returns {
        success   : Boolean;
        processed : Integer;
        failed    : Integer;
        results   : array of {
            alertId : String;
            success : Boolean;
            message : String;
        };
    };
    
    /**
     * Test scaling configuration
     */
    action testScalingConfig(
        config : String // JSON configuration
    ) returns {
        success     : Boolean;
        validation  : {
            valid   : Boolean;
            errors  : array of String;
            warnings : array of String;
        };
        simulation  : {
            wouldScale      : Boolean;
            targetInstances : Integer;
            reason          : String;
        };
    };
    
    /**
     * Export monitoring configuration
     */
    action exportConfiguration() returns {
        success       : Boolean;
        configuration : String; // JSON export
        timestamp     : Timestamp;
    };
    
    /**
     * Import monitoring configuration
     */
    @requires: 'admin'
    action importConfiguration(
        configuration : String,
        overwrite     : Boolean
    ) returns {
        success : Boolean;
        imported : Integer;
        skipped  : Integer;
        errors   : array of String;
    };
}

/**
 * Type definitions for monitoring service
 */

/**
 * System health status levels
 */
type SystemHealthStatus : String enum {
    healthy;
    warning;
    degraded;
    critical;
    unknown;
}

/**
 * Alert severity levels
 */
type AlertSeverity : String enum {
    info;
    warning;
    error;
    critical;
}

/**
 * Alert categories
 */
type AlertCategory : String enum {
    system;
    agents;
    network;
    performance;
    security;
    scaling;
    threshold;
}

/**
 * Scaling actions
 */
type ScalingAction : String enum {
    scale_up;
    scale_down;
    none;
}

/**
 * Scaling status
 */
type ScalingStatus : String enum {
    initiated;
    in_progress;
    completed;
    failed;
    cancelled;
}

/**
 * Performance trend directions
 */
type TrendDirection : String enum {
    improving;
    degrading;
    stable;
    increasing;
    decreasing;
    unknown;
}