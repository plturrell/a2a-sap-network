using { a2a.network as db } from '../db/schema';

/**
 * Operations Service
 * Provides monitoring, health checks, and operational endpoints
 * following SAP standards
 */
service OperationsService @(
    path: '/ops',
    requires: 'authenticated-user'
) {

    /**
     * Health check endpoint
     * Returns overall system health status
     */
    @readonly
    entity Health {
        key ID: UUID;
        status: String enum { healthy; degraded; unhealthy };
        score: Integer;
        timestamp: Timestamp;
        components: Composition of many HealthComponent on components.health = $self;
        issues: array of String;
        metrics: {
            cpu: Decimal;
            memory: Decimal;
            responseTime: Integer;
            activeAlerts: Integer;
        }
    }

    /**
     * Component health status
     */
    entity HealthComponent {
        key component: String;
        health: Association to Health;
        status: String enum { healthy; degraded; unhealthy; unknown };
        lastCheck: Timestamp;
        details: LargeString;
    }

    /**
     * System metrics
     */
    @readonly
    entity Metrics {
        key name: String;
        value: Decimal;
        unit: String;
        timestamp: Timestamp;
        tags: LargeString; // JSON format for flexible tags
    }

    /**
     * Active alerts
     */
    entity Alerts {
        key ID: UUID;
        name: String;
        severity: String enum { low; medium; high; critical };
        status: String enum { open; acknowledged; resolved };
        message: String;
        timestamp: Timestamp;
        acknowledgedBy: String;
        acknowledgedAt: Timestamp;
        resolvedBy: String;
        resolvedAt: Timestamp;
        metric: {
            name: String;
            value: Decimal;
            threshold: Decimal;
        }
    }

    /**
     * Application logs
     */
    @readonly
    entity Logs {
        key ID: UUID;
        timestamp: Timestamp;
        level: String enum { DEBUG; INFO; WARN; ERROR; FATAL };
        logger: String;
        message: LargeString;
        correlationId: String;
        tenant: String;
        user: String;
        details: LargeString;
    }

    /**
     * Performance traces
     */
    @readonly
    entity Traces {
        key traceId: String;
        spanId: String;
        parentSpanId: String;
        operationName: String;
        serviceName: String;
        startTime: Timestamp;
        endTime: Timestamp;
        duration: Integer; // milliseconds
        status: String enum { ok; error; timeout };
        tags: LargeString; // JSON format for flexible tags
    }

    /**
     * Configuration settings
     */
    entity Configuration {
        key name: String;
        value: LargeString;
        type: String enum { string; number; boolean; json };
        category: String;
        description: String;
        lastModified: Timestamp;
        modifiedBy: String;
    }

    /**
     * Get current health status
     */
    function getHealth() returns OperationsService.Health;

    /**
     * Get metrics for a time range
     */
    function getMetrics(
        @mandatory startTime: Timestamp,
        @mandatory endTime: Timestamp,
        metricNames: array of String,
        tags: LargeString // JSON format for flexible tags
    ) returns array of OperationsService.Metrics;

    /**
     * Get application logs
     */
    function getLogs(
        @mandatory startTime: Timestamp,
        @mandatory endTime: Timestamp,
        level: String,
        logger: String,
        correlationId: String,
        limit: Integer
    ) returns array of OperationsService.Logs;

    /**
     * Get performance traces
     */
    function getTraces(
        @mandatory startTime: Timestamp,
        @mandatory endTime: Timestamp,
        serviceName: String,
        operationName: String,
        minDuration: Integer,
        limit: Integer
    ) returns array of OperationsService.Traces;

    /**
     * Acknowledge an alert
     */
    action acknowledgeAlert(
        @mandatory alertId: UUID,
        message: String
    ) returns OperationsService.Alerts;

    /**
     * Resolve an alert
     */
    action resolveAlert(
        @mandatory alertId: UUID,
        resolution: String
    ) returns OperationsService.Alerts;

    /**
     * Update configuration
     */
    action updateConfiguration(
        @mandatory name: String,
        @mandatory value: String
    ) returns OperationsService.Configuration;

    /**
     * Trigger manual health check
     */
    action triggerHealthCheck() returns OperationsService.Health;

    /**
     * Export metrics to Cloud ALM
     */
    action exportToCloudALM(
        @mandatory startTime: Timestamp,
        @mandatory endTime: Timestamp
    ) returns LargeString; // JSON format export result

    /**
     * Create custom alert rule
     */
    action createAlertRule(
        @mandatory name: String,
        @mandatory metricName: String,
        @mandatory condition: String enum { gt; lt; eq; ne },
        @mandatory threshold: Decimal,
        @mandatory severity: String enum { low; medium; high; critical },
        description: String
    ) returns LargeString; // JSON format alert rule result

    /**
     * Get monitoring dashboard data
     */
    function getDashboard() returns LargeString; // JSON format dashboard data
}