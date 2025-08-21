/**
 * @fileoverview System Monitoring Service - CDS Definition
 * @since 1.0.0
 * @module systemMonitoringService
 * 
 * CDS service definition for system health, monitoring, and error reporting
 * Replaces Express routes with proper CAP service actions
 */

namespace a2a.monitoring;

/**
 * SystemMonitoringService - Health checks, metrics, and error reporting
 * Provides comprehensive system monitoring capabilities
 */
service SystemMonitoringService @(path: '/api/v1/monitoring') {
    
    /**
     * Basic health check
     */
    action getHealth() returns {
        status: String;
        timestamp: String;
        uptime: Integer;
        version: String;
    };
    
    /**
     * Detailed health check with component status
     */
    action getDetailedHealth() returns {
        status: String;
        timestamp: String;
        components: {
            database: {
                status: String;
                responseTime: Integer;
            };
            agents: {
                status: String;
                healthy: Integer;
                total: Integer;
            };
            blockchain: {
                status: String;
                connected: Boolean;
            };
            memory: {
                used: Integer64;
                total: Integer64;
                percentage: Decimal;
            };
            cpu: {
                usage: Decimal;
                load: array of Decimal;
            };
        };
    };
    
    /**
     * Readiness probe for Kubernetes
     */
    action getReadiness() returns {
        ready: Boolean;
        services: array of {
            name: String;
            ready: Boolean;
        };
    };
    
    /**
     * Liveness probe for Kubernetes
     */
    action getLiveness() returns {
        alive: Boolean;
        timestamp: String;
    };
    
    /**
     * System metrics in Prometheus format
     */
    action getMetrics() returns String;
    
    /**
     * Report an error
     */
    action reportError(error: {
        message: String;
        stack: String;
        context: String;
        severity: String;
        userId: String;
        timestamp: String;
    }) returns {
        success: Boolean;
        errorId: String;
    };
    
    /**
     * Get error statistics
     */
    action getErrorStats() returns {
        total: Integer;
        byLevel: {
            error: Integer;
            warn: Integer;
            info: Integer;
        };
        recent: array of {
            timestamp: String;
            level: String;
            message: String;
        };
    };
    
    /**
     * Get recent logs
     */
    action getLogs(limit: Integer, level: String) returns array of {
        timestamp: String;
        level: String;
        message: String;
        context: String;
    };
    
    /**
     * Get cache statistics
     */
    action getCacheStats() returns {
        hits: Integer64;
        misses: Integer64;
        hitRate: Decimal;
        size: Integer;
        memory: Integer64;
    };
    
    /**
     * Invalidate cache
     */
    action invalidateCache(pattern: String) returns {
        success: Boolean;
        cleared: Integer;
    };
}