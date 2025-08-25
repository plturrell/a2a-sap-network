/**
 * @fileoverview Notification Service - CDS Definition
 * @since 1.0.0
 * @module notificationService
 * 
 * CAP service definition for webhook management and notification delivery
 * Integrates with deployment events and system monitoring
 */

using { com.sap.a2a.notifications as notifications } from '../db/notificationSchema';
using { cuid, managed } from '@sap/cds/common';

namespace com.sap.a2a.services;

/**
 * NotificationService - Main service for webhook and notification management
 * Provides comprehensive notification delivery and webhook management capabilities
 */
@path: '/api/v1/notifications'
@requires: 'authenticated-user'
service NotificationService {
    
    /**
     * Webhook Endpoints Management
     */
    @odata.draft.enabled
    entity WebhookEndpoints as projection on notifications.WebhookEndpoints {
        *,
        virtual successRate : Decimal(5,2),
        virtual statusIcon  : String,
        virtual statusColor : String
    } actions {
        action test() returns {
            success    : Boolean;
            message    : String;
            duration   : Integer;
            timestamp  : Timestamp;
        };
        action enable() returns WebhookEndpoints;
        action disable() returns WebhookEndpoints;
        action resetStats() returns WebhookEndpoints;
    };
    
    /**
     * Notification History
     */
    @readonly
    entity NotificationHistory as projection on notifications.NotificationHistory {
        *,
        virtual statusIcon  : String,
        virtual statusColor : String,
        virtual canRetry    : Boolean
    } actions {
        action retry() returns NotificationHistory;
    };
    
    /**
     * Notification Templates
     */
    @odata.draft.enabled
    entity NotificationTemplates as projection on notifications.NotificationTemplates;
    
    /**
     * Notification Rules
     */
    @odata.draft.enabled
    entity NotificationRules as projection on notifications.NotificationRules;
    
    /**
     * Alert Configurations
     */
    @odata.draft.enabled
    entity AlertConfigurations as projection on notifications.AlertConfigurations;
    
    /**
     * Notification Subscriptions
     */
    @odata.draft.enabled
    entity NotificationSubscriptions as projection on notifications.NotificationSubscriptions;
    
    /**
     * Notification Queue (Admin only)
     */
    @requires: 'admin'
    entity NotificationQueue as projection on notifications.NotificationQueue {
        *,
        virtual isOverdue : Boolean,
        virtual canRetry  : Boolean
    } actions {
        action process() returns NotificationQueue;
        action cancel() returns NotificationQueue;
        action reschedule(scheduledFor: Timestamp) returns NotificationQueue;
    };
    
    /**
     * Views for reporting and analytics
     */
    @readonly
    entity WebhookStats as projection on notifications.WebhookStats;
    
    @readonly
    entity RecentNotifications as projection on notifications.RecentNotifications;
    
    /**
     * Functions for webhook management
     */
    
    /**
     * Get webhook statistics and performance metrics
     */
    function getWebhookStats() returns {
        totalWebhooks     : Integer;
        activeWebhooks    : Integer;
        queueSize         : Integer;
        totalCalls        : Integer;
        successfulCalls   : Integer;
        failedCalls       : Integer;
        avgSuccessRate    : Decimal(5,2);
        webhooks          : array of {
            id            : String;
            name          : String;
            type          : String;
            active        : Boolean;
            totalCalls    : Integer;
            successfulCalls : Integer;
            failedCalls   : Integer;
            successRate   : Decimal(5,2);
            lastTriggered : Timestamp;
        };
    };
    
    /**
     * Get notification history with filters
     */
    function getNotificationHistory(
        limit      : Integer,
        offset     : Integer,
        eventType  : String,
        webhookId  : String,
        status     : String,
        dateFrom   : Timestamp,
        dateTo     : Timestamp
    ) returns {
        history       : array of {
            id          : String;
            eventType   : String;
            status      : String;
            timestamp   : Timestamp;
            duration    : Integer;
            webhookName : String;
            webhookType : String;
            error       : String;
        };
        total         : Integer;
        limit         : Integer;
        offset        : Integer;
    };
    
    /**
     * Get system health metrics for notification monitoring
     */
    function getSystemHealth() returns {
        webhookService    : {
            status        : String;
            queueSize     : Integer;
            processing    : Boolean;
        };
        recentActivity    : array of {
            timestamp     : Timestamp;
            eventType     : String;
            status        : String;
            webhookCount  : Integer;
        };
        errorRate         : Decimal(5,2);
        avgResponseTime   : Integer;
    };
    
    /**
     * Actions for webhook operations
     */
    
    /**
     * Send test notification to webhook
     */
    action testWebhook(
        webhookId : String,
        eventType : String
    ) returns {
        success       : Boolean;
        notificationId : String;
        message       : String;
        duration      : Integer;
        timestamp     : Timestamp;
    };
    
    /**
     * Register new webhook endpoint
     */
    action registerWebhook(
        name          : String,
        url           : String,
        type          : String,
        events        : array of String,
        active        : Boolean,
        secret        : String,
        headers       : String,
        config        : String,
        retryAttempts : Integer
    ) returns {
        success       : Boolean;
        webhookId     : String;
        message       : String;
    };
    
    /**
     * Update webhook configuration
     */
    action updateWebhookConfig(
        webhookId     : String,
        config        : {
            name          : String;
            url           : String;
            active        : Boolean;
            events        : array of String;
            secret        : String;
            headers       : String;
            retryAttempts : Integer;
        }
    ) returns {
        success       : Boolean;
        webhook       : {
            id            : String;
            name          : String;
            active        : Boolean;
        };
        message       : String;
    };
    
    /**
     * Delete webhook endpoint
     */
    action deleteWebhook(
        webhookId : String
    ) returns {
        success   : Boolean;
        message   : String;
    };
    
    /**
     * Trigger manual notification
     */
    action triggerNotification(
        eventType     : String,
        data          : String,
        webhookIds    : array of String,
        priority      : String
    ) returns {
        success       : Boolean;
        notificationId : String;
        message       : String;
        queuedCount   : Integer;
    };
    
    /**
     * Bulk webhook operations
     */
    action bulkWebhookOperation(
        webhookIds    : array of String,
        operation     : String, // 'enable', 'disable', 'test', 'delete'
        config        : String  // JSON configuration if needed
    ) returns {
        success       : Boolean;
        processed     : Integer;
        failed        : Integer;
        results       : array of {
            webhookId : String;
            success   : Boolean;
            message   : String;
        };
    };
    
    /**
     * Process notification queue manually (admin only)
     */
    @requires: 'admin'
    action processQueue() returns {
        success       : Boolean;
        processed     : Integer;
        failed        : Integer;
        remaining     : Integer;
        message       : String;
    };
    
    /**
     * Clear notification history (admin only)
     */
    @requires: 'admin'
    action clearHistory(
        olderThan     : Timestamp,
        status        : String
    ) returns {
        success       : Boolean;
        deleted       : Integer;
        message       : String;
    };
    
    /**
     * Export webhook configuration
     */
    action exportConfiguration() returns {
        success       : Boolean;
        configuration : String; // JSON export
        timestamp     : Timestamp;
    };
    
    /**
     * Import webhook configuration
     */
    action importConfiguration(
        configuration : String,
        overwrite     : Boolean
    ) returns {
        success       : Boolean;
        imported      : Integer;
        skipped       : Integer;
        errors        : array of String;
    };
}

/**
 * Event types for notifications
 */
type EventType : String enum {
    deploymentStarted;
    deploymentCompleted;
    deploymentFailed;
    deploymentRollback;
    deploymentApproved;
    deploymentPaused;
    systemHealthDegraded;
    systemHealthRestored;
    agentConnected;
    agentDisconnected;
    serviceUnavailable;
    serviceRestored;
    performanceAlert;
    securityAlert;
    resourceExhaustion;
    transactionCompleted;
    transactionFailed;
}

/**
 * Webhook types supported
 */
type WebhookType : String enum {
    slack;
    teams;
    discord;
    email;
    pagerduty;
    datadog;
    generic;
}

/**
 * Notification priorities
 */
type NotificationPriority : String enum {
    low;
    medium;
    high;
    critical;
}

/**
 * Notification status
 */
type NotificationStatus : String enum {
    pending;
    processing;
    success;
    failed;
    timeout;
    cancelled;
    retry;
}