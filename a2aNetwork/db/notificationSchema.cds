/**
 * @fileoverview Notification and Webhook Schema - CDS Definition
 * @since 1.0.0 
 * @module notificationSchema
 *
 * Data model for webhook endpoints and notification history
 * Extends existing notification system with webhook capabilities
 */

using { cuid, managed, Currency } from '@sap/cds/common';

namespace com.sap.a2a.notifications;

/**
 * Webhook Endpoints Configuration
 * Stores webhook endpoint configurations for external notifications
 */
entity WebhookEndpoints : cuid, managed {
    name            : String(100) not null;
    url             : String(500) not null;
    type            : String(50) not null enum {
        slack;
        teams;
        discord;
        email;
        pagerduty;
        datadog;
        generic
    } default 'generic';
    
    active          : Boolean default true;
    
    // Event subscriptions
    events          : array of String;
    
    // Authentication and security
    secret          : String(255);
    headers         : LargeString; // JSON string of custom headers
    
    // Configuration
    config          : LargeString; // JSON configuration for specific webhook types
    retryAttempts   : Integer default 3;
    timeout         : Integer default 10000; // milliseconds
    
    // Statistics
    totalCalls      : Integer default 0;
    successfulCalls : Integer default 0;
    failedCalls     : Integer default 0;
    lastTriggered   : Timestamp;
    
    // Success rate calculation
    virtual successRate : Decimal(5,2);
    
    // Relationships
    history : Composition of many NotificationHistory on history.webhook = $self;
}

/**
 * Notification History
 * Tracks all notification attempts for auditing and monitoring
 */
entity NotificationHistory : cuid, managed {
    webhook     : Association to WebhookEndpoints not null;
    eventType   : String(100) not null;
    status      : String(50) not null enum {
        success;
        failed;
        retry;
        timeout
    };
    
    // Request/response data
    payload     : LargeString; // JSON payload sent
    response    : LargeString; // Response received
    error       : LargeString; // Error message if failed
    
    // Timing
    timestamp   : Timestamp not null;
    duration    : Integer; // milliseconds
    
    // Retry information
    attempt     : Integer default 1;
    maxAttempts : Integer default 3;
    nextRetry   : Timestamp;
    
    // Virtual fields for UI
    virtual statusIcon  : String;
    virtual statusColor : String;
}

/**
 * Notification Templates
 * Pre-defined templates for common notification scenarios
 */
entity NotificationTemplates : cuid, managed {
    name        : String(100) not null;
    eventType   : String(100) not null;
    webhookType : String(50) not null;
    
    // Template content
    title       : String(200);
    message     : LargeString;
    template    : LargeString; // JSON template for webhook-specific formatting
    
    // Configuration
    priority    : String(20) enum {
        low;
        medium;
        high;
        critical
    } default 'medium';
    
    active      : Boolean default true;
    
    // Usage statistics
    usageCount  : Integer default 0;
    lastUsed    : Timestamp;
}

/**
 * Notification Rules
 * Rules for when and how to send notifications
 */
entity NotificationRules : cuid, managed {
    name            : String(100) not null;
    description     : String(500);
    
    // Conditions
    eventType       : String(100) not null;
    environment     : String(50); // production, staging, development
    priority        : String(20);
    
    // Filters
    applicationName : String(100);
    minimumDuration : Integer; // Only notify if deployment takes longer than X seconds
    
    // Actions
    webhooks        : array of String; // Webhook IDs to notify
    delay           : Integer default 0; // Delay before sending (seconds)
    
    // Rate limiting
    cooldown        : Integer default 0; // Minimum time between notifications (seconds)
    lastTriggered   : Timestamp;
    
    // Status
    active          : Boolean default true;
    triggerCount    : Integer default 0;
}

/**
 * Alert Configurations
 * System-wide alert configurations for different scenarios
 */
entity AlertConfigurations : cuid, managed {
    name            : String(100) not null;
    type            : String(50) not null enum {
        deployment_failed;
        deployment_stuck;
        system_health;
        performance_degradation;
        security_breach;
        resource_exhaustion
    };
    
    // Thresholds
    threshold       : Decimal(10,2);
    timeWindow      : Integer default 300; // seconds
    
    // Escalation
    escalationLevel : Integer default 1;
    escalationDelay : Integer default 900; // 15 minutes
    
    // Notification settings
    webhooks        : array of String;
    template        : Association to NotificationTemplates;
    
    // Status
    active          : Boolean default true;
    lastTriggered   : Timestamp;
    triggerCount    : Integer default 0;
}

/**
 * Subscription Management
 * User/team subscriptions to different notification types
 */
entity NotificationSubscriptions : cuid, managed {
    userId          : String(100) not null;
    teamId          : String(100);
    
    // Subscription details
    eventTypes      : array of String;
    environments    : array of String;
    applications    : array of String;
    
    // Delivery preferences
    webhookType     : String(50) default 'email';
    webhook         : Association to WebhookEndpoints;
    
    // Scheduling
    quietHours      : LargeString; // JSON: { start: "22:00", end: "08:00", timezone: "UTC" }
    workingDays     : array of String; // ["monday", "tuesday", ...]
    
    // Status
    active          : Boolean default true;
    suspended       : Boolean default false;
    suspendedUntil  : Timestamp;
}

/**
 * Notification Queue
 * Queue for managing notification delivery with priority and retry logic
 */
entity NotificationQueue : cuid, managed {
    eventType       : String(100) not null;
    priority        : String(20) enum {
        low;
        medium; 
        high;
        critical
    } default 'medium';
    
    // Payload
    data            : LargeString not null; // JSON event data
    webhook         : Association to WebhookEndpoints not null;
    
    // Processing status
    status          : String(50) enum {
        pending;
        processing;
        completed;
        failed;
        cancelled
    } default 'pending';
    
    // Scheduling
    scheduledFor    : Timestamp;
    attempts        : Integer default 0;
    maxAttempts     : Integer default 3;
    nextRetry       : Timestamp;
    
    // Processing info
    processedAt     : Timestamp;
    error           : LargeString;
    
    // Virtual fields
    virtual isOverdue : Boolean;
    virtual canRetry  : Boolean;
}

/**
 * View: Webhook Statistics
 * Aggregated statistics for webhook performance monitoring
 */
view WebhookStats as 
    SELECT 
        webhook.ID,
        webhook.name,
        webhook.type,
        webhook.active,
        webhook.totalCalls,
        webhook.successfulCalls,
        webhook.failedCalls,
        CAST(CASE WHEN webhook.totalCalls > 0 
             THEN (webhook.successfulCalls * 100.0 / webhook.totalCalls) 
             ELSE 0 END AS Decimal(5,2)) as successRate : Decimal(5,2),
        webhook.lastTriggered,
        COUNT(history.ID) as historyCount : Integer,
        MAX(history.timestamp) as lastAttempt : Timestamp
    FROM WebhookEndpoints as webhook
    LEFT JOIN NotificationHistory as history ON history.webhook.ID = webhook.ID
    GROUP BY 
        webhook.ID,
        webhook.name, 
        webhook.type,
        webhook.active,
        webhook.totalCalls,
        webhook.successfulCalls,
        webhook.failedCalls,
        webhook.lastTriggered;

/**
 * View: Recent Notifications
 * Recent notification activity for dashboard display
 */
view RecentNotifications as
    SELECT 
        history.ID,
        history.eventType,
        history.status,
        history.timestamp,
        history.duration,
        history.attempt,
        webhook.name as webhookName,
        webhook.type as webhookType
    FROM NotificationHistory as history
    INNER JOIN WebhookEndpoints as webhook ON webhook.ID = history.webhook.ID
    WHERE history.timestamp >= $now - 86400000 // Last 24 hours
    ORDER BY history.timestamp DESC;