namespace a2a.network;

using { cuid, managed } from '@sap/cds/common';

entity Notifications : cuid, managed {
    userId          : String(100) not null;
    title           : String(200) not null;
    message         : String(1000) not null;
    type            : String(20) not null; // info, warning, error, success, system, project, agent, workflow, security
    priority        : String(20) not null; // low, medium, high, critical
    status          : String(20) not null default 'unread'; // unread, read, dismissed, archived
    source          : String(100) default 'system';
    category        : String(50) default 'general';
    metadata        : LargeString; // JSON string for flexible metadata
    actions         : Composition of many NotificationActions on actions.notification = $self;
    readAt          : DateTime;
    dismissedAt     : DateTime;
    expiresAt       : DateTime;
    deliveryStatus  : String(20) default 'pending'; // pending, delivered, failed, retrying
    deliveryAttempts: Integer default 0;
    lastDeliveryAt  : DateTime;
    
    // Indexes for performance
    @Core.Index: [userId, status]
    @Core.Index: [userId, createdAt]
    @Core.Index: [expiresAt]
}

entity NotificationActions : cuid {
    notification    : Association to Notifications;
    label           : String(100) not null;
    actionType      : String(20) not null; // navigate, api_call, external_link
    target          : String(500) not null; // URL, API endpoint, or route
    style           : String(20) default 'default'; // default, primary, success, warning, danger
}

entity NotificationPreferences : cuid, managed {
    userId          : String(100) not null @Core.Index;
    emailEnabled    : Boolean default true;
    pushEnabled     : Boolean default false;
    inAppEnabled    : Boolean default true;
    
    // Notification type preferences
    infoEnabled     : Boolean default true;
    warningEnabled  : Boolean default true;
    errorEnabled    : Boolean default true;
    successEnabled  : Boolean default true;
    systemEnabled   : Boolean default true;
    
    // Delivery preferences
    quietHoursStart : Time;
    quietHoursEnd   : Time;
    timezone        : String(50) default 'UTC';
    
    // Channel-specific settings
    emailAddress    : String(200);
    pushToken       : String(500);
    deviceInfo      : LargeString; // JSON string for device details
}

entity NotificationDeliveryLog : cuid, managed {
    notification    : Association to Notifications;
    channel         : String(20) not null; // websocket, email, push, in-app
    status          : String(20) not null; // success, failed, pending
    attemptNumber   : Integer not null;
    errorMessage    : String(500);
    deliveredAt     : DateTime;
    metadata        : LargeString; // JSON string for delivery details
}

// Views for common queries
view UnreadNotificationCounts as 
    select from Notifications {
        userId,
        count(*) as unreadCount : Integer
    } where status = 'unread' group by userId;

view NotificationStats as
    select from Notifications {
        userId,
        type,
        priority,
        count(*) as count : Integer
    } group by userId, type, priority;