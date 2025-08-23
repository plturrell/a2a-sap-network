namespace a2a.platform;

using { a2a.platform.Identifiable, a2a.platform.Manageable, a2a.platform.StatusTracking, a2a.platform.BusinessMetadata } from '../aspects/common';

// Security and audit entities  
entity Users : Identifiable, Manageable, StatusTracking {
    userId : String(100) not null unique;
    email : String(255) not null;
    firstName : String(100);
    lastName : String(100);
    department : String(100);
    
    // Authentication
    lastLogin : Timestamp;
    loginCount : Integer default 0;
    passwordLastChanged : Timestamp;
    accountLocked : Boolean default false;
    
    // Relationships
    roles : Association to many UserRoles on roles.user = $self;
    sessions : Composition of many UserSessions on sessions.user = $self;
    auditLogs : Composition of many AuditLogs on auditLogs.user = $self;
}

entity Roles : Identifiable, BusinessMetadata, StatusTracking {
    roleCode : String(50) not null unique;
    permissions : LargeString; // JSON array of permissions
    
    // Relationships
    users : Association to many UserRoles on users.role = $self;
}

entity UserRoles : Identifiable {
    user : Association to Users;
    role : Association to Roles;
    assignedAt : Timestamp not null;
    assignedBy : String(100);
    validFrom : Timestamp;
    validTo : Timestamp;
}

entity UserSessions : Identifiable {
    user : Association to Users;
    sessionId : String(255) not null unique;
    ipAddress : String(45); // IPv6 compatible
    userAgent : String(500);
    loginTime : Timestamp not null;
    logoutTime : Timestamp;
    lastActivity : Timestamp;
    isActive : Boolean default true;
}

entity AuditLogs : Identifiable {
    user : Association to Users;
    action : String(100) not null;
    entityType : String(100); // table/entity affected
    entityId : String(100); // ID of affected record
    oldValues : LargeString; // JSON of old values
    newValues : LargeString; // JSON of new values
    timestamp : Timestamp not null;
    ipAddress : String(45);
    userAgent : String(500);
    outcome : String(20) default 'SUCCESS'; // SUCCESS, FAILURE, PARTIAL
}

entity SecurityPolicies : Identifiable, Manageable, StatusTracking, BusinessMetadata {
    policyType : String(100) not null; // PASSWORD, ACCESS, DATA_RETENTION, etc.
    policyRules : LargeString not null; // JSON policy definition
    enforcement : String(20) default 'ENFORCING'; // ENFORCING, MONITORING, DISABLED
    
    // Applicability
    appliesTo : String(100); // ALL, ROLE_BASED, USER_SPECIFIC
    targetRoles : LargeString; // JSON array of role codes
    targetUsers : LargeString; // JSON array of user IDs
}

entity SecurityEvents : Identifiable {
    eventType : String(100) not null; // LOGIN_FAILURE, PRIVILEGE_ESCALATION, etc.
    severity : String(20) not null; // CRITICAL, HIGH, MEDIUM, LOW, INFO
    user : Association to Users;
    description : String(1000) not null;
    timestamp : Timestamp not null;
    ipAddress : String(45);
    resolved : Boolean default false;
    resolvedAt : Timestamp;
    resolvedBy : String(100);
}