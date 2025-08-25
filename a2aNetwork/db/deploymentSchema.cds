namespace com.sap.a2a.deployment;

using {
    cuid,
    managed,
    Currency,
    temporal
} from '@sap/cds/common';

// Deployment entity
entity Deployments : cuid, managed {
    appName         : String(100) not null;
    environment     : String(50) not null enum {
        development;
        staging;
        production
    };
    version         : String(50) not null;
    gitCommit       : String(40);
    dockerImage     : String(200);
    status          : String(50) not null enum {
        pending;
        in_progress;
        completed;
        failed;
        rolled_back
    } default 'pending';
    deploymentType  : String(50) enum {
        standard;
        zero_downtime;
        canary;
        blue_green
    } default 'standard';
    startTime       : Timestamp;
    endTime         : Timestamp;
    duration        : Integer; // in seconds
    deployedBy      : String(100);
    approvedBy      : String(100);
    notes           : String(500);
    
    // Relations
    healthChecks    : Composition of many HealthChecks on healthChecks.deployment = $self;
    stages          : Composition of many DeploymentStages on stages.deployment = $self;
    metrics         : Composition of many DeploymentMetrics on metrics.deployment = $self;
    notifications   : Composition of many Notifications on notifications.deployment = $self;
}

// Health check results
entity HealthChecks : cuid {
    deployment      : Association to Deployments;
    checkType       : String(50) enum {
        pre_deployment;
        post_deployment;
        continuous
    };
    serviceName     : String(100);
    endpoint        : String(200);
    status          : String(20) enum {
        healthy;
        unhealthy;
        degraded;
        unknown
    };
    responseTime    : Integer; // milliseconds
    message         : String(500);
    timestamp       : Timestamp;
}

// Deployment stages
entity DeploymentStages : cuid {
    deployment      : Association to Deployments;
    stageName       : String(100) not null;
    stageOrder      : Integer not null;
    status          : String(50) enum {
        pending;
        running;
        completed;
        failed;
        skipped
    } default 'pending';
    startTime       : Timestamp;
    endTime         : Timestamp;
    logs            : LargeString;
    errorMessage    : String(500);
}

// Deployment metrics
entity DeploymentMetrics : cuid {
    deployment      : Association to Deployments;
    metricType      : String(50) not null;
    metricName      : String(100) not null;
    metricValue     : Decimal(15, 3);
    unit            : String(20);
    timestamp       : Timestamp;
}

// Deployment configurations
entity DeploymentConfigs : cuid, managed {
    environment     : String(50) not null;
    configKey       : String(100) not null;
    configValue     : String(500) not null;
    isSecret        : Boolean default false;
    description     : String(200);
    validFrom       : Date;
    validTo         : Date;
}

// Rollback history
entity Rollbacks : cuid, managed {
    deployment      : Association to Deployments;
    rollbackFrom    : Association to Deployments;
    rollbackTo      : Association to Deployments;
    reason          : String(500) not null;
    initiatedBy     : String(100);
    timestamp       : Timestamp;
    status          : String(50) enum {
        initiated;
        in_progress;
        completed;
        failed
    };
}

// Notifications
entity Notifications : cuid {
    deployment      : Association to Deployments;
    notificationType: String(50) enum {
        email;
        slack;
        teams;
        webhook
    };
    recipient       : String(200);
    subject         : String(200);
    message         : LargeString;
    status          : String(50) enum {
        pending;
        sent;
        failed
    } default 'pending';
    sentAt          : Timestamp;
    error           : String(500);
}

// Deployment schedules
entity DeploymentSchedules : cuid, managed {
    appName         : String(100) not null;
    environment     : String(50) not null;
    scheduleType    : String(50) enum {
        once;
        daily;
        weekly;
        monthly;
        cron
    };
    cronExpression  : String(100);
    nextRun         : Timestamp;
    lastRun         : Timestamp;
    isActive        : Boolean default true;
    deploymentConfig: String(2000); // JSON configuration
}

// Deployment templates
entity DeploymentTemplates : cuid, managed {
    templateName    : String(100) not null;
    description     : String(500);
    templateType    : String(50) enum {
        fly_io;
        kubernetes;
        docker_compose;
        custom
    };
    templateContent : LargeString; // JSON or YAML content
    variables       : LargeString; // JSON variables definition
    isActive        : Boolean default true;
}

// Views for dashboard
@readonly
entity DeploymentSummary as 
    select from Deployments {
        key ID,
        appName,
        environment,
        version,
        status,
        deploymentType,
        startTime,
        endTime,
        duration,
        deployedBy,
        (
            select count(*) from HealthChecks as hc 
            where hc.deployment.ID = Deployments.ID 
            and hc.status = 'healthy'
        ) as healthyChecks : Integer,
        (
            select count(*) from HealthChecks as hc 
            where hc.deployment.ID = Deployments.ID
        ) as totalChecks : Integer
    };

@readonly
entity EnvironmentStatus as
    select from Deployments {
        environment,
        count(*) as totalDeployments : Integer,
        sum(case when status = 'completed' then 1 else 0 end) as successful : Integer,
        sum(case when status = 'failed' then 1 else 0 end) as failed : Integer,
        avg(duration) as avgDuration : Integer
    } group by environment;