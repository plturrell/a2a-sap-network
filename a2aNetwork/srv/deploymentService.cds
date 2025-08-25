using {com.sap.a2a.deployment as deployment} from '../db/deploymentSchema';
using {com.sap.a2a.operations as ops} from '../db/operationsSchema';

namespace com.sap.a2a;

@path: '/api/v1/deployment'
@requires: 'authenticated-user'
service DeploymentService {
    
    // Main entities
    @odata.draft.enabled
    entity Deployments as projection on deployment.Deployments {
        *,
        virtual statusIcon : String,
        virtual statusColor : String,
        virtual progressPercentage : Integer
    } actions {
        @Common.SideEffects: {
            TargetEntities: ['.']
        }
        action start() returns Deployments;
        
        @Common.SideEffects: {
            TargetEntities: ['.']
        }
        action approve() returns Deployments;
        
        @Common.SideEffects: {
            TargetEntities: ['.']
        }
        action rollback() returns Deployments;
        
        @Common.SideEffects: {
            TargetEntities: ['.']
        }
        action retry() returns Deployments;
    };
    
    entity DeploymentConfigs as projection on deployment.DeploymentConfigs;
    entity DeploymentSchedules as projection on deployment.DeploymentSchedules;
    entity DeploymentTemplates as projection on deployment.DeploymentTemplates;
    entity Rollbacks as projection on deployment.Rollbacks;
    
    // Read-only views
    @readonly entity DeploymentSummary as projection on deployment.DeploymentSummary;
    @readonly entity EnvironmentStatus as projection on deployment.EnvironmentStatus;
    @readonly entity HealthChecks as projection on deployment.HealthChecks;
    @readonly entity DeploymentStages as projection on deployment.DeploymentStages;
    @readonly entity DeploymentMetrics as projection on deployment.DeploymentMetrics;
    @readonly entity Notifications as projection on deployment.Notifications;
    
    // Functions
    function getDeploymentStatus(environment: String) returns {
        environment: String;
        lastDeployment: Deployments;
        activeDeployments: Integer;
        healthStatus: String;
        uptime: Decimal;
    };
    
    function getDeploymentHistory(
        environment: String,
        limit: Integer
    ) returns array of Deployments;
    
    function validateDeployment(
        appName: String,
        environment: String,
        version: String
    ) returns {
        isValid: Boolean;
        errors: array of String;
        warnings: array of String;
    };
    
    function getDeploymentMetrics(
        deploymentId: UUID,
        metricType: String
    ) returns array of deployment.DeploymentMetrics;
    
    // Actions
    action createDeployment(
        appName: String,
        environment: String,
        version: String,
        deploymentType: String,
        dockerImage: String,
        notes: String
    ) returns Deployments;
    
    action triggerHealthCheck(
        deploymentId: UUID,
        checkType: String
    ) returns deployment.HealthChecks;
    
    action sendNotification(
        deploymentId: UUID,
        notificationType: String,
        recipient: String,
        message: String
    ) returns deployment.Notifications;
    
    action scheduleDeployment(
        appName: String,
        environment: String,
        scheduleType: String,
        cronExpression: String,
        deploymentConfig: String
    ) returns deployment.DeploymentSchedules;
    
    // Real-time monitoring functions
    function getLiveDeploymentStatus() returns {
        activeDeployments: array of {
            id: UUID;
            appName: String;
            environment: String;
            status: String;
            progress: Integer;
            currentStage: String;
            startTime: Timestamp;
            estimatedCompletion: Timestamp;
        };
    };
    
    function getSystemHealth() returns {
        production: {
            status: String;
            healthScore: Integer;
            activeAgents: Integer;
            totalAgents: Integer;
        };
        staging: {
            status: String;
            healthScore: Integer;
            activeAgents: Integer;
            totalAgents: Integer;
        };
        alerts: array of {
            severity: String;
            message: String;
            timestamp: Timestamp;
        };
    };
    
    // Integration with Fly.io
    function getFlyStatus(appName: String) returns {
        status: String;
        instances: array of {
            id: String;
            status: String;
            region: String;
            cpu: Decimal;
            memory: Decimal;
        };
        ips: array of {
            type: String;
            address: String;
        };
        certificates: array of {
            domain: String;
            status: String;
            expiresAt: Date;
        };
    };
    
    action deployToFly(
        appName: String,
        environment: String,
        strategy: String
    ) returns {
        deploymentId: UUID;
        status: String;
        message: String;
    };
    
    action manageFlySecrets(
        appName: String,
        action: String, // set, unset, list
        secrets: array of {
            key: String;
            value: String;
        }
    ) returns {
        success: Boolean;
        message: String;
    };
    
    // Event definitions
    event deploymentStarted : {
        deploymentId: UUID;
        appName: String;
        environment: String;
        version: String;
        initiatedBy: String;
    };
    
    event deploymentCompleted : {
        deploymentId: UUID;
        appName: String;
        environment: String;
        status: String;
        duration: Integer;
    };
    
    event deploymentFailed : {
        deploymentId: UUID;
        appName: String;
        environment: String;
        error: String;
        stage: String;
    };
    
    event healthCheckFailed : {
        deploymentId: UUID;
        serviceName: String;
        endpoint: String;
        error: String;
    };
}