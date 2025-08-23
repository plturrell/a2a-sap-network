namespace com.a2a.network;

/**
 * Agent Proxy Service
 * Consolidates all agent proxy endpoints into a single CAP service
 * Replaces Express routes with proper CAP service handlers
 */
service AgentProxyService {
    
    // Generic proxy action for all agent requests
    action proxyRequest(
        agentId : String,
        path : String,
        method : String,
        body : String,
        query : String
    ) returns String;
    
    // Agent-specific health checks
    function getAgentHealth(agentId : String) returns {
        status : String;
        timestamp : DateTime;
        details : String;
    };
    
    // Batch operations across agents
    action executeBatchOperation(
        agents : array of String,
        operation : String,
        parameters : String
    ) returns array of {
        agentId : String;
        success : Boolean;
        result : String;
        error : String;
    };
    
    // WebSocket upgrade endpoints
    action upgradeToWebSocket(
        agentId : String,
        endpoint : String
    ) returns {
        wsUrl : String;
        token : String;
    };
    
    // OData proxies for agents
    entity Agent1Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent2Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent3Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent4Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent5Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent6Tasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent7RegisteredAgents {
        key ID : UUID;
        name : String;
        type : String;
        status : String;
        registeredAt : DateTime;
        lastHealthCheck : DateTime;
        metadata : LargeString;
    }
    
    entity Agent7ManagementTasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent8DataTasks {
        key ID : UUID;
        taskType : String;
        status : String;
        createdAt : DateTime;
        completedAt : DateTime;
        metadata : LargeString;
    }
    
    entity Agent8StorageBackends {
        key ID : UUID;
        name : String;
        type : String;
        status : String;
        configuration : LargeString;
    }
}