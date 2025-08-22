using A2AService as service from '../../../../srv/a2a-service';

// Agent 7 - Agent Manager UI Annotations
annotate service.AgentManagementTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            managedAgent,
            operationType,
            status,
            priority,
            createdAt
        ],
        
        LineItem : [
            {Value: managedAgent, Label: 'Managed Agent'},
            {Value: operationType, Label: 'Operation Type'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: priority, Label: 'Priority', Criticality: priorityCriticality},
            {Value: agentHealth, Label: 'Agent Health', Visualization: #Rating},
            {Value: performance, Label: 'Performance', Visualization: #Rating},
            {Value: lastHealthCheck, Label: 'Last Health Check'},
            {Value: uptime, Label: 'Uptime'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Agent Management Task',
            TypeNamePlural : 'Agent Management Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://manager'
        },
        
        // Facets for Object Page
        Facets : [
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#General',
                Label : 'General Information'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#AgentDetails',
                Label : 'Agent Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#HealthMonitoring',
                Label : 'Health Monitoring'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#PerformanceMetrics',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Coordination',
                Label : 'Agent Coordination'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'AgentRegistrations/@UI.LineItem',
                Label : 'Agent Registrations'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'HealthChecks/@UI.LineItem',
                Label : 'Health Check History'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'PerformanceLogs/@UI.LineItem',
                Label : 'Performance Logs'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : managedAgent},
                {Value : operationType},
                {Value : status},
                {Value : priority}
            ]
        },
        
        FieldGroup#AgentDetails : {
            Data : [
                {Value : agentType},
                {Value : agentVersion},
                {Value : agentEndpoint},
                {Value : blockchainAddress},
                {Value : registrationBlock},
                {Value : capabilities},
                {Value : dependencies},
                {Value : configurationStatus}
            ]
        },
        
        FieldGroup#HealthMonitoring : {
            Data : [
                {Value : agentHealth, Visualization : #Rating},
                {Value : lastHealthCheck},
                {Value : healthStatus},
                {Value : uptime},
                {Value : responseTime},
                {Value : errorRate, Visualization : #Rating},
                {Value : availabilityScore, Visualization : #Rating},
                {Value : lastRestart}
            ]
        },
        
        FieldGroup#PerformanceMetrics : {
            Data : [
                {Value : performance, Visualization : #Rating},
                {Value : throughput},
                {Value : averageResponseTime},
                {Value : memoryUsage},
                {Value : cpuUsage},
                {Value : diskUsage},
                {Value : networkLatency},
                {Value : loadScore}
            ]
        },
        
        FieldGroup#Coordination : {
            Data : [
                {Value : coordinationStatus},
                {Value : activeConnections},
                {Value : discoveryEnabled},
                {Value : loadBalancingEnabled},
                {Value : failoverStatus},
                {Value : workflowParticipation},
                {Value : trustLevel, Visualization : #Rating},
                {Value : lastCoordination}
            ]
        }
    },
    
    // Capabilities
    Capabilities : {
        SearchRestrictions : {Searchable : true},
        InsertRestrictions : {Insertable : true},
        UpdateRestrictions : {Updatable : true},
        DeleteRestrictions : {Deletable : true}
    }
);

// Agent Registrations as associated entity
annotate service.AgentRegistrations with @(
    UI : {
        LineItem : [
            {Value: agentId, Label: 'Agent ID'},
            {Value: agentName, Label: 'Agent Name'},
            {Value: agentType, Label: 'Type'},
            {Value: version, Label: 'Version'},
            {Value: endpoint, Label: 'Endpoint'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: capabilities, Label: 'Capabilities'},
            {Value: registeredAt, Label: 'Registered At'},
            {Value: lastSeen, Label: 'Last Seen'}
        ]
    }
);

// Health Checks as associated entity
annotate service.HealthChecks with @(
    UI : {
        LineItem : [
            {Value: checkId, Label: 'Check ID'},
            {Value: agentId, Label: 'Agent'},
            {Value: checkType, Label: 'Check Type'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: responseTime, Label: 'Response Time'},
            {Value: healthScore, Label: 'Health Score', Visualization: #Rating},
            {Value: errorMessage, Label: 'Error Message'},
            {Value: checkedAt, Label: 'Checked At'}
        ]
    }
);

// Performance Logs as associated entity
annotate service.PerformanceLogs with @(
    UI : {
        LineItem : [
            {Value: logId, Label: 'Log ID'},
            {Value: agentId, Label: 'Agent'},
            {Value: metricType, Label: 'Metric Type'},
            {Value: metricValue, Label: 'Value'},
            {Value: threshold, Label: 'Threshold'},
            {Value: deviation, Label: 'Deviation'},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: timestamp, Label: 'Timestamp'}
        ]
    }
);

// Agent Types
annotate service.AgentTypes with @(
    UI : {
        SelectionFields : [
            typeName,
            category,
            isActive
        ],
        
        LineItem : [
            {Value: typeName, Label: 'Type Name'},
            {Value: category, Label: 'Category'},
            {Value: description, Label: 'Description'},
            {Value: defaultPort, Label: 'Default Port'},
            {Value: requiredCapabilities, Label: 'Required Capabilities'},
            {Value: isActive, Label: 'Active'},
            {Value: instanceCount, Label: 'Instances'}
        ]
    }
);

// Agent Operations
annotate service.AgentOperations with @(
    UI : {
        SelectionFields : [
            operationName,
            operationType,
            category
        ],
        
        LineItem : [
            {Value: operationName, Label: 'Operation'},
            {Value: operationType, Label: 'Type'},
            {Value: category, Label: 'Category'},
            {Value: description, Label: 'Description'},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: successRate, Label: 'Success Rate', Visualization: #Rating},
            {Value: lastExecuted, Label: 'Last Executed'}
        ]
    }
);

// Value Help annotations
annotate service.AgentManagementTasks {
    managedAgent @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'RegisteredAgents',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : managedAgent,
            ValueListProperty : 'agentName'
        }]
    };
    
    operationType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'AgentOperations',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : operationType,
            ValueListProperty : 'operationName'
        }]
    };
    
    agentType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'AgentTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : agentType,
            ValueListProperty : 'typeName'
        }]
    };
}