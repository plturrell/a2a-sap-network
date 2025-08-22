using A2AService as service from '../../../../srv/a2a-service';

// Agent 15 - Orchestrator Agent UI Annotations
annotate service.Workflows with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            workflowType,
            orchestrationMode,
            status,
            priority,
            createdAt
        ],
        
        LineItem : [
            {Value: workflowName, Label: 'Workflow Name'},
            {Value: workflowType, Label: 'Workflow Type'},
            {Value: orchestrationMode, Label: 'Orchestration Mode'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: priority, Label: 'Priority'},
            {Value: activeSteps, Label: 'Active Steps'},
            {Value: completionRate, Label: 'Completion', Visualization: #Rating},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: nextStep, Label: 'Next Step'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Workflow',
            TypeNamePlural : 'Workflows',
            Title : {Value : workflowName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://workflow-tasks'
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
                Target : '@UI.FieldGroup#OrchestrationConfig',
                Label : 'Orchestration Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#ExecutionSettings',
                Label : 'Execution Settings'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#AgentCoordination',
                Label : 'Agent Coordination'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#MonitoringConfig',
                Label : 'Monitoring Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#PerformanceMetrics',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'WorkflowSteps/@UI.LineItem',
                Label : 'Workflow Steps'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'AgentAssignments/@UI.LineItem',
                Label : 'Agent Assignments'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ExecutionHistory/@UI.LineItem',
                Label : 'Execution History'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : workflowName},
                {Value : description},
                {Value : workflowType},
                {Value : orchestrationMode},
                {Value : status},
                {Value : priority},
                {Value : version}
            ]
        },
        
        FieldGroup#OrchestrationConfig : {
            Data : [
                {Value : executionStrategy},
                {Value : parallelization},
                {Value : maxConcurrency},
                {Value : taskDistribution},
                {Value : loadBalancing},
                {Value : failoverStrategy},
                {Value : retryPolicy},
                {Value : circuitBreaker}
            ]
        },
        
        FieldGroup#ExecutionSettings : {
            Data : [
                {Value : triggerType},
                {Value : schedule},
                {Value : timeout},
                {Value : maxDuration},
                {Value : checkpointEnabled},
                {Value : rollbackEnabled},
                {Value : compensationEnabled},
                {Value : transactional}
            ]
        },
        
        FieldGroup#AgentCoordination : {
            Data : [
                {Value : agentSelectionMode},
                {Value : communicationProtocol},
                {Value : messageQueuing},
                {Value : eventBus},
                {Value : coordinationPattern},
                {Value : consensusAlgorithm},
                {Value : conflictResolution},
                {Value : dataConsistency}
            ]
        },
        
        FieldGroup#MonitoringConfig : {
            Data : [
                {Value : monitoringEnabled},
                {Value : metricsCollection},
                {Value : tracingEnabled},
                {Value : loggingLevel},
                {Value : alertingEnabled},
                {Value : healthCheckInterval},
                {Value : performanceThresholds},
                {Value : anomalyDetection}
            ]
        },
        
        FieldGroup#PerformanceMetrics : {
            Data : [
                {Value : completionRate, Visualization : #Rating},
                {Value : successRate, Visualization : #Rating},
                {Value : averageExecutionTime},
                {Value : throughput},
                {Value : latency},
                {Value : resourceUtilization, Visualization : #Rating},
                {Value : costEfficiency, Visualization : #Rating},
                {Value : slaCompliance, Visualization : #Rating}
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

// Workflow Steps as associated entity
annotate service.WorkflowSteps with @(
    UI : {
        LineItem : [
            {Value: stepNumber, Label: 'Step #'},
            {Value: stepName, Label: 'Step Name'},
            {Value: stepType, Label: 'Type'},
            {Value: assignedAgent, Label: 'Assigned Agent'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: startTime, Label: 'Start Time'},
            {Value: duration, Label: 'Duration'},
            {Value: retries, Label: 'Retries'},
            {Value: dependencies, Label: 'Dependencies'}
        ]
    }
);

// Agent Assignments as associated entity
annotate service.AgentAssignments with @(
    UI : {
        LineItem : [
            {Value: agentName, Label: 'Agent Name'},
            {Value: agentType, Label: 'Agent Type'},
            {Value: role, Label: 'Role'},
            {Value: taskCount, Label: 'Tasks'},
            {Value: workload, Label: 'Workload', Visualization: #Rating},
            {Value: availability, Label: 'Availability'},
            {Value: performance, Label: 'Performance', Visualization: #Rating},
            {Value: lastAssigned, Label: 'Last Assigned'}
        ]
    }
);

// Execution History as associated entity
annotate service.ExecutionHistory with @(
    UI : {
        LineItem : [
            {Value: executionId, Label: 'Execution ID'},
            {Value: startTime, Label: 'Start Time'},
            {Value: endTime, Label: 'End Time'},
            {Value: duration, Label: 'Duration'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: completedSteps, Label: 'Completed Steps'},
            {Value: failedSteps, Label: 'Failed Steps'},
            {Value: cost, Label: 'Cost'},
            {Value: triggeredBy, Label: 'Triggered By'}
        ]
    }
);

// Pipeline Definitions
annotate service.PipelineDefinitions with @(
    UI : {
        SelectionFields : [
            pipelineName,
            pipelineType,
            isActive
        ],
        
        LineItem : [
            {Value: pipelineName, Label: 'Pipeline Name'},
            {Value: pipelineType, Label: 'Type'},
            {Value: stages, Label: 'Stages'},
            {Value: totalSteps, Label: 'Total Steps'},
            {Value: estimatedDuration, Label: 'Est. Duration'},
            {Value: reliability, Label: 'Reliability', Visualization: #Rating},
            {Value: lastExecuted, Label: 'Last Executed'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Task Queues
annotate service.TaskQueues with @(
    UI : {
        SelectionFields : [
            queueName,
            queueType,
            priority
        ],
        
        LineItem : [
            {Value: queueName, Label: 'Queue Name'},
            {Value: queueType, Label: 'Type'},
            {Value: pendingTasks, Label: 'Pending'},
            {Value: processingTasks, Label: 'Processing'},
            {Value: completedTasks, Label: 'Completed'},
            {Value: priority, Label: 'Priority'},
            {Value: throughput, Label: 'Throughput'},
            {Value: averageWaitTime, Label: 'Avg Wait Time'},
            {Value: status, Label: 'Status', Criticality: statusCriticality}
        ]
    }
);

// Coordination Patterns
annotate service.CoordinationPatterns with @(
    UI : {
        SelectionFields : [
            patternName,
            patternType,
            complexity
        ],
        
        LineItem : [
            {Value: patternName, Label: 'Pattern Name'},
            {Value: patternType, Label: 'Type'},
            {Value: complexity, Label: 'Complexity'},
            {Value: agentCount, Label: 'Agents'},
            {Value: communicationOverhead, Label: 'Comm. Overhead'},
            {Value: scalability, Label: 'Scalability', Visualization: #Rating},
            {Value: reliability, Label: 'Reliability', Visualization: #Rating},
            {Value: usageCount, Label: 'Usage Count'}
        ]
    }
);

// Orchestration Events
annotate service.OrchestrationEvents with @(
    UI : {
        SelectionFields : [
            eventType,
            severity,
            timeRange
        ],
        
        LineItem : [
            {Value: timestamp, Label: 'Timestamp'},
            {Value: eventType, Label: 'Event Type'},
            {Value: workflowName, Label: 'Workflow'},
            {Value: stepName, Label: 'Step'},
            {Value: agentName, Label: 'Agent'},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: message, Label: 'Message'},
            {Value: resolution, Label: 'Resolution'}
        ]
    }
);

// Value Help annotations
annotate service.Workflows {
    workflowType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'WorkflowTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : workflowType,
            ValueListProperty : 'typeName'
        }]
    };
    
    orchestrationMode @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'OrchestrationModes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : orchestrationMode,
            ValueListProperty : 'modeName'
        }]
    };
    
    executionStrategy @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ExecutionStrategies',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : executionStrategy,
            ValueListProperty : 'strategyName'
        }]
    };
}