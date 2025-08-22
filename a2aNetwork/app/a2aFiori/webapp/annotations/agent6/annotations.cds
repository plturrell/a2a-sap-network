using A2AService as service from '../../../../srv/a2a-service';

// Agent 6 - Quality Control Manager UI Annotations
annotate service.QualityControlTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            taskName,
            qualityGate,
            overallQuality,
            routingDecision,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: qualityGate, Label: 'Quality Gate'},
            {Value: overallQuality, Label: 'Overall Quality', Visualization: #Rating},
            {Value: componentsEvaluated, Label: 'Components'},
            {Value: issuesFound, Label: 'Issues'},
            {Value: routingDecision, Label: 'Routing Decision', Criticality: decisionCriticality},
            {Value: trustScore, Label: 'Trust Score', Visualization: #Rating},
            {Value: recommendedActions, Label: 'Actions'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Quality Control Task',
            TypeNamePlural : 'Quality Control Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://quality-issue'
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
                Target : '@UI.FieldGroup#QualityAssessment',
                Label : 'Quality Assessment'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#RoutingDecision',
                Label : 'Routing Decision'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#TrustVerification',
                Label : 'Trust Verification'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#WorkflowControl',
                Label : 'Workflow Control'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'QualityChecks/@UI.LineItem',
                Label : 'Quality Checks'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ImprovementRecommendations/@UI.LineItem',
                Label : 'Improvement Recommendations'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : qualityGate},
                {Value : dataSource},
                {Value : processingPipeline},
                {Value : overallQuality, Visualization : #Rating}
            ]
        },
        
        FieldGroup#QualityAssessment : {
            Data : [
                {Value : componentsEvaluated},
                {Value : qualityMetrics},
                {Value : complianceScore, Visualization : #Rating},
                {Value : performanceScore, Visualization : #Rating},
                {Value : securityScore, Visualization : #Rating},
                {Value : reliabilityScore, Visualization : #Rating},
                {Value : usabilityScore, Visualization : #Rating},
                {Value : maintainabilityScore, Visualization : #Rating}
            ]
        },
        
        FieldGroup#RoutingDecision : {
            Data : [
                {Value : routingDecision},
                {Value : decisionReason},
                {Value : nextAgent},
                {Value : alternativeRoutes},
                {Value : routingConfidence, Visualization : #Rating},
                {Value : estimatedProcessingTime},
                {Value : recommendedPriority}
            ]
        },
        
        FieldGroup#TrustVerification : {
            Data : [
                {Value : trustScore, Visualization : #Rating},
                {Value : trustFactors},
                {Value : verificationStatus},
                {Value : blockchainVerified},
                {Value : reputationCheck},
                {Value : integrityValidation},
                {Value : consensusRequired}
            ]
        },
        
        FieldGroup#WorkflowControl : {
            Data : [
                {Value : workflowStatus},
                {Value : currentStage},
                {Value : completionPercentage, Visualization : #Progress},
                {Value : bottlenecks},
                {Value : optimizationOpportunities},
                {Value : resourceUtilization},
                {Value : estimatedCompletion}
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

// Quality Checks as associated entity
annotate service.QualityChecks with @(
    UI : {
        LineItem : [
            {Value: checkId, Label: 'Check ID'},
            {Value: checkType, Label: 'Check Type'},
            {Value: checkName, Label: 'Check Name'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: score, Label: 'Score', Visualization: #Rating},
            {Value: threshold, Label: 'Threshold'},
            {Value: actualValue, Label: 'Actual Value'},
            {Value: deviation, Label: 'Deviation'},
            {Value: executedAt, Label: 'Executed At'}
        ]
    }
);

// Improvement Recommendations as associated entity
annotate service.ImprovementRecommendations with @(
    UI : {
        LineItem : [
            {Value: recommendationId, Label: 'ID'},
            {Value: category, Label: 'Category'},
            {Value: priority, Label: 'Priority', Criticality: priorityCriticality},
            {Value: description, Label: 'Description'},
            {Value: expectedImpact, Label: 'Expected Impact'},
            {Value: implementationEffort, Label: 'Effort'},
            {Value: status, Label: 'Status'},
            {Value: createdAt, Label: 'Created'}
        ]
    }
);

// Quality Gates
annotate service.QualityGates with @(
    UI : {
        SelectionFields : [
            gateName,
            category,
            severity
        ],
        
        LineItem : [
            {Value: gateName, Label: 'Gate Name'},
            {Value: category, Label: 'Category'},
            {Value: severity, Label: 'Severity'},
            {Value: thresholds, Label: 'Thresholds'},
            {Value: isActive, Label: 'Active'},
            {Value: passRate, Label: 'Pass Rate', Visualization: #Rating},
            {Value: lastUpdated, Label: 'Last Updated'}
        ]
    }
);

// Routing Rules
annotate service.RoutingRules with @(
    UI : {
        SelectionFields : [
            ruleName,
            sourceAgent,
            targetAgent
        ],
        
        LineItem : [
            {Value: ruleName, Label: 'Rule Name'},
            {Value: sourceAgent, Label: 'Source Agent'},
            {Value: targetAgent, Label: 'Target Agent'},
            {Value: conditions, Label: 'Conditions'},
            {Value: priority, Label: 'Priority'},
            {Value: isActive, Label: 'Active'},
            {Value: usageCount, Label: 'Usage Count'}
        ]
    }
);

// Value Help annotations
annotate service.QualityControlTasks {
    qualityGate @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'QualityGates',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : qualityGate,
            ValueListProperty : 'gateName'
        }]
    };
    
    routingDecision @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'RoutingDecisions',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : routingDecision,
            ValueListProperty : 'decision'
        }]
    };
    
    nextAgent @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Agents',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : nextAgent,
            ValueListProperty : 'name'
        }]
    };
}