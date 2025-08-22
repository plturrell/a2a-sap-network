using A2AService as service from '../../../../srv/a2a-service';

// Agent 9 - Reasoning Agent UI Annotations
annotate service.ReasoningTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            reasoningType,
            problemDomain,
            reasoningEngine,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: reasoningType, Label: 'Reasoning Type'},
            {Value: problemDomain, Label: 'Problem Domain'},
            {Value: reasoningEngine, Label: 'Engine'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: confidenceScore, Label: 'Confidence', Visualization: #Rating},
            {Value: factsProcessed, Label: 'Facts'},
            {Value: inferencesGenerated, Label: 'Inferences'},
            {Value: processingTime, Label: 'Duration'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Reasoning Task',
            TypeNamePlural : 'Reasoning Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://brain'
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
                Target : '@UI.FieldGroup#ReasoningConfig',
                Label : 'Reasoning Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#KnowledgeBase',
                Label : 'Knowledge Base'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#InferenceResults',
                Label : 'Inference Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#DecisionMaking',
                Label : 'Decision Making'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Performance',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'LogicalFacts/@UI.LineItem',
                Label : 'Logical Facts'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ReasoningRules/@UI.LineItem',
                Label : 'Reasoning Rules'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'Inferences/@UI.LineItem',
                Label : 'Generated Inferences'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : reasoningType},
                {Value : problemDomain},
                {Value : status},
                {Value : priority}
            ]
        },
        
        FieldGroup#ReasoningConfig : {
            Data : [
                {Value : reasoningEngine},
                {Value : confidenceThreshold},
                {Value : maxInferenceDepth},
                {Value : chainingStrategy},
                {Value : uncertaintyHandling},
                {Value : probabilisticModel},
                {Value : logicalFramework},
                {Value : parallelReasoning}
            ]
        },
        
        FieldGroup#KnowledgeBase : {
            Data : [
                {Value : knowledgeSource},
                {Value : factsLoaded},
                {Value : rulesLoaded},
                {Value : ontologyVersion},
                {Value : knowledgeConsistency, Visualization : #Rating},
                {Value : lastKnowledgeUpdate},
                {Value : domainExpertise},
                {Value : knowledgeCompleteness, Visualization : #Rating}
            ]
        },
        
        FieldGroup#InferenceResults : {
            Data : [
                {Value : factsProcessed},
                {Value : inferencesGenerated},
                {Value : conclusionsReached},
                {Value : confidenceScore, Visualization : #Rating},
                {Value : certaintyLevel},
                {Value : contradictionsFound},
                {Value : explanationDepth},
                {Value : validationStatus}
            ]
        },
        
        FieldGroup#DecisionMaking : {
            Data : [
                {Value : decisionCriteria},
                {Value : alternativesEvaluated},
                {Value : recommendedAction},
                {Value : decisionConfidence, Visualization : #Rating},
                {Value : riskAssessment},
                {Value : impactAnalysis},
                {Value : justification},
                {Value : fallbackOptions}
            ]
        },
        
        FieldGroup#Performance : {
            Data : [
                {Value : processingTime},
                {Value : memoryUsage},
                {Value : reasoningComplexity},
                {Value : engineEfficiency, Visualization : #Rating},
                {Value : convergenceTime},
                {Value : solutionOptimality, Visualization : #Rating},
                {Value : resourceUtilization},
                {Value : scalabilityScore, Visualization : #Rating}
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

// Logical Facts as associated entity
annotate service.LogicalFacts with @(
    UI : {
        LineItem : [
            {Value: factId, Label: 'Fact ID'},
            {Value: factType, Label: 'Type'},
            {Value: factStatement, Label: 'Statement'},
            {Value: confidence, Label: 'Confidence', Visualization: #Rating},
            {Value: source, Label: 'Source'},
            {Value: truthValue, Label: 'Truth Value'},
            {Value: temporalScope, Label: 'Temporal Scope'},
            {Value: createdAt, Label: 'Created At'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Reasoning Rules as associated entity
annotate service.ReasoningRules with @(
    UI : {
        LineItem : [
            {Value: ruleId, Label: 'Rule ID'},
            {Value: ruleName, Label: 'Rule Name'},
            {Value: ruleType, Label: 'Type'},
            {Value: condition, Label: 'Condition'},
            {Value: conclusion, Label: 'Conclusion'},
            {Value: priority, Label: 'Priority'},
            {Value: applicableCount, Label: 'Applications'},
            {Value: successRate, Label: 'Success Rate', Visualization: #Rating},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Inferences as associated entity
annotate service.Inferences with @(
    UI : {
        LineItem : [
            {Value: inferenceId, Label: 'Inference ID'},
            {Value: inferenceType, Label: 'Type'},
            {Value: premise, Label: 'Premise'},
            {Value: conclusion, Label: 'Conclusion'},
            {Value: confidence, Label: 'Confidence', Visualization: #Rating},
            {Value: supportingFacts, Label: 'Supporting Facts'},
            {Value: reasoning_chain, Label: 'Reasoning Chain'},
            {Value: validatedBy, Label: 'Validated By'},
            {Value: generatedAt, Label: 'Generated At'}
        ]
    }
);

// Reasoning Engines
annotate service.ReasoningEngines with @(
    UI : {
        SelectionFields : [
            engineName,
            engineType,
            isActive
        ],
        
        LineItem : [
            {Value: engineName, Label: 'Engine Name'},
            {Value: engineType, Label: 'Type'},
            {Value: capabilities, Label: 'Capabilities'},
            {Value: complexityHandling, Label: 'Complexity'},
            {Value: performanceRating, Label: 'Performance', Visualization: #Rating},
            {Value: accuracyRating, Label: 'Accuracy', Visualization: #Rating},
            {Value: isActive, Label: 'Active'},
            {Value: lastUsed, Label: 'Last Used'}
        ]
    }
);

// Problem Domains
annotate service.ProblemDomains with @(
    UI : {
        SelectionFields : [
            domainName,
            domainCategory,
            complexityLevel
        ],
        
        LineItem : [
            {Value: domainName, Label: 'Domain Name'},
            {Value: domainCategory, Label: 'Category'},
            {Value: complexityLevel, Label: 'Complexity'},
            {Value: knowledgeRequirement, Label: 'Knowledge Req.'},
            {Value: reasoningDepth, Label: 'Reasoning Depth'},
            {Value: expertiseLevel, Label: 'Expertise Level'},
            {Value: successRate, Label: 'Success Rate', Visualization: #Rating}
        ]
    }
);

// Value Help annotations
annotate service.ReasoningTasks {
    reasoningType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ReasoningTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : reasoningType,
            ValueListProperty : 'typeName'
        }]
    };
    
    reasoningEngine @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ReasoningEngines',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : reasoningEngine,
            ValueListProperty : 'engineName'
        }]
    };
    
    problemDomain @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ProblemDomains',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : problemDomain,
            ValueListProperty : 'domainName'
        }]
    };
}