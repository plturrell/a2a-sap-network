using A2AService as service from '../../../../srv/a2a-service';

// Agent 4 - Calculation Validation Agent UI Annotations
annotate service.CalculationValidationTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            taskName,
            calculationType,
            validationStatus,
            priority,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: calculationType, Label: 'Calculation Type'},
            {Value: formulaCount, Label: 'Formulas'},
            {Value: validationStatus, Label: 'Status', Criticality: statusCriticality},
            {Value: accuracyScore, Label: 'Accuracy', Visualization: #Rating},
            {Value: errorCount, Label: 'Errors'},
            {Value: priority, Label: 'Priority', Criticality: priorityCriticality},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Calculation Validation Task',
            TypeNamePlural : 'Calculation Validation Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://calculator'
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
                Target : '@UI.FieldGroup#ValidationSettings',
                Label : 'Validation Settings'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Results',
                Label : 'Validation Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Performance',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'Formulas/@UI.LineItem',
                Label : 'Formulas'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ValidationErrors/@UI.LineItem',
                Label : 'Validation Errors'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : calculationType},
                {Value : dataSource},
                {Value : priority},
                {Value : validationStatus}
            ]
        },
        
        FieldGroup#ValidationSettings : {
            Data : [
                {Value : precisionThreshold},
                {Value : toleranceLevel},
                {Value : validationMode},
                {Value : comparisonMethod},
                {Value : enableCrossValidation},
                {Value : enableStatisticalTests},
                {Value : customValidationRules},
                {Value : benchmarkDataset}
            ]
        },
        
        FieldGroup#Results : {
            Data : [
                {Value : accuracyScore, Visualization : #Rating},
                {Value : formulaCount},
                {Value : validatedFormulas},
                {Value : errorCount},
                {Value : warningCount},
                {Value : passedTests},
                {Value : failedTests},
                {Value : confidenceLevel}
            ]
        },
        
        FieldGroup#Performance : {
            Data : [
                {Value : validationStartTime},
                {Value : validationEndTime},
                {Value : validationDuration},
                {Value : formulasPerSecond},
                {Value : memoryUsage},
                {Value : cpuUsage},
                {Value : resourceEfficiency}
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

// Formulas as associated entity
annotate service.Formulas with @(
    UI : {
        LineItem : [
            {Value: formulaId, Label: 'Formula ID'},
            {Value: formulaExpression, Label: 'Expression'},
            {Value: formulaType, Label: 'Type'},
            {Value: validationResult, Label: 'Result', Criticality: resultCriticality},
            {Value: expectedValue, Label: 'Expected'},
            {Value: actualValue, Label: 'Actual'},
            {Value: variance, Label: 'Variance'},
            {Value: executionTime, Label: 'Exec Time (ms)'}
        ]
    }
);

// Validation Errors as associated entity
annotate service.ValidationErrors with @(
    UI : {
        LineItem : [
            {Value: errorCode, Label: 'Error Code'},
            {Value: errorType, Label: 'Type'},
            {Value: formulaId, Label: 'Formula ID'},
            {Value: errorMessage, Label: 'Error Message'},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: suggestedFix, Label: 'Suggested Fix'},
            {Value: errorLocation, Label: 'Location'}
        ]
    }
);

// Calculation Templates
annotate service.CalculationTemplates with @(
    UI : {
        SelectionFields : [
            templateName,
            category,
            complexity
        ],
        
        LineItem : [
            {Value: templateName, Label: 'Template Name'},
            {Value: category, Label: 'Category'},
            {Value: complexity, Label: 'Complexity'},
            {Value: formulaCount, Label: 'Formula Count'},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: usageCount, Label: 'Usage Count'}
        ]
    }
);

// Value Help annotations
annotate service.CalculationValidationTasks {
    calculationType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'CalculationTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : calculationType,
            ValueListProperty : 'type'
        }]
    };
    
    validationMode @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ValidationModes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : validationMode,
            ValueListProperty : 'mode'
        }]
    };
    
    priority @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Priorities',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : priority,
            ValueListProperty : 'level'
        }]
    };
}