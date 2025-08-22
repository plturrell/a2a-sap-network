using A2AService as service from '../../../../srv/a2a-service';

// Agent 10 - Calculation Agent UI Annotations
annotate service.CalculationTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            calculationType,
            formulaCategory,
            precisionLevel,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: calculationType, Label: 'Calculation Type'},
            {Value: formulaCategory, Label: 'Formula Category'},
            {Value: precisionLevel, Label: 'Precision Level'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: inputParameters, Label: 'Parameters'},
            {Value: resultValue, Label: 'Result'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Calculation Task',
            TypeNamePlural : 'Calculation Tasks',
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
                Target : '@UI.FieldGroup#CalculationConfig',
                Label : 'Calculation Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#FormulaDetails',
                Label : 'Formula Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#ResultsAnalysis',
                Label : 'Results & Analysis'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Performance',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#SelfHealing',
                Label : 'Self-Healing Status'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'FormulaParameters/@UI.LineItem',
                Label : 'Formula Parameters'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'CalculationResults/@UI.LineItem',
                Label : 'Calculation Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ValidationChecks/@UI.LineItem',
                Label : 'Validation Checks'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : calculationType},
                {Value : formulaCategory},
                {Value : status},
                {Value : priority}
            ]
        },
        
        FieldGroup#CalculationConfig : {
            Data : [
                {Value : precisionLevel},
                {Value : calculationEngine},
                {Value : parallelProcessing},
                {Value : threadCount},
                {Value : gpuAcceleration},
                {Value : cachingEnabled},
                {Value : resultValidation},
                {Value : confidenceInterval}
            ]
        },
        
        FieldGroup#FormulaDetails : {
            Data : [
                {Value : formulaExpression},
                {Value : formulaLanguage},
                {Value : inputParameters},
                {Value : expectedDataType},
                {Value : formulaComplexity},
                {Value : dependentVariables},
                {Value : formulaSource},
                {Value : lastModified}
            ]
        },
        
        FieldGroup#ResultsAnalysis : {
            Data : [
                {Value : resultValue},
                {Value : resultDataType},
                {Value : accuracy, Visualization : #Rating},
                {Value : precision},
                {Value : confidenceScore, Visualization : #Rating},
                {Value : errorMargin},
                {Value : statisticalSignificance},
                {Value : resultInterpretation}
            ]
        },
        
        FieldGroup#Performance : {
            Data : [
                {Value : executionTime},
                {Value : memoryUsage},
                {Value : cpuUtilization},
                {Value : throughput},
                {Value : operationsPerSecond},
                {Value : cacheHitRate, Visualization : #Rating},
                {Value : scalabilityScore, Visualization : #Rating},
                {Value : resourceEfficiency, Visualization : #Rating}
            ]
        },
        
        FieldGroup#SelfHealing : {
            Data : [
                {Value : selfHealingEnabled},
                {Value : redundantCalculations},
                {Value : consistencyChecks},
                {Value : boundaryValidation},
                {Value : precisionVerification},
                {Value : errorCorrection},
                {Value : healingAttempts},
                {Value : healingSuccessRate, Visualization : #Rating}
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

// Formula Parameters as associated entity
annotate service.FormulaParameters with @(
    UI : {
        LineItem : [
            {Value: parameterName, Label: 'Parameter Name'},
            {Value: parameterType, Label: 'Type'},
            {Value: parameterValue, Label: 'Value'},
            {Value: dataType, Label: 'Data Type'},
            {Value: unit, Label: 'Unit'},
            {Value: constraints, Label: 'Constraints'},
            {Value: defaultValue, Label: 'Default Value'},
            {Value: isRequired, Label: 'Required'},
            {Value: description, Label: 'Description'}
        ]
    }
);

// Calculation Results as associated entity
annotate service.CalculationResults with @(
    UI : {
        LineItem : [
            {Value: resultId, Label: 'Result ID'},
            {Value: calculationStep, Label: 'Step'},
            {Value: intermediateResult, Label: 'Intermediate Result'},
            {Value: finalResult, Label: 'Final Result'},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: memoryUsed, Label: 'Memory Used'},
            {Value: validationStatus, Label: 'Validation'},
            {Value: timestamp, Label: 'Timestamp'}
        ]
    }
);

// Validation Checks as associated entity
annotate service.ValidationChecks with @(
    UI : {
        LineItem : [
            {Value: checkId, Label: 'Check ID'},
            {Value: checkType, Label: 'Check Type'},
            {Value: checkResult, Label: 'Result'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: expectedValue, Label: 'Expected'},
            {Value: actualValue, Label: 'Actual'},
            {Value: deviation, Label: 'Deviation'},
            {Value: tolerance, Label: 'Tolerance'},
            {Value: executedAt, Label: 'Executed At'}
        ]
    }
);

// Calculation Engines
annotate service.CalculationEngines with @(
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
            {Value: supportedPrecision, Label: 'Precision'},
            {Value: parallelSupport, Label: 'Parallel Support'},
            {Value: gpuSupport, Label: 'GPU Support'},
            {Value: performanceRating, Label: 'Performance', Visualization: #Rating},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Formula Categories
annotate service.FormulaCategories with @(
    UI : {
        SelectionFields : [
            categoryName,
            domain,
            complexityLevel
        ],
        
        LineItem : [
            {Value: categoryName, Label: 'Category'},
            {Value: domain, Label: 'Domain'},
            {Value: complexityLevel, Label: 'Complexity'},
            {Value: formulaCount, Label: 'Formula Count'},
            {Value: description, Label: 'Description'},
            {Value: usageFrequency, Label: 'Usage Frequency'},
            {Value: averageExecutionTime, Label: 'Avg Execution Time'}
        ]
    }
);

// Mathematical Functions
annotate service.MathematicalFunctions with @(
    UI : {
        SelectionFields : [
            functionName,
            functionType,
            domain
        ],
        
        LineItem : [
            {Value: functionName, Label: 'Function Name'},
            {Value: functionType, Label: 'Type'},
            {Value: domain, Label: 'Domain'},
            {Value: syntax, Label: 'Syntax'},
            {Value: description, Label: 'Description'},
            {Value: examples, Label: 'Examples'},
            {Value: complexity, Label: 'Complexity'},
            {Value: usageCount, Label: 'Usage Count'}
        ]
    }
);

// Value Help annotations
annotate service.CalculationTasks {
    calculationType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'CalculationTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : calculationType,
            ValueListProperty : 'typeName'
        }]
    };
    
    formulaCategory @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'FormulaCategories',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : formulaCategory,
            ValueListProperty : 'categoryName'
        }]
    };
    
    calculationEngine @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'CalculationEngines',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : calculationEngine,
            ValueListProperty : 'engineName'
        }]
    };
}