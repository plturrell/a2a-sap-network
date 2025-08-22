using A2AService as service from '../../../../srv/a2a-service';

// Agent 5 - QA Validation Agent UI Annotations
annotate service.QAValidationTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            taskName,
            testType,
            validationStatus,
            severity,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: testSuite, Label: 'Test Suite'},
            {Value: testType, Label: 'Test Type'},
            {Value: totalTests, Label: 'Total Tests'},
            {Value: passedTests, Label: 'Passed'},
            {Value: failedTests, Label: 'Failed'},
            {Value: validationStatus, Label: 'Status', Criticality: statusCriticality},
            {Value: qualityScore, Label: 'Quality Score', Visualization: #Rating},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'QA Validation Task',
            TypeNamePlural : 'QA Validation Tasks',
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
                Target : '@UI.FieldGroup#TestConfiguration',
                Label : 'Test Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#QualityMetrics',
                Label : 'Quality Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#ComplianceSettings',
                Label : 'Compliance Settings'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Results',
                Label : 'Test Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'TestCases/@UI.LineItem',
                Label : 'Test Cases'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'QualityIssues/@UI.LineItem',
                Label : 'Quality Issues'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : testSuite},
                {Value : testType},
                {Value : targetApplication},
                {Value : validationStatus},
                {Value : severity}
            ]
        },
        
        FieldGroup#TestConfiguration : {
            Data : [
                {Value : testFramework},
                {Value : testEnvironment},
                {Value : testDataSource},
                {Value : automationLevel},
                {Value : parallelExecution},
                {Value : retryOnFailure},
                {Value : testTimeout},
                {Value : reportFormat}
            ]
        },
        
        FieldGroup#QualityMetrics : {
            Data : [
                {Value : qualityScore, Visualization : #Rating},
                {Value : coveragePercentage, Visualization : #Progress},
                {Value : totalTests},
                {Value : passedTests},
                {Value : failedTests},
                {Value : skippedTests},
                {Value : testSuccessRate},
                {Value : defectDensity}
            ]
        },
        
        FieldGroup#ComplianceSettings : {
            Data : [
                {Value : complianceStandard},
                {Value : requiresApproval},
                {Value : auditTrail},
                {Value : regulatoryCompliance},
                {Value : securityValidation},
                {Value : performanceValidation},
                {Value : accessibilityTesting},
                {Value : usabilityTesting}
            ]
        },
        
        FieldGroup#Results : {
            Data : [
                {Value : executionStartTime},
                {Value : executionEndTime},
                {Value : executionDuration},
                {Value : averageTestTime},
                {Value : resourceUtilization},
                {Value : memoryUsage},
                {Value : cpuUsage},
                {Value : networkUsage}
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

// Test Cases as associated entity
annotate service.TestCases with @(
    UI : {
        LineItem : [
            {Value: testCaseId, Label: 'Test Case ID'},
            {Value: testName, Label: 'Test Name'},
            {Value: testCategory, Label: 'Category'},
            {Value: priority, Label: 'Priority', Criticality: priorityCriticality},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: result, Label: 'Result'},
            {Value: lastExecuted, Label: 'Last Executed'}
        ]
    }
);

// Quality Issues as associated entity
annotate service.QualityIssues with @(
    UI : {
        LineItem : [
            {Value: issueId, Label: 'Issue ID'},
            {Value: issueType, Label: 'Type'},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: description, Label: 'Description'},
            {Value: affectedComponent, Label: 'Component'},
            {Value: status, Label: 'Status'},
            {Value: assignedTo, Label: 'Assigned To'},
            {Value: createdAt, Label: 'Created'}
        ]
    }
);

// Test Suites
annotate service.TestSuites with @(
    UI : {
        SelectionFields : [
            suiteName,
            category,
            framework
        ],
        
        LineItem : [
            {Value: suiteName, Label: 'Suite Name'},
            {Value: category, Label: 'Category'},
            {Value: framework, Label: 'Framework'},
            {Value: testCount, Label: 'Test Count'},
            {Value: successRate, Label: 'Success Rate', Visualization: #Rating},
            {Value: avgExecutionTime, Label: 'Avg Execution Time'},
            {Value: lastRun, Label: 'Last Run'}
        ]
    }
);

// Test Templates
annotate service.TestTemplates with @(
    UI : {
        SelectionFields : [
            templateName,
            testType,
            complexity
        ],
        
        LineItem : [
            {Value: templateName, Label: 'Template Name'},
            {Value: testType, Label: 'Test Type'},
            {Value: complexity, Label: 'Complexity'},
            {Value: estimatedTime, Label: 'Est. Time'},
            {Value: usageCount, Label: 'Usage Count'},
            {Value: rating, Label: 'Rating', Visualization: #Rating}
        ]
    }
);

// Value Help annotations
annotate service.QAValidationTasks {
    testType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'TestTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : testType,
            ValueListProperty : 'type'
        }]
    };
    
    testFramework @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'TestFrameworks',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : testFramework,
            ValueListProperty : 'framework'
        }]
    };
    
    severity @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'SeverityLevels',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : severity,
            ValueListProperty : 'level'
        }]
    };
    
    complianceStandard @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ComplianceStandards',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : complianceStandard,
            ValueListProperty : 'standard'
        }]
    };
}