using A2AService as service from '../../../../srv/a2a-service';

// Agent 11 - SQL Agent UI Annotations
annotate service.SQLQueries with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            queryType,
            databaseType,
            executionStatus,
            complexity,
            createdAt
        ],
        
        LineItem : [
            {Value: queryName, Label: 'Query Name'},
            {Value: queryType, Label: 'Query Type'},
            {Value: databaseType, Label: 'Database'},
            {Value: executionStatus, Label: 'Status', Criticality: statusCriticality},
            {Value: complexity, Label: 'Complexity'},
            {Value: executionTime, Label: 'Execution Time'},
            {Value: rowsAffected, Label: 'Rows Affected'},
            {Value: optimizationScore, Label: 'Optimization', Visualization: #Rating},
            {Value: performance, Label: 'Performance', Visualization: #Rating},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'SQL Query',
            TypeNamePlural : 'SQL Queries',
            Title : {Value : queryName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://database'
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
                Target : '@UI.FieldGroup#QueryDetails',
                Label : 'Query Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#DatabaseConfig',
                Label : 'Database Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#ExecutionMetrics',
                Label : 'Execution Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Optimization',
                Label : 'Query Optimization'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Security',
                Label : 'Security & Compliance'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'QueryParameters/@UI.LineItem',
                Label : 'Query Parameters'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'QueryResults/@UI.LineItem',
                Label : 'Query Results'
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
                {Value : queryName},
                {Value : description},
                {Value : queryType},
                {Value : databaseType},
                {Value : executionStatus},
                {Value : priority}
            ]
        },
        
        FieldGroup#QueryDetails : {
            Data : [
                {Value : sqlStatement},
                {Value : queryLanguage},
                {Value : dialectVersion},
                {Value : complexity},
                {Value : estimatedCost},
                {Value : tableCount},
                {Value : joinCount},
                {Value : subqueryCount},
                {Value : indexUsage}
            ]
        },
        
        FieldGroup#DatabaseConfig : {
            Data : [
                {Value : connectionString},
                {Value : schemaName},
                {Value : databaseVersion},
                {Value : connectionPool},
                {Value : timeoutSettings},
                {Value : transactionMode},
                {Value : isolationLevel},
                {Value : autoCommit}
            ]
        },
        
        FieldGroup#ExecutionMetrics : {
            Data : [
                {Value : executionTime},
                {Value : cpuTime},
                {Value : ioTime},
                {Value : memoryUsage},
                {Value : rowsAffected},
                {Value : rowsReturned},
                {Value : blocksRead},
                {Value : blocksWritten},
                {Value : cacheHitRatio, Visualization : #Rating}
            ]
        },
        
        FieldGroup#Optimization : {
            Data : [
                {Value : optimizationScore, Visualization : #Rating},
                {Value : queryPlan},
                {Value : indexRecommendations},
                {Value : statisticsUpdated},
                {Value : parallelExecution},
                {Value : partitionPruning},
                {Value : joinOptimization},
                {Value : costBasedOptimizer}
            ]
        },
        
        FieldGroup#Security : {
            Data : [
                {Value : securityValidation},
                {Value : sqlInjectionCheck},
                {Value : accessControlValidation},
                {Value : dataClassification},
                {Value : auditingEnabled},
                {Value : encryptionRequired},
                {Value : complianceScore, Visualization : #Rating},
                {Value : sensitiveDataDetected}
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

// Query Parameters as associated entity
annotate service.QueryParameters with @(
    UI : {
        LineItem : [
            {Value: parameterName, Label: 'Parameter Name'},
            {Value: parameterType, Label: 'Type'},
            {Value: parameterValue, Label: 'Value'},
            {Value: dataType, Label: 'Data Type'},
            {Value: isRequired, Label: 'Required'},
            {Value: defaultValue, Label: 'Default Value'},
            {Value: validation, Label: 'Validation'},
            {Value: description, Label: 'Description'}
        ]
    }
);

// Query Results as associated entity
annotate service.QueryResults with @(
    UI : {
        LineItem : [
            {Value: resultId, Label: 'Result ID'},
            {Value: columnName, Label: 'Column'},
            {Value: dataType, Label: 'Data Type'},
            {Value: sampleValue, Label: 'Sample Value'},
            {Value: rowCount, Label: 'Row Count'},
            {Value: uniqueValues, Label: 'Unique Values'},
            {Value: nullCount, Label: 'Null Count'},
            {Value: dataQuality, Label: 'Data Quality', Visualization: #Rating},
            {Value: timestamp, Label: 'Timestamp'}
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
            {Value: rowsAffected, Label: 'Rows Affected'},
            {Value: errorMessage, Label: 'Error Message'},
            {Value: executedBy, Label: 'Executed By'},
            {Value: executionContext, Label: 'Context'}
        ]
    }
);

// Database Connections
annotate service.DatabaseConnections with @(
    UI : {
        SelectionFields : [
            connectionName,
            databaseType,
            isActive,
            environment
        ],
        
        LineItem : [
            {Value: connectionName, Label: 'Connection Name'},
            {Value: databaseType, Label: 'Database Type'},
            {Value: serverHost, Label: 'Server Host'},
            {Value: databaseName, Label: 'Database Name'},
            {Value: connectionStatus, Label: 'Status', Criticality: statusCriticality},
            {Value: connectionPool, Label: 'Pool Size'},
            {Value: activeConnections, Label: 'Active Connections'},
            {Value: performance, Label: 'Performance', Visualization: #Rating},
            {Value: lastChecked, Label: 'Last Checked'}
        ]
    }
);

// Query Templates
annotate service.QueryTemplates with @(
    UI : {
        SelectionFields : [
            templateName,
            queryType,
            databaseType,
            category
        ],
        
        LineItem : [
            {Value: templateName, Label: 'Template Name'},
            {Value: queryType, Label: 'Query Type'},
            {Value: databaseType, Label: 'Database Type'},
            {Value: category, Label: 'Category'},
            {Value: complexity, Label: 'Complexity'},
            {Value: usageCount, Label: 'Usage Count'},
            {Value: rating, Label: 'Rating', Visualization: #Rating},
            {Value: lastUsed, Label: 'Last Used'}
        ]
    }
);

// Performance Metrics
annotate service.PerformanceMetrics with @(
    UI : {
        SelectionFields : [
            metricType,
            databaseType,
            timeframe
        ],
        
        LineItem : [
            {Value: metricName, Label: 'Metric Name'},
            {Value: metricType, Label: 'Type'},
            {Value: currentValue, Label: 'Current Value'},
            {Value: averageValue, Label: 'Average Value'},
            {Value: threshold, Label: 'Threshold'},
            {Value: trend, Label: 'Trend'},
            {Value: severity, Label: 'Severity', Criticality: severityCriticality},
            {Value: timestamp, Label: 'Timestamp'}
        ]
    }
);

// Value Help annotations
annotate service.SQLQueries {
    queryType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'QueryTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : queryType,
            ValueListProperty : 'typeName'
        }]
    };
    
    databaseType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'DatabaseTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : databaseType,
            ValueListProperty : 'databaseName'
        }]
    };
    
    schemaName @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'DatabaseSchemas',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : schemaName,
            ValueListProperty : 'schemaName'
        }]
    };
}