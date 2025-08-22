using A2AService as service from '../../../../srv/a2a-service';

// Agent 8 - Data Manager UI Annotations
annotate service.DataManagementTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            datasetName,
            operationType,
            storageType,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: datasetName, Label: 'Dataset Name'},
            {Value: operationType, Label: 'Operation Type'},
            {Value: storageType, Label: 'Storage Type'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: dataSize, Label: 'Data Size'},
            {Value: recordCount, Label: 'Records'},
            {Value: compressionRatio, Label: 'Compression', Visualization: #Rating},
            {Value: cacheHitRate, Label: 'Cache Hit Rate', Visualization: #Rating},
            {Value: lastModified, Label: 'Last Modified'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Data Management Task',
            TypeNamePlural : 'Data Management Tasks',
            Title : {Value : taskName},
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
                Target : '@UI.FieldGroup#StorageDetails',
                Label : 'Storage Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#CacheManagement',
                Label : 'Cache Management'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#VersionControl',
                Label : 'Version Control'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Performance',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'DataVersions/@UI.LineItem',
                Label : 'Data Versions'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'CacheEntries/@UI.LineItem',
                Label : 'Cache Entries'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'StorageOperations/@UI.LineItem',
                Label : 'Storage Operations'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : datasetName},
                {Value : operationType},
                {Value : status},
                {Value : priority}
            ]
        },
        
        FieldGroup#StorageDetails : {
            Data : [
                {Value : storageType},
                {Value : storageBackend},
                {Value : dataSize},
                {Value : recordCount},
                {Value : compressionType},
                {Value : compressionRatio, Visualization : #Rating},
                {Value : encryptionEnabled},
                {Value : partitionStrategy},
                {Value : retentionPolicy}
            ]
        },
        
        FieldGroup#CacheManagement : {
            Data : [
                {Value : cacheLevel},
                {Value : cacheHitRate, Visualization : #Rating},
                {Value : cacheMissRate, Visualization : #Rating},
                {Value : memoryCacheSize},
                {Value : redisCacheSize},
                {Value : cacheEvictionPolicy},
                {Value : cacheTTL},
                {Value : lastCacheRefresh}
            ]
        },
        
        FieldGroup#VersionControl : {
            Data : [
                {Value : currentVersion},
                {Value : versionStrategy},
                {Value : versionCount},
                {Value : retentionDays},
                {Value : lastVersionCreated},
                {Value : versionSize},
                {Value : incrementalBackup},
                {Value : checksumValidation}
            ]
        },
        
        FieldGroup#Performance : {
            Data : [
                {Value : readThroughput},
                {Value : writeThroughput},
                {Value : averageResponseTime},
                {Value : connectionPoolSize},
                {Value : activeConnections},
                {Value : diskIOPS},
                {Value : networkLatency},
                {Value : performanceScore, Visualization : #Rating}
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

// Data Versions as associated entity
annotate service.DataVersions with @(
    UI : {
        LineItem : [
            {Value: versionId, Label: 'Version ID'},
            {Value: versionNumber, Label: 'Version'},
            {Value: versionType, Label: 'Type'},
            {Value: dataSize, Label: 'Size'},
            {Value: changesSummary, Label: 'Changes'},
            {Value: checksum, Label: 'Checksum'},
            {Value: createdBy, Label: 'Created By'},
            {Value: createdAt, Label: 'Created At'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Cache Entries as associated entity
annotate service.CacheEntries with @(
    UI : {
        LineItem : [
            {Value: cacheKey, Label: 'Cache Key'},
            {Value: cacheLevel, Label: 'Cache Level'},
            {Value: dataSize, Label: 'Size'},
            {Value: hitCount, Label: 'Hit Count'},
            {Value: expiresAt, Label: 'Expires At'},
            {Value: lastAccessed, Label: 'Last Accessed'},
            {Value: isExpired, Label: 'Expired'},
            {Value: evictionScore, Label: 'Eviction Score'}
        ]
    }
);

// Storage Operations as associated entity
annotate service.StorageOperations with @(
    UI : {
        LineItem : [
            {Value: operationId, Label: 'Operation ID'},
            {Value: operationType, Label: 'Operation'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: recordsProcessed, Label: 'Records'},
            {Value: duration, Label: 'Duration'},
            {Value: throughput, Label: 'Throughput'},
            {Value: errorCount, Label: 'Errors'},
            {Value: executedAt, Label: 'Executed At'}
        ]
    }
);

// Storage Backends
annotate service.StorageBackends with @(
    UI : {
        SelectionFields : [
            backendName,
            backendType,
            isActive
        ],
        
        LineItem : [
            {Value: backendName, Label: 'Backend Name'},
            {Value: backendType, Label: 'Type'},
            {Value: endpoint, Label: 'Endpoint'},
            {Value: capacity, Label: 'Capacity'},
            {Value: usedSpace, Label: 'Used Space'},
            {Value: utilization, Label: 'Utilization', Visualization: #Rating},
            {Value: isActive, Label: 'Active'},
            {Value: healthStatus, Label: 'Health Status'}
        ]
    }
);

// Data Schemas
annotate service.DataSchemas with @(
    UI : {
        SelectionFields : [
            schemaName,
            schemaVersion,
            isActive
        ],
        
        LineItem : [
            {Value: schemaName, Label: 'Schema Name'},
            {Value: schemaVersion, Label: 'Version'},
            {Value: fieldCount, Label: 'Fields'},
            {Value: dataType, Label: 'Data Type'},
            {Value: validationRules, Label: 'Validation Rules'},
            {Value: isActive, Label: 'Active'},
            {Value: lastModified, Label: 'Last Modified'}
        ]
    }
);

// Value Help annotations
annotate service.DataManagementTasks {
    datasetName @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Datasets',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : datasetName,
            ValueListProperty : 'datasetName'
        }]
    };
    
    storageType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'StorageTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : storageType,
            ValueListProperty : 'typeName'
        }]
    };
    
    storageBackend @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'StorageBackends',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : storageBackend,
            ValueListProperty : 'backendName'
        }]
    };
}