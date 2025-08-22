using A2AService as service from '../../../../srv/a2a-service';

// Agent 3 - Vector Processing Agent UI Annotations
annotate service.VectorProcessingTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            taskName,
            embeddingModel,
            vectorDatabase,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: dataSource, Label: 'Data Source'},
            {Value: embeddingModel, Label: 'Embedding Model'},
            {Value: vectorCount, Label: 'Vectors'},
            {Value: dimensions, Label: 'Dimensions'},
            {Value: vectorDatabase, Label: 'Vector DB'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: progress, Label: 'Progress', Visualization: #Progress},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Vector Processing Task',
            TypeNamePlural : 'Vector Processing Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://scatter-chart'
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
                Target : '@UI.FieldGroup#EmbeddingConfiguration',
                Label : 'Embedding Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#VectorDatabase',
                Label : 'Vector Database'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Processing',
                Label : 'Processing Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#SearchConfiguration',
                Label : 'Search Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'SimilarityResults/@UI.LineItem',
                Label : 'Similarity Search Results'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : dataSource},
                {Value : dataType},
                {Value : status},
                {Value : progress, Visualization : #Progress}
            ]
        },
        
        FieldGroup#EmbeddingConfiguration : {
            Data : [
                {Value : embeddingModel},
                {Value : modelProvider},
                {Value : dimensions},
                {Value : normalization},
                {Value : chunkSize},
                {Value : chunkOverlap},
                {Value : batchSize},
                {Value : maxTokens}
            ]
        },
        
        FieldGroup#VectorDatabase : {
            Data : [
                {Value : vectorDatabase},
                {Value : indexType},
                {Value : distanceMetric},
                {Value : vectorCount},
                {Value : indexSize},
                {Value : queryLatency},
                {Value : compressionEnabled},
                {Value : shardCount}
            ]
        },
        
        FieldGroup#Processing : {
            Data : [
                {Value : processingStartTime},
                {Value : processingEndTime},
                {Value : processingDuration},
                {Value : vectorsPerSecond},
                {Value : cpuUsage},
                {Value : memoryUsage},
                {Value : gpuUsage},
                {Value : errorCount}
            ]
        },
        
        FieldGroup#SearchConfiguration : {
            Data : [
                {Value : searchAlgorithm},
                {Value : topK},
                {Value : similarityThreshold},
                {Value : preFilterEnabled},
                {Value : postFilterEnabled},
                {Value : reranking},
                {Value : hybridSearch}
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

// Similarity Results as associated entity
annotate service.SimilarityResults with @(
    UI : {
        LineItem : [
            {Value: documentId, Label: 'Document ID'},
            {Value: content, Label: 'Content'},
            {Value: similarity, Label: 'Similarity Score', Visualization: #Rating},
            {Value: distance, Label: 'Distance'},
            {Value: metadata, Label: 'Metadata'}
        ]
    }
);

// Vector Collections
annotate service.VectorCollections with @(
    UI : {
        SelectionFields : [
            collectionName,
            database,
            status
        ],
        
        LineItem : [
            {Value: collectionName, Label: 'Collection Name'},
            {Value: database, Label: 'Database'},
            {Value: vectorCount, Label: 'Vector Count'},
            {Value: dimensions, Label: 'Dimensions'},
            {Value: indexType, Label: 'Index Type'},
            {Value: lastUpdated, Label: 'Last Updated'},
            {Value: status, Label: 'Status'}
        ]
    }
);

// Embedding Models
annotate service.EmbeddingModels with @(
    UI : {
        SelectionFields : [
            modelName,
            provider,
            type
        ],
        
        LineItem : [
            {Value: modelName, Label: 'Model Name'},
            {Value: provider, Label: 'Provider'},
            {Value: type, Label: 'Type'},
            {Value: dimensions, Label: 'Output Dimensions'},
            {Value: maxTokens, Label: 'Max Tokens'},
            {Value: performance, Label: 'Performance', Visualization: #Rating}
        ]
    }
);

// Value Help annotations
annotate service.VectorProcessingTasks {
    embeddingModel @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'EmbeddingModels',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : embeddingModel,
            ValueListProperty : 'modelName'
        }]
    };
    
    vectorDatabase @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'VectorDatabases',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : vectorDatabase,
            ValueListProperty : 'name'
        }]
    };
    
    indexType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'IndexTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : indexType,
            ValueListProperty : 'type'
        }]
    };
}