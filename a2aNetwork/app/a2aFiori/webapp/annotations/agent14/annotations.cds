using A2AService as service from '../../../../srv/a2a-service';

// Agent 14 - Embedding Fine-Tuner Agent UI Annotations
annotate service.EmbeddingModels with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            modelType,
            embeddingDimension,
            status,
            performanceScore,
            createdAt
        ],
        
        LineItem : [
            {Value: modelName, Label: 'Model Name'},
            {Value: modelType, Label: 'Model Type'},
            {Value: embeddingDimension, Label: 'Dimensions'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: performanceScore, Label: 'Performance', Visualization: #Rating},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: trainingLoss, Label: 'Training Loss'},
            {Value: datasetSize, Label: 'Dataset Size'},
            {Value: lastFineTuned, Label: 'Last Fine-Tuned'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Embedding Model',
            TypeNamePlural : 'Embedding Models',
            Title : {Value : modelName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://machine-learning'
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
                Target : '@UI.FieldGroup#ModelConfiguration',
                Label : 'Model Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#TrainingSettings',
                Label : 'Training Settings'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#PerformanceMetrics',
                Label : 'Performance Metrics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#OptimizationSettings',
                Label : 'Optimization Settings'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#DatasetInfo',
                Label : 'Dataset Information'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'TrainingHistory/@UI.LineItem',
                Label : 'Training History'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'EvaluationResults/@UI.LineItem',
                Label : 'Evaluation Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ModelVersions/@UI.LineItem',
                Label : 'Model Versions'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : modelName},
                {Value : description},
                {Value : modelType},
                {Value : baseModel},
                {Value : embeddingDimension},
                {Value : status},
                {Value : version}
            ]
        },
        
        FieldGroup#ModelConfiguration : {
            Data : [
                {Value : architecture},
                {Value : layerCount},
                {Value : hiddenSize},
                {Value : attentionHeads},
                {Value : vocabularySize},
                {Value : maxSequenceLength},
                {Value : tokenizer},
                {Value : normalization}
            ]
        },
        
        FieldGroup#TrainingSettings : {
            Data : [
                {Value : learningRate},
                {Value : batchSize},
                {Value : epochs},
                {Value : optimizer},
                {Value : lossFunction},
                {Value : regularization},
                {Value : dropout},
                {Value : warmupSteps},
                {Value : gradientClipping}
            ]
        },
        
        FieldGroup#PerformanceMetrics : {
            Data : [
                {Value : accuracy, Visualization : #Rating},
                {Value : precision, Visualization : #Rating},
                {Value : recall, Visualization : #Rating},
                {Value : f1Score, Visualization : #Rating},
                {Value : trainingLoss},
                {Value : validationLoss},
                {Value : performanceScore, Visualization : #Rating},
                {Value : inferenceSpeed},
                {Value : memoryUsage}
            ]
        },
        
        FieldGroup#OptimizationSettings : {
            Data : [
                {Value : quantization},
                {Value : pruning},
                {Value : distillation},
                {Value : compressionRatio},
                {Value : optimizationTarget},
                {Value : hardwareTarget},
                {Value : autoOptimization},
                {Value : mixedPrecision}
            ]
        },
        
        FieldGroup#DatasetInfo : {
            Data : [
                {Value : datasetName},
                {Value : datasetSize},
                {Value : trainingSamples},
                {Value : validationSamples},
                {Value : testSamples},
                {Value : dataAugmentation},
                {Value : samplingStrategy},
                {Value : classBalance}
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

// Training History as associated entity
annotate service.TrainingHistory with @(
    UI : {
        LineItem : [
            {Value: trainingId, Label: 'Training ID'},
            {Value: startTime, Label: 'Start Time'},
            {Value: endTime, Label: 'End Time'},
            {Value: duration, Label: 'Duration'},
            {Value: epoch, Label: 'Epoch'},
            {Value: loss, Label: 'Loss'},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: learningRate, Label: 'Learning Rate'},
            {Value: status, Label: 'Status', Criticality: statusCriticality}
        ]
    }
);

// Evaluation Results as associated entity
annotate service.EvaluationResults with @(
    UI : {
        LineItem : [
            {Value: evaluationId, Label: 'Evaluation ID'},
            {Value: evaluationType, Label: 'Evaluation Type'},
            {Value: testDataset, Label: 'Test Dataset'},
            {Value: accuracy, Label: 'Accuracy', Visualization: #Rating},
            {Value: precision, Label: 'Precision', Visualization: #Rating},
            {Value: recall, Label: 'Recall', Visualization: #Rating},
            {Value: f1Score, Label: 'F1 Score', Visualization: #Rating},
            {Value: confusionMatrix, Label: 'Confusion Matrix'},
            {Value: evaluatedAt, Label: 'Evaluated At'}
        ]
    }
);

// Model Versions as associated entity
annotate service.ModelVersions with @(
    UI : {
        LineItem : [
            {Value: versionNumber, Label: 'Version'},
            {Value: versionTag, Label: 'Tag'},
            {Value: createdAt, Label: 'Created At'},
            {Value: modelSize, Label: 'Model Size'},
            {Value: performance, Label: 'Performance', Visualization: #Rating},
            {Value: isProduction, Label: 'Production'},
            {Value: deploymentStatus, Label: 'Deployment'},
            {Value: notes, Label: 'Notes'}
        ]
    }
);

// Fine-Tuning Jobs
annotate service.FineTuningJobs with @(
    UI : {
        SelectionFields : [
            jobName,
            jobStatus,
            modelType,
            priority
        ],
        
        LineItem : [
            {Value: jobName, Label: 'Job Name'},
            {Value: modelName, Label: 'Model'},
            {Value: jobStatus, Label: 'Status', Criticality: statusCriticality},
            {Value: progress, Label: 'Progress'},
            {Value: priority, Label: 'Priority'},
            {Value: startTime, Label: 'Started'},
            {Value: estimatedCompletion, Label: 'Est. Completion'},
            {Value: resourceAllocation, Label: 'Resources'},
            {Value: createdBy, Label: 'Created By'}
        ]
    }
);

// Vector Databases
annotate service.VectorDatabases with @(
    UI : {
        SelectionFields : [
            databaseName,
            databaseType,
            isActive
        ],
        
        LineItem : [
            {Value: databaseName, Label: 'Database Name'},
            {Value: databaseType, Label: 'Type'},
            {Value: vectorCount, Label: 'Vector Count'},
            {Value: dimensionality, Label: 'Dimensions'},
            {Value: indexType, Label: 'Index Type'},
            {Value: queryPerformance, Label: 'Query Performance', Visualization: #Rating},
            {Value: storageSize, Label: 'Storage Size'},
            {Value: lastOptimized, Label: 'Last Optimized'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Benchmark Results
annotate service.BenchmarkResults with @(
    UI : {
        SelectionFields : [
            benchmarkName,
            modelType,
            dateRange
        ],
        
        LineItem : [
            {Value: benchmarkName, Label: 'Benchmark'},
            {Value: modelName, Label: 'Model'},
            {Value: score, Label: 'Score', Visualization: #Rating},
            {Value: rank, Label: 'Rank'},
            {Value: dataset, Label: 'Dataset'},
            {Value: metrics, Label: 'Metrics'},
            {Value: comparisonBaseline, Label: 'Baseline'},
            {Value: improvement, Label: 'Improvement %'},
            {Value: executedAt, Label: 'Executed At'}
        ]
    }
);

// Hyperparameter Configurations
annotate service.HyperparameterConfigs with @(
    UI : {
        SelectionFields : [
            configName,
            optimizationMethod,
            isActive
        ],
        
        LineItem : [
            {Value: configName, Label: 'Config Name'},
            {Value: optimizationMethod, Label: 'Optimization Method'},
            {Value: searchSpace, Label: 'Search Space'},
            {Value: bestScore, Label: 'Best Score', Visualization: #Rating},
            {Value: trialsCompleted, Label: 'Trials'},
            {Value: convergenceStatus, Label: 'Convergence'},
            {Value: lastUpdated, Label: 'Last Updated'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Value Help annotations
annotate service.EmbeddingModels {
    modelType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ModelTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : modelType,
            ValueListProperty : 'typeName'
        }]
    };
    
    baseModel @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'BaseModels',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : baseModel,
            ValueListProperty : 'modelName'
        }]
    };
    
    optimizer @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Optimizers',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : optimizer,
            ValueListProperty : 'optimizerName'
        }]
    };
}