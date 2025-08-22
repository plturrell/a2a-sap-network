using A2AService as service from '../../../../srv/a2a-service';

// Agent 2 - AI Preparation Agent UI Annotations
annotate service.AIPreparationTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            taskName,
            modelType,
            status,
            dataType,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: datasetName, Label: 'Dataset'},
            {Value: modelType, Label: 'Model Type'},
            {Value: dataType, Label: 'Data Type'},
            {Value: recordCount, Label: 'Records'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: progress, Label: 'Progress', Visualization: #Progress},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'AI Preparation Task',
            TypeNamePlural : 'AI Preparation Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://artificial-intelligence'
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
                Target : '@UI.FieldGroup#DataPreparation',
                Label : 'Data Preparation'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#FeatureEngineering',
                Label : 'Feature Engineering'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#ModelConfiguration',
                Label : 'Model Configuration'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Preprocessing',
                Label : 'Preprocessing Pipeline'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'Features/@UI.LineItem',
                Label : 'Features'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : datasetName},
                {Value : modelType},
                {Value : dataType},
                {Value : status}
            ]
        },
        
        FieldGroup#DataPreparation : {
            Data : [
                {Value : recordCount},
                {Value : featureCount},
                {Value : targetVariable},
                {Value : splitRatio},
                {Value : validationStrategy},
                {Value : randomSeed}
            ]
        },
        
        FieldGroup#FeatureEngineering : {
            Data : [
                {Value : featureSelectionMethod},
                {Value : dimensionalityReduction},
                {Value : featureScaling},
                {Value : encodingStrategy},
                {Value : missingValueStrategy},
                {Value : outlierDetection}
            ]
        },
        
        FieldGroup#ModelConfiguration : {
            Data : [
                {Value : framework},
                {Value : modelArchitecture},
                {Value : hyperparameters},
                {Value : optimizationMetric},
                {Value : embeddingDimensions},
                {Value : batchSize}
            ]
        },
        
        FieldGroup#Preprocessing : {
            Data : [
                {Value : textNormalization},
                {Value : tokenization},
                {Value : imageAugmentation},
                {Value : audioProcessing},
                {Value : timeSeriesWindowing},
                {Value : dataBalancing}
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

// Features as associated entity
annotate service.Features with @(
    UI : {
        LineItem : [
            {Value: featureName, Label: 'Feature Name'},
            {Value: dataType, Label: 'Data Type'},
            {Value: importance, Label: 'Importance', Visualization: #Rating},
            {Value: nullPercentage, Label: 'Null %'},
            {Value: uniqueValues, Label: 'Unique Values'},
            {Value: statistics, Label: 'Statistics'}
        ]
    }
);

// Model Templates
annotate service.ModelTemplates with @(
    UI : {
        SelectionFields : [
            templateName,
            modelType,
            framework,
            useCase
        ],
        
        LineItem : [
            {Value: templateName, Label: 'Template Name'},
            {Value: modelType, Label: 'Model Type'},
            {Value: framework, Label: 'Framework'},
            {Value: useCase, Label: 'Use Case'},
            {Value: accuracy, Label: 'Baseline Accuracy'},
            {Value: popularity, Label: 'Usage', Visualization: #Rating}
        ]
    }
);

// Value Help annotations
annotate service.AIPreparationTasks {
    modelType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ModelTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : modelType,
            ValueListProperty : 'code'
        }]
    };
    
    dataType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'DataTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : dataType,
            ValueListProperty : 'code'
        }]
    };
    
    framework @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'MLFrameworks',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : framework,
            ValueListProperty : 'code'
        }]
    };
}