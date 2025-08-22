using A2AService as service from '../../../../srv/a2a-service';

// Agent 1 - Data Standardization Agent UI Annotations
annotate service.StandardizationTasks with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            sourceFormat,
            targetFormat,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: taskName, Label: 'Task Name'},
            {Value: sourceFormat, Label: 'Source Format'},
            {Value: targetFormat, Label: 'Target Format'},
            {Value: recordsProcessed, Label: 'Records'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: completionRate, Label: 'Progress', Visualization: #Progress},
            {Value: createdAt, Label: 'Started'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Standardization Task',
            TypeNamePlural : 'Standardization Tasks',
            Title : {Value : taskName},
            Description : {Value : description},
            ImageUrl : 'sap-icon://synchronize'
        },
        
        // Facets for Object Page
        Facets : [
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Overview',
                Label : 'Overview'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#SchemaMapping',
                Label : 'Schema Mapping'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Validation',
                Label : 'Validation Rules'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Results',
                Label : 'Standardization Results'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ValidationErrors/@UI.LineItem',
                Label : 'Validation Errors'
            }
        ],
        
        // Field Groups
        FieldGroup#Overview : {
            Data : [
                {Value : taskName},
                {Value : description},
                {Value : sourceFormat},
                {Value : targetFormat},
                {Value : status},
                {Value : completionRate, Visualization : #Progress}
            ]
        },
        
        FieldGroup#SchemaMapping : {
            Data : [
                {Value : sourceSchema},
                {Value : targetSchema},
                {Value : mappingRules},
                {Value : transformationScript},
                {Value : customMappings}
            ]
        },
        
        FieldGroup#Validation : {
            Data : [
                {Value : schemaValidation},
                {Value : dataTypeValidation},
                {Value : formatValidation},
                {Value : businessRuleValidation},
                {Value : validationThreshold}
            ]
        },
        
        FieldGroup#Results : {
            Data : [
                {Value : recordsProcessed},
                {Value : recordsSuccessful},
                {Value : recordsFailed},
                {Value : processingTime},
                {Value : averageRecordTime},
                {Value : outputFileSize}
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

// Validation Errors as associated entity
annotate service.ValidationErrors with @(
    UI : {
        LineItem : [
            {Value: recordId, Label: 'Record ID'},
            {Value: fieldName, Label: 'Field'},
            {Value: errorType, Label: 'Error Type', Criticality: criticalityIndicator},
            {Value: errorMessage, Label: 'Message'},
            {Value: suggestedFix, Label: 'Suggested Fix'}
        ]
    }
);

// Schema Templates
annotate service.SchemaTemplates with @(
    UI : {
        SelectionFields : [
            templateName,
            format,
            industry,
            version
        ],
        
        LineItem : [
            {Value: templateName, Label: 'Template Name'},
            {Value: format, Label: 'Format'},
            {Value: industry, Label: 'Industry'},
            {Value: version, Label: 'Version'},
            {Value: isActive, Label: 'Active'},
            {Value: usageCount, Label: 'Usage Count'}
        ]
    }
);

// Value Help annotations
annotate service.StandardizationTasks {
    sourceFormat @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'DataFormats',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : sourceFormat,
            ValueListProperty : 'code'
        }]
    };
    
    targetFormat @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'DataFormats',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : targetFormat,
            ValueListProperty : 'code'
        }]
    };
    
    status @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'TaskStatuses',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : status,
            ValueListProperty : 'code'
        }]
    };
}