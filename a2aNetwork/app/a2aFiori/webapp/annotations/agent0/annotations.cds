using A2AService as service from '../../../../srv/a2a-service';

// Agent 0 - Data Product Agent UI Annotations
annotate service.DataProducts with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            name,
            status,
            createdAt,
            format
        ],
        
        LineItem : [
            {Value: name, Label: 'Product Name'},
            {Value: description, Label: 'Description'},
            {Value: format, Label: 'Format'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: qualityScore, Label: 'Quality', Visualization: #Rating},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Data Product',
            TypeNamePlural : 'Data Products',
            Title : {Value : name},
            Description : {Value : description},
            ImageUrl : 'sap-icon://product'
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
                Target : '@UI.FieldGroup#Metadata',
                Label : 'Dublin Core Metadata'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Quality',
                Label : 'Quality & Validation'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Processing',
                Label : 'Processing Information'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : name},
                {Value : description},
                {Value : format},
                {Value : fileSize},
                {Value : status}
            ]
        },
        
        FieldGroup#Metadata : {
            Data : [
                {Value : dcTitle, Label : 'Title'},
                {Value : dcCreator, Label : 'Creator'},
                {Value : dcSubject, Label : 'Subject'},
                {Value : dcDescription, Label : 'Description'},
                {Value : dcPublisher, Label : 'Publisher'},
                {Value : dcContributor, Label : 'Contributor'},
                {Value : dcDate, Label : 'Date'},
                {Value : dcType, Label : 'Type'},
                {Value : dcFormat, Label : 'Format'},
                {Value : dcIdentifier, Label : 'Identifier'},
                {Value : dcSource, Label : 'Source'},
                {Value : dcLanguage, Label : 'Language'},
                {Value : dcRelation, Label : 'Relation'},
                {Value : dcCoverage, Label : 'Coverage'},
                {Value : dcRights, Label : 'Rights'}
            ]
        },
        
        FieldGroup#Quality : {
            Data : [
                {Value : qualityScore, Visualization : #Rating},
                {Value : integrityHash},
                {Value : validationStatus},
                {Value : validationErrors}
            ]
        },
        
        FieldGroup#Processing : {
            Data : [
                {Value : ingestionTimestamp},
                {Value : transformationRules},
                {Value : ordRegistryId},
                {Value : workflowId}
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

// Value Help annotations
annotate service.DataProducts {
    format @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Formats',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : format,
            ValueListProperty : 'code'
        }]
    };
    
    status @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Statuses',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : status,
            ValueListProperty : 'code'
        }]
    };
}