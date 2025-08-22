using A2AService as service from '../../../../srv/a2a-service';

// Agent 12 - Catalog Manager Agent UI Annotations
annotate service.CatalogEntries with @(
    UI : {
        // List Report Configuration
        SelectionFields : [
            catalogType,
            resourceType,
            category,
            status,
            createdAt
        ],
        
        LineItem : [
            {Value: resourceName, Label: 'Resource Name'},
            {Value: catalogType, Label: 'Catalog Type'},
            {Value: resourceType, Label: 'Resource Type'},
            {Value: category, Label: 'Category'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: visibility, Label: 'Visibility'},
            {Value: usageCount, Label: 'Usage Count'},
            {Value: rating, Label: 'Rating', Visualization: #Rating},
            {Value: lastAccessed, Label: 'Last Accessed'},
            {Value: createdAt, Label: 'Created'}
        ],
        
        // Object Page Configuration
        HeaderInfo : {
            TypeName : 'Catalog Entry',
            TypeNamePlural : 'Catalog Entries',
            Title : {Value : resourceName},
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
                Target : '@UI.FieldGroup#ResourceDetails',
                Label : 'Resource Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Metadata',
                Label : 'Metadata Information'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Discovery',
                Label : 'Discovery & Search'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Registration',
                Label : 'Registration Details'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : '@UI.FieldGroup#Usage',
                Label : 'Usage Analytics'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'MetadataProperties/@UI.LineItem',
                Label : 'Metadata Properties'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'ServiceEndpoints/@UI.LineItem',
                Label : 'Service Endpoints'
            },
            {
                $Type : 'UI.ReferenceFacet',
                Target : 'Dependencies/@UI.LineItem',
                Label : 'Dependencies'
            }
        ],
        
        // Field Groups
        FieldGroup#General : {
            Data : [
                {Value : resourceName},
                {Value : description},
                {Value : catalogType},
                {Value : resourceType},
                {Value : category},
                {Value : status},
                {Value : visibility}
            ]
        },
        
        FieldGroup#ResourceDetails : {
            Data : [
                {Value : resourceUrl},
                {Value : version},
                {Value : apiVersion},
                {Value : protocol},
                {Value : authenticationMethod},
                {Value : contentType},
                {Value : documentation},
                {Value : healthCheckUrl},
                {Value : swaggerUrl}
            ]
        },
        
        FieldGroup#Metadata : {
            Data : [
                {Value : metadataSchema},
                {Value : keywords},
                {Value : tags},
                {Value : owner},
                {Value : maintainer},
                {Value : license},
                {Value : compliance},
                {Value : dataClassification},
                {Value : retentionPeriod}
            ]
        },
        
        FieldGroup#Discovery : {
            Data : [
                {Value : searchable},
                {Value : indexedAt},
                {Value : searchTags},
                {Value : searchWeight},
                {Value : discoveryScore, Visualization : #Rating},
                {Value : searchRanking},
                {Value : autoDiscovery},
                {Value : discoveryFrequency}
            ]
        },
        
        FieldGroup#Registration : {
            Data : [
                {Value : registeredBy},
                {Value : registrationDate},
                {Value : approvalStatus},
                {Value : approvedBy},
                {Value : approvalDate},
                {Value : registrationSource},
                {Value : validationStatus},
                {Value : lastValidated}
            ]
        },
        
        FieldGroup#Usage : {
            Data : [
                {Value : usageCount},
                {Value : rating, Visualization : #Rating},
                {Value : reviewCount},
                {Value : lastAccessed},
                {Value : popularityScore, Visualization : #Rating},
                {Value : downloadCount},
                {Value : activeUsers},
                {Value : uptime, Visualization : #Rating}
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

// Metadata Properties as associated entity
annotate service.MetadataProperties with @(
    UI : {
        LineItem : [
            {Value: propertyName, Label: 'Property Name'},
            {Value: propertyType, Label: 'Type'},
            {Value: propertyValue, Label: 'Value'},
            {Value: dataType, Label: 'Data Type'},
            {Value: isRequired, Label: 'Required'},
            {Value: isSearchable, Label: 'Searchable'},
            {Value: displayOrder, Label: 'Display Order'},
            {Value: validation, Label: 'Validation'},
            {Value: description, Label: 'Description'}
        ]
    }
);

// Service Endpoints as associated entity
annotate service.ServiceEndpoints with @(
    UI : {
        LineItem : [
            {Value: endpointName, Label: 'Endpoint Name'},
            {Value: endpointUrl, Label: 'URL'},
            {Value: endpointType, Label: 'Type'},
            {Value: httpMethod, Label: 'HTTP Method'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: responseTime, Label: 'Response Time'},
            {Value: availability, Label: 'Availability', Visualization: #Rating},
            {Value: lastChecked, Label: 'Last Checked'},
            {Value: description, Label: 'Description'}
        ]
    }
);

// Dependencies as associated entity
annotate service.Dependencies with @(
    UI : {
        LineItem : [
            {Value: dependencyName, Label: 'Dependency Name'},
            {Value: dependencyType, Label: 'Type'},
            {Value: version, Label: 'Version'},
            {Value: isRequired, Label: 'Required'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: healthStatus, Label: 'Health Status'},
            {Value: lastValidated, Label: 'Last Validated'},
            {Value: documentation, Label: 'Documentation'},
            {Value: description, Label: 'Description'}
        ]
    }
);

// Service Categories
annotate service.ServiceCategories with @(
    UI : {
        SelectionFields : [
            categoryName,
            parentCategory,
            isActive
        ],
        
        LineItem : [
            {Value: categoryName, Label: 'Category Name'},
            {Value: parentCategory, Label: 'Parent Category'},
            {Value: description, Label: 'Description'},
            {Value: serviceCount, Label: 'Service Count'},
            {Value: subcategoryCount, Label: 'Subcategories'},
            {Value: displayOrder, Label: 'Display Order'},
            {Value: icon, Label: 'Icon'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Registry Configurations
annotate service.RegistryConfigurations with @(
    UI : {
        SelectionFields : [
            registryName,
            registryType,
            isActive
        ],
        
        LineItem : [
            {Value: registryName, Label: 'Registry Name'},
            {Value: registryType, Label: 'Type'},
            {Value: registryUrl, Label: 'Registry URL'},
            {Value: authenticationMethod, Label: 'Authentication'},
            {Value: syncFrequency, Label: 'Sync Frequency'},
            {Value: lastSync, Label: 'Last Sync'},
            {Value: entryCount, Label: 'Entry Count'},
            {Value: isActive, Label: 'Active'}
        ]
    }
);

// Search Indexes
annotate service.SearchIndexes with @(
    UI : {
        SelectionFields : [
            indexName,
            indexType,
            status
        ],
        
        LineItem : [
            {Value: indexName, Label: 'Index Name'},
            {Value: indexType, Label: 'Type'},
            {Value: documentCount, Label: 'Document Count'},
            {Value: lastUpdated, Label: 'Last Updated'},
            {Value: indexSize, Label: 'Index Size'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: searchPerformance, Label: 'Performance', Visualization: #Rating},
            {Value: refreshFrequency, Label: 'Refresh Frequency'}
        ]
    }
);

// Resource Discovery
annotate service.ResourceDiscovery with @(
    UI : {
        SelectionFields : [
            discoveryMethod,
            sourceType,
            status
        ],
        
        LineItem : [
            {Value: resourceName, Label: 'Resource Name'},
            {Value: discoveryMethod, Label: 'Discovery Method'},
            {Value: sourceType, Label: 'Source Type'},
            {Value: discoveredAt, Label: 'Discovered At'},
            {Value: confidence, Label: 'Confidence', Visualization: #Rating},
            {Value: validationStatus, Label: 'Validation Status'},
            {Value: status, Label: 'Status', Criticality: statusCriticality},
            {Value: autoRegistered, Label: 'Auto Registered'}
        ]
    }
);

// Value Help annotations
annotate service.CatalogEntries {
    catalogType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'CatalogTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : catalogType,
            ValueListProperty : 'typeName'
        }]
    };
    
    resourceType @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ResourceTypes',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : resourceType,
            ValueListProperty : 'typeName'
        }]
    };
    
    category @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'ServiceCategories',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : category,
            ValueListProperty : 'categoryName'
        }]
    };
    
    owner @Common.ValueList : {
        $Type : 'Common.ValueListType',
        CollectionPath : 'Users',
        Parameters : [{
            $Type : 'Common.ValueListParameterOut',
            LocalDataProperty : owner,
            ValueListProperty : 'userName'
        }]
    };
}