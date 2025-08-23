namespace a2a.platform;

using { Currency, Country } from '@sap/cds/common';
using { a2a.platform.Identifiable, a2a.platform.Manageable, a2a.platform.StatusTracking, a2a.platform.BusinessMetadata, a2a.platform.QualityTracking, a2a.platform.ConfigurationSettings } from '../aspects/common';

// Data Product domain entities
entity DataProducts : Identifiable, Manageable, StatusTracking, BusinessMetadata, QualityTracking {
    sourceSystem : String(100) not null;
    dataFormat : String(50);
    schemaVersion : String(20);
    
    // Business metadata
    assetClass : String(100);
    frequency : String(50);
    startDate : Date;
    endDate : Date;
    currency : Currency;
    country : Country;
    
    // Relationships  
    validationRules : Composition of many ValidationRules on validationRules.dataProduct = $self;
    processingSteps : Composition of many ProcessingSteps on processingSteps.dataProduct = $self;
    qualityMetrics : Composition of many QualityMetrics on qualityMetrics.dataProduct = $self;
    dataFiles : Composition of many DataFiles on dataFiles.dataProduct = $self;
}

entity ValidationRules : Identifiable, ConfigurationSettings {
    dataProduct : Association to DataProducts;
    ruleType : String(100) not null; // COMPLETENESS, ACCURACY, etc.
    threshold : Decimal(5,4);
    severity : String(20) default 'ERROR'; // ERROR, WARNING, INFO
    validationQuery : LargeString; // SQL or expression
}

entity DataFiles : Identifiable, Manageable, StatusTracking {
    dataProduct : Association to DataProducts;
    fileName : String(500) not null;
    fileSize : Integer; // bytes
    fileHash : String(64); // SHA-256
    uploadPath : String(1000);
    contentType : String(100);
    
    // Processing metadata
    rowCount : Integer;
    columnCount : Integer;
    schema : LargeString; // JSON schema definition
}

entity QualityMetrics : Identifiable {
    dataProduct : Association to DataProducts;
    metricType : String(100) not null;
    metricValue : Decimal(15,4) not null;
    metricUnit : String(50);
    calculatedAt : Timestamp not null;
    calculationMethod : String(200);
    
    // Quality dimensions
    dimension : String(50); // completeness, accuracy, consistency, etc.
    threshold : Decimal(5,4);
    passed : Boolean;
    details : LargeString; // JSON details
}