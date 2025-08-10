namespace a2a.network.i18n;

using { managed, cuid } from '@sap/cds/common';

/**
 * Translations entity for storing localized text
 */
entity Translations : cuid, managed {
    namespace   : String(100) not null;
    key        : String(255) not null;
    locale     : String(5) not null;
    value      : String(5000);
    
    // Metadata
    isPlural   : Boolean default false;
    context    : String(500);  // Additional context for translators
    
    // Audit fields
    translatedBy : String(100);
    reviewedBy   : String(100);
    reviewedAt   : DateTime;
    status     : String(20) enum { 
        draft; 
        translated; 
        reviewed; 
        approved; 
        deprecated 
    } default 'draft';
    
    // Quality metrics
    confidence : Decimal(3,2) default 0.0; // Translation confidence 0-100
    
    // Indices
    unique (namespace, key, locale);
}

/**
 * Supported locales configuration
 */
entity Locales : cuid, managed {
    locale     : String(5) not null; // e.g., 'en', 'de', 'fr'
    name       : String(100) not null; // e.g., 'English', 'Deutsch'
    nativeName : String(100) not null; // e.g., 'English', 'Deutsch'
    isDefault  : Boolean default false;
    isActive   : Boolean default true;
    rtl        : Boolean default false; // Right-to-left language
    
    // Regional settings
    dateFormat : String(20) default 'MM/dd/yyyy';
    numberFormat : String(20) default '#,##0.00';
    currencyCode : String(3);
    
    // Coverage metrics
    totalKeys  : Integer default 0;
    translatedKeys : Integer default 0;
    coverage   : Decimal(5,2) default 0.0; // Percentage
    
    unique (locale);
}

/**
 * User locale preferences
 */
entity UserLocalePreferences : cuid, managed {
    userId     : String(100) not null;
    locale     : String(5) not null;
    fallbackLocale : String(5) default 'en';
    
    // UI preferences
    dateFormat : String(20);
    timeFormat : String(20);
    numberFormat : String(20);
    currencyFormat : String(20);
    timezone   : String(50);
    
    unique (userId);
}

/**
 * View for missing translations
 */
view MissingTranslations as select from Translations {
    namespace,
    key,
    locale,
    status
} where status = 'draft' or value is null;

/**
 * View for translation coverage by locale
 */
view TranslationCoverage as select from Translations {
    locale,
    namespace,
    count(*) as totalKeys : Integer,
    sum(case when value is not null and value != '' then 1 else 0 end) as translatedKeys : Integer,
    cast(
        sum(case when value is not null and value != '' then 1 else 0 end) * 100.0 / count(*)
        as Decimal(5,2)
    ) as coverage : Decimal(5,2)
} group by locale, namespace;

/**
 * Audit log for translation changes
 */
entity TranslationAuditLog : cuid, managed {
    namespace  : String(100) not null;
    key        : String(255) not null;
    locale     : String(5) not null;
    
    oldValue   : String(5000);
    newValue   : String(5000);
    
    action     : String(20) enum { 
        created; 
        updated; 
        deleted; 
        imported; 
        exported 
    };
    
    changedBy  : String(100);
    reason     : String(500);
    timestamp  : DateTime not null;
    version    : Integer default 1;
    
    // Source information
    source     : String(50); // e.g., 'manual', 'import', 'api'
    sourceId   : String(100); // Import batch ID, API client, etc.
}

/**
 * Translation validation rules
 */
entity TranslationValidationRules : cuid, managed {
    namespace  : String(100);
    pattern    : String(500);
    ruleType   : String(20) enum { 
        required; 
        format; 
        length; 
        placeholder 
    };
    severity   : String(10) enum { 
        error; 
        warning; 
        info 
    } default 'warning';
    message    : String(500);
    isActive   : Boolean default true;
}

/**
 * Translation import/export batches
 */
entity TranslationBatches : cuid, managed {
    batchName  : String(100);
    operation  : String(10) enum { import; export };
    format     : String(10) enum { json; csv; xliff };
    locale     : String(5);
    namespace  : String(100);
    
    status     : String(20) enum { 
        pending; 
        processing; 
        completed; 
        failed 
    } default 'pending';
    
    // Statistics
    totalRecords : Integer default 0;
    successRecords : Integer default 0;
    errorRecords : Integer default 0;
    
    // File information
    fileName   : String(255);
    fileSize   : Integer;
    checksum   : String(64);
    
    // Process information
    processedBy : String(100);
    processedAt : DateTime;
    errorLog   : LargeString;
    
    // Results
    resultData : LargeString; // JSON format for export results
}

/**
 * Translation memory for reuse
 */
entity TranslationMemory : cuid, managed {
    sourceText : String(5000) not null;
    targetText : String(5000) not null;
    sourceLocale : String(5) not null;
    targetLocale : String(5) not null;
    
    // Quality metrics
    confidence : Decimal(3,2) default 0.0;
    matchType  : String(20) enum { 
        exact; 
        fuzzy; 
        machine 
    };
    
    // Usage statistics
    usageCount : Integer default 1;
    lastUsed   : DateTime;
    
    // Source information
    domain     : String(100);
    context    : String(500);
    
    unique (sourceText, sourceLocale, targetLocale);
}