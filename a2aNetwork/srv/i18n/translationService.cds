using { a2a.network.i18n as i18n } from '../../db/i18nSchema';

/**
 * Translation Management Service
 * Provides APIs for managing translations and localization
 */
service TranslationService @(path: '/api/v1/translations', requires: 'authenticated-user') {
    
    // Core entities
    entity Translations as projection on i18n.Translations;
    entity Locales as projection on i18n.Locales;
    entity UserLocalePreferences as projection on i18n.UserLocalePreferences;
    
    // Read-only views
    @readonly entity MissingTranslations as projection on i18n.MissingTranslations;
    @readonly entity TranslationCoverage as projection on i18n.TranslationCoverage;
    @readonly entity TranslationAuditLog as projection on i18n.TranslationAuditLog;
    
    // Actions for locale management
    action getLocales() returns array of Locales;
    action getDefaultLocale() returns Locales;
    action setDefaultLocale(locale: String) returns { success: Boolean; locale: String };
    
    // Actions for translation management
    action getMissingTranslations(locale: String) returns array of MissingTranslations;
    action getTranslationCoverage() returns array of TranslationCoverage;
    
    // Import/Export actions
    action exportTranslations(
        locale: String,
        namespace: String,
        format: String enum { json; csv; xliff; }
    ) returns LargeString;
    
    action importTranslations(
        data: LargeString,
        format: String enum { json; csv; xliff; },
        locale: String,
        overwrite: Boolean
    ) returns {
        success: Boolean;
        created: Integer;
        updated: Integer;
        skipped: Integer;
        errors: array of { translation: String; error: String };
        total: Integer;
    };
    
    // Validation action
    action validateTranslations(locale: String) returns {
        valid: Boolean;
        total: Integer;
        errors: Integer;
        warnings: Integer;
        info: Integer;
    };
    
    // User preference actions
    action getUserPreferences() returns UserLocalePreferences;
    action setUserPreferences(preferences: UserLocalePreferences) returns UserLocalePreferences;
}