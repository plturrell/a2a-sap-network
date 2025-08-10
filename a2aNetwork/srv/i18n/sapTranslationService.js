/**
 * @fileoverview SAP Translation Management Service
 * @description Enterprise translation service providing APIs for managing translations,
 * localization coverage tracking, and multi-language content management.
 * @module sapTranslationService
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.i18n
 */

const cds = require('@sap/cds');
const { normalizeLocale } = require('./i18n-config');

/**
 * Translation Management Service
 * Provides APIs for managing translations and localization
 */
class TranslationService extends cds.ApplicationService {
    async init() {
        const { Translations, Locales, TranslationAuditLog, MissingTranslations, TranslationCoverage } = this.entities;
        
        // Get all translations for a namespace
        this.on('READ', 'Translations', async (req) => {
            const { namespace, locale } = req.query.SELECT.where || {};
            
            if (namespace && locale) {
                return SELECT.from(Translations)
                    .where({ namespace, locale: normalizeLocale(locale) });
            }
            
            return req.query;
        });
        
        // Create or update translation
        this.on('CREATE', 'Translations', async (req) => {
            const { namespace, key, locale, value, context } = req.data;
            const normalizedLocale = normalizeLocale(locale);
            
            // Check if translation exists
            const existing = await SELECT.one.from(Translations)
                .where({ namespace, key, locale: normalizedLocale });
            
            if (existing) {
                // Update existing translation
                await this._updateTranslation(existing, value, context, req);
                return { ...existing, value, context };
            }
            
            // Create new translation
            req.data.locale = normalizedLocale;
            return req.data;
        });
        
        // Update translation
        this.on('UPDATE', 'Translations', async (req) => {
            const { namespace, key, locale } = req.params[0];
            const { value, context, isReviewed, reviewedBy } = req.data;
            
            // Get existing translation
            const existing = await SELECT.one.from(Translations)
                .where({ namespace, key, locale: normalizeLocale(locale) });
            
            if (!existing) {
                req.error(404, 'TRANSLATION_NOT_FOUND', 'Translation not found');
            }
            
            // Log changes
            await this._updateTranslation(existing, value, context, req);
            
            // Update review status
            if (isReviewed) {
                req.data.reviewedAt = new Date();
                req.data.reviewedBy = reviewedBy || req.user.id;
            }
            
            return req.data;
        });
        
        // Delete translation
        this.on('DELETE', 'Translations', async (req) => {
            const { namespace, key, locale } = req.params[0];
            
            const existing = await SELECT.one.from(Translations)
                .where({ namespace, key, locale });
            
            if (existing) {
                await INSERT.into(TranslationAuditLog).entries({
                    namespace,
                    key,
                    locale,
                    oldValue: existing.value,
                    newValue: null,
                    action: 'DELETE',
                    reason: req.data.reason || 'Deleted by user'
                });
            }
        });
        
        // Get supported locales
        this.on('getLocales', async () => {
            return SELECT.from(Locales)
                .where({ isActive: true })
                .orderBy('sortOrder', 'code');
        });
        
        // Get default locale
        this.on('getDefaultLocale', async () => {
            const defaultLocale = await SELECT.one.from(Locales)
                .where({ isDefault: true });
            
            return defaultLocale || { code: 'en' };
        });
        
        // Set default locale
        this.on('setDefaultLocale', async (req) => {
            const { locale } = req.data;
            const normalizedLocale = normalizeLocale(locale);
            
            // Reset all defaults
            await UPDATE(Locales).set({ isDefault: false });
            
            // Set new default
            await UPDATE(Locales)
                .set({ isDefault: true })
                .where({ code: normalizedLocale });
            
            return { success: true, locale: normalizedLocale };
        });
        
        // Get missing translations
        this.on('getMissingTranslations', async (req) => {
            const { locale } = req.data;
            
            if (locale) {
                return SELECT.from(MissingTranslations)
                    .where({ locale: normalizeLocale(locale) });
            }
            
            return SELECT.from(MissingTranslations);
        });
        
        // Get translation coverage
        this.on('getTranslationCoverage', async () => {
            return SELECT.from(TranslationCoverage)
                .orderBy('coveragePercentage desc');
        });
        
        // Export translations
        this.on('exportTranslations', async (req) => {
            const { locale, namespace, format = 'json' } = req.data;
            
            const translations = await SELECT.from(Translations)
                .where(
                    locale ? { locale: normalizeLocale(locale) } : {},
                    namespace ? { namespace } : {}
                );
            
            switch (format) {
                case 'json':
                    return this._exportAsJSON(translations);
                case 'csv':
                    return this._exportAsCSV(translations);
                case 'xliff':
                    return this._exportAsXLIFF(translations);
                default:
                    req.error(400, 'INVALID_FORMAT', 'Invalid export format');
            }
        });
        
        // Import translations
        this.on('importTranslations', async (req) => {
            const { data, format = 'json', locale, overwrite = false } = req.data;
            
            let translations;
            switch (format) {
                case 'json':
                    translations = this._importFromJSON(data, locale);
                    break;
                case 'csv':
                    translations = this._importFromCSV(data);
                    break;
                case 'xliff':
                    translations = this._importFromXLIFF(data);
                    break;
                default:
                    req.error(400, 'INVALID_FORMAT', 'Invalid import format');
            }
            
            return this._processImport(translations, overwrite, req);
        });
        
        // Validate translations
        this.on('validateTranslations', async (req) => {
            const { locale } = req.data;
            const normalizedLocale = locale ? normalizeLocale(locale) : null;
            
            const issues = [];
            
            // Check for missing translations
            const missing = await SELECT.from(MissingTranslations)
                .where(normalizedLocale ? { locale: normalizedLocale } : {});
            
            missing.forEach(m => {
                issues.push({
                    type: 'missing',
                    severity: 'warning',
                    namespace: m.namespace,
                    key: m.key,
                    locale: m.locale,
                    message: `Missing translation for key: ${m.key}`
                });
            });
            
            // Check for unreviewed translations
            const unreviewed = await SELECT.from(Translations)
                .where(
                    { isReviewed: false },
                    normalizedLocale ? { locale: normalizedLocale } : {}
                );
            
            unreviewed.forEach(t => {
                issues.push({
                    type: 'unreviewed',
                    severity: 'info',
                    namespace: t.namespace,
                    key: t.key,
                    locale: t.locale,
                    message: `Translation not reviewed: ${t.key}`
                });
            });
            
            // Check for placeholder mismatches
            const allTranslations = await SELECT.from(Translations)
                .where(normalizedLocale ? { locale: normalizedLocale } : {});
            
            for (const translation of allTranslations) {
                const placeholderIssues = await this._validatePlaceholders(translation);
                issues.push(...placeholderIssues);
            }
            
            return {
                valid: issues.filter(i => i.severity === 'error').length === 0,
                issues,
                summary: {
                    total: issues.length,
                    errors: issues.filter(i => i.severity === 'error').length,
                    warnings: issues.filter(i => i.severity === 'warning').length,
                    info: issues.filter(i => i.severity === 'info').length
                }
            };
        });
        
        await super.init();
    }
    
    /**
     * Update translation and create audit log
     */
    async _updateTranslation(existing, newValue, context, _req) {
        if (existing.value !== newValue) {
            await INSERT.into(this.entities.TranslationAuditLog).entries({
                namespace: existing.namespace,
                key: existing.key,
                locale: existing.locale,
                oldValue: existing.value,
                newValue: newValue,
                action: 'UPDATE',
                reason: context || 'Updated by user'
            });
        }
    }
    
    /**
     * Export translations as JSON
     */
    _exportAsJSON(translations) {
        const result = {};
        
        translations.forEach(t => {
            if (!result[t.locale]) {
                result[t.locale] = {};
            }
            if (!result[t.locale][t.namespace]) {
                result[t.locale][t.namespace] = {};
            }
            
            // Convert dot notation to nested object
            const keys = t.key.split('.');
            let current = result[t.locale][t.namespace];
            
            for (let i = 0; i < keys.length - 1; i++) {
                if (!current[keys[i]]) {
                    current[keys[i]] = {};
                }
                current = current[keys[i]];
            }
            
            current[keys[keys.length - 1]] = t.value;
        });
        
        return result;
    }
    
    /**
     * Export translations as CSV
     */
    _exportAsCSV(translations) {
        const headers = ['namespace', 'key', 'locale', 'value', 'context', 'isReviewed'];
        const rows = [headers.join(',')];
        
        translations.forEach(t => {
            const row = [
                t.namespace,
                t.key,
                t.locale,
                `"${t.value.replace(/"/g, '""')}"`,
                t.context || '',
                t.isReviewed
            ];
            rows.push(row.join(','));
        });
        
        return rows.join('\n');
    }
    
    /**
     * Export translations as XLIFF
     */
    _exportAsXLIFF(translations) {
        // Group by locale
        const byLocale = {};
        translations.forEach(t => {
            if (!byLocale[t.locale]) {
                byLocale[t.locale] = [];
            }
            byLocale[t.locale].push(t);
        });
        
        let xliff = '<?xml version="1.0" encoding="UTF-8"?>\n';
        xliff += '<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">\n';
        
        Object.entries(byLocale).forEach(([locale, trans]) => {
            xliff += `  <file source-language="en" target-language="${locale}" datatype="plaintext">\n`;
            xliff += '    <body>\n';
            
            trans.forEach(t => {
                xliff += `      <trans-unit id="${t.namespace}.${t.key}">\n`;
                xliff += `        <source>${this._escapeXML(t.key)}</source>\n`;
                xliff += `        <target>${this._escapeXML(t.value)}</target>\n`;
                if (t.context) {
                    xliff += `        <note>${this._escapeXML(t.context)}</note>\n`;
                }
                xliff += '      </trans-unit>\n';
            });
            
            xliff += '    </body>\n';
            xliff += '  </file>\n';
        });
        
        xliff += '</xliff>';
        return xliff;
    }
    
    /**
     * Import translations from JSON
     */
    _importFromJSON(data, defaultLocale) {
        const translations = [];
        const parsed = typeof data === 'string' ? JSON.parse(data) : data;
        
        // Handle flat structure { "key": "value" }
        if (defaultLocale && !parsed[Object.keys(parsed)[0]]?.namespace) {
            this._flattenObject(parsed, '', (key, value) => {
                translations.push({
                    namespace: 'common',
                    key,
                    locale: normalizeLocale(defaultLocale),
                    value
                });
            });
            return translations;
        }
        
        // Handle nested structure { "locale": { "namespace": { "key": "value" } } }
        Object.entries(parsed).forEach(([locale, namespaces]) => {
            Object.entries(namespaces).forEach(([namespace, keys]) => {
                this._flattenObject(keys, '', (key, value) => {
                    translations.push({
                        namespace,
                        key,
                        locale: normalizeLocale(locale),
                        value
                    });
                });
            });
        });
        
        return translations;
    }
    
    /**
     * Import translations from CSV
     */
    _importFromCSV(data) {
        const lines = data.split('\n');
        const headers = lines[0].split(',');
        const translations = [];
        
        for (let i = 1; i < lines.length; i++) {
            if (!lines[i].trim()) continue;
            
            const values = this._parseCSVLine(lines[i]);
            const translation = {};
            
            headers.forEach((header, index) => {
                translation[header.trim()] = values[index]?.trim() || '';
            });
            
            if (translation.namespace && translation.key && translation.locale && translation.value) {
                translation.locale = normalizeLocale(translation.locale);
                translations.push(translation);
            }
        }
        
        return translations;
    }
    
    /**
     * Process imported translations
     */
    async _processImport(translations, overwrite, _req) {
        let created = 0;
        let updated = 0;
        let skipped = 0;
        const errors = [];
        
        for (const translation of translations) {
            try {
                const existing = await SELECT.one.from(this.entities.Translations)
                    .where({
                        namespace: translation.namespace,
                        key: translation.key,
                        locale: translation.locale
                    });
                
                if (existing) {
                    if (overwrite) {
                        await UPDATE(this.entities.Translations)
                            .set({ value: translation.value, context: translation.context })
                            .where({
                                namespace: translation.namespace,
                                key: translation.key,
                                locale: translation.locale
                            });
                        updated++;
                    } else {
                        skipped++;
                    }
                } else {
                    await INSERT.into(this.entities.Translations).entries(translation);
                    created++;
                }
            } catch (error) {
                errors.push({
                    translation,
                    error: error.message
                });
            }
        }
        
        return {
            success: errors.length === 0,
            created,
            updated,
            skipped,
            errors,
            total: translations.length
        };
    }
    
    /**
     * Validate placeholders in translation
     */
    async _validatePlaceholders(translation) {
        const issues = [];
        
        // Get English version for comparison
        if (translation.locale !== 'en') {
            const englishVersion = await SELECT.one.from(this.entities.Translations)
                .where({
                    namespace: translation.namespace,
                    key: translation.key,
                    locale: 'en'
                });
            
            if (englishVersion) {
                const sourcePlaceholders = this._extractPlaceholders(englishVersion.value);
                const targetPlaceholders = this._extractPlaceholders(translation.value);
                
                // Check for missing placeholders
                sourcePlaceholders.forEach(ph => {
                    if (!targetPlaceholders.includes(ph)) {
                        issues.push({
                            type: 'placeholder',
                            severity: 'error',
                            namespace: translation.namespace,
                            key: translation.key,
                            locale: translation.locale,
                            message: `Missing placeholder: ${ph}`
                        });
                    }
                });
                
                // Check for extra placeholders
                targetPlaceholders.forEach(ph => {
                    if (!sourcePlaceholders.includes(ph)) {
                        issues.push({
                            type: 'placeholder',
                            severity: 'warning',
                            namespace: translation.namespace,
                            key: translation.key,
                            locale: translation.locale,
                            message: `Extra placeholder: ${ph}`
                        });
                    }
                });
            }
        }
        
        return issues;
    }
    
    /**
     * Extract placeholders from text
     */
    _extractPlaceholders(text) {
        const regex = /{{(\w+)}}/g;
        const placeholders = [];
        let match;
        
        while ((match = regex.exec(text)) !== null) {
            placeholders.push(match[0]);
        }
        
        return placeholders;
    }
    
    /**
     * Flatten nested object
     */
    _flattenObject(obj, prefix, callback) {
        Object.entries(obj).forEach(([key, value]) => {
            const fullKey = prefix ? `${prefix}.${key}` : key;
            
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                this._flattenObject(value, fullKey, callback);
            } else {
                callback(fullKey, value);
            }
        });
    }
    
    /**
     * Parse CSV line handling quoted values
     */
    _parseCSVLine(line) {
        const values = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            const nextChar = line[i + 1];
            
            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    current += '"';
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                values.push(current);
                current = '';
            } else {
                current += char;
            }
        }
        
        values.push(current);
        return values;
    }
    
    /**
     * Escape XML special characters
     */
    _escapeXML(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;');
    }
}

module.exports = TranslationService;