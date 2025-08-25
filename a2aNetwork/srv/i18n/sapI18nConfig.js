/**
 * @fileoverview SAP Internationalization Configuration Module
 * @description Configures i18n settings for A2A Network with SAP standard language codes,
 * regional variants, and enterprise-grade localization features.
 * @module sapI18nConfig
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.i18n
 */

const i18n = require('i18n');
const path = require('path');
const cds = require('@sap/cds');

/**
 * Initialize i18n configuration for A2A Network
 * Supports SAP's standard language codes and regional variants
 */
function initializeI18n(app) {
    i18n.configure({
        // Supported locales following SAP language codes
        locales: [
            'en',       // English (default)
            'de',       // German
            'fr',       // French
            'es',       // Spanish
            'it',       // Italian
            'pt',       // Portuguese
            'zh',       // Chinese (Simplified)
            'zh-TW',    // Chinese (Traditional)
            'ja',       // Japanese
            'ko',       // Korean
            'ru',       // Russian
            'ar',       // Arabic
            'he',       // Hebrew
            'hi',       // Hindi
            'nl',       // Dutch
            'pl',       // Polish
            'tr',       // Turkish
            'cs',       // Czech
            'sv',       // Swedish
            'da',       // Danish
            'no',       // Norwegian
            'fi'        // Finnish
        ],

        // Default locale
        defaultLocale: 'en',

        // Directory for translation files
        directory: path.join(__dirname, 'locales'),

        // Object notation for nested translations
        objectNotation: true,

        // Update files on missing translations (dev only)
        updateFiles: process.env.NODE_ENV === 'development',

        // Sync files on startup
        syncFiles: true,

        // Cookie name for locale persistence
        cookie: 'a2a-locale',

        // Query parameter for locale switching
        queryParameter: 'locale',

        // Register global helper
        register: global,

        // Preserve locale on redirect
        preserveLegacyCase: true,

        // Auto reload in development
        autoReload: process.env.NODE_ENV === 'development',

        // Indent JSON files
        indent: '  ',

        // Extension of json files
        extension: '.json',

        // Prefix for missing translations
        prefix: '[MISSING_TRANSLATION]',

        // Enable debug logging in development
        logDebugFn: function (msg) {
            if (process.env.NODE_ENV === 'development') {
                cds.log('service').info('i18n:', msg);
            }
        },

        // Log warnings for missing translations
        logWarnFn: function (msg) {
            cds.log('service').warn('i18n warning:', msg);
        },

        // Log errors
        logErrorFn: function (msg) {
            cds.log('service').error('i18n error:', msg);
        },

        // Missing key handler
        missingKeyFn: function(locale, value) {
            cds.log('service').error(`Missing translation: ${value} for locale: ${locale}`);
            return value;
        }
    });

    // Initialize i18n middleware
    app.use(i18n.init);

    // Custom middleware to handle SAP-specific locale formats
    app.use((req, res, next) => {
        // Check for SAP user locale from JWT token
        if (req.user && req.user.locale) {
            req.setLocale(normalizeLocale(req.user.locale));
        }

        // Check Accept-Language header
        else if (req.headers['accept-language']) {
            const locale = parseAcceptLanguage(req.headers['accept-language']);
            if (locale) {
                req.setLocale(locale);
            }
        }

        // Check for locale in query params or cookies
        else if (req.query.locale) {
            req.setLocale(normalizeLocale(req.query.locale));
        }

        // Set locale in response for client
        res.locals.locale = req.getLocale();
        res.locals.locales = i18n.getLocales();

        next();
    });
}

/**
 * Normalize SAP locale codes to standard format
 * @param {string} locale - SAP locale code (e.g., 'DE', 'EN_US')
 * @returns {string} Normalized locale code
 */
function normalizeLocale(locale) {
    if (!locale) return 'en';

    // Convert to lowercase and replace underscores
    locale = locale.toLowerCase().replace('_', '-');

    // Map SAP language codes to standard codes
    const sapToStandard = {
        '1': 'zh',      // Chinese
        '2': 'en',      // English
        '3': 'fr',      // French
        '4': 'de',      // German
        '5': 'it',      // Italian
        '6': 'ja',      // Japanese
        '7': 'pt',      // Portuguese
        '8': 'ru',      // Russian
        '9': 'es',      // Spanish
        'e': 'en',      // English
        'd': 'de',      // German
        'f': 'fr'       // French
    };

    if (sapToStandard[locale]) {
        return sapToStandard[locale];
    }

    // Handle regional variants
    if (locale.includes('-')) {
        const [lang, region] = locale.split('-');

        // Special cases for Chinese
        if (lang === 'zh' && ['hk', 'tw', 'mo'].includes(region)) {
            return 'zh-TW';
        }

        // Return base language if exact match not found
        return i18n.getLocales().includes(locale) ? locale : lang;
    }

    return locale;
}

/**
 * Parse Accept-Language header
 * @param {string} acceptLanguage - Accept-Language header value
 * @returns {string|null} Best matching locale
 */
function parseAcceptLanguage(acceptLanguage) {
    const languages = acceptLanguage
        .split(',')
        .map(lang => {
            const [locale, q = '1'] = lang.trim().split(';q=');
            return {
                locale: normalizeLocale(locale),
                quality: parseFloat(q)
            };
        })
        .sort((a, b) => b.quality - a.quality);

    // Find first supported locale
    for (const lang of languages) {
        if (i18n.getLocales().includes(lang.locale)) {
            return lang.locale;
        }

        // Try base language
        const baseLocale = lang.locale.split('-')[0];
        if (i18n.getLocales().includes(baseLocale)) {
            return baseLocale;
        }
    }

    return null;
}

/**
 * Get translations for a specific namespace
 * @param {string} namespace - Translation namespace
 * @param {string} locale - Target locale
 * @returns {object} Translations object
 */
function getTranslations(namespace, locale = 'en') {
    const catalog = i18n.getCatalog(locale);
    return catalog[namespace] || {};
}

/**
 * Format currency with locale
 * @param {number} amount - Amount to format
 * @param {string} currency - Currency code
 * @param {string} locale - Target locale
 * @returns {string} Formatted currency
 */
function formatCurrency(amount, currency = 'EUR', locale = 'en') {
    const localeMap = {
        'en': 'en-US',
        'de': 'de-DE',
        'fr': 'fr-FR',
        'es': 'es-ES',
        'it': 'it-IT',
        'pt': 'pt-PT',
        'zh': 'zh-CN',
        'zh-TW': 'zh-TW',
        'ja': 'ja-JP',
        'ko': 'ko-KR'
    };

    const formatLocale = localeMap[locale] || 'en-US';

    return new Intl.NumberFormat(formatLocale, {
        style: 'currency',
        currency: currency
    }).format(amount);
}

/**
 * Format date with locale
 * @param {Date} date - Date to format
 * @param {string} locale - Target locale
 * @param {string} format - Format style ('short', 'medium', 'long', 'full')
 * @returns {string} Formatted date
 */
function formatDate(date, locale = 'en', format = 'medium') {
    const localeMap = {
        'en': 'en-US',
        'de': 'de-DE',
        'fr': 'fr-FR',
        'es': 'es-ES',
        'it': 'it-IT',
        'pt': 'pt-PT',
        'zh': 'zh-CN',
        'zh-TW': 'zh-TW',
        'ja': 'ja-JP',
        'ko': 'ko-KR'
    };

    const formatLocale = localeMap[locale] || 'en-US';

    const options = {
        short: { dateStyle: 'short' },
        medium: { dateStyle: 'medium' },
        long: { dateStyle: 'long' },
        full: { dateStyle: 'full' }
    };

    return new Intl.DateTimeFormat(formatLocale, options[format] || options.medium).format(date);
}

/**
 * Format number with locale
 * @param {number} number - Number to format
 * @param {string} locale - Target locale
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(number, locale = 'en', decimals = 2) {
    const localeMap = {
        'en': 'en-US',
        'de': 'de-DE',
        'fr': 'fr-FR',
        'es': 'es-ES',
        'it': 'it-IT',
        'pt': 'pt-PT',
        'zh': 'zh-CN',
        'zh-TW': 'zh-TW',
        'ja': 'ja-JP',
        'ko': 'ko-KR'
    };

    const formatLocale = localeMap[locale] || 'en-US';

    return new Intl.NumberFormat(formatLocale, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(number);
}

/**
 * Get RTL (Right-to-Left) languages
 * @returns {array} Array of RTL language codes
 */
function getRTLLanguages() {
    return ['ar', 'he', 'fa', 'ur'];
}

/**
 * Check if locale is RTL
 * @param {string} locale - Locale code
 * @returns {boolean} True if RTL
 */
function isRTL(locale) {
    return getRTLLanguages().includes(locale.split('-')[0]);
}

module.exports = {
    initializeI18n,
    normalizeLocale,
    parseAcceptLanguage,
    getTranslations,
    formatCurrency,
    formatDate,
    formatNumber,
    getRTLLanguages,
    isRTL,
    i18n
};