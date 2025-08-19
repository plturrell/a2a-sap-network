/**
 * @fileoverview SAP Internationalization Middleware
 * @description CDS middleware for seamless i18n integration with request handling,
 * locale detection, and translation services for enterprise applications.
 * @module sapI18nMiddleware
 * @since 1.0.0
 * @author A2A Network Team
 * @namespace a2a.srv.i18n
 */

const cds = require('@sap/cds');
const { normalizeLocale, formatCurrency, formatDate, formatNumber, isRTL } = require('./sapI18nConfig');

/**
 * CDS middleware for i18n integration
 * Provides translation functions in CDS request handling
 */
class I18nMiddleware {
    constructor() {
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    \n        this.intervals = new Map(); // Track intervals for cleanup}

    /**
     * Initialize i18n middleware for CDS services
     */
    async init() {
        // const { i18n } = require('./i18n-config'); // Not used currently
        
        // Add i18n to CDS request context
        cds.on('bootstrap', (app) => {
            app.use((req, res, next) => {
                if (req.__ && req.locale) {
                    // Make i18n available in CDS context
                    req._.i18n = {
                        __: req.__,
                        __n: req.__n,
                        locale: req.locale,
                        setLocale: req.setLocale,
                        getLocale: req.getLocale,
                        getCatalog: req.getCatalog
                    };
                }
                next();
            });
        });

        // Extend CDS services with i18n
        cds.Service.prototype.translate = function(key, ...args) {
            const req = cds.context;
            if (req && req._ && req._.i18n) {
                return req._.i18n.__(key, ...args);
            }
            return key;
        };

        // Add translation helper to entities
        cds.extend(cds.ql.SELECT).with(class {
            localized(locale) {
                this._locale = normalizeLocale(locale);
                return this;
            }
        });

        // Handle localized error messages
        this.setupErrorHandling();
        
        // Setup localized validations
        this.setupValidations();
    }

    /**
     * Setup localized error handling
     */
    setupErrorHandling() {
        const originalError = cds.Request.prototype.error;
        
        cds.Request.prototype.error = function(code, message, ...args) {
            // Try to translate error message
            if (this._ && this._.i18n && typeof message === 'string') {
                const translationKey = `errors.${message}`;
                const translated = this._.i18n.__(translationKey);
                
                // Use translated message if different from key
                if (translated !== translationKey) {
                    message = translated;
                }
            }
            
            return originalError.call(this, code, message, ...args);
        };
    }

    /**
     * Setup localized validations
     */
    setupValidations() {
        // Add localized validation messages
        cds.on('serving', (service) => {
            service.before('*', (req) => {
                if (req._.i18n) {
                    // Add validation helper
                    req._.validate = (value, rules) => {
                        return this.validateWithI18n(value, rules, req._.i18n);
                    };
                }
            });
        });
    }

    /**
     * Validate with i18n messages
     * @param {any} value - Value to validate
     * @param {object} rules - Validation rules
     * @param {object} i18n - i18n instance
     * @returns {object} Validation result
     */
    validateWithI18n(value, rules, i18n) {
        const errors = [];
        
        if (rules.required && !value) {
            errors.push(i18n.__('validation.required'));
        }
        
        if (rules.email && value && !this.isValidEmail(value)) {
            errors.push(i18n.__('validation.email'));
        }
        
        if (rules.url && value && !this.isValidUrl(value)) {
            errors.push(i18n.__('validation.url'));
        }
        
        if (rules.min !== undefined && value < rules.min) {
            errors.push(i18n.__('validation.min', { min: rules.min }));
        }
        
        if (rules.max !== undefined && value > rules.max) {
            errors.push(i18n.__('validation.max', { max: rules.max }));
        }
        
        if (rules.minLength && value && value.length < rules.minLength) {
            errors.push(i18n.__('validation.minLength', { min: rules.minLength }));
        }
        
        if (rules.maxLength && value && value.length > rules.maxLength) {
            errors.push(i18n.__('validation.maxLength', { max: rules.maxLength }));
        }
        
        if (rules.pattern && value && !rules.pattern.test(value)) {
            errors.push(i18n.__('validation.pattern'));
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Get localized entity data
     * @param {string} entity - Entity name
     * @param {object} data - Entity data
     * @param {string} locale - Target locale
     * @returns {object} Localized data
     */
    async getLocalizedData(entity, data, locale) {
        const normalizedLocale = normalizeLocale(locale);
        const cacheKey = `${entity}:${data.ID}:${normalizedLocale}`;
        
        // Check cache
        const cached = this.cache.get(cacheKey);
        if (cached && cached.expires > Date.now()) {
            return cached.data;
        }
        
        // Get localized texts
        const texts = await SELECT.from(`${entity}.texts`)
            .where({ ID: data.ID, locale: normalizedLocale });
        
        // Merge with base data
        const localizedData = { ...data };
        if (texts.length > 0) {
            const text = texts[0];
            Object.keys(text).forEach(key => {
                if (key !== 'ID' && key !== 'locale' && text[key]) {
                    localizedData[key] = text[key];
                }
            });
        }
        
        // Cache result
        this.cache.set(cacheKey, {
            data: localizedData,
            expires: Date.now() + this.cacheTimeout
        });
        
        return localizedData;
    }

    /**
     * Format message with locale-specific formatting
     * @param {string} template - Message template
     * @param {object} data - Data for formatting
     * @param {string} locale - Target locale
     * @returns {string} Formatted message
     */
    formatMessage(template, data, locale) {
        let formatted = template;
        
        // Replace placeholders
        Object.keys(data).forEach(key => {
            const value = data[key];
            const placeholder = new RegExp(`{{${key}}}`, 'g');
            
            // Format based on type
            if (value instanceof Date) {
                formatted = formatted.replace(placeholder, formatDate(value, locale));
            } else if (typeof value === 'number') {
                // Check if it's a currency amount
                if (key.toLowerCase().includes('price') || key.toLowerCase().includes('amount')) {
                    formatted = formatted.replace(placeholder, formatCurrency(value, 'EUR', locale));
                } else {
                    formatted = formatted.replace(placeholder, formatNumber(value, locale));
                }
            } else {
                formatted = formatted.replace(placeholder, value);
            }
        });
        
        return formatted;
    }

    /**
     * Get locale-specific configuration
     * @param {string} locale - Target locale
     * @returns {object} Locale configuration
     */
    getLocaleConfig(locale) {
        const normalizedLocale = normalizeLocale(locale);
        
        return {
            locale: normalizedLocale,
            rtl: isRTL(normalizedLocale),
            dateFormat: this.getDateFormat(normalizedLocale),
            timeFormat: this.getTimeFormat(normalizedLocale),
            numberFormat: this.getNumberFormat(normalizedLocale),
            currency: this.getCurrency(normalizedLocale)
        };
    }

    /**
     * Get date format for locale
     * @param {string} locale - Target locale
     * @returns {object} Date format configuration
     */
    getDateFormat(locale) {
        const formats = {
            'en': { short: 'MM/DD/YYYY', medium: 'MMM D, YYYY', long: 'MMMM D, YYYY' },
            'de': { short: 'DD.MM.YYYY', medium: 'D. MMM YYYY', long: 'D. MMMM YYYY' },
            'fr': { short: 'DD/MM/YYYY', medium: 'D MMM YYYY', long: 'D MMMM YYYY' },
            'es': { short: 'DD/MM/YYYY', medium: 'D MMM YYYY', long: 'D de MMMM de YYYY' },
            'it': { short: 'DD/MM/YYYY', medium: 'D MMM YYYY', long: 'D MMMM YYYY' },
            'pt': { short: 'DD/MM/YYYY', medium: 'D de MMM de YYYY', long: 'D de MMMM de YYYY' },
            'zh': { short: 'YYYY/MM/DD', medium: 'YYYY年M月D日', long: 'YYYY年M月D日' },
            'ja': { short: 'YYYY/MM/DD', medium: 'YYYY年M月D日', long: 'YYYY年M月D日' },
            'ko': { short: 'YYYY.MM.DD', medium: 'YYYY년 M월 D일', long: 'YYYY년 M월 D일' }
        };
        
        return formats[locale] || formats['en'];
    }

    /**
     * Get time format for locale
     * @param {string} locale - Target locale
     * @returns {object} Time format configuration
     */
    getTimeFormat(locale) {
        const formats = {
            'en': { short: 'h:mm A', medium: 'h:mm:ss A', long: 'h:mm:ss A z' },
            'de': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'fr': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'es': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'it': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'pt': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'zh': { short: 'HH:mm', medium: 'HH:mm:ss', long: 'HH:mm:ss z' },
            'ja': { short: 'H:mm', medium: 'H:mm:ss', long: 'H時mm分ss秒 z' },
            'ko': { short: 'H:mm', medium: 'H:mm:ss', long: 'H시 mm분 ss초 z' }
        };
        
        return formats[locale] || formats['en'];
    }

    /**
     * Get number format for locale
     * @param {string} locale - Target locale
     * @returns {object} Number format configuration
     */
    getNumberFormat(locale) {
        const formats = {
            'en': { decimal: '.', thousand: ',' },
            'de': { decimal: ',', thousand: '.' },
            'fr': { decimal: ',', thousand: ' ' },
            'es': { decimal: ',', thousand: '.' },
            'it': { decimal: ',', thousand: '.' },
            'pt': { decimal: ',', thousand: '.' },
            'zh': { decimal: '.', thousand: ',' },
            'ja': { decimal: '.', thousand: ',' },
            'ko': { decimal: '.', thousand: ',' }
        };
        
        return formats[locale] || formats['en'];
    }

    /**
     * Get default currency for locale
     * @param {string} locale - Target locale
     * @returns {string} Currency code
     */
    getCurrency(locale) {
        const currencies = {
            'en': 'USD',
            'de': 'EUR',
            'fr': 'EUR',
            'es': 'EUR',
            'it': 'EUR',
            'pt': 'EUR',
            'zh': 'CNY',
            'ja': 'JPY',
            'ko': 'KRW'
        };
        
        return currencies[locale] || 'USD';
    }

    /**
     * Email validation
     * @param {string} email - Email to validate
     * @returns {boolean} Valid or not
     */
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * URL validation
     * @param {string} url - URL to validate
     * @returns {boolean} Valid or not
     */
    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }

    /**
     * Clean expired cache entries
     */
    cleanCache() {
        const now = Date.now();
        for (const [key, value] of this.cache.entries()) {
            if (value.expires <= now) {
                this.cache.delete(key);
            }
        }
    }
}

// Create singleton instance
const i18nMiddleware = new I18nMiddleware();

// Auto-clean cache every hour
this.intervals.set('interval_383', (function(intervalId) { this.intervals.add(intervalId); return intervalId; }).call(this, setInterval(() => {
    i18nMiddleware.cleanCache();
}, 60 * 60 * 1000));

module.exports = i18nMiddleware;