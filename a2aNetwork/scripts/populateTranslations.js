const cds = require('@sap/cds');
const fs = require('fs').promises;
const path = require('path');

/**
 * Script to populate initial translations into the database
 * Reads from locale files and inserts into Translations table
 */

async function populateTranslations() {
    log.info('🌍 Starting translation population...');

    try {
        // Connect to database
        await cds.connect.to('db');
        const { Translations, Locales } = cds.entities('a2a.network.i18n');

        // Define supported locales
        const locales = [
            { code: 'en', name: 'English', nativeName: 'English', isDefault: true, isRTL: false, currencyCode: 'USD', sortOrder: 1 },
            { code: 'de', name: 'German', nativeName: 'Deutsch', isDefault: false, isRTL: false, currencyCode: 'EUR', sortOrder: 2 },
            { code: 'fr', name: 'French', nativeName: 'Français', isDefault: false, isRTL: false, currencyCode: 'EUR', sortOrder: 3 },
            { code: 'es', name: 'Spanish', nativeName: 'Español', isDefault: false, isRTL: false, currencyCode: 'EUR', sortOrder: 4 },
            { code: 'it', name: 'Italian', nativeName: 'Italiano', isDefault: false, isRTL: false, currencyCode: 'EUR', sortOrder: 5 },
            { code: 'pt', name: 'Portuguese', nativeName: 'Português', isDefault: false, isRTL: false, currencyCode: 'EUR', sortOrder: 6 },
            { code: 'zh', name: 'Chinese (Simplified)', nativeName: '简体中文', isDefault: false, isRTL: false, currencyCode: 'CNY', sortOrder: 7 },
            { code: 'zh-TW', name: 'Chinese (Traditional)', nativeName: '繁體中文', isDefault: false, isRTL: false, currencyCode: 'TWD', sortOrder: 8 },
            { code: 'ja', name: 'Japanese', nativeName: '日本語', isDefault: false, isRTL: false, currencyCode: 'JPY', sortOrder: 9 },
            { code: 'ko', name: 'Korean', nativeName: '한국어', isDefault: false, isRTL: false, currencyCode: 'KRW', sortOrder: 10 },
            { code: 'ru', name: 'Russian', nativeName: 'Русский', isDefault: false, isRTL: false, currencyCode: 'RUB', sortOrder: 11 },
            { code: 'ar', name: 'Arabic', nativeName: 'العربية', isDefault: false, isRTL: true, currencyCode: 'SAR', sortOrder: 12 },
            { code: 'he', name: 'Hebrew', nativeName: 'עברית', isDefault: false, isRTL: true, currencyCode: 'ILS', sortOrder: 13 }
        ];

        log.debug('📝 Inserting locale configurations...');

        // Insert or update locales
        for (const locale of locales) {
            try {
                const existing = await SELECT.one.from(Locales).where({ code: locale.code });

                if (existing) {
                    await UPDATE(Locales)
                        .set({
                            name: locale.name,
                            nativeName: locale.nativeName,
                            isDefault: locale.isDefault,
                            isRTL: locale.isRTL,
                            currencyCode: locale.currencyCode,
                            sortOrder: locale.sortOrder,
                            isActive: true
                        })
                        .where({ code: locale.code });
                    log.debug(`  ✓ Updated locale: ${locale.code} (${locale.name})`);
                } else {
                    await INSERT.into(Locales).entries({
                        ...locale,
                        isActive: true,
                        dateFormat: getDateFormat(locale.code),
                        timeFormat: getTimeFormat(locale.code),
                        numberFormat: getNumberFormat(locale.code)
                    });
                    log.debug(`  ✓ Created locale: ${locale.code} (${locale.name})`);
                }
            } catch (error) {
                console.error(`  ✗ Failed to process locale ${locale.code}:`, error.message);
            }
        }

        // Load translations from locale files
        const localeDir = path.join(__dirname, '../srv/i18n/locales');

        try {
            const localeFiles = await fs.readdir(localeDir);
            const jsonFiles = localeFiles.filter(file => file.endsWith('.json'));

            log.debug(`📚 Loading translations from ${jsonFiles.length} files...`);

            for (const file of jsonFiles) {
                const locale = path.basename(file, '.json');
                const filePath = path.join(localeDir, file);

                try {
                    const content = await fs.readFile(filePath, 'utf8');
                    const translations = JSON.parse(content);

                    await processTranslationFile(translations, locale, Translations);
                    log.debug(`  ✓ Processed translations for locale: ${locale}`);
                } catch (error) {
                    console.error(`  ✗ Failed to process ${file}:`, error.message);
                }
            }

        } catch (error) {
            console.warn(`⚠️  Locale directory not found: ${localeDir}`);
            log.debug('💡 Creating minimal translations for available locales...');

            // Create minimal translations for core locales
            const coreTranslations = {
                'en': await createCoreTranslations('en'),
                'de': await createCoreTranslations('de')
            };

            for (const [locale, translations] of Object.entries(coreTranslations)) {
                await processTranslationFile(translations, locale, Translations);
                log.debug(`  ✓ Created core translations for: ${locale}`);
            }
        }

        // Generate translation statistics
        await generateStatistics(Translations, Locales);

        log.debug('🎉 Translation population completed successfully!');

    } catch (error) {
        console.error('❌ Error populating translations:', error);
        process.exit(1);
    }
}

/**
 * Process translation file and insert into database
 */
async function processTranslationFile(translations, locale, Translations) {
    const flatTranslations = flattenTranslations(translations);
    let inserted = 0;
    let updated = 0;
    let skipped = 0;

    for (const [key, value] of Object.entries(flatTranslations)) {
        try {
            const [namespace, ...keyParts] = key.split('.');
            const translationKey = keyParts.join('.');

            if (!namespace || !translationKey || !value) {
                skipped++;
                continue;
            }

            const existing = await SELECT.one.from(Translations).where({
                namespace,
                key: translationKey,
                locale
            });

            if (existing) {
                if (existing.value !== value.toString()) {
                    await UPDATE(Translations)
                        .set({ value: value.toString() })
                        .where({ namespace, key: translationKey, locale });
                    updated++;
                } else {
                    skipped++;
                }
            } else {
                await INSERT.into(Translations).entries({
                    namespace,
                    key: translationKey,
                    locale,
                    value: value.toString(),
                    isReviewed: locale === 'en' // English is considered reviewed
                });
                inserted++;
            }

        } catch (error) {
            console.error(`    ✗ Failed to process key ${key}:`, error.message);
            skipped++;
        }
    }

    log.debug(`    📊 ${locale}: ${inserted} inserted, ${updated} updated, ${skipped} skipped`);
}

/**
 * Flatten nested translation object
 */
function flattenTranslations(obj, prefix = '') {
    const result = {};

    for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            Object.assign(result, flattenTranslations(value, fullKey));
        } else {
            result[fullKey] = value;
        }
    }

    return result;
}

/**
 * Create core translations for a locale
 */
async function createCoreTranslations(locale) {
    const templates = {
        'en': {
            common: {
                welcome: 'Welcome to A2A Network',
                loading: 'Loading...',
                error: 'Error',
                success: 'Success',
                warning: 'Warning',
                info: 'Information',
                save: 'Save',
                cancel: 'Cancel',
                delete: 'Delete',
                edit: 'Edit',
                create: 'Create'
            },
            agents: {
                title: 'Agents',
                createTitle: 'Create New Agent',
                fields: {
                    name: 'Agent Name',
                    status: 'Status',
                    reputation: 'Reputation'
                },
                status: {
                    active: 'Active',
                    inactive: 'Inactive'
                }
            },
            errors: {
                general: 'An error occurred. Please try again.',
                network: 'Network error. Please check your connection.',
                unauthorized: 'You are not authorized to perform this action.'
            }
        },
        'de': {
            common: {
                welcome: 'Willkommen im A2A-Netzwerk',
                loading: 'Wird geladen...',
                error: 'Fehler',
                success: 'Erfolg',
                warning: 'Warnung',
                info: 'Information',
                save: 'Speichern',
                cancel: 'Abbrechen',
                delete: 'Löschen',
                edit: 'Bearbeiten',
                create: 'Erstellen'
            },
            agents: {
                title: 'Agenten',
                createTitle: 'Neuen Agenten erstellen',
                fields: {
                    name: 'Agentenname',
                    status: 'Status',
                    reputation: 'Reputation'
                },
                status: {
                    active: 'Aktiv',
                    inactive: 'Inaktiv'
                }
            },
            errors: {
                general: 'Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.',
                network: 'Netzwerkfehler. Bitte überprüfen Sie Ihre Verbindung.',
                unauthorized: 'Sie sind nicht berechtigt, diese Aktion auszuführen.'
            }
        }
    };

    return templates[locale] || templates['en'];
}

/**
 * Generate translation statistics
 */
async function generateStatistics(Translations, Locales) {
    log.debug('📈 Generating translation statistics...');

    try {
        // Get total keys (from English)
        const totalKeys = await SELECT.count().from(Translations).where({ locale: 'en' });

        // Get coverage by locale
        const locales = await SELECT.from(Locales).where({ isActive: true });

        log.debug('\n📊 Translation Coverage Report:');
        log.debug('─'.repeat(50));

        for (const locale of locales) {
            const translatedKeys = await SELECT.count().from(Translations).where({ locale: locale.code });
            const coverage = totalKeys > 0 ? Math.round((translatedKeys / totalKeys) * 100) : 0;
            const bar = '█'.repeat(Math.floor(coverage / 5)) + '░'.repeat(20 - Math.floor(coverage / 5));

            log.debug(`${locale.code.padEnd(6)} │ ${bar} │ ${coverage}% (${translatedKeys}/${totalKeys})`);
        }

        log.debug('─'.repeat(50));

        // Missing translations report
        const missingCount = await SELECT.count().from(Translations).where({
            locale: { '!=': 'en' },
            value: ''
        });

        if (missingCount > 0) {
            log.debug(`\n⚠️  ${missingCount} missing translations found`);
        }

    } catch (error) {
        console.error('Failed to generate statistics:', error.message);
    }
}

/**
 * Get date format for locale
 */
function getDateFormat(locale) {
    const formats = {
        'en': 'MM/DD/YYYY',
        'de': 'DD.MM.YYYY',
        'fr': 'DD/MM/YYYY',
        'es': 'DD/MM/YYYY',
        'it': 'DD/MM/YYYY',
        'pt': 'DD/MM/YYYY',
        'zh': 'YYYY/MM/DD',
        'zh-TW': 'YYYY/MM/DD',
        'ja': 'YYYY/MM/DD',
        'ko': 'YYYY.MM.DD',
        'ru': 'DD.MM.YYYY',
        'ar': 'DD/MM/YYYY',
        'he': 'DD/MM/YYYY'
    };
    return formats[locale] || 'MM/DD/YYYY';
}

/**
 * Get time format for locale
 */
function getTimeFormat(locale) {
    const formats = {
        'en': 'h:mm A',
        'de': 'HH:mm',
        'fr': 'HH:mm',
        'es': 'HH:mm',
        'it': 'HH:mm',
        'pt': 'HH:mm',
        'zh': 'HH:mm',
        'zh-TW': 'HH:mm',
        'ja': 'H:mm',
        'ko': 'H:mm',
        'ru': 'HH:mm',
        'ar': 'HH:mm',
        'he': 'HH:mm'
    };
    return formats[locale] || 'h:mm A';
}

/**
 * Get number format for locale
 */
function getNumberFormat(locale) {
    const formats = {
        'en': '1,234.56',
        'de': '1.234,56',
        'fr': '1 234,56',
        'es': '1.234,56',
        'it': '1.234,56',
        'pt': '1.234,56',
        'zh': '1,234.56',
        'zh-TW': '1,234.56',
        'ja': '1,234.56',
        'ko': '1,234.56',
        'ru': '1 234,56',
        'ar': '1,234.56',
        'he': '1,234.56'
    };
    return formats[locale] || '1,234.56';
}

// Export for testing
module.exports = {
    populateTranslations,
    flattenTranslations,
    createCoreTranslations
};

// Run if called directly
if (require.main === module) {
    populateTranslations()
        .then(() => {
            log.debug('✅ Translation population completed');
            process.exit(0);
        })
        .catch((error) => {
            console.error('❌ Translation population failed:', error);
            process.exit(1);
        });
}