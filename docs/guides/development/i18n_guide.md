# A2A Network Internationalization (i18n) Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Supported Languages](#supported-languages)
4. [Implementation Details](#implementation-details)
5. [Using i18n in Code](#using-i18n-in-code)
6. [Translation Management](#translation-management)
7. [Adding New Languages](#adding-new-languages)
8. [Testing Localization](#testing-localization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The A2A Network platform supports full internationalization (i18n) and localization (l10n) to enable global usage. This guide describes how to work with the multilingual features of the platform.

### Key Features

- **22 supported languages** including major business languages
- **Automatic locale detection** from user preferences and browser settings
- **Dynamic language switching** without page reload
- **Translation management system** for administrators
- **Localized data formats** for dates, numbers, and currencies
- **Right-to-Left (RTL) support** for Arabic and Hebrew
- **SAP UI5 integration** for consistent enterprise user experience

## Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│             Frontend Layer              │
├─────────────────────────────────────────┤
│ • SAP UI5 i18n Models                   │
│ • Resource Bundles (.properties)        │
│ • Dynamic Language Switching           │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│           Backend Services              │
├─────────────────────────────────────────┤
│ • i18n Middleware                       │
│ • Translation Service                   │
│ • Locale-aware Formatting              │
│ • Error Message Translation            │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│            Database Layer               │
├─────────────────────────────────────────┤
│ • Translation Tables                    │
│ • Locale Preferences                    │
│ • Audit Logs                           │
│ • Coverage Analytics                    │
└─────────────────────────────────────────┘
```

### Technology Stack

- **Backend**: Node.js with i18n module
- **Frontend**: SAP UI5 i18n framework
- **Database**: SAP HANA Cloud with i18n schema
- **Translation Format**: JSON, Properties files, XLIFF
- **Standards**: ICU MessageFormat, BCP 47 language tags

## Supported Languages

| Language | Code | Status | Coverage | RTL |
|----------|------|---------|----------|-----|
| English | `en` | ✅ Default | 100% | No |
| German | `de` | ✅ Complete | 95% | No |
| French | `fr` | 🔄 In Progress | 80% | No |
| Spanish | `es` | 🔄 In Progress | 75% | No |
| Italian | `it` | 📋 Planned | 0% | No |
| Portuguese | `pt` | 📋 Planned | 0% | No |
| Chinese (Simplified) | `zh` | 🔄 In Progress | 60% | No |
| Chinese (Traditional) | `zh-TW` | 📋 Planned | 0% | No |
| Japanese | `ja` | 📋 Planned | 0% | No |
| Korean | `ko` | 📋 Planned | 0% | No |
| Russian | `ru` | 📋 Planned | 0% | No |
| Arabic | `ar` | 📋 Planned | 0% | Yes |
| Hebrew | `he` | 📋 Planned | 0% | Yes |
| Hindi | `hi` | 📋 Planned | 0% | No |
| Dutch | `nl` | 📋 Planned | 0% | No |
| Polish | `pl` | 📋 Planned | 0% | No |
| Turkish | `tr` | 📋 Planned | 0% | No |
| Czech | `cs` | 📋 Planned | 0% | No |
| Swedish | `sv` | 📋 Planned | 0% | No |
| Danish | `da` | 📋 Planned | 0% | No |
| Norwegian | `no` | 📋 Planned | 0% | No |
| Finnish | `fi` | 📋 Planned | 0% | No |

## Implementation Details

### Backend Configuration

The i18n system is automatically initialized when the server starts:

```javascript
// srv/server.js
const { initializeI18n } = require('./i18n/i18n-config');
const i18nMiddleware = require('./i18n/i18n-middleware');

// Initialize i18n middleware
i18nMiddleware.init();

cds.on('bootstrap', (app) => {
    // Initialize i18n with Express app
    initializeI18n(app);
});
```

### Locale Detection Priority

1. **User Preference**: Stored in user profile
2. **JWT Token**: SAP user locale from authentication
3. **Query Parameter**: `?locale=de`
4. **Cookie**: `a2a-locale` cookie value
5. **Accept-Language Header**: Browser preference
6. **Default**: English (`en`)

### Database Schema

Key entities for translation management:

```cds
entity Translations {
    key namespace : String(100);
    key key : String(200);
    key locale : String(14);
    value : String(5000);
    context : String(200);
    isReviewed : Boolean;
}

entity Locales {
    key code : String(14);
    name : String(100);
    isActive : Boolean;
    isDefault : Boolean;
    isRTL : Boolean;
}
```

## Using i18n in Code

### Backend Services

```javascript
// In CDS service implementation
class MyService extends cds.ApplicationService {
    async init() {
        this.before('CREATE', 'Agents', (req) => {
            // Use translation in service
            const welcomeMsg = req._.i18n.__('common.welcome');
            
            // Validate with localized messages
            const validation = req._.validate(req.data.name, {
                required: true,
                minLength: 3
            });
            
            if (!validation.valid) {
                req.error(400, validation.errors.join(', '));
            }
        });
        
        await super.init();
    }
}
```

### Frontend (SAP UI5)

```javascript
// In UI5 controller
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/resource/ResourceModel"
], function (Controller, ResourceModel) {
    "use strict";
    
    return Controller.extend("a2a.Controller", {
        onInit: function() {
            // Create i18n model
            const i18nModel = new ResourceModel({
                bundleName: "a2a.i18n.i18n"
            });
            this.getView().setModel(i18nModel, "i18n");
        },
        
        onSaveAgent: function() {
            const i18n = this.getView().getModel("i18n").getResourceBundle();
            const successMsg = i18n.getText("message.save.success");
            
            sap.m.MessageToast.show(successMsg);
        }
    });
});
```

### XML Views

```xml
<!-- In UI5 XML views -->
<mvc:View xmlns:mvc="sap.ui.core.mvc" xmlns="sap.m">
    <Page title="{i18n>agent.title}">
        <content>
            <Table items="{/agents}">
                <columns>
                    <Column>
                        <Text text="{i18n>agent.field.name}" />
                    </Column>
                    <Column>
                        <Text text="{i18n>agent.field.status}" />
                    </Column>
                </columns>
                <items>
                    <ColumnListItem>
                        <Text text="{name}" />
                        <Text text="{i18n>agent.status.active}" />
                    </ColumnListItem>
                </items>
            </Table>
        </content>
    </Page>
</mvc:View>
```

## Translation Management

### Admin Interface

Access the translation management interface at `/translations`:

1. **Translation Overview**
   - View all translations by language
   - Check translation coverage
   - Identify missing translations

2. **Translation Editor**
   - Edit translations inline
   - Review and approve translations
   - Add context and comments

3. **Import/Export**
   - Export translations as JSON, CSV, or XLIFF
   - Import from translation tools
   - Bulk update operations

### API Endpoints

```bash
# Get all translations for a locale
GET /api/v1/translations/Translations?locale=de

# Create or update translation
POST /api/v1/translations/Translations
{
  "namespace": "common",
  "key": "welcome",
  "locale": "de",
  "value": "Willkommen"
}

# Export translations
POST /api/v1/translations/exportTranslations
{
  "locale": "de",
  "format": "json"
}

# Get missing translations
POST /api/v1/translations/getMissingTranslations
{
  "locale": "fr"
}
```

### Translation Workflow

1. **Development Phase**
   - Developers add translation keys in code
   - English translations added to base files
   - Missing translations logged automatically

2. **Translation Phase**
   - Export translation files
   - Send to professional translators
   - Review and quality assurance

3. **Integration Phase**
   - Import translated files
   - Validation and testing
   - Deployment to production

## Adding New Languages

### Step 1: Configure Locale

```sql
INSERT INTO Locales (
    code, name, nativeName, isRTL, isActive,
    dateFormat, timeFormat, currencyCode
) VALUES (
    'fr', 'French', 'Français', false, true,
    'DD/MM/YYYY', 'HH:mm', 'EUR'
);
```

### Step 2: Create Resource Bundles

```bash
# Create UI5 resource bundle
touch app/a2a-fiori/webapp/i18n/i18n_fr.properties

# Create backend translations
mkdir -p srv/i18n/locales
touch srv/i18n/locales/fr.json
```

### Step 3: Add Translations

```properties
# app/a2a-fiori/webapp/i18n/i18n_fr.properties
appTitle=Réseau A2A
agent.title=Gestion des Agents
common.welcome=Bienvenue dans le Réseau A2A
```

```json
// srv/i18n/locales/fr.json
{
  "common": {
    "welcome": "Bienvenue dans le Réseau A2A",
    "loading": "Chargement...",
    "error": "Erreur"
  }
}
```

### Step 4: Test and Validate

```bash
# Test locale switching
curl -H "Accept-Language: fr" http://localhost:4004/api/v1/agents

# Validate translations
POST /api/v1/translations/validateTranslations
{
  "locale": "fr"
}
```

## Testing Localization

### Automated Testing

```javascript
// test/i18n.test.js
describe('Internationalization', () => {
    test('should load German translations', async () => {
        const response = await request(app)
            .get('/api/v1/agents')
            .set('Accept-Language', 'de')
            .expect(200);
        
        expect(response.body.error).toBe(undefined);
    });
    
    test('should format currency for German locale', () => {
        const formatted = formatCurrency(100.50, 'EUR', 'de');
        expect(formatted).toBe('100,50 €');
    });
});
```

### Manual Testing

1. **Browser Testing**
   - Change browser language settings
   - Test language switching in UI
   - Verify date/number formats

2. **API Testing**
   - Test with different Accept-Language headers
   - Verify error messages in correct language
   - Check locale-specific data formats

3. **Pseudo-localization**
   - Use special characters to test UI layout
   - Verify text expansion handling
   - Test RTL languages

## Best Practices

### Translation Keys

```javascript
// ✅ Good - Descriptive, hierarchical keys
i18n.__('agent.validation.name.required')
i18n.__('service.message.subscription.success')
i18n.__('workflow.error.execution.timeout')

// ❌ Bad - Generic, flat keys
i18n.__('error1')
i18n.__('msg')
i18n.__('text123')
```

### Context and Comments

```json
{
  "agent.status.busy": {
    "message": "Beschäftigt",
    "context": "Status shown when agent is processing a task",
    "maxLength": 20
  }
}
```

### Placeholder Handling

```javascript
// ✅ Good - Named placeholders
i18n.__('message.itemCount', { count: 5, type: 'agents' })
// "Found {{count}} {{type}}" → "Found 5 agents"

// ❌ Bad - Positional placeholders
i18n.__('message.itemCount', 5, 'agents')
// "Found {0} {1}" → Hard to translate
```

### Date and Number Formatting

```javascript
// ✅ Good - Use locale-aware formatting
const date = formatDate(new Date(), req.getLocale());
const price = formatCurrency(19.99, 'EUR', req.getLocale());
const number = formatNumber(1234.56, req.getLocale());

// ❌ Bad - Hardcoded formats
const date = new Date().toLocaleDateString('en-US');
const price = '$' + amount.toFixed(2);
```

### RTL Support

```css
/* CSS for RTL languages */
html[dir="rtl"] .agent-list {
    text-align: right;
}

html[dir="rtl"] .workflow-step {
    margin-left: 0;
    margin-right: 20px;
}
```

### Performance Optimization

```javascript
// ✅ Good - Cache translations
const cachedTranslations = await cacheManager.get(`i18n:${locale}`);

// ✅ Good - Load only needed translations
const agentTranslations = await getTranslations('agents', locale);

// ❌ Bad - Load all translations always
const allTranslations = await getAllTranslations();
```

## Troubleshooting

### Common Issues

#### Missing Translations

**Problem**: Key shows in UI instead of translated text

**Solution**:
```bash
# Check if key exists
GET /api/v1/translations/Translations?key=missing.key&locale=de

# Add missing translation
POST /api/v1/translations/Translations
{
  "namespace": "common",
  "key": "missing.key",
  "locale": "de",
  "value": "German translation"
}
```

#### Locale Not Detected

**Problem**: Always shows English despite browser language

**Solution**:
```javascript
// Debug locale detection
app.use((req, res, next) => {
    console.log('Accept-Language:', req.headers['accept-language']);
    console.log('Detected locale:', req.getLocale());
    next();
});
```

#### Formatting Issues

**Problem**: Dates/numbers display in wrong format

**Solution**:
```javascript
// Verify locale configuration
const localeConfig = getLocaleConfig('de');
console.log('Date format:', localeConfig.dateFormat);

// Test formatting
const formatted = formatDate(new Date(), 'de');
console.log('Formatted date:', formatted);
```

#### Performance Issues

**Problem**: Slow response with many languages

**Solution**:
```javascript
// Enable caching
const cacheConfig = {
    ttl: 3600, // 1 hour
    max: 1000  // max entries
};

// Use lazy loading
const lazyLoad = true;
initializeI18n(app, { cacheConfig, lazyLoad });
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=i18n:*
npm start

# Check translation coverage
curl http://localhost:4004/api/v1/translations/getTranslationCoverage

# Validate all translations
curl -X POST http://localhost:4004/api/v1/translations/validateTranslations
```

### Support Resources

- **Translation API**: `/api/v1/translations/*`
- **Coverage Report**: Available in admin dashboard
- **Audit Logs**: Track all translation changes
- **Community**: SAP Community i18n forums

---

## Next Steps

1. **Complete Core Languages**: Focus on German, French, Spanish
2. **Add Asian Languages**: Chinese, Japanese, Korean
3. **Implement Visual Translation Tool**: In-context editing
4. **Add Machine Translation**: For initial drafts
5. **Mobile App Localization**: Extend to mobile interfaces

---

*Last updated: November 2024 | Version 1.0.0*