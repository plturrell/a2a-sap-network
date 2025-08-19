/**
 * Language Support and Localization Tests
 * Based on testCases1Common.md - Localization Tests section
 */

describe('Language Support and Localization Tests', () => {
  const supportedLanguages = ['en', 'de', 'fr', 'es', 'ja', 'zh'];
  
  beforeEach(() => {
    cy.visit('/');
  });

  describe('Language Switching', () => {
    it('should switch language within 500ms without page reload', () => {
      // Get initial URL
      cy.url().then((initialUrl) => {
        // Open language selector
        cy.get('[data-testid="language-selector"], [aria-label*="language"]').click();
        
        // Switch to German
        cy.get('[data-language="de"]').click();
        
        // Verify no page reload occurred
        cy.url().should('eq', initialUrl);
        
        // Verify language switched within 500ms
        cy.get('html', { timeout: 500 }).should('have.attr', 'lang', 'de');
        
        // Verify UI elements are translated
        cy.get('[data-i18n]').first().should('not.contain', 'Home').and('contain', 'Startseite');
      });
    });

    it('should persist language preference across sessions', () => {
      // Switch to French
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="fr"]').click();
      
      // Verify language is set
      cy.get('html').should('have.attr', 'lang', 'fr');
      
      // Check localStorage or cookie
      cy.window().then((win) => {
        const langPref = win.localStorage.getItem('preferredLanguage') || 
                         cy.getCookie('language');
        expect(langPref).to.include('fr');
      });
      
      // Reload page
      cy.reload();
      
      // Language should still be French
      cy.get('html').should('have.attr', 'lang', 'fr');
    });

    it('should support all required languages', () => {
      supportedLanguages.forEach((lang) => {
        cy.get('[data-testid="language-selector"]').click();
        cy.get(`[data-language="${lang}"]`).should('exist').and('be.visible');
      });
    });
  });

  describe('Translation Coverage', () => {
    it('should have zero hardcoded strings in UI', () => {
      supportedLanguages.forEach((lang) => {
        // Switch language
        cy.get('[data-testid="language-selector"]').click();
        cy.get(`[data-language="${lang}"]`).click();
        
        // Check common UI elements don't contain English when in other language
        if (lang !== 'en') {
          const commonEnglishWords = ['Home', 'Settings', 'Save', 'Cancel', 'Submit', 'Search'];
          
          commonEnglishWords.forEach((word) => {
            cy.get('nav, header, button, label').each(($el) => {
              const text = $el.text();
              if (text.length > 0 && text.length < 50) { // Skip long text blocks
                expect(text).to.not.equal(word);
              }
            });
          });
        }
      });
    });

    it('should show fallback with [MISSING] prefix for untranslated strings', () => {
      // Switch to a language with potentially missing translations
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="ja"]').click();
      
      // Check for missing translation markers
      cy.get('body').then(($body) => {
        const bodyText = $body.text();
        const missingTranslations = (bodyText.match(/\[MISSING\]/g) || []).length;
        
        // Log any missing translations found
        if (missingTranslations > 0) {
          cy.log(`Found ${missingTranslations} missing translations`);
          
          // Missing translations should show English fallback
          cy.contains('[MISSING]').each(($el) => {
            const text = $el.text();
            expect(text).to.match(/\[MISSING\].*[A-Za-z]+/);
          });
        }
      });
    });

    it('should translate all form labels and placeholders', () => {
      cy.visit('/forms/sample');
      
      // Test German translations
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="de"]').click();
      
      // Check form elements
      cy.get('label').each(($label) => {
        cy.wrap($label).should('have.attr', 'data-i18n-key');
      });
      
      cy.get('input[placeholder], textarea[placeholder]').each(($input) => {
        const placeholder = $input.attr('placeholder');
        expect(placeholder).to.not.match(/^(Name|Email|Password|Search)$/);
      });
    });

    it('should translate error messages', () => {
      cy.visit('/forms/sample');
      
      // Switch to Spanish
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="es"]').click();
      
      // Trigger validation error
      cy.get('form').submit();
      
      // Error messages should be in Spanish
      cy.get('.error-message, [role="alert"]').each(($error) => {
        const text = $error.text();
        expect(text).to.not.match(/required|invalid|error/i);
        expect(text).to.match(/requerido|inválido|error/i);
      });
    });
  });

  describe('Regional Date Formats', () => {
    const dateFormats = {
      'en-US': { format: 'MM/DD/YYYY', example: '12/31/2023' },
      'en-GB': { format: 'DD/MM/YYYY', example: '31/12/2023' },
      'de-DE': { format: 'DD.MM.YYYY', example: '31.12.2023' },
      'fr-FR': { format: 'DD/MM/YYYY', example: '31/12/2023' },
      'ja-JP': { format: 'YYYY/MM/DD', example: '2023/12/31' },
      'iso': { format: 'YYYY-MM-DD', example: '2023-12-31' }
    };

    it('should display dates in correct regional format', () => {
      Object.entries(dateFormats).forEach(([locale, { format, example }]) => {
        // Set locale
        cy.window().then((win) => {
          win.localStorage.setItem('locale', locale);
        });
        cy.reload();
        
        // Check date displays
        cy.get('[data-testid="date-display"], .date, time').first().then(($date) => {
          const dateText = $date.text();
          
          // Verify format matches expected pattern
          if (locale === 'en-US') {
            expect(dateText).to.match(/\d{1,2}\/\d{1,2}\/\d{4}/);
          } else if (locale === 'de-DE') {
            expect(dateText).to.match(/\d{1,2}\.\d{1,2}\.\d{4}/);
          } else if (locale === 'ja-JP' || locale === 'iso') {
            expect(dateText).to.match(/\d{4}[-\/]\d{1,2}[-\/]\d{1,2}/);
          }
        });
      });
    });

    it('should use correct date picker format', () => {
      cy.visit('/forms/sample');
      
      // Set German locale
      cy.window().then((win) => {
        win.localStorage.setItem('locale', 'de-DE');
      });
      cy.reload();
      
      // Open date picker
      cy.get('input[type="date"], [data-testid="date-input"]').first().click();
      
      // Verify date format placeholder
      cy.get('input[type="date"], [data-testid="date-input"]').first()
        .should('have.attr', 'placeholder').and('match', /DD\.MM\.YYYY|TT\.MM\.JJJJ/);
    });
  });

  describe('Regional Number Formats', () => {
    const numberFormats = {
      'en-US': { thousand: ',', decimal: '.', example: '1,234.56' },
      'de-DE': { thousand: '.', decimal: ',', example: '1.234,56' },
      'fr-FR': { thousand: ' ', decimal: ',', example: '1 234,56' }
    };

    it('should display numbers in correct regional format', () => {
      Object.entries(numberFormats).forEach(([locale, format]) => {
        // Set locale
        cy.window().then((win) => {
          win.localStorage.setItem('locale', locale);
        });
        cy.reload();
        
        // Check number displays
        cy.get('[data-testid="number-display"], .number, .price').each(($num) => {
          const numText = $num.text();
          
          if (numText.match(/[\d.,\s]+/)) {
            if (locale === 'en-US') {
              expect(numText).to.match(/\d{1,3}(,\d{3})*(\.\d+)?/);
            } else if (locale === 'de-DE') {
              expect(numText).to.match(/\d{1,3}(\.\d{3})*(,\d+)?/);
            } else if (locale === 'fr-FR') {
              expect(numText).to.match(/\d{1,3}(\s\d{3})*(,\d+)?/);
            }
          }
        });
      });
    });

    it('should format currency with correct symbol position', () => {
      const currencyFormats = {
        'en-US': { symbol: '$', position: 'before', example: '$1,234.56' },
        'de-DE': { symbol: '€', position: 'after', example: '1.234,56 €' },
        'fr-FR': { symbol: '€', position: 'after', example: '1 234,56 €' }
      };
      
      Object.entries(currencyFormats).forEach(([locale, format]) => {
        cy.window().then((win) => {
          win.localStorage.setItem('locale', locale);
        });
        cy.reload();
        
        cy.get('[data-testid="price"], .currency, .price').each(($price) => {
          const priceText = $price.text();
          
          if (format.position === 'before') {
            expect(priceText).to.match(new RegExp(`^\\${format.symbol}`));
          } else {
            expect(priceText).to.match(new RegExp(`${format.symbol}$`));
          }
        });
      });
    });
  });

  describe('Timezone Display', () => {
    it('should show times in user timezone with UTC offset', () => {
      // Get user's timezone
      const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      
      cy.get('[data-testid="time-display"], time[datetime]').each(($time) => {
        const timeText = $time.text();
        
        // Should include timezone abbreviation or offset
        expect(timeText).to.match(/([A-Z]{3,4}|UTC[+-]\d{1,2}|\([A-Z]{3,4}\))/);
      });
    });

    it('should allow timezone selection', () => {
      cy.get('[data-testid="timezone-selector"]').should('exist');
      
      // Change timezone
      cy.get('[data-testid="timezone-selector"]').select('America/New_York');
      
      // Verify times updated
      cy.get('[data-testid="time-display"]').first().then(($time) => {
        const timeText = $time.text();
        expect(timeText).to.include('EST').or.include('EDT').or.include('UTC-5').or.include('UTC-4');
      });
    });
  });

  describe('RTL Language Support', () => {
    const rtlLanguages = ['ar', 'he'];

    it('should mirror layout for RTL languages', () => {
      // Switch to Arabic
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="ar"]').click();
      
      // Check HTML dir attribute
      cy.get('html').should('have.attr', 'dir', 'rtl');
      
      // Check layout direction
      cy.get('body').should('have.css', 'direction', 'rtl');
      
      // Navigation should be right-aligned
      cy.get('nav').should('have.css', 'text-align', 'right');
      
      // Flex containers should be reversed
      cy.get('.flex, [class*="flex"]').each(($flex) => {
        cy.wrap($flex).should('have.css', 'flex-direction').and('match', /row-reverse/);
      });
    });

    it('should mirror icons and graphics for RTL', () => {
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="ar"]').click();
      
      // Check directional icons are flipped
      cy.get('[data-icon="arrow-right"], .icon-arrow-right').each(($icon) => {
        cy.wrap($icon).should('have.css', 'transform').and('include', 'scaleX(-1)');
      });
    });
  });

  describe('Text Expansion', () => {
    it('should accommodate text expansion for German (+30%)', () => {
      // Switch to German
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="de"]').click();
      
      // Check buttons and labels don't overflow
      cy.get('button, label').each(($el) => {
        cy.wrap($el).then(($element) => {
          const scrollWidth = $element[0].scrollWidth;
          const clientWidth = $element[0].clientWidth;
          
          // Text should not overflow
          expect(scrollWidth).to.be.lte(clientWidth + 2); // Allow 2px tolerance
        });
      });
    });

    it('should accommodate text expansion for Russian (+35%)', () => {
      // Switch to Russian
      cy.get('[data-testid="language-selector"]').click();
      cy.get('[data-language="ru"]').click();
      
      // Check containers have flexible width
      cy.get('.card, .panel, [class*="container"]').each(($container) => {
        cy.wrap($container).should('not.have.css', 'overflow', 'hidden');
      });
    });
  });

  describe('Unicode and Special Characters', () => {
    it('should render Unicode characters correctly', () => {
      const testStrings = {
        'ja': '日本語のテキスト',
        'zh': '中文文本',
        'ar': 'النص العربي',
        'emoji': '😀 🎉 ✨'
      };
      
      Object.entries(testStrings).forEach(([lang, text]) => {
        if (lang !== 'emoji') {
          cy.get('[data-testid="language-selector"]').click();
          cy.get(`[data-language="${lang}"]`).click();
        }
        
        // Verify characters render properly
        cy.get('body').should('contain', text);
      });
    });

    it('should have appropriate font stack for CJK languages', () => {
      ['ja', 'zh', 'ko'].forEach((lang) => {
        cy.get('[data-testid="language-selector"]').click();
        cy.get(`[data-language="${lang}"]`).click();
        
        cy.get('body').should('have.css', 'font-family').and('match', /Noto|Yu|Hiragino|Microsoft YaHei|Meiryo/);
      });
    });
  });

  describe('Locale-Specific Validation', () => {
    beforeEach(() => {
      cy.visit('/forms/sample');
    });

    it('should validate phone numbers based on locale', () => {
      // US format
      cy.window().then((win) => {
        win.localStorage.setItem('locale', 'en-US');
      });
      cy.reload();
      
      cy.get('input[type="tel"]').type('(555) 123-4567');
      cy.get('input[type="tel"]').should('have.class', 'valid').or('not.have.class', 'invalid');
      
      // German format
      cy.window().then((win) => {
        win.localStorage.setItem('locale', 'de-DE');
      });
      cy.reload();
      
      cy.get('input[type="tel"]').clear().type('+49 30 12345678');
      cy.get('input[type="tel"]').should('have.class', 'valid').or('not.have.class', 'invalid');
    });

    it('should validate postal codes based on locale', () => {
      const postalFormats = {
        'en-US': '12345',
        'en-GB': 'SW1A 1AA',
        'de-DE': '10115',
        'fr-FR': '75001'
      };
      
      Object.entries(postalFormats).forEach(([locale, postalCode]) => {
        cy.window().then((win) => {
          win.localStorage.setItem('locale', locale);
        });
        cy.reload();
        
        cy.get('input[name="postalCode"], input[name="zipCode"]').clear().type(postalCode);
        cy.get('input[name="postalCode"], input[name="zipCode"]').blur();
        cy.get('input[name="postalCode"], input[name="zipCode"]').should('not.have.class', 'invalid');
      });
    });
  });
});