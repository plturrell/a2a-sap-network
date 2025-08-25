/* global sap */
sap.ui.define([
    'sap/base/Log'
], (Log) => {
    'use strict';

    /**
     * SAP Enterprise Theme Configuration
     * Manages theme switching, customization, and enterprise compliance
     */
    return {
        /**
         * Available themes configuration
         */
        themes: {
            'sap_horizon': {
                name: 'SAP Horizon',
                description: 'Modern, fresh design for contemporary business applications',
                type: 'light',
                sapCompliant: true,
                accessibility: {
                    wcag: 'AA',
                    colorContrast: '4.5:1'
                },
                colors: {
                    brand: '#0070f2',
                    background: '#ffffff',
                    surface: '#f7f7f7',
                    text: '#32363a'
                },
                fonts: {
                    family: '72, Arial, sans-serif',
                    sizes: {
                        small: '0.75rem',
                        medium: '0.875rem',
                        large: '1rem',
                        title: '1.125rem'
                    }
                }
            },
            'sap_horizon_dark': {
                name: 'SAP Horizon Dark',
                description: 'Dark variant of SAP Horizon for reduced eye strain',
                type: 'dark',
                sapCompliant: true,
                accessibility: {
                    wcag: 'AA',
                    colorContrast: '4.5:1'
                },
                colors: {
                    brand: '#7996ff',
                    background: '#1d2d3e',
                    surface: '#233040',
                    text: '#ffffff'
                },
                fonts: {
                    family: '72, Arial, sans-serif',
                    sizes: {
                        small: '0.75rem',
                        medium: '0.875rem',
                        large: '1rem',
                        title: '1.125rem'
                    }
                }
            },
            'sap_horizon_hcb': {
                name: 'SAP Horizon High Contrast Black',
                description: 'High contrast black theme for accessibility',
                type: 'hc_black',
                sapCompliant: true,
                accessibility: {
                    wcag: 'AAA',
                    colorContrast: '7:1'
                },
                colors: {
                    brand: '#7dd3f0',
                    background: '#000000',
                    surface: '#000000',
                    text: '#ffffff'
                },
                fonts: {
                    family: '72, Arial, sans-serif',
                    sizes: {
                        small: '0.75rem',
                        medium: '0.875rem',
                        large: '1rem',
                        title: '1.125rem'
                    }
                }
            },
            'sap_horizon_hcw': {
                name: 'SAP Horizon High Contrast White',
                description: 'High contrast white theme for accessibility',
                type: 'hc_white',
                sapCompliant: true,
                accessibility: {
                    wcag: 'AAA',
                    colorContrast: '7:1'
                },
                colors: {
                    brand: '#0070f2',
                    background: '#ffffff',
                    surface: '#ffffff',
                    text: '#000000'
                },
                fonts: {
                    family: '72, Arial, sans-serif',
                    sizes: {
                        small: '0.75rem',
                        medium: '0.875rem',
                        large: '1rem',
                        title: '1.125rem'
                    }
                }
            }
        },

        /**
         * Theme switching configuration
         */
        switching: {
            enableUserSwitch: true,
            enableAutoDetection: true,
            storageKey: 'sap-ui-theme',
            defaultTheme: 'sap_horizon',
            fallbackTheme: 'sap_horizon'
        },

        /**
         * Enterprise customization settings
         */
        customization: {
            companyBranding: {
                enabled: true,
                logoPath: '/images/company-logo.png',
                brandColors: {
                    primary: '#0070f2',
                    secondary: '#42b883'
                }
            },
            customCSS: {
                enabled: true,
                paths: [
                    './css/company-theme-overrides.css',
                    './css/accessibility-enhancements.css'
                ]
            }
        },

        /**
         * Initialize theme system
         */
        initialize: function() {
            // Detect user preference
            const userTheme = this._detectUserPreference();
            
            // Apply theme
            this.applyTheme(userTheme);
            
            // Setup theme monitoring
            this._setupThemeMonitoring();
            
            // Load custom CSS if enabled
            if (this.customization.customCSS.enabled) {
                this._loadCustomCSS();
            }
            
            Log.info('Theme system initialized', { theme: userTheme });
        },

        /**
         * Apply theme
         * @param {string} themeId - Theme ID to apply
         */
        applyTheme: function(themeId) {
            if (!this.themes[themeId]) {
                Log.warning('Unknown theme requested, using fallback', themeId);
                themeId = this.switching.fallbackTheme;
            }

            // Apply via UI5 Core
            if (sap.ui.getCore && sap.ui.getCore()) {
                sap.ui.getCore().applyTheme(themeId);
            }

            // Store preference
            if (this.switching.enableUserSwitch) {
                localStorage.setItem(this.switching.storageKey, themeId);
            }

            // Update CSS custom properties
            this._updateCSSCustomProperties(themeId);
            
            // Trigger theme change event
            this._fireThemeChanged(themeId);
            
            Log.info('Theme applied', { theme: themeId });
        },

        /**
         * Get current theme
         * @returns {string} Current theme ID
         */
        getCurrentTheme: function() {
            if (sap.ui.getCore && sap.ui.getCore()) {
                return sap.ui.getCore().getConfiguration().getTheme();
            }
            return this.switching.defaultTheme;
        },

        /**
         * Get available themes
         * @returns {Object} Available themes
         */
        getAvailableThemes: function() {
            return Object.keys(this.themes).map(id => ({
                id: id,
                name: this.themes[id].name,
                description: this.themes[id].description,
                type: this.themes[id].type
            }));
        },

        /**
         * Check if theme is high contrast
         * @param {string} themeId - Theme ID
         * @returns {boolean} Is high contrast
         */
        isHighContrast: function(themeId) {
            const theme = this.themes[themeId];
            return theme && (theme.type === 'hc_black' || theme.type === 'hc_white');
        },

        /**
         * Check if theme is dark
         * @param {string} themeId - Theme ID
         * @returns {boolean} Is dark theme
         */
        isDarkTheme: function(themeId) {
            const theme = this.themes[themeId];
            return theme && (theme.type === 'dark' || theme.type === 'hc_black');
        },

        /**
         * Detect user preference
         * @private
         */
        _detectUserPreference: function() {
            // 1. Check stored preference
            if (this.switching.enableUserSwitch) {
                const stored = localStorage.getItem(this.switching.storageKey);
                if (stored && this.themes[stored]) {
                    return stored;
                }
            }

            // 2. Check system preference
            if (this.switching.enableAutoDetection && window.matchMedia) {
                if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    return 'sap_horizon_dark';
                }
                if (window.matchMedia('(prefers-contrast: high)').matches) {
                    return 'sap_horizon_hcb';
                }
            }

            // 3. Use default
            return this.switching.defaultTheme;
        },

        /**
         * Setup theme monitoring for system changes
         * @private
         */
        _setupThemeMonitoring: function() {
            if (!this.switching.enableAutoDetection || !window.matchMedia) {
                return;
            }

            // Monitor dark mode preference
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            darkModeQuery.addEventListener('change', (e) => {
                if (!localStorage.getItem(this.switching.storageKey)) {
                    // Only auto-switch if user hasn't manually set preference
                    this.applyTheme(e.matches ? 'sap_horizon_dark' : 'sap_horizon');
                }
            });

            // Monitor high contrast preference
            const contrastQuery = window.matchMedia('(prefers-contrast: high)');
            contrastQuery.addEventListener('change', (e) => {
                if (!localStorage.getItem(this.switching.storageKey)) {
                    this.applyTheme(e.matches ? 'sap_horizon_hcb' : 'sap_horizon');
                }
            });
        },

        /**
         * Update CSS custom properties for theme
         * @private
         */
        _updateCSSCustomProperties: function(themeId) {
            const theme = this.themes[themeId];
            if (!theme) return;

            const root = document.documentElement;
            
            // Update color properties
            if (theme.colors) {
                Object.entries(theme.colors).forEach(([key, value]) => {
                    root.style.setProperty(`--theme-${key}`, value);
                });
            }

            // Update font properties
            if (theme.fonts) {
                root.style.setProperty('--theme-font-family', theme.fonts.family);
                if (theme.fonts.sizes) {
                    Object.entries(theme.fonts.sizes).forEach(([key, value]) => {
                        root.style.setProperty(`--theme-font-size-${key}`, value);
                    });
                }
            }

            // Add theme class to body
            document.body.className = `${document.body.className
                .replace(/\btheme-\w+/g, '')  } theme-${themeId.replace('_', '-')}`;
        },

        /**
         * Load custom CSS files
         * @private
         */
        _loadCustomCSS: function() {
            this.customization.customCSS.paths.forEach(path => {
                const link = document.createElement('link');
                link.rel = 'stylesheet';
                link.href = path;
                link.onload = () => Log.debug('Custom CSS loaded', path);
                link.onerror = () => Log.warning('Failed to load custom CSS', path);
                document.head.appendChild(link);
            });
        },

        /**
         * Fire theme changed event
         * @private
         */
        _fireThemeChanged: function(themeId) {
            const event = new CustomEvent('sapThemeChanged', {
                detail: {
                    theme: themeId,
                    themeInfo: this.themes[themeId]
                }
            });
            window.dispatchEvent(event);
        }
    };
});