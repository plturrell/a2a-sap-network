sap.ui.define([], function() {
    "use strict";

    /**
     * SAP Enterprise Font Resource Configuration
     * Manages font loading, caching, and fallbacks
     */
    return {
        /**
         * Font resource definitions
         */
        fonts: {
            // SAP 72 Font Family
            "72": {
                name: "72",
                weights: {
                    light: {
                        weight: 300,
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Light.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Light.woff"
                        }
                    },
                    regular: {
                        weight: 400,
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Regular.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Regular.woff"
                        }
                    },
                    bold: {
                        weight: 700,
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Bold.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Bold.woff"
                        }
                    },
                    black: {
                        weight: 900,
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Black.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/base/fonts/72-Black.woff"
                        }
                    }
                }
            },
            
            // SAP Icons Font
            "SAP-icons": {
                name: "SAP-icons",
                weights: {
                    normal: {
                        weight: "normal",
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/sap_horizon/fonts/SAP-icons.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/sap_horizon/fonts/SAP-icons.woff",
                            ttf: "https://ui5.sap.com/1.120.0/resources/sap/ui/core/themes/sap_horizon/fonts/SAP-icons.ttf"
                        }
                    }
                }
            },
            
            // SAP Icons TNT (Technical Icons)
            "SAP-icons-TNT": {
                name: "SAP-icons-TNT",
                weights: {
                    normal: {
                        weight: "normal",
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/tnt/themes/sap_horizon/fonts/SAP-icons-TNT.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/tnt/themes/sap_horizon/fonts/SAP-icons-TNT.woff"
                        }
                    }
                }
            },
            
            // SAP Icons Business Suite
            "BusinessSuiteInAppSymbols": {
                name: "BusinessSuiteInAppSymbols",
                weights: {
                    normal: {
                        weight: "normal",
                        style: "normal",
                        urls: {
                            woff2: "https://ui5.sap.com/1.120.0/resources/sap/ushell/themes/sap_horizon/fonts/BusinessSuiteInAppSymbols.woff2",
                            woff: "https://ui5.sap.com/1.120.0/resources/sap/ushell/themes/sap_horizon/fonts/BusinessSuiteInAppSymbols.woff"
                        }
                    }
                }
            }
        },

        /**
         * Preload critical fonts for performance
         */
        preloadFonts: function() {
            const criticalFonts = [
                // Most important fonts to preload
                this.fonts["72"].weights.regular.urls.woff2,
                this.fonts["72"].weights.bold.urls.woff2,
                this.fonts["SAP-icons"].weights.normal.urls.woff2
            ];

            criticalFonts.forEach(url => {
                const link = document.createElement("link");
                link.rel = "preload";
                link.as = "font";
                link.type = "font/woff2";
                link.href = url;
                link.crossOrigin = "anonymous";
                document.head.appendChild(link);
            });
        },

        /**
         * Generate @font-face CSS rules
         */
        generateFontFaceRules: function() {
            let css = "";
            
            Object.values(this.fonts).forEach(font => {
                Object.values(font.weights).forEach(weight => {
                    css += `
@font-face {
    font-family: "${font.name}";
    font-weight: ${weight.weight};
    font-style: ${weight.style};
    src: ${this._generateSrcString(weight.urls)};
    font-display: swap;
}`;
                });
            });
            
            return css;
        },

        /**
         * Generate src string for @font-face
         * @private
         */
        _generateSrcString: function(urls) {
            const sources = [];
            
            if (urls.woff2) {
                sources.push(`url("${urls.woff2}") format("woff2")`);
            }
            if (urls.woff) {
                sources.push(`url("${urls.woff}") format("woff")`);
            }
            if (urls.ttf) {
                sources.push(`url("${urls.ttf}") format("truetype")`);
            }
            
            return sources.join(",\n         ");
        },

        /**
         * Initialize font loading
         */
        initialize: function() {
            // Preload critical fonts
            this.preloadFonts();
            
            // Add font-face rules
            const style = document.createElement("style");
            style.textContent = this.generateFontFaceRules();
            document.head.appendChild(style);
            
            // Monitor font loading
            if ("fonts" in document) {
                document.fonts.ready.then(() => {
                    document.documentElement.classList.add("fonts-loaded");
                    console.log("All fonts loaded successfully");
                });
            }
        },

        /**
         * Get icon font mappings
         */
        getIconMappings: function() {
            return {
                "sap-icon": {
                    fontFamily: "SAP-icons",
                    prefix: "sap-icon://",
                    codepoint: 0xe000 // Starting codepoint for SAP icons
                },
                "sap-icon-tnt": {
                    fontFamily: "SAP-icons-TNT",
                    prefix: "sap-icon://tnt/",
                    codepoint: 0xe000
                },
                "sap-icon-business": {
                    fontFamily: "BusinessSuiteInAppSymbols",
                    prefix: "sap-icon://business-suite/",
                    codepoint: 0xe000
                }
            };
        },

        /**
         * Cache fonts for offline use
         */
        cacheFontsForOffline: function() {
            if ('caches' in window) {
                const fontUrls = [];
                
                Object.values(this.fonts).forEach(font => {
                    Object.values(font.weights).forEach(weight => {
                        if (weight.urls.woff2) {
                            fontUrls.push(weight.urls.woff2);
                        }
                    });
                });
                
                caches.open('sap-fonts-v1').then(cache => {
                    cache.addAll(fontUrls).then(() => {
                        console.log("Fonts cached for offline use");
                    });
                });
            }
        }
    };
});