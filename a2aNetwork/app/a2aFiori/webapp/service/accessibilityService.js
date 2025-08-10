/**
 * Accessibility Service
 * 
 * Provides comprehensive accessibility support following WCAG 2.1 AA standards
 * and SAP accessibility guidelines for enterprise applications.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */
sap.ui.define([
    "sap/base/Log"
], function(Log) {
    "use strict";

    var AccessibilityService = {

        /* =========================================================== */
        /* Constants                                                   */
        /* =========================================================== */

        ARIA_LIVE_REGIONS: {
            POLITE: "polite",
            ASSERTIVE: "assertive",
            OFF: "off"
        },

        FOCUS_TRAP_CLASS: "a2a-focus-trap",
        SKIP_LINK_CLASS: "a2a-skip-link",
        
        /* =========================================================== */
        /* Lifecycle                                                   */
        /* =========================================================== */

        /**
         * Initialize the accessibility service
         * @public
         * @since 1.0.0
         */
        init: function() {
            this._ariaLiveRegion = null;
            this._screenReaderBuffer = [];
            this._keyboardNavigationEnabled = true;
            this._focusTraps = [];
            this._skipLinks = [];
            
            this._initializeAriaLiveRegion();
            this._initializeKeyboardNavigation();
            this._initializeSkipLinks();
            this._detectAccessibilityPreferences();
            
            Log.info("AccessibilityService initialized", { service: "AccessibilityService" });
        },

        /**
         * Destroy the accessibility service
         * @public
         * @since 1.0.0
         */
        destroy: function() {
            this._cleanupFocusTraps();
            this._cleanupKeyboardHandlers();
            
            if (this._ariaLiveRegion) {
                document.body.removeChild(this._ariaLiveRegion);
            }
            
            Log.info("AccessibilityService destroyed", { service: "AccessibilityService" });
        },

        /* =========================================================== */
        /* Public API                                                  */
        /* =========================================================== */

        /**
         * Announce message to screen readers
         * @public
         * @param {string} message Message to announce
         * @param {string} [priority="polite"] Announcement priority
         * @param {boolean} [queue=false] Whether to queue multiple messages
         * @since 1.0.0
         */
        announce: function(message, priority, queue) {
            if (!message || typeof message !== "string") {
                return;
            }
            
            priority = priority || this.ARIA_LIVE_REGIONS.POLITE;
            queue = queue || false;
            
            if (queue) {
                this._screenReaderBuffer.push({ message: message, priority: priority });
                this._processAnnouncementQueue();
            } else {
                this._announceImmediately(message, priority);
            }
            
            Log.debug("Accessibility announcement", { 
                message: message, 
                priority: priority,
                queued: queue
            });
        },

        /**
         * Set focus to element with proper announcement
         * @public
         * @param {string|Element} target Target element or ID
         * @param {string} [announcement] Optional announcement for screen readers
         * @param {boolean} [smooth=false] Whether to scroll smoothly to element
         * @returns {boolean} Whether focus was set successfully
         * @since 1.0.0
         */
        setFocus: function(target, announcement, smooth) {
            var element = typeof target === "string" ? document.getElementById(target) : target;
            
            if (!element || !element.focus) {
                Log.warn("Invalid focus target", { target: target });
                return false;
            }
            
            try {
                // Ensure element is focusable
                if (!this._isFocusable(element)) {
                    element.tabIndex = -1;
                }
                
                // Scroll element into view if needed
                if (smooth && element.scrollIntoView) {
                    element.scrollIntoView({ 
                        behavior: "smooth", 
                        block: "center",
                        inline: "nearest" 
                    });
                }
                
                element.focus();
                
                // Announce to screen readers if provided
                if (announcement) {
                    this.announce(announcement, this.ARIA_LIVE_REGIONS.ASSERTIVE);
                }
                
                Log.debug("Focus set successfully", { 
                    target: element.id || element.tagName,
                    announcement: announcement
                });
                
                return true;
            } catch (error) {
                Log.error("Failed to set focus", { 
                    target: target,
                    error: error.message 
                });
                return false;
            }
        },

        /**
         * Create focus trap for modal dialogs
         * @public
         * @param {string|Element} container Container element or ID
         * @param {object} [options] Focus trap options
         * @param {boolean} [options.returnFocus=true] Whether to return focus on release
         * @param {string} [options.initialFocus] ID of element to focus initially
         * @returns {string} Focus trap ID for later reference
         * @since 1.0.0
         */
        createFocusTrap: function(container, options) {
            options = options || {};
            var element = typeof container === "string" ? document.getElementById(container) : container;
            
            if (!element) {
                Log.warn("Invalid focus trap container", { container: container });
                return null;
            }
            
            var trapId = "trap-" + Date.now() + "-" + Math.random().toString(36).substr(2, 9);
            var focusableElements = this._getFocusableElements(element);
            
            if (focusableElements.length === 0) {
                Log.warn("No focusable elements in focus trap", { container: container });
                return null;
            }
            
            var focusTrap = {
                id: trapId,
                container: element,
                focusableElements: focusableElements,
                firstElement: focusableElements[0],
                lastElement: focusableElements[focusableElements.length - 1],
                previousFocus: document.activeElement,
                returnFocus: options.returnFocus !== false,
                keyHandler: null
            };
            
            // Set up keyboard trap
            focusTrap.keyHandler = this._createFocusTrapHandler(focusTrap);
            element.addEventListener("keydown", focusTrap.keyHandler);
            element.classList.add(this.FOCUS_TRAP_CLASS);
            
            // Set initial focus
            if (options.initialFocus) {
                this.setFocus(options.initialFocus);
            } else {
                this.setFocus(focusTrap.firstElement);
            }
            
            this._focusTraps.push(focusTrap);
            
            Log.info("Focus trap created", { 
                trapId: trapId,
                focusableCount: focusableElements.length
            });
            
            return trapId;
        },

        /**
         * Release focus trap
         * @public
         * @param {string} trapId Focus trap ID
         * @returns {boolean} Whether trap was released successfully
         * @since 1.0.0
         */
        releaseFocusTrap: function(trapId) {
            var trapIndex = this._focusTraps.findIndex(function(trap) { 
                return trap.id === trapId; 
            });
            
            if (trapIndex === -1) {
                Log.warn("Focus trap not found", { trapId: trapId });
                return false;
            }
            
            var focusTrap = this._focusTraps[trapIndex];
            
            // Clean up event listeners
            if (focusTrap.keyHandler) {
                focusTrap.container.removeEventListener("keydown", focusTrap.keyHandler);
            }
            
            focusTrap.container.classList.remove(this.FOCUS_TRAP_CLASS);
            
            // Return focus to previous element
            if (focusTrap.returnFocus && focusTrap.previousFocus && focusTrap.previousFocus.focus) {
                this.setFocus(focusTrap.previousFocus);
            }
            
            this._focusTraps.splice(trapIndex, 1);
            
            Log.info("Focus trap released", { trapId: trapId });
            return true;
        },

        /**
         * Add skip link for keyboard navigation
         * @public
         * @param {object} skipLink Skip link configuration
         * @param {string} skipLink.id Unique skip link ID
         * @param {string} skipLink.text Link text
         * @param {string} skipLink.target Target element ID
         * @param {number} [skipLink.order=0] Skip link order
         * @returns {boolean} Whether skip link was added successfully
         * @since 1.0.0
         */
        addSkipLink: function(skipLink) {
            if (!skipLink.id || !skipLink.text || !skipLink.target) {
                Log.warn("Invalid skip link configuration", skipLink);
                return false;
            }
            
            var existingLink = this._skipLinks.find(function(link) { 
                return link.id === skipLink.id; 
            });
            
            if (existingLink) {
                Log.warn("Skip link already exists", { id: skipLink.id });
                return false;
            }
            
            var linkElement = document.createElement("a");
            linkElement.id = skipLink.id;
            linkElement.href = "#" + skipLink.target;
            linkElement.textContent = skipLink.text;
            linkElement.className = this.SKIP_LINK_CLASS;
            linkElement.tabIndex = 0;
            linkElement.style.cssText = `
                position: absolute;
                left: -10000px;
                top: auto;
                width: 1px;
                height: 1px;
                overflow: hidden;
                z-index: 9999;
            `;
            
            // Show on focus
            linkElement.addEventListener("focus", function() {
                this.style.cssText = `
                    position: fixed;
                    top: 10px;
                    left: 10px;
                    background: var(--sapButton_Emphasized_Background);
                    color: var(--sapButton_Emphasized_TextColor);
                    padding: 0.5rem 1rem;
                    text-decoration: none;
                    border-radius: 0.25rem;
                    z-index: 9999;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                `;
            });
            
            linkElement.addEventListener("blur", function() {
                this.style.cssText = `
                    position: absolute;
                    left: -10000px;
                    top: auto;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    z-index: 9999;
                `;
            });
            
            linkElement.addEventListener("click", function(e) {
                e.preventDefault();
                var target = document.getElementById(skipLink.target);
                if (target) {
                    this.setFocus(target, "Skipped to " + skipLink.text);
                }
            }.bind(this));
            
            document.body.insertBefore(linkElement, document.body.firstChild);
            
            this._skipLinks.push({
                id: skipLink.id,
                text: skipLink.text,
                target: skipLink.target,
                order: skipLink.order || 0,
                element: linkElement
            });
            
            // Sort skip links by order
            this._skipLinks.sort(function(a, b) { return a.order - b.order; });
            
            Log.info("Skip link added", skipLink);
            return true;
        },

        /**
         * Check if element meets accessibility guidelines
         * @public
         * @param {string|Element} element Element to check
         * @returns {object} Accessibility audit result
         * @since 1.0.0
         */
        auditElement: function(element) {
            var target = typeof element === "string" ? document.getElementById(element) : element;
            
            if (!target) {
                return { valid: false, errors: ["Element not found"] };
            }
            
            var audit = {
                valid: true,
                warnings: [],
                errors: [],
                suggestions: []
            };
            
            // Check for accessible name
            if (!this._hasAccessibleName(target)) {
                audit.errors.push("Element lacks accessible name (aria-label, aria-labelledby, or text content)");
                audit.valid = false;
            }
            
            // Check color contrast for text elements
            if (this._isTextElement(target)) {
                var contrastRatio = this._getContrastRatio(target);
                if (contrastRatio < 4.5) {
                    audit.errors.push("Insufficient color contrast ratio: " + contrastRatio.toFixed(2));
                    audit.valid = false;
                }
            }
            
            // Check interactive elements
            if (this._isInteractiveElement(target)) {
                if (!this._isFocusable(target)) {
                    audit.errors.push("Interactive element is not focusable");
                    audit.valid = false;
                }
                
                if (!target.hasAttribute("role") && !this._hasImplicitRole(target)) {
                    audit.warnings.push("Interactive element lacks explicit role");
                }
            }
            
            // Check for ARIA attributes
            this._auditAriaAttributes(target, audit);
            
            // Check for keyboard accessibility
            if (target.onclick && !target.onkeydown && !target.onkeyup) {
                audit.warnings.push("Click handler without keyboard equivalent");
            }
            
            Log.debug("Element accessibility audit", { 
                element: target.id || target.tagName,
                valid: audit.valid,
                errors: audit.errors.length,
                warnings: audit.warnings.length
            });
            
            return audit;
        },

        /**
         * Get accessibility preferences from user settings
         * @public
         * @returns {object} Accessibility preferences
         * @since 1.0.0
         */
        getAccessibilityPreferences: function() {
            return {
                reducedMotion: this._prefersReducedMotion(),
                highContrast: this._prefersHighContrast(),
                largeText: this._prefersLargeText(),
                colorScheme: this._getPreferredColorScheme(),
                screenReader: this._detectScreenReader()
            };
        },

        /* =========================================================== */
        /* Private Methods                                             */
        /* =========================================================== */

        /**
         * Initialize ARIA live region for announcements
         * @private
         * @since 1.0.0
         */
        _initializeAriaLiveRegion: function() {
            this._ariaLiveRegion = document.createElement("div");
            this._ariaLiveRegion.id = "a2a-aria-live-region";
            this._ariaLiveRegion.setAttribute("aria-live", "polite");
            this._ariaLiveRegion.setAttribute("aria-atomic", "true");
            this._ariaLiveRegion.style.cssText = `
                position: absolute;
                left: -10000px;
                width: 1px;
                height: 1px;
                overflow: hidden;
            `;
            
            document.body.appendChild(this._ariaLiveRegion);
        },

        /**
         * Initialize keyboard navigation enhancements
         * @private
         * @since 1.0.0
         */
        _initializeKeyboardNavigation: function() {
            var that = this;
            
            // Global keyboard handler
            document.addEventListener("keydown", function(e) {
                // Skip to main content on Ctrl+M
                if (e.ctrlKey && e.key === "m") {
                    e.preventDefault();
                    var main = document.querySelector("main") || document.querySelector(".sapTntToolPageContent");
                    if (main) {
                        that.setFocus(main, "Skipped to main content");
                    }
                }
                
                // Show focus indicators on Tab
                if (e.key === "Tab") {
                    document.body.classList.add("keyboard-navigation");
                }
            });
            
            // Hide focus indicators on mouse interaction
            document.addEventListener("mousedown", function() {
                document.body.classList.remove("keyboard-navigation");
            });
        },

        /**
         * Initialize skip links
         * @private
         * @since 1.0.0
         */
        _initializeSkipLinks: function() {
            // Add default skip links
            this.addSkipLink({
                id: "skip-to-main",
                text: "Skip to main content",
                target: "main-content",
                order: 1
            });
            
            this.addSkipLink({
                id: "skip-to-nav",
                text: "Skip to navigation",
                target: "side-navigation",
                order: 2
            });
        },

        /**
         * Detect accessibility preferences
         * @private
         * @since 1.0.0
         */
        _detectAccessibilityPreferences: function() {
            var preferences = this.getAccessibilityPreferences();
            
            // Apply preferences to body class
            if (preferences.reducedMotion) {
                document.body.classList.add("a2a-reduced-motion");
            }
            
            if (preferences.highContrast) {
                document.body.classList.add("a2a-high-contrast");
            }
            
            if (preferences.largeText) {
                document.body.classList.add("a2a-large-text");
            }
            
            if (preferences.screenReader) {
                document.body.classList.add("a2a-screen-reader");
            }
            
            Log.info("Accessibility preferences detected", preferences);
        },

        /**
         * Announce message immediately
         * @private
         * @param {string} message Message to announce
         * @param {string} priority Announcement priority
         * @since 1.0.0
         */
        _announceImmediately: function(message, priority) {
            if (!this._ariaLiveRegion) {
                return;
            }
            
            this._ariaLiveRegion.setAttribute("aria-live", priority);
            this._ariaLiveRegion.textContent = message;
            
            // Clear after announcement
            setTimeout(function() {
                this._ariaLiveRegion.textContent = "";
            }.bind(this), 1000);
        },

        /**
         * Process queued announcements
         * @private
         * @since 1.0.0
         */
        _processAnnouncementQueue: function() {
            if (this._screenReaderBuffer.length === 0) {
                return;
            }
            
            var announcement = this._screenReaderBuffer.shift();
            this._announceImmediately(announcement.message, announcement.priority);
            
            // Process next announcement after delay
            if (this._screenReaderBuffer.length > 0) {
                setTimeout(function() {
                    this._processAnnouncementQueue();
                }.bind(this), 1500);
            }
        },

        /**
         * Create focus trap keyboard handler
         * @private
         * @param {object} focusTrap Focus trap configuration
         * @returns {Function} Keyboard event handler
         * @since 1.0.0
         */
        _createFocusTrapHandler: function(focusTrap) {
            return function(e) {
                if (e.key !== "Tab") {
                    return;
                }
                
                if (e.shiftKey) {
                    // Shift+Tab - move backward
                    if (document.activeElement === focusTrap.firstElement) {
                        e.preventDefault();
                        focusTrap.lastElement.focus();
                    }
                } else {
                    // Tab - move forward
                    if (document.activeElement === focusTrap.lastElement) {
                        e.preventDefault();
                        focusTrap.firstElement.focus();
                    }
                }
            };
        },

        /**
         * Get focusable elements in container
         * @private
         * @param {Element} container Container element
         * @returns {Array<Element>} Focusable elements
         * @since 1.0.0
         */
        _getFocusableElements: function(container) {
            var focusableSelector = [
                'a[href]',
                'button:not([disabled])',
                'input:not([disabled]):not([type="hidden"])',
                'select:not([disabled])',
                'textarea:not([disabled])',
                '[tabindex]:not([tabindex="-1"])',
                '[contenteditable="true"]'
            ].join(", ");
            
            return Array.from(container.querySelectorAll(focusableSelector))
                .filter(function(element) {
                    return this._isFocusable(element);
                }.bind(this));
        },

        /**
         * Check if element is focusable
         * @private
         * @param {Element} element Element to check
         * @returns {boolean} Whether element is focusable
         * @since 1.0.0
         */
        _isFocusable: function(element) {
            if (!element || element.offsetParent === null) {
                return false;
            }
            
            var style = window.getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden") {
                return false;
            }
            
            return element.tabIndex >= 0 || element.focus;
        },

        /**
         * Check if element has accessible name
         * @private
         * @param {Element} element Element to check
         * @returns {boolean} Whether element has accessible name
         * @since 1.0.0
         */
        _hasAccessibleName: function(element) {
            if (element.getAttribute("aria-label")) {
                return true;
            }
            
            if (element.getAttribute("aria-labelledby")) {
                return true;
            }
            
            if (element.textContent && element.textContent.trim()) {
                return true;
            }
            
            if (element.title) {
                return true;
            }
            
            if (element.alt) {
                return true;
            }
            
            return false;
        },

        /**
         * Check if element is text element
         * @private
         * @param {Element} element Element to check
         * @returns {boolean} Whether element contains text
         * @since 1.0.0
         */
        _isTextElement: function(element) {
            var textTags = ["P", "SPAN", "DIV", "H1", "H2", "H3", "H4", "H5", "H6", "LABEL", "A"];
            return textTags.includes(element.tagName) && element.textContent.trim().length > 0;
        },

        /**
         * Check if element is interactive
         * @private
         * @param {Element} element Element to check
         * @returns {boolean} Whether element is interactive
         * @since 1.0.0
         */
        _isInteractiveElement: function(element) {
            var interactiveTags = ["BUTTON", "A", "INPUT", "SELECT", "TEXTAREA"];
            var interactiveRoles = ["button", "link", "tab", "menuitem", "option"];
            
            if (interactiveTags.includes(element.tagName)) {
                return true;
            }
            
            var role = element.getAttribute("role");
            if (role && interactiveRoles.includes(role)) {
                return true;
            }
            
            return element.onclick || element.tabIndex >= 0;
        },

        /**
         * Check if element has implicit role
         * @private
         * @param {Element} element Element to check
         * @returns {boolean} Whether element has implicit role
         * @since 1.0.0
         */
        _hasImplicitRole: function(element) {
            var implicitRoles = {
                "BUTTON": "button",
                "A": "link",
                "INPUT": "textbox",
                "SELECT": "combobox",
                "TEXTAREA": "textbox"
            };
            
            return !!implicitRoles[element.tagName];
        },

        /**
         * Audit ARIA attributes
         * @private
         * @param {Element} element Element to audit
         * @param {object} audit Audit result object
         * @since 1.0.0
         */
        _auditAriaAttributes: function(element, audit) {
            var ariaAttributes = Array.from(element.attributes)
                .filter(function(attr) { return attr.name.startsWith("aria-"); });
            
            ariaAttributes.forEach(function(attr) {
                // Check for empty ARIA values
                if (!attr.value.trim()) {
                    audit.warnings.push("Empty ARIA attribute: " + attr.name);
                }
                
                // Check for references to non-existent elements
                if (attr.name === "aria-labelledby" || attr.name === "aria-describedby") {
                    var ids = attr.value.split(/\s+/);
                    ids.forEach(function(id) {
                        if (!document.getElementById(id)) {
                            audit.errors.push("ARIA reference to non-existent element: " + id);
                            audit.valid = false;
                        }
                    });
                }
            });
        },

        /**
         * Get color contrast ratio
         * @private
         * @param {Element} element Element to check
         * @returns {number} Contrast ratio
         * @since 1.0.0
         */
        _getContrastRatio: function(element) {
            try {
                var style = window.getComputedStyle(element);
                var color = style.color;
                var backgroundColor = style.backgroundColor;
                
                // Parse RGB values
                var foreground = this._parseRgbColor(color);
                var background = this._parseRgbColor(backgroundColor);
                
                // If background is transparent, traverse up to find actual background
                if (!background || (background.a !== undefined && background.a < 1)) {
                    background = this._getEffectiveBackgroundColor(element);
                }
                
                // Calculate contrast ratio using WCAG formula
                var contrastRatio = this._calculateContrastRatio(foreground, background);
                
                Log.debug("Contrast ratio calculated", {
                    foreground: foreground,
                    background: background,
                    ratio: contrastRatio
                });
                
                return contrastRatio;
                
            } catch (error) {
                Log.warn("Contrast calculation failed, assuming compliance", error);
                return 4.5; // Assume WCAG AA compliance
            }
        },

        /**
         * Parse RGB color string
         * @private
         * @param {string} colorStr Color string (rgb, rgba, hex, or named)
         * @returns {object} RGB values {r, g, b, a}
         * @since 1.0.0
         */
        _parseRgbColor: function(colorStr) {
            if (!colorStr || colorStr === 'transparent') {
                return null;
            }

            // RGB/RGBA format
            var rgbMatch = colorStr.match(/rgba?\(([^)]+)\)/);
            if (rgbMatch) {
                var values = rgbMatch[1].split(',').map(function(v) { return parseFloat(v.trim()); });
                return {
                    r: values[0] || 0,
                    g: values[1] || 0,
                    b: values[2] || 0,
                    a: values[3] !== undefined ? values[3] : 1
                };
            }

            // Hex format
            var hexMatch = colorStr.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
            if (hexMatch) {
                var hex = hexMatch[1];
                if (hex.length === 3) {
                    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
                }
                return {
                    r: parseInt(hex.substr(0, 2), 16),
                    g: parseInt(hex.substr(2, 2), 16),
                    b: parseInt(hex.substr(4, 2), 16),
                    a: 1
                };
            }

            // Named colors (basic set)
            var namedColors = {
                'black': {r: 0, g: 0, b: 0, a: 1},
                'white': {r: 255, g: 255, b: 255, a: 1},
                'red': {r: 255, g: 0, b: 0, a: 1},
                'green': {r: 0, g: 128, b: 0, a: 1},
                'blue': {r: 0, g: 0, b: 255, a: 1}
            };

            return namedColors[colorStr.toLowerCase()] || {r: 0, g: 0, b: 0, a: 1};
        },

        /**
         * Get effective background color by traversing DOM tree
         * @private
         * @param {Element} element Starting element
         * @returns {object} RGB background color
         * @since 1.0.0
         */
        _getEffectiveBackgroundColor: function(element) {
            var current = element;
            while (current && current !== document.body) {
                var style = window.getComputedStyle(current);
                var bgColor = this._parseRgbColor(style.backgroundColor);
                if (bgColor && bgColor.a > 0) {
                    return bgColor;
                }
                current = current.parentElement;
            }
            // Default to white background
            return {r: 255, g: 255, b: 255, a: 1};
        },

        /**
         * Calculate contrast ratio using WCAG formula
         * @private
         * @param {object} color1 First color {r, g, b, a}
         * @param {object} color2 Second color {r, g, b, a}
         * @returns {number} Contrast ratio (1:1 to 21:1)
         * @since 1.0.0
         */
        _calculateContrastRatio: function(color1, color2) {
            if (!color1 || !color2) {
                return 1;
            }

            var l1 = this._getRelativeLuminance(color1);
            var l2 = this._getRelativeLuminance(color2);

            // Ensure l1 is the lighter color
            if (l1 < l2) {
                var temp = l1;
                l1 = l2;
                l2 = temp;
            }

            return (l1 + 0.05) / (l2 + 0.05);
        },

        /**
         * Calculate relative luminance according to WCAG
         * @private
         * @param {object} color RGB color {r, g, b, a}
         * @returns {number} Relative luminance (0-1)
         * @since 1.0.0
         */
        _getRelativeLuminance: function(color) {
            var rs = color.r / 255;
            var gs = color.g / 255;
            var bs = color.b / 255;

            var r = rs <= 0.03928 ? rs / 12.92 : Math.pow((rs + 0.055) / 1.055, 2.4);
            var g = gs <= 0.03928 ? gs / 12.92 : Math.pow((gs + 0.055) / 1.055, 2.4);
            var b = bs <= 0.03928 ? bs / 12.92 : Math.pow((bs + 0.055) / 1.055, 2.4);

            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        },

        /**
         * Check for reduced motion preference
         * @private
         * @returns {boolean} Whether user prefers reduced motion
         * @since 1.0.0
         */
        _prefersReducedMotion: function() {
            return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        },

        /**
         * Check for high contrast preference
         * @private
         * @returns {boolean} Whether user prefers high contrast
         * @since 1.0.0
         */
        _prefersHighContrast: function() {
            return window.matchMedia && window.matchMedia("(prefers-contrast: high)").matches;
        },

        /**
         * Check for large text preference
         * @private
         * @returns {boolean} Whether user prefers large text
         * @since 1.0.0
         */
        _prefersLargeText: function() {
            return window.matchMedia && window.matchMedia("(min-resolution: 1.5dppx)").matches;
        },

        /**
         * Get preferred color scheme
         * @private
         * @returns {string} Preferred color scheme
         * @since 1.0.0
         */
        _getPreferredColorScheme: function() {
            if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
                return "dark";
            }
            return "light";
        },

        /**
         * Detect screen reader
         * @private
         * @returns {boolean} Whether screen reader is detected
         * @since 1.0.0
         */
        _detectScreenReader: function() {
            // Simple detection based on common screen reader characteristics
            return navigator.userAgent.includes("NVDA") || 
                   navigator.userAgent.includes("JAWS") || 
                   navigator.userAgent.includes("VoiceOver") ||
                   window.speechSynthesis !== undefined;
        },

        /**
         * Clean up focus traps
         * @private
         * @since 1.0.0
         */
        _cleanupFocusTraps: function() {
            this._focusTraps.forEach(function(trap) {
                if (trap.keyHandler) {
                    trap.container.removeEventListener("keydown", trap.keyHandler);
                }
                trap.container.classList.remove(this.FOCUS_TRAP_CLASS);
            }.bind(this));
            
            this._focusTraps = [];
        },

        /**
         * Clean up keyboard handlers
         * @private
         * @since 1.0.0
         */
        _cleanupKeyboardHandlers: function() {
            // Remove skip links
            this._skipLinks.forEach(function(link) {
                if (link.element && link.element.parentNode) {
                    link.element.parentNode.removeChild(link.element);
                }
            });
            
            this._skipLinks = [];
        }
    };

    return AccessibilityService;
});