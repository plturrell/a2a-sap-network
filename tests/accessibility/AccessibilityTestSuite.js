sap.ui.define([
    "sap/ui/test/Opa5",
    "sap/ui/test/actions/Press",
    "sap/ui/test/matchers/PropertyStrictEquals"
], function(Opa5, Press, PropertyStrictEquals) {
    "use strict";

    /**
     * Accessibility Test Suite
     * Automated tests for WCAG 2.1 AA compliance
     */

    QUnit.module("Accessibility Tests", {
        beforeEach() {
            // Initialize accessibility testing engine
            this.oAccessibilityEngine = new AccessibilityEngine();
        },

        afterEach() {
            this.oAccessibilityEngine.destroy();
        }
    });

    /**
     * Accessibility Testing Engine
     * Provides methods to test various accessibility requirements
     */
    function AccessibilityEngine() {
        this.aViolations = [];
        this.oResults = {};
    }

    AccessibilityEngine.prototype.destroy = function() {
        this.aViolations = null;
        this.oResults = null;
    };

    /**
     * Test keyboard navigation compliance
     */
    AccessibilityEngine.prototype.testKeyboardNavigation = function(oElement) {
        const aFocusableElements = this._getFocusableElements(oElement);
        const aResults = [];

        aFocusableElements.forEach((element, index) => {
            const oResult = {
                element,
                tabIndex: element.tabIndex,
                hasTabIndex: element.hasAttribute("tabindex"),
                isKeyboardAccessible: this._isKeyboardAccessible(element),
                hasVisualFocusIndicator: this._hasVisualFocusIndicator(element)
            };

            aResults.push(oResult);

            // Check for violations
            if (!oResult.isKeyboardAccessible) {
                this.aViolations.push({
                    type: "keyboard-navigation",
                    severity: "error",
                    element,
                    message: "Element is not keyboard accessible"
                });
            }

            if (!oResult.hasVisualFocusIndicator) {
                this.aViolations.push({
                    type: "focus-indicator",
                    severity: "error",
                    element,
                    message: "Element lacks visible focus indicator"
                });
            }
        });

        return aResults;
    };

    /**
     * Test color contrast compliance (WCAG AA: 4.5:1, AAA: 7:1)
     */
    AccessibilityEngine.prototype.testColorContrast = function(oElement) {
        const aTextElements = this._getTextElements(oElement);
        const aResults = [];

        aTextElements.forEach(element => {
            const oColors = this._getElementColors(element);
            const fContrastRatio = this._calculateContrastRatio(oColors.foreground, oColors.background);
            const bIsLargeText = this._isLargeText(element);

            const oResult = {
                element,
                foregroundColor: oColors.foreground,
                backgroundColor: oColors.background,
                contrastRatio: fContrastRatio,
                isLargeText: bIsLargeText,
                passesAA: bIsLargeText ? fContrastRatio >= 3.0 : fContrastRatio >= 4.5,
                passesAAA: bIsLargeText ? fContrastRatio >= 4.5 : fContrastRatio >= 7.0
            };

            aResults.push(oResult);

            // Check for AA violations
            if (!oResult.passesAA) {
                this.aViolations.push({
                    type: "color-contrast",
                    severity: "error",
                    element,
                    message: `Color contrast ratio ${fContrastRatio.toFixed(2)}:1 fails WCAG AA requirements`
                });
            }
        });

        return aResults;
    };

    /**
     * Test ARIA attributes and semantic structure
     */
    AccessibilityEngine.prototype.testARIACompliance = function(oElement) {
        const aAllElements = oElement.querySelectorAll("*");
        const aResults = [];

        Array.from(aAllElements).forEach(element => {
            const oResult = {
                element,
                role: element.getAttribute("role"),
                ariaLabel: element.getAttribute("aria-label"),
                ariaLabelledBy: element.getAttribute("aria-labelledby"),
                ariaDescribedBy: element.getAttribute("aria-describedby"),
                hasValidRole: this._hasValidARIARole(element),
                hasAccessibleName: this._hasAccessibleName(element),
                hasProperARIAAttributes: this._hasProperARIAAttributes(element)
            };

            aResults.push(oResult);

            // Check for violations
            if (this._requiresAccessibleName(element) && !oResult.hasAccessibleName) {
                this.aViolations.push({
                    type: "aria-missing-name",
                    severity: "error",
                    element,
                    message: "Interactive element lacks accessible name"
                });
            }

            if (oResult.role && !oResult.hasValidRole) {
                this.aViolations.push({
                    type: "aria-invalid-role",
                    severity: "error",
                    element,
                    message: `Invalid ARIA role: ${oResult.role}`
                });
            }
        });

        return aResults;
    };

    /**
     * Test heading hierarchy (H1-H6 structure)
     */
    AccessibilityEngine.prototype.testHeadingHierarchy = function(oElement) {
        const aHeadings = Array.from(oElement.querySelectorAll("h1, h2, h3, h4, h5, h6, [role=\"heading\"]"));
        const aResults = [];
        let iPreviousLevel = 0;

        aHeadings.forEach((heading, index) => {
            const iCurrentLevel = this._getHeadingLevel(heading);
            const bIsValidSequence = index === 0 ? iCurrentLevel === 1 : iCurrentLevel <= iPreviousLevel + 1;

            const oResult = {
                element: heading,
                level: iCurrentLevel,
                text: heading.textContent.trim(),
                isValidSequence: bIsValidSequence,
                hasAccessibleText: heading.textContent.trim().length > 0
            };

            aResults.push(oResult);

            // Check for violations
            if (!bIsValidSequence) {
                this.aViolations.push({
                    type: "heading-hierarchy",
                    severity: "error",
                    element: heading,
                    message: `Heading level ${iCurrentLevel} creates invalid hierarchy (previous: ${iPreviousLevel})`
                });
            }

            if (!oResult.hasAccessibleText) {
                this.aViolations.push({
                    type: "heading-empty",
                    severity: "error",
                    element: heading,
                    message: "Heading element has no accessible text"
                });
            }

            iPreviousLevel = iCurrentLevel;
        });

        return aResults;
    };

    /**
     * Test form accessibility (labels, fieldsets, etc.)
     */
    AccessibilityEngine.prototype.testFormAccessibility = function(oElement) {
        const aFormElements = Array.from(oElement.querySelectorAll("input, select, textarea, button"));
        const aResults = [];

        aFormElements.forEach(element => {
            const oResult = {
                element,
                type: element.type || element.tagName.toLowerCase(),
                hasLabel: this._hasAssociatedLabel(element),
                hasAccessibleName: this._hasAccessibleName(element),
                isRequired: element.hasAttribute("required") || element.getAttribute("aria-required") === "true",
                hasRequiredIndicator: this._hasRequiredIndicator(element),
                hasErrorMessage: this._hasErrorMessage(element)
            };

            aResults.push(oResult);

            // Check for violations
            if (["input", "select", "textarea"].includes(oResult.type) && !oResult.hasAccessibleName) {
                this.aViolations.push({
                    type: "form-missing-label",
                    severity: "error",
                    element,
                    message: "Form control lacks accessible label"
                });
            }

            if (oResult.isRequired && !oResult.hasRequiredIndicator) {
                this.aViolations.push({
                    type: "form-missing-required",
                    severity: "warning",
                    element,
                    message: "Required form field lacks clear indication"
                });
            }
        });

        return aResults;
    };

    // Helper methods
    AccessibilityEngine.prototype._getFocusableElements = function(oElement) {
        const sSelector = "a[href], button, input, select, textarea, [tabindex]:not([tabindex=\"-1\"]), [contenteditable=\"true\"]";
        return Array.from(oElement.querySelectorAll(sSelector));
    };

    AccessibilityEngine.prototype._isKeyboardAccessible = function(oElement) {
        return oElement.tabIndex >= 0 && !oElement.disabled;
    };

    AccessibilityEngine.prototype._hasVisualFocusIndicator = function(oElement) {
        // Simulate focus and check for visual changes
        const oOriginalStyle = window.getComputedStyle(oElement);
        oElement.focus();
        const oFocusedStyle = window.getComputedStyle(oElement, ":focus");

        // Check if outline or border changes on focus
        const bHasOutline = oFocusedStyle.outline !== "none" && oFocusedStyle.outline !== oOriginalStyle.outline;
        const bHasBorderChange = oFocusedStyle.border !== oOriginalStyle.border;
        const bHasBoxShadowChange = oFocusedStyle.boxShadow !== oOriginalStyle.boxShadow;

        return bHasOutline || bHasBorderChange || bHasBoxShadowChange;
    };

    AccessibilityEngine.prototype._getTextElements = function(oElement) {
        // Get all elements that contain visible text
        const aAllElements = Array.from(oElement.querySelectorAll("*"));
        return aAllElements.filter(el => {
            const sText = el.textContent.trim();
            const oStyle = window.getComputedStyle(el);
            return sText.length > 0 && oStyle.display !== "none" && oStyle.visibility !== "hidden";
        });
    };

    AccessibilityEngine.prototype._getElementColors = function(oElement) {
        const oStyle = window.getComputedStyle(oElement);
        return {
            foreground: this._parseColor(oStyle.color),
            background: this._parseColor(oStyle.backgroundColor)
        };
    };

    AccessibilityEngine.prototype._parseColor = function(sColor) {
        // Convert color to RGB values
        const oDiv = document.createElement("div");
        oDiv.style.color = sColor;
        document.body.appendChild(oDiv);
        const oComputedStyle = window.getComputedStyle(oDiv);
        const sComputedColor = oComputedStyle.color;
        document.body.removeChild(oDiv);

        const aRgbMatch = sComputedColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (aRgbMatch) {
            return {
                r: parseInt(aRgbMatch[1]),
                g: parseInt(aRgbMatch[2]),
                b: parseInt(aRgbMatch[3])
            };
        }

        return { r: 0, g: 0, b: 0 }; // Default to black
    };

    AccessibilityEngine.prototype._calculateContrastRatio = function(oForeground, oBackground) {
        const fL1 = this._getRelativeLuminance(oForeground);
        const fL2 = this._getRelativeLuminance(oBackground);
        const fLighter = Math.max(fL1, fL2);
        const fDarker = Math.min(fL1, fL2);
        return (fLighter + 0.05) / (fDarker + 0.05);
    };

    AccessibilityEngine.prototype._getRelativeLuminance = function(oColor) {
        const fR = this._getSRGB(oColor.r / 255);
        const fG = this._getSRGB(oColor.g / 255);
        const fB = this._getSRGB(oColor.b / 255);
        return 0.2126 * fR + 0.7152 * fG + 0.0722 * fB;
    };

    AccessibilityEngine.prototype._getSRGB = function(fColorValue) {
        if (fColorValue <= 0.03928) {
            return fColorValue / 12.92;
        }
        return Math.pow((fColorValue + 0.055) / 1.055, 2.4);
    };

    AccessibilityEngine.prototype._isLargeText = function(oElement) {
        const oStyle = window.getComputedStyle(oElement);
        const iFontSize = parseFloat(oStyle.fontSize);
        const sFontWeight = oStyle.fontWeight;

        // Large text is 18pt (24px) or 14pt (18.5px) if bold
        return iFontSize >= 24 || (iFontSize >= 18.5 && (sFontWeight === "bold" || parseInt(sFontWeight) >= 700));
    };

    AccessibilityEngine.prototype._hasValidARIARole = function(oElement) {
        const sRole = oElement.getAttribute("role");
        if (!sRole) {
            return true;
        } // No role is valid

        // List of valid ARIA roles
        const aValidRoles = [
            "alert", "alertdialog", "application", "article", "banner", "button", "checkbox",
            "columnheader", "combobox", "complementary", "contentinfo", "definition",
            "dialog", "directory", "document", "form", "grid", "gridcell", "group",
            "heading", "img", "link", "list", "listbox", "listitem", "log", "main",
            "marquee", "math", "menu", "menubar", "menuitem", "menuitemcheckbox",
            "menuitemradio", "navigation", "note", "option", "presentation", "progressbar",
            "radio", "radiogroup", "region", "row", "rowgroup", "rowheader", "scrollbar",
            "search", "separator", "slider", "spinbutton", "status", "tab", "tablist",
            "tabpanel", "textbox", "timer", "toolbar", "tooltip", "tree", "treegrid",
            "treeitem"
        ];

        return aValidRoles.includes(sRole);
    };

    AccessibilityEngine.prototype._hasAccessibleName = function(oElement) {
        return !!(
            oElement.getAttribute("aria-label") ||
            oElement.getAttribute("aria-labelledby") ||
            oElement.getAttribute("title") ||
            this._getAssociatedLabelText(oElement) ||
            oElement.textContent.trim()
        );
    };

    AccessibilityEngine.prototype._requiresAccessibleName = function(oElement) {
        const sTagName = oElement.tagName.toLowerCase();
        const sType = oElement.type;
        const sRole = oElement.getAttribute("role");

        return ["button", "a", "input", "select", "textarea"].includes(sTagName) ||
               ["button", "link", "textbox", "combobox"].includes(sRole) ||
               (sTagName === "input" && ["button", "submit", "reset", "image"].includes(sType));
    };

    AccessibilityEngine.prototype._getHeadingLevel = function(oElement) {
        const sTagName = oElement.tagName.toLowerCase();
        if (sTagName.match(/^h[1-6]$/)) {
            return parseInt(sTagName.charAt(1));
        }

        const sAriaLevel = oElement.getAttribute("aria-level");
        return sAriaLevel ? parseInt(sAriaLevel) : 2; // Default to level 2 for role="heading"
    };

    AccessibilityEngine.prototype._hasAssociatedLabel = function(oElement) {
        const sId = oElement.id;
        if (sId) {
            const oLabel = document.querySelector(`label[for="${sId}"]`);
            if (oLabel) {
                return true;
            }
        }

        const oParentLabel = oElement.closest("label");
        return !!oParentLabel;
    };

    AccessibilityEngine.prototype._getAssociatedLabelText = function(oElement) {
        const sLabelledBy = oElement.getAttribute("aria-labelledby");
        if (sLabelledBy) {
            const aIds = sLabelledBy.split(" ");
            return aIds.map(sId => {
                const oElement = document.getElementById(sId);
                return oElement ? oElement.textContent.trim() : "";
            }).join(" ");
        }

        const sId = oElement.id;
        if (sId) {
            const oLabel = document.querySelector(`label[for="${sId}"]`);
            if (oLabel) {
                return oLabel.textContent.trim();
            }
        }

        const oParentLabel = oElement.closest("label");
        return oParentLabel ? oParentLabel.textContent.trim() : "";
    };

    AccessibilityEngine.prototype._hasRequiredIndicator = function(oElement) {
        const bHasAriaRequired = oElement.getAttribute("aria-required") === "true";
        const bHasRequiredAttr = oElement.hasAttribute("required");
        const bHasVisualIndicator = this._hasVisualRequiredIndicator(oElement);

        return bHasAriaRequired || bHasRequiredAttr || bHasVisualIndicator;
    };

    AccessibilityEngine.prototype._hasVisualRequiredIndicator = function(oElement) {
        // Check for common visual indicators like asterisks
        const sAssociatedText = this._getAssociatedLabelText(oElement);
        return sAssociatedText.includes("*") || sAssociatedText.toLowerCase().includes("required");
    };

    AccessibilityEngine.prototype._hasErrorMessage = function(oElement) {
        const sDescribedBy = oElement.getAttribute("aria-describedby");
        if (sDescribedBy) {
            const aIds = sDescribedBy.split(" ");
            return aIds.some(sId => {
                const oElement = document.getElementById(sId);
                return oElement && oElement.getAttribute("role") === "alert";
            });
        }
        return false;
    };

    // Export for use in tests
    window.AccessibilityEngine = AccessibilityEngine;

    return {
        AccessibilityEngine
    };
});