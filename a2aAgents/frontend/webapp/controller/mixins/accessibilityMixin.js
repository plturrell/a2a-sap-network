/*! 
 * Accessibility Mixin for WCAG 2.1 AA Compliance
 * Provides comprehensive accessibility features for all controllers
 */

sap.ui.define([
    'sap/ui/base/Object',
    'sap/ui/core/library'
], (
    BaseObject,
    CoreLibrary
) => {
    'use strict';

    // Shortcuts
    const AccessibleRole = CoreLibrary.AccessibleRole;
    const AccessibleLandmarkRole = CoreLibrary.AccessibleLandmarkRole;

    /**
     * Accessibility Mixin
     * Provides WCAG 2.1 AA compliant accessibility features
     * @namespace com.sap.a2a.portal.controller.mixins
     */
    return {
        
        /**
         * Initialize accessibility features
         * Call this in onInit() of your controller
         */
        initAccessibility: function() {
            this._accessibilitySettings = {
                screenReaderEnabled: this._detectScreenReader(),
                highContrast: this._detectHighContrast(),
                reducedMotion: this._detectReducedMotion(),
                keyboardNavigation: true,
                skipLinks: []
            };
            
            this._setupKeyboardNavigation();
            this._setupSkipLinks();
            this._setupScreenReaderSupport();
            this._setupFocusManagement();
            
            // Listen for accessibility setting changes
            this._attachAccessibilityListeners();
        },

        /**
         * Detect if screen reader is active
         * @private
         * @returns {boolean} True if screen reader is detected
         */
        _detectScreenReader: function() {
            // Check for common screen reader indicators
            return !!(
                window.speechSynthesis ||
                navigator.userAgent.indexOf('NVDA') > -1 ||
                navigator.userAgent.indexOf('JAWS') > -1 ||
                navigator.userAgent.indexOf('VoiceOver') > -1 ||
                document.getElementById('nvda') ||
                window.navigator && window.navigator.userAgent.indexOf('Dragon') > -1
            );
        },

        /**
         * Detect if high contrast mode is enabled
         * @private
         * @returns {boolean} True if high contrast is enabled
         */
        _detectHighContrast: function() {
            // Create test element to detect high contrast
            const testElement = document.createElement('div');
            testElement.style.border = '1px solid';
            testElement.style.borderColor = 'red green';
            document.body.appendChild(testElement);
            
            const computedStyle = window.getComputedStyle(testElement);
            const isHighContrast = computedStyle.borderTopColor === computedStyle.borderRightColor;
            
            document.body.removeChild(testElement);
            return isHighContrast;
        },

        /**
         * Detect if reduced motion is preferred
         * @private
         * @returns {boolean} True if reduced motion is preferred
         */
        _detectReducedMotion: function() {
            return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        },

        /**
         * Set up keyboard navigation
         * @private
         */
        _setupKeyboardNavigation: function() {
            const oView = this.getView();
            if (!oView) return;

            // Add keyboard event handler to view
            oView.addEventDelegate({
                onkeydown: this._handleKeyDown.bind(this),
                onkeyup: this._handleKeyUp.bind(this)
            });

            // Set up roving tabindex for complex widgets
            this._setupRovingTabindex();
        },

        /**
         * Handle keyboard navigation
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         */
        _handleKeyDown: function(oEvent) {
            const oSource = oEvent.getSource();
            const iKeyCode = oEvent.which || oEvent.keyCode;

            // Handle escape key for dialogs and popups
            if (iKeyCode === 27) { // Escape
                this._handleEscapeKey(oEvent);
            }

            // Handle enter/space for custom controls
            if (iKeyCode === 13 || iKeyCode === 32) { // Enter or Space
                this._handleActivationKey(oEvent, oSource);
            }

            // Handle arrow keys for navigation
            if ([37, 38, 39, 40].indexOf(iKeyCode) > -1) { // Arrow keys
                this._handleArrowNavigation(oEvent, oSource, iKeyCode);
            }

            // Handle Tab key for focus management
            if (iKeyCode === 9) { // Tab
                this._handleTabNavigation(oEvent);
            }
        },

        /**
         * Handle key up events
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         */
        _handleKeyUp: function(oEvent) {
            // Implementation for key up handling if needed
        },

        /**
         * Handle escape key press
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         */
        _handleEscapeKey: function(oEvent) {
            // Close any open dialogs, popovers, or menus
            const aDialogs = sap.ui.getCore().byFieldGroupId('dialog');
            aDialogs.forEach((oDialog) => {
                if (oDialog.isOpen && oDialog.isOpen()) {
                    oDialog.close();
                }
            });

            // Close any open popovers
            const aPopovers = sap.ui.getCore().byFieldGroupId('popover');
            aPopovers.forEach((oPopover) => {
                if (oPopover.isOpen && oPopover.isOpen()) {
                    oPopover.close();
                }
            });
        },

        /**
         * Handle activation keys (Enter/Space)
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         * @param {sap.ui.core.Control} oSource Source control
         */
        _handleActivationKey: function(oEvent, oSource) {
            // Check if control has press event
            if (oSource && typeof oSource.firePress === 'function') {
                oEvent.preventDefault();
                oSource.firePress();
            }
        },

        /**
         * Handle arrow key navigation
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         * @param {sap.ui.core.Control} oSource Source control
         * @param {number} iKeyCode Key code
         */
        _handleArrowNavigation: function(oEvent, oSource, iKeyCode) {
            // Implement arrow key navigation for custom widgets
            if (oSource && oSource.getParent) {
                const oParent = oSource.getParent();
                
                // Handle table navigation
                if (oParent && oParent.getMetadata().getName() === 'sap.ui.table.Table') {
                    this._handleTableNavigation(oEvent, oParent, iKeyCode);
                }
                
                // Handle list navigation
                if (oParent && oParent.getMetadata().getName() === 'sap.m.List') {
                    this._handleListNavigation(oEvent, oParent, iKeyCode);
                }
            }
        },

        /**
         * Handle table keyboard navigation
         * @private
         * @param {sap.ui.base.Event} oEvent Keyboard event
         * @param {sap.ui.table.Table} oTable Table control
         * @param {number} iKeyCode Key code
         */
        _handleTableNavigation: function(oEvent, oTable, iKeyCode) {
            const iCurrentRow = oTable.getSelectedIndex();
            const iRowCount = oTable.getVisibleRowCount();
            let iNewRow = iCurrentRow;

            switch (iKeyCode) {
                case 38: // Up arrow
                    iNewRow = Math.max(0, iCurrentRow - 1);
                    break;
                case 40: // Down arrow
                    iNewRow = Math.min(iRowCount - 1, iCurrentRow + 1);
                    break;
                case 36: // Home
                    iNewRow = 0;
                    break;
                case 35: // End
                    iNewRow = iRowCount - 1;
                    break;
            }

            if (iNewRow !== iCurrentRow) {
                oEvent.preventDefault();
                oTable.setSelectedIndex(iNewRow);
                this._announceToScreenReader(`Row ${  iNewRow + 1  } of ${  iRowCount  } selected`);
            }
        },

        /**
         * Set up skip links for keyboard users
         * @private
         */
        _setupSkipLinks: function() {
            const oView = this.getView();
            if (!oView) return;

            // Create skip link container
            const oSkipLinkContainer = new sap.m.HBox({
                class: 'sapUiSkipLinks',
                visible: false
            });

            // Add common skip links
            this._addSkipLink(oSkipLinkContainer, 'skipToMainContent', '{i18n>accessibility.skipToContent}');
            this._addSkipLink(oSkipLinkContainer, 'skipToNavigation', '{i18n>accessibility.skipToNavigation}');
            
            // Insert skip links at the beginning of the view
            if (oView.getContent && oView.getContent().length > 0) {
                oView.insertContent(oSkipLinkContainer, 0);
            }

            this._skipLinkContainer = oSkipLinkContainer;
        },

        /**
         * Add a skip link
         * @private
         * @param {sap.m.HBox} oContainer Skip link container
         * @param {string} sId Skip link ID
         * @param {string} sText Skip link text
         */
        _addSkipLink: function(oContainer, sId, sText) {
            const oSkipLink = new sap.m.Link({
                id: `${this.getView().getId()  }--${  sId}`,
                text: sText,
                press: this._handleSkipLink.bind(this, sId),
                class: 'sapUiSkipLink'
            });

            // Show skip link on focus
            oSkipLink.addEventDelegate({
                onfocusin: function() {
                    oContainer.setVisible(true);
                },
                onfocusout: function() {
                    setTimeout(() => {
                        if (!oContainer.$().find(':focus').length) {
                            oContainer.setVisible(false);
                        }
                    }, 100);
                }
            });

            oContainer.addItem(oSkipLink);
            this._accessibilitySettings.skipLinks.push({
                id: sId,
                control: oSkipLink
            });
        },

        /**
         * Handle skip link activation
         * @private
         * @param {string} sSkipId Skip link ID
         */
        _handleSkipLink: function(sSkipId) {
            let oTargetElement;

            switch (sSkipId) {
                case 'skipToMainContent':
                    oTargetElement = this.getView().byId('mainContent') || 
                                   this.getView().$().find('[role=\'main\']').first();
                    break;
                case 'skipToNavigation':
                    oTargetElement = this.getView().byId('navigation') ||
                                   this.getView().$().find('[role=\'navigation\']').first();
                    break;
            }

            if (oTargetElement) {
                if (oTargetElement.focus) {
                    oTargetElement.focus();
                } else if (oTargetElement.length > 0) {
                    oTargetElement[0].focus();
                }
            }
        },

        /**
         * Set up screen reader support
         * @private
         */
        _setupScreenReaderSupport: function() {
            // Create live region for dynamic announcements
            this._createLiveRegion();
            
            // Set up ARIA labels and descriptions
            this._setupAriaLabels();
            
            // Set up landmark roles
            this._setupLandmarkRoles();
        },

        /**
         * Create live region for screen reader announcements
         * @private
         */
        _createLiveRegion: function() {
            const oLiveRegion = document.createElement('div');
            oLiveRegion.id = 'a2a-live-region';
            oLiveRegion.setAttribute('aria-live', 'polite');
            oLiveRegion.setAttribute('aria-atomic', 'true');
            oLiveRegion.className = 'sapUiInvisibleText';
            document.body.appendChild(oLiveRegion);
            
            this._liveRegion = oLiveRegion;
        },

        /**
         * Set up ARIA labels and descriptions
         * @private
         */
        _setupAriaLabels: function() {
            const oView = this.getView();
            if (!oView) return;

            // Add aria-label to main view
            oView.addCustomData(new sap.ui.core.CustomData({
                key: 'aria-label',
                value: '{i18n>app.title}',
                writeToDom: true
            }));

            // Set up form labels
            this._setupFormLabels();
            
            // Set up table headers
            this._setupTableHeaders();
        },

        /**
         * Set up form labels for accessibility
         * @private
         */
        _setupFormLabels: function() {
            const oView = this.getView();
            if (!oView) return;

            // Find all input controls and ensure they have labels
            const aInputs = oView.findAggregatedObjects(true, (oControl) => {
                return oControl.isA('sap.m.InputBase');
            });

            aInputs.forEach((oInput) => {
                if (!oInput.getAriaLabelledBy().length && !oInput.getAriaLabel()) {
                    // Try to find associated label
                    const oLabel = this._findAssociatedLabel(oInput);
                    if (oLabel) {
                        oInput.addAriaLabelledBy(oLabel);
                    }
                }
            });
        },

        /**
         * Find associated label for an input control
         * @private
         * @param {sap.m.InputBase} oInput Input control
         * @returns {sap.m.Label} Associated label
         */
        _findAssociatedLabel: function(oInput) {
            let oParent = oInput.getParent();
            
            while (oParent) {
                if (oParent.isA('sap.ui.layout.form.FormElement')) {
                    return oParent.getLabel();
                }
                if (oParent.isA('sap.m.VBox') || oParent.isA('sap.m.HBox')) {
                    const aItems = oParent.getItems();
                    const iInputIndex = aItems.indexOf(oInput);
                    if (iInputIndex > 0 && aItems[iInputIndex - 1].isA('sap.m.Label')) {
                        return aItems[iInputIndex - 1];
                    }
                }
                oParent = oParent.getParent();
            }
            
            return null;
        },

        /**
         * Set up table headers for screen readers
         * @private
         */
        _setupTableHeaders: function() {
            const oView = this.getView();
            if (!oView) return;

            const aTables = oView.findAggregatedObjects(true, (oControl) => {
                return oControl.isA('sap.ui.table.Table') || oControl.isA('sap.m.Table');
            });

            aTables.forEach((oTable) => {
                this._enhanceTableAccessibility(oTable);
            });
        },

        /**
         * Enhance table accessibility
         * @private
         * @param {sap.ui.table.Table|sap.m.Table} oTable Table control
         */
        _enhanceTableAccessibility: function(oTable) {
            // Add table caption if not present
            if (!oTable.getAriaLabelledBy().length) {
                const sTableTitle = oTable.getTitle && oTable.getTitle() ? 
                                 oTable.getTitle() : 'Data Table';
                
                const oInvisibleText = new sap.ui.core.InvisibleText({
                    text: `${sTableTitle  }. Use arrow keys to navigate.`
                });
                
                oTable.addAriaLabelledBy(oInvisibleText);
            }

            // Set appropriate ARIA role
            oTable.addCustomData(new sap.ui.core.CustomData({
                key: 'role',
                value: 'grid',
                writeToDom: true
            }));
        },

        /**
         * Set up landmark roles
         * @private
         */
        _setupLandmarkRoles: function() {
            const oView = this.getView();
            if (!oView) return;

            // Set main landmark
            const oMainContent = oView.byId('mainContent');
            if (oMainContent) {
                oMainContent.addCustomData(new sap.ui.core.CustomData({
                    key: 'role',
                    value: 'main',
                    writeToDom: true
                }));
            }

            // Set navigation landmarks
            const oNavigation = oView.byId('navigation');
            if (oNavigation) {
                oNavigation.addCustomData(new sap.ui.core.CustomData({
                    key: 'role',
                    value: 'navigation',
                    writeToDom: true
                }));
            }

            // Set banner and contentinfo if applicable
            this._setupHeaderFooterLandmarks();
        },

        /**
         * Set up header and footer landmarks
         * @private
         */
        _setupHeaderFooterLandmarks: function() {
            const oView = this.getView();
            if (!oView) return;

            const oHeader = oView.byId('pageHeader');
            if (oHeader) {
                oHeader.addCustomData(new sap.ui.core.CustomData({
                    key: 'role',
                    value: 'banner',
                    writeToDom: true
                }));
            }

            const oFooter = oView.byId('pageFooter');
            if (oFooter) {
                oFooter.addCustomData(new sap.ui.core.CustomData({
                    key: 'role',
                    value: 'contentinfo',
                    writeToDom: true
                }));
            }
        },

        /**
         * Set up focus management
         * @private
         */
        _setupFocusManagement: function() {
            // Set up focus trap for modal dialogs
            this._setupFocusTrap();
            
            // Set up focus restoration
            this._setupFocusRestoration();
            
            // Set up roving tabindex
            this._setupRovingTabindex();
        },

        /**
         * Set up focus trap for modal content
         * @private
         */
        _setupFocusTrap: function() {
            this._focusStack = [];
            
            // Listen for dialog open/close events
            sap.ui.getCore().getEventBus().subscribe('sap.m', 'Dialog', this._handleDialogOpen.bind(this));
        },

        /**
         * Handle dialog open for focus management
         * @private
         * @param {string} sChannelId Channel ID
         * @param {string} sEventId Event ID
         * @param {object} mParameters Event parameters
         */
        _handleDialogOpen: function(sChannelId, sEventId, mParameters) {
            if (mParameters && mParameters.dialog) {
                const oDialog = mParameters.dialog;
                
                // Store current focus
                this._focusStack.push(document.activeElement);
                
                // Set focus to first focusable element in dialog
                setTimeout(() => {
                    const oFirstFocusable = this._findFirstFocusableElement(oDialog);
                    if (oFirstFocusable) {
                        oFirstFocusable.focus();
                    }
                }, 100);
                
                // Set up focus trap
                this._trapFocusInDialog(oDialog);
            }
        },

        /**
         * Find first focusable element in container
         * @private
         * @param {sap.ui.core.Control} oContainer Container control
         * @returns {Element} First focusable element
         */
        _findFirstFocusableElement: function(oContainer) {
            const sFocusableSelector = 'input:not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])';
            const $focusable = oContainer.$().find(sFocusableSelector);
            return $focusable.length > 0 ? $focusable[0] : null;
        },

        /**
         * Announce text to screen readers
         * @param {string} sText Text to announce
         * @param {string} sPriority Priority level (polite, assertive)
         */
        _announceToScreenReader: function(sText, sPriority) {
            if (!this._liveRegion) {
                this._createLiveRegion();
            }
            
            this._liveRegion.setAttribute('aria-live', sPriority || 'polite');
            this._liveRegion.textContent = sText;
            
            // Clear after announcement
            setTimeout(() => {
                this._liveRegion.textContent = '';
            }, 1000);
        },

        /**
         * Get accessibility settings
         * @returns {object} Current accessibility settings
         */
        getAccessibilitySettings: function() {
            return this._accessibilitySettings;
        },

        /**
         * Update accessibility setting
         * @param {string} sSetting Setting name
         * @param {any} vValue Setting value
         */
        updateAccessibilitySetting: function(sSetting, vValue) {
            if (this._accessibilitySettings) {
                this._accessibilitySettings[sSetting] = vValue;
                this._handleAccessibilityChange(sSetting, vValue);
            }
        },

        /**
         * Handle accessibility setting changes
         * @private
         * @param {string} sSetting Setting name
         * @param {any} vValue Setting value
         */
        _handleAccessibilityChange: function(sSetting, vValue) {
            switch (sSetting) {
                case 'reducedMotion':
                    this._handleReducedMotionChange(vValue);
                    break;
                case 'highContrast':
                    this._handleHighContrastChange(vValue);
                    break;
                case 'screenReaderEnabled':
                    this._handleScreenReaderChange(vValue);
                    break;
            }
        },

        /**
         * Handle reduced motion preference change
         * @private
         * @param {boolean} bEnabled Whether reduced motion is enabled
         */
        _handleReducedMotionChange: function(bEnabled) {
            const oView = this.getView();
            if (!oView) return;

            if (bEnabled) {
                oView.addStyleClass('sapUiReducedMotion');
            } else {
                oView.removeStyleClass('sapUiReducedMotion');
            }
        },

        /**
         * Handle high contrast mode change
         * @private
         * @param {boolean} bEnabled Whether high contrast is enabled
         */
        _handleHighContrastChange: function(bEnabled) {
            const oView = this.getView();
            if (!oView) return;

            if (bEnabled) {
                oView.addStyleClass('sapUiHighContrast');
            } else {
                oView.removeStyleClass('sapUiHighContrast');
            }
        },

        /**
         * Attach accessibility listeners
         * @private
         */
        _attachAccessibilityListeners: function() {
            // Listen for media query changes
            if (window.matchMedia) {
                const oReducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
                oReducedMotionQuery.addListener((oQuery) => {
                    this.updateAccessibilitySetting('reducedMotion', oQuery.matches);
                });
            }
        }
    };
});