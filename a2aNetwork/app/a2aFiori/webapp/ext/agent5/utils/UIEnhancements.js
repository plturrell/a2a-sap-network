sap.ui.define([
    "sap/ui/Device",
    "sap/m/MessageToast"
], (Device, MessageToast) => {
    "use strict";

    /**
     * @class UIEnhancements
     * @description Utility class providing UI enhancement functions for Agent 5.
     * Includes responsive helpers, accessibility utilities, and performance optimizations.
     */
    return {

        /**
         * @function getResponsiveContentWidth
         * @description Returns appropriate content width based on device type.
         * @returns {string} Content width percentage
         */
        getResponsiveContentWidth() {
            if (Device.system.phone) {
                return "100%";
            } else if (Device.system.tablet) {
                return "95%";
            }
            return "90%";

        },

        /**
         * @function getResponsiveContentHeight
         * @description Returns appropriate content height based on device type.
         * @returns {string} Content height percentage
         */
        getResponsiveContentHeight() {
            if (Device.system.phone) {
                return "100%";
            } else if (Device.system.tablet) {
                return "90%";
            }
            return "85%";

        },

        /**
         * @function getResponsiveThreshold
         * @description Returns appropriate threshold for list growing based on device.
         * @returns {number} Growing threshold
         */
        getResponsiveThreshold() {
            if (Device.system.phone) {
                return 10;
            } else if (Device.system.tablet) {
                return 15;
            }
            return 20;

        },

        /**
         * @function enableAccessibility
         * @description Enhances accessibility for a given control.
         * @param {sap.ui.core.Control} oControl - Control to enhance
         * @param {Object} oOptions - Accessibility options
         */
        enableAccessibility(oControl, oOptions) {
            if (!oControl || !oOptions) {return;}

            // Set ARIA attributes
            if (oOptions.ariaLabel) {
                oControl.addAriaLabelledBy(oOptions.ariaLabel);
            }

            if (oOptions.ariaDescribedBy) {
                oControl.addAriaDescribedBy(oOptions.ariaDescribedBy);
            }

            if (oOptions.role) {
                oControl.$().attr("role", oOptions.role);
            }

            // Enable keyboard navigation
            if (oOptions.enableKeyboard) {
                this.enableKeyboardNavigation(oControl);
            }
        },

        /**
         * @function enableKeyboardNavigation
         * @description Enables comprehensive keyboard navigation for a control.
         * @param {sap.ui.core.Control} oControl - Control to enhance
         */
        enableKeyboardNavigation(oControl) {
            const $control = oControl.$();

            // Ensure focusability
            $control.attr("tabindex", "0");

            // Add keyboard event handlers
            $control.on("keydown", (e) => {
                switch (e.key) {
                case "Enter":
                case " ":
                    if (oControl.firePress) {
                        e.preventDefault();
                        oControl.firePress();
                    }
                    break;
                case "Escape":
                    if (oControl.close) {
                        e.preventDefault();
                        oControl.close();
                    }
                    break;
                case "ArrowUp":
                case "ArrowDown":
                    if (oControl.getItems) {
                        e.preventDefault();
                        this._navigateItems(oControl, e.key === "ArrowUp" ? -1 : 1);
                    }
                    break;
                }
            });
        },

        /**
         * @function _navigateItems
         * @description Navigates through items in a control.
         * @param {sap.ui.core.Control} oControl - Control with items
         * @param {number} iDirection - Navigation direction (-1 or 1)
         * @private
         */
        _navigateItems(oControl, iDirection) {
            const aItems = oControl.getItems();
            if (!aItems || aItems.length === 0) {return;}

            const iCurrentIndex = aItems.findIndex((item) => {
                return item.getDomRef() === document.activeElement;
            });

            const iNewIndex = iCurrentIndex + iDirection;
            if (iNewIndex >= 0 && iNewIndex < aItems.length) {
                aItems[iNewIndex].focus();
            }
        },

        /**
         * @function createVirtualizedList
         * @description Creates a virtualized list for better performance with large datasets.
         * @param {Object} oConfig - Configuration object
         * @returns {sap.m.List} Virtualized list control
         */
        createVirtualizedList(oConfig) {
            return new sap.m.List({
                growing: true,
                growingThreshold: this.getResponsiveThreshold(),
                growingScrollToLoad: true,
                rememberSelections: false,
                mode: oConfig.mode || "None",
                updateFinished: function(oEvent) {
                    // Optimize rendering for large lists
                    const aItems = oEvent.getSource().getItems();
                    aItems.forEach((oItem, iIndex) => {
                        // Defer non-visible item rendering
                        if (iIndex > this.getResponsiveThreshold()) {
                            oItem.addStyleClass("sapUiInvisibleText");
                            setTimeout(() => {
                                oItem.removeStyleClass("sapUiInvisibleText");
                            }, 100 * (iIndex - this.getResponsiveThreshold()));
                        }
                    });
                }.bind(this)
            });
        },

        /**
         * @function optimizeDialogPerformance
         * @description Optimizes dialog performance with lazy loading and caching.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         */
        optimizeDialogPerformance(oDialog) {
            // Enable content caching
            oDialog.setContentWidth(this.getResponsiveContentWidth());
            oDialog.setContentHeight(this.getResponsiveContentHeight());

            // Add resize handler for responsive behavior
            Device.resize.attachHandler(() => {
                oDialog.setContentWidth(this.getResponsiveContentWidth());
                oDialog.setContentHeight(this.getResponsiveContentHeight());
            });

            // Optimize rendering
            oDialog.attachAfterOpen(() => {
                // Focus first input element
                setTimeout(() => {
                    const $firstInput = oDialog.$().find("input:visible:first");
                    if ($firstInput.length > 0) {
                        $firstInput.focus();
                    }
                }, 100);
            });

            // Clean up on close
            oDialog.attachAfterClose(() => {
                // Clear any temporary data
                const oModel = oDialog.getModel();
                if (oModel && oModel.setData) {
                    // Keep structure but clear values
                    const oData = oModel.getData();
                    this._clearObjectValues(oData);
                    oModel.setData(oData);
                }
            });
        },

        /**
         * @function _clearObjectValues
         * @description Recursively clears object values while preserving structure.
         * @param {Object} oObject - Object to clear
         * @private
         */
        _clearObjectValues(oObject) {
            for (const sKey in oObject) {
                if (oObject.hasOwnProperty(sKey)) {
                    const value = oObject[sKey];
                    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
                        this._clearObjectValues(value);
                    } else if (Array.isArray(value)) {
                        oObject[sKey] = [];
                    } else if (typeof value === "string") {
                        oObject[sKey] = "";
                    } else if (typeof value === "number") {
                        oObject[sKey] = 0;
                    } else if (typeof value === "boolean") {
                        oObject[sKey] = false;
                    }
                }
            }
        },

        /**
         * @function showResponsiveMessage
         * @description Shows a message with responsive positioning.
         * @param {string} sMessage - Message to display
         * @param {Object} oOptions - Display options
         */
        showResponsiveMessage(sMessage, oOptions) {
            const oDefaults = {
                duration: 3000,
                width: Device.system.phone ? "15rem" : "20rem",
                my: Device.system.phone ? "center bottom" : "right bottom",
                at: Device.system.phone ? "center bottom" : "right bottom",
                of: window,
                offset: Device.system.phone ? "0 -50" : "-10 -10",
                animationDuration: 300
            };

            const oSettings = Object.assign({}, oDefaults, oOptions);
            MessageToast.show(sMessage, oSettings);
        },

        /**
         * @function setupLazyImageLoading
         * @description Sets up lazy loading for images to improve performance.
         * @param {sap.ui.core.Control} oContainer - Container control
         */
        setupLazyImageLoading(oContainer) {
            if (!window.IntersectionObserver) {return;}

            var imageObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove("lazy-image");
                        imageObserver.unobserve(img);
                    }
                });
            });

            // Find all lazy images
            const $lazyImages = oContainer.$().find("img.lazy-image");
            $lazyImages.each(function() {
                imageObserver.observe(this);
            });
        },

        /**
         * @function debounce
         * @description Creates a debounced version of a function.
         * @param {Function} fn - Function to debounce
         * @param {number} delay - Delay in milliseconds
         * @returns {Function} Debounced function
         */
        debounce(fn, delay) {
            let timeoutId;
            return function() {
                const context = this;
                const args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    fn.apply(context, args);
                }, delay);
            };
        }
    };
});