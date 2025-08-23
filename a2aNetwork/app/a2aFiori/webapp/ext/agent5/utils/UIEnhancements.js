sap.ui.define([
    "sap/ui/Device",
    "sap/m/MessageToast"
], function (Device, MessageToast) {
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
        getResponsiveContentWidth: function() {
            if (Device.system.phone) {
                return "100%";
            } else if (Device.system.tablet) {
                return "95%";
            } else {
                return "90%";
            }
        },
        
        /**
         * @function getResponsiveContentHeight
         * @description Returns appropriate content height based on device type.
         * @returns {string} Content height percentage
         */
        getResponsiveContentHeight: function() {
            if (Device.system.phone) {
                return "100%";
            } else if (Device.system.tablet) {
                return "90%";
            } else {
                return "85%";
            }
        },
        
        /**
         * @function getResponsiveThreshold
         * @description Returns appropriate threshold for list growing based on device.
         * @returns {number} Growing threshold
         */
        getResponsiveThreshold: function() {
            if (Device.system.phone) {
                return 10;
            } else if (Device.system.tablet) {
                return 15;
            } else {
                return 20;
            }
        },
        
        /**
         * @function enableAccessibility
         * @description Enhances accessibility for a given control.
         * @param {sap.ui.core.Control} oControl - Control to enhance
         * @param {Object} oOptions - Accessibility options
         */
        enableAccessibility: function(oControl, oOptions) {
            if (!oControl || !oOptions) return;
            
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
        enableKeyboardNavigation: function(oControl) {
            var $control = oControl.$();
            
            // Ensure focusability
            $control.attr("tabindex", "0");
            
            // Add keyboard event handlers
            $control.on("keydown", function(e) {
                switch(e.key) {
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
            }.bind(this));
        },
        
        /**
         * @function _navigateItems
         * @description Navigates through items in a control.
         * @param {sap.ui.core.Control} oControl - Control with items
         * @param {number} iDirection - Navigation direction (-1 or 1)
         * @private
         */
        _navigateItems: function(oControl, iDirection) {
            var aItems = oControl.getItems();
            if (!aItems || aItems.length === 0) return;
            
            var iCurrentIndex = aItems.findIndex(function(item) {
                return item.getDomRef() === document.activeElement;
            });
            
            var iNewIndex = iCurrentIndex + iDirection;
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
        createVirtualizedList: function(oConfig) {
            return new sap.m.List({
                growing: true,
                growingThreshold: this.getResponsiveThreshold(),
                growingScrollToLoad: true,
                rememberSelections: false,
                mode: oConfig.mode || "None",
                updateFinished: function(oEvent) {
                    // Optimize rendering for large lists
                    var aItems = oEvent.getSource().getItems();
                    aItems.forEach(function(oItem, iIndex) {
                        // Defer non-visible item rendering
                        if (iIndex > this.getResponsiveThreshold()) {
                            oItem.addStyleClass("sapUiInvisibleText");
                            setTimeout(function() {
                                oItem.removeStyleClass("sapUiInvisibleText");
                            }, 100 * (iIndex - this.getResponsiveThreshold()));
                        }
                    }.bind(this));
                }.bind(this)
            });
        },
        
        /**
         * @function optimizeDialogPerformance
         * @description Optimizes dialog performance with lazy loading and caching.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         */
        optimizeDialogPerformance: function(oDialog) {
            // Enable content caching
            oDialog.setContentWidth(this.getResponsiveContentWidth());
            oDialog.setContentHeight(this.getResponsiveContentHeight());
            
            // Add resize handler for responsive behavior
            Device.resize.attachHandler(function() {
                oDialog.setContentWidth(this.getResponsiveContentWidth());
                oDialog.setContentHeight(this.getResponsiveContentHeight());
            }.bind(this));
            
            // Optimize rendering
            oDialog.attachAfterOpen(function() {
                // Focus first input element
                setTimeout(function() {
                    var $firstInput = oDialog.$().find("input:visible:first");
                    if ($firstInput.length > 0) {
                        $firstInput.focus();
                    }
                }, 100);
            });
            
            // Clean up on close
            oDialog.attachAfterClose(function() {
                // Clear any temporary data
                var oModel = oDialog.getModel();
                if (oModel && oModel.setData) {
                    // Keep structure but clear values
                    var oData = oModel.getData();
                    this._clearObjectValues(oData);
                    oModel.setData(oData);
                }
            }.bind(this));
        },
        
        /**
         * @function _clearObjectValues
         * @description Recursively clears object values while preserving structure.
         * @param {Object} oObject - Object to clear
         * @private
         */
        _clearObjectValues: function(oObject) {
            for (var sKey in oObject) {
                if (oObject.hasOwnProperty(sKey)) {
                    var value = oObject[sKey];
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
        showResponsiveMessage: function(sMessage, oOptions) {
            var oDefaults = {
                duration: 3000,
                width: Device.system.phone ? "15rem" : "20rem",
                my: Device.system.phone ? "center bottom" : "right bottom",
                at: Device.system.phone ? "center bottom" : "right bottom",
                of: window,
                offset: Device.system.phone ? "0 -50" : "-10 -10",
                animationDuration: 300
            };
            
            var oSettings = Object.assign({}, oDefaults, oOptions);
            MessageToast.show(sMessage, oSettings);
        },
        
        /**
         * @function setupLazyImageLoading
         * @description Sets up lazy loading for images to improve performance.
         * @param {sap.ui.core.Control} oContainer - Container control
         */
        setupLazyImageLoading: function(oContainer) {
            if (!window.IntersectionObserver) return;
            
            var imageObserver = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting) {
                        var img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove("lazy-image");
                        imageObserver.unobserve(img);
                    }
                });
            });
            
            // Find all lazy images
            var $lazyImages = oContainer.$().find("img.lazy-image");
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
        debounce: function(fn, delay) {
            var timeoutId;
            return function() {
                var context = this;
                var args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
                    fn.apply(context, args);
                }, delay);
            };
        }
    };
});