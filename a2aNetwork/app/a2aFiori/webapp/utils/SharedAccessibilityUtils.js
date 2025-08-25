sap.ui.define([
    "sap/base/Log",
    "sap/ui/Device"
], (Log, Device) => {
    "use strict";

    /**
     * Shared Accessibility Utilities for A2A Platform
     * Provides comprehensive accessibility features for all agents including:
     * - ARIA label and description management
     * - Keyboard navigation support
     * - Screen reader optimization
     * - Focus management
     * - High contrast and color blind support
     * - Responsive design utilities
     */
    return {

        /**
         * Enhances dialog accessibility with proper ARIA attributes and keyboard handling
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @param {Object} options - Accessibility options
         */
        enhanceDialogAccessibility(oDialog, options = {}) {
            if (!oDialog) {
                Log.error("Dialog is required for accessibility enhancement");
                return;
            }

            // Set basic ARIA properties
            oDialog.addEventDelegate({
                onAfterRendering: function() {
                    const $dialog = oDialog.$();

                    // Set role and ARIA properties
                    $dialog.attr({
                        "role": options.role || "dialog",
                        "aria-modal": "true",
                        "aria-labelledby": options.titleId || `${oDialog.getId() }-title`,
                        "aria-describedby": options.descriptionId || `${oDialog.getId() }-content`
                    });

                    // Enhance form controls
                    this._enhanceFormControls($dialog);

                    // Setup keyboard navigation
                    this._setupKeyboardNavigation($dialog, oDialog);

                    // Setup focus management
                    this._setupFocusManagement($dialog, options);

                    // Add high contrast support
                    this._addHighContrastSupport($dialog);

                }.bind(this)
            });
        },

        /**
         * Enhances table accessibility with proper ARIA attributes and keyboard navigation
         * @param {sap.m.Table|sap.ui.table.Table} oTable - Table to enhance
         * @param {Object} options - Accessibility options
         */
        enhanceTableAccessibility(oTable, options = {}) {
            if (!oTable) {
                Log.error("Table is required for accessibility enhancement");
                return;
            }

            oTable.addEventDelegate({
                onAfterRendering: function() {
                    const $table = oTable.$();
                    const _tableType = oTable.getMetadata().getName();

                    // Set table ARIA properties
                    $table.find("table").first().attr({
                        "role": "grid",
                        "aria-label": options.ariaLabel || `Data table with ${ oTable.getItems().length } rows`,
                        "aria-rowcount": oTable.getItems().length,
                        "aria-colcount": this._getColumnCount(oTable)
                    });

                    // Enhance headers
                    this._enhanceTableHeaders($table, oTable, options);

                    // Enhance rows and cells
                    this._enhanceTableRows($table, oTable, options);

                    // Add sort and filter announcements
                    this._addTableAnnouncements($table, oTable);

                }.bind(this)
            });
        },

        /**
         * Enhances form accessibility with proper labels and validation messages
         * @param {sap.ui.core.Control} oForm - Form control to enhance
         * @param {Object} options - Accessibility options
         */
        enhanceFormAccessibility(oForm, options = {}) {
            if (!oForm) {
                Log.error("Form is required for accessibility enhancement");
                return;
            }

            oForm.addEventDelegate({
                onAfterRendering: function() {
                    const $form = oForm.$();

                    // Enhance input fields
                    this._enhanceFormInputs($form, options);

                    // Add validation message support
                    this._addValidationMessageSupport($form, options);

                    // Setup error announcement
                    this._setupErrorAnnouncement($form, options);

                }.bind(this)
            });
        },

        /**
         * Creates accessible announcements for screen readers
         * @param {string} message - Message to announce
         * @param {string} priority - Priority level (polite, assertive, off)
         */
        announceToScreenReader(message, priority = "polite") {
            const announcement = document.createElement("div");
            announcement.setAttribute("aria-live", priority);
            announcement.setAttribute("aria-atomic", "true");
            announcement.className = "a2a-sr-only";
            announcement.textContent = message;

            // Add to DOM temporarily
            document.body.appendChild(announcement);

            // Remove after announcement
            setTimeout(() => {
                document.body.removeChild(announcement);
            }, 1000);
        },

        /**
         * Sets up skip links for keyboard navigation
         * @param {jQuery} $container - Container to add skip links to
         * @param {Array} skipTargets - Array of skip target objects
         */
        addSkipLinks($container, skipTargets) {
            if (!skipTargets || skipTargets.length === 0) {return;}

            const $skipContainer = $("<div class=\"a2a-skip-links\" role=\"navigation\" aria-label=\"Skip links\"></div>");

            skipTargets.forEach(target => {
                const $skipLink = $(`<a href="#${target.id}" class="a2a-skip-link">${target.label}</a>`);
                $skipLink.on("click", (e) => {
                    e.preventDefault();
                    const $target = $(`#${ target.id}`);
                    if ($target.length) {
                        $target.focus();
                        this.announceToScreenReader(`Skipped to ${target.label}`);
                    }
                });
                $skipContainer.append($skipLink);
            });

            $container.prepend($skipContainer);
        },

        /**
         * Adds landmark roles to page sections
         * @param {jQuery} $container - Container to enhance
         * @param {Object} landmarks - Object mapping selectors to landmark roles
         */
        addLandmarkRoles($container, landmarks = {}) {
            const defaultLandmarks = {
                ".sapUiLocalBusyIndicator": "status",
                ".sapMMessageToast": "status",
                ".sapMDialog": "dialog",
                ".sapMPage-head": "banner",
                ".sapMPage-footer": "contentinfo",
                ".sapMPage-cont": "main",
                ".sapMPanel": "region"
            };

            const allLandmarks = Object.assign({}, defaultLandmarks, landmarks);

            Object.keys(allLandmarks).forEach(selector => {
                $container.find(selector).attr("role", allLandmarks[selector]);
            });
        },

        /**
         * Manages focus for complex interactions
         * @param {jQuery|string} element - Element to focus or selector
         * @param {Object} options - Focus options
         */
        manageFocus(element, options = {}) {
            const $element = typeof element === "string" ? $(element) : element;

            if (!$element.length) {
                Log.warning("Focus target not found");
                return;
            }

            // Store previous focus if needed
            if (options.storePrevious) {
                this._previousFocus = document.activeElement;
            }

            // Set focus with optional delay
            const focusElement = () => {
                $element.focus();

                // Announce focus change if needed
                if (options.announce) {
                    this.announceToScreenReader(options.announce);
                }

                // Scroll into view if needed
                if (options.scrollIntoView) {
                    $element[0].scrollIntoView({
                        behavior: "smooth",
                        block: options.scrollBlock || "center"
                    });
                }
            };

            if (options.delay) {
                setTimeout(focusElement, options.delay);
            } else {
                focusElement();
            }
        },

        /**
         * Restores previously stored focus
         */
        restorePreviousFocus() {
            if (this._previousFocus && this._previousFocus.focus) {
                this._previousFocus.focus();
                this._previousFocus = null;
            }
        },

        /**
         * Adds color blind friendly indicators
         * @param {jQuery} $container - Container to enhance
         * @param {Object} options - Color blind support options
         */
        addColorBlindSupport($container, options = {}) {
            // Add patterns/shapes in addition to colors
            $container.find(".sapUiIcon").each(function() {
                const $icon = $(this);
                const iconType = $icon.attr("data-sap-ui-icon");

                // Add descriptive title for screen readers
                if (!$icon.attr("title")) {
                    $icon.attr("title", this._getIconDescription(iconType));
                }
            }.bind(this));

            // Add text indicators for status colors
            $container.find("[class*=\"sapUiMessageType\"]").each(function() {
                const $element = $(this);
                const messageType = this._extractMessageType($element[0].className);
                const indicator = this._getStatusIndicator(messageType);

                if (indicator && !$element.find(".a2a-status-indicator").length) {
                    $element.prepend(`<span class="a2a-status-indicator" aria-hidden="true">${indicator}</span> `);
                }
            });
        },

        /**
         * Optimizes controls for mobile accessibility
         * @param {jQuery} $container - Container to optimize
         */
        optimizeForMobile($container) {
            if (!Device.system.phone && !Device.system.tablet) {return;}

            // Increase touch targets
            $container.find("button, .sapMBtn, .sapMLink, input[type=\"checkbox\"], input[type=\"radio\"]")
                .addClass("a2a-large-touch-target");

            // Add mobile-specific ARIA labels
            $container.find("input[type=\"text\"], input[type=\"email\"], input[type=\"number\"]")
                .each(function() {
                    const $input = $(this);
                    if (!$input.attr("aria-label") && !$input.attr("aria-labelledby")) {
                        const placeholder = $input.attr("placeholder");
                        if (placeholder) {
                            $input.attr("aria-label", placeholder);
                        }
                    }
                });

            // Enhance swipe gestures for screen readers
            this._addSwipeGestureSupport($container);
        },

        /**
         * Adds comprehensive error handling with accessibility features
         * @param {Object} errorInfo - Error information
         * @param {jQuery} $container - Container for error display
         * @param {Object} options - Error handling options
         */
        handleAccessibleError(errorInfo, $container, options = {}) {
            const errorId = `error-${ Date.now()}`;
            const isFieldError = options.fieldId || options.field;

            // Create accessible error message
            const $errorMessage = $(`
                <div id="${errorId}"
                     class="a2a-error-message"
                     role="alert"
                     aria-live="assertive"
                     tabindex="0">
                    <span class="a2a-error-icon" aria-hidden="true">⚠️</span>
                    <span class="a2a-error-text">${this._sanitizeErrorMessage(errorInfo.message)}</span>
                </div>
            `);

            // Add to container
            if (isFieldError) {
                // Associate with specific field
                const $field = options.field ? $(options.field) : $(`#${ options.fieldId}`);
                $field.attr("aria-describedby", errorId);
                $field.addClass("a2a-field-error");
                $field.after($errorMessage);

                // Focus the field for correction
                setTimeout(() => $field.focus(), 100);
            } else {
                // General error
                $container.prepend($errorMessage);
                setTimeout(() => $errorMessage.focus(), 100);
            }

            // Auto-remove after specified time
            if (options.autoRemove !== false) {
                setTimeout(() => {
                    $errorMessage.fadeOut(() => {
                        $errorMessage.remove();
                        if (isFieldError) {
                            const $field = options.field ? $(options.field) : $(`#${ options.fieldId}`);
                            $field.removeClass("a2a-field-error").removeAttr("aria-describedby");
                        }
                    });
                }, options.timeout || 8000);
            }

            // Log for debugging
            Log.error("Accessible error displayed", errorInfo);
        },

        // Private helper methods
        _enhanceFormControls($container) {
            // Add ARIA labels to form controls without labels
            $container.find("input, textarea, select").each(function() {
                const $control = $(this);

                if (!$control.attr("aria-label") && !$control.attr("aria-labelledby")) {
                    // Try to find associated label
                    const id = $control.attr("id");
                    let label = "";

                    if (id) {
                        const $label = $container.find(`label[for="${id}"]`);
                        if ($label.length) {
                            label = $label.text().trim();
                        }
                    }

                    // Fallback to placeholder or nearby text
                    if (!label) {
                        label = $control.attr("placeholder") ||
                               $control.prev("label").text().trim() ||
                               $control.closest(".sapMInputBase").find("label").text().trim();
                    }

                    if (label) {
                        $control.attr("aria-label", label);
                    }
                }
            });
        },

        _setupKeyboardNavigation($dialog, oDialog) {
            $dialog.on("keydown", (e) => {
                // Escape key handling
                if (e.key === "Escape" && oDialog.getEscapeHandler()) {
                    e.preventDefault();
                    oDialog.close();
                    return;
                }

                // Tab trapping within dialog
                if (e.key === "Tab") {
                    this._handleTabTrapping(e, $dialog);
                }

                // Enter key on buttons
                if (e.key === "Enter" && $(e.target).is("button, .sapMBtn")) {
                    e.preventDefault();
                    $(e.target).click();
                }
            });
        },

        _setupFocusManagement($dialog, options) {
            // Focus first focusable element when dialog opens
            setTimeout(() => {
                const $focusable = $dialog.find("input:visible:enabled, button:visible:enabled, select:visible:enabled, textarea:visible:enabled, [tabindex]:visible").first();
                if ($focusable.length) {
                    $focusable.focus();
                } else {
                    // Focus the dialog itself if no focusable elements
                    $dialog.attr("tabindex", "-1").focus();
                }
            }, options.focusDelay || 100);
        },

        _addHighContrastSupport($container) {
            // Add high contrast indicators
            if (window.matchMedia && window.matchMedia("(prefers-contrast: high)").matches) {
                $container.addClass("a2a-high-contrast");

                // Add extra visual indicators
                $container.find(".sapUiIcon").addClass("a2a-high-contrast-icon");
                $container.find("button, .sapMBtn").addClass("a2a-high-contrast-button");
            }
        },

        _handleTabTrapping(e, $dialog) {
            const $focusable = $dialog.find("input:visible:enabled, button:visible:enabled, select:visible:enabled, textarea:visible:enabled, [tabindex]:visible");
            const $first = $focusable.first();
            const $last = $focusable.last();

            if (e.shiftKey && $(e.target).is($first)) {
                e.preventDefault();
                $last.focus();
            } else if (!e.shiftKey && $(e.target).is($last)) {
                e.preventDefault();
                $first.focus();
            }
        },

        _enhanceTableHeaders($table, oTable, options) {
            $table.find("th").each(function(index) {
                const $header = $(this);
                $header.attr({
                    "scope": "col",
                    "role": "columnheader",
                    "aria-sort": "none"
                });

                // Add sort indicators
                const column = oTable.getColumns()[index];
                if (column && column.getSorted && column.getSorted()) {
                    $header.attr("aria-sort", column.getSortOrder() === "Ascending" ? "ascending" : "descending");
                }
            });
        },

        _enhanceTableRows($table, oTable, options) {
            $table.find("tbody tr").each(function(rowIndex) {
                const $row = $(this);
                $row.attr({
                    "role": "row",
                    "aria-rowindex": rowIndex + 1
                });

                $row.find("td").each(function(colIndex) {
                    const $cell = $(this);
                    $cell.attr({
                        "role": "gridcell",
                        "aria-colindex": colIndex + 1
                    });
                });
            });
        },

        _addTableAnnouncements($table, oTable) {
            // Add announcements for sorting
            if (oTable.attachSort) {
                oTable.attachSort((e) => {
                    const column = e.getParameter("column");
                    const sortOrder = e.getParameter("sortOrder");
                    this.announceToScreenReader(`Table sorted by ${column.getLabel()} ${sortOrder.toLowerCase()}`);
                });
            }

            // Add announcements for filtering
            if (oTable.attachFilter) {
                oTable.attachFilter((e) => {
                    const count = oTable.getItems().length;
                    this.announceToScreenReader(`Table filtered, ${count} rows visible`);
                });
            }
        },

        _getColumnCount(oTable) {
            const columns = oTable.getColumns ? oTable.getColumns() : [];
            return columns.length;
        },

        _enhanceFormInputs($form, options) {
            $form.find("input, textarea, select").each(function() {
                const $input = $(this);

                // Add required indicator
                if ($input.prop("required") || $input.hasClass("required")) {
                    $input.attr("aria-required", "true");

                    // Add visual indicator if not present
                    if (!$input.siblings(".a2a-required-indicator").length) {
                        $input.after("<span class=\"a2a-required-indicator\" aria-hidden=\"true\">*</span>");
                    }
                }

                // Add invalid state support
                if ($input.hasClass("error") || $input.attr("data-error")) {
                    $input.attr("aria-invalid", "true");
                }
            });
        },

        _addValidationMessageSupport($form, options) {
            // This would integrate with the form validation system
            // to announce validation messages to screen readers
            $form.on("validationError", (e) => {
                const field = e.field;
                const message = e.message;
                this.announceToScreenReader(`Validation error in ${field}: ${message}`, "assertive");
            });
        },

        _setupErrorAnnouncement($form, options) {
            // Monitor for error states and announce them
            const observer = new MutationObserver((mutations) => {
                mutations.forEach(function(mutation) {
                    if (mutation.type === "attributes" && mutation.attributeName === "class") {
                        const $target = $(mutation.target);
                        if ($target.hasClass("error") && !$target.hasClass("announced")) {
                            const label = $target.attr("aria-label") || $target.attr("placeholder") || "Field";
                            this.announceToScreenReader(`${label} has an error`);
                            $target.addClass("announced");
                        }
                    }
                });
            });

            observer.observe($form[0], { attributes: true, subtree: true });
        },

        _addSwipeGestureSupport($container) {
            // Add instructions for screen reader users about swipe gestures
            if (Device.system.phone || Device.system.tablet) {
                $container.find("[data-swipe]").each(function() {
                    const $element = $(this);
                    const swipeAction = $element.attr("data-swipe");
                    $element.attr("aria-label", `${$element.attr("aria-label") || ""} Swipe to ${swipeAction}`.trim());
                });
            }
        },

        _getIconDescription(iconType) {
            const iconDescriptions = {
                "sap-icon://accept": "Success",
                "sap-icon://decline": "Error",
                "sap-icon://warning": "Warning",
                "sap-icon://information": "Information",
                "sap-icon://edit": "Edit",
                "sap-icon://delete": "Delete",
                "sap-icon://add": "Add",
                "sap-icon://search": "Search"
            };

            return iconDescriptions[iconType] || "Icon";
        },

        _extractMessageType(className) {
            if (className.includes("sapUiMessageTypeError")) {return "Error";}
            if (className.includes("sapUiMessageTypeWarning")) {return "Warning";}
            if (className.includes("sapUiMessageTypeSuccess")) {return "Success";}
            if (className.includes("sapUiMessageTypeInformation")) {return "Information";}
            return "Message";
        },

        _getStatusIndicator(messageType) {
            const indicators = {
                "Error": "✗",
                "Warning": "⚠",
                "Success": "✓",
                "Information": "ℹ"
            };

            return indicators[messageType] || "";
        },

        _sanitizeErrorMessage(message) {
            if (!message || typeof message !== "string") {
                return "An error occurred. Please try again.";
            }

            // Remove potential HTML/script content
            return message.replace(/<[^>]*>/g, "").substring(0, 200);
        }
    };
});