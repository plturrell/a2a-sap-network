sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "./SharedAccessibilityUtils",
    "./SharedSecurityUtils"
], (Log, MessageBox, MessageToast, AccessibilityUtils, SecurityUtils) => {
    "use strict";

    /**
     * Shared Error Handling Utilities for A2A Platform
     * Provides comprehensive error handling with accessibility support including:
     * - Accessible error messages and announcements
     * - Graceful error recovery
     * - User-friendly error presentation
     * - Security-conscious error logging
     * - Retry mechanisms with exponential backoff
     */
    return {

        /**
         * Handles errors with accessibility support and user-friendly presentation
         * @param {Error|Object} error - Error object or error information
         * @param {Object} options - Error handling options
         * @param {jQuery} $container - Container for error display (optional)
         */
        handleError(error, options = {}, $container) {
            const errorInfo = this._parseError(error);

            // Log error securely
            this._logError(errorInfo, options);

            // Display error to user based on severity
            switch (options.severity || errorInfo.severity) {
            case "critical":
                this._handleCriticalError(errorInfo, options, $container);
                break;
            case "warning":
                this._handleWarningError(errorInfo, options, $container);
                break;
            case "info":
                this._handleInfoError(errorInfo, options, $container);
                break;
            default:
                this._handleStandardError(errorInfo, options, $container);
            }

            // Track error for analytics (if enabled)
            if (options.trackError !== false) {
                this._trackError(errorInfo, options);
            }

            // Trigger recovery if specified
            if (options.recovery && typeof options.recovery === "function") {
                setTimeout(() => {
                    try {
                        options.recovery(errorInfo);
                    } catch (recoveryError) {
                        Log.error("Error during recovery", recoveryError);
                    }
                }, options.recoveryDelay || 1000);
            }
        },

        /**
         * Handles validation errors with field-specific feedback
         * @param {Array|Object} validationErrors - Validation error(s)
         * @param {Object} options - Validation error options
         */
        handleValidationErrors(validationErrors, options = {}) {
            const errors = Array.isArray(validationErrors) ? validationErrors : [validationErrors];
            const $form = options.form ? $(options.form) : null;

            // Clear previous validation errors
            if (options.clearPrevious !== false) {
                this._clearValidationErrors($form);
            }

            errors.forEach(error => {
                const fieldId = error.fieldId || error.field;
                const message = SecurityUtils.sanitizeErrorMessage(error.message);

                if (fieldId) {
                    // Field-specific error
                    AccessibilityUtils.handleAccessibleError({
                        message,
                        type: "validation"
                    }, $form, {
                        fieldId,
                        autoRemove: false // Keep validation errors until corrected
                    });
                } else {
                    // General validation error
                    AccessibilityUtils.handleAccessibleError({
                        message,
                        type: "validation"
                    }, $form);
                }
            });

            // Announce validation summary to screen readers
            const errorCount = errors.length;
            const summary = errorCount === 1 ?
                "1 validation error found" :
                `${errorCount} validation errors found`;

            AccessibilityUtils.announceToScreenReader(summary, "assertive");

            // Focus first error field
            if (errors.length > 0 && errors[0].fieldId) {
                AccessibilityUtils.manageFocus(`#${ errors[0].fieldId}`, {
                    delay: 200,
                    announce: "Please correct the error and try again"
                });
            }
        },

        /**
         * Handles network errors with retry capabilities
         * @param {Object} networkError - Network error information
         * @param {Object} options - Network error handling options
         */
        handleNetworkError(networkError, options = {}) {
            const errorInfo = this._parseNetworkError(networkError);

            // Check if we should retry
            if (this._shouldRetry(errorInfo, options)) {
                this._scheduleRetry(options.originalRequest, options);
            } else {
                // Show user-friendly network error message
                const message = this._getNetworkErrorMessage(errorInfo);

                MessageBox.error(message, {
                    title: "Connection Error",
                    actions: [
                        MessageBox.Action.RETRY,
                        MessageBox.Action.CANCEL
                    ],
                    onClose: (action) => {
                        if (action === MessageBox.Action.RETRY && options.onRetry) {
                            options.onRetry();
                        }
                    }
                });

                // Announce to screen readers
                AccessibilityUtils.announceToScreenReader(
                    `Connection error: ${message}`,
                    "assertive"
                );
            }
        },

        /**
         * Creates graceful fallbacks for failed operations
         * @param {Function} operation - Operation to execute
         * @param {Array} fallbacks - Array of fallback operations
         * @param {Object} options - Fallback options
         */
        executeWithFallback(operation, fallbacks = [], options = {}) {
            return new Promise((resolve, reject) => {
                const attemptOperation = (currentOp, remainingFallbacks) => {
                    try {
                        const result = currentOp();

                        // Handle promise results
                        if (result && typeof result.then === "function") {
                            result
                                .then(resolve)
                                .catch(error => {
                                    this._handleOperationFailure(error, remainingFallbacks,
                                        attemptOperation, reject, options);
                                });
                        } else {
                            resolve(result);
                        }
                    } catch (error) {
                        this._handleOperationFailure(error, remainingFallbacks, attemptOperation, reject, options);
                    }
                };

                attemptOperation(operation, fallbacks);
            });
        },

        /**
         * Shows loading state with accessible indicators
         * @param {jQuery|sap.ui.core.Control} target - Target to show loading on
         * @param {Object} options - Loading options
         */
        showAccessibleLoading(target, options = {}) {
            const $target = target.$ ? target.$() : $(target);
            const loadingId = `loading-${ Date.now()}`;

            // Create accessible loading indicator
            const $loading = $(`
                <div id="${loadingId}"
                     class="a2a-loading-indicator"
                     role="status"
                     aria-live="polite"
                     aria-label="${options.message || "Loading, please wait"}">
                    <span class="a2a-loading-spinner" aria-hidden="true">⟳</span>
                    <span class="a2a-loading-text">${options.message || "Loading..."}</span>
                </div>
            `);

            // Position and show loading indicator
            if (options.overlay) {
                $loading.addClass("a2a-loading-overlay");
                $target.css("position", "relative").append($loading);
            } else {
                $target.prepend($loading);
            }

            // Auto-hide after timeout (prevent stuck loading states)
            if (options.timeout !== 0) {
                setTimeout(() => {
                    this.hideAccessibleLoading(loadingId);
                }, options.timeout || 30000);
            }

            return loadingId;
        },

        /**
         * Hides loading state
         * @param {string} loadingId - Loading indicator ID to hide
         */
        hideAccessibleLoading(loadingId) {
            const $loading = $(`#${ loadingId}`);
            if ($loading.length) {
                $loading.fadeOut(200, () => $loading.remove());
            }
        },

        /**
         * Provides user feedback for successful operations
         * @param {string} message - Success message
         * @param {Object} options - Success feedback options
         */
        showSuccess(message, options = {}) {
            const sanitizedMessage = SecurityUtils.sanitizeErrorMessage(message);

            if (options.useToast !== false) {
                MessageToast.show(sanitizedMessage, {
                    duration: options.duration || 3000,
                    width: options.width || "15em",
                    onClose: options.onClose
                });
            }

            // Announce to screen readers
            AccessibilityUtils.announceToScreenReader(sanitizedMessage, "polite");

            // Log success for debugging
            Log.info("Success message displayed", { message: sanitizedMessage });
        },

        // Private helper methods
        _parseError(error) {
            if (typeof error === "string") {
                return {
                    message: error,
                    type: "generic",
                    severity: "error"
                };
            }

            return {
                message: error.message || "An unexpected error occurred",
                type: error.type || "generic",
                severity: error.severity || "error",
                code: error.code,
                stack: error.stack,
                timestamp: new Date().toISOString()
            };
        },

        _logError(errorInfo, options) {
            const logEntry = {
                message: errorInfo.message,
                type: errorInfo.type,
                severity: errorInfo.severity,
                code: errorInfo.code,
                context: options.context || "unknown",
                agent: options.agentId || "shared",
                timestamp: errorInfo.timestamp
            };

            // Use security utils for safe logging
            SecurityUtils.logSecureOperation(
                "ERROR_HANDLED",
                errorInfo.severity.toUpperCase(),
                logEntry,
                options.agentId
            );
        },

        _handleCriticalError(errorInfo, options, $container) {
            const message = `Critical Error: ${SecurityUtils.sanitizeErrorMessage(errorInfo.message)}`;

            MessageBox.error(message, {
                title: "Critical System Error",
                actions: [MessageBox.Action.OK],
                onClose: () => {
                    if (options.onCriticalError) {
                        options.onCriticalError(errorInfo);
                    }
                }
            });

            // Critical errors should be announced immediately
            AccessibilityUtils.announceToScreenReader(message, "assertive");
        },

        _handleStandardError(errorInfo, options, $container) {
            const message = SecurityUtils.sanitizeErrorMessage(errorInfo.message);

            if ($container && options.useInlineError !== false) {
                // Show inline accessible error
                AccessibilityUtils.handleAccessibleError(errorInfo, $container, options);
            } else {
                // Show dialog error
                MessageBox.error(message, {
                    title: options.title || "Error",
                    actions: [MessageBox.Action.OK],
                    onClose: options.onClose
                });

                AccessibilityUtils.announceToScreenReader(message, "assertive");
            }
        },

        _handleWarningError(errorInfo, options, $container) {
            const message = SecurityUtils.sanitizeErrorMessage(errorInfo.message);

            MessageBox.warning(message, {
                title: options.title || "Warning",
                actions: [MessageBox.Action.OK],
                onClose: options.onClose
            });

            AccessibilityUtils.announceToScreenReader(`Warning: ${message}`, "polite");
        },

        _handleInfoError(errorInfo, options, $container) {
            const message = SecurityUtils.sanitizeErrorMessage(errorInfo.message);

            if (options.useToast !== false) {
                MessageToast.show(message, {
                    duration: options.duration || 4000
                });
            }

            AccessibilityUtils.announceToScreenReader(message, "polite");
        },

        _parseNetworkError(error) {
            const status = error.status || 0;
            const statusText = error.statusText || "";

            return {
                status,
                statusText,
                type: this._getNetworkErrorType(status),
                isTimeout: error.timeout || false,
                message: error.message || this._getDefaultNetworkMessage(status)
            };
        },

        _getNetworkErrorType(status) {
            if (status === 0) {return "connection";}
            if (status >= 400 && status < 500) {return "client";}
            if (status >= 500) {return "server";}
            return "unknown";
        },

        _getDefaultNetworkMessage(status) {
            switch (status) {
            case 0: return "Unable to connect to server";
            case 400: return "Invalid request";
            case 401: return "Authentication required";
            case 403: return "Access denied";
            case 404: return "Resource not found";
            case 408: return "Request timeout";
            case 429: return "Too many requests";
            case 500: return "Server error";
            case 502: return "Bad gateway";
            case 503: return "Service unavailable";
            case 504: return "Gateway timeout";
            default: return "Network error occurred";
            }
        },

        _getNetworkErrorMessage(errorInfo) {
            const baseMessage = errorInfo.message;

            switch (errorInfo.type) {
            case "connection":
                return `${baseMessage}. Please check your internet connection and try again.`;
            case "server":
                return `${baseMessage}. The server is experiencing issues. Please try again later.`;
            case "client":
                return `${baseMessage}. Please check your input and try again.`;
            default:
                return `${baseMessage}. Please try again.`;
            }
        },

        _shouldRetry(errorInfo, options) {
            const retryableStatuses = [0, 408, 429, 500, 502, 503, 504];
            const currentRetryCount = options._retryCount || 0;
            const maxRetries = options.maxRetries || 3;

            return retryableStatuses.includes(errorInfo.status) &&
                   currentRetryCount < maxRetries &&
                   options.enableRetry !== false;
        },

        _scheduleRetry(originalRequest, options) {
            const retryCount = (options._retryCount || 0) + 1;
            const delay = this._calculateRetryDelay(retryCount, options);

            setTimeout(() => {
                const retryOptions = {
                    ...options,
                    _retryCount: retryCount
                };

                AccessibilityUtils.announceToScreenReader(
                    `Retrying request, attempt ${retryCount}`,
                    "polite"
                );

                originalRequest(retryOptions);
            }, delay);
        },

        _calculateRetryDelay(retryCount, options) {
            const baseDelay = options.baseDelay || 1000;
            const maxDelay = options.maxDelay || 30000;

            // Exponential backoff with jitter
            const exponentialDelay = baseDelay * Math.pow(2, retryCount - 1);
            const jitter = Math.random() * 0.1 * exponentialDelay;

            return Math.min(exponentialDelay + jitter, maxDelay);
        },

        _handleOperationFailure(error, remainingFallbacks, attemptOperation, reject, options) {
            if (remainingFallbacks.length > 0) {
                const nextFallback = remainingFallbacks[0];
                const newFallbacks = remainingFallbacks.slice(1);

                // Log fallback attempt
                Log.info("Attempting fallback operation", {
                    error: error.message,
                    remainingFallbacks: newFallbacks.length
                });

                // Announce fallback to user if requested
                if (options.announceFallbacks) {
                    AccessibilityUtils.announceToScreenReader(
                        "Primary operation failed, trying alternative",
                        "polite"
                    );
                }

                setTimeout(() => {
                    attemptOperation(nextFallback, newFallbacks);
                }, options.fallbackDelay || 500);
            } else {
                reject(error);
            }
        },

        _clearValidationErrors($form) {
            if ($form) {
                $form.find(".a2a-error-message").remove();
                $form.find(".a2a-field-error")
                    .removeClass("a2a-field-error")
                    .removeAttr("aria-describedby");
            }
        },

        _trackError(errorInfo, options) {
            // In a real implementation, this would send error data to analytics service
            Log.info("Error tracked for analytics", {
                type: errorInfo.type,
                severity: errorInfo.severity,
                context: options.context
            });
        }
    };
});