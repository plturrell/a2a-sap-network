sap.ui.define([
    "sap/base/Log"
], (Log) => {
    "use strict";

    /**
     * @class AuthHandler
     * @description Authentication and authorization handler for Agent 9
     * Provides role-based access control and reasoning-specific permissions
     */
    const AuthHandler = {

        /**
         * @function hasPermission
         * @description Checks if current user has specified permission
         * @param {string} permission - Permission to check
         * @returns {boolean} True if user has permission
         */
        hasPermission(permission) {
            try {
                const user = this._getCurrentUser();
                if (!user) {
                    Log.warning("No user context available for permission check", "", "Agent9.AuthHandler");
                    return false;
                }

                // Admin users have all permissions
                if (this.hasRole("Admin")) {
                    return true;
                }

                // Define permission mappings for reasoning operations
                const permissionMap = {
                    "VIEW_REASONING_TASKS": ["authenticated-user", "ReasoningUser", "Admin"],
                    "CREATE_REASONING_TASK": ["ReasoningManager", "Admin"],
                    "EXECUTE_REASONING": ["ReasoningUser", "ReasoningManager", "Admin"],
                    "ANALYZE_CONTRADICTIONS": ["ReasoningAnalyst", "ReasoningManager", "Admin"],
                    "RESOLVE_CONTRADICTIONS": ["ReasoningManager", "Admin"],
                    "UPDATE_KNOWLEDGE": ["KnowledgeManager", "ReasoningManager", "Admin"],
                    "MAKE_DECISIONS": ["DecisionMaker", "ReasoningManager", "Admin"],
                    "VIEW_INFERENCE_RESULTS": ["ReasoningUser", "ReasoningAnalyst", "Admin"],
                    "MANAGE_REASONING_ENGINES": ["ReasoningManager", "Admin"],
                    "ACCESS_KNOWLEDGE_BASE": ["KnowledgeUser", "ReasoningUser", "Admin"],
                    "EXPORT_REASONING_DATA": ["DataExporter", "ReasoningManager", "Admin"]
                };

                const requiredRoles = permissionMap[permission];
                if (!requiredRoles) {
                    Log.warning(`Unknown permission requested: ${ permission}`, "", "Agent9.AuthHandler");
                    return false;
                }

                return requiredRoles.some(role => this.hasRole(role));

            } catch (error) {
                Log.error(`Error checking permission: ${ error.message}`, "", "Agent9.AuthHandler");
                return false;
            }
        },

        /**
         * @function hasRole
         * @description Checks if current user has specified role
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         */
        hasRole(role) {
            try {
                const user = this._getCurrentUser();
                if (!user || !user.roles) {
                    return false;
                }

                const userRoles = user.sapRoles || user.roles || [];
                return userRoles.includes(role) || userRoles.includes("authenticated-user");

            } catch (error) {
                Log.error(`Error checking role: ${ error.message}`, "", "Agent9.AuthHandler");
                return false;
            }
        },

        /**
         * @function validateReasoningAccess
         * @description Validates user access to reasoning operations with specific constraints
         * @param {string} operation - Reasoning operation type
         * @param {Object} context - Operation context (task details, etc.)
         * @returns {Object} Validation result with isValid and reason
         */
        validateReasoningAccess(operation, context) {
            const result = { isValid: false, reason: "Access denied" };

            try {
                const user = this._getCurrentUser();
                if (!user) {
                    result.reason = "User not authenticated";
                    return result;
                }

                // Check operation-specific constraints
                switch (operation) {
                case "CREATE_REASONING_TASK":
                    if (!this.hasPermission("CREATE_REASONING_TASK")) {
                        result.reason = "Insufficient permissions to create reasoning tasks";
                        return result;
                    }
                    break;

                case "ANALYZE_CONTRADICTIONS":
                    if (!this.hasPermission("ANALYZE_CONTRADICTIONS")) {
                        result.reason = "Insufficient permissions to analyze contradictions";
                        return result;
                    }
                    break;

                case "RESOLVE_CONTRADICTIONS":
                    if (!this.hasPermission("RESOLVE_CONTRADICTIONS")) {
                        result.reason = "Insufficient permissions to resolve contradictions";
                        return result;
                    }
                    break;

                case "UPDATE_KNOWLEDGE":
                    if (!this.hasPermission("UPDATE_KNOWLEDGE")) {
                        result.reason = "Insufficient permissions to update knowledge base";
                        return result;
                    }
                    // Additional validation for knowledge updates
                    if (context && context.scope === "GLOBAL" && !this.hasRole("Admin")) {
                        result.reason = "Global knowledge updates require admin privileges";
                        return result;
                    }
                    break;

                case "MAKE_DECISIONS":
                    if (!this.hasPermission("MAKE_DECISIONS")) {
                        result.reason = "Insufficient permissions to make decisions";
                        return result;
                    }
                    break;

                default:
                    result.reason = "Unknown reasoning operation";
                    return result;
                }

                result.isValid = true;
                result.reason = "Access granted";
                return result;

            } catch (error) {
                Log.error(`Error validating reasoning access: ${ error.message}`, "", "Agent9.AuthHandler");
                result.reason = "Authentication system error";
                return result;
            }
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user from SAP UI5 framework
         * @returns {object|null} User object or null
         * @private
         */
        _getCurrentUser() {
            try {
                // Try to get user from SAP UI5 framework
                if (sap && sap.ushell && sap.ushell.Container) {
                    const userInfoService = sap.ushell.Container.getService("UserInfo");
                    if (userInfoService) {
                        return {
                            id: userInfoService.getId(),
                            email: userInfoService.getEmail(),
                            fullName: userInfoService.getFullName(),
                            roles: userInfoService.getRoles ? userInfoService.getRoles() : ["authenticated-user"],
                            sapRoles: userInfoService.getSapRoles ? userInfoService.getSapRoles() : ["authenticated-user"]
                        };
                    }
                }

                // Fallback: try to get from session storage or other sources
                const sessionUser = sessionStorage.getItem("currentUser");
                if (sessionUser) {
                    return JSON.parse(sessionUser);
                }

                // Development fallback
                if (window.location.hostname === "localhost") {
                    return {
                        id: "dev-user",
                        email: "developer@a2a.local",
                        fullName: "Development User",
                        roles: ["authenticated-user", "ReasoningUser", "ReasoningManager", "Admin"],
                        sapRoles: ["authenticated-user", "ReasoningUser", "ReasoningManager", "Admin"],
                        isDevelopment: true
                    };
                }

                return null;

            } catch (error) {
                Log.error(`Error getting current user: ${ error.message}`, "", "Agent9.AuthHandler");
                return null;
            }
        },

        /**
         * @function logSecurityEvent
         * @description Logs security-related events for audit trail
         * @param {string} event - Event type
         * @param {object} details - Event details
         */
        logSecurityEvent(event, details) {
            try {
                const user = this._getCurrentUser();
                const logEntry = {
                    timestamp: new Date().toISOString(),
                    event,
                    userId: user ? user.id : "anonymous",
                    userEmail: user ? user.email : "unknown",
                    details: details || {},
                    component: "Agent9.AuthHandler",
                    reasoningContext: details.reasoningOperation || "unknown"
                };

                // Log to browser console in development
                if (window.location.hostname === "localhost") {
                    Log.info(`Agent 9 Security Event: ${ JSON.stringify(logEntry)}`, "", "Agent9.AuthHandler");
                }

                // In production, this would send to security logging service
                // Example: SecurityLoggingService.log(logEntry);

            } catch (error) {
                Log.error(`Failed to log security event: ${ error.message}`, "", "Agent9.AuthHandler");
            }
        }
    };

    return AuthHandler;
});