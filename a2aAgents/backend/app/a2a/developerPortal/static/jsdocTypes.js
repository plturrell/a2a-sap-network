"use strict";
/* global  */

/**
 * @fileoverview Type definitions for A2A Developer Portal
 * @module a2a/types
 * @description Common type definitions used throughout the A2A platform
 */

/**
 * @typedef {Object} AgentConfiguration
 * @property {string} agentId - Unique identifier for the agent
 * @property {string} name - Human-readable name of the agent
 * @property {string} description - Detailed description of agent capabilities
 * @property {string} version - Semantic version of the agent
 * @property {string[]} skills - Array of skill identifiers the agent possesses
 * @property {Object} metadata - Additional metadata for the agent
 * @property {string} metadata.created - ISO 8601 creation timestamp
 * @property {string} metadata.updated - ISO 8601 last update timestamp
 * @property {string} metadata.author - Agent creator identifier
 */

/**
 * @typedef {Object} DeploymentResult
 * @property {boolean} success - Whether deployment was successful
 * @property {string} deploymentId - Unique deployment identifier
 * @property {string} environment - Target environment (dev|staging|prod)
 * @property {string} status - Current deployment status
 * @property {string} url - URL where the agent can be accessed
 * @property {Object} [error] - Error details if deployment failed
 * @property {string} error.code - Error code
 * @property {string} error.message - Human-readable error message
 */

/**
 * @typedef {Object} ProjectData
 * @property {string} projectId - Unique project identifier
 * @property {string} name - Project name
 * @property {string} description - Project description
 * @property {string} status - Current project status
 * @property {AgentConfiguration[]} agents - Array of agents in the project
 * @property {WorkflowDefinition[]} workflows - Array of workflows
 * @property {Object} settings - Project-specific settings
 */

/**
 * @typedef {Object} WorkflowDefinition
 * @property {string} workflowId - Unique workflow identifier
 * @property {string} name - Workflow name
 * @property {string} bpmnXml - BPMN 2.0 XML definition
 * @property {Object[]} tasks - Array of workflow tasks
 * @property {string} tasks[].taskId - Task identifier
 * @property {string} tasks[].agentId - Agent responsible for the task
 * @property {Object} tasks[].parameters - Task parameters
 */

/**
 * @typedef {Object} NotificationData
 * @property {string} notificationId - Unique notification identifier
 * @property {('info'|'success'|'warning'|'error')} type - Notification type
 * @property {string} title - Notification title
 * @property {string} message - Notification message
 * @property {string} timestamp - ISO 8601 timestamp
 * @property {boolean} read - Whether notification has been read
 * @property {Object} [action] - Optional action for the notification
 * @property {string} action.label - Action button label
 * @property {string} action.url - Action URL or route
 */

/**
 * @typedef {Object} SecurityContext
 * @property {string} userId - Current user identifier
 * @property {string[]} roles - User roles
 * @property {string[]} permissions - User permissions
 * @property {string} token - Authentication token
 * @property {number} expiresAt - Token expiration timestamp
 */

/**
 * @typedef {Object} APIResponse
 * @template T
 * @property {boolean} success - Whether the API call was successful
 * @property {T} [data] - Response data if successful
 * @property {APIError} [error] - Error details if unsuccessful
 * @property {Object} metadata - Response metadata
 * @property {string} metadata.requestId - Request tracking ID
 * @property {number} metadata.timestamp - Response timestamp
 */

/**
 * @typedef {Object} APIError
 * @property {string} code - Error code
 * @property {string} message - Human-readable error message
 * @property {string} [details] - Additional error details
 * @property {string} [field] - Field that caused the error (for validation errors)
 */

/**
 * @typedef {Object} PaginationParams
 * @property {number} page - Current page number (1-based)
 * @property {number} pageSize - Number of items per page
 * @property {string} [sortBy] - Field to sort by
 * @property {('asc'|'desc')} [sortOrder='asc'] - Sort order
 */

/**
 * @typedef {Object} PaginatedResponse
 * @template T
 * @property {T[]} items - Array of items for current page
 * @property {number} total - Total number of items
 * @property {number} page - Current page number
 * @property {number} pageSize - Items per page
 * @property {number} totalPages - Total number of pages
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} valid - Whether validation passed
 * @property {ValidationError[]} errors - Array of validation errors
 */

/**
 * @typedef {Object} ValidationError
 * @property {string} field - Field that failed validation
 * @property {string} message - Validation error message
 * @property {string} [code] - Validation error code
 */

/**
 * @typedef {Object} EventPayload
 * @template T
 * @property {string} eventType - Type of event
 * @property {T} data - Event data
 * @property {string} timestamp - ISO 8601 timestamp
 * @property {string} [correlationId] - Correlation ID for tracking
 */

/**
 * @typedef {Function} EventHandler
 * @param {sap.ui.base.Event} oEvent - SAP UI5 event object
 * @returns {void}
 */

/**
 * @typedef {Function} AsyncEventHandler
 * @param {sap.ui.base.Event} oEvent - SAP UI5 event object
 * @returns {Promise<void>}
 */

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {};
}