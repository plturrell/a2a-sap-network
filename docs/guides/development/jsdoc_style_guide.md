# A2A Platform JSDoc Style Guide

## Overview

This guide establishes JSDoc documentation standards for the A2A Platform JavaScript and TypeScript codebase. Following these standards ensures consistent, comprehensive documentation that improves code maintainability, developer onboarding, and IDE support.

## Table of Contents

1. [General Principles](#general-principles)
2. [File Headers](#file-headers)
3. [Class Documentation](#class-documentation)
4. [Method Documentation](#method-documentation)
5. [Property Documentation](#property-documentation)
6. [Type Definitions](#type-definitions)
7. [Event Documentation](#event-documentation)
8. [Examples](#examples)
9. [SAP UI5 Specific Patterns](#sap-ui5-specific-patterns)
10. [Tools and Automation](#tools-and-automation)

## General Principles

- **Every file must have a file header** with `@fileoverview`, `@module`, and `@requires`
- **Every class/controller must have class-level documentation** with description and metadata
- **Every public method must have JSDoc** with parameters, return types, and description
- **Use TypeScript-style type annotations** in JSDoc for better IDE support
- **Include examples for complex methods** using `@example`
- **Document all events** with `@fires` and `@listens` annotations

## File Headers

Every JavaScript file must start with a file header:

```javascript
/**
 * @fileoverview Agent Builder Controller for A2A Developer Portal
 * @module a2a/portal/controller/AgentBuilder
 * @requires sap.ui.core.mvc.Controller
 * @requires sap.ui.model.json.JSONModel
 * @requires sap.m.MessageToast
 * @author A2A Development Team
 * @since 1.0.0
 */
```

## Class Documentation

### SAP UI5 Controllers

```javascript
/**
 * Agent Builder Controller
 * @class
 * @alias a2a.portal.controller.AgentBuilder
 * @extends sap.ui.core.mvc.Controller
 * @description Manages the agent creation and configuration interface, allowing users to build,
 * test, and deploy A2A agents with various templates and configurations
 * @author A2A Development Team
 * @version 1.0.0
 * @public
 */
sap.ui.define([...], function(...) {
    return Controller.extend("a2a.portal.controller.AgentBuilder", {
        // Controller implementation
    });
});
```

### ES6 Classes

```javascript
/**
 * Notification Service
 * @class NotificationService
 * @description Manages application notifications, including real-time updates,
 * persistence, and user preferences
 * @example
 * const notificationService = new NotificationService();
 * await notificationService.notify("Success", "Operation completed");
 */
export class NotificationService {
    // Class implementation
}
```

## Method Documentation

### Public Methods

```javascript
/**
 * Deploys an agent to the specified environment
 * @memberof a2a.portal.controller.AgentBuilder
 * @function deployAgent
 * @async
 * @public
 * @param {string} agentId - The unique agent identifier
 * @param {('dev'|'staging'|'prod')} environment - Target deployment environment
 * @param {Object} [options] - Optional deployment configuration
 * @param {number} [options.replicas=1] - Number of agent replicas
 * @param {string} [options.memory="512Mi"] - Memory allocation per replica
 * @param {boolean} [options.autoScale=false] - Enable auto-scaling
 * @returns {Promise<DeploymentResult>} The deployment result containing status and URL
 * @throws {DeploymentError} When deployment fails due to configuration or network issues
 * @fires deploymentStarted
 * @fires deploymentCompleted
 * @example
 * try {
 *     const result = await this.deployAgent("agent-123", "staging", {
 *         replicas: 2,
 *         memory: "1Gi",
 *         autoScale: true
 *     });
 *     console.log(`Deployed to: ${result.url}`);
 * } catch (error) {
 *     console.error("Deployment failed:", error);
 * }
 */
async deployAgent(agentId, environment, options = {}) {
    // Method implementation
}
```

### Private Methods

```javascript
/**
 * Validates agent configuration before deployment
 * @memberof a2a.portal.controller.AgentBuilder
 * @function _validateAgentConfig
 * @private
 * @param {AgentConfiguration} config - The agent configuration to validate
 * @returns {ValidationResult} Validation result with errors if any
 * @description Internal method that checks agent configuration for required fields,
 * validates skill compatibility, and ensures deployment readiness
 */
_validateAgentConfig(config) {
    // Method implementation
}
```

### Event Handlers

```javascript
/**
 * Handles template selection change event
 * @memberof a2a.portal.controller.AgentBuilder
 * @function onTemplateChange
 * @public
 * @param {sap.ui.base.Event} oEvent - The selection change event
 * @param {sap.ui.core.Item} oEvent.mParameters.selectedItem - The selected template item
 * @returns {void}
 * @listens sap.m.Select#change
 * @fires templateApplied
 * @description Updates the agent configuration based on the selected template,
 * loading predefined settings and skills
 */
onTemplateChange(oEvent) {
    // Event handler implementation
}
```

## Property Documentation

### Class Properties

```javascript
/**
 * @typedef {Object} AgentBuilderProperties
 * @property {string} projectId - Current project identifier
 * @property {AgentConfiguration} currentAgent - Agent being edited
 * @property {boolean} isDirty - Whether there are unsaved changes
 * @property {Object.<string, Function>} validators - Field validators
 */

/**
 * Agent Builder Controller
 * @class
 * @property {string} _projectId - Current project ID
 * @property {sap.ui.model.json.JSONModel} _agentModel - Agent data model
 * @property {boolean} _isDeploying - Deployment in progress flag
 */
```

### Model Properties

```javascript
/**
 * Agent model structure
 * @namespace AgentModel
 * @property {string} name - Agent display name
 * @property {string} id - Unique agent identifier
 * @property {('reactive'|'proactive'|'hybrid')} type - Agent behavior type
 * @property {string} description - Agent description
 * @property {string[]} skills - Array of skill identifiers
 * @property {Object[]} handlers - Event handler configurations
 * @property {string} handlers[].event - Event name to handle
 * @property {string} handlers[].action - Action to perform
 */
```

## Type Definitions

Define reusable types in a central location:

```javascript
/**
 * @typedef {Object} DeploymentResult
 * @property {boolean} success - Whether deployment succeeded
 * @property {string} deploymentId - Unique deployment identifier
 * @property {string} url - Deployed agent URL
 * @property {string} environment - Target environment
 * @property {Object} metrics - Deployment metrics
 * @property {number} metrics.duration - Deployment duration in ms
 * @property {number} metrics.resourcesCreated - Number of resources
 */

/**
 * @typedef {Object} ValidationResult
 * @property {boolean} valid - Whether validation passed
 * @property {ValidationError[]} errors - Validation errors
 * @property {ValidationWarning[]} warnings - Validation warnings
 */

/**
 * @typedef {Object} ValidationError
 * @property {string} field - Field with error
 * @property {string} message - Error description
 * @property {('required'|'format'|'range'|'custom')} type - Error type
 */
```

## Event Documentation

### Custom Events

```javascript
/**
 * @event a2a.portal.controller.AgentBuilder#deploymentStarted
 * @type {Object}
 * @property {string} agentId - Agent being deployed
 * @property {string} environment - Target environment
 * @property {Date} timestamp - Deployment start time
 */

/**
 * @event a2a.portal.controller.AgentBuilder#deploymentCompleted
 * @type {Object}
 * @property {string} agentId - Deployed agent ID
 * @property {string} environment - Target environment
 * @property {boolean} success - Deployment success status
 * @property {string} [url] - Deployed agent URL if successful
 * @property {Error} [error] - Error details if failed
 */
```

### Event Firing

```javascript
/**
 * Initiates agent deployment
 * @fires a2a.portal.controller.AgentBuilder#deploymentStarted
 * @fires a2a.portal.controller.AgentBuilder#deploymentCompleted
 */
async startDeployment(agentId, environment) {
    // Fire start event
    this.fireEvent("deploymentStarted", {
        agentId,
        environment,
        timestamp: new Date()
    });
    
    try {
        const result = await this._deploy(agentId, environment);
        
        // Fire completion event
        this.fireEvent("deploymentCompleted", {
            agentId,
            environment,
            success: true,
            url: result.url
        });
    } catch (error) {
        // Fire failure event
        this.fireEvent("deploymentCompleted", {
            agentId,
            environment,
            success: false,
            error
        });
    }
}
```

## Examples

### Complete Controller Example

```javascript
/**
 * @fileoverview Project Detail Controller
 * @module a2a/portal/controller/ProjectDetail
 * @requires sap.ui.core.mvc.Controller
 * @requires sap.ui.model.json.JSONModel
 */

sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel"
], function (Controller, JSONModel) {
    "use strict";

    /**
     * Project Detail Controller
     * @class
     * @alias a2a.portal.controller.ProjectDetail
     * @extends sap.ui.core.mvc.Controller
     * @description Manages project details view, including CRUD operations,
     * agent management, and workflow configuration
     */
    return Controller.extend("a2a.portal.controller.ProjectDetail", {
        
        /**
         * Controller initialization
         * @memberof a2a.portal.controller.ProjectDetail
         * @function onInit
         * @public
         * @returns {void}
         * @description Sets up models, bindings, and event handlers
         */
        onInit: function () {
            this._initializeModels();
            this._attachRouteHandlers();
        },

        /**
         * Initializes view models
         * @memberof a2a.portal.controller.ProjectDetail
         * @function _initializeModels
         * @private
         * @returns {void}
         */
        _initializeModels: function () {
            const oProjectModel = new JSONModel({
                project: null,
                agents: [],
                workflows: []
            });
            this.getView().setModel(oProjectModel, "project");
        },

        /**
         * Saves project changes
         * @memberof a2a.portal.controller.ProjectDetail
         * @function onSaveProject
         * @async
         * @public
         * @param {sap.ui.base.Event} oEvent - Button press event
         * @returns {Promise<void>}
         * @fires projectSaved
         * @throws {Error} When save operation fails
         * @example
         * // Triggered by Save button
         * <Button text="Save" press=".onSaveProject" />
         */
        async onSaveProject(oEvent) {
            const oProject = this.getView().getModel("project").getData().project;
            
            try {
                await this._saveProject(oProject);
                this.fireEvent("projectSaved", { project: oProject });
                MessageToast.show("Project saved successfully");
            } catch (error) {
                MessageBox.error(`Failed to save project: ${error.message}`);
            }
        }
    });
});
```

### Service Example

```javascript
/**
 * @fileoverview Notification Service for A2A Portal
 * @module a2a/portal/services/NotificationService
 */

/**
 * Notification Service
 * @class NotificationService
 * @description Handles all notification-related operations including
 * real-time updates, persistence, and user preferences
 * @singleton
 */
export class NotificationService {
    
    /**
     * Creates notification service instance
     * @constructor
     */
    constructor() {
        /**
         * Active notifications
         * @type {Map<string, NotificationData>}
         * @private
         */
        this._notifications = new Map();
        
        /**
         * WebSocket connection for real-time updates
         * @type {WebSocket}
         * @private
         */
        this._websocket = null;
    }

    /**
     * Sends a notification to the user
     * @method notify
     * @async
     * @param {string} title - Notification title
     * @param {string} message - Notification message
     * @param {('info'|'success'|'warning'|'error')} [type='info'] - Notification type
     * @param {Object} [options] - Additional options
     * @param {boolean} [options.persistent=false] - Keep notification until dismissed
     * @param {number} [options.duration=5000] - Auto-dismiss duration in ms
     * @param {Object} [options.action] - Optional action button
     * @returns {Promise<string>} Notification ID
     * @throws {Error} When notification fails
     * @example
     * const notificationId = await notificationService.notify(
     *     "Deployment Complete",
     *     "Your agent has been deployed successfully",
     *     "success",
     *     {
     *         persistent: true,
     *         action: {
     *             label: "View Agent",
     *             handler: () => window.open(agentUrl)
     *         }
     *     }
     * );
     */
    async notify(title, message, type = 'info', options = {}) {
        // Implementation
    }
}
```

## SAP UI5 Specific Patterns

### Fragment Controllers

```javascript
/**
 * @fileoverview Agent Configuration Dialog Fragment Controller
 * @module a2a/portal/fragments/AgentConfigDialog
 */

/**
 * Agent Configuration Dialog Controller
 * @namespace a2a.portal.fragments.AgentConfigDialog
 * @description Controls the agent configuration dialog fragment
 */
const AgentConfigDialog = {
    
    /**
     * Opens the configuration dialog
     * @memberof a2a.portal.fragments.AgentConfigDialog
     * @function open
     * @param {sap.ui.core.mvc.View} oView - Parent view
     * @param {AgentConfiguration} oAgent - Agent to configure
     * @returns {Promise<sap.m.Dialog>} The dialog instance
     */
    open: async function(oView, oAgent) {
        // Implementation
    }
};
```

### Formatter Functions

```javascript
/**
 * @fileoverview Formatting functions for A2A Portal
 * @module a2a/portal/model/formatter
 */

const formatter = {
    
    /**
     * Formats agent status for display
     * @memberof a2a.portal.model.formatter
     * @function statusText
     * @param {('active'|'inactive'|'deploying'|'error')} status - Agent status
     * @returns {string} Formatted status text
     * @example
     * // In XML view
     * <Text text="{path: 'status', formatter: '.formatter.statusText'}" />
     */
    statusText: function(status) {
        const statusMap = {
            'active': 'Active',
            'inactive': 'Inactive',
            'deploying': 'Deploying...',
            'error': 'Error'
        };
        return statusMap[status] || 'Unknown';
    },

    /**
     * Formats deployment state with icon
     * @memberof a2a.portal.model.formatter
     * @function deploymentState
     * @param {string} state - Deployment state
     * @returns {sap.ui.core.IconColor} Icon color
     */
    deploymentState: function(state) {
        // Implementation
    }
};
```

## Tools and Automation

### JSDoc Configuration (.jsdoc.json)

```json
{
  "source": {
    "include": ["src", "app"],
    "exclude": ["node_modules", "test"]
  },
  "opts": {
    "destination": "./docs/api",
    "recurse": true,
    "readme": "./README.md"
  },
  "plugins": ["plugins/markdown"],
  "templates": {
    "cleverLinks": true,
    "monospaceLinks": true
  }
}
```

### Package.json Scripts

```json
{
  "scripts": {
    "docs": "jsdoc -c .jsdoc.json",
    "docs:serve": "jsdoc -c .jsdoc.json && http-server ./docs/api",
    "lint:jsdoc": "eslint . --rule 'valid-jsdoc: error'",
    "migrate:jsdoc": "python scripts/migration/migrate_jsdoc.py"
  }
}
```

### ESLint Configuration

```javascript
{
  "rules": {
    "valid-jsdoc": ["error", {
      "requireReturn": true,
      "requireReturnType": true,
      "requireParamType": true,
      "requireParamDescription": true,
      "preferType": {
        "object": "Object",
        "array": "Array",
        "string": "string",
        "number": "number",
        "boolean": "boolean"
      }
    }],
    "require-jsdoc": ["error", {
      "require": {
        "FunctionDeclaration": true,
        "MethodDefinition": true,
        "ClassDeclaration": true
      }
    }]
  }
}
```

## Migration Strategy

1. **Phase 1: Critical Components (Week 1)**
   - All controllers (668 high priority items)
   - Core services (NotificationService, NavigationService)
   - Public APIs

2. **Phase 2: Supporting Components (Week 2)**
   - Utility functions
   - Formatters
   - Model definitions

3. **Phase 3: Complete Coverage (Week 3)**
   - Private methods
   - Test files
   - Build scripts

## Benefits

1. **Enhanced IDE Support**
   - IntelliSense autocomplete
   - Parameter hints
   - Type checking

2. **Automated Documentation**
   - API documentation generation
   - Interactive documentation site
   - Searchable reference

3. **Improved Maintainability**
   - Clear method contracts
   - Usage examples
   - Deprecation tracking

4. **Better Onboarding**
   - Self-documenting code
   - Clear examples
   - Consistent patterns

## Conclusion

Following this JSDoc style guide ensures that the A2A Platform maintains high-quality, well-documented code that is easy to understand, maintain, and extend. The investment in comprehensive documentation pays dividends in developer productivity and code quality.