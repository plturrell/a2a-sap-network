# A2A UI Components Documentation

## Overview
This documentation provides comprehensive guidance for all UI components used in the A2A Network platform, built on SAP UI5/Fiori framework.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Core Components](#core-components)
3. [Custom Controls](#custom-controls)
4. [Composite Components](#composite-components)
5. [Layout Patterns](#layout-patterns)
6. [Theming](#theming)
7. [Accessibility](#accessibility)
8. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- SAP UI5 1.120.0 or higher
- Node.js 18.x or 20.x
- Basic knowledge of SAP Fiori design principles

### Installation
```bash
npm install
npm run build:ui
```

### Development Setup
```bash
npm run start  # Start development server
npm run test   # Run component tests
npm run lint   # Check code quality
```

## Core Components

### Navigation Components

#### SideNavigation
The main navigation component providing hierarchical menu structure.

```xml
<tnt:SideNavigation
    id="sideNavigation"
    expanded="true"
    itemSelect="onNavigationItemSelect"
    class="a2a-tablet-navigation">
    <tnt:NavigationList>
        <tnt:NavigationListItem 
            text="{i18n>navProjects}" 
            icon="sap-icon://folder-blank" 
            key="projects"
            expanded="true">
            <!-- Nested items -->
        </tnt:NavigationListItem>
    </tnt:NavigationList>
</tnt:SideNavigation>
```

**Properties:**
- `expanded` (boolean): Initial expansion state
- `itemSelect` (function): Selection event handler
- `class` (string): CSS class for styling

**Events:**
- `itemSelect`: Fired when navigation item is selected

#### ShellBar
Enterprise-grade application header with integrated features.

```xml
<f:ShellBar
    id="shellBar"
    title="{i18n>appTitle}"
    homeIcon="./resources/img/sap-logo.svg"
    showNavButton="true"
    showCopilot="true"
    showSearch="true"
    showNotifications="true">
    <!-- Configuration -->
</f:ShellBar>
```

### Data Display Components

#### AgentCard
Displays agent information in a card format.

```xml
<custom:AgentCard
    agentId="{AgentID}"
    name="{Name}"
    status="{Status}"
    performance="{Performance}"
    press="onAgentPress"/>
```

**Properties:**
- `agentId` (string): Unique agent identifier
- `name` (string): Agent display name
- `status` (string): Current agent status
- `performance` (object): Performance metrics

#### WorkflowDiagram
Interactive workflow visualization component.

```xml
<custom:WorkflowDiagram
    workflowId="{WorkflowID}"
    nodes="{path: 'workflow>/nodes'}"
    connections="{path: 'workflow>/connections'}"
    interactive="true"/>
```

### Form Components

#### SmartForm
Intelligent form component with validation and data binding.

```xml
<smartForm:SmartForm
    id="agentForm"
    editable="true"
    title="{i18n>agentDetails}">
    <smartForm:Group>
        <smartForm:GroupElement>
            <smartField:SmartField value="{Name}"/>
        </smartForm:GroupElement>
    </smartForm:Group>
</smartForm:SmartForm>
```

## Custom Controls

### AgentStatusIndicator
Visual indicator for agent operational status.

```javascript
sap.ui.define([
    "sap/ui/core/Control"
], function (Control) {
    "use strict";
    
    return Control.extend("com.sap.a2a.control.AgentStatusIndicator", {
        metadata: {
            properties: {
                status: { type: "string", defaultValue: "inactive" },
                size: { type: "string", defaultValue: "Medium" }
            }
        },
        
        renderer: function (oRm, oControl) {
            oRm.openStart("div", oControl);
            oRm.class("agentStatusIndicator");
            oRm.class("status-" + oControl.getStatus());
            oRm.openEnd();
            oRm.close("div");
        }
    });
});
```

### LoadingIndicator
Custom loading animation for async operations.

```javascript
sap.ui.define([
    "sap/ui/core/Control"
], function (Control) {
    "use strict";
    
    return Control.extend("com.sap.a2a.control.LoadingIndicator", {
        metadata: {
            properties: {
                text: { type: "string", defaultValue: "Loading..." },
                type: { type: "string", defaultValue: "circular" }
            }
        }
    });
});
```

## Composite Components

### AgentDashboard
Complete dashboard view for agent management.

```xml
<mvc:View
    controllerName="com.sap.a2a.controller.AgentDashboard"
    xmlns:mvc="sap.ui.core.mvc"
    xmlns="sap.m"
    xmlns:custom="com.sap.a2a.control">
    
    <Page title="{i18n>agentDashboard}">
        <content>
            <custom:AgentOverview/>
            <custom:PerformanceMetrics/>
            <custom:RecentActivities/>
        </content>
    </Page>
</mvc:View>
```

## Layout Patterns

### Master-Detail Pattern
Used for project and agent management interfaces.

```xml
<SplitApp id="splitApp" initialDetail="detail" initialMaster="master">
    <masterPages>
        <Page id="master" title="{i18n>projects}">
            <List items="{/Projects}">
                <!-- List configuration -->
            </List>
        </Page>
    </masterPages>
    <detailPages>
        <Page id="detail" title="{i18n>projectDetails}">
            <!-- Detail content -->
        </Page>
    </detailPages>
</SplitApp>
```

### Flexible Column Layout
For complex multi-panel interfaces.

```xml
<f:FlexibleColumnLayout id="fcl" layout="TwoColumnsBeginExpanded">
    <f:beginColumnPages>
        <!-- List view -->
    </f:beginColumnPages>
    <f:midColumnPages>
        <!-- Detail view -->
    </f:midColumnPages>
    <f:endColumnPages>
        <!-- Additional info -->
    </f:endColumnPages>
</f:FlexibleColumnLayout>
```

## Theming

### Design Tokens
All components use standardized design tokens for consistent theming.

```css
/* Core tokens */
--sapBrandColor: #0070f2;
--sapHighlightColor: #0064d9;
--sapBaseColor: #fff;
--sapShellColor: #354a5f;

/* Semantic colors */
--sapPositiveColor: #30914c;
--sapNegativeColor: #bb0000;
--sapCriticalColor: #e76500;
```

### Custom Theme Implementation
```javascript
sap.ui.getCore().attachInit(function() {
    sap.ui.getCore().applyTheme("sap_horizon");
    
    // Apply custom CSS variables
    document.documentElement.style.setProperty('--custom-primary', '#0070f2');
});
```

## Accessibility

### ARIA Labels
All components include proper ARIA labeling.

```xml
<Button
    text="{i18n>save}"
    ariaLabelledBy="saveLabel"
    ariaDescribedBy="saveDescription"/>
```

### Keyboard Navigation
Components support full keyboard navigation.

```javascript
onInit: function() {
    this.getView().addEventDelegate({
        onAfterRendering: function() {
            // Enable keyboard navigation
            this._enableKeyboardNavigation();
        }
    }, this);
}
```

### Screen Reader Support
Components provide meaningful announcements.

```javascript
// Announce status changes
sap.ui.getCore().getMessageManager().addMessages(
    new Message({
        message: "Agent status updated",
        type: MessageType.Information,
        technical: false
    })
);
```

## Best Practices

### Performance
1. Use data binding efficiently
2. Implement lazy loading for large datasets
3. Minimize custom control complexity
4. Use view caching where appropriate

### Maintainability
1. Follow SAP naming conventions
2. Document all custom properties
3. Use i18n for all text
4. Implement proper error handling

### Testing
1. Write unit tests for custom controls
2. Implement OPA5 tests for user flows
3. Test accessibility compliance
4. Monitor performance metrics

## Component API Reference

### Full API documentation available at:
- [Control Reference](./api/controls.md)
- [View Reference](./api/views.md)
- [Fragment Reference](./api/fragments.md)
- [Controller Reference](./api/controllers.md)

## Examples and Samples

### Code Sandbox
Interactive examples available at: `/docs/ui-components/examples/`

### Sample Applications
- [Agent Management](./examples/agent-management/)
- [Workflow Designer](./examples/workflow-designer/)
- [Analytics Dashboard](./examples/analytics-dashboard/)