# A2A Developer Portal - Navigation Guide

## Navigation Architecture

The A2A Developer Portal uses a hybrid navigation pattern combining:
- **Side Navigation Panel** for main functional areas
- **Header Bar Actions** for user-specific features
- **Context-Dependent Navigation** for project-specific tools

## Navigation Patterns

### 1. Main Navigation (Side Panel)
These are always accessible from the left navigation panel:

| Navigation Item | Route | View | Description |
|----------------|-------|------|-------------|
| Projects | `/projects` | Projects.view.xml | Main project dashboard |
| Templates | `/templates` | Templates.view.xml | Agent template library |
| Testing | `/testing` | Testing.view.xml | Test management center |
| Deployment | `/deployment` | Deployment.view.xml | Deployment management |
| Monitoring | `/monitoring` | Monitoring.view.xml | System monitoring dashboard |
| A2A Network | `/a2a-network` | A2ANetworkManager.view.xml | Network topology viewer |

### 2. Header-Only Navigation
These are accessible only through header bar actions:

| Action | Route | View | Access Point |
|--------|-------|------|--------------|
| User Profile | `/profile` | UserProfile.view.xml | User avatar button in header |
| Notifications | N/A | NotificationPanel.fragment.xml | Bell icon in header |
| Help | N/A | HelpPanel.fragment.xml | Help icon in header (F1 key) |

### 3. Context-Dependent Navigation
These views are only accessible within a project context:

| View | Route Pattern | Access From | Description |
|------|---------------|-------------|-------------|
| Project Detail | `/project/{projectId}` | Projects list item click | Detailed project view |
| Agent Builder | `/project/{projectId}/agent-builder` | Project Detail > Build Agent | Visual agent configuration |
| BPMN Designer | `/project/{projectId}/bpmn-designer` | Project Detail > Design Workflow | BPMN workflow designer |
| Code Editor | `/project/{projectId}/code-editor` | Project Detail > Edit Code | Integrated code editor |

## Navigation Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Header Bar                             │
│  [Logo] [Title]                    [Help] [Notify] [Profile]  │
├───────────────┬─────────────────────────────────────────────┤
│               │                                               │
│  Side Nav     │              Main Content Area                │
│               │                                               │
│  ▼ Projects   │  ┌─────────────────────────────────────┐    │
│  Templates    │  │                                      │    │
│  Testing      │  │    Currently Selected View          │    │
│  Deployment   │  │                                      │    │
│  Monitoring   │  │    (e.g., Projects, Templates)      │    │
│  A2A Network  │  │                                      │    │
│               │  └─────────────────────────────────────┘    │
│               │                                               │
└───────────────┴─────────────────────────────────────────────┘
```

## Navigation Implementation

### Programmatic Navigation

```javascript
// Navigate to main routes
this.getOwnerComponent().getRouter().navTo("projects");
this.getOwnerComponent().getRouter().navTo("templates");

// Navigate with parameters
this.getOwnerComponent().getRouter().navTo("projectDetail", {
    projectId: "proj123"
});

// Navigate to nested routes
this.getOwnerComponent().getRouter().navTo("agentBuilder", {
    projectId: "proj123"
});
```

### Direct URL Access

All routes support direct URL access:
- `https://app.example.com/#/projects`
- `https://app.example.com/#/project/proj123`
- `https://app.example.com/#/project/proj123/agent-builder`

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Quick project search |
| `Ctrl+N` | New project |
| `F1` | Open help panel |
| `Ctrl+H` | Toggle help tooltips |
| `Ctrl+T` | Start guided tour |

## Design Rationale

### Why User Profile is Header-Only

The User Profile is intentionally kept in the header bar rather than the side navigation because:

1. **SAP Fiori Guidelines**: User-specific actions belong in the shell bar
2. **Space Efficiency**: Keeps side navigation focused on functional areas
3. **Consistency**: Matches SAP standard applications
4. **User Expectations**: Users expect profile access in the top-right corner

### Context-Dependent Views

Agent Builder, BPMN Designer, and Code Editor are context-dependent because:

1. **Project Context Required**: These tools need an active project context
2. **State Management**: They maintain project-specific state
3. **Navigation Hierarchy**: Reinforces the project-centric workflow
4. **Resource Optimization**: Only loaded when needed within a project

## Future Enhancements

1. **Breadcrumb Navigation**: Add breadcrumbs for context-dependent views
2. **Quick Actions**: Floating action button for common tasks
3. **Recent Items**: Quick access to recently viewed projects
4. **Favorites**: Pin frequently used projects or templates
5. **Search Integration**: Global search with navigation shortcuts

## Accessibility

All navigation elements support:
- Keyboard navigation (Tab, Arrow keys)
- Screen reader announcements
- High contrast mode
- Focus indicators
- ARIA labels and landmarks