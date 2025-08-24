# A2A Developer Portal - Complete UI Screens Catalog

## Overview
The A2A Developer Portal is a comprehensive web application built with SAP UI5 framework that provides a complete environment for developing, managing, and deploying A2A (Agent-to-Agent) solutions.

## UI Screens Inventory

### 1. Main Application Structure
| Screen | File | Purpose |
|--------|------|---------|
| **Application Shell** | `App.view.xml` | Root application container with navigation sidebar and header |
| **Entry Point** | `index.html` | HTML entry point that loads the UI5 application |

### 2. Project Management Screens
| Screen | File | Purpose |
|--------|------|---------|
| **Projects List** | `Projects.view.xml` | Main project listing with tiles/table view, search, and filters |
| **Projects Smart View** | `ProjectsSmart.view.xml` | Intelligent project view with recommendations and categorization |
| **Project Master-Detail** | `ProjectMasterDetail.view.xml` | Split-screen project management for desktop |
| **Project Details** | `ProjectDetail.view.xml` | Comprehensive project information and configuration |

#### Project Dialogs
| Dialog | File | Purpose |
|--------|------|---------|
| **Create Project** | `CreateProjectDialog.fragment.xml` | New project creation wizard |
| **Edit Project** | `EditProjectDialog.fragment.xml` | Project settings modification |
| **Import Project** | `ImportProjectDialog.fragment.xml` | Import existing projects |
| **Project Actions** | `ProjectActionsPopover.fragment.xml` | Quick actions menu |

### 3. Agent Development Screens
| Screen | File | Purpose |
|--------|------|---------|
| **Agent Builder** | `AgentBuilder.view.xml` | Visual agent development interface with configuration tabs |
| **Agent Details** | `AgentDetailDialog.fragment.xml` | Detailed agent information display |
| **Register Agent** | `RegisterAgentDialog.fragment.xml` | Agent registration to A2A network |

### 4. Workflow Design Screens
| Screen | File | Purpose |
|--------|------|---------|
| **BPMN Designer** | `BPMNDesigner.view.xml` | Visual workflow designer with BPMN 2.0 support |
| **Code Editor** | `CodeEditor.view.xml` | Integrated code editor with syntax highlighting |

### 5. A2A Network Management
| Screen | File | Purpose |
|--------|------|---------|
| **Network Manager** | `A2ANetworkManager.view.xml` | View and manage connected agents and network status |
| **Network Settings** | `NetworkSettingsDialog.fragment.xml` | Network configuration options |
| **Send Message** | `SendMessageDialog.fragment.xml` | Inter-agent communication interface |

### 6. User & System Management
| Screen | File | Purpose |
|--------|------|---------|
| **Dashboard** | `OverviewPage.view.xml` | System dashboard with metrics and recent activities |
| **User Profile** | `UserProfile.view.xml` | User account management and preferences |
| **Settings** | `SettingsDialog.fragment.xml` | Application-wide settings |
| **Change Password** | `ChangePasswordDialog.fragment.xml` | Password update interface |

### 7. Communication & Notifications
| Screen | File | Purpose |
|--------|------|---------|
| **Notification Panel** | `NotificationPanel.fragment.xml` | Central notification center |
| **Notification Actions** | `NotificationActions.fragment.xml` | Quick notification actions |
| **Webhooks Config** | `WebhooksDialog.fragment.xml` | Webhook management interface |
| **Add Webhook** | `AddWebhookDialog.fragment.xml` | New webhook creation |

### 8. Help & Support
| Screen | File | Purpose |
|--------|------|---------|
| **Help Panel** | `HelpPanel.fragment.xml` | Contextual help sidebar with search |
| **Guided Tours** | Managed by `GuidedTourManager.js` | Interactive feature walkthroughs |

### 9. Utility Screens
| Screen | File | Purpose |
|--------|------|---------|
| **Sort Dialog** | `SortDialog.fragment.xml` | Generic sorting options for lists |

## Navigation Structure

### Primary Navigation Routes
```
/ (root)
├── /projects (default)
│   ├── /projects/:id (Project Detail)
│   ├── /projects/:id/agents (Agent Builder)
│   ├── /projects/:id/workflows (BPMN Designer)
│   └── /projects/:id/code (Code Editor)
├── /profile (User Profile)
├── /network (A2A Network Manager)
└── /dashboard (Overview Page)
```

### Screen Relationships
```
Projects List
    ├── Create Project Dialog
    ├── Project Detail
    │   ├── Agent Builder
    │   │   ├── Agent Detail Dialog
    │   │   └── Register Agent Dialog
    │   ├── BPMN Designer
    │   └── Code Editor
    └── Project Actions Menu

A2A Network Manager
    ├── Network Settings Dialog
    └── Send Message Dialog

User Profile
    ├── Change Password Dialog
    └── Settings Dialog
```

## Screen Categories

### 1. **List/Grid Views** (4 screens)
- Projects List
- Projects Smart View
- Project Master-Detail
- Overview Dashboard

### 2. **Detail/Form Views** (5 screens)
- Project Detail
- Agent Builder
- BPMN Designer
- Code Editor
- User Profile

### 3. **Management Views** (1 screen)
- A2A Network Manager

### 4. **Dialogs** (14 dialogs)
- Project Management: 4 dialogs
- Agent Management: 2 dialogs
- Network/Communication: 4 dialogs
- User Management: 2 dialogs
- Utility: 2 dialogs

### 5. **Panels/Fragments** (3 panels)
- Help Panel
- Notification Panel
- Notification Actions

## Technical Details

### View Technologies
- **SAP UI5 XML Views**: Primary UI definition
- **JavaScript Controllers**: Business logic
- **XML Fragments**: Reusable UI components
- **HTML**: Entry point only

### UI Patterns
- **Master-Detail**: For project management
- **Tabbed Forms**: In Agent Builder
- **Visual Designer**: BPMN workflow creation
- **Code Editor**: Monaco/ACE editor integration
- **Responsive Design**: Mobile and desktop support

### Key Features by Screen
1. **Projects**: CRUD operations, search, filter, sort
2. **Agent Builder**: Visual configuration, code generation
3. **BPMN Designer**: Drag-drop design, validation
4. **Network Manager**: Real-time status, agent discovery
5. **Dashboard**: Analytics, charts, activity feed

## Total UI Component Count
- **Main Views**: 11
- **Dialogs**: 14
- **Fragments/Panels**: 3
- **Services**: 3
- **Utilities**: 2
- **Total**: 33 UI components