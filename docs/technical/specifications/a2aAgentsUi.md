# A2A Agents UI Test Cases - ISO/SAP Hybrid Standard

## Document Overview
**Document ID**: TC-UI-AGT-001  
**Version**: 1.0  
**Standard Compliance**: ISO/IEC/IEEE 29119-3:2021 + SAP Solution Manager Templates  
**Test Level**: System Integration Testing  
**Component**: A2A Agents User Interface (Developer Portal)  
**Business Process**: Agent Development and Management  

---

## Test Case ID: TC-UI-AGT-001
**Test Objective**: Verify project workspace initialization and development environment setup  
**Business Process**: Development Environment Initialization  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-001
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Environment Setup
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/Projects.view.xml:1-50`
- **Controller**: `a2a_agents/frontend/src/controller/Projects.controller.js:onInit()`
- **Functions Under Test**: `onInit()`, `_initializeWorkspace()`, `_loadProjectList()`

### Test Preconditions
1. **Developer Authentication**: Valid developer account with project access rights
2. **Development Environment**: Node.js 18+, npm/yarn package manager available
3. **Backend Services**: A2A Agents backend services running and accessible
4. **Database Connection**: Project database schema exists and is accessible
5. **File System Access**: Write permissions to project workspace directory

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Developer Role | Agent_Developer | String | User Management |
| Workspace Path | ~/a2a-workspace | String | Environment Config |
| Node Version | 18.17.0+ | String | Development Environment |
| Project Template | basic-agent | String | Template Registry |
| Git Repository | https://github.com/user/agent-project | URL | Version Control |

### Test Procedure Steps
1. **Step 1 - Workspace Access**
   - Action: Navigate to Developer Portal URL: `https://localhost:3000/projects`
   - Expected: Projects view loads with workspace selector
   - Verification: Workspace dropdown shows available workspaces

2. **Step 2 - Project List Loading**
   - Action: Select default workspace and wait for project enumeration
   - Expected: Project grid/list displays existing projects within 3 seconds
   - Verification: Project cards show name, status, last modified date

3. **Step 3 - Development Environment Check**
   - Action: System verifies development prerequisites automatically
   - Expected: Environment status indicator shows all dependencies satisfied
   - Verification: Node.js, npm, Git versions displayed and validated

4. **Step 4 - Create New Project**
   - Action: Click "Create New Project" button
   - Expected: Project creation dialog opens with template options
   - Verification: Template gallery displays available agent templates

5. **Step 5 - Project Template Selection**
   - Action: Select "Basic Agent" template and provide project details
   - Expected: Template preview shows project structure and dependencies
   - Verification: Preview pane displays files that will be created

6. **Step 6 - Project Initialization**
   - Action: Confirm project creation with name "TestAgent001"
   - Expected: Project scaffolding completes within 30 seconds
   - Verification: Project appears in project list with "INITIALIZING" status

7. **Step 7 - Workspace Synchronization**
   - Action: Wait for project workspace to fully initialize
   - Expected: Project status updates to "READY" and becomes clickable
   - Verification: File system contains project directory with template files

### Expected Results
- **Initialization Criteria**:
  - Projects view loads within 2 seconds
  - All existing projects enumerated correctly
  - Development environment validation passes
  - Project creation completes successfully
  
- **Environment Validation Criteria**:
  - Node.js version 18+ detected and validated
  - Package manager (npm/yarn) availability confirmed
  - Git version control system accessible
  - File system permissions verified for workspace

- **Project Creation Criteria**:
  - Template selection provides accurate preview
  - Scaffolding process completes without errors
  - Generated project structure matches template specification
  - Dependencies installation succeeds automatically

### Test Postconditions
- Developer workspace is initialized and functional
- At least one project exists in the workspace
- Development environment is validated and ready
- Project creation workflow is operational

### Error Scenarios & Recovery
1. **Environment Missing**: Display clear error message with installation instructions
2. **Permission Denied**: Show workspace permission error with resolution steps
3. **Template Download Failed**: Retry mechanism with fallback to cached templates
4. **Project Name Conflict**: Validate uniqueness and suggest alternatives

### Validation Points
- [ ] Projects view renders correctly
- [ ] Environment validation passes
- [ ] Project templates load successfully
- [ ] New project creation works
- [ ] Workspace synchronization functions
- [ ] Error handling provides clear guidance
- [ ] Performance meets specified timeouts

### Related Test Cases
- **Depends On**: TC-AUTH-AGT-001 (Developer Authentication)
- **Triggers**: TC-UI-AGT-002 (Project Detail View)
- **Related**: TC-UI-AGT-015 (Template Management)

### Standard Compliance
- **ISO 29119-3**: Complete test case specification with traceability
- **SAP Standards**: Developer portal UX follows SAP Build Work Zone patterns
- **Test Coverage**: Development environment setup and project initialization

---

## Test Case ID: TC-UI-AGT-002
**Test Objective**: Verify project detail view and code editor functionality  
**Business Process**: Agent Code Development  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-002
- **Test Priority**: Critical (P1)
- **Test Type**: IDE Functionality, Code Editor
- **Execution Method**: Manual
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/ProjectDetail.view.xml:1-100`
- **Controller**: `a2a_agents/frontend/src/controller/ProjectDetail.controller.js:onInit()`
- **Functions Under Test**: `onInit()`, `_loadProjectFiles()`, `_initializeCodeEditor()`

### Test Preconditions
1. **Project Exists**: TC-UI-AGT-001 completed with at least one project available
2. **Code Editor Library**: Monaco Editor or similar IDE component loaded
3. **File System Access**: Read/write permissions to project files
4. **Language Support**: Syntax highlighting for JavaScript/TypeScript available

### Test Input Data
| Component | Configuration | Expected Behavior |
|-----------|---------------|------------------|
| File Tree | Project file structure | Hierarchical display |
| Code Editor | Monaco Editor | Syntax highlighting, IntelliSense |
| Terminal | Integrated terminal | Command execution capability |
| Debugger | V8 debug adapter | Breakpoint and step debugging |

### Test Procedure Steps
1. **Step 1 - Project Detail Navigation**
   - Action: Click on project "TestAgent001" from project list
   - Expected: Project detail view opens with file explorer and editor
   - Verification: URL updates to /projects/TestAgent001, editor pane visible

2. **Step 2 - File Tree Exploration**
   - Action: Examine project file tree structure in left panel
   - Expected: All project files and folders displayed hierarchically
   - Verification: Folders expandable, files show correct icons by type

3. **Step 3 - Code Editor Loading**
   - Action: Click on main.js file in file tree
   - Expected: File content loads in code editor with syntax highlighting
   - Verification: JavaScript syntax coloring active, line numbers visible

4. **Step 4 - IntelliSense and Auto-completion**
   - Action: Begin typing code in editor (e.g., "console.")
   - Expected: Auto-completion suggestions appear
   - Verification: Method suggestions shown, documentation tooltips available

5. **Step 5 - File Modification and Save**
   - Action: Modify code content and press Ctrl+S
   - Expected: File saves successfully, unsaved indicator clears
   - Verification: File system updated, editor shows saved state

6. **Step 6 - Multiple File Tabs**
   - Action: Open additional files (package.json, README.md)
   - Expected: Each file opens in separate tab with proper content type
   - Verification: Tab switching works, content persists between tabs

7. **Step 7 - Integrated Terminal**
   - Action: Open integrated terminal panel
   - Expected: Terminal opens in project root directory
   - Verification: Can execute npm commands, output displays correctly

### Expected Results
- **Editor Functionality Criteria**:
  - Code editor loads within 2 seconds
  - Syntax highlighting works for supported languages
  - Auto-completion provides relevant suggestions
  - File save/load operations complete successfully
  
- **Navigation Criteria**:
  - File tree accurately reflects project structure
  - File clicking opens content in editor
  - Tab management works for multiple files
  - Terminal integration functions properly

- **Performance Criteria**:
  - Large files (>1MB) load within 5 seconds
  - Typing response time < 50ms
  - Memory usage remains stable during extended editing
  - Auto-save triggers appropriately

### Test Postconditions
- Project files are accessible and editable
- Code editor is fully functional
- File changes are persisted correctly
- Development environment is ready for coding

### Error Scenarios & Recovery
1. **File Load Error**: Display error message, offer reload option
2. **Save Failure**: Show save error, offer retry with different location
3. **Editor Crash**: Auto-save recovery, restore from backup
4. **Large File Performance**: Pagination or virtual scrolling for huge files

### Validation Points
- [ ] Project detail view opens correctly
- [ ] File tree displays project structure
- [ ] Code editor loads with syntax highlighting
- [ ] Auto-completion works properly
- [ ] File save/load operations succeed
- [ ] Multiple tabs function correctly
- [ ] Integrated terminal is operational

### Related Test Cases
- **Depends On**: TC-UI-AGT-001 (Project Initialization)
- **Triggers**: TC-UI-AGT-003 (Agent Builder Interface)
- **Related**: TC-UI-AGT-012 (Code Editor Advanced Features)

---

## Test Case ID: TC-UI-AGT-003
**Test Objective**: Verify agent builder interface and configuration management  
**Business Process**: Agent Configuration and Build  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-003
- **Test Priority**: High (P2)
- **Test Type**: Functional, Configuration Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/agentBuilder.view.xml:1-150`
- **Controller**: `a2a_agents/frontend/src/controller/agentBuilder.controller.js:onInit()`
- **Functions Under Test**: `onInit()`, `_loadAgentConfig()`, `_validateConfiguration()`

### Test Preconditions
1. **Project Loaded**: TC-UI-AGT-002 completed with project open in editor
2. **Agent Configuration**: Valid agent.config.json exists in project
3. **Build System**: Agent build tools and dependencies available
4. **Validation Rules**: Agent configuration schema loaded

### Test Input Data
| Configuration Section | Required Fields | Optional Fields | Validation Rules |
|--------------------|----------------|-----------------|------------------|
| Agent Metadata | name, version, description | author, license | Name must be unique |
| Capabilities | skills[], triggers[] | dependencies[] | At least one skill required |
| Runtime Config | runtime, memory, timeout | environment | Memory > 128MB |
| Network Config | endpoints[], ports[] | security[] | Endpoints must be valid URLs |

### Test Procedure Steps
1. **Step 1 - Agent Builder Access**
   - Action: Click "Build Agent" button or tab in project interface
   - Expected: Agent builder interface opens with configuration form
   - Verification: All configuration sections visible, current config loaded

2. **Step 2 - Metadata Configuration**
   - Action: Update agent name, version, and description fields
   - Expected: Changes reflected immediately, validation occurs on blur
   - Verification: Field validation messages appear for invalid inputs

3. **Step 3 - Capability Management**
   - Action: Add new skill to agent capabilities list
   - Expected: Skill selection dialog opens with available skills
   - Verification: Selected skills added to configuration, dependencies updated

4. **Step 4 - Runtime Configuration**
   - Action: Adjust memory allocation and timeout settings
   - Expected: Sliders/inputs update configuration values
   - Verification: Configuration JSON updates in real-time

5. **Step 5 - Configuration Validation**
   - Action: Click "Validate Configuration" button
   - Expected: Comprehensive validation runs, results displayed
   - Verification: Validation errors highlighted, warnings shown

6. **Step 6 - Build Process Initiation**
   - Action: Click "Build Agent" with valid configuration
   - Expected: Build process starts, progress indicator shows status
   - Verification: Build logs stream in real-time, build artifacts created

### Expected Results
- **Configuration Management Criteria**:
  - All configuration sections load correctly
  - Field validation occurs in real-time
  - Configuration changes persist automatically
  - JSON schema validation passes
  
- **Build Process Criteria**:
  - Build initiates within 2 seconds of request
  - Progress indication shows accurate status
  - Build completes within 60 seconds for standard agent
  - Build artifacts available for download/deployment

### Test Postconditions
- Agent configuration is valid and complete
- Build artifacts are generated successfully
- Agent is ready for testing or deployment
- Configuration changes are saved to project

### Related Test Cases
- **Depends On**: TC-UI-AGT-002 (Project Detail View)
- **Triggers**: TC-UI-AGT-004 (Agent Testing Interface)
- **Related**: TC-UI-AGT-013 (Build System Integration)

---

## Test Case ID: TC-UI-AGT-004
**Test Objective**: Verify agent testing interface and debugging capabilities  
**Business Process**: Agent Testing and Quality Assurance  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-004
- **Test Priority**: High (P2)
- **Test Type**: Testing Framework, Debugging
- **Execution Method**: Manual
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/Testing.view.xml:1-120`
- **Controller**: `a2a_agents/frontend/src/controller/Testing.controller.js:onInit()`
- **Functions Under Test**: `onInit()`, `_runTests()`, `_debugAgent()`

### Test Preconditions
1. **Agent Built**: TC-UI-AGT-003 completed with successful agent build
2. **Test Framework**: Testing infrastructure available (Jest, Mocha, etc.)
3. **Debug Tools**: Debugging capabilities integrated and functional
4. **Test Data**: Sample test cases and mock data available

### Test Input Data
| Test Type | Configuration | Expected Result | Validation Method |
|-----------|---------------|-----------------|------------------|
| Unit Tests | Individual functions | Pass/Fail status | Assertion results |
| Integration Tests | Agent API endpoints | Response validation | HTTP status codes |
| Performance Tests | Load simulation | Response time metrics | Benchmark comparison |
| Debugging Session | Breakpoint placement | Variable inspection | Debug console output |

### Test Procedure Steps
1. **Step 1 - Test Suite Loading**
   - Action: Navigate to Testing tab in project interface
   - Expected: Test suite interface loads with available test categories
   - Verification: Unit, integration, and performance tests listed

2. **Step 2 - Unit Test Execution**
   - Action: Click "Run Unit Tests" button
   - Expected: Unit tests execute and results displayed in real-time
   - Verification: Pass/fail status for each test, execution time shown

3. **Step 3 - Test Result Analysis**
   - Action: Review test results and identify any failures
   - Expected: Detailed failure information with stack traces
   - Verification: Failed tests show specific error messages and line numbers

4. **Step 4 - Debug Mode Activation**
   - Action: Click "Debug" button for a failing test
   - Expected: Debug interface opens with breakpoint capabilities
   - Verification: Code execution pauses at breakpoints, variables inspectable

5. **Step 5 - Live Agent Testing**
   - Action: Start agent in test mode and send test messages
   - Expected: Agent responds correctly, message flow visible
   - Verification: Request/response logged, performance metrics captured

### Expected Results
- **Test Execution Criteria**:
  - Tests complete within reasonable time (< 30 seconds)
  - Results clearly indicate pass/fail status
  - Detailed error information available for failures
  - Performance benchmarks within expected ranges
  
- **Debugging Criteria**:
  - Breakpoints can be set and triggered correctly
  - Variable values are inspectable at runtime
  - Step-through debugging works properly
  - Debug console provides useful information

### Test Postconditions
- Test results are saved and accessible
- Debugging session can be resumed
- Agent performance baseline established
- Quality metrics updated in project dashboard

### Related Test Cases
- **Depends On**: TC-UI-AGT-003 (Agent Builder)
- **Triggers**: TC-UI-AGT-005 (Deployment Pipeline)
- **Related**: TC-UI-AGT-014 (Performance Monitoring)

---

## Test Case ID: TC-UI-AGT-005
**Test Objective**: Verify deployment pipeline and environment management  
**Business Process**: Agent Deployment and Operations  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-005
- **Test Priority**: Critical (P1)
- **Test Type**: Deployment, Operations
- **Execution Method**: Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/Deployment.view.xml:1-100`
- **Controller**: `a2a_agents/frontend/src/controller/Deployment.controller.js:onInit()`
- **Functions Under Test**: `onInit()`, `_deployAgent()`, `_monitorDeployment()`

### Test Preconditions
1. **Agent Tested**: TC-UI-AGT-004 completed with passing tests
2. **Deployment Environment**: Target environment configured and accessible
3. **Credentials**: Deployment credentials and permissions available
4. **Infrastructure**: Container orchestration platform ready

### Test Input Data
| Environment | Configuration | Resource Limits | Expected Outcome |
|-------------|---------------|-----------------|------------------|
| Development | Single instance | 256MB RAM, 0.5 CPU | Fast deployment |
| Staging | Load balanced | 512MB RAM, 1 CPU | Production-like |
| Production | High availability | 1GB RAM, 2 CPU | Zero downtime |

### Test Procedure Steps
1. **Step 1 - Deployment Environment Selection**
   - Action: Navigate to Deployment view and select target environment
   - Expected: Environment details and configuration displayed
   - Verification: Resource requirements and deployment strategy shown

2. **Step 2 - Pre-deployment Validation**
   - Action: System runs pre-deployment checks automatically
   - Expected: All validation checks pass (tests, security, dependencies)
   - Verification: Green checkmarks for all validation criteria

3. **Step 3 - Deployment Initiation**
   - Action: Click "Deploy" button with selected configuration
   - Expected: Deployment pipeline starts, progress tracking begins
   - Verification: Pipeline stages show progress, logs stream in real-time

4. **Step 4 - Deployment Monitoring**
   - Action: Monitor deployment progress through pipeline visualization
   - Expected: Each stage completes successfully within expected timeframes
   - Verification: Build, test, deploy stages all show success status

5. **Step 5 - Post-deployment Verification**
   - Action: Wait for deployment completion and health checks
   - Expected: Agent instance starts successfully, health checks pass
   - Verification: Agent status shows "Running", endpoint responds correctly

### Expected Results
- **Deployment Process Criteria**:
  - Deployment completes within 10 minutes for standard agent
  - Zero downtime deployment to production environment
  - Rollback capability available in case of issues
  - Environment-specific configuration applied correctly
  
- **Monitoring and Health Criteria**:
  - Agent health monitoring activates automatically
  - Performance metrics collection begins
  - Alerting configured for critical issues
  - Logs aggregation and analysis available

### Test Postconditions
- Agent successfully deployed to target environment
- Monitoring and alerting systems active
- Agent is accessible and responsive
- Deployment history recorded for audit

### Related Test Cases
- **Depends On**: TC-UI-AGT-004 (Agent Testing)
- **Triggers**: TC-UI-AGT-006 (Production Monitoring)
- **Related**: TC-UI-AGT-016 (Rollback Procedures)

---

## Test Case ID: TC-UI-AGT-006 (Migrated: TC-AA-001)
**Test Objective**: Verify workspace selector displays and functions correctly in application shell  
**Business Process**: Workspace Management and Context Switching  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-006 (Legacy: TC-AA-001)
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, UI Component
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/view/App.view.xml:25-36`
- **Controller**: `a2aAgents/frontend/webapp/controller/App.controller.ts:285-342`
- **Functions Under Test**: `onWorkspaceSelectionChange()`, `_refreshWorkspaceData()`, workspace model initialization

### Test Preconditions
1. **User Authentication**: Valid user session with workspace access permissions
2. **Workspace Data**: Multiple workspaces available for user account
3. **Application Shell**: Main application loaded with ShellBar rendered
4. **Backend Services**: Workspace API endpoints accessible and operational
5. **Browser Environment**: Modern browser with JavaScript enabled

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Workspace ID | default, personal, team, enterprise | String | Workspace Model |
| User Permissions | workspace_read, workspace_switch | Array | User Model |
| API Endpoint | /api/v1/workspaces/{id}/data | String | Backend Configuration |
| Selector Width | 200px | String | CSS Configuration |
| Animation Duration | 300ms | Number | UI Framework |

### Test Procedure Steps
1. **Step 1 - Workspace Selector Visibility**
   - Action: Navigate to main application URL and wait for complete page load
   - Expected: Workspace selector ComboBox displays in ShellBar additional content area
   - Verification: Element with ID "workspaceSelector" is visible and has placeholder text

2. **Step 2 - Workspace Options Loading**
   - Action: Click workspace selector dropdown to expand options
   - Expected: Dropdown opens showing all available workspaces for user
   - Verification: ComboBox items display correct workspace names (Default, Personal, Team, Enterprise)

3. **Step 3 - Default Workspace Selection**
   - Action: Verify default workspace selection on page load
   - Expected: "Default Workspace" is pre-selected as current workspace
   - Verification: ComboBox selectedKey matches "default", display shows "Default Workspace"

4. **Step 4 - Workspace Selection Change**
   - Action: Select "Personal Workspace" from dropdown options
   - Expected: Selection changes, confirmation toast appears, workspace model updates
   - Verification: MessageToast shows "Switched to Personal Workspace", selectedKey = "personal"

5. **Step 5 - Workspace Data Refresh**
   - Action: Monitor network requests and app state during workspace change
   - Expected: API call to `/api/v1/workspaces/personal/data`, app shows busy indicator
   - Verification: Network tab shows GET request, app.busy = true during refresh

6. **Step 6 - Event Bus Communication**
   - Action: Verify workspace change event is published to application event bus
   - Expected: Event "app/workspaceChanged" published with workspace details
   - Verification: Event contains workspaceId "personal" and workspaceName "Personal Workspace"

7. **Step 7 - Persistence Verification**
   - Action: Refresh page and verify workspace selection is maintained
   - Expected: Previously selected workspace remains active after page reload
   - Verification: localStorage contains "selectedWorkspace" = "personal"

8. **Step 8 - Error Handling**
   - Action: Simulate workspace API failure (network disconnection)
   - Expected: Error message displayed, workspace reverts to previous selection
   - Verification: MessageToast shows "Error loading workspace data", no data corruption

### Expected Results
- **Display Criteria**:
  - Workspace selector renders within 1 second of page load
  - ComboBox displays with 200px width and appropriate styling
  - Placeholder text shows "Select Workspace" when no selection
  - All workspace options are visible and selectable

- **Functionality Criteria**:
  - Workspace selection changes update model immediately
  - API requests complete within 3 seconds under normal conditions
  - Confirmation messages appear for successful workspace switches
  - Busy indicator shows during data refresh operations

- **Data Management Criteria**:
  - Workspace selection persists across browser sessions
  - Event bus notifications enable other components to react to changes
  - Network errors are handled gracefully without breaking functionality
  - Workspace data refresh only occurs when selection actually changes

### Test Postconditions
- Selected workspace is active and data is loaded correctly
- Workspace preference is saved in localStorage
- Application components are aware of current workspace context
- User can continue working in selected workspace environment

### Error Scenarios & Recovery
1. **Network Failure**: Show error message, maintain previous workspace selection
2. **Invalid Workspace ID**: Revert to default workspace, log warning
3. **Permission Denied**: Display access denied message, disable workspace switching
4. **API Timeout**: Show timeout message, allow retry after 30 seconds
5. **Model Update Failure**: Restore previous state, alert development team

### Validation Points
- [ ] Workspace selector ComboBox displays in ShellBar additional content
- [ ] All available workspaces load correctly in dropdown options
- [ ] Default workspace selection works on initial page load
- [ ] Workspace selection change triggers model update and API call
- [ ] Confirmation toast message appears on successful workspace switch
- [ ] Event bus publishes workspace change event with correct parameters
- [ ] Workspace selection persists in localStorage across sessions
- [ ] Error handling works for network failures and invalid responses

### Related Test Cases
- **Depends On**: TC-UI-AGT-001 (Project Workspace Initialization)
- **Triggers**: TC-UI-AGT-007 (Workspace Data Synchronization), TC-UI-AGT-008 (Multi-Workspace Projects)
- **Related**: TC-UI-AGT-002 (Project Detail View), TC-BE-AGT-006 (User Authentication)

### Standard Compliance
- **ISO 29119-3**: Complete workspace selector functional test specification
- **SAP Standards**: SAP Fiori UX Guidelines for workspace selection patterns
- **UI Standards**: SAP UI5 ComboBox component standards, ShellBar integration patterns

---

## Test Case ID: TC-UI-AGT-007 (Migrated: TC-AA-002)
**Test Objective**: Verify workspace switching functionality and data synchronization  
**Business Process**: Workspace Context Management and Data Isolation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-007 (Legacy: TC-AA-002)
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:285-342`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:26-35`
- **Functions Under Test**: `onWorkspaceSelectionChange()`, `_refreshWorkspaceData()`, event bus communication

### Test Preconditions
1. **Workspace Selector Active**: TC-UI-AGT-006 completed with workspace selector functional
2. **Multiple Workspaces**: At least 2 different workspaces available for user
3. **Active Session**: User authenticated with valid session token
4. **Backend API**: Workspace data endpoints operational and responsive
5. **Event Bus**: Application event bus initialized and functional

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Source Workspace | default | String | Current Selection |
| Target Workspace | personal | String | User Selection |
| API Response Time | < 3 seconds | Number | Performance SLA |
| Event Bus Channel | app | String | Application Framework |
| Local Storage Key | selectedWorkspace | String | Persistence Layer |

### Test Procedure Steps
1. **Step 1 - Initial Workspace Verification**
   - Action: Verify current workspace is "Default Workspace"
   - Expected: Workspace selector shows "Default Workspace" as selected
   - Verification: ComboBox selectedKey = "default", display text matches

2. **Step 2 - Workspace Switch Initiation**
   - Action: Click workspace selector and select "Personal Workspace"
   - Expected: Selection change event triggers, model update begins
   - Verification: `onWorkspaceSelectionChange()` handler called with correct parameters

3. **Step 3 - Model Update Verification**
   - Action: Monitor workspace model during selection change
   - Expected: Workspace model current property updates to "personal"
   - Verification: `oWorkspaceModel.setProperty("/current", "personal")` executed

4. **Step 4 - Confirmation Message Display**
   - Action: Verify user feedback during workspace switch
   - Expected: MessageToast displays "Switched to Personal Workspace"
   - Verification: Toast message appears with correct workspace name

5. **Step 5 - Event Bus Publication**
   - Action: Monitor application event bus for workspace change event
   - Expected: Event "app/workspaceChanged" published with workspace details
   - Verification: Event payload includes workspaceId: "personal", workspaceName: "Personal Workspace"

6. **Step 6 - App Busy State Management**
   - Action: Monitor app model busy state during data refresh
   - Expected: App busy indicator shows during API call, clears after completion
   - Verification: `app.busy = true` during fetch, `app.busy = false` after response

7. **Step 7 - API Request Execution**
   - Action: Monitor network requests during workspace switch
   - Expected: GET request to `/api/v1/workspaces/personal/data` with auth header
   - Verification: Network tab shows API call with correct endpoint and authorization

8. **Step 8 - Workspace Data Processing**
   - Action: Verify workspace data is processed and logged
   - Expected: API response data logged to console, no processing errors
   - Verification: Console shows "Workspace data loaded:" with response data

9. **Step 9 - Persistence Verification**
   - Action: Check localStorage after successful workspace switch
   - Expected: selectedWorkspace key updated to "personal"
   - Verification: `localStorage.getItem("selectedWorkspace") === "personal"`

10. **Step 10 - Cross-Component Impact**
    - Action: Verify other components receive workspace change notification
    - Expected: Components subscribed to "app/workspaceChanged" event react appropriately
    - Verification: Event subscribers update their data/state based on new workspace

11. **Step 11 - Reverse Switch Testing**
    - Action: Switch back to "Default Workspace" from "Personal Workspace"
    - Expected: All switching functionality works in reverse direction
    - Verification: Same validation steps apply with workspace IDs swapped

12. **Step 12 - Error Handling Validation**
    - Action: Simulate API failure during workspace switch
    - Expected: Error message shown, workspace selection reverts gracefully
    - Verification: MessageToast shows error, app.busy = false, no data corruption

### Expected Results
- **Switch Process Criteria**:
  - Workspace switching completes within 5 seconds under normal conditions
  - Model updates reflect new workspace selection immediately
  - User feedback provided through confirmation messages
  - No UI freezing or unresponsive behavior during switch

- **Data Management Criteria**:
  - API requests made only when workspace actually changes
  - Workspace-specific data fetched and processed correctly
  - Previous workspace data cleared/updated appropriately
  - Data isolation maintained between different workspaces

- **Integration Criteria**:
  - Event bus notifications enable cross-component communication
  - Workspace preferences persist across browser sessions
  - Error conditions handled gracefully without system compromise
  - Performance remains acceptable with multiple workspace switches

### Test Postconditions
- User successfully working in selected workspace environment
- All components aware of current workspace context
- Workspace preference saved for future sessions
- Data integrity maintained across workspace boundaries

### Error Scenarios & Recovery
1. **API Timeout**: Show timeout message, maintain previous workspace selection
2. **Network Disconnection**: Queue workspace change for retry when connection restored
3. **Invalid Workspace ID**: Revert to last known good workspace, log error
4. **Concurrent Switch Requests**: Handle race conditions gracefully
5. **Event Bus Failure**: Continue with workspace switch, log communication issue

### Validation Points
- [ ] Workspace selection change triggers model update immediately
- [ ] Confirmation toast message shows correct workspace name
- [ ] Event bus publishes workspace change event with proper payload
- [ ] App busy indicator displays during API data refresh
- [ ] API request sent to correct endpoint with authentication
- [ ] Workspace data processed and logged successfully
- [ ] localStorage updated with new workspace preference
- [ ] Cross-component event notifications work correctly
- [ ] Reverse workspace switching functions properly
- [ ] Error handling prevents data corruption

### Related Test Cases
- **Depends On**: TC-UI-AGT-006 (Workspace Selector Display)
- **Triggers**: TC-UI-AGT-008 (Workspace Data Isolation), TC-UI-AGT-009 (Multi-Workspace Projects)
- **Related**: TC-UI-AGT-002 (Project Detail View), TC-BE-AGT-001 (Runtime Environment)

### Standard Compliance
- **ISO 29119-3**: Complete workspace switching functional test specification
- **SAP Standards**: SAP Fiori UX Guidelines for context switching patterns
- **Integration Standards**: Event-driven architecture patterns, RESTful API integration

---

## Test Case ID: TC-UI-AGT-008 (Migrated: TC-AA-003)
**Test Objective**: Verify navigation menu items display correctly and are accessible  
**Business Process**: Application Navigation and Menu Structure  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-008 (Legacy: TC-AA-003)
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, UI Structure
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/view/App.view.xml:68-135`
- **Controller**: `a2aAgents/frontend/webapp/controller/App.controller.ts:70-77`
- **Functions Under Test**: `_initializeSideNavigation()`, menu item rendering, i18n text binding

### Test Preconditions
1. **Application Loaded**: Main application shell rendered with SideNavigation component
2. **I18n Resources**: Internationalization resources loaded for navigation text
3. **User Authentication**: Valid user session with appropriate menu access rights
4. **Navigation Framework**: SAP TNT SideNavigation component properly initialized
5. **Menu Icons**: SAP icon font loaded and accessible

### Test Input Data
| Menu Section | Items Count | Default State | Key Pattern |
|--------------|-------------|---------------|-------------|
| Projects | 4 items | Expanded | projects, myProjects, allProjects, templates |
| Agents | 10 items | Collapsed | agents, agent0-5, agentManager, dataManager, catalogManager |
| Tools | 5 items | Collapsed | tools, agentBuilder, workflowDesigner, testing, deployment |
| Monitoring | 5 items | Collapsed | monitoring, dashboard, metrics, logs, alerts |
| Network | 1 item | Collapsed | network |
| ORD Registry | 1 item | Collapsed | ordRegistry |

### Test Procedure Steps
1. **Step 1 - Navigation Container Visibility**
   - Action: Verify SideNavigation component renders within ToolPage layout
   - Expected: Navigation panel visible on left side with proper styling
   - Verification: Element with ID "sideNavigation" exists and has class "a2a-tablet-navigation"

2. **Step 2 - Main Navigation Sections Count**
   - Action: Count top-level navigation items in the NavigationList
   - Expected: Exactly 6 main sections (Projects, Agents, Tools, Monitoring, Network, ORD Registry)
   - Verification: NavigationList contains 6 root-level NavigationListItem elements

3. **Step 3 - Projects Section Verification**
   - Action: Examine Projects navigation section structure and content
   - Expected: Projects item expanded by default with 3 sub-items
   - Verification: Projects section shows "My Projects", "All Projects", "Templates" sub-items

4. **Step 4 - Agents Section Structure**
   - Action: Verify Agents section contains all expected agent-related items
   - Expected: 9 sub-items including 6 numbered agents plus 3 manager items
   - Verification: Agent0-5, AgentManager, DataManager, CatalogManager all present

5. **Step 5 - Tools Section Content**
   - Action: Check Tools section for development and deployment tools
   - Expected: 4 sub-items for agent development workflow
   - Verification: AgentBuilder, WorkflowDesigner, Testing, Deployment items present

6. **Step 6 - Monitoring Section Items**
   - Action: Verify Monitoring section contains system monitoring tools
   - Expected: 4 sub-items for system observability
   - Verification: Dashboard, Metrics, Logs, Alerts items displayed

7. **Step 7 - Network and Registry Items**
   - Action: Check single-item sections for external integrations
   - Expected: Network and ORD Registry as standalone items
   - Verification: A2A Network and ORD Registry items present without sub-items

8. **Step 8 - Icon Verification**
   - Action: Verify each main section displays appropriate SAP icons
   - Expected: Icons match functionality (folder-blank, group, wrench, line-charts, chain-link, database)
   - Verification: CSS classes for SAP icons properly applied and visible

9. **Step 9 - Text Localization**
   - Action: Verify all menu item texts use i18n resource bindings
   - Expected: All text attributes use {i18n>key} binding pattern
   - Verification: No hardcoded text, all strings from resource bundle

10. **Step 10 - Default Selection State**
    - Action: Verify Projects section is selected by default on page load
    - Expected: Projects navigation item highlighted as active selection
    - Verification: `_initializeSideNavigation()` sets selectedKey to "projects"

11. **Step 11 - Expand/Collapse Behavior**
    - Action: Test expanding and collapsing navigation sections
    - Expected: Sections expand to show sub-items, collapse to hide them
    - Verification: Click interactions toggle expanded property correctly

12. **Step 12 - Key Attribution Consistency**
    - Action: Verify each navigation item has unique key for routing
    - Expected: All items have distinct key values matching routing patterns
    - Verification: Key values align with controller routing map

### Expected Results
- **Structure Criteria**:
  - All 6 main navigation sections render correctly
  - Sub-item counts match specification for each section
  - Hierarchical structure properly nested and displayed
  - Projects section expanded by default, others collapsed

- **Visual Criteria**:
  - SAP icons display correctly for each main section
  - Text labels load from i18n resources without errors
  - Active/selected states provide clear visual feedback
  - Navigation styling follows SAP Fiori design guidelines

- **Functionality Criteria**:
  - All navigation items are clickable and respond to interaction
  - Expand/collapse behavior works smoothly for parent items
  - Key values properly set for routing integration
  - Default selection state established on initialization

### Test Postconditions
- Navigation menu fully functional and accessible
- All menu items properly labeled and visually distinct
- Navigation state properly initialized for user interaction
- Menu structure ready for navigation testing

### Error Scenarios & Recovery
1. **I18n Resource Missing**: Display key names as fallback, log missing resources
2. **Icon Font Not Loaded**: Show placeholder icons, maintain menu functionality
3. **Navigation Framework Error**: Graceful degradation to basic menu structure
4. **Model Binding Failure**: Use static fallback menu structure
5. **CSS Styling Issues**: Maintain functional navigation with basic styling

### Validation Points
- [ ] SideNavigation component renders correctly in ToolPage layout
- [ ] All 6 main navigation sections (Projects, Agents, Tools, Monitoring, Network, ORD Registry) present
- [ ] Projects section expanded by default with 3 sub-items visible
- [ ] Agents section contains 9 sub-items (agent0-5, managers)
- [ ] Tools section shows 4 development workflow items
- [ ] Monitoring section displays 4 observability items
- [ ] Network and ORD Registry appear as standalone items
- [ ] SAP icons properly displayed for each main section
- [ ] All text labels use i18n resource bindings
- [ ] Default selection set to "projects" on initialization
- [ ] Expand/collapse functionality works for hierarchical items
- [ ] Unique key values assigned for routing integration

### Related Test Cases
- **Depends On**: TC-UI-AGT-001 (Project Workspace Initialization)
- **Triggers**: TC-UI-AGT-009 (Menu Navigation Testing), TC-UI-AGT-010 (Active Item Highlighting)
- **Related**: TC-UI-AGT-006 (Workspace Selector), TC-UI-AGT-002 (Project Detail View)

### Standard Compliance
- **ISO 29119-3**: Complete navigation menu structure test specification
- **SAP Standards**: SAP Fiori UX Guidelines for side navigation patterns
- **UI Standards**: SAP TNT NavigationList component standards, accessibility guidelines

---

## Test Case ID: TC-UI-AGT-009 (Migrated: TC-AA-004)
**Test Objective**: Verify menu navigation functionality and route handling  
**Business Process**: Application Navigation and Route Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-009 (Legacy: TC-AA-004)
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Navigation
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:124-165`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:71`
- **Functions Under Test**: `onNavigationItemSelect()`, router navigation, route parameter handling

### Test Preconditions
1. **Navigation Menu Active**: TC-UI-AGT-008 completed with all menu items visible
2. **Router Initialized**: Application router configured with all route definitions
3. **User Authentication**: Valid session with access to navigation features
4. **Target Views**: All target views/pages exist and are accessible
5. **Event Handling**: NavigationList itemSelect event properly bound

### Test Input Data
| Navigation Item | Expected Route | Parameters | Route Type |
|----------------|----------------|------------|------------|
| Projects | ProjectsList | None | Basic |
| My Projects | ProjectsList | None | Basic |
| All Projects | ProjectsList | None | Basic |
| Templates | ProjectsList | None | Basic |
| Agents | AgentsList | None | Basic |
| Agent0 | AgentDetail | {agentId: "agent0"} | Parameterized |
| Agent Builder | AgentBuilder | None | Basic |
| Testing | Testing | None | Basic |
| Dashboard | Dashboard | None | Basic |
| A2A Network | A2ANetwork | None | Basic |
| ORD Registry | ORDRegistry | None | Basic |

### Test Procedure Steps
1. **Step 1 - Navigation Event Binding**
   - Action: Verify NavigationList itemSelect event is bound to controller method
   - Expected: XML attribute itemSelect="onNavigationItemSelect" properly configured
   - Verification: Event handler method exists and is callable

2. **Step 2 - Basic Navigation Testing**
   - Action: Click "Projects" navigation item
   - Expected: Router navigates to "ProjectsList" route without parameters
   - Verification: `oRouter.navTo("ProjectsList")` called, URL updates correctly

3. **Step 3 - Sub-Item Navigation**
   - Action: Click "My Projects" sub-item under Projects section
   - Expected: Same ProjectsList route triggered as parent Projects item
   - Verification: Navigation consolidation works, route mapping consistent

4. **Step 4 - Agent Parameter Navigation**
   - Action: Click "Agent0" navigation item
   - Expected: Router navigates to AgentDetail with agentId parameter
   - Verification: `oRouter.navTo("AgentDetail", { agentId: "agent0" })` called

5. **Step 5 - Multiple Agent Navigation**
   - Action: Click each agent item (Agent1, Agent2, Agent3, Agent4, Agent5)
   - Expected: Each navigates to AgentDetail with correct agentId parameter
   - Verification: Parameter values match clicked agent keys

6. **Step 6 - Manager Agent Navigation**
   - Action: Click agentManager, dataManager, catalogManager items
   - Expected: All navigate to AgentDetail with appropriate agentId parameters
   - Verification: Manager agents treated as parameterized routes

7. **Step 7 - Tools Section Navigation**
   - Action: Click Agent Builder, Workflow Designer, Testing, Deployment items
   - Expected: Each navigates to corresponding route without parameters
   - Verification: Tools routes mapped correctly (AgentBuilder, WorkflowDesigner, Testing, Deployment)

8. **Step 8 - Monitoring Navigation**
   - Action: Click Dashboard, Metrics, Logs, Alerts items
   - Expected: Each navigates to respective monitoring route
   - Verification: Monitoring routes (Dashboard, Metrics, Logs, Alerts) work correctly

9. **Step 9 - External Integration Navigation**
   - Action: Click A2A Network and ORD Registry items
   - Expected: Navigate to A2ANetwork and ORDRegistry routes respectively
   - Verification: External integration routes function properly

10. **Step 10 - Route Parameter Validation**
    - Action: Test agent navigation with parameter inspection
    - Expected: Agent routes include correct agentId in navigation parameters
    - Verification: Parameters passed to target views for data loading

11. **Step 11 - Navigation State Management**
    - Action: Navigate through multiple menu items and verify state consistency
    - Expected: Selected navigation item updates to reflect current route
    - Verification: NavigationList selectedKey updates with route changes

12. **Step 12 - Invalid Route Handling**
    - Action: Simulate navigation item with unmapped route key
    - Expected: Navigation attempt ignored gracefully, no errors thrown
    - Verification: Route mapping check prevents invalid navigation

13. **Step 13 - Back Navigation**
    - Action: Use browser back button after menu navigation
    - Expected: Navigation selection updates to match previous route
    - Verification: Route change events update navigation state correctly

14. **Step 14 - URL Direct Access**
    - Action: Access route URLs directly and verify navigation selection
    - Expected: Navigation menu reflects current route from URL
    - Verification: Route-to-navigation mapping works in reverse

### Expected Results
- **Navigation Response Criteria**:
  - Menu item clicks trigger navigation within 200ms
  - Route changes complete successfully without errors
  - Navigation selection state updates correctly
  - URL updates reflect navigation changes immediately

- **Route Mapping Criteria**:
  - All mapped navigation items navigate to correct routes
  - Parameterized routes (agents) include proper parameter values
  - Route consolidation works for sub-items (Projects section)
  - External routes (Network, ORD Registry) function correctly

- **State Management Criteria**:
  - Selected navigation item reflects current application state
  - Navigation state persists through route changes
  - Back/forward browser navigation updates menu selection
  - Direct URL access synchronizes with navigation menu

### Test Postconditions
- All navigation items functional and routing correctly
- Navigation state synchronized with application routes
- User can navigate through all application sections
- Route parameters properly handled for dynamic content

### Error Scenarios & Recovery
1. **Route Not Found**: Log warning, maintain current navigation state
2. **Navigation Permission Denied**: Show access denied message, revert selection
3. **Router Initialization Failure**: Fallback to basic page routing
4. **Parameter Validation Error**: Use default parameters, log validation issue
5. **Target View Load Failure**: Show error page, maintain navigation consistency

### Validation Points
- [ ] NavigationList itemSelect event bound to onNavigationItemSelect handler
- [ ] Basic navigation items (Projects, Agents, Tools) route correctly
- [ ] Sub-items (My Projects, All Projects, Templates) navigate properly
- [ ] Agent items (Agent0-5) include correct agentId parameters
- [ ] Manager agents (agentManager, dataManager, catalogManager) route with parameters
- [ ] Tools section items navigate to development workflow routes
- [ ] Monitoring items route to observability views
- [ ] External integration items (Network, ORD Registry) work correctly
- [ ] Route parameter validation and passing functions properly
- [ ] Navigation state updates reflect current route
- [ ] Back navigation and direct URL access synchronize menu selection
- [ ] Invalid route attempts handled gracefully without errors

### Related Test Cases
- **Depends On**: TC-UI-AGT-008 (Navigation Menu Items)
- **Triggers**: TC-UI-AGT-010 (Active Item Highlighting), TC-UI-AGT-011 (Route Parameter Handling)
- **Related**: TC-UI-AGT-002 (Project Detail View), TC-UI-AGT-003 (Agent Builder)

### Standard Compliance
- **ISO 29119-3**: Complete navigation functionality test specification
- **SAP Standards**: SAP Fiori UX Guidelines for navigation patterns and routing
- **UI Standards**: SAP UI5 Router integration, TNT NavigationList interaction patterns

---

## Test Case ID: TC-UI-AGT-010 (Migrated: TC-AA-005)
**Test Objective**: Verify active navigation item highlighting and selection state management  
**Business Process**: Navigation Visual Feedback and State Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-010 (Legacy: TC-AA-005)
- **Test Priority**: High (P2)
- **Test Type**: UI Visual, State Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:71-121`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:56-72`
- **Functions Under Test**: `_initializeSideNavigation()`, `onRouteMatched()`, `setSelectedKey()`

### Test Preconditions
1. **Navigation Functional**: TC-UI-AGT-009 completed with navigation working correctly
2. **Router Active**: Application router initialized and route matching operational
3. **Visual Framework**: SAP TNT NavigationList styling and selection states available
4. **Route Definitions**: All application routes properly defined and accessible
5. **CSS Styling**: Navigation selection styles loaded and applied

### Test Input Data
| Test Scenario | Route | Expected Selected Key | Visual State |
|---------------|-------|---------------------|--------------|
| Default Load | / (root) | projects | Projects highlighted |
| Projects Route | ProjectsList | projects | Projects highlighted |
| Project Detail | ProjectDetail | projects | Projects highlighted |
| Agents List | AgentsList | agents | Agents highlighted |
| Agent Builder | AgentBuilder | agentBuilder | Agent Builder highlighted |
| Network Route | A2ANetwork | network | A2A Network highlighted |
| Direct URL Access | /agents | agents | Agents highlighted |
| Browser Back/Forward | Various | Route-dependent | Correct item highlighted |

### Test Procedure Steps
1. **Step 1 - Default Selection Initialization**
   - Action: Load application and verify default navigation selection
   - Expected: Projects navigation item highlighted by default
   - Verification: `_initializeSideNavigation()` sets selectedKey to "projects"

2. **Step 2 - Visual Selection Verification**
   - Action: Examine visual appearance of selected Projects item
   - Expected: Projects item displays active/selected visual styling
   - Verification: Selected item has different background, border, or text styling

3. **Step 3 - Navigation Selection Update**
   - Action: Click Agents navigation item
   - Expected: Agents item becomes highlighted, Projects item unhighlighted
   - Verification: Selection state transfers correctly between items

4. **Step 4 - Route-Based Selection Update**
   - Action: Navigate to AgentsList route programmatically
   - Expected: Navigation selection updates to Agents item
   - Verification: `onRouteMatched()` calls `setSelectedKey("agents")`

5. **Step 5 - Route-to-Navigation Mapping**
   - Action: Test route-to-navigation key mapping for all defined routes
   - Expected: Each route maps to correct navigation selection
   - Verification: ProjectsList/ProjectDetail → projects, AgentsList → agents, etc.

6. **Step 6 - Sub-Item Parent Selection**
   - Action: Navigate to ProjectDetail route
   - Expected: Parent Projects item remains highlighted, not sub-items
   - Verification: Route mapping consolidates sub-routes to parent navigation

7. **Step 7 - Agent Builder Selection**
   - Action: Navigate to Agent Builder tool
   - Expected: Agent Builder item highlighted independently
   - Verification: Tools section child item receives selection highlight

8. **Step 8 - External Route Selection**
   - Action: Navigate to A2A Network route
   - Expected: A2A Network item highlighted as standalone selection
   - Verification: External integration items properly highlighted

9. **Step 9 - Direct URL Access**
   - Action: Access route URL directly in browser (e.g., /agents)
   - Expected: Navigation menu reflects route with correct item highlighted
   - Verification: Route matching on page load updates navigation selection

10. **Step 10 - Browser Navigation Impact**
    - Action: Use browser back/forward buttons after navigation
    - Expected: Navigation selection updates with browser history changes
    - Verification: Route change events update navigation highlighting

11. **Step 11 - Unmapped Route Handling**
    - Action: Navigate to route not in route-to-navigation mapping
    - Expected: Previous selection maintained, no selection change
    - Verification: Unmapped routes don't clear or corrupt navigation state

12. **Step 12 - Selection Persistence**
    - Action: Navigate through multiple routes and verify selection consistency
    - Expected: Selection always reflects current route accurately
    - Verification: No selection state drift or incorrect highlighting

13. **Step 13 - Multiple Selection Prevention**
    - Action: Verify only one navigation item highlighted at a time
    - Expected: Single selection maintained, no multiple selections
    - Verification: NavigationList single-selection behavior enforced

14. **Step 14 - Selection Visual Feedback**
    - Action: Test hover and focus states on navigation items
    - Expected: Clear visual distinction between selected, hovered, and normal states
    - Verification: CSS states provide appropriate user feedback

### Expected Results
- **Initialization Criteria**:
  - Default selection set to "projects" on application load
  - Initial visual highlighting correctly applied
  - Navigation state consistent with default route
  - No delay in selection state establishment

- **Selection Update Criteria**:
  - Navigation selection updates within 100ms of route change
  - Visual highlighting transitions smoothly between items
  - Route-to-navigation mapping works for all defined routes
  - Selection state remains synchronized with current route

- **Visual Feedback Criteria**:
  - Selected item clearly distinguished from unselected items
  - Hover states provide immediate visual feedback
  - Focus indicators meet accessibility standards
  - Selection styling follows SAP Fiori design guidelines

### Test Postconditions
- Navigation selection accurately reflects current application state
- Visual feedback system properly guides user orientation
- Selection state maintained consistently across route changes
- User can easily identify current application context

### Error Scenarios & Recovery
1. **Route Matching Failure**: Maintain previous selection, log routing issue
2. **Navigation Reference Lost**: Re-initialize navigation, restore selection
3. **CSS Styling Missing**: Fallback to basic selection indicators
4. **Multiple Selection State**: Force single selection, correct state
5. **Selection State Corruption**: Reset to default selection, navigate to projects

### Validation Points
- [ ] Default selection set to "projects" on application initialization
- [ ] Selected navigation item displays distinct visual highlighting
- [ ] Navigation selection updates correctly when routes change
- [ ] Route-to-navigation mapping works for all defined routes (ProjectsList→projects, AgentsList→agents)
- [ ] Sub-routes (ProjectDetail) map to parent navigation item (projects)
- [ ] Tools section items (Agent Builder) receive proper selection highlighting
- [ ] External routes (A2A Network) highlight correctly as standalone items
- [ ] Direct URL access updates navigation selection appropriately
- [ ] Browser back/forward navigation updates selection state
- [ ] Unmapped routes don't corrupt navigation selection state
- [ ] Only one navigation item highlighted at any time
- [ ] Visual feedback includes appropriate hover and focus states
- [ ] Selection styling follows SAP Fiori design standards

### Related Test Cases
- **Depends On**: TC-UI-AGT-009 (Menu Navigation Testing)
- **Triggers**: TC-UI-AGT-011 (Visual State Management), TC-UI-AGT-012 (Accessibility Testing)
- **Related**: TC-UI-AGT-008 (Navigation Menu Items), TC-UI-AGT-002 (Project Detail View)

### Standard Compliance
- **ISO 29119-3**: Complete navigation state management test specification
- **SAP Standards**: SAP Fiori UX Guidelines for navigation selection and visual feedback
- **Accessibility Standards**: WCAG 2.1 visual indicators and focus management

---

## Test Case ID: TC-UI-AGT-011 (Migrated: TC-AA-006)
**Test Objective**: Verify global search functionality and search suggestions in ShellBar  
**Business Process**: Global Search and Content Discovery  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-011 (Legacy: TC-AA-006)
- **Test Priority**: High (P2)
- **Test Type**: Functional, Search Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:238-271`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:48-55`
- **Functions Under Test**: `onSearch()`, `onSearchSuggest()`, SearchManager component integration

### Test Preconditions
1. **Application Shell Loaded**: TC-UI-AGT-006 completed with ShellBar rendered
2. **Search API Available**: Backend search endpoints operational and accessible
3. **Authentication Active**: Valid user session with search permissions
4. **Event Bus Functional**: Application event bus ready for search events
5. **Search Data**: Searchable content available in system (projects, agents, etc.)

### Test Input Data
| Search Term | Expected Suggestions | Search Results | API Response |
|-------------|---------------------|----------------|--------------|
| "proj" | ["Project Alpha", "Project Beta"] | Projects list | 200 OK |
| "agent" | ["Agent Builder", "Agent Manager"] | Agents list | 200 OK |
| "test" | ["Testing Suite", "Test Data"] | Testing tools | 200 OK |
| "xyz123" | [] | No results | 200 OK (empty) |
| "a" | [] | No suggestions (< 2 chars) | No API call |
| "" | [] | No suggestions | No API call |

### Test Procedure Steps
1. **Step 1 - Search Component Visibility**
   - Action: Verify SearchManager component displays in ShellBar
   - Expected: Search field visible with placeholder text and search icon
   - Verification: Element exists with showSearch="true" and searchFieldWidth="50%"

2. **Step 2 - Search Field Placeholder**
   - Action: Check search field placeholder text displays correctly
   - Expected: Localized placeholder text from i18n resources
   - Verification: Placeholder attribute bound to {i18n>searchPlaceholder}

3. **Step 3 - Search Input Response**
   - Action: Type "project" in search field
   - Expected: Input accepted, characters display correctly
   - Verification: Search field value updates as user types

4. **Step 4 - Search Suggestions Trigger**
   - Action: Type 2+ characters in search field
   - Expected: onSearchSuggest method triggered after brief delay
   - Verification: API call made to `/api/v1/search/suggest?q=project`

5. **Step 5 - Suggestions API Call**
   - Action: Monitor network requests during search input
   - Expected: GET request to search suggest endpoint with query parameter
   - Verification: Request includes Authorization Bearer token header

6. **Step 6 - Suggestions Display**
   - Action: Verify search suggestions appear below search field
   - Expected: Dropdown with relevant suggestions based on API response
   - Verification: Suggestions mapped from API response and displayed

7. **Step 7 - Suggestion Selection**
   - Action: Click on a search suggestion from dropdown
   - Expected: Search field populated with selected suggestion
   - Verification: Selected suggestion becomes search field value

8. **Step 8 - Search Execution**
   - Action: Press Enter or click search button with query "project"
   - Expected: onSearch method triggered with search query
   - Verification: Event published to app event bus with search parameters

9. **Step 9 - Global Search Event**
   - Action: Verify global search event published correctly
   - Expected: Event "app/globalSearch" published with query parameter
   - Verification: Event payload includes {query: "project"} data

10. **Step 10 - Short Query Handling**
    - Action: Type single character "a" in search field
    - Expected: No suggestions API call made (< 2 characters)
    - Verification: onSearchSuggest returns early, no network request

11. **Step 11 - Empty Search Handling**
    - Action: Clear search field and attempt search
    - Expected: Empty search handled gracefully
    - Verification: No errors, appropriate empty state handling

12. **Step 12 - Search API Error Handling**
    - Action: Simulate search API failure (disconnect network)
    - Expected: Error logged, suggestions not shown, search continues
    - Verification: Console shows "Search suggest failed" error message

13. **Step 13 - Search Performance**
    - Action: Type rapidly in search field and measure response time
    - Expected: Suggestions appear within 500ms of API response
    - Verification: Reasonable debouncing and response performance

14. **Step 14 - Cross-Component Search Integration**
    - Action: Subscribe to global search event in test component
    - Expected: Other components can listen and respond to search events
    - Verification: Event bus propagates search events to subscribers

### Expected Results
- **Search Interface Criteria**:
  - Search field displays prominently in ShellBar with 50% width
  - Placeholder text provides clear guidance to users
  - Search input responsive and accepts all character types
  - Search suggestions appear after 2+ character input

- **Suggestions Functionality Criteria**:
  - API calls made only when necessary (2+ characters)
  - Suggestions API response processed and displayed correctly
  - Network errors handled gracefully without breaking search
  - Suggestion selection populates search field appropriately

- **Search Integration Criteria**:
  - Search execution publishes global event for system integration
  - Event bus communication enables cross-component search response
  - Search authentication includes proper Bearer token
  - Performance remains acceptable with rapid input

### Test Postconditions
- Global search functionality operational and integrated
- Search suggestions provide helpful user guidance
- Cross-component search communication established
- Error handling prevents search failures from breaking application

### Error Scenarios & Recovery
1. **Search API Unavailable**: Continue without suggestions, log error
2. **Authentication Expired**: Refresh token or redirect to login
3. **Network Timeout**: Show timeout message, allow retry
4. **Malformed API Response**: Handle gracefully, show no suggestions
5. **Event Bus Failure**: Log error, search continues without event propagation

### Validation Points
- [ ] SearchManager component displays in ShellBar with correct configuration
- [ ] Search field placeholder text loads from i18n resources
- [ ] Search input accepts and displays user typing correctly
- [ ] Search suggestions triggered after 2+ characters typed
- [ ] API call made to `/api/v1/search/suggest` with query parameter
- [ ] Authorization Bearer token included in search API requests
- [ ] Search suggestions displayed in dropdown from API response
- [ ] Suggestion selection populates search field value
- [ ] Search execution (Enter/click) triggers onSearch method
- [ ] Global search event published to app event bus with query
- [ ] Short queries (< 2 chars) don't trigger API calls
- [ ] Empty search handled without errors
- [ ] Search API failures logged and handled gracefully
- [ ] Search performance meets 500ms response time requirement

### Related Test Cases
- **Depends On**: TC-UI-AGT-006 (Workspace Selector Display)
- **Triggers**: TC-UI-AGT-012 (Search Results Integration), TC-UI-AGT-013 (Cross-Component Communication)
- **Related**: TC-UI-AGT-008 (Navigation Menu Items), TC-BE-AGT-001 (Runtime Environment)

### Standard Compliance
- **ISO 29119-3**: Complete search functionality test specification
- **SAP Standards**: SAP Fiori UX Guidelines for search patterns and user experience
- **UI Standards**: SAP F ShellBar SearchManager integration, search suggestion patterns

---

## Test Case ID: TC-UI-AGT-012 (Migrated: TC-AA-007)
**Test Objective**: Verify notification system functionality and notification display  
**Business Process**: User Notification Management and Alert System  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-012 (Legacy: TC-AA-007)
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:176-200`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:19-21`
- **Fragment**: `com.sap.a2a.portal.view.NotificationPopover`
- **Functions Under Test**: `onNotificationPress()`, notification model management, popover loading

### Test Preconditions
1. **Application Shell Loaded**: TC-UI-AGT-006 completed with ShellBar rendered
2. **Notification API Available**: Backend notification endpoints operational
3. **Authentication Active**: Valid user session with notification access
4. **Notification Data**: Test notifications available in system
5. **Fragment Resources**: NotificationPopover fragment exists and is accessible

### Test Input Data
| Test Scenario | Notification Count | API Response | Expected Display |
|---------------|-------------------|--------------|------------------|
| Default State | 3 | N/A | Badge shows "3" |
| No Notifications | 0 | [] | No badge or "0" |
| New Notifications | 5 | [{id: 1, title: "Test", message: "..."}] | Badge shows "5" |
| API Error | 3 | 500 Error | Badge shows "3", error toast |
| Empty Response | 0 | [] | No badge, empty popover |

### Test Procedure Steps
1. **Step 1 - Notification Icon Visibility**
   - Action: Verify notification bell icon displays in ShellBar
   - Expected: Notification icon visible with showNotifications="true"
   - Verification: Notification bell icon present and clickable

2. **Step 2 - Notification Badge Display**
   - Action: Check notification count badge displays correctly
   - Expected: Badge shows "3" (default count from notification model)
   - Verification: Badge bound to {notification>/count} shows correct number

3. **Step 3 - Notification Model Initialization**
   - Action: Verify notification model initialized with default values
   - Expected: Notification model contains count: 3, items: []
   - Verification: Model created and bound to view correctly

4. **Step 4 - Notification Click Response**
   - Action: Click notification icon in ShellBar
   - Expected: onNotificationPress method triggered
   - Verification: Event handler called with correct event source

5. **Step 5 - Notifications API Call**
   - Action: Monitor network requests when notification icon clicked
   - Expected: GET request to `/api/v1/notifications` with auth header
   - Verification: API call includes Authorization Bearer token

6. **Step 6 - Notification Data Loading**
   - Action: Verify notification data loaded from API response
   - Expected: Notification items updated in model from API response
   - Verification: Model property "/items" set with API response data

7. **Step 7 - Notification Popover Loading**
   - Action: Verify NotificationPopover fragment loads correctly
   - Expected: Fragment loaded and cached for subsequent use
   - Verification: `loadFragment()` called with correct fragment name

8. **Step 8 - Popover Display**
   - Action: Verify notification popover opens correctly
   - Expected: Popover opens positioned by notification icon
   - Verification: `openBy()` called with event source, popover visible

9. **Step 9 - Popover Content**
   - Action: Check popover displays notification items correctly
   - Expected: All notifications from API response displayed in popover
   - Verification: Popover content matches notification model data

10. **Step 10 - Notification Error Handling**
    - Action: Simulate notification API failure
    - Expected: Error handled gracefully, toast message shown
    - Verification: MessageToast displays "notificationLoadError" text

11. **Step 11 - Popover Caching**
    - Action: Click notification icon multiple times
    - Expected: Popover fragment cached, not reloaded each time
    - Verification: Fragment loaded once, reused for subsequent opens

12. **Step 12 - Notification Count Update**
    - Action: Simulate notification count change in model
    - Expected: Badge updates to reflect new count
    - Verification: Count changes trigger badge update

13. **Step 13 - Empty Notifications Handling**
    - Action: Test with API returning empty notifications array
    - Expected: Popover shows empty state, no errors
    - Verification: Empty array handled gracefully in popover

14. **Step 14 - Notification Persistence**
    - Action: Navigate between views and return to main page
    - Expected: Notification state maintained across navigation
    - Verification: Count and items persist through route changes

### Expected Results
- **Notification Display Criteria**:
  - Notification icon displays prominently in ShellBar
  - Badge shows correct count from notification model
  - Badge updates dynamically when count changes
  - No badge shown when count is 0 or undefined

- **Interaction Criteria**:
  - Notification icon responds to click within 100ms
  - API call made within 500ms of notification icon click
  - Popover opens within 1 second of successful API response
  - Popover positioned correctly relative to notification icon

- **Data Management Criteria**:
  - Notification API includes proper authentication
  - API response data correctly updates notification model
  - Fragment caching prevents unnecessary reloads
  - Error states handled without breaking functionality

### Test Postconditions
- Notification system fully functional and responsive
- Notification data correctly loaded and displayed
- Error handling prevents system failures
- User can access and view all available notifications

### Error Scenarios & Recovery
1. **Notification API Unavailable**: Show error toast, maintain current count
2. **Authentication Expired**: Refresh token or redirect to login
3. **Fragment Load Failure**: Show basic notification list, log error
4. **Malformed API Response**: Handle gracefully, show empty state
5. **Popover Display Error**: Fall back to basic notification display

### Validation Points
- [ ] Notification icon displays in ShellBar with proper visibility
- [ ] Notification badge shows correct count from model
- [ ] Notification model initialized with default values (count: 3, items: [])
- [ ] Notification icon click triggers onNotificationPress method
- [ ] API call made to `/api/v1/notifications` with Authorization header
- [ ] Notification data loaded and updates model "/items" property
- [ ] NotificationPopover fragment loads correctly on first use
- [ ] Popover opens positioned by notification icon source
- [ ] Popover displays notification items from model data
- [ ] API error handling shows appropriate error message
- [ ] Fragment caching prevents multiple loads of same popover
- [ ] Badge updates dynamically when notification count changes
- [ ] Empty notifications array handled gracefully
- [ ] Notification state persists across navigation changes

### Related Test Cases
- **Depends On**: TC-UI-AGT-006 (Workspace Selector Display)
- **Triggers**: TC-UI-AGT-013 (Notification Detail View), TC-UI-AGT-014 (Real-time Notifications)
- **Related**: TC-UI-AGT-011 (Global Search), TC-BE-AGT-001 (Runtime Environment)

### Standard Compliance
- **ISO 29119-3**: Complete notification system test specification
- **SAP Standards**: SAP Fiori UX Guidelines for notification patterns and user alerts
- **UI Standards**: SAP F ShellBar notification integration, popover display patterns

---

## Test Case ID: TC-UI-AGT-013 (Migrated: TC-AA-008)
**Test Objective**: Verify user menu dropdown functionality and user profile management  
**Business Process**: User Profile Management and System Access  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-013 (Legacy: TC-AA-008)
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:206-236`
- **View File**: `a2aAgents/frontend/webapp/view/App.view.xml:37-46`
- **Functions Under Test**: `onUserSettingsPress()`, `onSystemSettingsPress()`, `onHelpPress()`, `onLogoutPress()`

### Test Preconditions
1. **Application Shell Loaded**: TC-UI-AGT-006 completed with ShellBar rendered
2. **User Authentication**: Valid user session with profile data loaded
3. **Router Initialized**: Application router ready for settings navigation
4. **User Model**: User model initialized with profile information
5. **External Links**: Help documentation URL accessible

### Test Input Data
| Menu Item | Action Type | Expected Result | Verification |
|-----------|-------------|----------------|--------------|
| User Settings | Navigation | Route to UserSettings | Router.navTo() called |
| System Settings | Navigation | Route to SystemSettings | Router.navTo() called |
| Help | External Link | Open help.sap.com | window.open() called |
| Logout | API + Redirect | Logout API + redirect to login | fetch() + window.location |

### Test Procedure Steps
1. **Step 1 - User Avatar Display**
   - Action: Verify user avatar displays in ShellBar profile section
   - Expected: Avatar shows user initials with Accent6 background color
   - Verification: Avatar bound to {user>/initials} with correct styling

2. **Step 2 - User Menu Access**
   - Action: Click user avatar to open user menu dropdown
   - Expected: Menu dropdown opens with 4 menu items
   - Verification: Menu contains User Settings, System Settings, Help, Logout items

3. **Step 3 - Menu Item Icons and Text**
   - Action: Verify each menu item displays correct icon and localized text
   - Expected: All items show appropriate SAP icons and i18n text
   - Verification: Icons and text bound correctly from i18n resources

4. **Step 4 - User Settings Navigation**
   - Action: Click "User Settings" menu item
   - Expected: Navigation to UserSettings route triggered
   - Verification: `onUserSettingsPress()` calls `router.navTo("UserSettings")`

5. **Step 5 - System Settings Navigation**
   - Action: Click "System Settings" menu item  
   - Expected: Navigation to SystemSettings route triggered
   - Verification: `onSystemSettingsPress()` calls `router.navTo("SystemSettings")`

6. **Step 6 - Help Documentation Access**
   - Action: Click "Help" menu item
   - Expected: External help documentation opens in new tab
   - Verification: `onHelpPress()` calls `window.open("https://help.sap.com/a2a-portal", "_blank")`

7. **Step 7 - Logout Process Initiation**
   - Action: Click "Logout" menu item
   - Expected: Logout process begins with API call and cleanup
   - Verification: `onLogoutPress()` triggered with async logout flow

8. **Step 8 - Logout API Call**
   - Action: Monitor network requests during logout process
   - Expected: POST request to `/api/v1/auth/logout` with auth token
   - Verification: Request includes Authorization Bearer header

9. **Step 9 - Local Storage Cleanup**
   - Action: Verify local storage cleared during logout
   - Expected: authToken removed from localStorage
   - Verification: `localStorage.removeItem("authToken")` called

10. **Step 10 - Logout Redirect**
    - Action: Verify redirect to login page after logout
    - Expected: Browser redirected to /login page
    - Verification: `window.location.href = "/login"` executed

11. **Step 11 - Logout Error Handling**
    - Action: Simulate logout API failure
    - Expected: Error logged, cleanup still performed
    - Verification: Console shows "Logout failed" error, cleanup continues

12. **Step 12 - Menu Accessibility**
    - Action: Test menu keyboard navigation and screen reader support
    - Expected: Menu items accessible via keyboard and ARIA labels
    - Verification: Tab navigation works, appropriate ARIA attributes present

13. **Step 13 - Menu State Management**
    - Action: Open menu, navigate away, return to page
    - Expected: Menu state resets appropriately
    - Verification: Menu closes properly, no state persistence issues

14. **Step 14 - User Profile Data Display**
    - Action: Verify user model data correctly displayed in avatar
    - Expected: Avatar initials match user model data
    - Verification: Avatar reflects current user profile information

### Expected Results
- **Menu Display Criteria**:
  - User avatar displays prominently in ShellBar with correct initials
  - Menu dropdown opens smoothly when avatar clicked
  - All 4 menu items visible with correct icons and localized text
  - Menu positioning appropriate relative to avatar

- **Navigation Criteria**:
  - User Settings and System Settings navigate to correct routes
  - Navigation completes within 500ms
  - Menu closes after navigation selection
  - Router integration works correctly

- **External Integration Criteria**:
  - Help link opens external documentation in new tab
  - Help URL loads correctly without popup blocking
  - External link maintains user session in original tab
  - Window focus handling appropriate

- **Logout Security Criteria**:
  - Logout API call includes proper authentication
  - Local storage completely cleared of sensitive data
  - Redirect to login page occurs after cleanup
  - Error conditions handled without security compromise

### Test Postconditions
- User menu fully functional with all navigation options working
- User profile information correctly displayed
- Logout process secure and complete
- External help resources accessible

### Error Scenarios & Recovery
1. **Navigation Route Missing**: Show error message, maintain menu functionality
2. **External Help URL Unavailable**: Handle popup blocking, provide alternative
3. **Logout API Failure**: Complete local cleanup, log error for investigation
4. **Profile Data Missing**: Show generic avatar, log data loading issue
5. **Menu Display Error**: Fallback to basic profile display

### Validation Points
- [ ] User avatar displays in ShellBar with user initials from model
- [ ] Menu dropdown opens when user avatar clicked
- [ ] All 4 menu items display with correct SAP icons
- [ ] Menu item text loads from i18n resources correctly
- [ ] User Settings menu item navigates to UserSettings route
- [ ] System Settings menu item navigates to SystemSettings route
- [ ] Help menu item opens external documentation in new tab
- [ ] Logout menu item triggers logout API call with auth token
- [ ] Local storage authToken removed during logout process
- [ ] Browser redirected to /login after successful logout
- [ ] Logout errors logged and handled gracefully
- [ ] Menu accessibility features work for keyboard navigation
- [ ] User profile data correctly bound to avatar display

### Related Test Cases
- **Depends On**: TC-UI-AGT-006 (Workspace Selector Display)
- **Triggers**: TC-UI-AGT-014 (User Settings View), TC-UI-AGT-015 (System Settings View)
- **Related**: TC-UI-AGT-012 (Notification System), TC-BE-AGT-009 (User Logout)

### Standard Compliance
- **ISO 29119-3**: Complete user menu functionality test specification  
- **SAP Standards**: SAP Fiori UX Guidelines for user profile and menu patterns
- **Accessibility Standards**: WCAG 2.1 keyboard navigation and screen reader support

---

## Test Case ID: TC-UI-AGT-014 (Migrated: TC-AA-009)
**Test Objective**: Verify theme persistence functionality and user preference management  
**Business Process**: User Experience Customization and Preference Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-014 (Legacy: TC-AA-009)
- **Test Priority**: Medium (P3)
- **Test Type**: Functional, User Preferences
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:69-94,369-451`
- **Settings Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/settingsDialog.fragment.xml:38-47`
- **Functions Under Test**: `_applyTheme()`, `onSaveSettings()`, `onResetSettings()`, settings model initialization

### Test Preconditions
1. **Application Shell Loaded**: TC-UI-AGT-013 completed with user menu functional
2. **Settings Fragment**: Settings dialog fragment accessible and loadable
3. **Local Storage Available**: Browser localStorage functionality operational
4. **Theme Resources**: SAP theme resources available and accessible
5. **Settings Model**: Settings model initialized with default or stored values

### Test Input Data
| Theme Option | Theme ID | Visual Characteristics | Storage Key |
|--------------|----------|----------------------|-------------|
| SAP Horizon | sap_horizon | Modern light theme | userTheme |
| SAP Fiori 3 | sap_fiori_3 | Classic light theme | userTheme |
| SAP Fiori 3 Dark | sap_fiori_3_dark | Dark theme | userTheme |
| SAP Belize | sap_belize | Legacy light theme | userTheme |

### Test Procedure Steps
1. **Step 1 - Default Theme Initialization**
   - Action: Load application and verify default theme applied
   - Expected: SAP Horizon theme applied by default
   - Verification: `sap.ui.getCore().getConfiguration().getTheme()` returns "sap_horizon"

2. **Step 2 - Settings Model Initialization**
   - Action: Verify settings model initialized with theme preference
   - Expected: Settings model contains theme from localStorage or default
   - Verification: Model property "/theme" matches stored or default value

3. **Step 3 - Settings Dialog Access**
   - Action: Navigate to settings via user menu System Settings
   - Expected: Settings dialog opens with theme selector visible
   - Verification: Theme Select control displays with current theme selected

4. **Step 4 - Theme Options Display**
   - Action: Open theme selector dropdown in settings dialog
   - Expected: All 4 theme options visible (Horizon, Fiori 3, Fiori 3 Dark, Belize)
   - Verification: Select items match expected theme list

5. **Step 5 - Theme Selection Change**
   - Action: Select "SAP Fiori 3 Dark" from theme dropdown
   - Expected: Selection updates in dropdown, model reflects change
   - Verification: Settings model "/theme" property updates to "sap_fiori_3_dark"

6. **Step 6 - Theme Application on Save**
   - Action: Click "Save" button in settings dialog
   - Expected: Dark theme applied immediately, localStorage updated
   - Verification: UI changes to dark theme, localStorage contains new theme

7. **Step 7 - Theme Persistence Verification**
   - Action: Refresh browser page or reload application
   - Expected: Dark theme persists after reload
   - Verification: Theme remains "sap_fiori_3_dark" on application restart

8. **Step 8 - Multiple Theme Changes**
   - Action: Change theme multiple times: Horizon → Belize → Fiori 3
   - Expected: Each change applies immediately and persists
   - Verification: Visual theme changes match selection, localStorage updates

9. **Step 9 - Settings Reset Functionality**
   - Action: Click "Reset" button in settings dialog
   - Expected: Theme reverts to default "sap_horizon"
   - Verification: Theme selector shows Horizon, UI applies default theme

10. **Step 10 - Cancel Changes**
    - Action: Change theme selection, then click "Cancel"
    - Expected: Theme change discarded, original theme maintained
    - Verification: Settings dialog closes, no theme change applied

11. **Step 11 - Cross-Session Persistence**
    - Action: Set theme, close browser, reopen application
    - Expected: Theme preference maintained across browser sessions
    - Verification: Stored theme applies on fresh browser launch

12. **Step 12 - Theme Performance**
    - Action: Measure theme application time during changes
    - Expected: Theme changes apply within 500ms
    - Verification: Visual theme transition completes quickly

13. **Step 13 - Error Handling**
    - Action: Simulate localStorage unavailable or theme resource missing
    - Expected: Graceful fallback to default theme, no errors
    - Verification: Application continues with default theme, errors logged

14. **Step 14 - Settings Integration**
    - Action: Verify theme setting integrates with other preference settings
    - Expected: Theme changes save with other settings atomically
    - Verification: All settings saved together, consistent state maintained

### Expected Results
- **Theme Application Criteria**:
  - Default theme (SAP Horizon) applied on first application load
  - Theme changes apply immediately without page refresh
  - Visual appearance changes correctly for each theme option
  - Theme application completes within 500ms

- **Persistence Criteria**:
  - Theme preference stored in localStorage with key "userTheme"
  - Theme persists across browser sessions and page refreshes
  - Settings model correctly initialized from stored preferences
  - Multiple theme changes handled correctly

- **User Interface Criteria**:
  - Settings dialog displays all available theme options
  - Current theme selection highlighted in dropdown
  - Save/Reset/Cancel operations work as expected
  - Visual feedback provided for successful theme changes

### Test Postconditions
- Theme persistence system fully functional
- User can customize and maintain preferred theme
- Settings integrated with overall preferences management
- No performance degradation from theme changes

### Error Scenarios & Recovery
1. **Theme Resource Missing**: Fall back to default theme, log warning
2. **localStorage Unavailable**: Use session-only theme, continue operation
3. **Invalid Theme ID**: Reset to default theme, show user message
4. **Theme Application Failure**: Retry with default theme, log error
5. **Settings Dialog Error**: Maintain current theme, allow manual theme setting

### Validation Points
- [ ] Default theme "sap_horizon" applied on application initialization
- [ ] Settings model initialized with theme from localStorage or default
- [ ] Settings dialog theme selector displays all 4 theme options
- [ ] Theme selection changes update settings model correctly
- [ ] Save button applies new theme using sap.ui.getCore().applyTheme()
- [ ] Theme preference stored in localStorage with key "userTheme"
- [ ] Theme persists across page refreshes and browser sessions
- [ ] Multiple theme changes handled correctly and persist
- [ ] Reset functionality restores default theme settings
- [ ] Cancel operation discards unsaved theme changes
- [ ] Theme changes apply within 500ms performance requirement
- [ ] Error conditions handled gracefully with fallback to default
- [ ] Theme setting integrates properly with other user preferences
- [ ] Visual appearance changes correctly for each theme option

### Related Test Cases
- **Depends On**: TC-UI-AGT-013 (User Menu Dropdown)
- **Triggers**: TC-UI-AGT-015 (Settings Dialog Integration), TC-UI-AGT-016 (User Preferences Management)
- **Related**: TC-UI-AGT-006 (Workspace Selector), TC-UI-AGT-012 (Notification System)

### Standard Compliance
- **ISO 29119-3**: Complete theme preference management test specification
- **SAP Standards**: SAP Fiori UX Guidelines for theme consistency and user customization
- **UI Standards**: SAP UI5 theming standards and user preference patterns

---

## Test Case ID: TC-UI-AGT-015 (Migrated: TC-AA-010)
**Test Objective**: Verify keyboard shortcuts functionality and accessibility navigation  
**Business Process**: Keyboard Accessibility and User Productivity  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-015 (Legacy: TC-AA-010)
- **Test Priority**: High (P2)
- **Test Type**: Accessibility, Keyboard Navigation
- **Execution Method**: Manual
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/App.controller.ts:32-33,456-598`
- **Functions Under Test**: `_initializeKeyboardShortcuts()`, `_handleKeyboardShortcut()`, shortcut handlers

### Test Preconditions
1. **Application Shell Loaded**: TC-UI-AGT-014 completed with theme system functional
2. **Keyboard Event Handling**: Browser keyboard event support operational
3. **Navigation System**: All navigation components functional and accessible
4. **Search Component**: Global search system operational
5. **Dialog System**: Settings and notification dialogs available

### Test Input Data
| Shortcut | Key Combination | Expected Action | Context |
|----------|----------------|-----------------|---------|
| Focus Search | Ctrl+K (Cmd+K on Mac) | Focus search field | Global |
| Navigate Home | Ctrl+H (Cmd+H on Mac) | Navigate to Projects | Global |
| Open Settings | Ctrl+, (Cmd+, on Mac) | Open settings dialog | Global |
| Show Help | Ctrl+/ (Cmd+/) | Show keyboard shortcuts | Global |
| Projects | Alt+1 | Navigate to Projects section | Global |
| Agents | Alt+2 | Navigate to Agents section | Global |
| Tools | Alt+3 | Navigate to Tools section | Global |
| Monitoring | Alt+4 | Navigate to Monitoring section | Global |
| Notifications | Alt+N | Open notifications panel | Global |
| Close Dialog | ESC | Close active dialog | Dialog context |
| Show Shortcuts | Shift+? | Display shortcuts help | Global |

### Test Procedure Steps
1. **Step 1 - Keyboard Shortcuts Initialization**
   - Action: Load application and verify keyboard event listener attached
   - Expected: Keyboard shortcuts system initialized on app startup
   - Verification: Console shows "Keyboard shortcuts initialized" message

2. **Step 2 - Search Focus Shortcut (Ctrl+K)**
   - Action: Press Ctrl+K anywhere in the application
   - Expected: Search field receives focus, confirmation toast appears
   - Verification: Search input focused, toast shows "Search focused (Ctrl+K)"

3. **Step 3 - Home Navigation Shortcut (Ctrl+H)**
   - Action: Press Ctrl+H from any application page
   - Expected: Navigate to Projects page, confirmation toast appears
   - Verification: Router navigates to ProjectsList, toast shows "Navigated to Projects (Ctrl+H)"

4. **Step 4 - Settings Shortcut (Ctrl+,)**
   - Action: Press Ctrl+, to open settings
   - Expected: System settings dialog opens, confirmation toast appears
   - Verification: Settings dialog displays, toast shows "Opening Settings (Ctrl+,)"

5. **Step 5 - Help Shortcut (Ctrl+/)**
   - Action: Press Ctrl+/ to show keyboard shortcuts help
   - Expected: Help message displays with available shortcuts
   - Verification: Toast shows comprehensive shortcuts list for 5 seconds

6. **Step 6 - Section Navigation Shortcuts (Alt+1-4)**
   - Action: Test Alt+1, Alt+2, Alt+3, Alt+4 shortcuts
   - Expected: Navigate to Projects, Agents, Tools, Monitoring respectively
   - Verification: Navigation selection updates, routes change correctly

7. **Step 7 - Notifications Shortcut (Alt+N)**
   - Action: Press Alt+N to open notifications
   - Expected: Notifications panel opens, confirmation toast appears
   - Verification: Notification popover displays, toast shows "Notifications opened (Alt+N)"

8. **Step 8 - Dialog Close Shortcut (ESC)**
   - Action: Open a dialog, then press ESC key
   - Expected: Dialog closes, confirmation toast appears
   - Verification: Dialog dismissed, toast shows "Dialog closed (ESC)"

9. **Step 9 - Help Display Shortcut (Shift+?)**
   - Action: Press Shift+? to show shortcuts help
   - Expected: Same help message as Ctrl+/ displays
   - Verification: Comprehensive shortcuts list shown for 5 seconds

10. **Step 10 - Input Field Behavior**
    - Action: Focus search field, type text, then press shortcut keys
    - Expected: Shortcuts ignored when typing in input fields
    - Verification: Text entry continues normally, shortcuts don't interfere

11. **Step 11 - Mac Compatibility**
    - Action: Test Cmd+K, Cmd+H, Cmd+, shortcuts on Mac
    - Expected: Command key works same as Ctrl key
    - Verification: All Cmd shortcuts function identically to Ctrl

12. **Step 12 - Modifier Key Combinations**
    - Action: Test shortcuts with incorrect modifier combinations
    - Expected: Invalid combinations ignored, no unintended actions
    - Verification: Only correct key combinations trigger shortcuts

13. **Step 13 - Event Propagation**
    - Action: Test shortcuts in different application contexts
    - Expected: Shortcuts work globally except when typing in inputs
    - Verification: Shortcuts function from any view or dialog state

14. **Step 14 - Performance Impact**
    - Action: Rapidly press various shortcuts and monitor performance
    - Expected: No performance degradation, responsive shortcut handling
    - Verification: Shortcuts respond within 100ms, no UI lag

### Expected Results
- **Initialization Criteria**:
  - Keyboard event listener attached on application startup
  - Shortcuts system initializes without errors
  - Console confirmation message appears
  - No interference with normal keyboard usage

- **Shortcut Functionality Criteria**:
  - All 11 defined shortcuts function correctly
  - Search focus (Ctrl+K) works from any application state
  - Navigation shortcuts (Alt+1-4) change views appropriately
  - Settings shortcut (Ctrl+,) opens system settings
  - Dialog close (ESC) works with any open dialog

- **Accessibility Criteria**:
  - Shortcuts follow standard conventions (Ctrl+K for search)
  - Mac command key support for cross-platform compatibility
  - Input field detection prevents shortcut interference
  - Visual feedback provided for all shortcut actions

- **User Experience Criteria**:
  - Confirmation toasts provide clear feedback
  - Help system (Ctrl+/ or Shift+?) shows all available shortcuts
  - Shortcuts enhance productivity without disrupting workflow
  - Error handling prevents unintended actions

### Test Postconditions
- Keyboard shortcuts system fully operational and accessible
- All major application functions accessible via keyboard
- User productivity enhanced through quick navigation
- Accessibility compliance improved for keyboard users

### Error Scenarios & Recovery
1. **Event Listener Failure**: Log error, continue without shortcuts
2. **Invalid Key Combination**: Ignore silently, no action taken
3. **Target Element Missing**: Log warning, show error toast
4. **Dialog State Conflicts**: Handle gracefully, maintain application state
5. **Performance Issues**: Throttle events, maintain responsiveness

### Validation Points
- [ ] Keyboard shortcuts initialized on application startup
- [ ] Ctrl+K (Cmd+K) focuses search field with confirmation
- [ ] Ctrl+H (Cmd+H) navigates to Projects with confirmation
- [ ] Ctrl+, (Cmd+,) opens settings dialog with confirmation
- [ ] Ctrl+/ (Cmd+/) shows comprehensive shortcuts help
- [ ] Alt+1-4 navigate to correct application sections
- [ ] Alt+N opens notifications panel with confirmation
- [ ] ESC key closes active dialogs with confirmation
- [ ] Shift+? displays keyboard shortcuts help
- [ ] Input field detection prevents shortcut interference
- [ ] Mac command key compatibility works correctly
- [ ] Invalid modifier combinations ignored appropriately
- [ ] Shortcuts work globally across all application views
- [ ] Performance remains responsive with rapid shortcut use

### Related Test Cases
- **Depends On**: TC-UI-AGT-014 (Theme Persistence)
- **Triggers**: TC-UI-AGT-016 (Accessibility Compliance), TC-UI-AGT-017 (User Productivity Features)
- **Related**: TC-UI-AGT-011 (Global Search), TC-UI-AGT-013 (User Menu Dropdown), TC-UI-AGT-012 (Notification System)

### Standard Compliance
- **ISO 29119-3**: Complete keyboard accessibility test specification
- **SAP Standards**: SAP Fiori UX Guidelines for keyboard navigation and shortcuts
- **Accessibility Standards**: WCAG 2.1 keyboard accessibility and navigation standards

---

## Test Case ID: TC-UI-AGT-016
**Test Objective**: Verify project grid view display and interaction functionality  
**Business Process**: Project Management and Visualization  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-016
- **Test Priority**: High (P1)
- **Test Type**: Functional, UI/UX
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:106-117`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:121-125`
- **Functions Under Test**: `onViewChange()`, GridContainer rendering

### Test Preconditions
1. **Authentication**: User authenticated with project access permissions
2. **Project Data**: Multiple test projects exist in the system
3. **UI State**: Projects view loaded and initialized
4. **View Mode**: Grid/tiles view mode selected
5. **Screen Resolution**: Minimum 1024x768 for proper grid layout

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| View Mode | tiles | String | SegmentedButton selection |
| Grid Layout | 4 columns, 6 rows | Layout | GridContainerSettings |
| Item Size | 20rem x 5rem | Dimensions | CSS Configuration |
| Gap Size | 1rem | Spacing | Layout Configuration |
| Sample Projects | 12 test projects | Array | Mock Data |

### Test Procedure Steps
1. **Step 1 - Grid View Activation**
   - Action: Click tiles view button in segmented control (grid icon)
   - Expected: View mode switches to tiles, GridContainer becomes visible
   - Verification: `${view>/viewMode} === 'tiles'` evaluates to true

2. **Step 2 - Grid Layout Validation**
   - Action: Observe project tiles arrangement in grid container
   - Expected: Projects displayed in 4-column responsive grid layout
   - Verification: GridContainer shows proper column count and spacing

3. **Step 3 - Project Tile Rendering**
   - Action: Verify individual project tiles display complete information
   - Expected: Each tile shows project name, description, status, and actions
   - Verification: Project data binds correctly to tile template

4. **Step 4 - Grid Responsiveness**
   - Action: Resize browser window to test responsive behavior
   - Expected: Grid adapts to available screen space, maintains readability
   - Verification: Tiles reflow appropriately at different breakpoints

5. **Step 5 - Tile Interaction**
   - Action: Click on project tiles to test selection and navigation
   - Expected: Tile click triggers `onProjectPress()` with correct context
   - Verification: Selected project ID displayed in message toast

6. **Step 6 - View Mode Toggle**
   - Action: Switch between grid and table views repeatedly
   - Expected: Views toggle smoothly without data loss or UI glitches
   - Verification: View state persists, data binding remains intact

7. **Step 7 - Empty State Handling**
   - Action: Clear all projects or filter to empty result set
   - Expected: Empty state message and create project prompt displayed
   - Verification: "No projects" message visible when grid is empty

8. **Step 8 - Grid Performance**
   - Action: Load grid with large number of projects (50+ items)
   - Expected: Grid renders within 2 seconds without UI blocking
   - Verification: No performance degradation or scrolling issues

### Expected Results
- **Grid Display Criteria**:
  - Projects displayed in clean, organized tile layout
  - 4-column grid on desktop, responsive on smaller screens  
  - Consistent tile sizing and spacing (20rem x 5rem, 1rem gap)
  - All project information visible and readable

- **Interaction Criteria**:
  - Tiles respond to hover and click interactions
  - Project selection triggers appropriate navigation/action
  - View mode switching works smoothly between grid and table
  - Empty states handled gracefully with clear messaging

- **Performance Criteria**:
  - Grid renders within 2 seconds for up to 50 projects
  - Smooth scrolling and responsive behavior
  - No memory leaks during view mode switching
  - Proper cleanup when navigating away from view

### Error Scenarios
1. **Grid Layout Failure**: If GridContainer fails to render, fallback to table view
2. **Data Binding Issues**: Missing project data should show placeholder tiles
3. **Responsive Failure**: Grid should degrade gracefully on very small screens
4. **Performance Degradation**: Large datasets should implement virtual scrolling

### Validation Points
- Grid container visible when tiles view selected: `document.querySelector('#tilesContainer[aria-hidden="false"]')`
- Proper column count in grid layout: CSS grid-template-columns verification
- Project tile data binding: Each tile contains expected project properties
- View mode persistence: Selected view mode maintained during session
- Error handling: Graceful degradation when grid rendering fails

### Test Data Requirements
- Minimum 12 test projects with varied names, descriptions, statuses
- Projects with different lengths of description text for layout testing
- Mix of active, inactive, and error status projects for visual variety
- Unicode characters in project names for internationalization testing

---

## Test Case ID: TC-UI-AGT-017
**Test Objective**: Verify project table view display, sorting, and multi-selection functionality  
**Business Process**: Project Management and Batch Operations  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-017
- **Test Priority**: High (P1)
- **Test Type**: Functional, UI/UX, Data Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:120-210`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:121-125,245-279`
- **Functions Under Test**: `onViewChange()`, `onExportSelected()`, `onDeleteSelected()`, table sorting

### Test Preconditions
1. **Authentication**: User authenticated with project management permissions
2. **Project Data**: At least 25 test projects with varied attributes
3. **UI State**: Projects view loaded with table view available
4. **Browser Support**: Modern browser with table rendering support
5. **Test Data**: Projects with different statuses, dates, agent counts

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| View Mode | table | String | SegmentedButton selection |
| Sort Field | last_modified | String | Default sorter configuration |
| Sort Order | descending | Boolean | Sorter configuration |
| Page Size | 20 | Number | Growing threshold |
| Selection Mode | MultiSelect | String | Table mode property |

### Test Procedure Steps
1. **Step 1 - Table View Activation**
   - Action: Click table view button in segmented control (table icon)
   - Expected: View switches from tiles to table, table becomes visible
   - Verification: `${view>/viewMode} === 'table'` evaluates to true

2. **Step 2 - Table Structure Validation**
   - Action: Verify table displays all required columns
   - Expected: 6 columns: Name, Description, Agents, Last Modified, Status, Actions
   - Verification: All column headers present and properly labeled

3. **Step 3 - Data Rendering Verification**
   - Action: Check project data displays correctly in table cells
   - Expected: Each row shows complete project information with proper formatting
   - Verification: Project ID shown below name, dates formatted, status colored

4. **Step 4 - Sorting Functionality**
   - Action: Click column headers to test sorting
   - Expected: Table sorts by clicked column, sort indicator updates
   - Verification: Data reorders correctly, descending/ascending toggle works

5. **Step 5 - Multi-Selection Testing**
   - Action: Select multiple projects using checkboxes
   - Expected: Rows highlight, selection count updates in toolbar
   - Verification: Selected items array populated correctly

6. **Step 6 - Batch Export Operation**
   - Action: Select 3 projects and click Export button
   - Expected: Export dialog/process initiated for selected projects
   - Verification: `onExportSelected()` receives correct project IDs

7. **Step 7 - Batch Delete Operation**
   - Action: Select 2 projects and click Delete button
   - Expected: Confirmation dialog shows with selected project count
   - Verification: Confirmation message includes "Delete 2 selected project(s)?"

8. **Step 8 - Growing Table Behavior**
   - Action: Scroll to bottom of table with 25+ projects
   - Expected: "More" button appears after 20 items, loads next batch
   - Verification: Growing threshold triggers at 20, smooth data loading

9. **Step 9 - Row Action Buttons**
   - Action: Test Edit, Clone, Delete buttons in action column
   - Expected: Each button triggers appropriate action for specific project
   - Verification: Correct project context passed to action handlers

10. **Step 10 - Table Responsiveness**
    - Action: Resize window to test table adaptation
    - Expected: Table remains usable, horizontal scroll if needed
    - Verification: Critical columns remain visible, actions accessible

### Expected Results
- **Table Display Criteria**:
  - Clean, organized table with clear column headers
  - Proper data alignment and formatting in cells
  - Status indicators with appropriate colors
  - Action buttons clearly visible and accessible

- **Interaction Criteria**:
  - Column sorting works bi-directionally
  - Multi-selection checkboxes function properly
  - Batch operations process selected items correctly
  - Row-level actions maintain proper context

- **Performance Criteria**:
  - Table renders within 1 second for initial 20 items
  - Sorting completes within 500ms
  - Growing functionality loads smoothly
  - No UI freezing during batch operations

### Error Scenarios
1. **Empty Selection**: Batch operations should show warning if no items selected
2. **Sort Failure**: Table should maintain current order if sort fails
3. **Load More Error**: Growing should show error message if data fetch fails
4. **Action Failure**: Individual row actions should show specific error messages

### Validation Points
- Table visibility when mode is table: `document.querySelector('#projectsTable[aria-hidden="false"]')`
- Column count matches specification: 6 columns present
- Multi-selection state: `oTable.getSelectedItems().length` returns correct count
- Sort indicator visible on active column
- Growing button appears at threshold: More button after 20 items

### Test Data Requirements
- 25+ test projects for growing functionality testing
- Projects with various last modified dates for sort testing
- Mix of project statuses (active, inactive, error) for visual testing
- Projects with 0-10 agents for agent count display
- Long project names and descriptions for text truncation testing

### Accessibility Requirements
- Table has proper ARIA labels and descriptions
- Column headers marked with appropriate roles
- Sort direction announced to screen readers
- Selection state communicated via ARIA attributes
- Action buttons have descriptive labels

---

## Test Case ID: TC-UI-AGT-018
**Test Objective**: Verify project search functionality with real-time filtering across multiple fields  
**Business Process**: Project Discovery and Search  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-018
- **Test Priority**: High (P1)
- **Test Type**: Functional, Search, Filtering
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:75-80`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:127-142`
- **Functions Under Test**: `onSearch()`, Filter creation and application

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Project Data**: At least 20 projects with varied names, descriptions, and IDs
3. **UI State**: Projects view loaded with search field visible
4. **Test Data**: Projects with searchable terms in different fields
5. **View Mode**: Either grid or table view active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Search Query | "agent" | String | User input |
| Search Width | 20rem | Dimension | SearchField width |
| Filter Fields | name, description, project_id | Array | Filter configuration |
| Filter Operator | Contains | FilterOperator | SAP UI5 Filter |
| Case Sensitivity | false | Boolean | Filter parameter |

### Test Procedure Steps
1. **Step 1 - Search Field Verification**
   - Action: Locate search field in projects toolbar
   - Expected: Search field visible with placeholder text "Search projects"
   - Verification: SearchField element present with correct properties

2. **Step 2 - Basic Search Execution**
   - Action: Type "agent" in search field and press Enter
   - Expected: Projects filtered to show only those containing "agent"
   - Verification: Only matching projects visible in current view

3. **Step 3 - Multi-Field Search Validation**
   - Action: Search for term that exists in project description but not name
   - Expected: Projects with matching descriptions displayed
   - Verification: Filter applies across name, description, and project_id fields

4. **Step 4 - Project ID Search**
   - Action: Enter partial project ID (e.g., "proj-123")
   - Expected: Projects with matching IDs displayed
   - Verification: Project ID field included in search scope

5. **Step 5 - Case-Insensitive Search**
   - Action: Search with different case variations ("AGENT", "Agent", "agent")
   - Expected: Same results regardless of case
   - Verification: Case-insensitive matching confirmed

6. **Step 6 - Special Character Handling**
   - Action: Search with special characters (e.g., "project-2024", "agent_v2")
   - Expected: Special characters treated as literal search terms
   - Verification: No regex interpretation, literal matching works

7. **Step 7 - Empty Search Reset**
   - Action: Clear search field and press Enter
   - Expected: All projects displayed again, filter removed
   - Verification: `oBinding.filter([])` called, full list restored

8. **Step 8 - Real-Time Search Updates**
   - Action: Type search term character by character
   - Expected: Results update as user types (if live search enabled)
   - Verification: Search triggered on appropriate events

9. **Step 9 - Search Persistence Across Views**
   - Action: Search for "test", then switch between grid and table views
   - Expected: Search filter remains active after view change
   - Verification: Filter state preserved during view mode switch

10. **Step 10 - No Results Handling**
    - Action: Search for non-existent term "xyz123abc"
    - Expected: Empty state shown with appropriate message
    - Verification: "No projects found" or similar message displayed

### Expected Results
- **Search Functionality Criteria**:
  - Search field accepts and processes user input correctly
  - Filtering applies to name, description, and project_id fields
  - Case-insensitive matching provides intuitive results
  - Special characters handled without errors

- **UI Response Criteria**:
  - Search results appear within 500ms of query submission
  - Clear visual indication of active search filter
  - Smooth transition between filtered and unfiltered states
  - Search field remains focused after search execution

- **Data Accuracy Criteria**:
  - Only projects matching search criteria displayed
  - All matching projects included (no false negatives)
  - No non-matching projects shown (no false positives)
  - Empty search returns to full project list

### Error Scenarios
1. **Invalid Characters**: Malformed input should be sanitized, not cause errors
2. **Long Search Terms**: Extremely long queries should be truncated gracefully
3. **Concurrent Searches**: Rapid search changes should cancel previous requests
4. **Network Failure**: Client-side filtering should continue working if possible

### Validation Points
- Search field present and accessible: `document.querySelector('#searchField')`
- Filter applied to table binding: `oTable.getBinding("items").getFilters().length > 0`
- Correct filter configuration: Multiple fields with OR condition
- Search results accuracy: Manual verification of displayed vs expected projects
- Performance: Search completion time under 500ms

### Test Data Requirements
- Projects with "agent" in name: "Agent Manager", "Multi-Agent System"
- Projects with "agent" in description: Various projects mentioning agents
- Projects with searchable IDs: "proj-agent-001", "test-2024-03"
- Projects with special characters: "AI-Agent_v2.0", "Project (Beta)"
- Edge cases: Empty descriptions, very long names, Unicode characters

### Accessibility Requirements
- Search field has proper ARIA labels and descriptions
- Search status announced to screen readers
- Keyboard navigation fully supported (Tab, Enter, Escape)
- Clear indication when search filter is active
- Results count communicated accessibly

---

## Test Case ID: TC-UI-AGT-019
**Test Objective**: Verify project sorting functionality with sort dialog and column-based ordering  
**Business Process**: Project Organization and Data Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-019
- **Test Priority**: Medium (P2)
- **Test Type**: Functional, Data Organization
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:96-98`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:148-167`
- **Functions Under Test**: `onOpenSortDialog()`, `onSortConfirm()`, Sorter application

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Project Data**: Minimum 15 projects with varied attributes for sorting
3. **UI State**: Projects view loaded in table mode (sorting most relevant here)
4. **Test Data**: Projects with different dates, names, statuses, agent counts
5. **Dialog Fragments**: SortDialog fragment available and loadable

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Default Sort | last_modified | String | Initial configuration |
| Sort Direction | descending | Boolean | Default sort order |
| Sort Fields | name, last_modified, status, agents | Array | Available sort options |
| Dialog Type | ViewSettingsDialog | Component | SAP M library |
| Persistence | session | String | Sort preference storage |

### Test Procedure Steps
1. **Step 1 - Sort Button Verification**
   - Action: Locate sort button in projects toolbar (sort icon)
   - Expected: Sort button visible and enabled
   - Verification: Button present with tooltip "Sort"

2. **Step 2 - Sort Dialog Opening**
   - Action: Click sort button to open sort dialog
   - Expected: ViewSettingsDialog opens with sort options
   - Verification: Dialog displays available sort fields

3. **Step 3 - Default Sort Indication**
   - Action: Observe initial sort selection in dialog
   - Expected: "Last Modified" selected with descending order
   - Verification: Default sort matches table's initial state

4. **Step 4 - Sort by Name (Ascending)**
   - Action: Select "Name" field and "Ascending" option, confirm
   - Expected: Projects reorder alphabetically A-Z
   - Verification: First project name starts with A or lowest character

5. **Step 5 - Sort by Status**
   - Action: Open sort dialog, select "Status" field, confirm
   - Expected: Projects grouped by status (Active, Inactive, Error)
   - Verification: Status groups appear consecutively

6. **Step 6 - Sort by Agent Count**
   - Action: Sort by "Agents" field in descending order
   - Expected: Projects with most agents appear first
   - Verification: Agent counts decrease down the list

7. **Step 7 - Sort Direction Toggle**
   - Action: Re-open dialog, toggle sort direction only
   - Expected: Same field sorted in opposite direction
   - Verification: Order reverses without changing sort field

8. **Step 8 - Sort Persistence Check**
   - Action: Navigate away and return to projects view
   - Expected: Previous sort preference maintained
   - Verification: Sort state persists during session

9. **Step 9 - Combined Sort and Search**
   - Action: Apply search filter, then change sort order
   - Expected: Filtered results sort correctly
   - Verification: Sort applies only to visible filtered items

10. **Step 10 - Sort Dialog Cancellation**
    - Action: Open sort dialog, make changes, click Cancel
    - Expected: No changes applied, original sort maintained
    - Verification: Table order unchanged after cancellation

### Expected Results
- **Sort Dialog Criteria**:
  - Dialog opens promptly with clear sort options
  - Current sort field and direction pre-selected
  - All relevant project fields available for sorting
  - Clear indication of ascending/descending options

- **Sorting Behavior Criteria**:
  - Sort applies immediately upon confirmation
  - Correct ordering based on selected field
  - Stable sort maintaining relative order
  - Performance acceptable for large datasets

- **UI Feedback Criteria**:
  - Visual indication of active sort (column header icon)
  - Smooth animation during reordering if implemented
  - No data loss or corruption during sort
  - Loading indicator if sort takes time

### Error Scenarios
1. **Dialog Load Failure**: Fallback to default sort if dialog fails to load
2. **Invalid Sort Field**: System should ignore invalid sort configurations
3. **Performance Issues**: Large datasets should show progress indicator
4. **Sort Conflict**: Multiple sort requests should queue or cancel previous

### Validation Points
- Sort button accessible: `document.querySelector('[aria-label*="sort"]')`
- Sort dialog opens: ViewSettingsDialog instance created and visible
- Sort applied to binding: `oBinding.getSort()` returns active Sorter
- Correct sort order: Manual verification of first/last items
- Performance timing: Sort completion under 1 second for 100 items

### Test Data Requirements
- Projects with names covering full alphabet range
- Various last modified dates (today, yesterday, last week, last month)
- Mix of all possible status values
- Agent counts ranging from 0 to 10+
- Special cases: Same names, same dates, null values

### Accessibility Requirements
- Sort button has descriptive ARIA label
- Sort dialog fully keyboard navigable
- Screen reader announces sort changes
- Current sort state communicated accessibly
- Focus management when dialog opens/closes

### Extended Test Scenarios
- **Multi-Column Sort**: If supported, test sorting by multiple columns
- **Custom Sort**: Test any custom sort algorithms (e.g., smart project ranking)
- **Locale-Aware Sort**: Verify proper handling of international characters
- **Null Handling**: Check how null/undefined values are sorted

---

## Test Case ID: TC-UI-AGT-020
**Test Objective**: Verify project status badges display correctly with appropriate colors and states  
**Business Process**: Project Status Visualization and Monitoring  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-020
- **Test Priority**: High (P1)
- **Test Type**: Functional, UI/Visual
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:187-189`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:302-309`
- **Functions Under Test**: `formatStatusState()`, ObjectStatus component rendering

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Project Data**: Projects exist with various status values (active, inactive, error, etc.)
3. **UI State**: Projects view loaded in either grid or table mode
4. **Theme**: Standard SAP theme applied for consistent color rendering
5. **Browser**: Modern browser with CSS3 support

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Status Values | active, inactive, error, pending | Array | Project data model |
| Status States | Success, Warning, Error, None | Enum | SAP UI5 ValueState |
| Badge Component | ObjectStatus | Control | SAP M library |
| Color Mapping | Green, Orange, Red, Gray | Colors | Theme CSS |
| Icon Display | Optional status icons | Boolean | Configuration |

### Test Procedure Steps
1. **Step 1 - Status Badge Visibility**
   - Action: Navigate to Projects view and observe status badges
   - Expected: Each project displays a status badge/indicator
   - Verification: ObjectStatus control rendered for each project

2. **Step 2 - Active Status Display**
   - Action: Locate projects with "active" status
   - Expected: Green badge with "Active" text displayed
   - Verification: CSS class indicates Success state (green color)

3. **Step 3 - Inactive Status Display**
   - Action: Find projects with "inactive" status
   - Expected: Orange/yellow badge with "Inactive" text
   - Verification: Warning state applied with appropriate color

4. **Step 4 - Error Status Display**
   - Action: Identify projects with "error" status
   - Expected: Red badge with "Error" text displayed
   - Verification: Error state with red color and possible icon

5. **Step 5 - Pending Status Display**
   - Action: Look for projects with "pending" or unknown status
   - Expected: Gray/neutral badge with status text
   - Verification: None state applied for neutral appearance

6. **Step 6 - Badge Text Accuracy**
   - Action: Compare badge text with actual project status
   - Expected: Badge text matches project status exactly
   - Verification: No text truncation or formatting issues

7. **Step 7 - Color Contrast Verification**
   - Action: Check badge visibility against background
   - Expected: All badges clearly visible with good contrast
   - Verification: WCAG 2.1 contrast requirements met

8. **Step 8 - Responsive Badge Display**
   - Action: Resize browser window to various sizes
   - Expected: Badges remain visible and properly sized
   - Verification: No overlap or layout issues at different sizes

9. **Step 9 - Theme Compatibility**
   - Action: Switch between light and dark themes if available
   - Expected: Badge colors adapt appropriately to theme
   - Verification: Status meaning remains clear in all themes

10. **Step 10 - Status Update Reflection**
    - Action: If possible, trigger a status change and observe
    - Expected: Badge updates immediately to reflect new status
    - Verification: No refresh required for status badge update

### Expected Results
- **Visual Criteria**:
  - Status badges clearly visible for all projects
  - Colors match SAP Fiori design standards
  - Text is readable and not truncated
  - Icons (if present) enhance status recognition

- **Functional Criteria**:
  - Correct status-to-color mapping:
    - Active → Green (Success)
    - Inactive → Orange (Warning)
    - Error → Red (Error)
    - Other → Gray (None)
  - Badge updates reflect real-time status changes
  - No performance impact from badge rendering

- **Accessibility Criteria**:
  - Status conveyed through text, not just color
  - Screen readers announce status correctly
  - Sufficient color contrast for visibility
  - Status badges keyboard accessible if interactive

### Error Scenarios
1. **Unknown Status**: System should display neutral badge for unrecognized status
2. **Missing Status**: Projects without status should show default indicator
3. **Long Status Text**: Text should truncate gracefully with tooltip
4. **Theme Issues**: Badges should remain visible even if theme fails to load

### Validation Points
- Status badge present: `document.querySelector('.sapMObjectStatus')`
- Correct color mapping: Inspect CSS classes for state indicators
- Text content accuracy: Badge text matches project status property
- Accessibility: ARIA labels present for status information
- Performance: Badge rendering doesn't slow table/grid display

### Test Data Requirements
- Projects with all possible status values
- At least 5 projects per status type for pattern verification
- Projects with null/undefined status for edge case testing
- Very long status text for truncation testing
- Custom status values to test fallback behavior

### Visual Design Requirements
- Badge styling consistent with SAP Fiori guidelines
- Clear visual hierarchy with primary content
- No visual conflicts with other UI elements
- Smooth transitions if status changes dynamically
- Print-friendly display for reports

---

## Test Case ID: TC-UI-AGT-021
**Test Objective**: Verify create project button functionality and project creation dialog workflow  
**Business Process**: Project Initialization and Creation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-021
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, CRUD Operation
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:19-22`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:43-84`
- **Functions Under Test**: `onCreateProject()`, `onCreateProjectConfirm()`, dialog management

### Test Preconditions
1. **Authentication**: User authenticated with project creation permissions
2. **UI State**: Projects view loaded and initialized
3. **Permissions**: User has rights to create new projects
4. **Backend**: Project creation API endpoint available
5. **Dialog Fragment**: CreateProjectDialog.fragment.xml loaded

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Button Type | Emphasized | String | UI5 ButtonType |
| Icon | sap-icon://add | String | SAP Icon font |
| Dialog Fragment | CreateProjectDialog | String | Fragment name |
| Project Name | Test Project 2024 | String | User input |
| Description | A2A agent development project | String | User input |

### Test Procedure Steps
1. **Step 1 - Create Button Visibility**
   - Action: Navigate to Projects view
   - Expected: "Create Project" button visible in page header
   - Verification: Button has emphasized type and add icon

2. **Step 2 - Button Click Response**
   - Action: Click "Create Project" button
   - Expected: Create project dialog opens within 500ms
   - Verification: Dialog fragment loaded and displayed

3. **Step 3 - Dialog Content Verification**
   - Action: Observe dialog contents
   - Expected: Name field, description field, and action buttons present
   - Verification: All form elements accessible and properly labeled

4. **Step 4 - Empty Name Validation**
   - Action: Leave name empty and click Create/Confirm
   - Expected: Validation message "Please enter a project name"
   - Verification: Dialog remains open, focus on name field

5. **Step 5 - Valid Project Creation**
   - Action: Enter "Test Project 2024" as name and description
   - Expected: Form accepts input without errors
   - Verification: No validation errors displayed

6. **Step 6 - Cancel Operation**
   - Action: Click Cancel button in dialog
   - Expected: Dialog closes without creating project
   - Verification: No API call made, project list unchanged

7. **Step 7 - Successful Creation Flow**
   - Action: Re-open dialog, enter valid data, click Create
   - Expected: API call initiated, loading state shown
   - Verification: POST request to /api/projects with correct data

8. **Step 8 - Success Feedback**
   - Action: Wait for creation to complete
   - Expected: Success message "Project created successfully"
   - Verification: Dialog closes, project list refreshes

9. **Step 9 - New Project Visibility**
   - Action: Check project list after creation
   - Expected: New project appears in list/grid
   - Verification: Project data matches entered values

10. **Step 10 - Error Handling**
    - Action: Attempt to create duplicate project name
    - Expected: Error message with specific details
    - Verification: Dialog remains open for correction

### Expected Results
- **UI Interaction Criteria**:
  - Button responds immediately to click (< 100ms)
  - Dialog opens with smooth animation
  - Form fields are focused appropriately
  - Loading states clearly indicated

- **Validation Criteria**:
  - Name field is required and validated
  - Whitespace is trimmed from inputs
  - Error messages are specific and helpful
  - Validation prevents empty project names

- **Creation Process Criteria**:
  - API call includes correct project data
  - Success/error responses handled properly
  - Project list updates automatically
  - No duplicate API calls on double-click

### Error Scenarios
1. **Network Failure**: Show "Failed to create project" with retry option
2. **Duplicate Name**: Display specific error from backend
3. **Permission Denied**: Show authorization error message
4. **Server Error**: Display generic error with support contact

### Validation Points
- Button present: `document.querySelector('[aria-label*="Create Project"]')`
- Dialog opens: Fragment instance created and visible
- Form validation: Required fields marked and validated
- API call: Network request to POST /api/projects
- Success handling: Project list refreshed after creation

### Test Data Requirements
- Valid project names: Alphanumeric with spaces, hyphens, underscores
- Invalid names: Empty string, only whitespace, special characters
- Description: Optional field, supports longer text
- Duplicate names: For testing uniqueness validation
- Edge cases: Very long names, Unicode characters

### Dialog Behavior Requirements
- Modal dialog prevents background interaction
- Escape key closes dialog (with confirmation if data entered)
- Tab navigation cycles through form fields
- Enter key submits form when valid
- Loading state disables form during API call

### API Integration Points
- **Endpoint**: POST /api/projects
- **Request Body**: `{ name: string, description: string }`
- **Success Response**: 201 Created with project data
- **Error Response**: 4xx/5xx with error details
- **Side Effects**: Project list refresh, possible navigation

---

## Test Case ID: TC-UI-AGT-022
**Test Objective**: Verify open project functionality through various interaction methods  
**Business Process**: Project Access and Navigation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-022
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Navigation
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:174-181`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:113-119`
- **Functions Under Test**: `onProjectPress()`, navigation handling

### Test Preconditions
1. **Authentication**: User authenticated with project access permissions
2. **Project Data**: At least one existing project in the system
3. **UI State**: Projects view loaded with projects displayed
4. **View Mode**: Works in both grid and table views
5. **Navigation**: Routing configured for project detail view

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Project ID | proj-123-abc | String | Project data |
| Project Name | Test Project | String | Project data |
| Row Type | Active | String | ColumnListItem type |
| Click Target | Row/Card/Link | Element | User interaction |
| Navigation | Detail view | Target | Router config |

### Test Procedure Steps
1. **Step 1 - Table Row Click (Table View)**
   - Action: In table view, click on a project row
   - Expected: Row highlights and navigation triggered
   - Verification: onProjectPress event fired with correct context

2. **Step 2 - Project Name Link Click**
   - Action: Click specifically on project name link
   - Expected: Link responds to click, navigation initiated
   - Verification: Same navigation as row click

3. **Step 3 - Grid Card Click (Grid View)**
   - Action: Switch to grid view, click project card
   - Expected: Card shows click feedback, navigation occurs
   - Verification: Project context passed correctly

4. **Step 4 - Keyboard Navigation**
   - Action: Use Tab to focus project, press Enter
   - Expected: Keyboard activation triggers navigation
   - Verification: Same behavior as mouse click

5. **Step 5 - Touch Interaction (Mobile)**
   - Action: On touch device, tap project item
   - Expected: Touch feedback shown, navigation works
   - Verification: No double-tap required

6. **Step 6 - Navigation Feedback**
   - Action: Observe UI during navigation
   - Expected: Loading indicator or transition shown
   - Verification: User aware of navigation in progress

7. **Step 7 - Project Context Validation**
   - Action: Check data passed to navigation
   - Expected: Correct project_id and data transferred
   - Verification: Target view receives proper context

8. **Step 8 - Double-Click Prevention**
   - Action: Double-click quickly on project
   - Expected: Only one navigation triggered
   - Verification: No duplicate navigation events

9. **Step 9 - Disabled Project Handling**
   - Action: If disabled projects exist, try clicking
   - Expected: No navigation for disabled projects
   - Verification: Visual indication of disabled state

10. **Step 10 - Navigation History**
    - Action: After opening project, check browser history
    - Expected: Browser back button returns to project list
    - Verification: Navigation state properly managed

### Expected Results
- **Interaction Criteria**:
  - All clickable areas respond within 100ms
  - Visual feedback (hover/active states) provided
  - Consistent behavior across view modes
  - Touch-friendly interaction on mobile

- **Navigation Criteria**:
  - Correct project opens based on selection
  - Project ID passed to detail view
  - URL updates to reflect navigation
  - Browser history maintained properly

- **Accessibility Criteria**:
  - Keyboard navigation fully supported
  - Screen reader announces navigation
  - Focus management during transition
  - ARIA labels indicate clickable items

### Error Scenarios
1. **Navigation Failure**: Show error message, remain on list
2. **Project Not Found**: Display 404-style message
3. **Permission Denied**: Show authorization error
4. **Network Error**: Offline message with retry option

### Validation Points
- Click handler attached: Event listener on project items
- Context retrieval: `oContext.getProperty("project_id")` works
- Navigation triggered: Router or navigation method called
- Visual feedback: Active/hover states visible
- URL update: Browser address reflects navigation

### Test Data Requirements
- Multiple projects for selection testing
- Projects with various states (active, archived, etc.)
- Long project names for truncation testing
- Projects with special characters in IDs
- Recently accessed projects for history testing

### Navigation Implementation Notes
- Current implementation shows toast message (temporary)
- Future: Should navigate to ProjectDetail view
- URL pattern: #/projects/{project_id}
- Deep linking support required
- Back navigation must return to same scroll position

### Performance Requirements
- Navigation initiation < 100ms after click
- Page transition smooth (no flicker)
- Previous view cleanup prevents memory leaks
- Large project lists don't slow navigation
- Smooth scrolling preserved on back navigation

---

## Test Case ID: TC-UI-AGT-023
**Test Objective**: Verify clone project functionality with confirmation dialog and duplication process  
**Business Process**: Project Duplication and Template Creation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-023
- **Test Priority**: High (P1)
- **Test Type**: Functional, CRUD Operation
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:196-200`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:198-212`
- **Functions Under Test**: `onCloneProject()`, MessageBox confirmation

### Test Preconditions
1. **Authentication**: User authenticated with project creation permissions
2. **Project Data**: At least one clonable project exists
3. **UI State**: Projects view loaded with clone buttons visible
4. **Permissions**: User has rights to create new projects
5. **Storage**: Sufficient space for cloned project

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Source Project | Original Project | String | Selected project |
| Clone Name | Original Project (Copy) | String | Auto-generated |
| Icon | sap-icon://copy | String | SAP Icon font |
| Button Type | Transparent | String | UI5 ButtonType |
| Confirmation | MessageBox.confirm | Dialog | SAP M library |

### Test Procedure Steps
1. **Step 1 - Clone Button Visibility**
   - Action: Navigate to project row/card actions
   - Expected: Clone button visible with copy icon
   - Verification: Button has tooltip "Clone"

2. **Step 2 - Clone Button Click**
   - Action: Click clone button for a project
   - Expected: Confirmation dialog appears immediately
   - Verification: MessageBox.confirm displayed

3. **Step 3 - Confirmation Dialog Content**
   - Action: Read confirmation message
   - Expected: "Clone project 'Project Name'?" displayed
   - Verification: Project name shown in message

4. **Step 4 - Cancel Clone Operation**
   - Action: Click Cancel/No in confirmation dialog
   - Expected: Dialog closes, no action taken
   - Verification: No API call made, project list unchanged

5. **Step 5 - Confirm Clone Operation**
   - Action: Click OK/Yes in confirmation dialog
   - Expected: Clone process initiated
   - Verification: Loading indicator shown (when implemented)

6. **Step 6 - Clone Name Generation**
   - Action: Observe cloned project name
   - Expected: Automatic name like "Original (Copy)" or "Original_2"
   - Verification: Unique name generated to avoid conflicts

7. **Step 7 - Clone Process Completion**
   - Action: Wait for clone operation to complete
   - Expected: Success message displayed
   - Verification: New project appears in list

8. **Step 8 - Clone Content Verification**
   - Action: Open cloned project
   - Expected: All files and structure duplicated
   - Verification: Clone is independent of original

9. **Step 9 - Multiple Clones**
   - Action: Clone the same project again
   - Expected: Second clone gets unique name
   - Verification: "Original (Copy 2)" or similar

10. **Step 10 - Error Handling**
    - Action: Simulate clone failure (quota exceeded)
    - Expected: Error message with details
    - Verification: Original project unaffected

### Expected Results
- **UI Interaction Criteria**:
  - Clone button responds immediately
  - Confirmation dialog modal and centered
  - Clear project identification in message
  - Smooth dialog animations

- **Clone Process Criteria**:
  - Unique name automatically generated
  - All project content duplicated
  - No references to original project
  - Metadata updated (creation date, owner)

- **Feedback Criteria**:
  - Clear progress indication during clone
  - Success/error messages informative
  - New project immediately visible
  - No page refresh required

### Error Scenarios
1. **Insufficient Permissions**: Show "Cannot clone project" message
2. **Storage Quota Exceeded**: Display quota error with cleanup suggestion
3. **Name Conflict**: Auto-increment number suffix
4. **Network Failure**: Show retry option with error details

### Validation Points
- Button present: Clone icon in action column/menu
- Dialog triggered: MessageBox.confirm called with project name
- Clone logic: Implementation currently shows "coming soon"
- Future API: Should call POST /api/projects/{id}/clone
- Success handling: Project list refresh after clone

### Test Data Requirements
- Projects with various sizes for clone testing
- Projects with special characters in names
- Large projects to test timeout handling
- Projects with complex file structures
- Projects near storage quota limits

### Clone Implementation Requirements
- **Current State**: Shows "Clone functionality - coming soon"
- **Target Implementation**:
  - API call to backend clone endpoint
  - Progress tracking for large projects
  - Automatic name generation with conflict resolution
  - Option to customize clone name before creation
  - Selective cloning (files/folders to include/exclude)

### Performance Requirements
- Clone initiation < 500ms after confirmation
- Small projects (<10MB) clone within 5 seconds
- Large projects show accurate progress
- No UI blocking during clone operation
- Efficient storage usage (no unnecessary duplication)

### Future Enhancements
- Clone with modifications dialog
- Template creation from project
- Cross-workspace cloning
- Version control integration
- Shared project templates

---

## Test Case ID: TC-UI-AGT-024
**Test Objective**: Verify archive project functionality with confirmation and status change  
**Business Process**: Project Lifecycle Management and Archival  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-024
- **Test Priority**: High (P1)
- **Test Type**: Functional, Status Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:201-205`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:214-248`
- **Functions Under Test**: `onArchiveProject()`, `_archiveProject()`, MessageBox confirmation

### Test Preconditions
1. **Authentication**: User authenticated with project archival permissions
2. **Project Data**: Active projects available for archiving
3. **UI State**: Projects view loaded with archive buttons visible
4. **Status**: Projects in "active" or "inactive" status
5. **Backend**: Archive API endpoint available

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Target Project | Active Project | String | Selected project |
| Icon | sap-icon://inbox | String | SAP Icon font |
| Confirmation Title | Archive Project | String | Dialog title |
| API Endpoint | /api/projects/{id}/archive | String | Backend route |
| New Status | archived | String | Target status |

### Test Procedure Steps
1. **Step 1 - Archive Button Visibility**
   - Action: Navigate to project row actions
   - Expected: Archive button visible with inbox icon
   - Verification: Button has tooltip "Archive"

2. **Step 2 - Archive Button Click**
   - Action: Click archive button for active project
   - Expected: Confirmation dialog appears
   - Verification: MessageBox with question icon displayed

3. **Step 3 - Confirmation Message Content**
   - Action: Read confirmation dialog text
   - Expected: "Archive project 'Name'? This will move the project to archived status and hide it from the active projects list."
   - Verification: Clear explanation of archive action

4. **Step 4 - Cancel Archive Operation**
   - Action: Click Cancel in confirmation dialog
   - Expected: Dialog closes, no changes made
   - Verification: Project remains in active list

5. **Step 5 - Confirm Archive Operation**
   - Action: Click OK in confirmation dialog
   - Expected: Archive API call initiated
   - Verification: POST to /api/projects/{id}/archive

6. **Step 6 - Archive Success Feedback**
   - Action: Wait for archive completion
   - Expected: Success message "Project 'Name' archived successfully"
   - Verification: Toast message displayed

7. **Step 7 - Project List Update**
   - Action: Observe project list after archival
   - Expected: Archived project removed from active view
   - Verification: Project no longer visible in default list

8. **Step 8 - Archive Multiple Projects**
   - Action: Archive another project
   - Expected: Same flow works consistently
   - Verification: Multiple projects can be archived

9. **Step 9 - Archive Permission Check**
   - Action: Try archiving as user without permission
   - Expected: Permission error displayed
   - Verification: Project remains unarchived

10. **Step 10 - Network Error Handling**
    - Action: Simulate network failure during archive
    - Expected: Error message with details
    - Verification: Project status unchanged

### Expected Results
- **UI Interaction Criteria**:
  - Archive button clearly visible and accessible
  - Confirmation dialog provides clear context
  - Dialog title "Archive Project" displayed
  - Question icon indicates decision required

- **Archive Process Criteria**:
  - Project status changes to "archived"
  - Project removed from active project list
  - Archive action is logged for audit
  - Related resources remain intact

- **Feedback Criteria**:
  - Success message includes project name
  - Error messages are specific and helpful
  - List refreshes automatically after archive
  - No page reload required

### Error Scenarios
1. **Insufficient Permissions**: "You don't have permission to archive this project"
2. **Active Workflows**: "Cannot archive project with active workflows"
3. **Network Failure**: "Failed to archive project: [error details]"
4. **Already Archived**: "Project is already archived"

### Validation Points
- Button present: Archive icon in action buttons
- Dialog triggered: MessageBox.confirm with detailed message
- API call: POST /api/projects/{projectId}/archive
- Success handling: Project list refreshed
- Error handling: Appropriate error messages displayed

### Test Data Requirements
- Active projects ready for archival
- Projects with different statuses
- Projects with dependencies to test constraints
- User accounts with/without archive permissions
- Recently modified projects

### Archive Business Rules
- Only active/inactive projects can be archived
- Archived projects hidden from default view
- Archive action is reversible (unarchive)
- Project data preserved during archival
- Audit trail maintained for compliance

### Performance Requirements
- Archive operation completes < 2 seconds
- UI remains responsive during operation
- Batch archive for multiple projects (future)
- No data loss during archival
- Efficient status update in database

### Related Features
- Unarchive functionality (restore)
- Archived projects view/filter
- Bulk archive operations
- Archive retention policies
- Automatic archival rules

---

## Test Case ID: TC-UI-AGT-025
**Test Objective**: Verify delete project functionality with confirmation dialog and permanent removal  
**Business Process**: Project Deletion and Data Cleanup  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-025
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Destructive Operation
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:206-210`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:250-279`
- **Functions Under Test**: `onDeleteProject()`, `_deleteProject()`, MessageBox confirmation

### Test Preconditions
1. **Authentication**: User authenticated with project deletion permissions
2. **Project Data**: Deletable test projects exist
3. **UI State**: Projects view loaded with delete buttons visible
4. **Backup**: Test projects backed up if needed
5. **Permissions**: User has delete rights on target projects

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Target Project | Test Project | String | Selected project |
| Icon | sap-icon://delete | String | SAP Icon font |
| Confirmation Icon | WARNING | Icon | MessageBox.Icon |
| Dialog Title | Confirm Deletion | String | Confirmation title |
| API Method | DELETE | String | HTTP method |

### Test Procedure Steps
1. **Step 1 - Delete Button Visibility**
   - Action: Locate delete button in project actions
   - Expected: Delete button visible with trash icon
   - Verification: Button has tooltip "Delete"

2. **Step 2 - Delete Button Click**
   - Action: Click delete button for a test project
   - Expected: Confirmation dialog appears immediately
   - Verification: MessageBox with warning icon displayed

3. **Step 3 - Warning Message Content**
   - Action: Read confirmation dialog message
   - Expected: "Delete project 'Name'? This action cannot be undone."
   - Verification: Clear warning about permanence

4. **Step 4 - Dialog Visual Elements**
   - Action: Observe dialog styling
   - Expected: Warning icon (triangle), "Confirm Deletion" title
   - Verification: Visual indicators of serious action

5. **Step 5 - Cancel Delete Operation**
   - Action: Click Cancel/No in dialog
   - Expected: Dialog closes, no deletion occurs
   - Verification: Project remains in list

6. **Step 6 - Confirm Delete Operation**
   - Action: Re-open dialog, click OK/Yes
   - Expected: Delete API call initiated
   - Verification: DELETE /api/projects/{id} called

7. **Step 7 - Delete Progress**
   - Action: Observe during deletion
   - Expected: Loading state or progress indicator
   - Verification: UI shows operation in progress

8. **Step 8 - Success Feedback**
   - Action: Wait for deletion completion
   - Expected: "Project deleted successfully" message
   - Verification: Success toast displayed

9. **Step 9 - Project List Update**
   - Action: Check project list after deletion
   - Expected: Deleted project removed from view
   - Verification: Project no longer in list

10. **Step 10 - Permanent Deletion Verification**
    - Action: Try to access deleted project directly
    - Expected: Project not found (404 error)
    - Verification: Data permanently removed

### Expected Results
- **UI Interaction Criteria**:
  - Delete button clearly marked with warning color
  - Confirmation dialog emphasizes irreversibility
  - Warning icon draws attention to risk
  - Two-step process prevents accidents

- **Deletion Process Criteria**:
  - Project completely removed from system
  - All associated files deleted
  - No orphaned resources remain
  - Audit log records deletion

- **Feedback Criteria**:
  - Clear success confirmation
  - Error messages explain failures
  - List updates without refresh
  - No UI glitches during update

### Error Scenarios
1. **Insufficient Permissions**: "You don't have permission to delete this project"
2. **Project In Use**: "Cannot delete project with active sessions"
3. **Network Failure**: "Failed to delete project: [error]"
4. **Server Error**: "Server error occurred during deletion"

### Validation Points
- Button present: Delete icon in actions
- Dialog type: MessageBox.confirm with WARNING icon
- API call: DELETE /api/projects/{projectId}
- Success handling: Project removed from list
- Error handling: Descriptive error messages

### Test Data Requirements
- Test projects safe to delete
- Projects with no critical data
- Projects owned by test user
- Projects with various states
- Recently created test projects

### Delete Safety Measures
- Confirmation dialog mandatory
- Warning icon and message
- "Cannot be undone" clearly stated
- No batch delete without individual confirmations
- Audit trail for compliance

### Performance Requirements
- Delete confirmation immediate
- Deletion completes < 3 seconds
- UI updates smoothly
- No hanging requests
- Proper timeout handling

### Security Considerations
- Only project owner can delete
- Admin override with audit
- Soft delete option (future)
- Backup before delete (future)
- Recovery period (future)

### Related Features
- Batch delete (with confirmations)
- Soft delete/trash functionality
- Project export before delete
- Deletion audit reports
- Undo/recovery options

---

## Test Case ID: TC-UI-AGT-026
**Test Objective**: Verify project filter panel functionality with multiple filter criteria  
**Business Process**: Project Filtering and Advanced Search  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-026
- **Test Priority**: High (P1)
- **Test Type**: Functional, Data Filtering
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:93-96`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:144-205`
- **Functions Under Test**: `onOpenFilterDialog()`, `onFilterConfirm()`, `onClearFilters()`

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Project Data**: Multiple projects with varied attributes
3. **UI State**: Projects view loaded with filter button visible
4. **Test Data**: Projects with different statuses, dates, agent counts
5. **Dialog Fragment**: FilterDialog.fragment.xml available

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Status Options | active, inactive, error, archived | Array | Filter options |
| Date Range | Last 7/30/90 days, Custom | Options | Date filters |
| Agent Count | 0-10, 10+, Custom range | Range | Numeric filter |
| Filter Icon | sap-icon://filter | String | SAP Icon font |
| Active Indicator | hasActiveFilters | CSS Class | Style indicator |

### Test Procedure Steps
1. **Step 1 - Filter Button Visibility**
   - Action: Locate filter button in toolbar
   - Expected: Filter button visible with funnel icon
   - Verification: Button has tooltip "Filter"

2. **Step 2 - Open Filter Dialog**
   - Action: Click filter button
   - Expected: Filter dialog opens with multiple criteria
   - Verification: Dialog displays status, date, and count filters

3. **Step 3 - Status Filter Selection**
   - Action: Select "Active" and "Inactive" status checkboxes
   - Expected: Multiple status selections allowed
   - Verification: Checkboxes reflect selection state

4. **Step 4 - Date Range Filter**
   - Action: Select "Last 30 days" from date options
   - Expected: Date range automatically calculated
   - Verification: From/To dates populated correctly

5. **Step 5 - Agent Count Filter**
   - Action: Set minimum agents to 2
   - Expected: Numeric input accepts value
   - Verification: Validation prevents negative numbers

6. **Step 6 - Apply Filters**
   - Action: Click "OK" or "Apply" button
   - Expected: Dialog closes, filters applied
   - Verification: Project list updates to show filtered results

7. **Step 7 - Filter Indicator**
   - Action: Observe filter button after applying filters
   - Expected: Visual indicator shows active filters
   - Verification: Button has highlight or badge

8. **Step 8 - Verify Filter Results**
   - Action: Check displayed projects
   - Expected: Only projects matching ALL criteria shown
   - Verification: Each project meets filter conditions

9. **Step 9 - Clear Filters**
   - Action: Click "Clear Filters" option
   - Expected: All filters removed
   - Verification: Full project list restored

10. **Step 10 - Filter Persistence**
    - Action: Navigate away and return
    - Expected: Filter state maintained (or cleared)
    - Verification: Consistent filter behavior

### Expected Results
- **Dialog Behavior Criteria**:
  - Filter dialog opens smoothly
  - All filter options clearly labeled
  - Multiple selections supported
  - Dialog responsive and accessible

- **Filter Application Criteria**:
  - Filters apply immediately on confirmation
  - Multiple filters work with AND logic
  - Results update without page refresh
  - Count shows filtered results

- **Visual Feedback Criteria**:
  - Active filter indicator visible
  - Filter button changes appearance
  - Clear indication of applied filters
  - Easy to identify filtered state

### Error Scenarios
1. **Invalid Date Range**: Show "End date must be after start date"
2. **No Results**: Display "No projects match the selected filters"
3. **Filter Conflict**: Handle mutually exclusive filters gracefully
4. **Performance**: Show loading for large filter operations

### Validation Points
- Dialog opens: Fragment loaded and displayed
- Filter options: All criteria types available
- Filter logic: Correct AND/OR combinations
- Visual indicator: CSS class applied when filters active
- Results accuracy: Filtered data matches criteria

### Test Data Requirements
- Projects with all status values
- Projects spanning different date ranges
- Projects with 0 to 10+ agents
- Edge cases: Today's date, empty projects
- Large dataset for performance testing

### Filter Combinations
- **Single Filter**: Status only, Date only, Count only
- **Multiple Filters**: Status + Date, Status + Count, All three
- **Edge Cases**: No matches, All match, Single result
- **Complex Filters**: Multiple statuses + date range
- **Clear Operation**: Reset from any filter state

### Performance Requirements
- Dialog opens < 300ms
- Filter application < 500ms
- Large datasets handled efficiently
- No UI freezing during filter
- Smooth result transitions

### UI/UX Requirements
- Clear filter option labels
- Intuitive multi-select behavior
- Visual feedback during filtering
- Result count indication
- Easy filter removal process

---

## Test Case ID: TC-UI-AGT-027
**Test Objective**: Verify filter combinations work correctly with search and sort  
**Business Process**: Advanced Project Discovery with Combined Filters  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-027
- **Test Priority**: High (P1)
- **Test Type**: Functional, Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:127-205`
- **Secondary**: Search (127-142), Filter (144-205), Sort (156-167)
- **Functions Under Test**: Combined filter/search/sort operations

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Project Data**: Diverse projects for complex filtering
3. **UI State**: Projects view with all filter controls visible
4. **Test Data**: Projects with overlapping attributes
5. **Features**: Search, filter, and sort all functional

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Search Term | "agent" | String | Search field |
| Status Filter | ["active", "inactive"] | Array | Filter dialog |
| Sort Field | "name" | String | Sort dialog |
| Sort Order | "ascending" | Boolean | Sort dialog |
| Date Filter | "Last 30 days" | Range | Filter dialog |

### Test Procedure Steps
1. **Step 1 - Apply Search First**
   - Action: Enter "agent" in search field
   - Expected: Projects filtered by search term
   - Verification: Only projects containing "agent" visible

2. **Step 2 - Add Status Filter**
   - Action: Open filter dialog, select Active and Inactive
   - Expected: Search results further filtered by status
   - Verification: Projects match search AND status criteria

3. **Step 3 - Apply Sort Order**
   - Action: Open sort dialog, sort by name ascending
   - Expected: Filtered results sorted alphabetically
   - Verification: Correct order maintained with filters

4. **Step 4 - Add Date Range Filter**
   - Action: In filter dialog, add "Last 30 days"
   - Expected: Results filtered by search + status + date
   - Verification: All three filters apply correctly

5. **Step 5 - Modify Search Term**
   - Action: Change search to "test"
   - Expected: Filters remain, search updates
   - Verification: New results match all criteria

6. **Step 6 - Remove One Filter**
   - Action: Clear status filter, keep others
   - Expected: Date and search filters still active
   - Verification: Results expand to include all statuses

7. **Step 7 - Change Sort Order**
   - Action: Sort by date descending
   - Expected: Filtered results re-sort correctly
   - Verification: Sort doesn't affect filter criteria

8. **Step 8 - Complex Filter Scenario**
   - Action: Apply all filters + sort + search
   - Expected: All operations work together
   - Verification: Correct subset in correct order

9. **Step 9 - Clear Individual vs All**
   - Action: Test clearing filters individually
   - Expected: Each filter can be removed independently
   - Verification: Other filters remain active

10. **Step 10 - Performance Check**
    - Action: Apply maximum filters on large dataset
    - Expected: Response time remains acceptable
    - Verification: No UI freezing or timeouts

### Expected Results
- **Filter Combination Criteria**:
  - All filters work with AND logic
  - Search + Filter + Sort combine properly
  - Each operation maintains others' state
  - Results always reflect all active criteria

- **UI State Criteria**:
  - Visual indicators for all active filters
  - Clear which operations are applied
  - Easy to see combined filter state
  - Counts update with combinations

- **Performance Criteria**:
  - Combined operations < 1 second
  - Smooth transitions between states
  - No conflicts between operations
  - Efficient query execution

### Error Scenarios
1. **Conflicting Filters**: No results found message
2. **Invalid Combinations**: Graceful handling
3. **Performance Degradation**: Loading indicators
4. **State Corruption**: Filters remain consistent

### Validation Points
- Filter state persistence: All filters maintain state
- Operation order: Results same regardless of apply order
- Visual indicators: All active filters shown
- Result accuracy: Data matches all criteria
- Performance: Acceptable with multiple filters

### Test Data Requirements
- Projects with overlapping search terms
- Various status and date combinations
- Large dataset (100+ projects)
- Projects with similar names
- Edge cases for each filter type

### Filter Combination Matrix
| Search | Status | Date | Sort | Expected Result |
|--------|--------|------|------|-----------------|
| Yes | No | No | No | Search only |
| Yes | Yes | No | No | Search + Status |
| Yes | Yes | Yes | No | All filters |
| Yes | Yes | Yes | Yes | All + sorted |
| No | Yes | Yes | Yes | Filter + sort only |

### Complex Scenarios
- Search "project" + Active status + Last week + Sort by date
- Empty search + Multiple statuses + Custom date range
- Special characters in search + All filters
- Clear search while filters active
- Maximum filters on minimum dataset

### UI Behavior Requirements
- Filter indicators stack/combine visually
- Clear option for each filter type
- Breadcrumb showing active filters
- Result count updates live
- No jarring UI updates

---

## Test Case ID: TC-UI-AGT-028
**Test Objective**: Verify clear filters functionality removes all active filters  
**Business Process**: Filter Reset and Data Restoration  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-028
- **Test Priority**: Medium (P2)
- **Test Type**: Functional, State Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:199-205`
- **Functions Under Test**: `onClearFilters()`, `_updateFilterIndicator()`
- **Related**: Filter dialog, search field, sort options

### Test Preconditions
1. **Authentication**: User authenticated with project view permissions
2. **Active Filters**: At least one filter currently applied
3. **UI State**: Projects view with filtered results
4. **Test Data**: Sufficient projects to show filter effect
5. **Controls**: Clear filters option available

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Initial Filters | Status + Date + Search | Multiple | Various controls |
| Clear Action | Button/Menu click | Event | User interaction |
| Toast Message | "All filters cleared" | String | Success feedback |
| Filter Indicator | hasActiveFilters | CSS Class | Visual state |
| Result Set | All projects | Array | Unfiltered data |

### Test Procedure Steps
1. **Step 1 - Apply Multiple Filters**
   - Action: Apply status, date, and search filters
   - Expected: Projects filtered, count reduced
   - Verification: Filter indicator shows active state

2. **Step 2 - Locate Clear Option**
   - Action: Find clear filters button/link
   - Expected: Option visible when filters active
   - Verification: Clear option enabled and accessible

3. **Step 3 - Clear All Filters**
   - Action: Click clear filters option
   - Expected: All filters removed simultaneously
   - Verification: No confirmation required

4. **Step 4 - Verify Complete Reset**
   - Action: Check all filter controls
   - Expected: Search empty, filters unchecked, sort default
   - Verification: All controls in initial state

5. **Step 5 - Check Results Update**
   - Action: Observe project list
   - Expected: All projects visible again
   - Verification: Count matches total projects

6. **Step 6 - Filter Indicator Update**
   - Action: Check filter button appearance
   - Expected: Active indicator removed
   - Verification: hasActiveFilters class not present

7. **Step 7 - Success Feedback**
   - Action: Watch for feedback message
   - Expected: "All filters cleared" toast
   - Verification: Message appears briefly

8. **Step 8 - Partial Clear Test**
   - Action: Apply filters, clear search manually, then clear all
   - Expected: Clear all still works correctly
   - Verification: All remaining filters cleared

9. **Step 9 - No Active Filters**
   - Action: Try clear when no filters active
   - Expected: No error, possible disabled state
   - Verification: Graceful handling

10. **Step 10 - Performance Check**
    - Action: Clear filters on large dataset
    - Expected: Quick response time
    - Verification: UI updates smoothly

### Expected Results
- **Clear Operation Criteria**:
  - All filters removed in single action
  - No confirmation dialog needed
  - Immediate effect on results
  - Complete state reset

- **UI Update Criteria**:
  - Filter controls reset to defaults
  - Visual indicators updated
  - Result count reflects all projects
  - Smooth transition animation

- **Feedback Criteria**:
  - Clear success message shown
  - Filter button returns to normal
  - No residual filter effects
  - User aware of state change

### Error Scenarios
1. **Partial Clear Failure**: Some filters remain active
2. **UI State Mismatch**: Visual indicators incorrect
3. **Performance Issue**: Slow clear on large datasets
4. **Search Not Cleared**: Search field retains value

### Validation Points
- Function called: `onClearFilters()` executes
- Binding reset: `oBinding.filter([])` called
- Indicator update: `_updateFilterIndicator(false)`
- Toast display: Success message shown
- Complete reset: All filter states cleared

### Test Data Requirements
- Projects for initial filtered state
- Multiple filter types active
- Large dataset for performance
- Edge cases: Single filter, all filters
- Various filter combinations

### Clear Operation Scenarios
- **Clear from Dialog**: Clear button in filter dialog
- **Clear from Toolbar**: Dedicated clear button
- **Clear from Menu**: Clear option in menu
- **Keyboard Shortcut**: If implemented (e.g., Ctrl+Shift+C)
- **Right-Click Option**: Context menu clear

### State Management
- Search field cleared
- Filter dialog selections reset
- Sort returns to default
- Active indicators removed
- URL parameters updated (if used)

### User Experience
- One-click clear operation
- No accidental clear protection
- Visual feedback immediate
- Clear option always findable
- Undo option (future enhancement)

---

## Test Case ID: TC-UI-AGT-029
**Test Objective**: Verify virtual scrolling performance with large datasets  
**Business Process**: Efficient Data Display and Navigation  
**SAP Module**: A2A Agents Developer Portal - Projects List  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-029
- **Test Priority**: High (P1)
- **Test Type**: Performance, UI Rendering
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/view/ProjectsList.view.xml:92-104`
- **Controller**: `a2aAgents/frontend/webapp/controller/ProjectsList.controller.js:_setupVirtualScrolling()`
- **Functions Under Test**: Virtual scrolling, lazy loading, scroll performance

### Test Preconditions
1. **Authentication**: User authenticated with project access
2. **Large Dataset**: 1000+ projects loaded in memory
3. **Table View**: sap.ui.table.Table with virtual scrolling enabled
4. **Browser**: Modern browser with smooth scrolling support
5. **Performance Tools**: Browser dev tools for monitoring

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Dataset Size | 1000+ items | Number | Test data |
| Visible Rows | 15 | Number | Table config |
| Threshold | 100 | Number | Initial load |
| Scroll Speed | Various | Action | User input |
| Memory Limit | Browser default | Resource | System |

### Test Procedure Steps
1. **Step 1 - Initial Load Performance**
   - Action: Navigate to Projects List view
   - Expected: Table loads with first 100 rows quickly
   - Verification: Load time < 2 seconds

2. **Step 2 - DOM Element Count**
   - Action: Check DOM elements in dev tools
   - Expected: Only visible rows rendered in DOM
   - Verification: ~15-20 table rows in DOM, not 1000+

3. **Step 3 - Smooth Scroll Test**
   - Action: Scroll slowly through the list
   - Expected: Smooth scrolling without jank
   - Verification: 60 FPS maintained during scroll

4. **Step 4 - Fast Scroll Performance**
   - Action: Scroll quickly from top to bottom
   - Expected: No freezing or lag
   - Verification: UI remains responsive

5. **Step 5 - Jump Navigation**
   - Action: Use scrollbar to jump to middle/end
   - Expected: Quick render of target position
   - Verification: < 500ms to display new rows

6. **Step 6 - Memory Usage Check**
   - Action: Monitor memory in dev tools
   - Expected: Stable memory usage
   - Verification: No memory leaks during scroll

7. **Step 7 - Row Recycling**
   - Action: Inspect DOM during scroll
   - Expected: Rows recycled, not created new
   - Verification: DOM elements reused

8. **Step 8 - Data Integrity**
   - Action: Scroll and verify row data
   - Expected: Correct data in each position
   - Verification: No data duplication/loss

9. **Step 9 - Scroll Position Restore**
   - Action: Navigate away and return
   - Expected: Scroll position remembered
   - Verification: Returns to same position

10. **Step 10 - Filtered Data Scrolling**
    - Action: Apply filter, then scroll
    - Expected: Virtual scrolling works with filters
    - Verification: Performance maintained

### Expected Results
- **Performance Criteria**:
  - Initial load < 2 seconds
  - Scroll maintains 60 FPS
  - Jump navigation < 500ms
  - Memory usage stable

- **Rendering Criteria**:
  - Only visible rows in DOM
  - Smooth visual experience
  - No flashing or artifacts
  - Consistent row heights

- **Functionality Criteria**:
  - All data accessible via scroll
  - Selection works during scroll
  - Actions available on all rows
  - Filters don't break scrolling

### Error Scenarios
1. **Browser Limit**: Graceful degradation if too many items
2. **Slow Network**: Loading indicators during data fetch
3. **Memory Pressure**: Reduced performance but no crash
4. **Rapid Scrolling**: Temporary placeholder rows

### Validation Points
- Initial threshold applied: First 100 rows loaded
- DOM optimization: Limited row elements rendered
- Event handling: FirstVisibleRowChanged fires correctly
- Performance metrics: FPS and response times logged
- Memory stability: No leaks over extended use

### Test Data Requirements
- Minimum 1000 projects for testing
- Varied data in each row
- Mix of statuses and dates
- Some rows with long text
- Performance baseline metrics

### Performance Benchmarks
- **Initial Load**: < 2 seconds for first view
- **Scroll FPS**: Maintain 60 FPS (55+ acceptable)
- **Jump Time**: < 500ms to render new position
- **Memory**: < 50MB increase during session
- **CPU**: < 30% usage during scroll

### Browser Compatibility
- Chrome/Edge: Full virtual scrolling
- Firefox: Full support expected
- Safari: Test for any quirks
- Mobile: Touch scrolling smooth
- Tablet: Optimized for touch

### Implementation Details
- Uses sap.ui.table.Table
- Built-in virtual scrolling
- Threshold property for initial load
- Row recycling automatic
- Event hooks for monitoring

---

## Test Case ID: TC-UI-AGT-030
**Test Objective**: Verify lazy loading functionality for large project datasets  
**Business Process**: Project List Performance and Data Loading  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-030
- **Test Priority**: High (P1)
- **Test Type**: Functional, Performance, Data Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/ProjectsList.controller.js:76-114`
- **View File**: `a2aAgents/frontend/webapp/view/ProjectsList.view.xml:92-233`
- **Functions Under Test**: `_loadMoreProjects()`, `_setupVirtualScrolling()`, scroll event handling

### Test Preconditions
1. **Large Dataset**: 1000+ projects available in test data
2. **View State**: ProjectsList view loaded with table
3. **Browser**: Modern browser with DevTools access
4. **Network**: Simulated latency for realistic testing
5. **Initial Load**: First 100 projects loaded

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Initial Batch | 100 | Number | Controller config |
| Load More Batch | 100 | Number | Controller config |
| Total Projects | 1000 | Number | Test dataset |
| Scroll Trigger | 50 rows before end | Number | Lazy load threshold |
| Load Delay | 500ms | Number | Simulated network |

### Test Procedure Steps
1. **Step 1 - Initial Load Verification**
   - Action: Load ProjectsList view
   - Expected: Only first 100 projects loaded
   - Verification: Check model data count = 100

2. **Step 2 - Scroll Near Bottom**
   - Action: Scroll to row 50 (halfway)
   - Expected: No additional load triggered
   - Verification: Still 100 projects in model

3. **Step 3 - Trigger Lazy Load**
   - Action: Scroll to row 60+ (within 50 of end)
   - Expected: Loading indicator briefly shown
   - Verification: _loadMoreProjects() called

4. **Step 4 - Additional Data Load**
   - Action: Wait for load completion
   - Expected: Toast "Loaded 100 more projects"
   - Verification: Model now has 200 projects

5. **Step 5 - Continuous Scrolling**
   - Action: Continue scrolling down
   - Expected: Seamless experience, no gaps
   - Verification: No UI freezing or jank

6. **Step 6 - Multiple Load Cycles**
   - Action: Scroll to trigger 3rd batch
   - Expected: Another 100 projects loaded
   - Verification: Model has 300 projects

7. **Step 7 - Prevent Duplicate Loads**
   - Action: Rapid scroll during loading
   - Expected: No duplicate load requests
   - Verification: _bLoadingMore flag prevents

8. **Step 8 - End of Data**
   - Action: Scroll after all 1000 loaded
   - Expected: No more load attempts
   - Verification: _iLoadedCount = _iTotalCount

9. **Step 9 - Performance Check**
   - Action: Monitor scroll performance
   - Expected: Console logs show position
   - Verification: Smooth 60 FPS maintained

10. **Step 10 - Data Integrity**
    - Action: Check loaded project data
    - Expected: All projects unique, ordered
    - Verification: No duplicates or gaps

### Expected Results
- **Loading Behavior**:
  - Initial batch loads immediately
  - Additional batches load on demand
  - Loading indicator during fetch
  - Toast notification on completion
  
- **Scroll Triggering**:
  - Triggers 50 rows before end
  - No trigger if already loading
  - No trigger at data end
  - Smooth scroll continues
  
- **Performance Criteria**:
  - No UI blocking during load
  - < 500ms simulated delay
  - Memory efficient loading
  - No scroll position jump

### Error Scenarios
1. **Load Failure**: Error toast, retry capability
2. **Network Timeout**: Graceful failure, manual retry
3. **Memory Limit**: Stop loading, show warning
4. **Rapid Scrolling**: Queue management, no duplicates

### Validation Points
- Initial load count: `oModel.getProperty("/Projects").length === 100`
- Load trigger point: Within 50 rows of current end
- Loading flag: `this._bLoadingMore` prevents duplicates
- Total tracking: `this._iTotalCount` defines limit
- Scroll events: FirstVisibleRowChanged fires correctly

### Test Data Requirements
- Sequential project IDs (PRJ000000-PRJ000999)
- Consistent data structure per project
- Varied statuses for visual testing
- Timestamps for sorting verification
- Agent counts for data variety

### Performance Monitoring
- **Initial Load**: < 2 seconds
- **Lazy Load Batch**: < 1 second per 100
- **Scroll Performance**: 60 FPS maintained
- **Memory Growth**: Linear, not exponential
- **Network Calls**: One per batch only

### Implementation Verification
- Scroll event attached in setup
- Threshold calculation correct
- Loading flag management proper
- Data append (not replace)
- Count tracking accurate

### User Experience Requirements
- Clear loading feedback
- No scroll position loss
- Smooth visual transition
- Informative messages
- No duplicate data

---

## Test Case ID: TC-UI-AGT-031
**Test Objective**: Verify system performance and stability with 1000+ project items  
**Business Process**: Large Dataset Handling and System Scalability  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-031
- **Test Priority**: Critical (P1)
- **Test Type**: Performance, Stress Testing, Scalability
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/ProjectsList.controller.js`
- **View File**: `a2aAgents/frontend/webapp/view/ProjectsList.view.xml`
- **API Endpoint**: `/api/projects`
- **Functions Under Test**: Full table performance with large datasets

### Test Preconditions
1. **Database**: 1000+ real project records
2. **API**: Backend configured for pagination
3. **Browser**: Chrome/Edge with DevTools
4. **Network**: Production-like latency
5. **System**: Adequate memory (8GB+)

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Total Projects | 1000-5000 | Number | Database |
| Page Size | 100 | Number | API config |
| API Response Time | < 1s | Time | SLA |
| Memory Baseline | Current usage | MB | DevTools |
| Network Speed | Varies | Mbps | Test conditions |

### Test Procedure Steps
1. **Step 1 - Initial Performance Baseline**
   - Action: Open DevTools Performance tab
   - Expected: Record baseline metrics
   - Verification: Memory, CPU, FPS recorded

2. **Step 2 - Load View with Large Dataset**
   - Action: Navigate to ProjectsList view
   - Expected: Initial 100 projects load
   - Verification: API returns totalCount > 1000

3. **Step 3 - Verify Initial Render Time**
   - Action: Measure time to interactive
   - Expected: < 3 seconds for first paint
   - Verification: Performance timeline shows

4. **Step 4 - Scroll Through All Data**
   - Action: Scroll progressively to bottom
   - Expected: Lazy loading triggers properly
   - Verification: All 1000+ items accessible

5. **Step 5 - Memory Usage Monitoring**
   - Action: Check memory after each 500 items
   - Expected: Linear growth, not exponential
   - Verification: No memory leaks detected

6. **Step 6 - Interaction Performance**
   - Action: Click, select, hover on items
   - Expected: < 100ms response time
   - Verification: No UI lag or freezing

7. **Step 7 - Search Performance**
   - Action: Search within 1000+ items
   - Expected: < 500ms filter application
   - Verification: Smooth filtering experience

8. **Step 8 - Sort Large Dataset**
   - Action: Sort by different columns
   - Expected: < 1 second sort time
   - Verification: Correct sort order

9. **Step 9 - Concurrent Operations**
   - Action: Scroll while data loads
   - Expected: No blocking or crashes
   - Verification: Smooth multitasking

10. **Step 10 - Extended Usage Test**
    - Action: Use view for 10+ minutes
    - Expected: Stable performance
    - Verification: No degradation over time

### Expected Results
- **Performance Metrics**:
  - Initial load: < 3 seconds
  - Scroll FPS: > 30 (target 60)
  - Memory growth: < 200MB total
  - CPU usage: < 50% average
  
- **Functionality**:
  - All items accessible via scroll
  - No data loss or corruption
  - Consistent interaction speed
  - Proper error handling
  
- **Stability**:
  - No crashes or freezes
  - No memory leaks
  - Predictable behavior
  - Graceful degradation

### Error Scenarios
1. **API Timeout**: Show error, allow retry
2. **Memory Pressure**: Reduce rendered rows
3. **Network Failure**: Maintain loaded data
4. **Browser Limits**: Inform user of constraints

### Validation Points
- API pagination works: `skip` and `limit` parameters
- Total count accurate: Matches database count
- Memory efficient: Uses virtual scrolling
- No duplicate requests: Check network tab
- Data integrity: All items unique and complete

### Performance Benchmarks
- **Initial Load**: < 3 seconds (100 items)
- **Subsequent Loads**: < 1 second per batch
- **Total Load Time**: < 15 seconds for 1000 items
- **Memory Usage**: < 200MB increase
- **Frame Rate**: Maintain 30+ FPS

### Stress Test Scenarios
- 1000 projects: Normal operation
- 2500 projects: Good performance
- 5000 projects: Acceptable performance
- 10000 projects: Graceful handling
- 50000 projects: Consider alternatives

### Browser Compatibility
- **Chrome/Edge**: Full support expected
- **Firefox**: Test virtual scrolling
- **Safari**: Check memory handling
- **Mobile**: Not recommended for 1000+
- **Tablet**: Up to 2500 items

### Production Monitoring
- API response times logged
- Client-side performance metrics
- Error rates tracked
- User experience scores
- Resource usage alerts

---

## Test Case ID: TC-UI-AGT-032
**Test Objective**: Verify smooth scrolling behavior and visual performance in project table  
**Business Process**: User Interface Smoothness and Visual Performance  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-032
- **Test Priority**: High (P1)
- **Test Type**: Performance, User Experience, Visual
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/ProjectsList.controller.js:97-111`
- **View File**: `a2aAgents/frontend/webapp/view/ProjectsList.view.xml:92-110`
- **Functions Under Test**: `_optimizeTableRendering()`, scroll throttling, virtual scrolling

### Test Preconditions
1. **Dataset**: 200+ projects loaded for scroll testing
2. **Browser**: Chrome/Edge with performance monitoring
3. **Hardware**: Modern device (4GB+ RAM)
4. **Display**: Standard resolution (1920x1080+)
5. **CSS**: Smooth scroll styles applied

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Row Height | 48px | Number | CSS/Controller |
| Visible Rows | 15 | Number | Table config |
| Threshold | 50 | Number | Virtual scroll |
| Debounce Delay | 100ms | Time | Scroll throttling |
| Target FPS | 60 | Number | Performance target |

### Test Procedure Steps
1. **Step 1 - Initial Scroll Test**
   - Action: Load ProjectsList with 200+ items
   - Expected: Table renders with fixed row heights
   - Verification: All rows have consistent 48px height

2. **Step 2 - Smooth Scroll Down**
   - Action: Slowly scroll down using mouse wheel
   - Expected: Smooth 60 FPS scrolling
   - Verification: No jank or stuttering visible

3. **Step 3 - Fast Scroll Test**
   - Action: Rapidly scroll from top to bottom
   - Expected: Maintains visual smoothness
   - Verification: Virtual scrolling handles speed

4. **Step 4 - Keyboard Scrolling**
   - Action: Use arrow keys to scroll row by row
   - Expected: Smooth single-row transitions
   - Verification: No visual jumps or flicker

5. **Step 5 - Page Up/Down Test**
   - Action: Use Page Up/Page Down keys
   - Expected: Smooth page transitions
   - Verification: Consistent visual behavior

6. **Step 6 - Momentum Scrolling**
   - Action: Use trackpad/touch for momentum scroll
   - Expected: Natural deceleration curve
   - Verification: Smooth momentum physics

7. **Step 7 - Scroll Position Precision**
   - Action: Scroll to specific position
   - Expected: Accurate positioning
   - Verification: No overshoot or undershoot

8. **Step 8 - During Data Loading**
   - Action: Scroll while lazy loading triggers
   - Expected: No interruption to scrolling
   - Verification: Loading doesn't block scroll

9. **Step 9 - Performance Monitoring**
   - Action: Monitor FPS during scroll
   - Expected: Maintain 55+ FPS consistently  
   - Verification: DevTools performance tab

10. **Step 10 - Extended Scroll Session**
    - Action: Scroll continuously for 30 seconds
    - Expected: No performance degradation
    - Verification: Consistent frame rate maintained

### Expected Results
- **Visual Smoothness**:
  - 60 FPS target (55+ acceptable)
  - No visible stuttering or jank
  - Consistent row transitions
  - Natural scrolling feel
  
- **Performance Metrics**:
  - Frame time < 16.67ms (60 FPS)
  - No dropped frames during scroll
  - CPU usage < 30% during scroll
  - Memory stable during extended use
  
- **User Experience**:
  - Responsive to all input methods
  - Predictable scroll behavior
  - No visual artifacts
  - Smooth during data operations

### Error Scenarios
1. **Performance Degradation**: FPS drops below 30
2. **Visual Glitches**: Row flickering or jumping
3. **Scroll Blocking**: Loading operations interrupt scroll
4. **Memory Issues**: Performance degrades over time

### Validation Points
- Fixed row height set: `oTable.setRowHeight(48)`
- CSS class applied: `.a2a-smooth-scroll` present
- Throttling active: 100ms debounce on scroll events
- Virtual scrolling: Only visible rows rendered in DOM
- Performance metrics: FPS logged in development mode

### Test Data Requirements
- Minimum 200 projects for meaningful scroll
- Varied content lengths in cells
- Mixed data types (text, dates, numbers)
- Some rows with longer descriptions
- Representative real-world data

### Performance Benchmarks
- **Target FPS**: 60 (minimum 55)
- **Frame Time**: < 16.67ms average
- **Scroll Response**: < 10ms input to visual
- **Memory Growth**: < 5MB during test
- **CPU Usage**: < 30% peak during scroll

### Browser-Specific Considerations
- **Chrome/Edge**: Hardware acceleration enabled
- **Firefox**: Check for any scroll quirks
- **Safari**: Test momentum scrolling
- **Mobile Safari**: Touch scrolling behavior
- **IE11**: Fallback behavior (if supported)

### Accessibility Requirements
- Keyboard navigation smooth
- Screen reader compatible
- High contrast mode support
- Reduced motion preference respected
- Focus indicators visible during scroll

### Implementation Details
- Uses `setTimeout` for scroll event debouncing
- Fixed row heights eliminate reflow
- Virtual scrolling reduces DOM nodes
- CSS `scroll-behavior: smooth` applied
- Hardware acceleration via CSS transforms

### CSS Optimizations Required
```css
.a2a-smooth-scroll {
    scroll-behavior: smooth;
    will-change: scroll-position;
    -webkit-overflow-scrolling: touch;
}

.a2a-smooth-scroll .sapUiTableTr {
    will-change: transform;
    backface-visibility: hidden;
}
```

### Development Monitoring
- Console logs scroll position (dev only)
- Performance metrics collected
- Frame rate monitoring available
- Memory usage tracked
- Scroll event frequency logged

---

## Test Case ID: TC-UI-AGT-033
**Test Objective**: Verify scroll position persistence and restoration across navigation events  
**Business Process**: User Experience Continuity and Navigation State Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-033
- **Test Priority**: Medium (P2)
- **Test Type**: Functional, User Experience, State Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/frontend/webapp/controller/ProjectsList.controller.js:29-73,248-281`
- **Storage**: SessionStorage for scroll position persistence
- **Functions Under Test**: `_saveScrollPosition()`, `_restoreScrollPosition()`, `onExit()`

### Test Preconditions
1. **Project Data**: 50+ projects loaded for scrolling
2. **Browser**: SessionStorage supported and enabled
3. **Navigation**: Multiple routes/views available
4. **User Session**: Active session maintained
5. **JavaScript**: Enabled and working

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Storage Key | "ProjectsList_ScrollPosition" | String | Controller |
| Scroll Positions | Various row indices | Number | User interaction |
| Session Duration | 30+ minutes | Time | Browser session |
| Navigation Targets | Other views/routes | Routes | App routing |
| Data Changes | Add/remove projects | Operations | User actions |

### Test Procedure Steps
1. **Step 1 - Initial Navigation**
   - Action: Navigate to ProjectsList view
   - Expected: View loads with scroll at top (row 0)
   - Verification: No saved position in sessionStorage

2. **Step 2 - Scroll to Middle Position**
   - Action: Scroll down to around row 25 of 50+ projects
   - Expected: Smooth scrolling to position
   - Verification: Visual confirmation of position

3. **Step 3 - Navigate Away**
   - Action: Click to navigate to another view
   - Expected: Navigation proceeds normally
   - Verification: Scroll position saved to sessionStorage

4. **Step 4 - Check Storage**
   - Action: Inspect sessionStorage in DevTools
   - Expected: Key "ProjectsList_ScrollPosition" with value "25"
   - Verification: Storage contains correct row index

5. **Step 5 - Return to ProjectsList**
   - Action: Navigate back to ProjectsList view
   - Expected: View loads and restores to row 25
   - Verification: Same projects visible as before

6. **Step 6 - Multiple Navigation Cycles**
   - Action: Scroll to row 40, navigate away, return
   - Expected: Position restored to row 40
   - Verification: Correct position maintained

7. **Step 7 - Browser Refresh Test**
   - Action: While at row 30, refresh browser
   - Expected: Position restored after page reload
   - Verification: SessionStorage persists across refresh

8. **Step 8 - Data Change Impact**
   - Action: Scroll to row 20, add new projects, return
   - Expected: Position adjusted if data count changed
   - Verification: Scroll position remains reasonable

9. **Step 9 - Session Cleanup**
   - Action: Close tab or end session
   - Expected: Clean exit without errors
   - Verification: Event handlers properly detached

10. **Step 10 - Edge Case Testing**
    - Action: Try to restore position beyond data range
    - Expected: Position adjusted to valid range
    - Verification: No errors, reasonable position set

### Expected Results
- **Position Persistence**:
  - Scroll position saved before navigation
  - Position restored after returning
  - Survives browser refresh
  - Handles data changes gracefully
  
- **User Experience**:
  - Seamless navigation feel
  - No jarring position jumps
  - Consistent behavior
  - No performance impact
  
- **Technical Implementation**:
  - SessionStorage used correctly
  - Memory leaks prevented
  - Event handlers cleaned up
  - Error handling robust

### Error Scenarios
1. **Storage Disabled**: Graceful degradation, no errors
2. **Invalid Position**: Adjust to valid range automatically
3. **No Data**: Handle empty list scenario
4. **Storage Full**: Clear old positions if needed

### Validation Points
- Storage key created: `sessionStorage.getItem("ProjectsList_ScrollPosition")`
- Position saved on navigation: Value matches current scroll row
- Position restored on return: `setFirstVisibleRow()` called with saved value
- Cleanup on exit: `onExit()` removes event handlers
- Adjustment for data changes: Position within valid range

### Test Data Requirements
- Minimum 50 projects for meaningful scrolling
- Multiple views/routes for navigation testing
- Test scenarios with different data counts
- Projects with varied content for visual verification
- Stable project IDs for position reference

### Browser Storage Testing
- **SessionStorage**: Primary storage mechanism
- **Storage Limits**: Test within reasonable limits
- **Privacy Modes**: Test behavior in private browsing
- **Storage Events**: Verify proper event handling
- **Cross-Tab**: Test behavior across multiple tabs

### Performance Considerations
- **Save Operation**: < 1ms to save position
- **Restore Operation**: < 100ms to restore position
- **Memory Usage**: Minimal storage footprint
- **Event Overhead**: No noticeable impact on navigation
- **Cleanup**: Proper resource management

### Accessibility Requirements
- Position restoration doesn't affect screen readers
- Focus management remains intact
- Keyboard navigation works after restoration
- High contrast mode compatibility
- Reduced motion preferences respected

### Integration Points
- **Router Events**: BeforeRouteMatched handler
- **View Lifecycle**: onInit and onExit methods
- **Table Component**: FirstVisibleRow property
- **Model Changes**: Data update handling
- **Storage API**: SessionStorage operations

### Edge Cases
- Very large scroll positions (1000+ rows)
- Rapid navigation before restore completes
- Concurrent storage operations
- Storage quota exceeded
- Malformed storage values

### User Stories Covered
- "As a user, I want to return to the same scroll position when navigating back"
- "As a user, I want my scroll position to persist across browser refreshes"
- "As a developer, I want scroll position to adjust when data changes"

---

## Test Case ID: TC-UI-AGT-034
**Test Objective**: Verify hover actions and quick action functionality on project rows  
**Business Process**: Quick Project Actions and User Experience Enhancement  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-034
- **Test Priority**: High (P1)
- **Test Type**: Functional, User Experience, Interaction
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:29-87,271-316`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:175-242`
- **Functions Under Test**: `_onProjectRowHover()`, `_showQuickActions()`, `_hideQuickActions()`, quick action handlers

### Test Preconditions
1. **Projects Data**: Multiple projects loaded in table view
2. **Mouse/Touch Input**: Device with hover capability
3. **CSS Support**: Hover states and animations working
4. **JavaScript**: Event delegation and jQuery functional
5. **Browser**: Modern browser with CSS3 support

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Hover Duration | 500ms+ | Time | User interaction |
| Quick Actions | Open, Favorite, Share | Buttons | UI actions |
| Animation Speed | 200ms | Time | CSS transitions |
| Row Identifier | .a2a-project-row | CSS Class | View markup |
| Action Visibility | On hover only | Boolean | UX design |

### Test Procedure Steps
1. **Step 1 - Initial State Verification**
   - Action: Load projects table view
   - Expected: No quick actions visible initially
   - Verification: Quick action buttons hidden by default

2. **Step 2 - Mouse Hover Entry**
   - Action: Move mouse over first project row
   - Expected: Row highlights with hover effect
   - Verification: CSS class .a2a-project-row-hover applied

3. **Step 3 - Quick Actions Appearance**
   - Action: Wait 100ms after hover
   - Expected: Quick action buttons fade in smoothly
   - Verification: .a2a-quick-actions becomes visible with animation

4. **Step 4 - Quick Action Buttons**
   - Action: Verify quick action buttons visible
   - Expected: Open folder, favorite, and share icons shown
   - Verification: Three buttons with proper tooltips

5. **Step 5 - Hover Exit**
   - Action: Move mouse away from project row
   - Expected: Quick actions fade out and hide
   - Verification: .a2a-project-row-hover class removed

6. **Step 6 - Quick Open Action**
   - Action: Hover over row and click quick open button
   - Expected: Quick open action triggered
   - Verification: Toast message or navigation occurs

7. **Step 7 - Toggle Favorite Action**
   - Action: Click favorite button on project row
   - Expected: Favorite status toggles
   - Verification: Toast shows "Added to/Removed from favorites"

8. **Step 8 - Share Project Action**
   - Action: Click share button on project row
   - Expected: Share link copied to clipboard
   - Verification: Toast shows "Share link copied"

9. **Step 9 - Multiple Row Testing**
   - Action: Hover over different project rows
   - Expected: Each row shows its own quick actions
   - Verification: Actions work independently per row

10. **Step 10 - Performance Testing**
    - Action: Rapidly hover over multiple rows
    - Expected: Smooth hover transitions without lag
    - Verification: No flickering or performance issues

### Expected Results
- **Hover Interaction**:
  - Immediate visual feedback on hover entry
  - Smooth fade-in animation for quick actions
  - Clean fade-out when hover exits
  - No interference between rows
  
- **Quick Actions Functionality**:
  - Open button navigates to project workspace
  - Favorite button toggles project favorite status
  - Share button copies project link to clipboard
  - All actions provide user feedback
  
- **Performance and UX**:
  - < 100ms hover response time
  - Smooth animations without jank
  - Accessible keyboard navigation
  - Touch-friendly on tablets

### Error Scenarios
1. **JavaScript Disabled**: Quick actions not available, fallback to main actions
2. **CSS Disabled**: Actions visible but without styling
3. **Touch Device**: Touch and hold for hover simulation
4. **Clipboard API Unavailable**: Share shows URL in toast message

### Validation Points
- Hover event attached: Event delegation on table element
- CSS classes applied: .a2a-project-row-hover on hover
- Quick actions visibility: Display property changes on hover
- Action functionality: Each button triggers correct handler
- Cleanup on exit: Event handlers removed in onExit

### Test Data Requirements
- At least 5 projects for hover testing
- Projects with different statuses and names
- Mix of favorited and non-favorited projects
- Valid project IDs for share functionality
- Representative project data for testing

### Interaction Design Requirements
- **Hover Entry**: 100ms delay before showing actions
- **Hover Exit**: Immediate hide (no delay)
- **Animation**: Smooth 200ms CSS transitions
- **Spacing**: Adequate space for touch targets
- **Visibility**: Clear visual distinction on hover

### Accessibility Considerations
- Keyboard focus shows same actions
- Screen reader announces action availability
- High contrast mode compatibility
- Reduced motion preference respected
- ARIA labels on all action buttons

### Browser Compatibility
- **Chrome/Edge**: Full hover and animation support
- **Firefox**: Test CSS transitions
- **Safari**: Check webkit prefixes if needed
- **Mobile**: Touch and hold simulation
- **Tablet**: Hover via stylus or long press

### CSS Animation Requirements
```css
.a2a-project-row:hover {
    background-color: rgba(0, 120, 215, 0.05);
    transition: background-color 200ms ease;
}

.a2a-quick-actions {
    opacity: 0;
    transform: translateY(5px);
    transition: all 200ms ease;
}

.a2a-fade-in {
    opacity: 1;
    transform: translateY(0);
}
```

### Performance Benchmarks
- **Hover Response**: < 100ms visual feedback
- **Animation Duration**: 200ms for smooth feel
- **Event Handling**: No memory leaks on repeated hover
- **Multiple Rows**: Smooth interaction with 50+ rows
- **Touch Performance**: < 300ms touch response

### Touch Device Support
- Long press (500ms) triggers hover state
- Touch outside row hides actions
- Large enough touch targets (44px minimum)
- No accidental triggers from scrolling
- Proper touch feedback visual states

---

## Test Case ID: TC-UI-AGT-035
**Test Objective**: Verify tooltip display functionality and context-sensitive help across UI elements  
**Business Process**: User Assistance and Interface Guidance  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-035
- **Test Priority**: Medium (P2)
- **Test Type**: Functional, User Experience, Accessibility
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:92-169,344-362`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml` (tooltip attributes)
- **Functions Under Test**: `_enhanceTooltips()`, `_updateFilterButtonTooltip()`, `_updateExportButtonTooltip()`, `_updateDeleteButtonTooltip()`

### Test Preconditions
1. **UI Elements**: All buttons and controls with tooltips loaded
2. **I18n Resources**: Tooltip text resources available
3. **Mouse Input**: Device with hover capability
4. **JavaScript**: Tooltip enhancement functions working
5. **Browser**: Modern browser with CSS tooltip support

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Hover Duration | 500ms | Time | Tooltip show delay |
| Tooltip Position | Top center | Position | CSS configuration |
| Hide Delay | 100ms | Time | Tooltip hide delay |
| Context Changes | Selection count | Number | Dynamic content |
| Resource Keys | i18n tooltip keys | String | Text resources |

### Test Procedure Steps
1. **Step 1 - Static Tooltip Verification**
   - Action: Hover over Create Project button
   - Expected: Tooltip shows "Create a new agent project"
   - Verification: Tooltip appears after 500ms delay

2. **Step 2 - Icon Button Tooltips**
   - Action: Hover over refresh button (icon only)
   - Expected: Descriptive tooltip appears
   - Verification: "Refresh projects list" tooltip shown

3. **Step 3 - Search Field Tooltip**
   - Action: Hover over search field
   - Expected: Search help tooltip appears
   - Verification: "Search projects by name, description, or ID"

4. **Step 4 - View Toggle Tooltips**
   - Action: Hover over tiles view and table view buttons
   - Expected: Each shows specific view description
   - Verification: "Show projects as tiles/table" tooltips

5. **Step 5 - Context-Sensitive Filter Tooltip**
   - Action: Hover over filter button (no active filters)
   - Expected: "Filter projects by status, date, and criteria"
   - Verification: Default filter tooltip shown

6. **Step 6 - Active Filter Tooltip**
   - Action: Apply filters, then hover filter button
   - Expected: "Clear active filters" or similar
   - Verification: Tooltip changes based on filter state

7. **Step 7 - Selection-Based Export Tooltip**
   - Action: Select 0 projects, hover export button
   - Expected: "Export all projects to file"
   - Verification: Default export tooltip

8. **Step 8 - Selected Items Export Tooltip**
   - Action: Select 3 projects, hover export button
   - Expected: "Export 3 selected projects to file"
   - Verification: Dynamic count in tooltip

9. **Step 9 - Delete Button Context Tooltip**
   - Action: Select 2 projects, hover delete button
   - Expected: "Delete 2 selected projects"
   - Verification: Selection count reflected in tooltip

10. **Step 10 - Action Button Tooltips**
    - Action: Hover over edit, clone, archive, delete icons
    - Expected: Each shows clear action description
    - Verification: All action tooltips display correctly

### Expected Results
- **Static Tooltips**:
  - All buttons show descriptive tooltips
  - Tooltips appear within 500ms of hover
  - Text is clear and helpful
  - Position is appropriate and readable
  
- **Dynamic Tooltips**:
  - Content updates based on context
  - Selection counts reflected accurately
  - Filter state changes tooltip content
  - Real-time updates without page refresh
  
- **Tooltip Behavior**:
  - Consistent positioning across elements
  - Proper hide delay when hover exits
  - No tooltip flickering or overlap
  - Keyboard accessible alternatives

### Error Scenarios
1. **Missing I18n Text**: Show fallback text or key
2. **JavaScript Disabled**: Rely on native browser tooltips
3. **Context Update Failure**: Show static tooltip content
4. **Positioning Issues**: Ensure tooltips stay on screen

### Validation Points
- Tooltip attribute present: `tooltip="{i18n>keyName}"`
- Dynamic update called: Selection change triggers tooltip refresh
- Resource bundle access: `getResourceBundle().getText()` works
- Positioning configured: jQuery UI tooltip settings applied
- Timing correct: Show/hide delays as specified

### Test Data Requirements
- I18n resource bundle with tooltip texts
- Projects data for selection testing
- Filter configurations for state testing
- Multiple UI elements with tooltips
- Various selection scenarios (0, 1, multiple)

### Tooltip Content Examples
- **Create Project**: "Create a new agent development project"
- **Import**: "Import an existing project from file"
- **Search**: "Search projects by name, description, or agent ID"
- **Filter**: "Filter projects by status, date range, and criteria"
- **Export (none selected)**: "Export all visible projects to Excel"
- **Export (selected)**: "Export {0} selected projects to Excel"
- **Delete (selected)**: "Delete {0} selected projects permanently"

### Accessibility Requirements
- Screen reader compatible tooltips
- Keyboard focus shows tooltip equivalent
- High contrast mode support
- ARIA describedby relationships where appropriate
- Alternative access methods for tooltip content

### Browser Compatibility Testing
- **Chrome/Edge**: Full tooltip support
- **Firefox**: CSS and JavaScript tooltip behavior  
- **Safari**: WebKit tooltip rendering
- **Mobile**: Touch-based tooltip alternatives
- **Screen Readers**: ARIA tooltip support

### Performance Considerations
- **Show Delay**: 500ms prevents accidental triggers
- **Hide Delay**: 100ms allows for cursor adjustment
- **Memory**: No tooltip object leaks
- **DOM Impact**: Minimal tooltip element creation
- **Update Frequency**: Context updates only when needed

### I18n Resource Requirements
```json
{
  "createProjectTooltip": "Create a new agent development project",
  "importProjectTooltip": "Import an existing project from file or repository",
  "refreshProjectsTooltip": "Refresh the projects list to show latest changes",
  "searchProjectsTooltip": "Search projects by name, description, or agent ID",
  "filterProjectsTooltip": "Filter projects by status, date range, and other criteria",
  "filterActiveTooltip": "Clear active filters to show all projects",
  "exportAllProjectsTooltip": "Export all visible projects to Excel format",
  "exportSelectedProjectsTooltip": "Export {0} selected projects to Excel format",
  "deleteSelectedProjectsTooltip": "Delete {0} selected projects permanently",
  "selectProjectsToDeleteTooltip": "Select projects to enable delete function"
}
```

### Tooltip Implementation Requirements
- Dynamic content based on application state
- Consistent positioning and styling
- Proper z-index to appear above content
- Responsive design for different screen sizes
- Touch device alternatives (long press)

---

## Test Case ID: TC-UI-AGT-036
**Test Objective**: Verify quick open functionality with multiple access methods and navigation  
**Business Process**: Rapid Project Access and Productivity Enhancement  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-036
- **Test Priority**: High (P1)
- **Test Type**: Functional, User Experience, Productivity
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:365-609`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:224-241` (quick action buttons)
- **Functions Under Test**: `_performQuickOpen()`, `_validateProjectAccess()`, `_navigateToProjectWorkspace()`, keyboard shortcuts

### Test Preconditions
1. **Project Data**: Multiple projects with different statuses available
2. **User Permissions**: Access rights to test projects
3. **Navigation Setup**: Router and routing configuration working
4. **API Access**: Backend API endpoints responding
5. **Keyboard Input**: Physical or virtual keyboard available

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Keyboard Shortcuts | Ctrl+Enter, Ctrl+Shift+O | Keys | User input |
| Project Statuses | Active, Draft, Archived | States | Project data |
| Navigation Routes | projectWorkspace, projectEditor | Routes | Router config |
| Access Validation | /api/projects/{id}/access | API | Backend service |
| Analytics Tracking | Quick Open Project event | Event | Analytics service |

### Test Procedure Steps
1. **Step 1 - Hover Quick Open**
   - Action: Hover over project row, click quick open icon
   - Expected: Project workspace opens immediately
   - Verification: Navigation to correct project workspace

2. **Step 2 - Keyboard Quick Open (Selection)**
   - Action: Select project, press Ctrl+Enter (Cmd+Enter on Mac)
   - Expected: Selected project opens in workspace
   - Verification: Correct project loaded in appropriate view

3. **Step 3 - Keyboard Quick Open (No Selection)**
   - Action: Clear selection, press Ctrl+Enter
   - Expected: Message "Select a project to quick open"
   - Verification: User guidance provided

4. **Step 4 - Multiple Selection Quick Open**
   - Action: Select 2+ projects, press Ctrl+Enter
   - Expected: Message "Select only one project for quick open"
   - Verification: Clear instructions for single selection

5. **Step 5 - Quick Access Dialog**
   - Action: Press Ctrl+Shift+O
   - Expected: Quick open dialog appears with search
   - Verification: Dialog opens, search field focused

6. **Step 6 - Quick Access Search**
   - Action: Type partial project name in search field
   - Expected: Project list filters in real-time
   - Verification: Only matching projects visible

7. **Step 7 - Dialog Project Selection**
   - Action: Click on project in quick access dialog
   - Expected: Project opens immediately
   - Verification: Dialog closes, project workspace loads

8. **Step 8 - Access Validation Test**
   - Action: Attempt quick open on restricted project
   - Expected: "Project access denied" message
   - Verification: Access check performed, error handled

9. **Step 9 - Status-Based Navigation**
   - Action: Quick open projects with different statuses
   - Expected: Route varies by status (workspace/editor/viewer)
   - Verification: Correct view opens for each status

10. **Step 10 - Performance and Analytics**
    - Action: Monitor quick open response time and tracking
    - Expected: < 1 second load time, analytics recorded
    - Verification: Performance meets target, events logged

### Expected Results
- **Access Methods**:
  - Hover button triggers quick open
  - Ctrl+Enter opens selected project
  - Ctrl+Shift+O opens quick access dialog
  - Dialog search filters projects real-time
  
- **Navigation Behavior**:
  - Active projects → projectWorkspace route
  - Draft projects → projectEditor route  
  - Archived projects → projectViewer route
  - Default → projectDetails route
  
- **Validation and Security**:
  - Access permissions checked before navigation
  - Proper error messages for access denied
  - Loading states during validation
  - Fallback navigation methods

### Error Scenarios
1. **Network Failure**: Show error message, provide retry option
2. **Access Denied**: Clear permission error message
3. **Invalid Project**: Handle missing or corrupted project data
4. **Router Failure**: Fallback to direct URL navigation

### Validation Points
- Access validation: GET /api/projects/{id}/access called
- Route navigation: Router.navTo() called with correct parameters
- Analytics tracking: Quick Open Project event recorded
- Loading states: Busy indicator shown during validation
- Keyboard handling: Event listeners attached and responsive

### Test Data Requirements
- Projects with different statuses (Active, Draft, Archived)
- Mix of accessible and restricted projects
- Projects with various names for search testing
- Valid project IDs and routing configuration
- Analytics service or mock for event tracking

### Quick Open Access Methods
1. **Hover Action**: Quick open button in row hover overlay
2. **Keyboard Selection**: Ctrl+Enter on selected project
3. **Quick Access Dialog**: Ctrl+Shift+O opens searchable dialog
4. **Direct Click**: Project name link (traditional method)
5. **Context Menu**: Right-click quick open (future enhancement)

### Performance Requirements
- **Quick Open Response**: < 500ms from action to navigation
- **Access Validation**: < 1 second API response
- **Dialog Open**: < 200ms dialog appearance
- **Search Filtering**: Real-time with no lag
- **Navigation**: < 1 second to workspace load

### Keyboard Shortcuts
- **Ctrl+Enter** (Cmd+Enter): Quick open selected project
- **Ctrl+Shift+O** (Cmd+Shift+O): Open quick access dialog
- **Escape**: Close quick access dialog
- **Enter** in dialog: Open selected/highlighted project
- **Arrow keys**: Navigate dialog project list

### Accessibility Features
- Clear keyboard shortcuts with standard conventions
- Screen reader announcements for quick open actions
- Focus management in quick access dialog
- High contrast mode support for dialog
- ARIA labels for quick open buttons and actions

### Router Configuration Requirements
```javascript
{
  "projectWorkspace": {
    "pattern": "projects/{projectId}/workspace",
    "target": ["workspace"]
  },
  "projectEditor": {
    "pattern": "projects/{projectId}/edit", 
    "target": ["editor"]
  },
  "projectViewer": {
    "pattern": "projects/{projectId}/view",
    "target": ["viewer"]
  }
}
```

### Analytics Event Schema
```javascript
{
  "eventName": "Quick Open Project",
  "properties": {
    "projectId": "string",
    "projectName": "string", 
    "projectStatus": "string",
    "accessMethod": "hover|keyboard|dialog",
    "timestamp": "ISO string"
  }
}
```

### Browser Compatibility
- **Chrome/Edge**: Full keyboard shortcut support
- **Firefox**: Keyboard event handling tested
- **Safari**: Cmd key shortcuts on macOS
- **Mobile**: Touch alternatives for quick access
- **Keyboard Navigation**: Full accessibility support

---

## Test Case ID: TC-UI-AGT-037
**Test Objective**: Verify inline quick edit functionality for project properties  
**Business Process**: Rapid Project Information Updates and Content Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-037
- **Test Priority**: High (P1)
- **Test Type**: Functional, User Experience, Data Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:614-801`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:194-266`
- **Functions Under Test**: `_startQuickEdit()`, `_validateAndSaveQuickEdit()`, `_cancelQuickEdit()`, validation methods

### Test Preconditions
1. **Project Data**: Editable projects available in table view
2. **User Permissions**: Edit rights for test projects
3. **API Access**: PATCH /api/projects/{id} endpoint working
4. **UI State**: Projects table loaded and visible
5. **Input Validation**: Client-side validation working

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Project Name | "Test Project Updated" | String | User input |
| Description | "Updated description text" | String | User input |
| Name Length Limit | 100 characters | Number | Validation rule |
| Description Limit | 500 characters | Number | Validation rule |
| API Endpoint | /api/projects/{id} | URL | Backend service |

### Test Procedure Steps
1. **Step 1 - Enter Edit Mode**
   - Action: Hover over project row, click quick edit (pencil) icon
   - Expected: Row switches to edit mode with input fields
   - Verification: Name input and description textarea visible

2. **Step 2 - Focus Management**
   - Action: Observe focus after entering edit mode
   - Expected: Project name field is focused and selected
   - Verification: Text cursor in name field, text highlighted

3. **Step 3 - Edit Project Name**
   - Action: Change project name to "Updated Project Name"
   - Expected: Input accepts changes, no validation errors
   - Verification: Name field shows new value

4. **Step 4 - Name Validation - Empty**
   - Action: Clear project name completely
   - Expected: Error state with "Project name is required"
   - Verification: Input shows error state, validation message

5. **Step 5 - Name Validation - Length**
   - Action: Enter name longer than 100 characters
   - Expected: Error state with length warning
   - Verification: Character limit validation active

6. **Step 6 - Edit Description**
   - Action: Modify project description
   - Expected: TextArea accepts multi-line input
   - Verification: Description updates correctly

7. **Step 7 - Description Length Warning**
   - Action: Enter description approaching 500 characters
   - Expected: Warning state with character count
   - Verification: "Description is getting long (450/500 chars)"

8. **Step 8 - Save Changes (Button)**
   - Action: Click save (checkmark) button
   - Expected: API call made, success message shown
   - Verification: PATCH request sent, "Project updated successfully"

9. **Step 9 - Save Changes (Enter Key)**
   - Action: Press Enter in name or description field
   - Expected: Changes saved automatically
   - Verification: Same as button save behavior

10. **Step 10 - Cancel Changes**
    - Action: Make changes, then click cancel (X) button
    - Expected: Original values restored, edit mode exits
    - Verification: Changes reverted, "Changes cancelled" message

### Expected Results
- **Edit Mode Entry**:
  - Row transforms to editable form
  - Input fields replace display text
  - Save/cancel buttons appear
  - Focus set to name field
  
- **Real-time Validation**:
  - Name required validation immediate
  - Character length limits enforced
  - Error states clearly indicated
  - Warning states for approaching limits
  
- **Save Functionality**:
  - API call sends correct data
  - Success feedback provided
  - Model updated with server response
  - Edit mode exits cleanly

### Error Scenarios
1. **Network Failure**: Show error message, maintain edit mode
2. **Validation Failure**: Prevent save, show specific errors
3. **Server Error**: Display server error message
4. **Concurrent Edit**: Handle edit conflicts gracefully

### Validation Points
- Edit mode state: `view>/editMode` set to project ID
- Original values stored: `view>/originalValues` populated
- Input validation: ValueState set on validation errors
- API call: PATCH /api/projects/{id} with correct payload
- Model binding: Changes reflected in UI immediately

### Test Data Requirements
- Projects with different name and description lengths
- Valid and invalid input scenarios
- Server responses for success and error cases
- Projects with special characters in names/descriptions
- Edge cases for validation limits

### Inline Editing Features
- **Direct Editing**: Click to edit without dialog
- **Auto-focus**: Name field focused on edit start
- **Real-time Validation**: Immediate feedback on input
- **Keyboard Shortcuts**: Enter to save, Escape to cancel
- **Visual Indicators**: Clear edit/save/cancel buttons

### Validation Rules
- **Project Name**: Required, 1-100 characters, trimmed
- **Description**: Optional, 0-500 characters, trimmed
- **Special Characters**: Allowed in both fields
- **Whitespace**: Leading/trailing spaces trimmed
- **Unicode**: Full Unicode support for international names

### User Experience Requirements
- **Response Time**: < 200ms to enter edit mode
- **Save Time**: < 2 seconds for save operation
- **Visual Feedback**: Clear states for view/edit/saving
- **Error Handling**: Specific, helpful error messages
- **Accessibility**: Screen reader support for state changes

### API Integration
- **Endpoint**: PATCH /api/projects/{projectId}
- **Request Body**: `{ "name": "string", "description": "string" }`
- **Success Response**: Updated project object
- **Error Response**: Error details with specific message
- **Optimistic Updates**: UI updates before API confirmation

### Accessibility Features
- ARIA labels for edit mode controls
- Screen reader announcements for state changes
- Keyboard navigation through edit controls
- High contrast support for edit states
- Focus indicators clearly visible

### Browser Compatibility
- **Input Focus**: Works across browsers
- **Validation States**: Consistent visual feedback
- **Keyboard Events**: Enter/Escape handling
- **API Calls**: XHR/Fetch compatibility
- **Mobile Support**: Touch-friendly edit controls

### Performance Considerations
- **Edit Mode Toggle**: < 100ms transition
- **Validation**: Real-time without lag
- **Save Operation**: Optimistic UI updates
- **Memory**: No leaks from edit state
- **Concurrent Edits**: Only one row editable at time

### Security Considerations
- Input sanitization on client and server
- XSS prevention in project names/descriptions
- Authentication required for edit operations
- Authorization check before allowing edits
- Audit trail for project modifications

---

## Test Case ID: TC-UI-AGT-038
**Test Objective**: Verify action responsiveness and performance monitoring across all UI interactions  
**Business Process**: User Interface Performance and Response Time Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-038
- **Test Priority**: Critical (P1)
- **Test Type**: Performance, User Experience, System Monitoring
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:807-1044`
- **Functions Under Test**: `_initializeResponseMonitoring()`, `_recordActionStart()`, `_recordActionComplete()`, performance tracking

### Test Preconditions
1. **Performance API**: Browser supports performance.now() timing
2. **Project Data**: Sufficient data for performance testing
3. **Network Conditions**: Various network speeds for API testing
4. **Browser**: Modern browser with DevTools access
5. **Monitoring**: Action responsiveness monitoring active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Response Target | < 100ms | Time | UX standards |
| Slow Action Threshold | 300ms | Time | Warning threshold |
| API Response Target | < 1000ms | Time | Performance SLA |
| Render Target | < 100ms | Time | Smooth UI |
| Success Rate Target | > 95% | Percentage | Quality metric |

### Test Procedure Steps
1. **Step 1 - Initialize Performance Monitoring**
   - Action: Load Projects view
   - Expected: Monitoring system initialized
   - Verification: `_actionMetrics` object created with arrays

2. **Step 2 - Button Click Response Time**
   - Action: Click various buttons (Create, Edit, Delete, etc.)
   - Expected: Visual feedback within 100ms
   - Verification: Loading indicators appear immediately

3. **Step 3 - Hover Response Testing**
   - Action: Hover over project rows rapidly
   - Expected: Hover effects within 50ms
   - Verification: No lag in hover state changes

4. **Step 4 - Search Input Responsiveness**
   - Action: Type quickly in search field
   - Expected: Real-time filtering without delay
   - Verification: Results update with each keystroke

5. **Step 5 - Scroll Performance**
   - Action: Scroll through project list rapidly
   - Expected: Smooth 60 FPS scrolling
   - Verification: No frame drops or stuttering

6. **Step 6 - API Call Performance**
   - Action: Trigger save, delete, and update operations
   - Expected: API responses < 1 second
   - Verification: Network tab shows response times

7. **Step 7 - Concurrent Action Testing**
   - Action: Perform multiple actions simultaneously
   - Expected: No blocking or queue delays
   - Verification: Actions process independently

8. **Step 8 - Slow Network Simulation**
   - Action: Throttle network, perform actions
   - Expected: Appropriate loading states shown
   - Verification: User feedback during slow operations

9. **Step 9 - Memory Performance**
   - Action: Perform 50+ actions, monitor memory
   - Expected: No memory leaks or excessive growth
   - Verification: DevTools memory tab stable

10. **Step 10 - Performance Metrics Collection**
    - Action: Call `getPerformanceMetrics()` method
    - Expected: Detailed performance statistics
    - Verification: Metrics show response times and success rates

### Expected Results
- **Response Time Standards**:
  - Button clicks: < 100ms visual feedback
  - Hover effects: < 50ms state change
  - Search filtering: Real-time (< 50ms)
  - API calls: < 1000ms average
  
- **Performance Monitoring**:
  - All actions tracked automatically
  - Slow actions logged and identified
  - API performance measured
  - Render times monitored
  
- **User Experience**:
  - No perceptible delays in interactions
  - Smooth animations and transitions
  - Clear loading states for slow operations
  - Responsive feedback for all actions

### Error Scenarios
1. **Slow Actions**: Warning logged, visual indicator shown
2. **Failed API Calls**: Error tracked, user notified
3. **Memory Leaks**: Performance degradation detected
4. **Network Issues**: Appropriate timeout handling

### Validation Points
- Action start recorded: `_recordActionStart()` called on button clicks
- Action complete tracked: Duration and success status recorded
- Performance thresholds: Warnings for actions > 100ms
- Metrics collection: `_actionMetrics` populated with data
- Visual feedback: Loading indicators shown during actions

### Test Data Requirements
- Various action types for comprehensive testing
- Different data sizes for performance scaling
- Network conditions (fast, slow, offline)
- Multiple concurrent user scenarios
- Edge cases for error handling

### Performance Monitoring Features
- **Automatic Tracking**: All button clicks monitored
- **Threshold Warnings**: Slow actions flagged in console
- **API Performance**: Request/response times recorded
- **Render Monitoring**: View rendering performance tracked
- **Success Rates**: Action success/failure statistics

### Responsiveness Targets
- **Immediate Feedback**: < 50ms for hover states
- **Click Response**: < 100ms for button feedback
- **Search Results**: < 50ms for real-time filtering
- **Navigation**: < 200ms for view transitions
- **API Operations**: < 1000ms for backend calls

### Performance Metrics Available
```javascript
{
  clickResponses: { average, min, max, count },
  apiCalls: { average, min, max, count },
  totalActions: number,
  slowActions: Array,
  failedActions: Array,
  apiSuccessRate: percentage
}
```

### Visual Feedback Requirements
- **Loading States**: Immediate visual indication
- **Progress Indicators**: For operations > 500ms
- **Success Confirmation**: Clear completion feedback
- **Error Handling**: Specific error messages
- **State Changes**: Smooth visual transitions

### Browser Performance Tools
- **Performance Tab**: Frame rate and timing analysis
- **Network Tab**: API response time measurement
- **Memory Tab**: Memory usage and leak detection
- **Console**: Performance warnings and metrics
- **DevTools API**: Programmatic performance access

### Automated Performance Testing
- Response time measurement for all actions
- Success rate calculation over time
- Performance regression detection
- Automated slow action reporting
- Memory usage trend analysis

### Accessibility Performance
- Screen reader response times
- Keyboard navigation speed
- Focus management performance
- ARIA updates timing
- High contrast mode impact

### Mobile Performance Considerations
- Touch response times
- Gesture recognition speed
- Battery usage optimization
- Network efficiency on mobile
- Reduced animation on low-power mode

### Performance Debugging Tools
- `getPerformanceMetrics()`: Get current statistics
- `logPerformanceSummary()`: Console performance report
- Action timing in browser DevTools
- Network waterfall analysis
- Memory profiling capabilities

### Continuous Performance Monitoring
- Real-time metrics collection
- Performance trend analysis
- Automatic slow action detection
- User experience impact measurement
- Performance regression alerts

---

## Summary Statistics
**Total Test Cases**: 43 UI Test Cases for A2A Agents (38 migrated legacy cases)
**Coverage**: Primary agent development workflow functionality + workspace, navigation, search, notification, user profile, theme management, keyboard accessibility, project views, search, sorting, filtering, filter combinations, clear filters, status visualization, project CRUD operations, virtual scrolling, lazy loading, large dataset performance, smooth scrolling, scroll position persistence, hover actions, tooltip display, quick open functionality, quick edit functionality, action responsiveness, column configuration, density settings, saved views, bulk operations, and comprehensive export functionality  
**Compliance**: ISO/IEC/IEEE 29119-3:2021 + SAP Solution Manager Format  
**Priority Distribution**: 12 Critical, 24 High, 7 Medium  

### Standard Compliance Verification
- ✅ **ISO 29119-3 Elements**: Complete test specification with all required components
- ✅ **SAP Elements**: Developer portal UX aligned with SAP Build Work Zone patterns
- ✅ **Development Workflow**: Complete IDE and development lifecycle coverage
- ✅ **Quality Assurance**: Testing and debugging capabilities validated
- ✅ **Operations**: Deployment and monitoring functionality tested

---

## Test Case ID: TC-UI-AGT-039
**Test Objective**: Verify table column configuration functionality  
**Business Process**: Projects View Customization  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-039
- **Test Priority**: Medium (P3)
- **Test Type**: Functional, UI Customization
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:1046-1223`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:106-108`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/ColumnConfigDialog.fragment.xml:1-80`
- **Functions Under Test**: `onOpenColumnConfigDialog()`, `_initializeColumnConfigModel()`, `onColumnVisibilityChange()`, `onSaveColumnConfig()`, `_saveColumnConfiguration()`, `_loadColumnConfiguration()`

### Test Preconditions
1. **User Authentication**: Valid developer account with project access rights
2. **Projects Data**: At least 5 projects visible in table view
3. **Table View**: Projects table displayed with all default columns
4. **Browser Support**: Modern browser with localStorage support
5. **Responsive Layout**: Screen width sufficient for dialog display

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Table Columns | Name, Description, Agents, Last Modified, Status, Actions | Array | Table Definition |
| Default Visibility | All visible | Boolean Array | Default Configuration |
| User ID | current_user | String | Session Context |
| Storage Key | a2a_projects_column_config_current_user | String | Local Storage |
| API Endpoint | /api/user/preferences/column-config | URL | Backend Service |

### Test Procedure Steps
1. **Step 1 - Access Column Configuration**
   - Action: Click column configuration button (gear icon) in toolbar
   - Expected: Column configuration dialog opens
   - Verification: Dialog title "Configure Columns", list of all columns displayed

2. **Step 2 - Review Current Configuration**
   - Action: Examine column list in dialog
   - Expected: All 6 columns listed with current visibility states
   - Verification: Switch states match actual column visibility

3. **Step 3 - Hide Column**
   - Action: Toggle off "Description" column switch
   - Expected: Switch updates to off position, preview updates
   - Verification: Description column immediately hidden in background table

4. **Step 4 - Show Hidden Column**
   - Action: Toggle on previously hidden column
   - Expected: Switch updates to on position, column reappears
   - Verification: Column visible again with correct content

5. **Step 5 - Multiple Column Changes**
   - Action: Hide "Agents" and "Last Modified" columns
   - Expected: Both switches update, multiple columns hidden
   - Verification: Table shows only Name, Status, Actions columns

6. **Step 6 - Preview Functionality**
   - Action: Expand preview section in dialog
   - Expected: Mini table shows current visibility configuration
   - Verification: Preview reflects actual column states

7. **Step 7 - Save Configuration**
   - Action: Click Save button
   - Expected: Configuration saved, success message shown
   - Verification: "Column configuration saved" toast appears

8. **Step 8 - Persistence Test**
   - Action: Refresh page or navigate away and back
   - Expected: Saved column configuration persists
   - Verification: Previously hidden columns remain hidden

9. **Step 9 - Reset to Defaults**
   - Action: Open dialog, click Reset button
   - Expected: All columns visible, default configuration restored
   - Verification: All 6 columns shown, switches all on

10. **Step 10 - Cancel Changes**
    - Action: Make changes but click Cancel
    - Expected: Changes discarded, original state restored
    - Verification: Table returns to state before dialog opened

### Expected Results
- **Dialog Behavior Criteria**:
  - Configuration dialog opens within 300ms
  - Column list shows all table columns with labels
  - Switch controls responsive (< 100ms toggle)
  - Real-time preview of changes in background table

- **Configuration Management Criteria**:
  - Changes applied immediately for preview
  - Save persists configuration in localStorage and backend
  - Cancel restores original state completely
  - Reset returns to system default visibility

- **Persistence Criteria**:
  - Configuration survives page refresh
  - Settings restored on view initialization
  - Backend sync maintains cross-session consistency
  - Error handling if storage unavailable

### Error Scenarios
1. **Storage Unavailable**: Show warning "Configuration changes will not persist"
2. **Backend Save Failed**: Show message "Local changes saved, sync will retry"  
3. **Invalid Configuration**: Reset to defaults with notification
4. **Dialog Load Error**: Show basic column toggle fallback

### Validation Points
- Button present: Column config button with gear icon in toolbar
- Dialog implementation: Fragment correctly loaded and bound
- Live preview: Changes reflected immediately in background table
- Persistence: localStorage saves/loads configuration correctly
- Backend sync: API calls made for cross-session persistence

### Test Data Requirements
- Projects table with all 6 standard columns populated
- Test user with preference save permissions
- Various screen sizes for responsive dialog testing

### Browser Storage Testing
- **localStorage Available**: Full functionality with persistence
- **localStorage Disabled**: Warning shown, session-only changes
- **localStorage Full**: Graceful degradation with error message
- **Cross-Tab Sync**: Configuration updates across browser tabs

### API Integration Testing
| Endpoint | Method | Payload | Response |
|----------|--------|---------|----------|
| `/api/user/preferences/column-config` | POST | `{"view": "projects", "configuration": [...]}` | 200 OK |
| Error Handling | POST | Invalid data | 400/500 with fallback |

### Accessibility Requirements
- Dialog keyboard accessible (Tab, Enter, Escape)
- Switch controls have proper ARIA labels
- Screen reader announces column state changes
- Focus management within dialog
- High contrast mode support

### Performance Considerations
- **Dialog Load**: < 300ms open time
- **Switch Response**: < 100ms toggle feedback  
- **Save Operation**: < 500ms persistence complete
- **Preview Updates**: < 50ms background table changes
- **Memory Usage**: Minimal dialog instance management

---

## Test Case ID: TC-UI-AGT-040
**Test Objective**: Verify table density settings functionality  
**Business Process**: Projects View Customization  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-040
- **Test Priority**: Medium (P3)
- **Test Type**: Functional, UI Customization
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:1231-1356`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:109-129`
- **Functions Under Test**: `onSetDensity()`, `_applyDensity()`, `_updateTableRowHeights()`, `_loadDensitySettings()`, `_saveDensitySettings()`, `getCurrentDensity()`

### Test Preconditions
1. **User Authentication**: Valid developer account with project access rights
2. **Projects Data**: At least 5 projects visible in table view for density comparison
3. **Table View**: Projects table displayed in default cozy density
4. **Browser Support**: Modern browser with localStorage and CSS support
5. **Responsive Layout**: Screen width sufficient for menu display

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Density Options | Cozy, Compact, Condensed | String Array | Menu Items |
| Row Heights | 48px (Cozy), 32px (Compact), 28px (Condensed) | Integer | Density Configuration |
| CSS Classes | sapUiSizeCozy, sapUiSizeCompact, sapUiSizeCondensed | String | SAP UI5 Classes |
| Storage Key | a2a_projects_density_current_user | String | Local Storage |
| API Endpoint | /api/user/preferences/density | URL | Backend Service |

### Test Procedure Steps
1. **Step 1 - Access Density Menu**
   - Action: Click density settings menu button (customize icon) in toolbar
   - Expected: Dropdown menu opens with three density options
   - Verification: Menu shows Cozy, Compact, and Condensed options with icons

2. **Step 2 - Default Density Verification**
   - Action: Observe current table appearance and row heights
   - Expected: Table displayed in cozy density (48px rows)
   - Verification: Adequate spacing between rows, comfortable reading

3. **Step 3 - Switch to Compact Density**
   - Action: Select "Compact" from density menu
   - Expected: Table row height reduces to 32px immediately
   - Verification: More rows visible in same viewport, toast message shown

4. **Step 4 - Visual Density Validation**
   - Action: Compare table appearance before and after density change
   - Expected: Rows closer together, text still readable
   - Verification: No text overlap or clipping occurs

5. **Step 5 - Switch to Condensed Density**
   - Action: Select "Condensed" from density menu
   - Expected: Table row height reduces to 28px, most compact view
   - Verification: Maximum information density while maintaining usability

6. **Step 6 - Return to Cozy Density**
   - Action: Select "Cozy" from density menu
   - Expected: Table returns to comfortable spacing (48px rows)
   - Verification: Original appearance restored

7. **Step 7 - Persistence Test**
   - Action: Change to compact density, refresh page
   - Expected: Compact density setting persists across page loads
   - Verification: Table loads with saved density setting

8. **Step 8 - Cross-Component Impact**
   - Action: Navigate to different view and return
   - Expected: Density setting applies globally to the view
   - Verification: All UI components use selected density

9. **Step 9 - Error Handling Test**
   - Action: Simulate localStorage unavailable scenario
   - Expected: Graceful fallback to default cozy density
   - Verification: No errors, functionality continues

10. **Step 10 - API Persistence**
    - Action: Change density and verify backend call
    - Expected: API call made to save preference
    - Verification: POST to /api/user/preferences/density with correct data

### Expected Results
- **Menu Interaction Criteria**:
  - Menu opens within 200ms of button click
  - Three density options clearly labeled with appropriate icons
  - Menu items respond to selection immediately
  - Menu closes after selection

- **Visual Density Criteria**:
  - Cozy: 48px row height, comfortable spacing
  - Compact: 32px row height, reduced spacing
  - Condensed: 28px row height, minimal spacing
  - Text readability maintained in all densities

- **Persistence Criteria**:
  - Selection saved to localStorage immediately
  - Settings restored on view initialization
  - Backend API called for cross-session sync
  - Error handling if storage unavailable

### Error Scenarios
1. **Storage Unavailable**: Default to cozy, show temporary warning
2. **Invalid Density Value**: Reset to default cozy density
3. **Backend Save Failed**: Local setting preserved, retry on next change
4. **CSS Class Conflicts**: Remove all density classes before applying new

### Validation Points
- Menu button present: Customize icon in toolbar after column config button
- Menu items configured: Three density options with correct customData
- Row height changes: Table updates immediately on selection
- CSS classes applied: Correct SAP UI5 density classes on view
- Persistence working: localStorage and API calls successful

### Test Data Requirements
- Projects table with sufficient rows to show density differences
- Test user with preference save permissions
- Various screen sizes for density impact testing

### SAP UI5 Density Standards
| Density | Row Height | Use Case | CSS Class |
|---------|------------|----------|-----------|
| Cozy | 48px | Touch devices, accessibility | sapUiSizeCozy |
| Compact | 32px | Desktop standard | sapUiSizeCompact |
| Condensed | 28px | Data-heavy scenarios | sapUiSizeCondensed |

### Accessibility Considerations
- **Touch Targets**: Cozy density maintains 44px minimum touch target
- **Visual Clarity**: All densities maintain adequate contrast
- **Screen Readers**: Density changes announced appropriately
- **Keyboard Navigation**: Focus indicators work in all densities
- **Low Vision**: Condensed density may require zoom support

### Performance Considerations
- **CSS Application**: < 50ms for class changes to take effect
- **Menu Response**: < 200ms from click to menu open
- **Row Rendering**: Table reflow completed within 300ms
- **Storage Operations**: < 100ms for localStorage read/write
- **API Calls**: Non-blocking backend preference save

### Browser Compatibility
- **Modern Browsers**: Full CSS density class support
- **Mobile Browsers**: Touch-optimized density handling
- **High-DPI Displays**: Consistent density appearance
- **Reduced Motion**: Respect prefers-reduced-motion settings
- **Legacy Browsers**: Graceful fallback to default density

---

## Test Case ID: TC-UI-AGT-041
**Test Objective**: Verify saved views functionality for complete state persistence  
**Business Process**: Projects View Personalization  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-041
- **Test Priority**: High (P2)
- **Test Type**: Functional, State Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:1361-1788`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:130-144`
- **Save Dialog**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/SaveViewDialog.fragment.xml:1-55`
- **Manage Dialog**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/ManageViewsDialog.fragment.xml:1-90`
- **Functions Under Test**: `_initializeSavedViews()`, `getCurrentViewState()`, `onSaveCurrentView()`, `_saveView()`, `onApplySavedView()`, `_applyViewState()`, `onManageSavedViews()`

### Test Preconditions
1. **User Authentication**: Valid developer account with project access rights
2. **Projects Data**: At least 10 projects for meaningful state testing
3. **View Customization**: Various filters, sorting, and column configs available
4. **Browser Support**: Modern browser with localStorage and JSON support
5. **Screen Size**: Sufficient space for dialogs and complex state display

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| View States | Multiple combinations | Object | User Configurations |
| View Names | "My Daily View", "Status Review", "Full Details" | String | User Input |
| Default View | Boolean flag | Boolean | User Selection |
| Storage Key | a2a_projects_saved_views_current_user | String | Local Storage |
| API Endpoint | /api/user/preferences/saved-views | URL | Backend Service |

### Test Procedure Steps
1. **Step 1 - Setup Custom View State**
   - Action: Apply filters (status=active), change density (compact), hide description column
   - Expected: View reflects all customizations
   - Verification: Table shows filtered/customized data

2. **Step 2 - Access Save View Dialog**
   - Action: Click saved views menu button, select "Save Current View"
   - Expected: Save view dialog opens with current state summary
   - Verification: Dialog shows view mode, density, filters, sorting details

3. **Step 3 - Save View with Name**
   - Action: Enter "My Daily View" as name, "Filtered active projects" as description
   - Expected: Input validation works, save button enabled
   - Verification: Required field validation, character limits enforced

4. **Step 4 - Confirm View Save**
   - Action: Click Save button
   - Expected: View saved successfully, toast message shown
   - Verification: "View 'My Daily View' saved successfully" message

5. **Step 5 - Verify Menu Update**
   - Action: Open saved views menu again
   - Expected: Menu shows saved view with bookmark icon
   - Verification: "My Daily View" appears in menu below save/manage options

6. **Step 6 - Change View State**
   - Action: Clear filters, change to cozy density, show all columns
   - Expected: View changes to different configuration
   - Verification: Visual differences from saved state

7. **Step 7 - Apply Saved View**
   - Action: Open saved views menu, select "My Daily View"
   - Expected: All saved settings restored instantly
   - Verification: Filters, density, columns match saved state

8. **Step 8 - Save as Default View**
   - Action: Create another view and mark "Set as Default"
   - Expected: View saved with default flag
   - Verification: Default view shows with star icon in menu

9. **Step 9 - Page Refresh Test**
   - Action: Refresh page or navigate away and return
   - Expected: Default view automatically applied
   - Verification: View state matches default view configuration

10. **Step 10 - Manage Views Dialog**
    - Action: Open saved views menu, select "Manage Views"
    - Expected: Management dialog opens with views table
    - Verification: All saved views listed with details and actions

11. **Step 11 - Delete Saved View**
    - Action: Click delete button for non-default view
    - Expected: Confirmation dialog, then view removed
    - Verification: View no longer appears in menu or management table

12. **Step 12 - View State Persistence**
    - Action: Close browser, restart, navigate to projects
    - Expected: Saved views and default view persist across sessions
    - Verification: All saved views intact, default applied automatically

### Expected Results
- **Save Functionality Criteria**:
  - Complete view state captured (filters, sorting, columns, density, view mode)
  - Dialog validation prevents empty names
  - Save operation completes within 500ms
  - Success feedback clearly displayed

- **Apply Functionality Criteria**:
  - Saved state restored completely and accurately
  - All UI elements update to match saved configuration
  - Apply operation completes within 1 second
  - Visual feedback confirms view application

- **Management Criteria**:
  - Views table shows all saved views with metadata
  - Delete operations work with confirmation
  - Default view designation properly enforced
  - Menu updates reflect changes immediately

### Error Scenarios
1. **Storage Unavailable**: Show warning "Views will not persist across sessions"
2. **Invalid View Data**: Reset to system default with notification
3. **Backend Sync Failed**: Local views maintained, retry on next save
4. **Name Conflict**: Offer to overwrite existing view with same name

### Validation Points
- Menu button present: Save icon in toolbar after density menu
- Complete state capture: All UI customizations included in saved state
- Menu updates: Dynamic menu reflects current saved views
- Persistence working: localStorage and backend API calls successful
- Default view logic: Proper default handling and application

### Test Data Requirements
- Multiple projects with different statuses for filter testing
- Various project data to test column visibility impact
- Multiple view configurations to test state differences

### State Elements Tested
| Element | Test Coverage | Validation Method |
|---------|---------------|-------------------|
| View Mode | Tiles/Table switching | Visual verification |
| Density | Cozy/Compact/Condensed | Row height measurement |
| Column Config | Show/Hide columns | Column visibility check |
| Filters | Search, status, date | Filter application check |
| Sorting | Field and direction | Table sort order verification |
| Search Terms | Text input preservation | Search field value check |

### User Experience Testing
- **Save Workflow**: Intuitive dialog flow with clear options
- **Apply Speed**: Instant state restoration without flicker
- **Menu Organization**: Logical grouping with clear visual hierarchy  
- **Default Handling**: Seamless automatic application of default views
- **Error Recovery**: Graceful handling of corrupted or missing views

### Cross-Session Persistence Testing
- **Browser Restart**: Views survive complete browser restart
- **Tab Management**: Views work correctly across multiple tabs
- **Session Storage**: Temporary session state vs persistent saved views
- **Backend Sync**: Cross-device synchronization if backend available

### Performance Considerations
- **State Capture**: < 200ms to capture current view state
- **Save Operation**: < 500ms for complete save including backend sync
- **Apply Operation**: < 1000ms for complete state restoration
- **Menu Loading**: < 300ms to populate saved views menu
- **Storage Optimization**: Efficient JSON serialization of view state

### Accessibility Requirements
- **Dialog Navigation**: Full keyboard support in save/manage dialogs
- **Screen Reader**: Proper announcements for view state changes
- **Focus Management**: Correct focus handling during apply operations
- **ARIA Labels**: Descriptive labels for all saved view menu items
- **State Communication**: Clear feedback when views are applied

---

## Test Case ID: TC-UI-AGT-042
**Test Objective**: Verify bulk operations functionality for multiple project management  
**Business Process**: Project Management Operations  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-042
- **Test Priority**: High (P2)
- **Test Type**: Functional, Bulk Operations
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:1790-2176`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:189-227`
- **Export Dialog**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/ExportDialog.fragment.xml:1-65`
- **Status Dialog**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/BulkStatusDialog.fragment.xml:1-55`
- **Functions Under Test**: `onSelectionChange()`, `onSelectAll()`, `onDeselectAll()`, `onExportSelected()`, `onDeleteSelected()`, `onArchiveSelected()`, `onBulkStatusChange()`, `_performBulkDelete()`, `_performBulkArchive()`

### Test Preconditions
1. **User Authentication**: Valid developer account with project management rights
2. **Projects Data**: At least 15 projects with various statuses for bulk testing
3. **Table View**: Projects table displayed with MultiSelect mode enabled
4. **Permissions**: User has delete, archive, and export permissions
5. **Backend Services**: Bulk operation APIs available and responsive

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Selection Methods | Individual, Select All, Range Select | User Action | Table Interaction |
| Bulk Operations | Export, Delete, Archive, Status Change | Operation Type | Toolbar Buttons |
| Export Formats | Excel, CSV, JSON | String | Export Options |
| Status Values | Active, Inactive, Maintenance, Archived | String | Status Dialog |
| Project Count | 1-100 projects | Integer | Selection Size |

### Test Procedure Steps
1. **Step 1 - Multi-Selection Testing**
   - Action: Click checkboxes for 3 individual projects
   - Expected: Projects selected, selection count displayed
   - Verification: Toolbar shows "3 projects selected", bulk buttons enabled

2. **Step 2 - Select All Functionality**
   - Action: Click "Select All" button in table toolbar
   - Expected: All visible projects selected instantly
   - Verification: All checkboxes checked, count shows total projects

3. **Step 3 - Deselect All Testing**
   - Action: Click "Deselect All" button
   - Expected: All selections cleared immediately
   - Verification: No checkboxes checked, bulk buttons disabled

4. **Step 4 - Selection Change Feedback**
   - Action: Select different numbers of projects (1, 5, 10)
   - Expected: Dynamic tooltip updates reflect selection count
   - Verification: Button tooltips show "Export X selected projects"

5. **Step 5 - Bulk Export Dialog**
   - Action: Select 5 projects, click Export button
   - Expected: Export dialog opens with project list and options
   - Verification: Dialog shows selected projects, format options, file name field

6. **Step 6 - Export Format Selection**
   - Action: Test Excel, CSV, and JSON format options
   - Expected: Format selection updates export options
   - Verification: File extension updates in filename field

7. **Step 7 - Export Execution**
   - Action: Configure export options and click Export
   - Expected: API call initiated, download triggered
   - Verification: POST to /api/projects/export, file download starts

8. **Step 8 - Bulk Delete Confirmation**
   - Action: Select 3 projects, click Delete button
   - Expected: Confirmation dialog with specific count and warning
   - Verification: "Delete 3 projects" dialog with irreversible action warning

9. **Step 9 - Bulk Delete Execution**
   - Action: Confirm bulk delete operation
   - Expected: Progress dialog, API call, success feedback
   - Verification: Projects removed from list, selection cleared

10. **Step 10 - Bulk Archive Operation**
    - Action: Select 5 projects, click Archive button
    - Expected: Archive confirmation with reversible action note
    - Verification: Projects archived, status changed to archived

11. **Step 11 - Bulk Status Change**
    - Action: Select projects, click Change Status button
    - Expected: Status change dialog with dropdown options
    - Verification: Status options available, selected projects listed

12. **Step 12 - Status Change Execution**
    - Action: Select "Maintenance" status and confirm
    - Expected: All selected projects status updated
    - Verification: Project list refreshed, statuses changed

### Expected Results
- **Selection Management Criteria**:
  - Individual selection works via checkboxes
  - Select All/Deselect All functions correctly
  - Selection count accurately tracked and displayed
  - Bulk action buttons enabled/disabled based on selection

- **Bulk Export Criteria**:
  - Export dialog shows all selected projects
  - Multiple format options available and functional
  - File naming options and validation working
  - Actual file download triggered successfully

- **Bulk Operations Criteria**:
  - Delete operations require confirmation
  - Archive operations reversible with clear messaging
  - Status change operations update all selected items
  - Progress feedback during long-running operations

### Error Scenarios
1. **No Selection**: Show message "Please select projects to [operation]"
2. **Partial Failures**: Show "Completed X of Y operations" with details
3. **Network Errors**: Graceful error handling with retry options
4. **Permission Errors**: Clear messaging about insufficient permissions

### Validation Points
- Table multiselect mode: Table configured with mode="MultiSelect"
- Bulk action buttons: All buttons present with proper IDs and event handlers
- Progress feedback: BusyDialog shown during long operations
- API integration: Correct endpoints called with proper payload format
- List refresh: Project list refreshed after bulk operations

### Test Data Requirements
- Projects with different statuses (active, inactive, maintenance, archived)
- Various project sizes for export testing
- Mixed permission scenarios for error testing
- Large datasets (100+ projects) for performance testing

### Bulk Operation Types Tested
| Operation | API Endpoint | Confirmation Required | Reversible |
|-----------|-------------|---------------------|------------|
| Export | `/api/projects/export` | No | N/A |
| Delete | `/api/projects/bulk-delete` | Yes (Strong warning) | No |
| Archive | `/api/projects/bulk-archive` | Yes (Soft warning) | Yes |
| Status Change | `/api/projects/bulk-status` | Yes (Informational) | Yes |

### Performance Testing
- **Selection Performance**: Large selections (1000+ items) complete within 2 seconds
- **Export Performance**: Export operations complete within 30 seconds
- **Bulk Operations**: Database operations complete within 60 seconds
- **UI Responsiveness**: Interface remains responsive during operations
- **Memory Usage**: No memory leaks during repeated bulk operations

### User Experience Considerations
- **Progressive Disclosure**: Complex options revealed in dialogs
- **Confirmation Patterns**: Appropriate confirmation levels for destructive actions
- **Progress Indication**: Clear progress feedback for long operations
- **Error Recovery**: Graceful handling of partial failures
- **Undo Options**: Archive operations clearly marked as reversible

### API Payload Examples
```json
// Bulk Delete
{
  "projectIds": ["proj_123", "proj_456", "proj_789"]
}

// Bulk Export
{
  "projects": [...],
  "options": {
    "format": "excel",
    "includeDetails": true,
    "includeAgents": false,
    "fileName": "projects_2024-01-15"
  }
}

// Bulk Status Change
{
  "projectIds": ["proj_123", "proj_456"],
  "status": "maintenance"
}
```

### Accessibility Requirements
- **Selection Feedback**: Screen reader announces selection changes
- **Bulk Operations**: Keyboard access to all bulk action buttons
- **Dialog Navigation**: Full keyboard support in confirmation dialogs
- **Progress Updates**: Screen reader announces operation progress
- **Error Messages**: Accessible error messaging for all failure scenarios

### Security Considerations
- **Authorization**: Verify user permissions before each bulk operation
- **Data Validation**: Validate all project IDs and operation parameters
- **Audit Logging**: Log all bulk operations for security audit trail
- **Rate Limiting**: Prevent abuse of bulk operation endpoints
- **Data Integrity**: Maintain referential integrity during bulk operations

---

## Test Case ID: TC-UI-AGT-043
**Test Objective**: Verify comprehensive export functionality for project data  
**Business Process**: Project Data Export and Analysis  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-043
- **Test Priority**: High (P2)
- **Test Type**: Functional, Data Export
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/Projects.controller.js:2182-2297`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/Projects.view.xml:28-45`
- **Export Dialog**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/ExportDialog.fragment.xml:1-65`
- **Functions Under Test**: `onExportAll()`, `onExportWithFilters()`, `onExportSelected()`, `onExportTemplateDownload()`, `onFormatChange()`, `_performExport()`, `_triggerDownload()`

### Test Preconditions
1. **User Authentication**: Valid developer account with export permissions
2. **Projects Data**: At least 20 projects with varied data for comprehensive testing
3. **File System Access**: Browser can download files to local file system
4. **Network Connection**: Stable connection for export API calls
5. **Multiple Formats**: Backend supports Excel, CSV, and JSON export formats

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Export Types | Selected, All, Filtered, Template | String | Export Options |
| Export Formats | Excel (.xlsx), CSV (.csv), JSON (.json) | String | Format Selection |
| Content Options | Details, Agents, Files | Boolean | Inclusion Checkboxes |
| File Names | Custom names with date stamps | String | User Input |
| Project Counts | 1-1000 projects | Integer | Export Scope |

### Test Procedure Steps
1. **Step 1 - Export Menu Access**
   - Action: Click Export MenuButton in main toolbar
   - Expected: Dropdown menu opens with 4 export options
   - Verification: "Export All", "Export Filtered", "Export Template" options visible

2. **Step 2 - Export All Projects**
   - Action: Select "Export All" from menu
   - Expected: Export dialog opens with all projects pre-loaded
   - Verification: Dialog shows total project count, default filename "all_projects_YYYY-MM-DD"

3. **Step 3 - Format Selection Testing**
   - Action: Test each format option (Excel, CSV, JSON)
   - Expected: File extension updates automatically in filename
   - Verification: .xlsx → .csv → .json extension changes

4. **Step 4 - Content Inclusion Options**
   - Action: Toggle checkboxes for Details, Agents, Files
   - Expected: Options affect export payload structure
   - Verification: Checkbox states properly captured in model

5. **Step 5 - Custom Filename Validation**
   - Action: Enter custom filename with special characters
   - Expected: Input validation prevents invalid characters
   - Verification: Filename field shows validation state

6. **Step 6 - Export All Execution**
   - Action: Configure options and click Export
   - Expected: Progress indicator, API call, file download
   - Verification: POST to /api/projects/export with all projects

7. **Step 7 - Export Selected Projects**
   - Action: Select 5 projects, use bulk export from table toolbar
   - Expected: Export dialog with selected projects only
   - Verification: Dialog shows "5 projects selected", filename "selected_projects_DATE"

8. **Step 8 - Export Filtered Projects**
   - Action: Apply filters (e.g., status=active), then export filtered
   - Expected: Only filtered projects included in export
   - Verification: Export scope matches current filter results

9. **Step 9 - Export Template Download**
   - Action: Select "Export Template" from menu
   - Expected: Template file downloads immediately
   - Verification: project_import_template.xlsx file downloaded

10. **Step 10 - Large Dataset Export**
    - Action: Export all projects when dataset > 100 items
    - Expected: Progress dialog, successful completion
    - Verification: Large file generates without timeout

11. **Step 11 - Format Comparison Testing**
    - Action: Export same data in all 3 formats
    - Expected: Data consistency across formats
    - Verification: Excel, CSV, JSON contain equivalent data structures

12. **Step 12 - Error Handling Testing**
    - Action: Simulate network failure during export
    - Expected: Graceful error handling with retry option
    - Verification: Error message displayed, export can be retried

### Expected Results
- **Export Menu Criteria**:
  - MenuButton provides clear export options
  - Export scope clearly differentiated (All vs Selected vs Filtered)
  - Template download available for import workflows
  - Menu items have appropriate icons and descriptions

- **Dialog Functionality Criteria**:
  - Export dialog shows accurate project counts
  - Format selection updates filename extensions automatically
  - Content inclusion options affect export payload
  - Filename validation prevents invalid characters

- **File Generation Criteria**:
  - Excel format includes proper headers, formatting, multiple sheets
  - CSV format uses proper delimiters and encoding
  - JSON format provides structured, parseable data
  - All formats preserve data integrity and completeness

### Error Scenarios
1. **No Data Available**: Show "No projects available to export" message
2. **Network Timeout**: Show retry option with progress indication
3. **Invalid Filename**: Highlight field with validation message
4. **Backend Error**: Display specific error message with support contact

### Validation Points
- Export menu present: MenuButton in main toolbar with dropdown options
- Dialog integration: ExportDialog fragment loads with proper model binding
- Format handling: File extensions update based on format selection
- Download mechanism: Browser download triggered with correct MIME types
- API integration: Correct endpoints called with appropriate payloads

### Test Data Requirements
- Projects with full metadata (name, description, status, dates, agents)
- Projects with minimal data to test edge cases
- Large dataset (500+ projects) for performance testing
- Projects with special characters in names/descriptions

### Export Format Specifications
| Format | MIME Type | Features | Use Case |
|--------|-----------|----------|----------|
| Excel | application/vnd.openxmlformats-officedocument.spreadsheetml.sheet | Multiple sheets, formatting, formulas | Business analysis, reporting |
| CSV | text/csv | Simple delimited, universal compatibility | Data import, simple analysis |
| JSON | application/json | Structured data, nested objects | API integration, data processing |

### File Content Validation
- **Basic Data**: Project ID, name, description, status, dates
- **Agent Details**: Agent count, agent types, configurations (if selected)
- **File Structure**: Project file tree, sizes, types (if selected)
- **Metadata**: Export timestamp, user info, filter criteria
- **Data Integrity**: No truncation, proper encoding, complete records

### Performance Testing
- **Small Export (1-10 projects)**: Complete within 5 seconds
- **Medium Export (10-100 projects)**: Complete within 30 seconds
- **Large Export (100-1000 projects)**: Complete within 120 seconds
- **Memory Usage**: No browser memory issues during large exports
- **File Size Limits**: Handle exports up to 100MB file size

### Browser Compatibility
- **Chrome/Edge**: Full download functionality with proper MIME types
- **Firefox**: Download handling with appropriate file extensions
- **Safari**: WebKit download behavior with security considerations
- **Mobile Browsers**: Touch-optimized export dialog, download handling

### Security Considerations
- **Data Access Control**: Export respects user project access permissions
- **Audit Logging**: All export operations logged for compliance
- **Data Sanitization**: Sensitive data excluded or masked in exports
- **File Security**: Generated files don't expose system information
- **Rate Limiting**: Prevent abuse of export functionality

### API Payload Examples
```json
// Export All Projects
{
  "exportType": "all",
  "format": "excel",
  "options": {
    "includeDetails": true,
    "includeAgents": false,
    "includeFiles": false,
    "fileName": "all_projects_2024-01-15.xlsx"
  }
}

// Export Selected Projects
{
  "exportType": "selected", 
  "projects": [...],
  "format": "csv",
  "options": {
    "includeDetails": true,
    "includeAgents": true,
    "fileName": "selected_projects_2024-01-15.csv"
  }
}
```

### Integration Testing
- **Backend API**: Verify correct data transformation for each format
- **File Generation**: Ensure proper file structure and content
- **Download Service**: Test file serving and cleanup mechanisms
- **Template System**: Validate import template matches export format

---

## Test Case ID: TC-UI-AGT-044
**Test Objective**: Verify file tree display functionality in ProjectDetail view  
**Business Process**: Project File Management  
**SAP Module**: A2A Agents Developer Portal - File Explorer  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-044
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, File System Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:166-255`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:22-566`
- **Functions Under Test**: `_loadProjectFiles()`, `_buildFileTree()`, `onToggleFolder()`, `onFilePress()`

### Test Preconditions
1. **Project Context**: Valid project loaded in ProjectDetail view with project ID
2. **File System Access**: Project has associated file system structure in backend
3. **API Endpoints**: File management endpoints `/api/projects/{id}/files` are accessible
4. **User Permissions**: Developer has read access to project files
5. **UI Components**: Files tab in IconTabBar is accessible and functional

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Project ID | test-project-123 | String | Project Context |
| File Structure | Multi-level directory tree | Object | Backend API |
| File Types | js, json, xml, html, css, txt, md | Array | File System |
| Folder Hierarchy | /src/components/ui/ | String | Project Structure |
| File Count | 50+ files across 5+ folders | Number | Test Data Set |

### Test Procedure Steps
1. **Step 1 - Navigate to Files Tab**
   - Action: Open ProjectDetail view and click on "Files" tab
   - Expected: Files tab activates and file explorer panel loads
   - Verification: Tree component is rendered with loading indicator

2. **Step 2 - Initial Tree Loading**
   - Action: Wait for API call to `/api/projects/{id}/files` to complete
   - Expected: File tree structure displays within 3 seconds
   - Verification: Root level files and folders are visible in tree

3. **Step 3 - File Tree Structure Validation**
   - Action: Inspect tree structure and file hierarchy
   - Expected: Files and folders display appropriate icons and names
   - Verification: Folders show folder icon, files show type-specific icons

4. **Step 4 - File Type Icon Display**
   - Action: Check various file types (.js, .json, .xml, .html, .css, .txt, .md, images)
   - Expected: Each file type displays correct icon (syntax, document, web, palette, text, notes, image-viewer)
   - Verification: Icon mapping matches file extension correctly

5. **Step 5 - Folder Navigation Items**
   - Action: Locate folders in file tree
   - Expected: Folders display as Navigation type tree items with expand/collapse indicators
   - Verification: Folder items have different styling and behavior than files

6. **Step 6 - File Active Items**
   - Action: Locate files in file tree
   - Expected: Files display as Active type tree items that respond to clicks
   - Verification: File items are clickable and show appropriate hover states

7. **Step 7 - Tree Item Selection**
   - Action: Click on individual files and folders to select them
   - Expected: Selection changes trigger `onFileSelectionChange` event
   - Verification: Selected files appear in files model selectedFiles array

8. **Step 8 - File Actions Panel Visibility**
   - Action: Select one or more files from tree
   - Expected: File actions panel becomes visible with action buttons
   - Verification: Panel shows "X files selected" and enables appropriate actions

### Expected Results
- **Tree Display Criteria**:
  - File tree loads within 3 seconds of tab selection
  - All files and folders from backend API are displayed
  - Tree structure accurately reflects file system hierarchy
  - Icons correctly match file types and folder status

- **Navigation and Selection Criteria**:
  - Tree supports multi-select mode with checkboxes
  - Individual tree items respond to press events
  - Selection state updates correctly in files model
  - Tree maintains selection across expand/collapse operations

- **Performance Criteria**:
  - Initial tree loading completes within 3 seconds
  - Tree rendering handles 100+ files without performance issues
  - File icon rendering is instantaneous
  - Tree selection updates occur within 100ms

### Error Handling Scenarios
1. **API Failure Scenario**:
   - Action: Simulate API endpoint `/api/projects/{id}/files` returning error
   - Expected: Error message displays: "Failed to load project files: {error}"
   - Verification: User receives clear feedback about loading failure

2. **Empty Project Scenario**:
   - Action: Load project with no files
   - Expected: Tree shows empty state or root folder only
   - Verification: No JavaScript errors occur with empty data

3. **Large File Set Scenario**:
   - Action: Load project with 1000+ files across deep folder structure
   - Expected: Tree handles large dataset with lazy loading
   - Verification: Performance remains acceptable, UI stays responsive

### Post-Conditions
- File tree accurately represents current project file structure
- Selected files are tracked in files model for subsequent operations
- File explorer is ready for file operations (create, delete, rename, etc.)
- No memory leaks from tree rendering or event handlers

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Screen Resolution**: 1920x1080 minimum for full tree visibility
- **Network**: Stable connection for API calls
- **Backend**: File management API endpoints operational

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Performance Efficiency, Usability)
- **SAP Standard**: SAP Fiori Design Guidelines - Tree Control Patterns
- **Accessibility**: WCAG 2.1 Level AA (tree navigation, keyboard support)

---

## Test Case ID: TC-UI-AGT-045
**Test Objective**: Verify folder expand/collapse functionality in file tree  
**Business Process**: Project File Navigation  
**SAP Module**: A2A Agents Developer Portal - File Explorer Navigation  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-045
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interaction
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:122-136`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:195-220`
- **Functions Under Test**: `onToggleFolder()`, `_loadFolderContents()`, `_updateFolderContents()`

### Test Preconditions
1. **File Tree Loaded**: Project file tree is displayed with at least one folder
2. **Folder Structure**: Project contains hierarchical folder structure with nested directories
3. **API Accessibility**: Folder content endpoints are responsive and operational
4. **UI State**: File explorer is visible and interactive in Files tab
5. **Data State**: Tree model contains folders with expandable/collapsible state

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Project Structure | Multi-level folders | Object | File System |
| Root Folders | /src, /docs, /tests, /config | Array | Project Layout |
| Nested Depth | 4+ levels deep | Number | Directory Structure |
| Folder Children | 5-20 items per folder | Number | Content Density |
| Mixed Content | Folders and files combined | Mixed | Realistic Structure |

### Test Procedure Steps
1. **Step 1 - Identify Expandable Folders**
   - Action: Locate folders in file tree with expand/collapse indicators
   - Expected: Folders display with triangle/chevron indicators for expansion
   - Verification: Visual indicators clearly distinguish folders from files

2. **Step 2 - Initial Folder State**
   - Action: Check initial expansion state of folders after tree load
   - Expected: Folders start in collapsed state (expanded: false)
   - Verification: Tree model shows expanded property as false for folders

3. **Step 3 - Expand Folder Action**
   - Action: Click on folder expand indicator or folder name
   - Expected: `onToggleFolder` event triggers with expanded=true parameter
   - Verification: Event handler receives correct tree item and expansion state

4. **Step 4 - Folder Content Loading**
   - Action: Wait for folder expansion to complete
   - Expected: API call to `/api/projects/{id}/files?path={folderPath}` executes
   - Verification: Network request shows correct folder path parameter

5. **Step 5 - Child Content Display**
   - Action: Observe folder contents after expansion
   - Expected: Child files and folders appear under expanded folder within 2 seconds
   - Verification: Tree structure updates with nested items indented properly

6. **Step 6 - Lazy Loading Verification**
   - Action: Expand folder that hasn't been loaded before
   - Expected: Loading occurs only when folder is expanded for first time
   - Verification: API call made only on initial expansion, not subsequent

7. **Step 7 - Collapse Folder Action**
   - Action: Click expanded folder to collapse it
   - Expected: `onToggleFolder` event triggers with expanded=false parameter
   - Verification: Folder children hide but remain in model for caching

8. **Step 8 - Visual State Updates**
   - Action: Toggle folder expansion multiple times
   - Expected: Expand/collapse icons rotate or change to reflect state
   - Verification: Visual feedback provides clear indication of current state

9. **Step 9 - Nested Folder Navigation**
   - Action: Expand multiple levels of nested folders
   - Expected: Each folder level expands independently
   - Verification: Deep folder structures navigate correctly with proper indentation

10. **Step 10 - State Persistence**
    - Action: Expand folders, navigate away, and return to Files tab
    - Expected: Folder expansion states are maintained in files model
    - Verification: Previously expanded folders remain expanded

### Expected Results
- **Interaction Response Criteria**:
  - Folder expansion triggers within 100ms of click
  - Visual indicators update immediately upon interaction
  - Expand/collapse animations are smooth (300ms duration)
  - No UI freezing during folder operations

- **Content Loading Criteria**:
  - Folder contents load within 2 seconds of expansion
  - Lazy loading prevents unnecessary API calls
  - Tree structure accurately reflects folder hierarchy
  - Mixed content (files + folders) displays correctly

- **State Management Criteria**:
  - Folder expanded state tracked accurately in model
  - Multiple folder expansions work independently
  - Collapsed folders hide children while preserving data
  - State persistence works across tab switches

### Error Handling Scenarios
1. **Folder Loading Failure**:
   - Action: Simulate API failure when loading folder contents
   - Expected: Error message displays: "Failed to load folder contents: {error}"
   - Verification: Folder remains in loading state, user gets clear feedback

2. **Empty Folder Scenario**:
   - Action: Expand folder with no children
   - Expected: Folder expands but shows no child items
   - Verification: No JavaScript errors, clean empty state presentation

3. **Deep Nesting Performance**:
   - Action: Expand folders nested 10+ levels deep
   - Expected: Performance remains acceptable, no stack overflow
   - Verification: Deep structures handle gracefully without UI lag

### Post-Conditions
- Folder expansion states accurately reflect user interactions
- Tree structure maintains correct hierarchical relationships
- Loaded folder contents are cached for performance
- File tree remains navigable and responsive

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Performance**: Tree operations complete within response time limits
- **Network**: API endpoints respond within 2 seconds
- **Memory**: No memory leaks from repeated expand/collapse operations

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Performance Efficiency, Functional Suitability)
- **SAP Standard**: SAP Fiori Design Guidelines - Tree Navigation Patterns
- **Accessibility**: WCAG 2.1 Level AA (keyboard navigation, screen reader support)

### Integration Testing
- **Backend API**: Verify folder path parameters and response structure
- **Model Synchronization**: Ensure UI state matches model data
- **Event Handling**: Confirm proper event propagation and handling
- **Performance**: Validate acceptable response times under load

---

## Test Case ID: TC-UI-AGT-046
**Test Objective**: Verify comprehensive file operations (open, rename, copy, move, delete)  
**Business Process**: Project File Management Operations  
**SAP Module**: A2A Agents Developer Portal - File Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-046
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, CRUD Operations
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:334-490`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:224-254`
- **Functions Under Test**: `onOpenSelectedFile()`, `onRenameFile()`, `onCopyFiles()`, `onMoveFiles()`, `onDeleteFiles()`

### Test Preconditions
1. **File Selection**: At least one file is selected in file tree
2. **File Actions Panel**: Selected file actions panel is visible and accessible
3. **User Permissions**: Developer has read/write permissions for project files
4. **API Endpoints**: All file operation endpoints are operational
5. **File System State**: Test files exist in project file system

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Test Files | sample.js, config.json, readme.md | Array | File System |
| Target Folder | /src/components/ | String | Directory Structure |
| Rename Target | component-updated.js | String | User Input |
| File Content | Valid JavaScript/JSON/Markdown | String | File System |
| Operation Count | 1-5 selected files | Number | User Selection |

### Test Procedure Steps
1. **Step 1 - File Selection for Operations**
   - Action: Select single file from file tree
   - Expected: File actions panel becomes visible with enabled action buttons
   - Verification: Panel shows "1 files selected" and all relevant actions enabled

2. **Step 2 - Open File Operation**
   - Action: Click "Open" button in file actions panel
   - Expected: `onOpenSelectedFile()` method executes and file content dialog opens
   - Verification: File content loads via `/api/projects/{id}/files/{path}` endpoint

3. **Step 3 - File Content Display**
   - Action: Wait for file content dialog to load
   - Expected: Dialog shows file name, path, content, and editability status
   - Verification: Content matches actual file and editable flag set correctly

4. **Step 4 - Rename File Operation**
   - Action: Select file and click "Rename" button
   - Expected: Rename dialog appears with current filename pre-filled
   - Verification: MessageBox.prompt displays with initial text set to current name

5. **Step 5 - Execute Rename**
   - Action: Enter new filename "updated-component.js" and confirm
   - Expected: PUT request to `/api/projects/{id}/files/{path}/rename` with newPath
   - Verification: File renamed successfully, tree refreshes, success toast shown

6. **Step 6 - Copy Files Operation**
   - Action: Select multiple files and click "Copy" button
   - Expected: Files copied to clipboard with operation type "copy"
   - Verification: Toast shows "X file(s) copied to clipboard", clipboard model updated

7. **Step 7 - Move Files Operation**
   - Action: Select files and click "Move" button  
   - Expected: Files added to clipboard with operation type "cut"
   - Verification: Toast shows "X file(s) moved to clipboard", clipboard operation set

8. **Step 8 - Delete Files Confirmation**
   - Action: Select files and click "Delete" button
   - Expected: Confirmation dialog appears with file count and warning
   - Verification: MessageBox.confirm shows "Are you sure you want to delete X selected file(s)?"

9. **Step 9 - Execute Delete Operation**
   - Action: Confirm deletion in dialog
   - Expected: DELETE request to `/api/projects/{id}/files/delete` with filePaths array
   - Verification: Files deleted successfully, tree refreshes, selection cleared

10. **Step 10 - Multi-File Operations**
    - Action: Select 3+ files and test each operation
    - Expected: All operations handle multiple files correctly
    - Verification: Bulk operations process all selected files as expected

### Expected Results
- **File Opening Criteria**:
  - Single file opens in content dialog within 2 seconds
  - File content loads accurately from backend
  - Editable files show edit capability status correctly
  - Dialog displays proper file metadata (name, path, size)

- **File Modification Criteria**:
  - Rename operation updates filename and refreshes tree
  - Copy operation places files in clipboard without moving originals
  - Move operation marks files for relocation
  - Delete operation removes files permanently with confirmation

- **User Interface Criteria**:
  - Action buttons enable/disable based on selection count
  - "Open" and "Rename" enabled only for single file selection
  - "Copy", "Move", "Delete" enabled for any number of selected files
  - Success/error messages provide clear feedback for all operations

### Error Handling Scenarios
1. **File Open Failure**:
   - Action: Attempt to open file that no longer exists
   - Expected: Error toast: "Failed to load file content"
   - Verification: User gets clear feedback, no dialog opens

2. **Rename Validation**:
   - Action: Attempt to rename file with empty or invalid name
   - Expected: Validation prevents operation, user prompted to enter valid name
   - Verification: No API call made with invalid data

3. **Delete Permission Error**:
   - Action: Try to delete read-only or system files
   - Expected: Error toast: "Failed to delete files: {error}"
   - Verification: Partial success handled gracefully, tree state consistent

4. **Network Failure During Operations**:
   - Action: Simulate network failure during file operations
   - Expected: Operation fails gracefully with error messages
   - Verification: File tree state remains consistent, no data corruption

### Post-Conditions
- File operations complete successfully with appropriate user feedback
- File tree accurately reflects current file system state
- Clipboard operations prepare files for subsequent paste operations
- Deleted files are permanently removed and selection is cleared
- File system integrity maintained throughout all operations

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Network**: Stable connection for API operations
- **Backend**: File management endpoints respond within 5 seconds
- **File System**: Test files with appropriate permissions available

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Functional Suitability, Reliability)
- **SAP Standard**: SAP Fiori Design Guidelines - File Management Patterns
- **Data Protection**: Secure file operations without data leakage
- **Accessibility**: WCAG 2.1 Level AA (screen reader support for confirmations)

### Integration Testing
- **Backend API**: Verify all CRUD endpoints handle operations correctly
- **File System**: Ensure operations properly modify file system state
- **Error Handling**: Test network failures and permission errors
- **Performance**: Validate acceptable response times for large file operations

---

## Test Case ID: TC-UI-AGT-047
**Test Objective**: Verify file and folder creation functionality with dialog interfaces  
**Business Process**: Project Asset Creation  
**SAP Module**: A2A Agents Developer Portal - File Creation Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-047
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Creation Operations
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:168-285`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:177-191`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/CreateFileDialog.fragment.xml:1-62`
- **Functions Under Test**: `onCreateFile()`, `onCreateFolder()`, `onConfirmCreateFile()`, `onConfirmCreateFolder()`

### Test Preconditions
1. **Files Tab Access**: Files tab is open and file explorer is displayed
2. **Creation Permissions**: Developer has write permissions to project file system
3. **Dialog Availability**: CreateFileDialog and CreateFolderDialog fragments are accessible
4. **API Endpoints**: File creation endpoints are operational
5. **Parent Directory**: Valid parent directory exists for new file/folder creation

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| New File Name | TestComponent.js | String | User Input |
| File Type | js (JavaScript) | String | Selection |
| Parent Path | /src/components/ | String | Directory Context |
| Initial Content | // New component code | String | Template/User Input |
| New Folder Name | utils | String | User Input |
| Folder Path | /src/utils/ | String | Directory Structure |

### Test Procedure Steps
1. **Step 1 - Access File Creation**
   - Action: Click "Create File" button in file explorer toolbar
   - Expected: CreateFileDialog fragment opens with empty form
   - Verification: Dialog displays with title "Create New File" and form fields

2. **Step 2 - File Creation Form Validation**
   - Action: Inspect form fields and validation states
   - Expected: File name field shows error state when empty, required indicator visible
   - Verification: valueState="Error" when fileName is empty, "None" when filled

3. **Step 3 - File Type Selection**
   - Action: Open file type dropdown and inspect options
   - Expected: Dropdown shows predefined file types (txt, js, json, xml, html, css, md, yaml)
   - Verification: Each option displays proper extension and description

4. **Step 4 - Fill File Creation Form**
   - Action: Enter file name "TestComponent.js", select "JavaScript" type, set parent path
   - Expected: Form validation passes, file name field shows "None" state
   - Verification: Create button becomes enabled when required fields filled

5. **Step 5 - Add Initial Content**
   - Action: Enter initial JavaScript code in TextArea
   - Expected: Content area accepts multi-line input up to 10000 characters
   - Verification: TextArea displays content correctly with proper formatting

6. **Step 6 - Execute File Creation**
   - Action: Click "Create" button to confirm file creation
   - Expected: `onConfirmCreateFile()` executes, API call to POST `/api/projects/{id}/files`
   - Verification: Request contains correct path, type, extension, and content data

7. **Step 7 - File Creation Success**
   - Action: Wait for file creation to complete
   - Expected: Success toast "File created successfully", dialog closes, tree refreshes
   - Verification: New file appears in file tree at specified location

8. **Step 8 - Access Folder Creation**
   - Action: Click "Create Folder" button in file explorer toolbar
   - Expected: CreateFolderDialog opens with folder name and parent path fields
   - Verification: Dialog displays with proper form validation and required indicators

9. **Step 9 - Fill Folder Creation Form**
   - Action: Enter folder name "utilities" and parent path "/src/"
   - Expected: Form validation passes, folder name field shows valid state
   - Verification: Create button enabled when required fields completed

10. **Step 10 - Execute Folder Creation**
    - Action: Click create button to confirm folder creation
    - Expected: `onConfirmCreateFolder()` executes, POST request with type="folder"
    - Verification: Folder created successfully, tree refreshes with new folder visible

### Expected Results
- **Dialog Interface Criteria**:
  - CreateFileDialog opens within 200ms of button click
  - All form fields display proper labels, placeholders, and validation states
  - File type selection provides comprehensive options for common file types
  - Dialog supports proper keyboard navigation and accessibility

- **Form Validation Criteria**:
  - Required fields show error states when empty
  - File name validation prevents invalid characters
  - Maximum length constraints enforced (255 for filename, 10000 for content)
  - Parent path validation ensures valid directory structure

- **Creation Operation Criteria**:
  - File creation completes within 3 seconds
  - Folder creation completes within 2 seconds
  - API requests contain properly formatted data
  - File tree refreshes automatically after successful creation

### Error Handling Scenarios
1. **Empty File Name Validation**:
   - Action: Attempt to create file with empty name
   - Expected: Validation prevents creation, error message displayed
   - Verification: Toast shows "Please enter a file name", no API call made

2. **Duplicate File Name**:
   - Action: Create file with name that already exists
   - Expected: Backend returns error, user receives clear feedback
   - Verification: Error toast displays server error message appropriately

3. **Invalid Parent Path**:
   - Action: Specify non-existent parent directory
   - Expected: Creation fails with path validation error
   - Verification: Error handling graceful, user informed of path issue

4. **Network Failure During Creation**:
   - Action: Simulate network failure during file/folder creation
   - Expected: Operation fails with network error message
   - Verification: Dialog remains open, user can retry operation

### Post-Conditions
- New files and folders appear in file tree at correct locations
- File content matches what was entered in creation dialog
- File system maintains consistency and proper hierarchy
- Creation dialogs reset to clean state for subsequent use
- No memory leaks from dialog instantiation and cleanup

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Dialog Rendering**: Proper dialog sizing and positioning across screen sizes
- **Network**: API endpoints respond within acceptable timeframes
- **File System**: Backend supports file creation with various types and content

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Usability, Functional Suitability)
- **SAP Standard**: SAP Fiori Design Guidelines - Dialog Patterns, Form Validation
- **Accessibility**: WCAG 2.1 Level AA (dialog focus management, form labeling)
- **UX Standards**: Intuitive file creation workflow with clear feedback

### Integration Testing
- **Backend API**: Verify file creation endpoints handle all parameters correctly
- **File System**: Ensure created files persist and are accessible
- **UI Synchronization**: Confirm file tree updates reflect actual file system state
- **Template System**: Validate different file types create with appropriate structure

---

## Test Case ID: TC-UI-AGT-048
**Test Objective**: Verify file/folder deletion with comprehensive confirmation workflow  
**Business Process**: Secure File Deletion Operations  
**SAP Module**: A2A Agents Developer Portal - File Deletion with Safeguards  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-048
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Security, User Protection
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:446-490`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:247-252`
- **Functions Under Test**: `onDeleteFiles()`, `_deleteSelectedFiles()`, MessageBox.confirm handling

### Test Preconditions
1. **File Selection**: One or more files/folders selected in file tree
2. **Delete Permission**: User has delete permissions for selected items
3. **UI State**: File actions panel visible with delete button enabled
4. **Test Data**: Mix of files and folders in various states (empty/non-empty)
5. **Backup State**: Test performed in non-production environment

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Single File | test-component.js | String | File Selection |
| Multiple Files | [file1.js, file2.json, file3.md] | Array | Multi-Selection |
| Folder with Contents | /src/old-components/ (5 files) | Object | Folder Selection |
| Mixed Selection | 2 files + 1 folder | Mixed | Multi-Selection |
| System Files | .gitignore, package.json | Array | Protected Files |

### Test Procedure Steps
1. **Step 1 - Single File Deletion Initiation**
   - Action: Select single file and click delete button
   - Expected: MessageBox.confirm dialog appears with file-specific message
   - Verification: Dialog shows "Are you sure you want to delete 1 selected file(s)?"

2. **Step 2 - Confirmation Dialog Validation**
   - Action: Inspect confirmation dialog content and options
   - Expected: Dialog displays warning "This action cannot be undone."
   - Verification: Dialog shows DELETE and CANCEL actions, DELETE emphasized

3. **Step 3 - Cancel Delete Operation**
   - Action: Click CANCEL button in confirmation dialog
   - Expected: Dialog closes, no deletion occurs, selection maintained
   - Verification: File remains in tree, no API call made

4. **Step 4 - Confirm Single File Delete**
   - Action: Click delete again and choose DELETE action
   - Expected: `_deleteSelectedFiles()` executes with single file path
   - Verification: DELETE request to `/api/projects/{id}/files/delete` with array

5. **Step 5 - Multiple File Selection Delete**
   - Action: Select 3 files and initiate delete
   - Expected: Confirmation shows "delete 3 selected file(s)?"
   - Verification: Dialog accurately reflects selection count

6. **Step 6 - Bulk Delete Execution**
   - Action: Confirm deletion of multiple files
   - Expected: All selected files deleted in single API call
   - Verification: Request body contains array of all file paths

7. **Step 7 - Folder Deletion Warning**
   - Action: Select folder containing files and initiate delete
   - Expected: Confirmation emphasizes folder contains files
   - Verification: Extra warning for non-empty folder deletion

8. **Step 8 - Mixed Selection Delete**
   - Action: Select combination of files and folders, then delete
   - Expected: Confirmation shows total count of items
   - Verification: Both files and folders included in deletion request

9. **Step 9 - Delete Success Feedback**
   - Action: Confirm deletion and wait for completion
   - Expected: Success toast "Successfully deleted X file(s)"
   - Verification: Tree refreshes, deleted items removed, selection cleared

10. **Step 10 - Partial Delete Handling**
    - Action: Simulate partial deletion failure (some files locked)
    - Expected: Detailed feedback on which files deleted/failed
    - Verification: Tree shows accurate state after partial deletion

### Expected Results
- **Confirmation Dialog Criteria**:
  - Dialog appears within 100ms of delete button click
  - Message accurately reflects number and type of selected items
  - Warning about irreversibility is prominently displayed
  - Dialog actions follow SAP Fiori button emphasis patterns

- **Deletion Operation Criteria**:
  - Delete API call includes all selected file paths
  - Operation completes within 5 seconds for reasonable selection
  - Success feedback indicates exact number of deleted items
  - File tree automatically refreshes after deletion

- **Safety and Recovery Criteria**:
  - Cancel action truly cancels without side effects
  - No accidental deletions possible without confirmation
  - Selection cleared after successful deletion
  - Clear error messages for failed deletions

### Error Handling Scenarios
1. **Permission Denied Error**:
   - Action: Attempt to delete read-only or system files
   - Expected: Error message specifies which files couldn't be deleted
   - Verification: Other deletable files in selection still processed

2. **File In Use Error**:
   - Action: Try to delete file currently open in editor
   - Expected: Error indicates file is locked/in use
   - Verification: User advised to close file before deletion

3. **Network Failure During Delete**:
   - Action: Simulate network disconnection during deletion
   - Expected: Error toast about network failure, tree state preserved
   - Verification: No partial deletions, operation atomic

4. **Large Selection Performance**:
   - Action: Select 100+ files and delete
   - Expected: Progress indication during deletion
   - Verification: UI remains responsive, operation completes

### Post-Conditions
- Successfully deleted files permanently removed from file system
- File tree accurately reflects post-deletion state
- No orphaned references or broken links remain
- Selection state cleared, ready for new operations
- Audit trail of deletion operation logged (if applicable)

### Test Environment Requirements
- **Browser Support**: All modern browsers with MessageBox support
- **Confirmation Dialog**: Proper rendering and focus management
- **Network**: Stable connection for deletion operations
- **Backend**: Transactional delete support for atomicity

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Safety, Error Prevention)
- **SAP Standard**: SAP Fiori Design Guidelines - Destructive Actions
- **Security**: Proper authorization checks before deletion
- **Accessibility**: WCAG 2.1 Level AA (confirmation dialog accessibility)

### Integration Testing
- **Confirmation Flow**: Verify proper dialog lifecycle and event handling
- **Batch Operations**: Test deletion of large file sets
- **Rollback Capability**: Ensure failed deletions don't corrupt state
- **Audit Logging**: Verify deletion events are properly logged

---

## Test Case ID: TC-UI-AGT-049
**Test Objective**: Verify project save functionality with data persistence  
**Business Process**: Project Configuration Management  
**SAP Module**: A2A Agents Developer Portal - Project Save Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-049
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Data Persistence
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:641-682`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:25-29`
- **Functions Under Test**: `onSaveProject()`, EventBus publish mechanism

### Test Preconditions
1. **Project Context**: Valid project loaded in ProjectDetail view
2. **Modification State**: At least one project property has been modified
3. **User Permissions**: Developer has write permissions for project
4. **Backend Availability**: Project save API endpoint is operational
5. **Network Connectivity**: Stable connection to backend services

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Project Name | Updated Test Project | String | User Input |
| Description | Enhanced project description with new features | String | User Input |
| Status | active | String | Dropdown Selection |
| Deployment Config | {environment: "staging", autoDeploy: true} | Object | Settings Tab |
| Last Modified | Auto-generated timestamp | DateTime | System |

### Test Procedure Steps
1. **Step 1 - Access Project Details**
   - Action: Navigate to ProjectDetail view for existing project
   - Expected: Project data loads with current values displayed
   - Verification: All project fields show current saved state

2. **Step 2 - Modify Project Properties**
   - Action: Change project name in Settings tab input field
   - Expected: Input field accepts new value, model updates
   - Verification: Binding reflects change immediately in UI

3. **Step 3 - Update Description**
   - Action: Edit description TextArea with enhanced content
   - Expected: TextArea accepts multi-line text up to character limit
   - Verification: Description field shows modified state

4. **Step 4 - Click Save Button**
   - Action: Click save button in page header actions
   - Expected: View shows busy state, save operation initiates
   - Verification: Busy indicator appears immediately

5. **Step 5 - API Request Validation**
   - Action: Monitor network request to PUT `/api/projects/{id}`
   - Expected: Request contains all modified fields with correct values
   - Verification: Request body includes name, description, status, deployment

6. **Step 6 - Save Success Feedback**
   - Action: Wait for save operation to complete
   - Expected: Success toast "Project saved successfully" appears
   - Verification: Busy state removed, toast displayed for 3 seconds

7. **Step 7 - Model Update Verification**
   - Action: Check project model data after save
   - Expected: Model updated with server response including last_modified
   - Verification: Timestamp reflects current save time

8. **Step 8 - Event Bus Notification**
   - Action: Monitor EventBus for "project:saved" event
   - Expected: Event published with projectId and updated data
   - Verification: Other components receive save notification

9. **Step 9 - Persistence Verification**
   - Action: Navigate away and return to project
   - Expected: Saved changes persist after navigation
   - Verification: All modified values remain after reload

10. **Step 10 - Concurrent Save Handling**
    - Action: Attempt save while previous save in progress
    - Expected: Second save queued or blocked with appropriate feedback
    - Verification: No data corruption from concurrent saves

### Expected Results
- **Save Operation Criteria**:
  - Save completes within 3 seconds for normal payload
  - All modified fields included in save request
  - Server response updates local model completely
  - Last modified timestamp updates to current time

- **User Interface Criteria**:
  - Save button shows loading state during operation
  - Busy indicator prevents user interaction during save
  - Success feedback clearly indicates completion
  - Error messages provide actionable information

- **Data Integrity Criteria**:
  - No data loss during save operation
  - Optimistic locking prevents overwrite conflicts
  - Partial saves not possible (all or nothing)
  - Related components notified of changes

### Error Handling Scenarios
1. **Network Failure During Save**:
   - Action: Simulate network disconnection during save
   - Expected: Error toast "Failed to save project: Network error"
   - Verification: Local changes preserved, retry possible

2. **Validation Error Response**:
   - Action: Submit invalid data (empty name)
   - Expected: Server returns 400 with validation errors
   - Verification: Specific field errors displayed to user

3. **Concurrent Edit Conflict**:
   - Action: Another user saves while editing
   - Expected: Conflict detection with merge options
   - Verification: User can choose to overwrite or merge

4. **Permission Denied**:
   - Action: Save with insufficient permissions
   - Expected: Error message about missing permissions
   - Verification: Clear guidance on required permissions

### Post-Conditions
- Project data successfully persisted to backend
- All project views reflect updated information
- Event notifications sent to dependent components
- Save history/audit trail updated (if applicable)
- UI returns to normal non-busy state

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **API Response Time**: Save endpoint responds within 5 seconds
- **Data Validation**: Backend validates all required fields
- **Concurrency**: System handles multiple users editing projects

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Reliability, Data Integrity)
- **SAP Standard**: SAP Fiori Design Guidelines - Save Patterns
- **Data Protection**: No sensitive data exposed in save operations
- **Audit Requirements**: Save operations logged for compliance

### Integration Testing
- **Backend Validation**: Verify all validation rules enforced
- **Event Propagation**: Ensure all listeners receive save events
- **Cache Synchronization**: Check caches update after save
- **Database Persistence**: Confirm data written to database

---

## Test Case ID: TC-UI-AGT-050
**Test Objective**: Verify project build functionality with progress tracking  
**Business Process**: Project Compilation and Build Management  
**SAP Module**: A2A Agents Developer Portal - Build Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-050
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Build Process
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:684-802`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:30-33`
- **Functions Under Test**: `onBuildProject()`, `_updateBuildProgress()`, `_cancelBuild()`

### Test Preconditions
1. **Project State**: Valid project with source code ready for build
2. **Build Configuration**: Project has valid build configuration files
3. **Build Server**: Build service endpoints are operational
4. **User Permissions**: Developer has build execution permissions
5. **Resources**: Sufficient server resources for build process

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Build Configuration | production | String | Build Settings |
| Clean Build | true | Boolean | Build Options |
| Optimize | true | Boolean | Build Options |
| Build Timeout | 300000ms (5 min) | Number | System Config |
| Progress Updates | Every 1000ms | Number | Polling Interval |

### Test Procedure Steps
1. **Step 1 - Initiate Build Process**
   - Action: Click "Build" button in project header actions
   - Expected: Build progress dialog opens immediately
   - Verification: Dialog shows "Building Project" title and initial status

2. **Step 2 - Progress Dialog Validation**
   - Action: Inspect build progress dialog components
   - Expected: Dialog contains progress indicator, status text, action buttons
   - Verification: Progress indicator at 0%, status shows "Initializing build process..."

3. **Step 3 - Build API Request**
   - Action: Monitor POST request to `/api/projects/{id}/build`
   - Expected: Request includes configuration, clean, and optimize parameters
   - Verification: Request body contains correct build settings

4. **Step 4 - Build ID Assignment**
   - Action: Check server response for build initiation
   - Expected: Response includes unique buildId for tracking
   - Verification: BuildId stored for progress polling

5. **Step 5 - Progress Updates**
   - Action: Observe progress indicator updates
   - Expected: Progress updates every second via status polling
   - Verification: GET requests to `/api/builds/{buildId}/status` every 1000ms

6. **Step 6 - Status Message Updates**
   - Action: Monitor status text changes during build
   - Expected: Messages reflect current build phase (compiling, bundling, optimizing)
   - Verification: Status text updates with each progress response

7. **Step 7 - Background Execution**
   - Action: Click "Run in Background" button
   - Expected: Dialog closes, build continues, toast message appears
   - Verification: "Build continues in background" toast displayed

8. **Step 8 - Build Completion**
   - Action: Wait for build to complete successfully
   - Expected: Progress reaches 100%, success state shown
   - Verification: Progress indicator turns green, success message displayed

9. **Step 9 - Auto-Close Behavior**
   - Action: Wait 2 seconds after successful completion
   - Expected: Dialog automatically closes
   - Verification: Dialog closes gracefully without user action

10. **Step 10 - Build Cancellation**
    - Action: Start new build and click "Cancel" button
    - Expected: Cancel request sent, build terminated
    - Verification: POST to `/api/builds/{buildId}/cancel`, appropriate feedback

### Expected Results
- **Build Initiation Criteria**:
  - Build starts within 1 second of button click
  - Progress dialog displays all required elements
  - Build configuration properly sent to server
  - Unique build ID received and tracked

- **Progress Tracking Criteria**:
  - Real-time progress updates every second
  - Accurate percentage completion shown
  - Status messages reflect actual build phases
  - No missed polling cycles during active build

- **User Experience Criteria**:
  - Smooth progress animation without jumps
  - Clear status communication throughout process
  - Option to run in background maintains functionality
  - Automatic cleanup after completion

### Error Handling Scenarios
1. **Build Initialization Failure**:
   - Action: Simulate build start failure
   - Expected: Error message "Failed to start build: {error}"
   - Verification: Dialog closes, user can retry

2. **Build Process Failure**:
   - Action: Simulate build failure mid-process
   - Expected: Progress indicator turns red, error state shown
   - Verification: Status text shows "Build failed: {error details}"

3. **Network Interruption**:
   - Action: Disconnect network during progress polling
   - Expected: Error state with "Failed to get build status"
   - Verification: Build may continue on server, user informed

4. **Build Timeout**:
   - Action: Simulate build exceeding timeout threshold
   - Expected: Build cancelled automatically, timeout message shown
   - Verification: Resources cleaned up, user can restart

### Post-Conditions
- Build artifacts generated successfully (if completed)
- Build logs available for review
- Project status updated to reflect build state
- Server resources properly released
- UI returns to normal interactive state

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Build Server**: Adequate CPU and memory for compilation
- **Network**: Stable connection for progress updates
- **Storage**: Sufficient space for build artifacts

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Performance, Reliability)
- **SAP Standard**: SAP Fiori Design Guidelines - Progress Indicators
- **Build Standards**: Industry standard build practices
- **Security**: Secure build environment isolation

### Integration Testing
- **Build Pipeline**: Verify integration with CI/CD systems
- **Artifact Storage**: Ensure build outputs stored correctly
- **Notification System**: Test build completion notifications
- **Resource Management**: Verify proper cleanup after builds

---

## Test Case ID: TC-UI-AGT-051
**Test Objective**: Verify project run functionality with configuration dialog  
**Business Process**: Project Execution and Runtime Management  
**SAP Module**: A2A Agents Developer Portal - Run Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-051
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Runtime Execution
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:804-861`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:34-37`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/RunConfigDialog.fragment.xml`
- **Functions Under Test**: `onRunProject()`, `onConfirmRunProject()`, `onCancelRunProject()`

### Test Preconditions
1. **Project Build Status**: Project successfully built and ready to run
2. **Runtime Environment**: Development/test environment available
3. **Port Availability**: Default port 3000 or configured port available
4. **User Permissions**: Developer has project execution permissions
5. **Dependencies**: All project dependencies installed

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Environment | development | String | Environment Select |
| Port | 3000 | Number | Port Configuration |
| Debug Mode | true | Boolean | Debug Switch |
| Watch Files | true | Boolean | Watch Switch |
| Environment Variables | [{name: "API_KEY", value: "test123"}] | Array | Env Config |

### Test Procedure Steps
1. **Step 1 - Open Run Configuration**
   - Action: Click "Run" button in project header actions
   - Expected: RunConfigDialog fragment opens with default values
   - Verification: Dialog displays all configuration options

2. **Step 2 - Environment Selection**
   - Action: Open environment dropdown and inspect options
   - Expected: Dropdown shows development, staging, production, test
   - Verification: Development selected by default

3. **Step 3 - Port Configuration**
   - Action: Verify port input field with default value
   - Expected: Port field shows 3000 as default with number validation
   - Verification: Field accepts only numeric input

4. **Step 4 - Debug Mode Toggle**
   - Action: Check debug mode switch state
   - Expected: Switch shows ON state with custom text
   - Verification: Toggle functions properly with visual feedback

5. **Step 5 - Watch Files Option**
   - Action: Verify watch files switch and tooltip
   - Expected: Switch enabled by default with informative tooltip
   - Verification: Tooltip explains file watching functionality

6. **Step 6 - Add Environment Variable**
   - Action: Click "Add Variable" button
   - Expected: New row added to environment variables table
   - Verification: Empty inputs for name and value appear

7. **Step 7 - Configure Environment Variable**
   - Action: Enter "API_KEY" as name and "test123" as value
   - Expected: Inputs accept text, binding updates model
   - Verification: Table shows entered values correctly

8. **Step 8 - Delete Environment Variable**
   - Action: Click delete icon on variable row
   - Expected: Row removed from table via onDeleteEnvVariable
   - Verification: Model array updated, UI reflects deletion

9. **Step 9 - Execute Run Command**
   - Action: Click "Run" button to confirm configuration
   - Expected: Dialog closes, POST to `/api/projects/{id}/run`
   - Verification: Request contains all configuration parameters

10. **Step 10 - Terminal Launch**
    - Action: Wait for run response with terminal URL
    - Expected: New browser tab opens with terminal interface
    - Verification: Project status updates to "running"

### Expected Results
- **Configuration Dialog Criteria**:
  - Dialog opens within 200ms with all fields populated
  - Form controls properly bound to configuration model
  - All options have appropriate defaults for development
  - Dialog supports keyboard navigation and accessibility

- **Configuration Validation Criteria**:
  - Port number validated as numeric within valid range
  - Environment selection limited to valid options
  - Environment variables allow flexible key-value pairs
  - No validation errors for default configuration

- **Execution Criteria**:
  - Run command sent with complete configuration
  - Terminal URL received and launched successfully
  - Project status updates reflect running state
  - Success feedback provided to user

### Error Handling Scenarios
1. **Port Already in Use**:
   - Action: Attempt to run with occupied port
   - Expected: Error message about port conflict
   - Verification: Suggests alternative port or kill process

2. **Missing Dependencies**:
   - Action: Run project with missing npm packages
   - Expected: Clear error about missing dependencies
   - Verification: Suggests running npm install first

3. **Invalid Environment Variables**:
   - Action: Enter invalid variable names (spaces, special chars)
   - Expected: Validation prevents invalid names
   - Verification: User guided to correct format

4. **Resource Limitations**:
   - Action: Run when system resources exhausted
   - Expected: Error about insufficient memory/CPU
   - Verification: Advises closing other applications

### Post-Conditions
- Project running in specified environment
- Terminal accessible for monitoring output
- Process can be stopped through UI or terminal
- Configuration saved for future runs (optional)
- Project status accurately reflects running state

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Terminal Support**: Browser supports opening new tabs/windows
- **Network**: Local network access for development server
- **Resources**: Adequate CPU/memory for project execution

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Usability, Functional Suitability)
- **SAP Standard**: SAP Fiori Design Guidelines - Dialog Patterns
- **Security**: Environment variables handled securely
- **Development Standards**: Standard Node.js execution patterns

### Integration Testing
- **Process Management**: Verify process lifecycle handling
- **Terminal Integration**: Ensure terminal properly connected
- **Hot Reload**: Test file watching functionality
- **Debug Integration**: Verify debug mode enables breakpoints

---

## Test Case ID: TC-UI-AGT-052
**Test Objective**: Verify project deployment functionality with configuration and progress tracking  
**Business Process**: Application Deployment and Release Management  
**SAP Module**: A2A Agents Developer Portal - Deployment Operations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-052
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Deployment Process
- **Execution Method**: Manual/Automated
- **Risk Level**: Very High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:116-317`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:41-43`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/fragment/DeployConfigDialog.fragment.xml`
- **Functions Under Test**: `onDeployProject()`, `onConfirmDeployProject()`, `_updateDeploymentProgress()`

### Test Preconditions
1. **Build Status**: Project successfully built and artifacts available
2. **Deployment Permissions**: User has deployment rights for target environment
3. **Target Environment**: Staging/production infrastructure available
4. **Health Checks**: Application health endpoints implemented
5. **Rollback Capability**: Previous version available for rollback

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Target Environment | staging | String | Environment Select |
| Deployment Strategy | rolling | String | Strategy Selection |
| Replicas | 2 | Number | Instance Configuration |
| Health Check Path | /health | String | Health Endpoint |
| Auto Scaling | Enabled | Boolean | Scaling Switch |
| Min/Max Instances | 1/5 | Numbers | Scaling Config |
| CPU Threshold | 70% | Number | Performance Metric |
| Memory Limit | 512Mi | String | Resource Allocation |
| Rollback on Failure | true | Boolean | Safety Setting |

### Test Procedure Steps
1. **Step 1 - Open Deployment Configuration**
   - Action: Click "Deploy" button in project header
   - Expected: DeployConfigDialog opens with three tabs
   - Verification: Basic Config, Auto Scaling, and Advanced tabs visible

2. **Step 2 - Basic Configuration Tab**
   - Action: Review and modify basic deployment settings
   - Expected: Environment selector, strategy options, replicas, health check visible
   - Verification: Staging selected by default, rolling deployment chosen

3. **Step 3 - Select Deployment Strategy**
   - Action: Choose between Rolling and Blue-Green deployment
   - Expected: Radio button selection updates deployment approach
   - Verification: Strategy change reflected in configuration model

4. **Step 4 - Configure Auto Scaling**
   - Action: Switch to Auto Scaling tab and enable scaling
   - Expected: Scaling controls become enabled when switch is ON
   - Verification: Min/max instances, CPU threshold, memory limit configurable

5. **Step 5 - Advanced Options Setup**
   - Action: Navigate to Advanced tab for scripts and rollback
   - Expected: Pre/post deployment script areas and rollback checkbox
   - Verification: Scripts accept shell commands, rollback enabled by default

6. **Step 6 - Initiate Deployment**
   - Action: Click "Deploy" button to start deployment
   - Expected: Config dialog closes, progress dialog opens
   - Verification: Progress dialog shows target environment and initialization

7. **Step 7 - Deployment Progress Tracking**
   - Action: Monitor deployment progress and step execution
   - Expected: Real-time progress updates with step-by-step status
   - Verification: Progress bar advances, current step displayed

8. **Step 8 - Step Status Visualization**
   - Action: Observe deployment steps with status icons
   - Expected: Each step shows icon (pending/running/completed/failed)
   - Verification: Icons update as steps progress through states

9. **Step 9 - Successful Deployment**
   - Action: Wait for deployment completion
   - Expected: Success message, option to open deployed application
   - Verification: Confirmation dialog offers to open deployment URL

10. **Step 10 - View Deployment Logs**
    - Action: Click "View Logs" button during or after deployment
    - Expected: New tab opens with deployment logs
    - Verification: Comprehensive logs accessible for troubleshooting

### Expected Results
- **Configuration Criteria**:
  - All deployment options properly saved and validated
  - Environment-specific settings applied correctly
  - Auto-scaling parameters within valid ranges
  - Health check endpoints validated before deployment

- **Progress Tracking Criteria**:
  - Real-time progress updates every 2 seconds
  - Step-by-step execution visibility
  - Clear status indicators for each phase
  - Accurate completion percentage

- **Deployment Success Criteria**:
  - Application deployed to correct environment
  - All replicas healthy and serving traffic
  - Deployment URL accessible and functional
  - Rollback capability preserved

### Error Handling Scenarios
1. **Pre-deployment Script Failure**:
   - Action: Configure failing pre-deployment script
   - Expected: Deployment stops, error clearly indicated
   - Verification: Failed step highlighted, logs available

2. **Health Check Failure**:
   - Action: Deploy with incorrect health check path
   - Expected: Deployment detects unhealthy instances
   - Verification: Automatic rollback initiated if enabled

3. **Resource Constraints**:
   - Action: Deploy with insufficient resources
   - Expected: Clear error about resource limitations
   - Verification: Suggestions for resolution provided

4. **Network/Infrastructure Issues**:
   - Action: Simulate infrastructure unavailability
   - Expected: Deployment fails gracefully with retry option
   - Verification: State preserved for retry attempt

### Post-Conditions
- Application successfully deployed to target environment
- Previous version preserved for potential rollback
- Deployment history updated with new entry
- Monitoring and alerts configured for new deployment
- All deployment artifacts and logs archived

### Test Environment Requirements
- **Target Infrastructure**: Kubernetes/Cloud platform ready
- **Load Balancer**: Configured for traffic distribution
- **Monitoring**: Metrics and logging infrastructure active
- **Rollback System**: Previous versions accessible

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Reliability, Security)
- **SAP Standard**: SAP Fiori Design Guidelines - Progressive Disclosure
- **Deployment Standards**: Blue-Green, Rolling deployment patterns
- **Security**: Secure credential management, encrypted communications

### Integration Testing
- **CI/CD Pipeline**: Integration with build artifacts
- **Container Registry**: Image availability verification
- **Service Mesh**: Traffic routing validation
- **Monitoring Systems**: Metrics and alerts configuration

---

## Test Case ID: TC-UI-AGT-053
**Test Objective**: Verify integrated terminal functionality with WebSocket support  
**Business Process**: Development Environment Terminal Access  
**SAP Module**: A2A Agents Developer Portal - Terminal Integration  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-053
- **Test Priority**: High (P2)
- **Test Type**: Functional, Real-time Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectDetail.controller.js:1064-1319`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectDetail.view.xml:301-356`
- **Functions Under Test**: `_initializeTerminal()`, `onTerminalCommandSubmit()`, `_connectTerminalWebSocket()`

### Test Preconditions
1. **Project Access**: Valid project loaded in ProjectDetail view
2. **Terminal Permissions**: User has terminal access rights
3. **WebSocket Support**: Browser supports WebSocket connections
4. **Backend Terminal Service**: Terminal service endpoints operational
5. **Network Connectivity**: Stable connection for real-time communication

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Initial Directory | /projects/{projectId} | String | Project Context |
| Test Commands | ls, pwd, echo "test", npm --version | Array | Command Tests |
| Terminal Settings | {fontSize: 14, theme: "dark"} | Object | User Preferences |
| Session Count | 1-3 concurrent sessions | Number | Multi-Terminal |
| Output Buffer | 1000 lines | Number | Scrollback Limit |

### Test Procedure Steps
1. **Step 1 - Access Terminal Tab**
   - Action: Click on Terminal tab in project view
   - Expected: Terminal interface loads with welcome message
   - Verification: Terminal shows connected status and working directory

2. **Step 2 - WebSocket Connection**
   - Action: Monitor network for WebSocket establishment
   - Expected: WS connection to `/api/projects/{id}/terminal` successful
   - Verification: Connection status shows "Terminal connected"

3. **Step 3 - Execute Basic Command**
   - Action: Type "ls" in command input and press Enter
   - Expected: Command echoed in terminal, output displayed
   - Verification: Directory listing shows project files

4. **Step 4 - Command History Navigation**
   - Action: Press up arrow key in command input
   - Expected: Previous command "ls" appears in input
   - Verification: Command history navigation works

5. **Step 5 - Execute Multiple Commands**
   - Action: Run "pwd", "echo test", "npm --version" sequentially
   - Expected: Each command executes with appropriate output
   - Verification: Output displayed with proper formatting

6. **Step 6 - Create New Terminal Session**
   - Action: Click "New Terminal" button
   - Expected: Additional terminal session created
   - Verification: New session with unique ID and separate output

7. **Step 7 - Clear Terminal Output**
   - Action: Select Terminal menu > Clear Terminal
   - Expected: Current session output cleared
   - Verification: Only "Terminal cleared" message remains

8. **Step 8 - Export Terminal Output**
   - Action: Select Terminal menu > Export Output
   - Expected: Terminal history downloaded as text file
   - Verification: File contains timestamped command history

9. **Step 9 - Terminal Settings**
   - Action: Open Terminal Settings dialog
   - Expected: Settings dialog with font size, theme options
   - Verification: Settings changes apply to terminal display

10. **Step 10 - Error Command Handling**
    - Action: Execute invalid command "invalidcmd123"
    - Expected: Error message displayed in terminal
    - Verification: Error shown in red/error styling

### Expected Results
- **Connection Criteria**:
  - WebSocket connects within 2 seconds
  - Fallback to REST API if WebSocket fails
  - Connection status clearly indicated
  - Auto-reconnect on disconnection

- **Command Execution Criteria**:
  - Commands execute within 500ms
  - Output streams (stdout/stderr) properly distinguished
  - Command history maintained across session
  - Multi-line output handled correctly

- **User Interface Criteria**:
  - Terminal displays with monospace font
  - Scrollback buffer works smoothly
  - Auto-scroll to bottom on new output
  - Color coding for different output types

### Error Handling Scenarios
1. **WebSocket Connection Failure**:
   - Action: Simulate WebSocket connection error
   - Expected: Fallback to REST API for command execution
   - Verification: Commands still execute via HTTP POST

2. **Command Timeout**:
   - Action: Execute long-running command
   - Expected: Timeout warning after 30 seconds
   - Verification: Option to terminate or continue waiting

3. **Invalid Command Syntax**:
   - Action: Enter malformed command
   - Expected: Shell error message displayed
   - Verification: Terminal remains functional

4. **Session Disconnection**:
   - Action: Simulate network interruption
   - Expected: "Terminal disconnected" message
   - Verification: Reconnect option available

### Post-Conditions
- Terminal sessions properly cleaned up
- Command history persisted for session
- WebSocket connections closed gracefully
- No memory leaks from output buffering
- Settings preferences saved

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **WebSocket Protocol**: WS/WSS support required
- **Terminal Backend**: Shell access configured
- **Security**: Sandboxed command execution

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Usability, Performance)
- **SAP Standard**: SAP Fiori Design Guidelines - Embedded Tools
- **Security**: Command injection prevention
- **Accessibility**: Screen reader support for terminal output

### Integration Testing
- **Shell Integration**: Verify bash/sh command compatibility
- **File System Access**: Test file operations via terminal
- **Process Management**: Verify process lifecycle handling
- **Output Streaming**: Test real-time output for long processes

---

## Test Case ID: TC-UI-AGT-054
**Test Objective**: Verify split view (master-detail) rendering and layout management  
**Business Process**: Project Management with Master-Detail Navigation  
**SAP Module**: A2A Agents Developer Portal - FlexibleColumnLayout  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-054
- **Test Priority**: High (P2)
- **Test Type**: Functional, Layout Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:11-15`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:16-46`
- **Functions Under Test**: `onInit()`, `onStateChanged()`, `_updateLayout()`

### Test Preconditions
1. **Browser Compatibility**: Modern browser with flexbox support
2. **Screen Resolution**: Minimum 1024px width for split view
3. **Projects Available**: At least 3 projects in the system
4. **User Permissions**: Read access to project list
5. **Route Access**: Navigation to projectMasterDetail route

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Initial Layout | OneColumn | String | Layout Model |
| Screen Width | 1920px, 1024px, 768px | Number | Viewport |
| Project Count | 5+ projects | Number | Test Data |
| Selected Project | project-001 | String | Master List |
| Layout States | OneColumn, TwoColumns, ThreeColumns | Array | FCL States |

### Test Procedure Steps
1. **Step 1 - Initial View Render**
   - Action: Navigate to ProjectMasterDetail view
   - Expected: FlexibleColumnLayout renders in OneColumn mode
   - Verification: Only master list visible, no detail pane

2. **Step 2 - Master List Display**
   - Action: Observe master column content
   - Expected: Project list displays with title showing count
   - Verification: Title shows "Projects (X)" where X is count

3. **Step 3 - Master Column Elements**
   - Action: Check master column header and controls
   - Expected: Create and Refresh buttons visible in header
   - Verification: Search field present below header

4. **Step 4 - Select Project Item**
   - Action: Click on a project in master list
   - Expected: Layout transitions to TwoColumns mode
   - Verification: Detail view appears in mid column

5. **Step 5 - Layout State Change**
   - Action: Monitor layout state change event
   - Expected: onStateChanged handler executes
   - Verification: Layout model updates to TwoColumns

6. **Step 6 - Detail View Content**
   - Action: Inspect detail view rendering
   - Expected: Selected project details display correctly
   - Verification: Detail model populated with project data

7. **Step 7 - Responsive Behavior**
   - Action: Resize browser window to 768px width
   - Expected: Layout adapts to single column on mobile
   - Verification: Detail view takes full width when selected

8. **Step 8 - Navigation Buttons**
   - Action: Check navigation button visibility
   - Expected: Back button appears in detail view on mobile
   - Verification: Navigation works between master and detail

9. **Step 9 - Full Screen Mode**
   - Action: Click full screen button in detail view
   - Expected: Detail view expands to full screen
   - Verification: Master column hidden, layout = OneColumn

10. **Step 10 - Exit Full Screen**
    - Action: Click exit full screen button
    - Expected: Return to previous TwoColumns layout
    - Verification: Both master and detail visible again

### Expected Results
- **Layout Rendering Criteria**:
  - FlexibleColumnLayout initializes correctly
  - Default OneColumn layout on load
  - Smooth transitions between layout states
  - No layout flickering or jumps

- **Responsive Behavior Criteria**:
  - Automatic layout adaptation based on screen size
  - Proper column widths at different breakpoints
  - Touch-friendly navigation on mobile devices
  - Consistent behavior across browsers

- **State Management Criteria**:
  - Layout model accurately tracks current state
  - Navigation buttons appear/disappear correctly
  - Selected item remains highlighted in master
  - State persists during view navigation

### Error Handling Scenarios
1. **No Projects Available**:
   - Action: Load view with empty project list
   - Expected: Empty state message in master column
   - Verification: No JavaScript errors, clean UI

2. **Detail Load Failure**:
   - Action: Select project that fails to load
   - Expected: Error message in detail pane
   - Verification: Master list remains functional

3. **Invalid Layout State**:
   - Action: Manually set invalid layout type
   - Expected: Fallback to OneColumn layout
   - Verification: Error logged, UI recovers

4. **Browser Resize During Transition**:
   - Action: Resize window while layout animating
   - Expected: Graceful handling of resize
   - Verification: Layout settles to correct state

### Post-Conditions
- Layout state correctly reflects current view
- Selected project highlighted in master list
- Detail view shows accurate project information
- Navigation history maintained
- No memory leaks from view transitions

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Screen Sizes**: Desktop (1920px), Tablet (1024px), Mobile (768px)
- **Network**: Stable connection for data loading
- **Performance**: Smooth animations at 60fps

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Usability, Compatibility)
- **SAP Standard**: SAP Fiori Design Guidelines - Flexible Column Layout
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG 2.1 Level AA navigation support

### Integration Testing
- **Router Integration**: Verify deep linking to specific projects
- **Model Binding**: Test two-way binding between views
- **Event Propagation**: Ensure state changes propagate correctly
- **Performance**: Measure layout transition performance

---

## Test Case ID: TC-UI-AGT-055
**Test Objective**: Verify pane resizing functionality in split view layout  
**Business Process**: Customizable Layout Management  
**SAP Module**: A2A Agents Developer Portal - Resizable Columns  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-055
- **Test Priority**: Medium (P3)
- **Test Type**: Functional, User Interaction
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:440-533`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml`
- **Functions Under Test**: `onBeginColumnResize()`, `_onResize()`, `_applyColumnWidths()`

### Test Preconditions
1. **Split View Active**: ProjectMasterDetail view in TwoColumns layout
2. **Resize Handle Visible**: Resize handle rendered between columns
3. **Project Selected**: Detail view populated with content
4. **Browser Support**: Mouse/touch events supported
5. **Sufficient Width**: Screen width > 1024px for resizing

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Min Column Width | 320px | Number | Layout Model |
| Max Begin Column | 600px | Number | Layout Model |
| Default Split | 30%/70% | String | Initial State |
| Drag Distance | 100px, -100px | Number | User Input |
| Touch Events | touchstart, touchmove | Events | Mobile Testing |

### Test Procedure Steps
1. **Step 1 - Locate Resize Handle**
   - Action: Find resize handle between master and detail columns
   - Expected: Resize handle visible with appropriate cursor
   - Verification: Cursor changes to resize indicator on hover

2. **Step 2 - Start Resize Operation**
   - Action: Click and hold resize handle (mousedown)
   - Expected: Resize mode activated, body gets resizing class
   - Verification: Text selection disabled during resize

3. **Step 3 - Drag to Increase Master Width**
   - Action: Drag handle 100px to the right
   - Expected: Master column expands, detail column shrinks
   - Verification: Columns resize in real-time during drag

4. **Step 4 - Verify Constraints**
   - Action: Continue dragging beyond max width (600px)
   - Expected: Master column stops at maximum width
   - Verification: Width constrained to maxBeginColumnWidth

5. **Step 5 - Release Resize Handle**
   - Action: Release mouse button (mouseup)
   - Expected: Resize operation completes, preferences saved
   - Verification: Column widths persist in localStorage

6. **Step 6 - Drag to Decrease Master Width**
   - Action: Drag handle left to shrink master column
   - Expected: Master column shrinks, detail expands
   - Verification: Smooth resize animation

7. **Step 7 - Test Minimum Constraint**
   - Action: Drag to shrink below 320px minimum
   - Expected: Master column stops at minimum width
   - Verification: Width constrained to minColumnWidth

8. **Step 8 - Touch Device Resize**
   - Action: Use touch events to resize on tablet
   - Expected: Touch drag works same as mouse drag
   - Verification: Touch events properly handled

9. **Step 9 - Reload and Verify Persistence**
   - Action: Refresh page and return to split view
   - Expected: Column widths restored from preferences
   - Verification: Previous resize state maintained

10. **Step 10 - Reset Column Widths**
    - Action: Trigger reset column widths function
    - Expected: Columns return to 30%/70% default
    - Verification: Toast message confirms reset

### Expected Results
- **Resize Interaction Criteria**:
  - Resize handle clearly visible and interactive
  - Smooth real-time resizing during drag
  - No flickering or jumping during resize
  - Touch and mouse events work equally well

- **Constraint Enforcement Criteria**:
  - Minimum width (320px) enforced
  - Maximum width (600px) enforced
  - Proportional sizing maintained
  - Total width always equals 100%

- **Persistence Criteria**:
  - Column widths saved to localStorage
  - Preferences loaded on view initialization
  - User settings survive page refresh
  - Reset function restores defaults

### Error Handling Scenarios
1. **Rapid Resize Movements**:
   - Action: Quickly drag handle back and forth
   - Expected: Smooth handling without lag
   - Verification: No performance degradation

2. **Window Resize During Drag**:
   - Action: Resize browser window while dragging
   - Expected: Graceful handling of viewport change
   - Verification: Column percentages recalculated

3. **LocalStorage Disabled**:
   - Action: Test with localStorage disabled
   - Expected: Resize still works, no persistence
   - Verification: No JavaScript errors thrown

4. **Touch Gesture Conflicts**:
   - Action: Perform conflicting touch gestures
   - Expected: Resize takes precedence when active
   - Verification: No unintended scrolling

### Post-Conditions
- Column widths set to user preference
- Resize preferences saved successfully
- UI remains responsive and functional
- No memory leaks from event handlers
- Resize handle ready for next operation

### Test Environment Requirements
- **Browser Support**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Input Devices**: Mouse, touchpad, touchscreen
- **Screen Resolution**: Minimum 1024px width
- **Performance**: 60fps during resize operations

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Usability, Performance)
- **SAP Standard**: SAP Fiori Design Guidelines - Flexible Layouts
- **Accessibility**: Keyboard alternative for resizing
- **Touch Standards**: Touch target minimum 44x44px

### Integration Testing
- **State Management**: Verify layout model updates correctly
- **CSS Application**: Test dynamic style updates
- **Event Handling**: Ensure proper event cleanup
- **Performance Impact**: Measure resize operation performance

---

## Test Case ID: TC-UI-AGT-056
**Test Objective**: Verify minimum size constraints in split view layout  
**Business Process**: Layout Constraint Management  
**SAP Module**: A2A Agents Developer Portal - Size Constraints  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-056
- **Test Priority**: Medium (P3)
- **Test Type**: Functional, Constraint Validation
- **Execution Method**: Manual/Automated
- **Risk Level**: Low

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:34-37,464-467`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml`
- **Functions Under Test**: `_onResize()`, constraint enforcement logic

### Test Preconditions
1. **Split View Active**: ProjectMasterDetail in TwoColumns layout
2. **Resize Enabled**: Resize functionality operational
3. **Constraints Defined**: Min/max width values set in model
4. **Multiple Projects**: Content available for testing
5. **Various Screen Sizes**: Test on different viewport widths

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Min Column Width | 320px | Number | Layout Model |
| Max Begin Column | 600px | Number | Layout Model |
| Screen Widths | 768px, 1024px, 1920px | Array | Test Viewports |
| Content Types | Short text, Long text, Tables | Array | Test Content |
| Extreme Values | 0px, 9999px | Array | Edge Cases |

### Test Procedure Steps
1. **Step 1 - Verify Default Minimum**
   - Action: Check initial column width configuration
   - Expected: Minimum width set to 320px in model
   - Verification: Model property "/minColumnWidth" equals 320

2. **Step 2 - Test Resize Below Minimum**
   - Action: Attempt to resize master column below 320px
   - Expected: Column width stops at 320px minimum
   - Verification: Width cannot be reduced further

3. **Step 3 - Content Visibility at Minimum**
   - Action: Set master column to minimum width
   - Expected: Essential content remains visible and usable
   - Verification: No horizontal scrolling, text readable

4. **Step 4 - Responsive Breakpoint Test**
   - Action: Resize browser window to 768px width
   - Expected: Layout switches to single column mode
   - Verification: Minimum constraints don't apply in mobile

5. **Step 5 - Maximum Width Constraint**
   - Action: Attempt to resize master beyond 600px
   - Expected: Column width stops at 600px maximum
   - Verification: Detail column maintains minimum space

6. **Step 6 - Percentage-Based Constraints**
   - Action: Test constraints on different screen widths
   - Expected: Pixel constraints convert to percentages correctly
   - Verification: Proportions maintained across viewports

7. **Step 7 - Content Overflow Handling**
   - Action: Add long content to minimum-width column
   - Expected: Content wraps or shows ellipsis appropriately
   - Verification: No layout breaking or overlap

8. **Step 8 - Programmatic Size Setting**
   - Action: Set column width programmatically below minimum
   - Expected: Constraint enforcement in setter method
   - Verification: Width adjusted to minimum automatically

9. **Step 9 - Touch Device Constraints**
   - Action: Test resize constraints on touch device
   - Expected: Same constraints apply for touch resize
   - Verification: Touch-friendly minimum sizes

10. **Step 10 - Constraint Validation Messages**
    - Action: Attempt extreme resize values
    - Expected: No error messages, graceful constraint
    - Verification: User experience remains smooth

### Expected Results
- **Constraint Enforcement Criteria**:
  - Minimum width (320px) strictly enforced
  - Maximum width (600px) prevents oversizing
  - Constraints apply during resize and programmatic setting
  - No way to bypass constraints through UI

- **Usability at Extremes Criteria**:
  - Content remains functional at minimum width
  - Navigation elements stay accessible
  - Text remains readable without horizontal scroll
  - Action buttons don't overlap

- **Responsive Behavior Criteria**:
  - Constraints adapt to viewport changes
  - Mobile layout ignores desktop constraints
  - Smooth transitions between breakpoints
  - No layout jumps or flickers

### Error Handling Scenarios
1. **Invalid Constraint Values**:
   - Action: Set minimum > maximum in configuration
   - Expected: Validation prevents invalid state
   - Verification: Defaults applied, error logged

2. **Viewport Smaller Than Minimum**:
   - Action: Resize window smaller than total minimums
   - Expected: Layout adapts to single column
   - Verification: No content cut off or hidden

3. **Dynamic Content Changes**:
   - Action: Load content requiring more space
   - Expected: Layout maintains constraints
   - Verification: Scrolling enabled if needed

4. **Browser Zoom Effects**:
   - Action: Test with browser zoom 50%-200%
   - Expected: Constraints scale appropriately
   - Verification: Layout remains functional

### Post-Conditions
- Column widths respect all constraints
- Layout remains stable and usable
- No broken states possible through UI
- Performance unaffected by constraint checks
- User preferences within valid ranges

### Test Environment Requirements
- **Browser Support**: All modern browsers
- **Screen Sizes**: 768px to 2560px width
- **Zoom Levels**: 50% to 200% browser zoom
- **Device Types**: Desktop, tablet, mobile

### Compliance and Standards
- **ISO Standard**: ISO/IEC 25010 (Reliability, Usability)
- **SAP Standard**: SAP Fiori Design Guidelines - Responsive Design
- **Accessibility**: WCAG 2.1 minimum target sizes
- **Mobile Standards**: Touch-friendly minimum dimensions

### Integration Testing
- **Model Validation**: Test constraint property binding
- **CSS Calculations**: Verify pixel to percentage conversion
- **Event Handling**: Ensure constraints during all interactions
- **Performance**: Measure constraint check overhead

---

## Test Case ID: TC-UI-AGT-057
**Test Objective**: Verify responsive behavior and layout adaptation across different screen sizes  
**Business Process**: Responsive UI Adaptation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-057
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Responsive Design
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:562-691`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml`
- **Functions Under Test**: `_initializeResponsiveHandler()`, `_checkResponsiveBreakpoint()`, `_applyResponsiveLayout()`

### Test Preconditions
1. **Authentication**: Valid user session with project access
2. **Projects**: At least 3 existing projects in the workspace
3. **Test Devices**: Phone (320-599px), Tablet (600-1023px), Desktop (1024px+)
4. **Browser**: Chrome DevTools or actual devices for testing
5. **Initial State**: Desktop view with two-column layout

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Phone Width | 375px | Number | Device Emulation |
| Tablet Width | 768px | Number | Device Emulation |
| Desktop Width | 1280px | Number | Device Emulation |
| Transition Width | 599px, 600px, 1023px, 1024px | Number | Breakpoint Testing |
| Resize Speed | 200ms debounce | Number | Configuration |

### Test Procedure Steps
1. **Step 1 - Desktop View Verification**
   - Action: Load ProjectMasterDetail view on desktop (1280px width)
   - Expected: Two-column layout with 30%/70% split
   - Verification: Both master and detail columns visible, resize handle present

2. **Step 2 - Tablet View Adaptation**
   - Action: Resize browser window to 768px width
   - Expected: Two-column layout adjusts to 40%/60% split
   - Verification: Columns resize, constraints update (max 450px for begin column)

3. **Step 3 - Phone View Transformation**
   - Action: Resize browser window to 375px width
   - Expected: One-column layout, only master list visible
   - Verification: Detail column hidden, resize handle removed

4. **Step 4 - Phone Navigation**
   - Action: Select a project item on phone view
   - Expected: Detail view replaces master view with back button
   - Verification: Full-screen detail view, navigation button visible

5. **Step 5 - Breakpoint Transition**
   - Action: Slowly resize from 595px to 605px
   - Expected: Smooth transition from phone to tablet layout
   - Verification: Layout changes at exactly 600px

6. **Step 6 - Orientation Change**
   - Action: Rotate tablet from portrait (768x1024) to landscape (1024x768)
   - Expected: Layout adapts from tablet to desktop breakpoint
   - Verification: Column widths adjust to desktop defaults

7. **Step 7 - User Resize Persistence**
   - Action: Manually resize columns on desktop, then change to tablet
   - Expected: Manual resize overrides default tablet widths
   - Verification: User preferences maintained across breakpoints

### Expected Results
- **Responsive Breakpoints**:
  - Phone (≤599px): One-column layout, full-width views
  - Tablet (600-1023px): Two-column with 40%/60% split
  - Desktop (≥1024px): Two-column with 30%/70% split
  
- **Layout Behavior**:
  - Smooth transitions between breakpoints
  - Resize handle visibility based on device type
  - Navigation adapts to screen size
  - CSS classes applied: a2a-fcl-phone, a2a-fcl-tablet, a2a-fcl-desktop

- **User Experience**:
  - No content loss during transitions
  - Debounced resize prevents jarring updates
  - Selected state maintained across layouts
  - Back navigation available on phone view

### Error Scenarios
1. **Rapid Resize**: Quick window resizing should be debounced
2. **Boundary Testing**: Exact breakpoint pixels (599, 600, 1023, 1024)
3. **Memory Cleanup**: Event listeners removed on view exit
4. **State Persistence**: Selected project maintained during transitions

### Post-Execution Tasks
- **Verification**: Device model properties correctly updated
- **Cleanup**: Window resize listeners properly removed
- **Performance**: No memory leaks from event handlers
- **Documentation**: Responsive behavior matches design specs

### Test Data Management
- **Breakpoint Configuration**: Stored in _oResponsiveConfig
- **Device State**: Managed in device model
- **Layout State**: Synchronized with oLayoutModel
- **User Preferences**: _bUserResized flag tracks manual changes

### Dependencies
- **jQuery**: Window resize event handling
- **SAP UI5**: FlexibleColumnLayout responsive features
- **CSS**: Breakpoint-specific styling classes
- **Device Detection**: Window width measurement

### Performance Criteria
- **Resize Debounce**: 200ms delay for performance
- **Transition Speed**: < 300ms layout change
- **Memory Usage**: No increase after multiple resizes
- **Event Cleanup**: All listeners removed on exit

---

## Test Case ID: TC-UI-AGT-058
**Test Objective**: Verify agent template selection and automatic configuration application  
**Business Process**: Agent Creation with Templates  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-058
- **Test Priority**: High (P2)
- **Test Type**: Functional, Configuration Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/AgentBuilder.controller.js:onTemplateChange`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:78-88`
- **Functions Under Test**: `onTemplateChange()`, `_applyTemplate()`, `_getTemplateConfig()`

### Test Preconditions
1. **Authentication**: Valid user session with agent creation permissions
2. **Project Context**: Active project selected with agent creation rights
3. **Agent Builder**: Clean agent builder instance without prior configurations
4. **Templates**: All 5 template types available in the system
5. **Browser State**: No cached template configurations

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Template Types | blank, data-processor, api-integrator, ml-analyzer, workflow-coordinator | Array | Template Registry |
| Agent Name | TestAgent_[Template] | String | Dynamic Generation |
| Agent Type | reactive, proactive, collaborative | String | Template Config |
| Skill Count | 0-5 skills per template | Number | Template Definition |
| Handler Count | 0-3 handlers per template | Number | Template Definition |

### Test Procedure Steps
1. **Step 1 - Access Agent Builder**
   - Action: Navigate to Agent Builder from project context
   - Expected: Agent Builder loads with blank template selected
   - Verification: Template dropdown shows "Blank Template" as default

2. **Step 2 - Data Processor Template**
   - Action: Select "Data Processor" from template dropdown
   - Expected: Agent configuration auto-populates with data processing skills
   - Verification: 3 skills added (Data Ingestion, Transformation, Validation)

3. **Step 3 - API Integrator Template**
   - Action: Switch to "API Integrator" template
   - Expected: Previous config cleared, API skills loaded
   - Verification: REST Handler, Authentication, Response Mapping skills present

4. **Step 4 - ML Analyzer Template**
   - Action: Select "ML Analyzer" template
   - Expected: Machine learning specific configuration applied
   - Verification: Model Loading, Prediction, Training skills configured

5. **Step 5 - Workflow Coordinator Template**
   - Action: Choose "Workflow Coordinator" template
   - Expected: Workflow management configuration loaded
   - Verification: Task Scheduling, State Management, Event Routing skills

6. **Step 6 - Template Persistence Check**
   - Action: Make custom changes, then switch templates
   - Expected: Confirmation dialog warns about losing changes
   - Verification: User can cancel or proceed with template switch

7. **Step 7 - Blank Template Reset**
   - Action: Select "Blank Template" after using other templates
   - Expected: All auto-configured elements cleared
   - Verification: Empty skills, handlers, and default agent type

### Expected Results
- **Template Application**:
  - Each template applies complete configuration set
  - Skills appropriate to template type
  - Handlers match template requirements
  - Agent type set based on template nature
  
- **Configuration Details**:
  - Data Processor: 3 skills, reactive type, data event handlers
  - API Integrator: 4 skills, reactive type, HTTP event handlers
  - ML Analyzer: 3 skills, proactive type, prediction handlers
  - Workflow Coordinator: 4 skills, collaborative type, workflow handlers
  - Blank: No skills, reactive type, no handlers

- **User Experience**:
  - Immediate template application on selection
  - Clear visual feedback of configuration changes
  - Warning before overwriting custom configurations
  - Smooth transition between templates

### Error Scenarios
1. **Template Load Failure**: Graceful fallback to blank template
2. **Partial Configuration**: Complete rollback if any part fails
3. **Concurrent Edits**: Lock template during application
4. **Invalid Template**: Error message and maintain current state

### Post-Execution Tasks
- **Verification**: All template configurations correctly applied
- **State Check**: Model consistency after template switches
- **Memory**: No lingering configurations from previous templates
- **Performance**: Template application completes < 500ms

### Test Data Management
- **Template Definitions**: Stored in _templateConfigs object
- **Skill Library**: Predefined skill objects with icons
- **Handler Templates**: Event-handler mapping configurations
- **Validation Rules**: Template-specific validation sets

### Dependencies
- **Agent Model**: JSONModel for agent configuration
- **Template Service**: Template definition provider
- **Skill Registry**: Available skills catalog
- **Event System**: Handler configuration system

### Performance Criteria
- **Template Load Time**: < 200ms for configuration retrieval
- **Application Time**: < 300ms for full template application
- **UI Update**: < 100ms for visual feedback
- **Memory Usage**: < 5MB per template configuration

---

## Test Case ID: TC-UI-AGT-059
**Test Objective**: Verify master-detail selection synchronization across different interaction patterns  
**Business Process**: Project Selection and Navigation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-059
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, UI Synchronization
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:698-736`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:46-94`
- **Functions Under Test**: `_syncMasterSelection()`, `_ensureItemVisible()`, `onSelectionChange()`, `onItemPress()`

### Test Preconditions
1. **Authentication**: Valid user session with project access
2. **Projects**: At least 10 projects in the master list
3. **Initial State**: Two-column layout with no selection
4. **Browser**: Desktop view (1280px+ width)
5. **Data**: Projects loaded in master list

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project IDs | proj_001 through proj_010 | Array | Test Data |
| Selection Methods | Click, Keyboard, API, URL | Array | User Actions |
| Scroll Positions | Top, Middle, Bottom | Array | List State |
| Selection States | None, Single, Changed | Array | Selection Model |
| Layout States | OneColumn, TwoColumns, ThreeColumns | Array | FCL States |

### Test Procedure Steps
1. **Step 1 - Initial Selection**
   - Action: Click on first project in master list
   - Expected: Project selected, detail view loads, selection highlighted
   - Verification: Selected item has CSS class "sapMLIBSelected"

2. **Step 2 - Selection Persistence**
   - Action: Scroll master list to bottom, then back to top
   - Expected: Original selection remains visible and highlighted
   - Verification: Selected project still marked, detail view unchanged

3. **Step 3 - Programmatic Selection**
   - Action: Navigate via URL to specific project (e.g., #/projects/proj_005)
   - Expected: Master list syncs to show proj_005 selected
   - Verification: Item scrolls into view if needed, selection updated

4. **Step 4 - Selection Change**
   - Action: Click different project while one is selected
   - Expected: Previous selection cleared, new selection applied
   - Verification: Only one item selected, detail view updates

5. **Step 5 - Keyboard Navigation Sync**
   - Action: Use arrow keys to navigate through list
   - Expected: Selection follows keyboard focus, detail updates
   - Verification: Selection and focus remain synchronized

6. **Step 6 - Selection After Search**
   - Action: Search for project, select from filtered results
   - Expected: Selection maintained when clearing search
   - Verification: Selected item visible in full list

7. **Step 7 - Selection Recovery**
   - Action: Refresh page with project selected via URL
   - Expected: Selection restored after page load
   - Verification: Correct project selected and detail loaded

### Expected Results
- **Visual Feedback**:
  - Selected item has distinct highlight color
  - Selection indicator visible on list item
  - No multiple selections allowed
  - Selection persists during scrolling
  
- **Synchronization Behavior**:
  - Master selection updates when detail loads
  - URL updates reflect current selection
  - Selection visible after list operations
  - Smooth scroll to selected item when needed

- **State Management**:
  - Selection stored in model property
  - Selection ID tracked for recovery
  - Selection cleared on invalid project
  - Selection maintained across layout changes

### Error Scenarios
1. **Invalid Project ID**: Clear selection, show empty state
2. **Network Failure**: Maintain selection, show error in detail
3. **Concurrent Updates**: Last selection wins, no conflicts
4. **Missing Project**: Remove from selection, show message

### Post-Execution Tasks
- **Verification**: Selection model consistency
- **UI State**: Visual selection indicators correct
- **Performance**: Selection sync < 100ms
- **Memory**: No leaked event handlers

### Test Data Management
- **Selection Model**: masterModel>/selectedProjectId
- **List Reference**: byId("masterList")
- **Selection State**: oList.getSelectedItem()
- **Scroll Position**: DOM scrollIntoView tracking

### Dependencies
- **SAP UI5 List**: Selection mode "SingleSelectMaster"
- **Router**: Navigation parameter handling
- **Models**: Master and Detail model binding
- **DOM**: ScrollIntoView API support

### Performance Criteria
- **Selection Time**: < 50ms for visual update
- **Scroll Animation**: < 300ms smooth scroll
- **Model Update**: < 20ms for property set
- **Total Sync Time**: < 100ms end-to-end

---

## Test Case ID: TC-UI-AGT-060
**Test Objective**: Verify detail view updates with loading states, animations, and error handling  
**Business Process**: Project Detail Information Display  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-060
- **Test Priority**: High (P2)
- **Test Type**: Functional, UI Updates
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:120-237`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:126-139`
- **Functions Under Test**: `_loadProjectDetails()`, `getUpdateStatistics()`, `forceDetailRefresh()`

### Test Preconditions
1. **Authentication**: Valid user session with project access
2. **Projects**: At least 5 projects with different data sizes
3. **Network**: Variable network conditions for testing
4. **Browser**: CSS animation support enabled
5. **Initial State**: Master-detail view loaded

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Update Delays | 0ms, 500ms, 2000ms | Array | Network Simulation |
| Project Sizes | Small, Medium, Large | Array | Test Data |
| Error Types | 404, 500, Timeout | Array | Error Scenarios |
| Animation Duration | 300ms | Number | Controller Config |
| Update History Size | 10 items | Number | History Limit |

### Test Procedure Steps
1. **Step 1 - Initial Detail Load**
   - Action: Select first project from master list
   - Expected: Loading indicator appears, detail content fades out
   - Verification: Loading state true, busy indicator visible

2. **Step 2 - Successful Update**
   - Action: Wait for detail load to complete
   - Expected: Content fades in, loading indicator hidden, toast message
   - Verification: Loading false, update count incremented

3. **Step 3 - Rapid Selection Changes**
   - Action: Quickly select different projects (< 300ms intervals)
   - Expected: Previous load cancelled, only latest loads
   - Verification: No overlapping updates, correct final state

4. **Step 4 - Update Statistics**
   - Action: After 5 selections, check update statistics
   - Expected: Correct counts, average duration calculated
   - Verification: Statistics match actual update history

5. **Step 5 - Error Handling**
   - Action: Select project that returns 404 error
   - Expected: Error message box, detail shows error state
   - Verification: Error tracked in history, loading false

6. **Step 6 - Force Refresh**
   - Action: Call forceDetailRefresh() on current project
   - Expected: Detail reloads with animation
   - Verification: Same project reloaded, update count increased

7. **Step 7 - Animation Verification**
   - Action: Observe fade out/in during updates
   - Expected: Smooth 300ms transitions
   - Verification: CSS class a2a-detail-updating applied/removed

### Expected Results
- **Loading States**:
  - Immediate loading indicator on selection
  - Fade out animation during load
  - Fade in animation after load
  - Toast notification on success
  
- **Update Tracking**:
  - Update count increments correctly
  - Last update timestamp accurate
  - History limited to 10 items
  - Duration tracking in milliseconds

- **Error Handling**:
  - Error message box for failures
  - Error state in detail model
  - Failed updates tracked in history
  - Loading state cleared on error

### Error Scenarios
1. **Network Timeout**: Show timeout error after 30s
2. **Server Error**: Display server error message
3. **Invalid Data**: Handle malformed response gracefully
4. **Concurrent Updates**: Cancel previous, load latest

### Post-Execution Tasks
- **Verification**: Update history integrity
- **Performance**: Average load time < 1000ms
- **Memory**: No memory leaks from animations
- **UI State**: All transitions completed

### Test Data Management
- **Update History**: _detailUpdateHistory array
- **Loading State**: detailModel>/loading
- **Update Count**: detailModel>/updateCount
- **Error State**: detailModel>/error, errorMessage

### Dependencies
- **jQuery Ajax**: HTTP requests
- **SAP UI5 Models**: Data binding
- **CSS Animations**: Fade effects
- **MessageToast**: Success notifications
- **MessageBox**: Error dialogs

### Performance Criteria
- **Load Start**: < 50ms after selection
- **Animation Duration**: 300ms fade effects
- **Data Processing**: < 100ms after response
- **Total Update Time**: < 1500ms typical

### Integration Testing
- **Model Updates**: Verify all properties updated atomically
- **Animation Timing**: Ensure no visual glitches
- **Error Recovery**: Subsequent loads work after error
- **History Management**: Array operations performant

---

## Test Case ID: TC-UI-AGT-061
**Test Objective**: Verify comprehensive keyboard navigation and accessibility shortcuts in master-detail interface  
**Business Process**: Keyboard Accessibility and Navigation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-061
- **Test Priority**: High (P2)
- **Test Type**: Functional, Accessibility
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:559-753`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:40`
- **Functions Under Test**: `_initializeKeyboardNavigation()`, `_onGlobalKeyDown()`, `_navigateList()`, `_navigateToMaster()`, `_navigateToDetail()`

### Test Preconditions
1. **Authentication**: Valid user session with project access
2. **Projects**: At least 5 projects in master list
3. **Browser**: Keyboard events support enabled
4. **Screen Reader**: Optional NVDA/JAWS for accessibility testing
5. **Initial State**: Master-detail view loaded with focus on page

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Keyboard Shortcuts | Ctrl+F, Ctrl+R, Ctrl+N, Esc | Array | Accessibility Standards |
| Navigation Keys | Up, Down, Left, Right + modifiers | Array | Standard Navigation |
| Focus States | Master, Detail, Search, Buttons | Array | UI Elements |
| Preload Cache | 5MB limit | Number | Performance Config |
| Focus Delay | 100ms | Number | UI Responsiveness |

### Test Procedure Steps
1. **Step 1 - Search Field Focus (Ctrl+F)**
   - Action: Press Ctrl+F anywhere in the master-detail view
   - Expected: Search field receives focus, cursor appears
   - Verification: Search field is focused, can type immediately

2. **Step 2 - List Navigation (Arrow Keys)**
   - Action: Focus master list, use Up/Down arrows to navigate
   - Expected: Visual focus moves between items
   - Verification: Focus indicator visible, selection follows focus

3. **Step 3 - Quick List Navigation (Ctrl+Up/Down)**
   - Action: Press Ctrl+Up/Down to navigate list items quickly
   - Expected: Selection changes, detail view updates
   - Verification: Selected project changes, detail loads

4. **Step 4 - Column Navigation (Alt+Left/Right)**
   - Action: Press Alt+Left to focus master, Alt+Right for detail
   - Expected: Focus switches between columns
   - Verification: Focus moves to appropriate column

5. **Step 5 - Create Project Shortcut (Ctrl+N)**
   - Action: Press Ctrl+N from master-detail view
   - Expected: Create project action triggered
   - Verification: Navigation or dialog opens

6. **Step 6 - Refresh Shortcut (Ctrl+R)**
   - Action: Press Ctrl+R to refresh current view
   - Expected: Current data refreshes, toast notification
   - Verification: Data reloaded, appropriate feedback shown

7. **Step 7 - Escape Key Navigation**
   - Action: Press Esc when in detail view
   - Expected: Return to master view (one column layout)
   - Verification: Layout changes to one column

8. **Step 8 - Focus Management**
   - Action: Navigate using Tab key through all interactive elements
   - Expected: Logical tab order, no focus traps
   - Verification: All buttons/fields reachable, skip links work

9. **Step 9 - Preload on Focus**
   - Action: Use keyboard to focus different list items
   - Expected: Project details preloaded for smoother navigation
   - Verification: Cache populated, faster subsequent loads

### Expected Results
- **Keyboard Shortcuts**:
  - Ctrl+F: Focus search field
  - Ctrl+R: Refresh current view
  - Ctrl+N: Create new project
  - Esc: Return to master view
  - Alt+Left/Right: Navigate between columns
  - Ctrl+Up/Down: Navigate list items

- **Navigation Behavior**:
  - Arrow keys navigate within components
  - Tab key follows logical reading order
  - Enter/Space activate focused elements
  - Focus visible at all times
  - No focus traps or dead ends

- **Accessibility Features**:
  - Screen reader announcements
  - Focus indicators clearly visible
  - Keyboard alternatives for all mouse actions
  - Preloading improves navigation speed

### Error Scenarios
1. **Focus Loss**: Restore focus to last valid element
2. **Invalid Navigation**: Ignore invalid key combinations
3. **Preload Failure**: Continue navigation, cache miss acceptable
4. **Event Conflicts**: Prevent default browser shortcuts when appropriate

### Post-Execution Tasks
- **Focus Verification**: All interactive elements keyboard accessible
- **Performance**: Keyboard navigation responsive < 200ms
- **Memory**: Event listeners properly cleaned up
- **Accessibility**: Screen reader compatibility verified

### Test Data Management
- **Focus History**: Track focus changes for debugging
- **Cache Management**: _projectCache size monitoring
- **Event Cleanup**: jQuery document event removal
- **Performance Metrics**: Navigation timing measurements

### Dependencies
- **jQuery**: Event handling and DOM manipulation
- **SAP UI5**: Focus management and accessibility
- **Browser**: Keyboard event support
- **Screen Readers**: ARIA attribute support

### Performance Criteria
- **Key Response**: < 50ms for key event handling
- **Focus Movement**: < 100ms visual focus change
- **Preload Request**: < 200ms cache population
- **Navigation Flow**: < 150ms between selections

### Accessibility Standards
- **WCAG 2.1 Level AA**: Keyboard accessibility compliance
- **SAP Fiori**: Keyboard shortcuts follow SAP standards
- **Focus Management**: Visible focus indicators
- **Screen Reader**: Proper ARIA labels and descriptions

### Integration Testing
- **Multi-Browser**: Chrome, Firefox, Safari, Edge keyboard handling
- **Operating Systems**: Windows, macOS key modifier support
- **Screen Readers**: NVDA, JAWS, VoiceOver compatibility
- **Mobile**: Touch device keyboard navigation

---

## Test Case ID: TC-UI-AGT-062
**Test Objective**: Verify comprehensive loading states management across all operations and user interactions  
**Business Process**: Loading State Management and User Feedback  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-062
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, User Experience
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:56-66, 120-444`
- **View File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:37-40, 53-57, 142-173`
- **Functions Under Test**: `_loadProjects()`, `_loadProjectDetails()`, `onSearch()`, `onRefresh()`, `_deployProject()`

### Test Preconditions
1. **Authentication**: Valid user session with project access
2. **Network**: Variable network conditions for testing
3. **Projects**: At least 10 projects for comprehensive testing
4. **Browser**: Support for CSS animations and progress indicators
5. **Initial State**: Master-detail view with no active operations

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Loading States | masterLoading, detailLoading, operationLoading, searchLoading, deployLoading, refreshLoading | Array | Loading Model |
| Progress Values | 0-100% | Number | Operation Progress |
| Timeout Values | 10s, 30s | Number | Network Timeouts |
| Search Delays | 200ms | Number | Search Debounce |
| Operation Delays | 500ms-2000ms | Number | Simulated Processing |

### Test Procedure Steps
1. **Step 1 - Initial Master Loading**
   - Action: Navigate to master-detail view
   - Expected: Master loading indicator appears, projects load
   - Verification: masterLoading true->false, busy indicator visible->hidden

2. **Step 2 - Detail Loading State**
   - Action: Select project from master list
   - Expected: Detail loading indicator shown, content fades out->in
   - Verification: detailLoading true->false, fade animation smooth

3. **Step 3 - Search Loading State**
   - Action: Type search query in search field
   - Expected: Search field disabled, loading indicator appears
   - Verification: searchLoading true->false, search results updated

4. **Step 4 - Refresh Loading State**
   - Action: Click refresh button
   - Expected: Button disabled, loading indicator in header
   - Verification: refreshLoading true->false, button re-enabled

5. **Step 5 - Deployment Progress**
   - Action: Click deploy project button
   - Expected: Progress indicator shows 0->100%, deploy button disabled
   - Verification: deployLoading true->false, progress updates incrementally

6. **Step 6 - Operation Loading with Progress**
   - Action: Trigger long-running operation
   - Expected: Full-screen progress indicator with percentage
   - Verification: operationProgress 0->100%, operation text updates

7. **Step 7 - Concurrent Loading States**
   - Action: Trigger multiple operations simultaneously
   - Expected: Each loading state independent, no conflicts
   - Verification: Multiple loading indicators can coexist

8. **Step 8 - Loading Timeout Handling**
   - Action: Simulate network timeout during load
   - Expected: Loading state cleared, error message shown
   - Verification: Loading indicators removed, error handling graceful

9. **Step 9 - Loading State Persistence**
   - Action: Navigate away and back during loading
   - Expected: Loading states maintained correctly
   - Verification: Loading indicators reflect actual operation status

### Expected Results
- **Visual Indicators**:
  - BusyIndicator components for short operations
  - ProgressIndicator for long operations with progress
  - Button disabled states during operations
  - Fade animations for content updates
  
- **Loading State Management**:
  - masterLoading: Project list loading
  - detailLoading: Project detail loading
  - searchLoading: Search operation
  - refreshLoading: Refresh operation
  - deployLoading: Deployment operation
  - operationLoading: General operations

- **Progress Tracking**:
  - operationProgress: 0-100% for long operations
  - currentOperation: Descriptive text for user
  - Progress updates at reasonable intervals
  - Completion feedback with success/error states

### Error Scenarios
1. **Timeout Handling**: Clear loading states, show error message
2. **Network Failure**: Reset loading indicators, maintain UI stability
3. **Rapid Operations**: Prevent loading state conflicts
4. **Memory Management**: Clean up loading timers and intervals

### Post-Execution Tasks
- **State Verification**: All loading properties correctly reset
- **UI Consistency**: No orphaned loading indicators
- **Performance**: Loading animations smooth, no janky updates
- **Memory**: Progress intervals properly cleared

### Test Data Management
- **Loading Model**: loading>/[state] properties
- **Progress Tracking**: operationProgress percentage
- **Operation Text**: currentOperation descriptive strings
- **Timeout Handling**: Ajax timeout configurations

### Dependencies
- **SAP UI5**: BusyIndicator, ProgressIndicator controls
- **jQuery**: Ajax timeout handling
- **CSS**: Fade transition animations
- **Browser**: Animation frame support

### Performance Criteria
- **Loading Indicator Delay**: < 100ms to show loading state
- **Progress Updates**: Every 300ms for smooth progression
- **State Transitions**: < 50ms for state model updates
- **Animation Duration**: 300ms for fade effects

### Accessibility Considerations
- **Screen Reader**: Loading state announcements
- **Focus Management**: Focus preserved during loading
- **Color Contrast**: Loading indicators clearly visible
- **Text Alternatives**: Progress percentage as text

### Integration Testing
- **Model Synchronization**: Loading states reflect actual operations
- **Animation Coordination**: Multiple loading indicators work together
- **Error Recovery**: Failed operations properly reset loading states
- **Performance Impact**: Loading indicators don't degrade performance

---

## Test Case ID: TC-UI-AGT-063
**Test Objective**: Verify comprehensive error handling system with categorization, tracking, and recovery options  
**Business Process**: Error Management and System Reliability  
**SAP Module**: A2A Agents Developer Portal - Project Management Interface  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-063
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Error Handling, System Reliability
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectMasterDetail.controller.js:_initializeErrorHandling()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectMasterDetail.view.xml:42-50`
- **Fragment**: `a2a_agents/backend/app/a2a/developerPortal/static/fragment/ErrorDialog.fragment.xml`
- **Functions Under Test**: `_handleAjaxError()`, `_categorizeError()`, `onShowErrorDetails()`, `onRetryError()`

### Test Preconditions
1. **Error Tracking System**: Error handling initialized with history tracking
2. **Network Conditions**: Ability to simulate network failures and timeouts
3. **Backend Services**: Test endpoints that can return various error conditions
4. **Error Dialog**: Fragment loaded and properly bound to error model
5. **User Interface**: Error indicator visible in master page header

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Network Error | Connection timeout | Object | Network simulation |
| Server Error | 500 Internal Server Error | Object | Backend service |
| Validation Error | Invalid project data | Object | Form validation |
| API Timeout | 30 second timeout | Number | Configuration |
| Retry Attempts | Maximum 3 retries | Number | Error handling config |

### Test Procedure Steps
1. **Step 1 - Error System Initialization**
   - Action: Load project master-detail view
   - Expected: Error handling system initializes with empty error history
   - Verification: Error model shows hasErrors: false, errorCount: 0

2. **Step 2 - Network Error Simulation**
   - Action: Disconnect network and trigger project refresh
   - Expected: Network error caught and categorized as "network"
   - Verification: Error indicator appears with count 1, error stored in history

3. **Step 3 - Server Error Simulation** 
   - Action: Trigger API call that returns HTTP 500 error
   - Expected: Server error categorized and added to error history
   - Verification: Error count increments, error type marked as "server"

4. **Step 4 - Validation Error Testing**
   - Action: Submit form with invalid data
   - Expected: Validation error caught and properly categorized
   - Verification: Error appears as "validation" type with appropriate message

5. **Step 5 - Error Dialog Display**
   - Action: Click error indicator button in header
   - Expected: Error dialog opens showing error statistics and history
   - Verification: Dialog displays total errors, categorized counts, error list

6. **Step 6 - Error Details Verification**
   - Action: Expand error details panel in dialog
   - Expected: Detailed error information displayed with context
   - Verification: Error details show timestamp, ID, browser info, URL

7. **Step 7 - Error Retry Functionality**
   - Action: Click retry button for a failed operation
   - Expected: Original operation retries automatically
   - Verification: Loading state activates, operation executes again

8. **Step 8 - Error Reporting Feature**
   - Action: Click report button for critical error
   - Expected: Error reporting dialog opens with pre-filled information
   - Verification: Error details, browser info, and context captured for reporting

9. **Step 9 - Error History Management**
   - Action: Accumulate multiple errors then clear all
   - Expected: All errors removed from history and UI updated
   - Verification: Error count resets to 0, error indicator hidden

10. **Step 10 - Timeout Error Handling**
    - Action: Trigger operation that exceeds timeout threshold
    - Expected: Timeout error caught and categorized appropriately
    - Verification: Error appears as "timeout" with relevant context

### Expected Results
- **Error Categorization Criteria**:
  - Network errors detected for connectivity issues
  - Server errors identified for HTTP 4xx/5xx responses
  - Validation errors caught for client-side data validation
  - Timeout errors recognized for long-running operations

- **Error Tracking Criteria**:
  - Error history maintains chronological order
  - Error statistics calculated correctly by category
  - Error context captured (timestamp, URL, operation)
  - Error indicators update in real-time

- **Error Recovery Criteria**:
  - Retry functionality works for recoverable errors
  - Error reporting collects comprehensive diagnostic information
  - Error clearing removes all traces from UI and models
  - System remains stable after error recovery

### Technical Implementation Details

#### Error Model Structure
```javascript
{
    hasErrors: boolean,
    errorCount: number,
    lastError: object,
    networkError: boolean,
    serverError: boolean,
    timeoutError: boolean,
    validationError: boolean
}
```

#### Error History Object
```javascript
{
    id: string,
    timestamp: Date,
    message: string,
    category: 'network'|'server'|'validation'|'timeout',
    context: {
        operation: string,
        url: string,
        status: number,
        statusText: string,
        responseData: object
    },
    retryCount: number,
    resolved: boolean
}
```

### Error Categories and Handling

#### Network Errors
- **Detection**: Connection failures, DNS resolution errors
- **UI Indicator**: Orange warning icon with network symbol
- **Recovery**: Automatic retry after network connectivity restored
- **User Action**: Manual retry button, network diagnostics

#### Server Errors  
- **Detection**: HTTP status codes 500-599
- **UI Indicator**: Red error icon with server symbol
- **Recovery**: Limited retry with exponential backoff
- **User Action**: Report error, contact support option

#### Validation Errors
- **Detection**: Client-side validation failures
- **UI Indicator**: Yellow warning with validation symbol  
- **Recovery**: User corrects input, validation re-runs
- **User Action**: Field highlighting, inline error messages

#### Timeout Errors
- **Detection**: Operations exceeding configured timeout
- **UI Indicator**: Blue information with clock symbol
- **Recovery**: Operation cancellation, retry with extended timeout
- **User Action**: Retry operation, adjust timeout settings

### Dialog Components

#### Error Statistics Panel
- Total errors count with state-based styling
- Category breakdown (Network, Server, Validation, Timeout)
- Visual indicators using ObjectNumber controls
- Summary of error frequency and patterns

#### Error History Table
- Chronological list of all errors with growing table
- Sortable by timestamp, category, operation
- Expandable rows showing full error details
- Action buttons for retry and reporting

#### Error Details Expansion
- Complete error context and diagnostic information
- Browser environment details (User Agent, URL)
- Operation-specific context and parameters
- JSON-formatted technical details for developers

### Validation Points
- Error indicator presence: Button visible when errors exist
- Dialog integration: Fragment loads with proper error model binding
- Categorization accuracy: Errors classified correctly by type
- History persistence: Error tracking maintained during session
- Recovery mechanisms: Retry and reporting functions work properly

### Test Data Requirements
- Network connectivity scenarios (online, offline, intermittent)
- Backend endpoints configured to return specific error codes
- Form validation rules for testing validation errors
- Timeout configurations for testing long operations
- Error templates for consistent error messaging

### Performance Testing
- **Error Processing**: Error handling completes within 100ms
- **Dialog Loading**: Error dialog opens within 500ms
- **History Management**: Support for 100+ error entries
- **Memory Usage**: No memory leaks from error tracking
- **UI Responsiveness**: Error indicators don't block interface

### Browser Compatibility
- **Chrome/Edge**: Full error handling with proper error object support
- **Firefox**: Error categorization with Firefox-specific error handling
- **Safari**: WebKit error handling with iOS considerations
- **Mobile Browsers**: Touch-optimized error dialog and indicators

### Security Considerations
- **Error Sanitization**: Sensitive data excluded from error messages
- **Information Disclosure**: Error details don't expose system internals
- **Audit Logging**: Error occurrences logged for security monitoring  
- **Rate Limiting**: Error reporting prevented from abuse
- **Data Privacy**: User data protected in error context

### Integration Testing
- **Model Synchronization**: Error states reflect actual system state
- **Animation Coordination**: Error indicators work with loading states
- **Event Propagation**: Error handling doesn't interfere with normal operations
- **Resource Management**: Error system releases resources properly

### Accessibility Features
- **Screen Reader Support**: Error messages announced to assistive technology
- **Keyboard Navigation**: Error dialog fully navigable via keyboard
- **High Contrast**: Error indicators visible in high contrast modes
- **Focus Management**: Focus properly managed when errors occur
- **Text Alternatives**: Error icons have descriptive alt text

---

## Test Case ID: TC-UI-AGT-064
**Test Objective**: Verify anchor bar navigation functionality with section highlighting and smooth scrolling  
**Business Process**: Object Page Section Navigation  
**SAP Module**: A2A Agents Developer Portal - Project Object Page  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-064
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Navigation, UX Interaction
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:_initializeAnchorBar()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:10-20`
- **Functions Under Test**: `_navigateToSection()`, `_updateCurrentSection()`, `onAnchorBarPress()`, `_smoothScrollToSection()`

### Test Preconditions
1. **Project Data Loaded**: Project object page displayed with all sections populated
2. **Anchor Bar Visible**: Navigation bar showing all available sections
3. **Section Content**: Each section contains representative data (agents, workflows, members, etc.)
4. **Scroll Container**: Object page scroll container properly initialized
5. **Browser Support**: Modern browser with smooth scroll and animation support

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project ID | PROJ_001 | String | Project model |
| Section Count | 6 sections | Number | Layout configuration |
| Scroll Duration | 500ms | Number | Animation config |
| Scroll Offset | 60px | Number | Header offset |
| Keyboard Shortcuts | Ctrl+1-6, Alt+A | String | Key binding config |

### Test Procedure Steps
1. **Step 1 - Anchor Bar Display Verification**
   - Action: Load project object page for valid project
   - Expected: Anchor bar displays with 6 navigation items (General, Agents, Workflows, Team, Metrics, Activity)
   - Verification: Each anchor item shows section title and appropriate icon

2. **Step 2 - Section Badge Updates**
   - Action: Verify badges on anchor items reflect actual data counts
   - Expected: Agents, Workflows, Team, and Activity sections show correct counts
   - Verification: Badge numbers match section content (e.g., Agents (3), Team (3), Activity (3))

3. **Step 3 - Anchor Bar Click Navigation**
   - Action: Click "Agents" anchor bar item
   - Expected: Page smoothly scrolls to Agents section within 500ms
   - Verification: Agents section visible at top of scroll container, anchor item highlighted

4. **Step 4 - Section Highlighting During Scroll**
   - Action: Manually scroll through sections using mouse wheel or touch
   - Expected: Anchor bar highlights current section as user scrolls
   - Verification: Current section indicator updates in real-time during scrolling

5. **Step 5 - Keyboard Navigation Testing**
   - Action: Press Ctrl+3 (Workflows section shortcut)
   - Expected: Page navigates to Workflows section automatically
   - Verification: Workflows section displayed, anchor bar updates highlight

6. **Step 6 - Alt+A Focus Testing**
   - Action: Press Alt+A keyboard shortcut
   - Expected: Focus moves to anchor bar for keyboard navigation
   - Verification: Anchor bar receives keyboard focus, visual focus indicator present

7. **Step 7 - Smooth Scroll Animation**
   - Action: Click distant section (Activity) while viewing General section
   - Expected: Smooth animated scroll with easing function over 500ms
   - Verification: No jarring jumps, smooth transition visible to user

8. **Step 8 - Anchor Bar Toggle Functionality**
   - Action: Click anchor bar toggle button in header actions
   - Expected: Anchor bar visibility toggles on/off
   - Verification: Anchor bar disappears/reappears, toggle state persisted

9. **Step 9 - Section Navigation Edge Cases**
   - Action: Navigate to section with minimal/no content
   - Expected: Navigation completes successfully even for empty sections
   - Verification: Scroll position correct, no JavaScript errors

10. **Step 10 - Multiple Rapid Navigation**
    - Action: Rapidly click different anchor bar items in sequence
    - Expected: Navigation queue handled gracefully, ends at final selection
    - Verification: No animation conflicts, final section displayed correctly

### Expected Results
- **Navigation Accuracy Criteria**:
  - Click navigation scrolls to correct section within 500ms
  - Section highlighting tracks actual scroll position
  - Keyboard shortcuts navigate to corresponding sections
  - Focus management works properly with Alt+A

- **Visual Feedback Criteria**:
  - Current section clearly highlighted in anchor bar
  - Badge counts accurately reflect section content
  - Smooth scroll animation provides good user experience
  - Toggle functionality shows immediate visual feedback

- **Performance Criteria**:
  - Navigation completes within 500ms for all sections
  - Scroll animations are smooth without frame drops
  - Rapid navigation doesn't cause performance degradation
  - Memory usage remains stable during extensive navigation

### Technical Implementation Details

#### Anchor Bar Configuration
```javascript
{
    autoScrollToSection: true,
    highlightCurrentSection: true,
    smoothScrollBehavior: true,
    scrollOffset: 60 // Offset for fixed header
}
```

#### Section Navigation Items
```javascript
[
    { key: "general", title: "General Information", icon: "sap-icon://hint" },
    { key: "agents", title: "Agents", icon: "sap-icon://person-placeholder", badge: 0 },
    { key: "workflows", title: "Workflows", icon: "sap-icon://process" },
    { key: "team", title: "Team Members", icon: "sap-icon://group" },
    { key: "metrics", title: "Metrics & Analytics", icon: "sap-icon://line-chart" },
    { key: "activity", title: "Recent Activity", icon: "sap-icon://history" }
]
```

### Navigation Methods

#### Smooth Scroll Implementation
- **Animation Duration**: 500ms with easeInOutQuad easing
- **Offset Calculation**: Account for fixed header height (60px)
- **Scroll Container**: Target ObjectPageLayout scroll container
- **Fallback**: Standard scrollToSection for unsupported browsers

#### Section Detection Logic
- **Event Listeners**: Navigate and SectionChange events on ObjectPageLayout
- **ID Matching**: Section IDs contain anchor key for identification
- **Current Section Tracking**: Real-time updates during scroll operations
- **Highlight Synchronization**: Anchor bar reflects current viewport section

### Keyboard Accessibility
- **Shortcut Keys**: Ctrl+1-6 for direct section navigation
- **Focus Management**: Alt+A moves focus to anchor bar
- **Tab Navigation**: Standard tab order through anchor items
- **Screen Reader Support**: ARIA labels and announcements

### Browser Compatibility
- **Chrome/Edge**: Full smooth scroll and animation support
- **Firefox**: Smooth scroll with requestAnimationFrame fallback
- **Safari**: WebKit scroll animation with CSS transitions
- **Mobile Browsers**: Touch-optimized scrolling with momentum

### Validation Points
- Anchor bar visibility: Toggle button controls anchor bar display
- Navigation accuracy: Clicks scroll to correct section positions
- Section highlighting: Current section indicator updates during scroll
- Animation quality: Smooth transitions without jarring movements
- Keyboard support: Shortcuts provide alternative navigation method

### Test Data Requirements
- Project with all 6 section types populated with representative data
- Agents list with at least 3 entries for badge count testing
- Workflows with different statuses for visual variety
- Team members with roles and join dates
- Metrics with realistic performance data
- Activity timeline with recent entries

### Performance Testing
- **Navigation Speed**: Section navigation completes within 500ms
- **Scroll Performance**: Smooth animation at 60fps minimum
- **Memory Usage**: No memory leaks during extensive navigation
- **Event Handling**: Rapid clicks handled without conflicts
- **Mobile Performance**: Touch scrolling maintains responsiveness

### Accessibility Features
- **Screen Reader Support**: Section announcements during navigation
- **High Contrast**: Anchor bar visible in high contrast modes  
- **Keyboard Navigation**: Full functionality accessible via keyboard
- **Focus Indicators**: Clear visual focus on anchor bar items
- **ARIA Labels**: Descriptive labels for all interactive elements

### Integration Testing
- **Object Page Layout**: Integration with SAP UI5 ObjectPageLayout
- **Model Synchronization**: Badge updates reflect model changes
- **Event Coordination**: Navigation events don't conflict with other interactions
- **Responsive Behavior**: Anchor bar adapts to different screen sizes

---

## Test Case ID: TC-UI-AGT-065
**Test Objective**: Verify advanced section scrolling functionality with position memory, direction tracking, and programmatic controls  
**Business Process**: Object Page Content Navigation and Scrolling  
**SAP Module**: A2A Agents Developer Portal - Project Object Page  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-065
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interaction, Performance
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:_setupAdvancedScrolling()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:48-55`
- **Functions Under Test**: `_setupScrollSpy()`, `_handleScrollSpyUpdate()`, `onScrollToTop()`, `onScrollByPage()`, `_saveScrollPosition()`

### Test Preconditions
1. **Object Page Loaded**: Project object page with multiple sections and sufficient content height
2. **Scroll Container**: Object page scroll container properly initialized and scrollable
3. **Advanced Scrolling**: Scroll spy, memory, and direction tracking enabled
4. **Keyboard Navigation**: Enhanced keyboard shortcuts configured
5. **Section Content**: Each section has enough content to enable meaningful scrolling

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Scroll Debounce | 100ms | Number | Scroll configuration |
| Scroll Threshold | 50px | Number | Section change threshold |
| Page Scroll Factor | 80% | Number | Page scrolling percentage |
| Animation Duration | 300ms | Number | Smooth scroll timing |
| Memory Positions | 6 sections | Object | Section position storage |

### Test Procedure Steps
1. **Step 1 - Scroll Direction Detection**
   - Action: Scroll down through multiple sections using mouse wheel
   - Expected: Scroll direction tracked and updated in view model as "down"
   - Verification: View model property "/scrollDirection" updates to "down" during downward scrolling

2. **Step 2 - Scroll Position Memory**
   - Action: Navigate to Metrics section, scroll partially down, then navigate to another section
   - Expected: Scroll position for Metrics section saved in memory
   - Verification: Return to Metrics section restores previous scroll position

3. **Step 3 - Scroll Spy Functionality**
   - Action: Manually scroll through content without clicking anchor bar
   - Expected: Anchor bar highlighting updates based on current section in viewport
   - Verification: Current section highlighting changes automatically during scrolling

4. **Step 4 - Scroll to Top Functionality**
   - Action: Scroll to bottom of page, then click "Scroll to Top" button
   - Expected: Smooth animated scroll to top of page within 300ms
   - Verification: Page scrolls to top position (scrollTop = 0) with smooth animation

5. **Step 5 - Ctrl+Home Keyboard Shortcut**
   - Action: Scroll to middle of page, then press Ctrl+Home
   - Expected: Page scrolls to top automatically
   - Verification: Keyboard shortcut triggers scroll to top functionality

6. **Step 6 - Scroll to Bottom Functionality**
   - Action: From top of page, click "Scroll to Bottom" button
   - Expected: Smooth scroll to bottom of content within 300ms
   - Verification: Page scrolls to maximum scroll position with animation

7. **Step 7 - Ctrl+End Keyboard Shortcut**
   - Action: From top of page, press Ctrl+End
   - Expected: Page scrolls to bottom automatically
   - Verification: Keyboard shortcut triggers scroll to bottom functionality

8. **Step 8 - Page Up/Down Scrolling**
   - Action: Press Page Down key multiple times
   - Expected: Each press scrolls down by 80% of viewport height
   - Verification: Consistent scroll distances, smooth transitions between positions

9. **Step 9 - Alt+Up/Down Section Navigation**
   - Action: Navigate to Workflows section, then press Alt+Down
   - Expected: Page navigates to next section (Team Members)
   - Verification: Section navigation moves to adjacent sections in sequence

10. **Step 10 - Scroll Event Debouncing**
    - Action: Rapidly scroll up and down with mouse wheel
    - Expected: Scroll events debounced to prevent performance issues
    - Verification: Section highlighting updates smoothly without excessive event firing

### Expected Results
- **Scroll Tracking Criteria**:
  - Direction detection accurately identifies up/down scrolling
  - Position memory saves and restores scroll positions per section
  - Scroll spy updates anchor bar highlighting in real-time
  - Debouncing prevents excessive event handling during rapid scrolling

- **Programmatic Scroll Criteria**:
  - Scroll to top/bottom functions complete within 300ms
  - Page scrolling moves by consistent 80% viewport distances
  - Section navigation moves to correct adjacent sections
  - All scroll animations are smooth without frame drops

- **Keyboard Navigation Criteria**:
  - Ctrl+Home/End shortcuts work consistently
  - Page Up/Down keys provide expected scroll distances
  - Alt+Up/Down navigate between sections correctly
  - All keyboard shortcuts prevent default browser behavior

### Technical Implementation Details

#### Scroll Configuration Object
```javascript
{
    scrollOffset: 60,           // Header offset in pixels
    scrollThreshold: 50,        // Section change threshold
    scrollDebounce: 100,        // Event debouncing (ms)
    enableScrollMemory: true,   // Position memory feature
    enableScrollSpy: true       // Auto section highlighting
}
```

#### Scroll Position Storage
```javascript
{
    _sectionScrollPositions: {
        "generalSection": 0,
        "agentsSection": 150,
        "workflowsSection": 300,
        "teamSection": 450,
        "metricsSection": 600,
        "activitySection": 750
    }
}
```

### Scrolling Methods

#### Scroll Spy Implementation
- **Event Debouncing**: 100ms debounce prevents excessive scroll event handling
- **Section Detection**: Viewport intersection calculation determines current section
- **Highlight Synchronization**: Anchor bar updates reflect current viewport content
- **Performance Optimization**: Efficient DOM queries minimize impact on scrolling performance

#### Position Memory System
- **Automatic Saving**: Section scroll positions saved when navigating away
- **Restoration Logic**: Previous scroll positions restored when returning to sections
- **Memory Management**: Position storage cleaned up when no longer needed
- **Smooth Restoration**: Animated scroll to saved positions for better UX

### Keyboard Shortcuts

#### Enhanced Navigation Keys
- **Ctrl+Home**: Jump to top of entire page
- **Ctrl+End**: Jump to bottom of entire page
- **Page Up/Down**: Scroll by 80% of viewport height
- **Alt+Up**: Navigate to previous section in sequence
- **Alt+Down**: Navigate to next section in sequence

#### Scroll Direction Tracking
- **Real-time Detection**: Immediate direction updates during scrolling
- **View Model Integration**: Direction available for UI animations and indicators
- **State Management**: Current direction maintained for enhanced user feedback
- **Event Coordination**: Direction changes coordinate with other scroll features

### Performance Optimization
- **Event Debouncing**: Scroll events debounced to 100ms intervals
- **Efficient Calculations**: Minimal DOM queries during scroll operations
- **Animation Performance**: Hardware-accelerated smooth scrolling where possible
- **Memory Management**: Scroll positions cleaned up appropriately

### Browser Compatibility
- **Chrome/Edge**: Full support for smooth scroll behavior and animations
- **Firefox**: Fallback scroll implementation with polyfill animations
- **Safari**: WebKit-specific scroll optimizations and touch handling
- **Mobile Browsers**: Touch scroll integration with momentum preservation

### Validation Points
- Scroll direction tracking: Model updates reflect actual scroll direction
- Position memory: Saved positions accurately restored when returning to sections
- Scroll spy accuracy: Section highlighting matches actual viewport content
- Animation quality: Smooth scrolling without performance degradation
- Keyboard functionality: All shortcuts work reliably across browsers

### Test Data Requirements
- Object page with significant content height requiring scrolling
- Multiple sections with sufficient individual content for position memory testing
- Content variety to test different scroll distances and positions
- Realistic section heights for scroll spy threshold testing

### Performance Testing
- **Scroll Response Time**: Direction detection within 50ms of scroll event
- **Animation Performance**: Smooth scrolling maintains 60fps
- **Memory Efficiency**: No memory leaks from position tracking
- **Event Handling**: Debounced events don't cause UI lag
- **Mobile Performance**: Touch scrolling remains responsive

### Accessibility Features
- **Screen Reader Support**: Scroll position announcements for assistive technology
- **Keyboard Navigation**: Full scroll control accessible via keyboard only
- **Focus Management**: Focus remains visible during programmatic scrolling
- **High Contrast**: Scroll indicators visible in high contrast modes
- **Reduced Motion**: Respects user preferences for reduced motion

### Integration Testing
- **Anchor Bar Coordination**: Scroll functionality works seamlessly with anchor navigation
- **Section Navigation**: Scroll position memory integrates with section switching
- **Event Management**: Scroll events don't interfere with other page interactions
- **State Synchronization**: Scroll state remains consistent across different navigation methods

---

## Test Case ID: TC-UI-AGT-066
**Test Objective**: Verify advanced section highlighting with visual animations, intersection detection, and coordinated anchor bar highlighting  
**Business Process**: Visual Section Navigation and User Feedback  
**SAP Module**: A2A Agents Developer Portal - Project Object Page  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-066
- **Test Priority**: High (P2)
- **Test Type**: Functional, Visual, User Experience
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:_initializeSectionHighlighting()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:56-59`
- **Functions Under Test**: `_highlightSection()`, `_setupIntersectionObserver()`, `_applySectionHighlight()`, `onToggleSectionHighlighting()`

### Test Preconditions
1. **Object Page Loaded**: Project object page with all sections rendered and visible
2. **Highlighting System**: Section highlighting initialized with CSS styles and animations
3. **Intersection Observer**: Modern browser with Intersection Observer API support
4. **Visual Styles**: CSS highlighting styles injected and active
5. **Animation Support**: Browser supports CSS transitions and animations

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Animation Duration | 200ms | Number | Highlight configuration |
| Intersection Thresholds | [0.1, 0.5, 0.9] | Array | Observer configuration |
| Root Margin | -60px 0px -60px 0px | String | Header offset adjustment |
| Highlight Color | rgba(0,123,191,0.05) | String | CSS styling |
| Border Color | #007bff | String | CSS styling |

### Test Procedure Steps
1. **Step 1 - Initial Section Highlighting**
   - Action: Load project object page and navigate to Agents section
   - Expected: Agents section highlighted with blue background and left border
   - Verification: Section has highlight class applied with visual border and background

2. **Step 2 - Highlight Animation on Navigation**
   - Action: Click on Workflows section in anchor bar
   - Expected: Workflows section highlights with 500ms pulse animation
   - Verification: Animation plays from transparent to highlighted state smoothly

3. **Step 3 - Section Title Highlighting**
   - Action: Navigate to Team Members section
   - Expected: Section title color changes to blue (#007bff) with font-weight 600
   - Verification: Section title styling updates coordinate with section highlighting

4. **Step 4 - Intersection Observer Detection**
   - Action: Manually scroll through sections without clicking anchor bar
   - Expected: Section highlighting updates automatically based on viewport intersection
   - Verification: Current section highlights change as sections enter/leave viewport

5. **Step 5 - Multiple Threshold Detection**
   - Action: Slowly scroll so section is partially visible (10%, 50%, 90% thresholds)
   - Expected: Section highlighting activates when section becomes most visible
   - Verification: Most visible section (highest intersection ratio) becomes highlighted

6. **Step 6 - Highlight Toggle Functionality**
   - Action: Click "Toggle Section Highlighting" button in header
   - Expected: All section highlighting disabled, toggle button changes to Transparent
   - Verification: No sections show highlighting, button visual state updates

7. **Step 7 - Re-enable Highlighting**
   - Action: Click highlighting toggle button again
   - Expected: Section highlighting re-enabled, current section highlighted immediately
   - Verification: Toggle button becomes Emphasized, current section highlights

8. **Step 8 - Anchor Bar Coordination**
   - Action: Navigate to Metrics section via anchor bar click
   - Expected: Both section and corresponding anchor bar button highlighted
   - Verification: Anchor button scales (1.05x), changes to blue background with white text

9. **Step 9 - Highlight Cleanup**
   - Action: Navigate between multiple sections rapidly
   - Expected: Previous section highlighting removed cleanly before new highlighting
   - Verification: No multiple sections highlighted simultaneously, smooth transitions

10. **Step 10 - CSS Animation Performance**
    - Action: Trigger highlighting animations on different sections repeatedly
    - Expected: Smooth 200ms transitions without performance degradation
    - Verification: Animations complete within specified duration, no frame drops

### Expected Results
- **Visual Highlighting Criteria**:
  - Sections display blue background (rgba(0,123,191,0.05)) with left border
  - Section titles change to blue color with increased font weight
  - Highlight transitions smooth with 200ms CSS animations
  - Anchor bar buttons coordinate highlighting with section state

- **Intersection Detection Criteria**:
  - Most visible section automatically highlighted during scrolling
  - Multiple threshold detection (0.1, 0.5, 0.9) provides accurate section tracking
  - Root margin adjustment (-60px) accounts for fixed header correctly
  - Section highlighting updates in real-time during scroll operations

- **Animation Performance Criteria**:
  - Pulse animation (500ms) plays smoothly when navigating to sections
  - Transition animations (200ms) provide smooth highlight state changes
  - No visual glitches or frame drops during highlighting animations
  - Animation cleanup prevents overlapping or conflicting animations

### Technical Implementation Details

#### Highlighting Configuration
```javascript
{
    enabled: true,
    animationDuration: 200,
    highlightClass: "a2a-section-highlighted",
    fadeClass: "a2a-section-fade",
    currentHighlighted: null
}
```

#### CSS Highlight Styles
```css
.a2a-section-highlighted {
    background-color: rgba(0, 123, 191, 0.05) !important;
    border-left: 4px solid #007bff !important;
    transition: all 200ms ease-in-out !important;
    box-shadow: 0 2px 8px rgba(0, 123, 191, 0.15) !important;
}

.a2a-section-highlight-animation {
    animation: sectionHighlight 500ms ease-in-out;
}
```

### Intersection Observer Configuration

#### Observer Settings
- **Root Margin**: `-60px 0px -60px 0px` to account for fixed header
- **Thresholds**: `[0.1, 0.5, 0.9]` for granular visibility detection
- **Root Element**: `null` (viewport) for accurate section tracking
- **Target Elements**: All ObjectPageSection DOM elements

#### Visibility Calculation
- **Maximum Intersection Ratio**: Section with highest visibility becomes highlighted
- **Real-time Updates**: Highlighting changes immediately during scroll operations
- **Performance Optimization**: Efficient DOM queries minimize impact on scroll performance
- **Browser Fallback**: Graceful degradation for browsers without Intersection Observer

### Animation System

#### Highlight Transitions
- **Fade In**: Smooth transition to highlighted state (200ms)
- **Fade Out**: Smooth transition from highlighted state (200ms)
- **Pulse Animation**: Initial highlight animation when navigating (500ms)
- **Anchor Coordination**: Anchor bar highlighting with scale transform (150ms)

#### Animation Classes
- **Static Highlight**: `a2a-section-highlighted` for persistent highlighting
- **Animation Pulse**: `a2a-section-highlight-animation` for navigation animation
- **Anchor Highlight**: `a2a-anchor-item-highlighted` for anchor bar coordination
- **Title Highlight**: `a2a-section-title-highlighted` for section title styling

### Visual Design Elements

#### Section Highlighting
- **Background Color**: Subtle blue tint (rgba(0,123,191,0.05))
- **Left Border**: Prominent blue border (4px solid #007bff)
- **Box Shadow**: Subtle elevation (0 2px 8px rgba(0,123,191,0.15))
- **Transition**: Smooth 200ms ease-in-out for all properties

#### Title Enhancement
- **Color Change**: Blue (#007bff) for highlighted section titles
- **Font Weight**: Bold (600) for visual emphasis
- **Transition**: Smooth color change (200ms ease-in-out)
- **Typography**: Maintains existing font family and size

### Browser Compatibility
- **Chrome/Edge**: Full support for Intersection Observer and CSS animations
- **Firefox**: Complete feature support with native API implementation
- **Safari**: WebKit Intersection Observer with CSS transition support
- **Mobile Browsers**: Touch-optimized highlighting with momentum scrolling integration
- **Legacy Fallback**: Graceful degradation to scroll-based highlighting

### Validation Points
- Highlighting toggle: Button controls highlighting system enable/disable state
- Visual consistency: Highlight styling matches design specifications
- Animation quality: Smooth transitions without visual artifacts
- Intersection accuracy: Most visible section correctly identified and highlighted
- Performance impact: Highlighting system doesn't affect scroll performance

### Test Data Requirements
- Object page with sufficient section content for intersection testing
- Multiple sections with varying heights for visibility threshold testing
- Realistic content amounts for performance evaluation
- Different section types for visual consistency verification

### Performance Testing
- **Animation Performance**: Highlighting animations maintain 60fps
- **Scroll Performance**: Intersection detection doesn't impact scroll smoothness
- **Memory Usage**: No memory leaks from observer or animation cleanup
- **CPU Usage**: Highlighting calculations remain efficient during rapid scrolling
- **Mobile Performance**: Touch scrolling remains responsive with highlighting active

### Accessibility Features
- **Screen Reader Support**: Highlighted sections announced to assistive technology
- **High Contrast**: Highlighting visible in high contrast modes with alternative styling
- **Focus Management**: Keyboard focus coordination with section highlighting
- **Reduced Motion**: Respects user preferences for reduced motion in animations
- **Color Blind Support**: Highlighting relies on border and shadow, not just color

### Integration Testing
- **Anchor Bar Coordination**: Highlighting synchronizes with anchor bar navigation
- **Scroll Integration**: Works seamlessly with scroll spy and position memory
- **Navigation Coordination**: Highlighting integrates with all navigation methods
- **State Management**: Highlighting state persists correctly across interactions

### Error Handling
- **API Fallback**: Graceful degradation when Intersection Observer unavailable
- **CSS Injection**: Safe CSS injection with collision prevention
- **Animation Cleanup**: Proper cleanup prevents animation artifacts
- **Memory Management**: Observer disconnection on component destruction

---

## Test Case ID: TC-UI-AGT-067
**Test Objective**: Verify section expand/collapse functionality with smooth animations, state persistence, and keyboard shortcuts  
**Business Process**: Content Management and Space Optimization  
**SAP Module**: A2A Agents Developer Portal - Project Object Page  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-067
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interaction, State Management
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:_initializeSectionCollapse()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:60-67`
- **Functions Under Test**: `_toggleSectionCollapse()`, `_collapseSection()`, `_expandSection()`, `onCollapseAllSections()`

### Test Preconditions
1. **Object Page Loaded**: Project object page with all sections rendered and populated
2. **Collapse System**: Section collapse functionality initialized with CSS animations
3. **State Persistence**: LocalStorage available for saving collapsed section states
4. **Section Controls**: Collapse icons and click handlers added to all section headers
5. **Keyboard Support**: Collapse keyboard shortcuts configured and active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Animation Duration | 300ms | Number | Collapse configuration |
| Stagger Delay | 100ms | Number | Batch operation timing |
| Storage Key | a2a-collapsed-sections-{projectId} | String | State persistence |
| Icon Rotation | -90deg | String | Collapsed state styling |
| Opacity Transition | 0.7 | Number | Header fade effect |

### Test Procedure Steps
1. **Step 1 - Section Collapse Controls**
   - Action: Load project object page and examine section headers
   - Expected: Each section header has collapse icon and clickable title container
   - Verification: Down arrow icons visible, hover effects active on section titles

2. **Step 2 - Individual Section Collapse**
   - Action: Click on Agents section title to collapse
   - Expected: Section content collapses with 300ms smooth animation, icon rotates -90deg
   - Verification: Section content max-height transitions to 0, opacity fades to 0

3. **Step 3 - Section Expand Animation**
   - Action: Click collapsed Agents section title to expand
   - Expected: Section content expands to natural height with smooth animation
   - Verification: Content becomes visible, icon rotates back, opacity returns to 1

4. **Step 4 - Section Badge Display**
   - Action: Verify badges on collapsible sections
   - Expected: Agents, Workflows, Team, Activity sections show content count badges
   - Verification: Badge numbers match actual content (e.g., "3" for agents count)

5. **Step 5 - Header State Changes**
   - Action: Collapse Team Members section
   - Expected: Section header opacity reduces to 0.7, title container shows collapsed state
   - Verification: Visual indication that section is collapsed without losing readability

6. **Step 6 - Keyboard Collapse Current Section**
   - Action: Navigate to Workflows section, press Ctrl+[
   - Expected: Current section (Workflows) collapses via keyboard shortcut
   - Verification: Workflows section content collapses, keyboard shortcut works

7. **Step 7 - Keyboard Expand Current Section**
   - Action: While on collapsed Workflows section, press Ctrl+]
   - Expected: Current section expands via keyboard shortcut
   - Verification: Workflows section content expands, smooth transition

8. **Step 8 - Collapse All Sections**
   - Action: Press Ctrl+Shift+C or click "Collapse All" button
   - Expected: All sections collapse with staggered animation (100ms delay between sections)
   - Verification: Sections collapse sequentially, visual wave effect

9. **Step 9 - Expand All Sections**
   - Action: Press Ctrl+Shift+E or click "Expand All" button
   - Expected: All sections expand with staggered animation
   - Verification: Sections expand sequentially in same order

10. **Step 10 - State Persistence**
    - Action: Collapse multiple sections, refresh page
    - Expected: Page loads with previously collapsed sections remaining collapsed
    - Verification: LocalStorage contains collapsed section IDs, state restored correctly

### Expected Results
- **Animation Performance Criteria**:
  - Section collapse/expand animations complete within 300ms
  - Smooth transitions without visual glitches or frame drops
  - Icon rotation animations (200ms) coordinate with content animations
  - Staggered animations provide pleasing visual sequence

- **State Management Criteria**:
  - Individual section states tracked accurately in view model
  - Collapsed section IDs persisted to localStorage per project
  - State restoration works correctly on page refresh/navigation
  - Multiple sections can be collapsed simultaneously

- **User Interaction Criteria**:
  - Click handlers work consistently on all section titles
  - Hover effects provide visual feedback on interactive elements
  - Keyboard shortcuts work reliably for current and all sections
  - Badge counts update dynamically and display correctly

### Technical Implementation Details

#### Collapse Configuration
```javascript
{
    enabled: true,
    animationDuration: 300,
    collapsedSections: [],
    persistState: true,
    allowMultipleExpanded: true
}
```

#### CSS Animation Classes
```css
.a2a-section-collapsed .sapUxAPObjectPageSectionContent {
    max-height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    transition: all 300ms ease-in-out !important;
}

.a2a-collapse-icon {
    transition: transform 200ms ease-in-out !important;
}

.a2a-collapse-icon.collapsed {
    transform: rotate(-90deg) !important;
}
```

### Animation System

#### Collapse Animation Sequence
1. **Height Measurement**: Calculate current content height for smooth transition
2. **Animation Start**: Set explicit max-height to current height
3. **Transition Trigger**: Change max-height to 0px after 10ms delay
4. **Icon Rotation**: Rotate collapse icon -90deg simultaneously
5. **Header Fade**: Reduce header opacity to 0.7 for collapsed state
6. **Cleanup**: Remove animation classes after completion

#### Expand Animation Sequence
1. **Natural Height**: Calculate natural expanded height for target
2. **Animation Start**: Begin with max-height: 0px
3. **Transition Trigger**: Change max-height to natural height
4. **Icon Reset**: Rotate collapse icon back to 0deg
5. **Header Restore**: Return header opacity to 1.0
6. **Completion**: Set max-height: none for responsive content

### State Persistence System

#### LocalStorage Integration
- **Storage Key**: `a2a-collapsed-sections-{projectId}` for project-specific state
- **Data Format**: JSON array of collapsed section IDs
- **Error Handling**: Graceful degradation when localStorage unavailable
- **Cleanup**: Automatic cleanup of invalid section IDs

#### State Restoration Logic
- **Load Time**: Restore collapsed states after view rendering
- **Validation**: Verify section IDs exist before applying collapsed state
- **Animation**: Apply collapsed state without animation during restoration
- **Synchronization**: Update view model to match restored state

### Keyboard Shortcuts

#### Individual Section Controls
- **Ctrl+[**: Collapse current section (based on scroll position)
- **Ctrl+]**: Expand current section (based on scroll position)
- **Current Section Detection**: Use scroll spy to identify active section

#### Batch Operations
- **Ctrl+Shift+C**: Collapse all sections with staggered animation
- **Ctrl+Shift+E**: Expand all sections with staggered animation
- **Animation Timing**: 100ms delay between sections for visual effect

### Visual Design Elements

#### Section Title Enhancement
- **Flex Container**: Title, icon, and badge in flexbox layout
- **Hover Effects**: Subtle background color change on title hover
- **Click Target**: Entire title container clickable for better UX
- **Visual Feedback**: Cursor pointer and hover states for interactivity

#### Badge System
- **Dynamic Counts**: Real-time update of content counts in badges
- **Visual Design**: Blue background, white text, rounded corners
- **Positioning**: Right-aligned within title container
- **Responsive**: Adapts to different content lengths

### Browser Compatibility
- **Chrome/Edge**: Full CSS transition and transform support
- **Firefox**: Complete animation support with vendor prefixes
- **Safari**: WebKit transitions with transform animations
- **Mobile Browsers**: Touch-optimized collapse controls with gesture support
- **Legacy Fallback**: Instant collapse/expand for unsupported browsers

### Validation Points
- Animation quality: Smooth 300ms transitions without visual artifacts
- State persistence: Collapsed sections restore correctly on page reload
- Keyboard functionality: All shortcuts work consistently across browsers
- Performance impact: Animations don't affect overall page performance
- Badge accuracy: Content counts reflect actual data in sections

### Test Data Requirements
- Project with all section types populated for badge testing
- Sufficient content in each section to make collapse/expand meaningful
- Multiple projects for testing project-specific state persistence
- Various content amounts for performance testing of animations

### Performance Testing
- **Animation Performance**: Collapse/expand animations maintain 60fps
- **Batch Operations**: Staggered animations don't cause performance degradation
- **Memory Usage**: No memory leaks from animation event handlers
- **Storage Performance**: LocalStorage operations don't block UI
- **Mobile Performance**: Touch interactions remain responsive

### Accessibility Features
- **Screen Reader Support**: Collapse state announced to assistive technology
- **Keyboard Navigation**: Full functionality accessible via keyboard only
- **Focus Management**: Focus remains visible during collapse/expand operations
- **ARIA States**: Collapsed/expanded states communicated via ARIA attributes
- **High Contrast**: Collapse controls visible in high contrast modes

### Integration Testing
- **Highlight Coordination**: Section highlighting works with collapsed sections
- **Scroll Integration**: Scroll spy functions correctly with collapsed content
- **Anchor Navigation**: Anchor bar works with collapsed sections
- **State Coordination**: All navigation methods respect collapsed states

### Error Handling
- **Storage Errors**: Graceful handling of localStorage quota exceeded
- **Animation Conflicts**: Prevention of conflicting expand/collapse operations
- **DOM Availability**: Safe DOM manipulation with existence checks
- **Invalid States**: Recovery from inconsistent section states

---

## Test Case ID: TC-UI-AGT-068
**Test Objective**: Verify section lazy loading with intersection detection, placeholder animations, error handling, and performance optimization  
**Business Process**: Performance Optimization and Progressive Content Loading  
**SAP Module**: A2A Agents Developer Portal - Project Object Page  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-068
- **Test Priority**: High (P2)
- **Test Type**: Functional, Performance, User Experience
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:_initializeLazyLoading()`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:68-71`
- **Functions Under Test**: `_loadSectionContent()`, `_setupLazyLoadingObserver()`, `_createLazyPlaceholder()`, `_fetchSectionData()`

### Test Preconditions
1. **Object Page Loaded**: Project object page with multiple sections requiring lazy loading
2. **Intersection Observer**: Modern browser with Intersection Observer API support
3. **Loading System**: Lazy loading initialized with placeholder styling and metrics tracking
4. **Network Simulation**: Ability to simulate different loading times and failures
5. **Performance Monitoring**: Metrics tracking system active for load time measurement

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Intersection Threshold | 0.1 | Number | Observer configuration |
| Preload Distance | 200px | Number | Root margin setting |
| Loading Timeout | 5000ms | Number | Maximum load time |
| Failure Rate | 10% | Number | Error simulation |
| Section Load Times | 300-1500ms | Range | Performance simulation |

### Test Procedure Steps
1. **Step 1 - Initial Load State**
   - Action: Load project object page
   - Expected: First section loads immediately, other sections show placeholder content
   - Verification: Only General Information section fully loaded, others show spinning loading indicators

2. **Step 2 - Placeholder Content Display**
   - Action: Examine placeholder content in unloaded sections
   - Expected: Skeleton screens with loading spinners and section-appropriate placeholders
   - Verification: Each section shows "Loading [Section Name]..." with animated skeleton elements

3. **Step 3 - Intersection-Based Loading**
   - Action: Scroll down to bring Agents section into viewport
   - Expected: Agents section triggers loading when 10% visible, content loads after 800ms
   - Verification: Loading spinner appears, then content fades in with animation

4. **Step 4 - Preload Distance Testing**
   - Action: Scroll to 200px above Workflows section
   - Expected: Workflows section begins loading before fully visible
   - Verification: Loading starts when section is 200px away from viewport

5. **Step 5 - Different Section Load Times**
   - Action: Navigate through all sections observing load behavior
   - Expected: Sections load at different speeds (Metrics: 1500ms, Team: 400ms, etc.)
   - Verification: Load times match expected durations, heavier sections take longer

6. **Step 6 - Loading Error Handling**
   - Action: Trigger section loading that fails (simulated 10% failure rate)
   - Expected: Error placeholder displays with retry button
   - Verification: Red error state with "Failed to Load Content" message and retry option

7. **Step 7 - Error Recovery**
   - Action: Click "Retry Loading" button on failed section
   - Expected: Section resets to loading state and attempts to load again
   - Verification: Error clears, loading spinner returns, successful load on retry

8. **Step 8 - Content Fade-In Animation**
   - Action: Observe content loading animations
   - Expected: Smooth fade-in with 500ms duration and upward slide (20px translateY)
   - Verification: Content appears with lazyFadeIn animation class

9. **Step 9 - Lazy Loading Toggle**
   - Action: Click "Toggle Lazy Loading" button in header
   - Expected: All unloaded sections load immediately when lazy loading disabled
   - Verification: All placeholder content replaced with actual content, no loading states

10. **Step 10 - Performance Metrics**
    - Action: Check lazy loading performance information
    - Expected: Metrics track successful loads, error counts, and section-specific load times
    - Verification: getLazyLoadingInfo() returns accurate metrics data

### Expected Results
- **Loading Performance Criteria**:
  - First section loads immediately on page load
  - Subsequent sections load only when approaching viewport (200px preload)
  - Section-specific load times reflect content complexity
  - Loading animations provide smooth user feedback

- **Error Handling Criteria**:
  - Loading failures display user-friendly error messages
  - Retry functionality allows recovery from temporary failures
  - Error states visually distinct from loading states
  - Failed sections don't block other section loading

- **Animation Quality Criteria**:
  - Placeholder skeletons provide realistic loading preview
  - Content fade-in animations smooth and performant
  - Loading spinners indicate active loading state
  - No jarring content shifts during loading

### Technical Implementation Details

#### Lazy Loading Configuration
```javascript
{
    enabled: true,
    threshold: 0.1,
    loadedSections: [],
    preloadDistance: 200,
    enablePlaceholders: true,
    loadingTimeout: 5000
}
```

#### Intersection Observer Setup
```javascript
{
    root: null,
    rootMargin: '200px',
    threshold: 0.1
}
```

### Loading System Architecture

#### Section Loading Lifecycle
1. **Initial State**: Section marked for lazy loading with placeholder content
2. **Intersection Detection**: Intersection Observer triggers when section approaches viewport
3. **Loading State**: Placeholder shows loading spinner and skeleton screens
4. **Data Fetching**: Asynchronous data loading with section-specific timing
5. **Content Rendering**: Original content rendered with fade-in animation
6. **Loaded State**: Section marked as loaded, observer stops watching

#### Placeholder System
- **Skeleton Screens**: Animated gradient placeholders matching expected content layout
- **Loading Indicators**: Spinning indicators showing active loading state
- **Section-Specific**: Different placeholder types for tables, grids, lists, charts, timelines
- **Error States**: Distinct error styling with retry functionality

### Performance Optimization

#### Load Time Differentiation
- **Metrics Sections**: 1500ms (complex data processing)
- **Activity Timeline**: 1200ms (historical data loading)
- **Agents Table**: 800ms (structured data display)
- **Workflows Grid**: 600ms (medium complexity)
- **Team List**: 400ms (simple list data)
- **General Content**: 300ms (static information)

#### Observer Efficiency
- **Single Observer**: One Intersection Observer monitors all sections
- **Cleanup**: Observer properly disconnected on component destruction
- **Threshold Optimization**: 0.1 threshold balances early loading with performance
- **Root Margin**: 200px preload distance provides smooth user experience

### Animation System

#### Loading Animations
```css
.a2a-lazy-loading-spinner {
    animation: lazySpinner 1s linear infinite;
}

.a2a-section-skeleton {
    animation: skeletonLoading 1.5s infinite;
}

.a2a-section-lazy-loaded {
    animation: lazyFadeIn 500ms ease-in-out;
}
```

#### Transition Effects
- **Spinner Rotation**: Continuous rotation indicating active loading
- **Skeleton Shimmer**: Gradient animation suggesting content structure
- **Fade-In Effect**: Opacity and transform transition for loaded content
- **Smooth Transitions**: Coordinated animations prevent visual jarring

### Error Handling System

#### Error Types
- **Network Errors**: Connection failures during data fetching
- **Timeout Errors**: Loading exceeds 5000ms timeout threshold
- **Server Errors**: API failures or invalid responses
- **Parsing Errors**: Malformed data handling

#### Recovery Mechanisms
- **Retry Functionality**: User-initiated retry attempts
- **Exponential Backoff**: Progressive retry delays for automatic recovery
- **Fallback Content**: Default content when loading completely fails
- **Error Logging**: Client-side error tracking for debugging

### Browser Compatibility
- **Chrome/Edge**: Full Intersection Observer and CSS animation support
- **Firefox**: Complete lazy loading functionality with native API
- **Safari**: WebKit Intersection Observer with optimized animations
- **Mobile Browsers**: Touch-optimized loading states and reduced animation complexity
- **Legacy Fallback**: Graceful degradation to immediate loading without Observer support

### Validation Points
- Loading behavior: Sections load only when approaching viewport
- Placeholder quality: Skeleton screens provide meaningful loading preview
- Animation performance: Loading transitions smooth without performance impact
- Error recovery: Failed sections can be retried successfully
- Performance metrics: Accurate tracking of loading success/failure rates

### Test Data Requirements
- Multiple sections with varying content complexity for load time testing
- Network conditions allowing simulation of loading failures
- Sufficient scroll distance for intersection testing
- Different content types for placeholder variety testing

### Performance Testing
- **Initial Load Time**: Page loads with first section in under 1 second
- **Lazy Load Performance**: Section loading doesn't block UI interactions
- **Memory Usage**: No memory leaks from observers or cached content
- **Animation Performance**: Loading animations maintain 60fps
- **Mobile Performance**: Touch scrolling remains responsive during loading

### Accessibility Features
- **Screen Reader Support**: Loading states announced to assistive technology
- **Keyboard Navigation**: Loading sections accessible via keyboard navigation
- **Focus Management**: Focus handling during content loading transitions
- **High Contrast**: Loading indicators visible in high contrast modes
- **Reduced Motion**: Respects user preferences for reduced animation

### Integration Testing
- **Highlight Coordination**: Section highlighting works with lazy-loaded content
- **Scroll Integration**: Smooth scrolling functions with loading sections
- **Collapse Integration**: Expand/collapse works correctly with lazy-loaded content
- **Anchor Navigation**: Anchor bar navigation triggers appropriate loading

### Metrics and Monitoring
- **Load Success Rate**: Percentage of sections loading successfully
- **Average Load Time**: Mean loading duration across all sections
- **Error Rate**: Frequency of loading failures requiring retry
- **User Interaction**: Click-through rates on retry buttons
- **Performance Impact**: Effect on overall page performance metrics

---

## Test Case ID: TC-UI-AGT-069
**Test Objective**: Verify project edit toggle functionality and form field management  
**Business Process**: Project Data Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-069
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:41-84`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:273-518`
- **Functions Under Test**: `onEdit()`, `_enableEditMode()`, `_disableEditMode()`, `_validateEditForm()`

### Test Preconditions
1. **User Authentication**: Valid developer account with project edit permissions
2. **Project Access**: Project details loaded in ObjectPage view
3. **Data State**: Project contains editable information (name, description, dates, budget)
4. **UI State**: ObjectPage in default read-only mode
5. **Form Controls**: All form fields rendered and accessible

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project Name | Updated Analytics Platform | String | Form Input |
| Description | Enhanced multi-agent analytics system | Text | TextArea Input |
| Start Date | 2024-02-01 | Date | DatePicker |
| End Date | 2024-11-30 | Date | DatePicker |
| Budget | 275000 | Number | Number Input |
| Cost Center | CC-IT-002 | String | Text Input |

### Test Procedure Steps
1. **Step 1 - Toggle Edit Mode On**
   - Action: Click "Edit" button in ObjectPage header actions
   - Expected: Button text changes to "Save Changes" with emphasized type
   - Verification: Edit mode indicator appears, form controls become editable

2. **Step 2 - Form Control State Verification**  
   - Action: Inspect all form controls in General Information section
   - Expected: Input fields, TextArea, DatePickers become enabled and editable
   - Verification: Blue border styling applied to editable controls

3. **Step 3 - Visual Edit Mode Indicators**
   - Action: Observe ObjectPage visual changes after enabling edit mode
   - Expected: Page shows blue left border, "EDIT MODE" indicator in header
   - Verification: Section titles change to blue color, pulsing green indicator visible

4. **Step 4 - Form Field Editing**
   - Action: Modify project name, description, start date, budget values
   - Expected: Changes reflected immediately in form controls
   - Verification: Input validation occurs (required fields, date ranges)

5. **Step 5 - Cancel Edit Functionality**
   - Action: Click "Cancel" button after making changes
   - Expected: Confirmation dialog appears asking to discard changes
   - Verification: Original data restored after confirmation

6. **Step 6 - Form Validation Testing**
   - Action: Clear required fields (name, description) and attempt save
   - Expected: Validation error dialog shows specific error messages
   - Verification: Edit mode remains active until validation passes

7. **Step 7 - Date Range Validation**
   - Action: Set end date before start date and attempt save
   - Expected: Validation error for invalid date range
   - Verification: Error message: "End date must be after start date"

8. **Step 8 - Budget Validation**
   - Action: Enter negative budget value and attempt save
   - Expected: Validation error for invalid budget amount
   - Verification: Error message: "Budget must be a positive number"

9. **Step 9 - Successful Save Operation**
   - Action: Enter valid data and click "Save Changes" button
   - Expected: Loading indicator appears, success message displayed
   - Verification: Edit mode disabled, form returns to read-only state

10. **Step 10 - Action Button State Management**
    - Action: Verify Deploy/Archive buttons during edit mode
    - Expected: Action buttons disabled during edit mode
    - Verification: Only Edit/Cancel buttons functional in edit mode

### Expected Results
- **Edit Mode Toggle Criteria**:
  - Edit button transforms to Save Changes button with proper styling
  - Cancel button appears only in edit mode
  - Visual indicators clearly show edit state
  - Form controls respond properly to edit state changes
  
- **Form Validation Criteria**:
  - Required field validation prevents invalid saves
  - Date range validation ensures logical date sequences
  - Numeric validation prevents invalid budget values
  - Clear error messages guide user corrections

- **Data Persistence Criteria**:
  - Changes saved successfully via API call
  - Loading states shown during save operations
  - Success feedback provided to user
  - Last modified timestamp updated after save

### Post-conditions
- Project data updated with new values in backend
- Edit mode indicator removed from UI
- Form controls returned to read-only state
- User receives confirmation of successful save operation

## Test Case ID: TC-UI-AGT-070
**Test Objective**: Verify role-based field editability and conditional form permissions  
**Business Process**: Security and Access Control  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-070
- **Test Priority**: Critical (P1)
- **Test Type**: Security, Authorization, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:171-277`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:549-834`
- **Functions Under Test**: `_initializeFieldEditability()`, `_updateFieldEditability()`, `_applyFieldEditabilityToControls()`

### Test Preconditions
1. **User Authentication**: Valid user account with specific role assignments
2. **Project Access**: Project in various states (ACTIVE, DEPLOYED, ARCHIVED, DRAFT)
3. **Role Configuration**: Role permissions properly configured in system
4. **UI State**: ObjectPage loaded with project data
5. **Edit Mode**: Form can be toggled between read-only and edit modes

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| User Roles | ADMIN, PROJECT_MANAGER, DEVELOPER, VIEWER | Enum | Role Selector |
| Project Status | ACTIVE, DEPLOYED, ARCHIVED, DRAFT | String | Project Data |
| Field Names | name, description, startDate, endDate, budget, costCenter | Array | Form Fields |
| Permission Matrix | Role-based field access permissions | Object | Configuration |
| Status Restrictions | Project status-based field locks | Object | Business Rules |

### Test Procedure Steps
1. **Step 1 - Admin Role Field Access**
   - Action: Set role to "ADMIN" via role selector in edit mode
   - Expected: All fields show green checkmark permission indicators
   - Verification: All form controls editable (name, description, dates, budget, cost center)

2. **Step 2 - Project Manager Role Restrictions**
   - Action: Change role to "PROJECT_MANAGER" in edit mode
   - Expected: Most fields editable except advanced admin fields
   - Verification: Budget and cost center fields remain editable, critical fields accessible

3. **Step 3 - Developer Role Limited Access**
   - Action: Set role to "DEVELOPER" and observe field permissions
   - Expected: Only description field editable, others show red decline icons
   - Verification: Name, dates, budget, cost center fields become read-only with gray styling

4. **Step 4 - Viewer Role No Edit Access**
   - Action: Change role to "VIEWER" in edit mode
   - Expected: All fields show red decline icons and become read-only
   - Verification: All form controls disabled with "not-allowed" cursor styling

5. **Step 5 - Status-Based Field Restrictions**
   - Action: Change project status to "DEPLOYED" (Admin role)
   - Expected: Name, start date, end date fields become read-only
   - Verification: Only description and budget remain editable for deployed projects

6. **Step 6 - Archived Project Complete Lockdown**
   - Action: Set project status to "ARCHIVED" (Admin role)
   - Expected: All fields become read-only regardless of role
   - Verification: All form controls show read-only styling and decline icons

7. **Step 7 - Field Focus Behavior Testing**
   - Action: Attempt to focus on read-only fields in edit mode
   - Expected: Red focus border appears, toast message shows restriction reason
   - Verification: Message explains specific reason (role vs status restriction)

8. **Step 8 - Tooltip Information Display**
   - Action: Hover over read-only fields during edit mode
   - Expected: Tooltip shows specific reason for read-only state
   - Verification: Messages like "Insufficient permissions for role 'DEVELOPER'"

9. **Step 9 - Permission Matrix Validation**
   - Action: Click "Test Field Editability" button to show current permissions
   - Expected: Dialog displays current role, project status, and editable fields list
   - Verification: Information accurately reflects current field editability state

10. **Step 10 - Dynamic Permission Updates**
    - Action: Change role while in edit mode, observe real-time field updates
    - Expected: Form controls immediately reflect new permission levels
    - Verification: Visual indicators, styling, and editability update without page refresh

### Expected Results
- **Role-Based Access Criteria**:
  - ADMIN: Full access to all fields including system-level controls
  - PROJECT_MANAGER: Business fields editable, restricted from system settings
  - DEVELOPER: Limited to content fields like descriptions
  - VIEWER: No edit access to any fields regardless of mode
  
- **Status-Based Restriction Criteria**:
  - ACTIVE/DRAFT projects: Permissions follow role-based rules
  - DEPLOYED projects: Critical fields (name, dates) locked for stability
  - ARCHIVED projects: All fields read-only to preserve historical data

- **Visual Feedback Criteria**:
  - Permission indicators show green checkmarks for editable fields
  - Red decline icons indicate read-only fields with specific reasons
  - Read-only fields have grayed styling with "not-allowed" cursor
  - Focus states provide immediate feedback with colored borders

### Security Validation
- **Authorization Enforcement**: Field permissions enforced at UI level
- **Role Verification**: User roles validated against permission matrix
- **Status Compliance**: Project status restrictions properly applied
- **Bypass Prevention**: No client-side workarounds for permission restrictions

### Accessibility Compliance
- **Screen Reader**: Permission indicators announced to screen readers
- **Keyboard Navigation**: Tab order respects field editability states
- **High Contrast**: Visual indicators maintain contrast ratios
- **Focus Management**: Clear focus indication for accessible vs restricted fields

### Post-conditions
- Field permissions accurately reflect user role and project status
- Visual indicators provide clear feedback on edit capabilities
- Security restrictions properly enforced without client-side bypasses
- User receives appropriate guidance for restricted field access

## Test Case ID: TC-UI-AGT-071
**Test Objective**: Verify comprehensive save changes functionality with validation, conflict detection, and progress tracking  
**Business Process**: Data Persistence and Change Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-071
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Data Integrity, Concurrent Operations
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:514-890`
- **View**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:155-163`
- **Functions Under Test**: `_saveChanges()`, `_preSaveValidation()`, `_performSave()`, `_handleSaveError()`

### Test Preconditions
1. **Edit Mode Active**: Project ObjectPage in edit mode with editable fields
2. **Data Changes**: Modified project data ready for persistence
3. **User Permissions**: Valid user role with save permissions
4. **Backend Connectivity**: API endpoints available for save operations
5. **Validation Rules**: Business validation rules configured and active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project Name | Analytics Platform v2.0 | String | Text Input |
| Description | Enhanced AI-powered analytics system | Text | TextArea |
| Start Date | 2024-03-01 | Date | DatePicker |
| End Date | 2024-12-31 | Date | DatePicker |
| Budget | 350000 | Number | Number Input |
| Budget Limit | 500000 | Number | Configuration |
| Project Duration | 305 days | Calculated | Date Range |
| Change Summary | name, description, budget | Array | Change Tracking |

### Test Procedure Steps
1. **Step 1 - No Changes Detection**
   - Action: Click "Save Changes" without modifying any fields
   - Expected: "No changes detected to save" message appears
   - Verification: Save operation aborted, edit mode remains active

2. **Step 2 - Basic Field Validation**
   - Action: Clear project name and attempt to save
   - Expected: Validation error dialog shows "Project name is required"
   - Verification: Save blocked, specific error highlighted in dialog

3. **Step 3 - Business Rule Validation**
   - Action: Set project duration to 5 days (less than 7-day minimum)
   - Expected: Error message "Project duration must be at least 7 days"
   - Verification: Save prevented, date range validation enforced

4. **Step 4 - Warning Validation with Confirmation**
   - Action: Set budget to $1,200,000 (exceeds $1M warning threshold)
   - Expected: Warning dialog appears asking for confirmation to continue
   - Verification: User can choose to proceed or cancel save operation

5. **Step 5 - Save Progress Tracking**
   - Action: Make valid changes and click "Save Changes"
   - Expected: Progress indicator appears showing 5-step save process
   - Verification: Progress updates through: Prepare → Validate → Save → Cache → Finalize

6. **Step 6 - Change Tracking Verification**
   - Action: Complete successful save and click "View Change History"
   - Expected: Change history dialog shows all modified fields with before/after values
   - Verification: Changes logged with timestamps and user information

7. **Step 7 - Concurrent Edit Conflict Detection**
   - Action: Simulate concurrent modification and attempt save
   - Expected: Conflict warning appears with options (Overwrite/Reload/Cancel)
   - Verification: User can choose resolution strategy for concurrent edits

8. **Step 8 - Network Error Handling**
   - Action: Simulate network failure during save operation
   - Expected: Error dialog with retry option appears
   - Verification: User can retry save operation or cancel gracefully

9. **Step 9 - Server Validation Error**
   - Action: Send invalid data that passes client validation but fails server validation
   - Expected: Server error message displayed with specific details
   - Verification: Error details explain server-side validation failures

10. **Step 10 - Force Save Override**
    - Action: Click "Force Save" button to override concurrent modifications
    - Expected: Confirmation dialog warns about overwriting other user changes
    - Verification: Force save bypasses optimistic locking checks

### Expected Results
- **Change Detection Criteria**:
  - Only modified fields included in save payload
  - Change tracking captures before/after values with timestamps
  - No unnecessary API calls made when no changes exist
  - Change history maintained for audit purposes
  
- **Validation Enforcement Criteria**:
  - Client-side validation prevents invalid data submission
  - Server-side validation provides additional security layer
  - Business rules consistently enforced (duration, budget, status)
  - Warning vs error distinction properly implemented

- **Progress Feedback Criteria**:
  - Multi-step save process clearly communicated to user
  - Progress indicator shows meaningful step descriptions
  - Loading states prevent accidental double-saves
  - Success confirmation includes change summary

- **Error Recovery Criteria**:
  - Network failures gracefully handled with retry options
  - Validation errors provide specific, actionable guidance
  - Concurrent edit conflicts offer resolution choices
  - Force save option available for administrative overrides

### Data Integrity Validation
- **Optimistic Locking**: Prevents concurrent modification conflicts
- **Change Audit Trail**: All modifications logged with user and timestamp
- **Validation Consistency**: Client and server validation rules aligned
- **Transaction Safety**: Save operations atomic with proper rollback

### Performance Requirements
- **Save Operation**: Complete within 5 seconds under normal conditions
- **Progress Updates**: Real-time feedback during multi-step process
- **Change Calculation**: Efficient diff algorithm for large data sets
- **History Storage**: Last 50 changes maintained per session

### Security Considerations
- **Permission Validation**: User authorization checked before save
- **Data Sanitization**: Input data properly sanitized and validated
- **Audit Logging**: All save operations logged for security review
- **Concurrent Access**: Proper handling of multi-user editing scenarios

### Accessibility Features
- **Screen Reader**: Save progress announced to assistive technologies
- **Keyboard Navigation**: Save operation accessible via keyboard shortcuts
- **High Contrast**: Progress indicators visible in high contrast modes
- **Focus Management**: Proper focus handling during save operations

### Post-conditions
- Modified data successfully persisted to backend database
- Change history updated with comprehensive audit information
- Edit mode automatically disabled after successful save
- User receives confirmation with summary of saved changes
- Field editability permissions refreshed based on new data state

## Test Case ID: TC-UI-AGT-072
**Test Objective**: Verify comprehensive cancel changes functionality with selective reversion, draft management, and navigation protection  
**Business Process**: Change Management and Data Protection  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-072
- **Test Priority**: High (P1)
- **Test Type**: Functional, User Experience, Data Integrity
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:494-833`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/view/fragments/ChangeReviewDialog.fragment.xml`
- **Functions Under Test**: `onCancelEdit()`, `_showAdvancedCancelDialog()`, `_discardSelectedChanges()`, `onAutoSaveDraft()`

### Test Preconditions
1. **Edit Mode Active**: Project ObjectPage in edit mode with modified fields
2. **Unsaved Changes**: Multiple fields modified from original values
3. **Change Detection**: System capable of detecting and tracking field changes
4. **Local Storage**: Browser local storage available for draft functionality
5. **Navigation Events**: Browser navigation events can be intercepted

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Original Name | Analytics Platform | String | Initial Data |
| Modified Name | Advanced Analytics Platform | String | User Input |
| Original Budget | 250000 | Number | Initial Data |
| Modified Budget | 350000 | Number | User Input |
| Original Description | Basic analytics system | Text | Initial Data |
| Modified Description | AI-powered advanced analytics platform | Text | User Input |
| Draft Timestamp | 2024-01-15T10:30:00Z | DateTime | System Generated |

### Test Procedure Steps
1. **Step 1 - No Changes Cancel**
   - Action: Enter edit mode without making changes, click "Cancel"
   - Expected: "No changes to cancel - exiting edit mode" message
   - Verification: Edit mode exits immediately without confirmation dialog

2. **Step 2 - Advanced Cancel Dialog**
   - Action: Modify multiple fields and click "Cancel"
   - Expected: Advanced dialog shows with 4 options and change summary
   - Verification: Dialog displays: Discard All, Review Changes, Save & Exit, Continue Editing

3. **Step 3 - Change Summary Display**
   - Action: Review the change summary in the advanced cancel dialog
   - Expected: All modified fields listed with "original → modified" format
   - Verification: Change descriptions show proper field names and truncated values

4. **Step 4 - Continue Editing Option**
   - Action: Click "Continue Editing" in the cancel dialog
   - Expected: Dialog closes, edit mode remains active, changes preserved
   - Verification: All modified values remain in form fields

5. **Step 5 - Save & Exit Option**
   - Action: Click "Save & Exit" in the cancel dialog
   - Expected: Save process initiated, edit mode exits on successful save
   - Verification: Changes persisted to backend, form returns to read-only mode

6. **Step 6 - Selective Change Review**
   - Action: Click "Review Changes" to open detailed change review dialog
   - Expected: Table shows all changes with checkboxes for selective reversion
   - Verification: Each row shows field name, original value, and current value

7. **Step 7 - Select/Deselect All Changes**
   - Action: Use "Select All" and "Deselect All" buttons in review dialog
   - Expected: All checkboxes toggle appropriately
   - Verification: Button states update based on selection count

8. **Step 8 - Selective Field Reversion**
   - Action: Select specific fields and click "Discard Selected"
   - Expected: Only selected fields revert to original values
   - Verification: Unselected changes preserved, selected fields restored

9. **Step 9 - Auto-Save Draft Functionality**
   - Action: Make changes and click "Auto-Save Draft" button
   - Expected: "Draft auto-saved locally" confirmation message
   - Verification: Draft data stored in browser local storage with timestamp

10. **Step 10 - Draft Restoration**
    - Action: Reload page and click "Restore Draft" button
    - Expected: Confirmation dialog with draft timestamp, option to restore
    - Verification: Draft data restored to form fields when confirmed

### Expected Results
- **Change Detection Criteria**:
  - System accurately detects all field modifications
  - Change summaries display meaningful before/after comparisons  
  - Long values properly truncated for dialog display
  - Field display names use user-friendly labels
  
- **Cancel Option Criteria**:
  - Four distinct cancellation options clearly presented
  - User can choose granular vs bulk cancellation approaches
  - Continue editing preserves all current work
  - Save & Exit provides seamless transition to persistence

- **Selective Reversion Criteria**:
  - Individual field changes can be selectively discarded
  - Bulk selection/deselection operations work efficiently
  - Partial reversions maintain form consistency
  - Remaining changes preserved after selective operations

- **Draft Management Criteria**:
  - Auto-save functionality prevents accidental data loss
  - Draft restoration works across browser sessions
  - Timestamp information helps identify draft currency
  - Local storage gracefully handles quota limitations

### User Experience Validation
- **Dialog Usability**: Clear action buttons with appropriate emphasis and focus
- **Change Visualization**: Intuitive before/after value comparison display
- **Progress Feedback**: Loading indicators during reversion operations
- **Error Handling**: Graceful handling of local storage failures

### Data Protection Features
- **Navigation Prevention**: Warns user before leaving page with unsaved changes
- **Audit Logging**: All cancellation operations logged with session details
- **Clean Exit**: Proper cleanup of edit indicators and temporary data
- **State Recovery**: Ability to restore previous state from multiple sources

### Performance Considerations
- **Change Calculation**: Efficient diff operations for large datasets
- **Dialog Rendering**: Quick response time for change review dialog
- **Local Storage**: Optimized storage usage with automatic cleanup
- **Memory Management**: Proper cleanup of event listeners and references

### Security Aspects
- **Data Sanitization**: Draft data properly sanitized before storage
- **Session Management**: Draft data associated with current user session
- **Audit Trail**: Cancellation events logged for security review
- **Permission Validation**: User authorization verified before operations

### Post-conditions
- Edit mode properly exited with clean state restoration
- All temporary change indicators and event handlers removed
- Draft data appropriately managed in local storage
- User receives clear confirmation of cancellation results
- System ready for new edit session without interference

## Test Case ID: TC-UI-AGT-073
**Test Objective**: Verify comprehensive validation display system with real-time feedback, detailed error reporting, and visual indicators  
**Business Process**: Form Validation and Data Quality Assurance  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-073
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, User Experience, Data Quality
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:352-783`
- **View**: `a2aAgents/backend/app/a2a/developerPortal/static/view/ProjectObjectPage.view.xml:185-198`
- **Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/view/fragments/ValidationDetailsDialog.fragment.xml`
- **Functions Under Test**: `_performComprehensiveValidation()`, `onRealTimeValidation()`, `_updateValidationDisplay()`

### Test Preconditions
1. **Edit Mode Active**: Project ObjectPage in edit mode with form fields enabled
2. **Validation Rules**: Comprehensive validation rules configured for all field types
3. **Real-time Events**: Form controls configured with live change events
4. **Visual System**: Validation styling and indicator system initialized
5. **Error Handling**: Validation error display mechanisms functional

### Test Input Data
| Parameter | Value | Type | Validation Rule |
|-----------|--------|------|-----------------|
| Valid Name | Analytics Platform Pro | String | 3-100 chars, alphanumeric |
| Invalid Name | AB | String | Too short (< 3 chars) |
| Reserved Name | System Test | String | Contains reserved word |
| Valid Description | Comprehensive analytics system for enterprise data processing | Text | 10-2000 chars |
| Invalid Description | Short | Text | Too short (< 10 chars) |
| Past Start Date | 2023-01-01 | Date | Date in past (warning) |
| Future End Date | 2030-12-31 | Date | More than 5 years future |
| Invalid Budget | -1000 | Number | Negative number (error) |
| High Budget | 15000000 | Number | > 10M (warning) |
| Invalid Cost Center | INVALID | String | Wrong format |

### Test Procedure Steps
1. **Step 1 - Real-time Name Validation**
   - Action: Enter "AB" in project name field
   - Expected: Red error styling appears, error message "Project name must be at least 3 characters long"
   - Verification: Field shows error state, live validation indicator updates

2. **Step 2 - Reserved Word Detection**
   - Action: Enter "System Test Platform" in project name
   - Expected: Yellow warning styling, message "Project name contains reserved words"
   - Verification: Warning icon displayed, field accepts input with advisory

3. **Step 3 - Description Length Validation**
   - Action: Enter "Short" in description field
   - Expected: Warning message "Description is quite short - provide more details"
   - Verification: Field styling updates, character count guidance provided

4. **Step 4 - Date Range Cross-Validation**
   - Action: Set end date before start date
   - Expected: Error on end date field "End date must be after start date"
   - Verification: Cross-field validation triggers, both date fields show context

5. **Step 5 - Budget Range Validation**
   - Action: Enter negative budget value "-1000"
   - Expected: Error styling, message "Budget must be a positive number"
   - Verification: Numeric validation enforced, input restriction applied

6. **Step 6 - Live Validation Indicator**
   - Action: Observe header validation indicator during field modifications
   - Expected: Percentage updates in real-time, error/warning counts change
   - Verification: "X% valid" display reflects current form state accurately

7. **Step 7 - Validation Summary Dialog**
   - Action: Click "Validate All Fields" button with mixed valid/invalid fields
   - Expected: Validation summary appears with detailed breakdown
   - Verification: Error dialog lists all issues categorized by severity

8. **Step 8 - Detailed Validation Dialog**
   - Action: Click "Show Details" in validation summary dialog
   - Expected: Comprehensive validation details dialog opens
   - Verification: Field-by-field results with status icons and messages

9. **Step 9 - Validation Panel Toggle**
   - Action: Click "Toggle Validation Panel" button
   - Expected: Validation panel appears/disappears with current validation state
   - Verification: Panel shows real-time validation status and statistics

10. **Step 10 - Cross-Field Business Rules**
    - Action: Set status to "DEPLOYED" without any agents configured
    - Expected: Business rule error "Cannot deploy project without agents"
    - Verification: Cross-field validation enforces complex business logic

### Expected Results
- **Real-time Validation Criteria**:
  - Field validation occurs immediately during user input
  - Visual feedback provided through styling and icons
  - ValueState properly set (Error/Warning/Success)
  - Validation messages appear in field tooltips and value state text
  
- **Validation Display Criteria**:
  - Live validation indicator shows accurate percentage and counts
  - Error/warning/success states visually distinct and accessible
  - Comprehensive validation summary with categorized issues
  - Detailed validation dialog provides field-by-field breakdown

- **Business Rule Enforcement Criteria**:
  - Individual field validation rules properly enforced
  - Cross-field validation detects relationship violations
  - Business logic rules prevent invalid state combinations
  - Validation severity appropriately categorized (Error vs Warning)

- **User Experience Criteria**:
  - Validation feedback immediate and non-intrusive
  - Clear guidance provided for fixing validation issues
  - Progressive disclosure from summary to detailed views
  - Accessible validation states for screen readers

### Validation Rule Coverage
- **Field-Level Rules**: Required fields, length limits, format validation, range checking
- **Cross-Field Rules**: Date sequences, budget-duration correlation, status-dependency checks
- **Business Rules**: Deployment prerequisites, priority-budget alignment, reserved word detection
- **Format Rules**: Cost center patterns, special character restrictions, numeric constraints

### Performance Validation
- **Real-time Response**: Validation feedback within 100ms of input change
- **Large Form Handling**: Efficient validation for forms with 20+ fields
- **Memory Usage**: Proper cleanup of validation event handlers
- **Calculation Speed**: Complex cross-field validation completes within 200ms

### Accessibility Compliance
- **Screen Reader Support**: Validation messages announced to assistive technology
- **Keyboard Navigation**: All validation controls accessible via keyboard
- **High Contrast**: Validation indicators maintain proper contrast ratios
- **Focus Management**: Error fields receive appropriate focus indication

### Visual Design Standards
- **Color Coding**: Red for errors, yellow for warnings, green for success
- **Icon Usage**: Consistent iconography for validation states
- **Typography**: Clear hierarchy for validation messages
- **Spacing**: Adequate spacing for validation indicators and messages

### Post-conditions
- All form fields display appropriate validation state based on current values
- Live validation indicator accurately reflects overall form validity
- Validation messages provide clear guidance for issue resolution
- Validation system ready for integration with save/submit operations

---

## Test Case ID: TC-UI-AGT-074
**Test Objective**: Verify AI suggestions display functionality during form editing  
**Business Process**: Intelligent Form Assistance with AI-powered Suggestions  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-074
- **Test Priority**: High (P1)
- **Test Type**: Functional, AI/UX Integration
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:3693-4129`
- **UI Fragment**: `a2aAgents/backend/app/a2a/developerPortal/static/view/fragments/SuggestionsPopover.fragment.xml:1-142`
- **Functions Under Test**: `_initializeSuggestions()`, `_generateContextualSuggestions()`, `onAcceptSuggestion()`, `onRejectSuggestion()`

### Test Preconditions
1. **Edit Mode Active**: User has enabled edit mode for project data
2. **Suggestions System**: AI suggestions functionality is enabled and initialized
3. **Project Data**: Valid project context with existing or partial field values
4. **Field Focus**: Form fields are interactive and support live input changes
5. **AI Backend**: Suggestions engine is running and responsive (simulated patterns in test)

### Test Input Data
| Field | Input Value | Expected Suggestion Type | Description |
|-------|-------------|-------------------------|-------------|
| Project Name | "AI Ass" | Completion | Should suggest "AI Assistant Development Project" |
| Description | Short text | Validation | Tips for better descriptions |
| Cost Center | "CC-" | Completion | Format suggestions like "CC-IT-001" |
| Budget | 50000 | Optimization | Budget optimization tips for project scope |
| Invalid Name | "A" | Validation | Too short validation tip |

### Test Procedure Steps

1. **Step 1 - Enable Suggestions**
   - Action: Click "AI Suggestions" button in toolbar (lightbulb icon)
   - Expected: Button becomes emphasized, tooltip shows "0 suggestions"
   - Verification: Suggestions are enabled in view model (`/suggestions/enabled = true`)

2. **Step 2 - Trigger Field Suggestions**
   - Action: Focus on "Project Name" field and type "AI Ass"
   - Expected: After 1 second delay, suggestions popover appears below field
   - Verification: Popover contains completion suggestion "AI Assistant Development Project"

3. **Step 3 - Accept Completion Suggestion**
   - Action: Click "Apply" button on the completion suggestion
   - Expected: Field value updates to full suggestion text
   - Verification: Project name input shows "AI Assistant Development Project"

4. **Step 4 - Test Validation Suggestions**
   - Action: Clear project name field and enter single character "A"
   - Expected: Validation suggestions appear with tips about naming best practices
   - Verification: Suggestion shows "Project names should be descriptive and unique"

5. **Step 5 - Reject Suggestion**
   - Action: Click "Dismiss" button on validation suggestion
   - Expected: Suggestion removes from list, feedback recorded
   - Verification: Toast message "Suggestion dismissed" appears

6. **Step 6 - Test Context-Aware Suggestions**
   - Action: Set start date to 12+ months from end date, focus on budget field
   - Expected: Optimization suggestion appears about long-term project budgeting
   - Verification: Context suggestion shows "Consider phased budget allocation for projects over 12 months"

7. **Step 7 - Configure Suggestions**
   - Action: Click "Configure Suggestions" button (learning-assistant icon)
   - Expected: Configuration dialog opens with current settings
   - Verification: Dialog shows switches for different suggestion types

8. **Step 8 - Modify Suggestion Settings**
   - Action: Disable "Auto-completion" switch and save configuration
   - Expected: Configuration saves, completion suggestions no longer appear
   - Verification: Only validation and optimization suggestions display

9. **Step 9 - Test Suggestions Statistics**
   - Action: Open suggestions config, expand "Learning Statistics" panel
   - Expected: Shows count of accepted/rejected suggestions and success rate
   - Verification: Statistics reflect previous test interactions

10. **Step 10 - Disable Suggestions System**
    - Action: Toggle "AI Suggestions" button to disabled state
    - Expected: No suggestions appear even with field interactions
    - Verification: Button becomes default type, no popovers trigger

### Expected Results

**Suggestions Display Criteria:**
- Suggestions popover appears within 1 second of field input
- Maximum 5 suggestions displayed at once, prioritized by relevance
- Each suggestion shows type icon, confidence percentage, and clear description
- Accept/dismiss buttons provide appropriate actions for each suggestion type

**Suggestion Types Verification:**
- **Completion**: Auto-complete based on common patterns with "Apply" action
- **Validation**: Tips and best practices with "Got it" acknowledgment
- **Optimization**: Context-aware recommendations for better project setup
- **Context-Aware**: Suggestions based on other field values and project data

**User Experience Criteria:**
- Suggestions appear contextually without disrupting input flow
- Clear visual hierarchy distinguishes suggestion types with appropriate icons
- Feedback is recorded for machine learning improvement
- Configuration allows users to customize suggestion behavior

**Performance Requirements:**
- Suggestions generation completes within 200ms of trigger
- Popover rendering under 100ms for smooth user experience
- No impact on form input responsiveness during suggestion display
- Efficient cleanup when suggestions disabled or edit mode exited

**Accessibility Compliance:**
- Suggestions popover accessible via keyboard navigation
- Screen reader announcements for suggestion appearance/dismissal
- High contrast support for suggestion type indicators
- Focus management maintains proper tab order with popover open

### Business Impact Validation
- **User Productivity**: Suggestions reduce time to complete form fields accurately
- **Data Quality**: Validation suggestions prevent common input errors
- **Learning System**: User feedback improves suggestion accuracy over time
- **Usability**: Configuration options provide personalized experience

### Error Handling Verification
- Graceful handling when AI backend unavailable (fallback to pattern-based suggestions)
- Proper cleanup of suggestion timeouts when switching fields rapidly
- Error recovery if suggestion popover fails to render
- Fallback behavior when suggestion data is invalid or corrupted

### Integration Points
- Suggestions integrate with real-time validation system
- Feedback data flows to learning service API endpoints
- Configuration persists to user profile for cross-session consistency
- Suggestions respect field editability and permission restrictions

### Post-conditions
- AI suggestions system demonstrates intelligent assistance during form editing
- User interactions with suggestions improve system learning and future recommendations
- Configuration settings provide appropriate control over suggestion behavior
- Feedback tracking enables continuous improvement of suggestion accuracy and relevance

---

## Test Case ID: TC-UI-AGT-075
**Test Objective**: Verify comprehensive suggestion acceptance functionality with multiple action types  
**Business Process**: Advanced AI Suggestion Acceptance with Preview and Rollback  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-075
- **Test Priority**: High (P1)
- **Test Type**: Functional, Integration, User Experience
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:3948-4564`
- **Enhanced Methods**: `_processSuggestionAcceptance()`, `_applySuggestionReplace()`, `_applySuggestionAppend()`, `_applySuggestionInsert()`
- **Functions Under Test**: Multiple suggestion action types, validation, rollback, preview functionality

### Test Preconditions
1. **Edit Mode**: Project object page in edit mode with suggestions enabled
2. **Suggestion System**: AI suggestions initialized with enhanced patterns supporting multiple action types
3. **Field Data**: Form fields contain various values for testing different action types
4. **Backend Mock**: Acceptance feedback endpoint ready to receive detailed tracking data
5. **Validation System**: Real-time validation active for post-acceptance verification

### Test Input Data
| Action Type | Field | Current Value | Suggestion Value | Expected Result |
|-------------|-------|---------------|------------------|------------------|
| replace | Project Name | "Test" | "AI Assistant Development Project" | Complete replacement |
| append | Project Name | "Customer Portal" | " - Phase 1" | "Customer Portal - Phase 1" |
| insert | Project Name | "Portal Enhancement" | "Customer " @ pos 0 | "Customer Portal Enhancement" |
| modify | Cost Center | "CC-OLD-001" | "IT" (pattern: "OLD") | "CC-IT-001" |
| info | Budget | "50000" | Validation tip only | No field change |

### Test Procedure Steps

1. **Step 1 - Test Replace Action**
   - Action: Type "AI Ass" in project name field, wait for suggestions
   - Expected: Completion suggestion "AI Assistant Development Project" appears with "Replace" button
   - Verification: Suggestion shows action="replace" and previewable=true

2. **Step 2 - Preview Replace Suggestion**
   - Action: Click "Preview" button on the replace suggestion
   - Expected: Preview dialog shows current vs. new value comparison
   - Verification: Dialog displays "Current: 'AI Ass'" and "After: 'AI Assistant Development Project'"

3. **Step 3 - Accept Replace Suggestion**
   - Action: Click "Apply Now" in preview dialog or "Replace" button directly
   - Expected: Field value completely replaced, success toast shown, undo bar appears
   - Verification: Field now contains "AI Assistant Development Project", validation updates

4. **Step 4 - Test Rollback Functionality**
   - Action: Click the undo bar that appeared after acceptance
   - Expected: Field reverts to original value before suggestion
   - Verification: Field returns to "AI Ass", toast shows "Suggestion undone"

5. **Step 5 - Test Append Action**
   - Action: Enter "Customer Portal" in project name, trigger suggestions
   - Expected: Advanced suggestion appears with "Append" action for " - Phase 1"
   - Verification: Button shows "Append" with tooltip "Add to end of current value"

6. **Step 6 - Accept Append Suggestion**
   - Action: Click "Append" button on phase suffix suggestion
   - Expected: Text appended to existing value without replacement
   - Verification: Field shows "Customer Portal - Phase 1", original text preserved

7. **Step 7 - Test Insert Action**  
   - Action: Clear field, enter "Portal Enhancement", position cursor at start
   - Expected: Insert suggestion "Customer " appears for position 0
   - Verification: Suggestion shows action="insert" with insertPosition=0

8. **Step 8 - Accept Insert Suggestion**
   - Action: Click "Insert" button on the positional suggestion
   - Expected: Text inserted at specified position maintaining existing content
   - Verification: Field shows "Customer Portal Enhancement" with proper insertion

9. **Step 9 - Test Modify Action with Pattern**
   - Action: Enter "CC-OLD-001" in cost center field
   - Expected: Pattern-based suggestion to replace "OLD" with "IT" appears
   - Verification: Suggestion shows action="modify" with modifyPattern and replacement value

10. **Step 10 - Test Acceptance Error Handling**
    - Action: Make field read-only, then try to accept a suggestion
    - Expected: Error dialog appears with clear message about field not being editable
    - Verification: No field changes, error logged to feedback tracking

11. **Step 11 - Test Validation After Acceptance**
    - Action: Accept suggestion that results in validation warnings
    - Expected: Suggestion applied but user informed about validation issues
    - Verification: Field updated, toast shows validation warning count

12. **Step 12 - Verify Feedback Tracking**
    - Action: Open suggestions config, check learning statistics
    - Expected: Accepted suggestions counter increased, success rate updated
    - Verification: Statistics reflect all acceptance interactions from test steps

### Expected Results

**Acceptance Mechanism Verification:**
- Each action type (replace, append, insert, modify, info) handled correctly
- Pre-acceptance validation prevents invalid applications
- Post-acceptance validation provides warnings without blocking
- Original values preserved for rollback functionality

**User Experience Validation:**
- Clear visual feedback for each action type with appropriate button labels
- Preview functionality shows accurate before/after comparison
- Success notifications provide meaningful feedback with undo options
- Error handling provides clear guidance without data loss

**Technical Implementation Verification:**
- Field control identification works across different control types
- Value calculation accurate for all action types including edge cases
- Event firing maintains proper data model synchronization
- Rollback functions restore exact previous state

**Integration Points Testing:**
- Real-time validation updates after suggestion acceptance
- Feedback data correctly structured and sent to backend learning service  
- Session tracking maintains proper user context for personalization
- Undo functionality integrates with existing change management

**Performance Requirements:**
- Acceptance processing completes within 100ms for immediate user feedback
- Preview calculation and display under 50ms for smooth interaction
- Rollback operations execute instantaneously with no perceptible delay
- Backend feedback transmission doesn't block UI responsiveness

**Error Recovery Validation:**
- Graceful handling of network failures during feedback submission
- Proper cleanup of UI elements if acceptance partially fails
- Clear error messages guide user to resolution without confusion
- Failed acceptance attempts tracked for system improvement

### Business Impact Assessment
- **Productivity Enhancement**: Multiple action types accommodate diverse user preferences
- **Data Accuracy**: Validation integration prevents acceptance of invalid suggestions  
- **Learning Optimization**: Detailed feedback improves future suggestion quality
- **User Confidence**: Preview and undo capabilities reduce acceptance hesitation

### Accessibility and Usability
- **Keyboard Navigation**: All acceptance actions accessible via keyboard shortcuts
- **Screen Reader Support**: Action types and results properly announced
- **Visual Indicators**: Clear distinction between action types with appropriate iconography
- **Error Communication**: Validation warnings and errors clearly communicated to all users

### Post-conditions
- Suggestion acceptance system demonstrates robust handling of multiple action types
- User feedback loop actively improves system learning and personalization
- Preview functionality empowers confident suggestion adoption
- Rollback capabilities provide safety net for experimental usage

---

## Test Case ID: TC-UI-AGT-076
**Test Objective**: Verify comprehensive suggestion rejection functionality with learning and feedback  
**Business Process**: Smart Suggestion Rejection with Pattern Learning and User Preference Adaptation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-076
- **Test Priority**: High (P1)
- **Test Type**: Functional, Learning System, User Experience
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:4375-4732`
- **Enhanced Methods**: `_processSuggestionRejection()`, `_showRejectionReasonDialog()`, `_recordEnhancedRejection()`, `_updateUserPreferencesFromRejection()`
- **Functions Under Test**: Rejection reason collection, pattern learning, temporary suppression, preference adaptation

### Test Preconditions
1. **Edit Mode Active**: Project object page in edit mode with suggestions enabled and displayed
2. **Multiple Suggestions**: Various suggestion types available (completion, validation, optimization)
3. **Clean State**: No existing rejection patterns to ensure consistent test results
4. **Backend Ready**: Rejection feedback endpoint configured to accept enhanced data
5. **User Context**: Sufficient project and user data for contextual rejection analysis

### Test Input Data
| Rejection Type | Suggestion | Field | Reason | Expected Behavior |
|----------------|------------|-------|---------|-------------------|
| Info Dismissal | "Validation tip" | Project Name | acknowledged | Temporary suppression, friendly message |
| Relevance | "AI Assistant..." | Description | not_relevant | Block similar, disable completion suggestions |
| Incorrectness | "CC-WRONG-001" | Cost Center | incorrect | Pattern rejection, backend review flag |
| Overwhelming | Multiple suggestions | Any field | too_many | Reduce frequency, lower max suggestions |
| Bad Timing | Early suggestion | Budget | wrong_timing | Increase delay, retry later allowed |
| Different Approach | Any actionable | Any field | different_approach | Learn user preference pattern |

### Test Procedure Steps

1. **Step 1 - Test Info Suggestion Acknowledgment**
   - Action: Trigger validation suggestion, click "Got it" button
   - Expected: Friendly toast "Got it! Tip acknowledged", suggestion removed
   - Verification: Suggestion type temporarily suppressed for 5 minutes

2. **Step 2 - Test Actionable Suggestion Rejection Dialog**
   - Action: Trigger completion suggestion, click "Dismiss" button
   - Expected: Rejection reason dialog opens with 6 radio button options
   - Verification: Dialog shows suggestion title and "Don't show similar" checkbox

3. **Step 3 - Test Relevance Rejection with Block Similar**
   - Action: Select "Not relevant for this field", check "Don't show similar", submit
   - Expected: Completion suggestions disabled, pattern added to rejection list
   - Verification: Toast "Thanks! We'll improve relevance for this field type (similar suggestions blocked)"

4. **Step 4 - Test Incorrectness Rejection Feedback**
   - Action: Select "Suggestion is incorrect or inappropriate", submit feedback
   - Expected: Backend receives enhanced rejection data with suggestion review flag
   - Verification: Toast "Feedback noted - we'll review this suggestion pattern"

5. **Step 5 - Test Overwhelming Rejection Auto-Adjustment**
   - Action: Select "Too many suggestions (overwhelming)" reason
   - Expected: Max suggestions reduced by 1, show delay increased by 500ms
   - Verification: Configuration shows updated values, fewer suggestions appear

6. **Step 6 - Test Wrong Timing Rejection Delay Increase**
   - Action: Select "Suggestion appears at wrong time" reason
   - Expected: Show delay increased by 250ms, but suggestion not permanently blocked
   - Verification: Same suggestion can appear later with increased delay

7. **Step 7 - Test Different Approach Learning**
   - Action: Select "I prefer a different approach" for multiple suggestions
   - Expected: System learns user preferences for suggestion types
   - Verification: Toast "Got it! We'll learn from your preferred approach"

8. **Step 8 - Test Quick Dismiss Without Reason**
   - Action: Click "Just Dismiss" without selecting reason
   - Expected: Suggestion removed with minimal feedback collection
   - Verification: Recorded as "dismissed_without_reason" in feedback data

9. **Step 9 - Test Rejection Pattern Filtering**
   - Action: Trigger suggestions for field where patterns were rejected
   - Expected: Previously rejected suggestions don't reappear
   - Verification: Filtered suggestions list excludes rejected patterns

10. **Step 10 - Test Suppression Expiry**
    - Action: Wait 5+ minutes after info suggestion acknowledgment, trigger again
    - Expected: Previously suppressed suggestion type can appear again
    - Verification: Temporary suppression period expired, suggestions shown

11. **Step 11 - Test Backend Improved Suggestions**
    - Action: Trigger rejection that returns improved suggestions from backend
    - Expected: New suggestions marked with learning assistant icon appear
    - Verification: Improved suggestions have high confidence and priority

12. **Step 12 - Verify Comprehensive Feedback Tracking**
    - Action: Open suggestions config, check learning statistics
    - Expected: Rejection statistics updated with detailed breakdown by reason
    - Verification: Statistics show rejection count, reasons distribution, and success rate impact

### Expected Results

**Rejection Mechanism Verification:**
- Info suggestions handled with acknowledgment rather than rejection
- Actionable suggestions prompt detailed reason collection via dialog
- Rejection reasons mapped to appropriate system adjustments
- User preferences automatically updated based on rejection patterns

**Learning System Validation:**
- Rejected patterns filtered from future suggestions effectively
- Temporary suppression prevents overwhelming while allowing later retry
- User preference adaptation improves suggestion relevance over time
- Backend integration enables continuous suggestion quality improvement

**User Experience Excellence:**
- Rejection reason dialog provides comprehensive but not overwhelming options
- Feedback messages acknowledge user input and communicate system learning
- Quick dismissal option available for users who don't want to provide details
- Suggestion frequency and timing adapt to individual user preferences

**Technical Implementation Verification:**
- Enhanced rejection data includes comprehensive context and user state
- Pattern matching prevents exact suggestion repetition appropriately
- Suppression timing mechanism works correctly with expiry handling
- Backend communication includes detailed headers for proper context

**Performance and Responsiveness:**
- Rejection processing completes within 50ms for immediate feedback
- Reason dialog opens instantly without perceptible delay
- Preference updates don't affect current session suggestion performance
- Backend feedback transmission happens asynchronously without blocking UI

**Data Quality and Privacy:**
- Rejection feedback includes sufficient context for learning without privacy violation
- User context captured appropriately (role, project status, field completion)
- Session tracking maintains user privacy while enabling personalization
- Feedback data structured for effective machine learning processing

### Business Impact Assessment
- **User Empowerment**: Detailed rejection reasons give users control over suggestion experience
- **System Learning**: Rich feedback data enables continuous AI improvement
- **Personalization**: Individual user preferences create tailored suggestion experience
- **Efficiency Gains**: Reduced irrelevant suggestions increase user productivity

### Advanced Functionality Verification
- **Smart Filtering**: Rejection patterns prevent repetitive poor suggestions
- **Contextual Adaptation**: System learns field-specific and role-specific preferences
- **Temporal Intelligence**: Wrong-timing rejections increase delays while preserving suggestion value
- **Collaborative Learning**: Individual feedback contributes to system-wide improvement

### Error Handling and Edge Cases
- **Network Failures**: Graceful handling of backend communication failures during feedback
- **Rapid Rejections**: System handles multiple quick rejections without performance degradation
- **Invalid Selections**: Dialog validation ensures proper reason selection before submission
- **Storage Limits**: Rejection pattern storage managed to prevent excessive memory usage

### Integration Testing
- **Validation System**: Rejection patterns work correctly with real-time validation
- **Edit Mode Integration**: Rejection functionality respects edit mode state and permissions
- **Configuration Sync**: Preference changes from rejections reflected in configuration dialog
- **Session Persistence**: Rejection patterns and suppressions maintained across browser refresh

### Post-conditions
- Suggestion rejection system demonstrates intelligent learning from user feedback
- Individual user preferences significantly improve suggestion relevance and timing
- System-wide suggestion quality benefits from aggregated rejection pattern analysis
- User confidence in AI suggestions increases due to responsive feedback incorporation

---

## Test Case ID: TC-UI-AGT-077
**Test Objective**: Verify comprehensive AI insights panel functionality with multi-category analysis  
**Business Process**: AI-Powered Project Intelligence and Decision Support  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-077
- **Test Priority**: High (P1)
- **Test Type**: Functional, AI Analytics, Decision Support
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:4809-5365`
- **UI Fragments**: `InsightsPanel.fragment.xml`, `InsightsConfigDialog.fragment.xml`
- **Functions Under Test**: `_generateInsights()`, `_generateRiskInsights()`, `_generateOptimizationInsights()`, `_generateComplianceInsights()`, `_generatePredictionInsights()`

### Test Preconditions
1. **Project Data Available**: Complete project data including budget, timeline, team, and status information
2. **Insights System Enabled**: AI insights functionality initialized and configured
3. **Real-time Analysis Active**: Auto-refresh enabled with 30-second interval for dynamic testing
4. **Multiple Categories Enabled**: All insight categories (risk, optimization, compliance, prediction, recommendation) active
5. **Sufficient Context**: Project contains enough data points to trigger various insight types

### Test Input Data
| Project Attribute | Test Value | Expected Insight Type | Insight Trigger |
|-------------------|------------|----------------------|-----------------|
| Budget | $150,000 | Optimization | Cost optimization opportunity |
| Estimated Cost | $140,000 | Risk | High budget utilization (93%) |
| Duration | 20 days | Risk | Aggressive timeline warning |
| Team Size | 2 members | Risk | Limited team size concern |
| Progress | 30% (timeline 60%) | Prediction | Delivery delay prediction |
| Description | "Test" (4 chars) | Recommendation | Documentation improvement |
| Security Level | 1 | Compliance | Security standards review |
| Data Processing | "personal" | Compliance | GDPR/privacy compliance |

### Test Procedure Steps

1. **Step 1 - Access Insights Panel**
   - Action: Click "AI Insights Panel" button in project header toolbar
   - Expected: Insights dialog opens with current analysis results and summary cards
   - Verification: Dialog shows total insights count, critical issues, warnings, and average confidence

2. **Step 2 - Verify Risk Analysis Insights**
   - Action: Navigate to "Risk Analysis" tab in insights panel
   - Expected: Risk insights display for budget, timeline, and resource concerns
   - Verification: Budget risk shows 93% utilization, timeline risk flags 20-day duration, team size concern noted

3. **Step 3 - Review Optimization Opportunities**
   - Action: Click "Optimization" tab to view optimization insights
   - Expected: Cost optimization recommendations appear for large budget projects
   - Verification: Milestone-based funding suggestion with 15-25% potential savings displayed

4. **Step 4 - Check Compliance Insights**
   - Action: Navigate to "Compliance" insights and verify security/privacy alerts
   - Expected: Security standards review and data privacy compliance insights shown
   - Verification: GDPR compliance requirements listed with specific recommendations

5. **Step 5 - Examine Predictive Insights**
   - Action: View "Predictions" tab for timeline and delivery forecasts
   - Expected: Delivery delay prediction based on progress variance
   - Verification: Predicted delay calculation shown with schedule variance percentage

6. **Step 6 - Test Auto-Refresh Functionality**
   - Action: Modify project budget to $160,000, wait 30+ seconds
   - Expected: Insights automatically refresh with updated budget risk analysis
   - Verification: New insights reflect changed budget with recalculated risk levels

7. **Step 7 - Test Insight Prioritization**
   - Action: Observe insight ordering across all categories
   - Expected: Critical insights appear first, followed by warnings, then info
   - Verification: Insights sorted by severity, confidence, and category priority

8. **Step 8 - Verify Insight Details and Recommendations**
   - Action: Examine detailed insight cards for recommendations and metrics
   - Expected: Each insight shows actionable recommendations and relevant metrics
   - Verification: Recommendations list specific actions, metrics show quantified impacts

9. **Step 9 - Test Export Functionality**
   - Action: Click "Export Insights" button in insights panel
   - Expected: CSV file downloads with comprehensive insight data
   - Verification: Export includes all insights with categories, confidence scores, and recommendations

10. **Step 10 - Configure Insights Settings**
    - Action: Click "Configure" button to open insights configuration dialog
    - Expected: Configuration dialog shows all insight categories and analysis settings
    - Verification: Switches for each category, confidence threshold slider, refresh interval settings

11. **Step 11 - Test Confidence Threshold Adjustment**
    - Action: Increase confidence threshold to 90%, save configuration
    - Expected: Lower confidence insights filtered out, only high-confidence insights remain
    - Verification: Insights panel shows fewer items, all with 90%+ confidence scores

12. **Step 12 - Test Category Filtering**
    - Action: Disable "Risk Analysis" category in configuration, refresh insights
    - Expected: Risk insights no longer appear in panel
    - Verification: Risk tab shows 0 count, no risk insights in "All Insights" view

### Expected Results

**Insights Generation Verification:**
- Risk analysis identifies budget, timeline, and resource concerns accurately
- Optimization insights provide actionable cost and performance improvement suggestions
- Compliance insights flag security and privacy requirements appropriately
- Predictive insights calculate delivery delays based on actual progress variance
- Recommendation insights suggest documentation and process improvements

**UI/UX Excellence:**
- Insights panel displays comprehensive dashboard with summary metrics
- Category tabs organize insights logically with appropriate counts
- Color-coded severity indicators (red=critical, yellow=warning, blue=info)
- Confidence scores and trend indicators provide context for decision-making
- Actionable recommendations clearly listed for each insight

**Technical Implementation Verification:**
- Auto-refresh updates insights every 30 seconds when enabled
- Real-time analysis reflects project data changes immediately
- Confidence threshold filtering works accurately
- Category enable/disable controls function properly
- Export functionality generates complete CSV with all insight data

**Analytics Quality Assessment:**
- Budget risk calculations accurate (utilization percentages, thresholds)
- Timeline analysis considers project complexity appropriately
- Team size recommendations align with industry best practices
- Progress variance predictions use statistical analysis
- Compliance checks cover relevant regulatory requirements

**Configuration and Personalization:**
- All insight categories can be individually enabled/disabled
- Confidence threshold adjustable from 50% to 99%
- Refresh interval configurable from 10 to 300 seconds
- Display settings control insight presentation and detail level
- Configuration persists across browser sessions

**Performance Requirements:**
- Initial insights generation completes within 2 seconds of panel opening
- Auto-refresh analysis executes within 1 second for responsive updates  
- Configuration changes apply immediately without requiring page refresh
- Export functionality processes and downloads within 3 seconds for typical datasets
- Panel remains responsive with up to 50+ insights across all categories

### Business Impact Validation
- **Risk Mitigation**: Early identification of budget, timeline, and resource risks
- **Decision Support**: Data-driven insights guide project planning and execution decisions
- **Proactive Management**: Predictive insights enable preventive action before issues escalate
- **Compliance Assurance**: Automated compliance checking reduces regulatory risks
- **Optimization Opportunities**: Cost and performance improvements identified automatically

### Advanced Analytics Features
- **Multi-dimensional Analysis**: Considers budget, timeline, resources, and progress simultaneously
- **Contextual Intelligence**: Insights adapt based on project type, size, and industry patterns
- **Confidence Scoring**: Machine learning confidence levels guide insight reliability
- **Trend Analysis**: Historical pattern recognition improves prediction accuracy
- **Comparative Analysis**: Benchmarking against similar projects for better recommendations

### Integration Testing
- **Real-time Data Integration**: Insights reflect current project state immediately
- **Configuration Persistence**: Settings maintained across browser sessions and users
- **Export Integration**: CSV export compatible with external analysis tools
- **Backend Analytics**: Insight generation leverages backend AI/ML services appropriately
- **User Preference Sync**: Configuration synchronized with user profile preferences

### Error Handling and Edge Cases
- **No Data Available**: Graceful handling when project data insufficient for analysis
- **Network Failures**: Robust error handling for backend analytics service unavailability
- **Large Datasets**: Efficient processing of projects with extensive historical data
- **Concurrent Updates**: Proper handling of simultaneous project data changes during analysis

### Post-conditions
- AI insights panel provides comprehensive, actionable intelligence for project decision-making
- Multi-category analysis covers all critical aspects of project management and risk assessment
- Configuration system allows personalized insight delivery based on user preferences and roles
- Export functionality enables integration with external reporting and analysis workflows
- Real-time updates ensure insights remain current and relevant throughout project lifecycle

---

## Test Case ID: TC-UI-AGT-078
**Test Objective**: Verify comprehensive learning feedback system and analytics dashboard functionality  
**Business Process**: AI Learning Analytics and Feedback Processing  
**SAP Module**: A2A Agents Developer Portal - Learning Analytics System  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-078
- **Test Priority**: High (P2)
- **Test Type**: Functional, AI Learning, Analytics
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectObjectPage.controller.js:4800-5500`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/fragments/LearningAnalyticsDialog.fragment.xml:1-400`
- **Functions Under Test**: `onViewLearningAnalytics()`, `_generateLearningAnalytics()`, `_analyzeRejectionPatterns()`, `_analyzeAcceptancePatterns()`

### Test Preconditions
1. **User Authentication**: Valid user session with AI suggestions enabled
2. **Learning Data**: Existing feedback data with accepted/rejected suggestions
3. **AI System**: Learning analytics engine functional and accessible
4. **UI State**: Project object page in view or edit mode
5. **Data Requirements**: Minimum 10 suggestion interactions for meaningful analytics

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Total Interactions | 25 | Number | Generated Feedback |
| Acceptance Rate | 68% | Percentage | Calculated Metric |
| User Type | balanced | String | Analytics Engine |
| Session Count | 5 | Number | Session Tracking |
| Learning Score | 72 | Number | Calculated Score |

### Test Procedure Steps

1. **Step 1 - Access Learning Analytics**
   - Action: Click "Learning Analytics" button in header toolbar
   - Expected: Learning Analytics Dashboard opens with comprehensive metrics display
   - Verification: Dialog shows current learning score, acceptance rate, total interactions, system confidence

2. **Step 2 - Overview Metrics Validation**
   - Action: Review key metrics cards in dashboard header
   - Expected: Four metric cards display learning score (72%), acceptance rate (68%), total interactions (25), system confidence (75%)
   - Verification: Metrics cards show appropriate color coding (green for good performance, red for poor performance)

3. **Step 3 - User Engagement Analysis**
   - Action: Navigate to "Overview" tab and examine user engagement panel
   - Expected: Panel shows engagement level (moderate/high), user type (balanced), applied suggestions count, session statistics
   - Verification: Engagement metrics align with user interaction patterns, session data displays correctly

4. **Step 4 - Learning Evolution Tracking**
   - Action: Expand "Learning Evolution" panel in overview tab
   - Expected: Evolution panel shows overall trend (improving/declining/stable), current performance percentage, improvement periods count
   - Verification: Trend analysis displays meaningful data points, performance tracking shows progression over time

5. **Step 5 - Rejection Pattern Analysis**
   - Action: Switch to "Rejection Analysis" tab
   - Expected: Detailed breakdown of rejection patterns including most rejected reason, problematic field, rejected type
   - Verification: Rejection data categorized by reason (not_relevant, incorrect, too_many), field analysis, time patterns

6. **Step 6 - Acceptance Pattern Analysis**
   - Action: Switch to "Acceptance Analysis" tab
   - Expected: Success patterns showing most accepted field, type, preferred time, confidence distribution
   - Verification: Acceptance data shows high/medium/low confidence breakdown, field success rates, optimal timing

7. **Step 7 - AI Improvement Recommendations**
   - Action: Navigate to "Recommendations" tab
   - Expected: List of actionable improvement recommendations with priority levels, categories, impact assessment
   - Verification: Recommendations based on rejection patterns, each shows title, description, priority (high/medium/low), impact level

8. **Step 8 - Learning Data Export**
   - Action: Click "Export Learning Data" button
   - Expected: CSV file downloads with comprehensive learning analytics report including metrics, patterns, recommendations
   - Verification: Downloaded file contains all analytics data in structured CSV format with headers and categories

9. **Step 9 - Learning Preferences Integration**
   - Action: Review how learning feedback affects user preferences and suggestion behavior
   - Expected: System adapts suggestion timing, frequency, types based on user feedback patterns
   - Verification: User preferences updated automatically, suggestion engine respects learned patterns

10. **Step 10 - Real-time Learning Updates**
    - Action: Generate new suggestion feedback and refresh analytics
    - Expected: Analytics dashboard updates with new interaction data, metrics recalculated
    - Verification: Learning score and acceptance rate update correctly, new patterns incorporated

11. **Step 11 - Reset Learning Data**
    - Action: Click "Reset Learning Data" button and confirm reset
    - Expected: Confirmation dialog appears, upon confirmation all learning data cleared, metrics reset to zero
    - Verification: All feedback arrays emptied, user preferences reset to defaults, analytics show clean state

12. **Step 12 - Empty State Handling**
    - Action: View learning analytics with no interaction data
    - Expected: Empty state messages displayed appropriately, no errors in calculations with zero data
    - Verification: Dashboard handles zero interactions gracefully, shows appropriate "insufficient data" messages

### Expected Results
- **Analytics Dashboard Criteria**:
  - Learning analytics dialog opens within 1 second
  - All metric calculations mathematically correct
  - Tab navigation smooth with no UI glitches
  - Real-time updates reflect current data state

- **Learning Intelligence Criteria**:
  - Rejection patterns accurately categorized and analyzed
  - Acceptance patterns provide actionable insights
  - Improvement recommendations relevant and prioritized
  - System confidence correlates with interaction quality

- **Data Export Criteria**:
  - CSV export contains all analytics data
  - File format properly structured with headers
  - Export completes within 3 seconds
  - Downloaded data matches dashboard display

- **Feedback Integration Criteria**:
  - Learning affects future suggestion behavior
  - User preferences automatically adjusted
  - Temporal patterns influence suggestion timing
  - Field-specific learning improves relevance

### Post-Conditions
- Learning analytics dashboard displays accurately
- Exported data available for external analysis
- User feedback patterns incorporated into AI behavior
- System learning continuously improves suggestion quality

### Error Handling
- **No Data Scenarios**: Graceful empty state display with helpful guidance
- **Calculation Errors**: Safe mathematical operations with division by zero protection
- **Export Failures**: User notification with retry option and error details
- **Performance Issues**: Efficient data processing with reasonable response times

---

---

## Test Case ID: TC-UI-AGT-079
**Test Objective**: Verify comprehensive column sorting functionality in projects list report table  
**Business Process**: List Report Data Sorting and Column Management  
**SAP Module**: A2A Agents Developer Portal - Projects List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-079
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface, Data Sorting
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:790-1015`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectsListReport.view.xml:248-350`
- **Functions Under Test**: `onColumnSort()`, `_applySingleColumnSort()`, `_priorityComparator()`, `_statusComparator()`

### Test Preconditions
1. **Data Availability**: Projects list populated with minimum 10 projects having varied data
2. **User Authentication**: Valid user session with list report access permissions
3. **Table State**: Projects table rendered in standard table view mode (not card view)
4. **Column Visibility**: All sortable columns visible and accessible
5. **Data Variety**: Projects with different statuses, priorities, dates, and names for meaningful sorting

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project Count | 15 | Number | Test Data Set |
| Status Varieties | ACTIVE, DRAFT, DEPLOYED, COMPLETED, ERROR | Array | Project Status Enum |
| Priority Levels | CRITICAL, HIGH, MEDIUM, LOW | Array | Priority Enum |
| Date Range | Last 6 months | DateRange | Test Data Generator |
| Name Patterns | Mixed alphanumeric | String | Test Project Names |

### Test Procedure Steps

1. **Step 1 - Default Sort Verification**
   - Action: Load Projects List Report page
   - Expected: Table displays projects sorted by "Modified Date" descending by default, sort indicator visible on Modified Date column
   - Verification: Most recently modified projects appear first, sort icon highlighted on correct column

2. **Step 2 - Name Column Ascending Sort**
   - Action: Click on "Project Name" column header
   - Expected: Table re-sorts alphabetically by project name (A-Z), sort indicator changes to ascending
   - Verification: Projects arranged alphabetically, first project starts with earliest letter, sort icon shows ascending direction

3. **Step 3 - Name Column Descending Sort**
   - Action: Click on "Project Name" column header again
   - Expected: Table sorts by project name in reverse alphabetical order (Z-A), sort indicator shows descending
   - Verification: Projects in reverse alphabetical order, sort direction toggled correctly

4. **Step 4 - Status Column Custom Sort**
   - Action: Click on "Status" column header
   - Expected: Table sorts by status using business logic order (ACTIVE → DEPLOYED → TESTING → DRAFT → PAUSED → COMPLETED → ARCHIVED → ERROR)
   - Verification: Projects grouped by status in business priority order, not simple alphabetical sorting

5. **Step 5 - Priority Column Custom Sort**
   - Action: Click on "Priority" column header
   - Expected: Table sorts by priority using business logic (CRITICAL → HIGH → MEDIUM → LOW), not alphabetical
   - Verification: Critical priority projects appear first, followed by high, medium, then low priority

6. **Step 6 - Date Column Chronological Sort**
   - Action: Click on "Last Deployment" column header
   - Expected: Table sorts by deployment date, most recent deployments first (descending default for dates)
   - Verification: Projects with recent deployments at top, null/empty deployment dates handled appropriately

7. **Step 7 - Created By Column Sort**
   - Action: Click on "Created By" column header
   - Expected: Table sorts alphabetically by creator name (ascending default for names)
   - Verification: Projects sorted by creator username/display name in alphabetical order

8. **Step 8 - Quick Sort Menu Access**
   - Action: Click on "Sort" menu button in toolbar
   - Expected: Dropdown menu opens showing quick sort options (Newest First, A-Z by Name, High Priority First, etc.)
   - Verification: Menu contains predefined sort options with appropriate icons and labels

9. **Step 9 - Quick Sort Application**
   - Action: Select "High Priority First" from sort menu
   - Expected: Table immediately sorts by priority descending, toast message confirms quick sort applied
   - Verification: High and critical priority projects move to top of list

10. **Step 10 - Sort Direction Indicators**
    - Action: Observe column headers after various sort operations
    - Expected: Currently sorted column shows appropriate sort direction indicator (up/down arrow or equivalent)
    - Verification: Visual indicators match actual sort direction, other columns show neutral/no sort indicator

11. **Step 11 - Sort Reset Functionality**
    - Action: Select "Reset Sorting" from sort menu
    - Expected: Table returns to default sort (Modified Date descending), toast message confirms reset
    - Verification: Sort indicators reset, table order matches initial default state

12. **Step 12 - Sort Persistence During Navigation**
    - Action: Apply custom sort, navigate to project detail, return to list report
    - Expected: Sort order maintained when returning to list, user's sort preference preserved during session
    - Verification: Table displays same sort order as before navigation

### Expected Results
- **Column Sorting Criteria**:
  - All sortable columns respond to header clicks
  - Sort direction toggles correctly (ascending ↔ descending)
  - Custom comparators work for status and priority fields
  - Date columns sort chronologically, not as text strings

- **Visual Feedback Criteria**:
  - Sort indicators visible and accurate on column headers
  - Sort direction clearly distinguished (up/down arrows)
  - Currently sorted column highlighted appropriately
  - Quick sort menu accessible and functional

- **Performance Criteria**:
  - Column sort completes within 500ms for up to 100 records
  - Visual indicators update immediately upon sort
  - No UI blocking during sort operations
  - Smooth user experience with appropriate feedback

- **Business Logic Criteria**:
  - Priority sorting follows business hierarchy (CRITICAL > HIGH > MEDIUM > LOW)
  - Status sorting reflects workflow precedence (ACTIVE > DEPLOYED > TESTING...)
  - Date fields handle null values appropriately (typically sorted last)
  - Text fields use proper collation and locale-specific sorting

### Post-Conditions
- Table data sorted according to user selection
- Sort state reflected in visual indicators
- Sort configuration available for session persistence
- Quick sort options reflect current capabilities

### Error Handling
- **Non-sortable Columns**: Appropriate message for columns without sort capability
- **Empty Data**: Graceful handling when no projects available for sorting
- **Data Type Errors**: Safe handling of unexpected data types in sort fields
- **Performance Issues**: User feedback for large dataset sorting operations

---

---

## Test Case ID: TC-UI-AGT-080
**Test Objective**: Verify advanced table grouping functionality for organizing projects by various criteria  
**Business Process**: List Report Data Grouping and Organization  
**SAP Module**: A2A Agents Developer Portal - Projects List Report Grouping  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-080
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface, Data Grouping
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:830-1450`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectsListReport.view.xml:296-331`
- **Functions Under Test**: `_applyGrouping()`, `_getDateRangeGrouper()`, `formatStatusGroup()`, `onCollapseAllGroups()`

### Test Preconditions
1. **Data Diversity**: Projects with various statuses, priorities, business units, and creation dates
2. **User Authentication**: Valid user session with full list report access
3. **Table State**: Projects table loaded with minimum 20 projects for meaningful grouping
4. **Browser Support**: Modern browser with proper table grouping support
5. **Initial State**: No grouping applied, standard table view active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project Count | 25+ | Number | Test Data Set |
| Status Distribution | All status types | Array | Status Enum |
| Priority Mix | All priority levels | Array | Priority Enum |
| Business Units | 3-5 different units | Array | Organization Data |
| Date Range | Past 6 months | DateRange | Project Creation Dates |

### Test Procedure Steps

1. **Step 1 - Group Options Menu Access**
   - Action: Click "Group" menu button in toolbar
   - Expected: Dropdown menu opens showing grouping options (Status, Priority, Business Unit, Date Range, plus control options)
   - Verification: Menu contains all expected grouping options with appropriate icons

2. **Step 2 - Group by Status**
   - Action: Select "Group by Status" from group menu
   - Expected: Table reorganizes with projects grouped by status (ACTIVE, DRAFT, DEPLOYED, etc.), group headers visible
   - Verification: Each status group shows header with status name and item count, projects correctly categorized

3. **Step 3 - Group Header Interaction**
   - Action: Click on any group header (e.g., "Active Projects (5)")
   - Expected: Group collapses, hiding contained projects; click again to expand
   - Verification: Smooth collapse/expand animation, group state persists during session

4. **Step 4 - Group by Priority with Custom Order**
   - Action: Select "Group by Priority" from group menu
   - Expected: Table groups by priority in business order (CRITICAL → HIGH → MEDIUM → LOW), not alphabetical
   - Verification: Critical priority group appears first regardless of alphabetical order

5. **Step 5 - Group by Business Unit**
   - Action: Select "Group by Business Unit" from menu
   - Expected: Projects grouped by their assigned business units, "Unassigned" group for projects without unit
   - Verification: Each business unit has its own group with count, proper handling of null/empty values

6. **Step 6 - Group by Date Range**
   - Action: Select "Group by Date Range" from menu
   - Expected: Projects grouped into date buckets (Today, Yesterday, This Week, This Month, Last 3 Months, Older)
   - Verification: Projects correctly categorized by creation date, most recent groups appear first

7. **Step 7 - Collapse All Groups**
   - Action: Select "Collapse All Groups" from group menu
   - Expected: All group headers remain visible but all project rows hidden, toast message confirms action
   - Verification: All groups collapsed simultaneously, can still see group headers with counts

8. **Step 8 - Expand All Groups**
   - Action: Select "Expand All Groups" from menu
   - Expected: All collapsed groups expand to show their projects, toast message confirms
   - Verification: All groups expanded, all projects visible again

9. **Step 9 - Combined Sorting and Grouping**
   - Action: Apply grouping by status, then sort by priority within groups
   - Expected: Projects grouped by status, and within each group sorted by priority
   - Verification: Grouping maintained while sorting affects order within groups

10. **Step 10 - Group Count Display**
    - Action: Observe group headers during various grouping operations
    - Expected: Each group header displays count of items in parentheses (e.g., "Active Projects (8)")
    - Verification: Counts accurate and update when projects added/removed from groups

11. **Step 11 - Remove Grouping**
    - Action: Select "Remove Grouping" from menu
    - Expected: Table returns to flat list view, previous sort order maintained, toast confirms removal
    - Verification: All group headers gone, projects displayed in single continuous list

12. **Step 12 - Group Persistence During Navigation**
    - Action: Apply grouping, navigate to project detail, return to list
    - Expected: Grouping configuration maintained, including collapsed/expanded states
    - Verification: Same grouping active upon return, user preferences preserved

### Expected Results
- **Grouping Application Criteria**:
  - All grouping options create proper hierarchical view
  - Group headers display with counts and appropriate formatting
  - Projects correctly assigned to groups based on criteria
  - Custom business logic applied (priority/status ordering)

- **Group Interaction Criteria**:
  - Group headers clickable for collapse/expand
  - Smooth animations for group state changes
  - Multiple groups can be collapsed independently
  - Collapse/expand all functions work correctly

- **Visual Presentation Criteria**:
  - Group headers visually distinct from data rows
  - Proper indentation for grouped items
  - Clear group boundaries and separators
  - Icon indicators for group state (collapsed/expanded)

- **Performance Criteria**:
  - Grouping applies within 1 second for up to 200 records
  - Collapse/expand animations smooth without lag
  - No performance degradation with multiple groups
  - Memory efficient for large grouped datasets

### Post-Conditions
- Table displays grouped or ungrouped based on user selection
- Group states (collapsed/expanded) maintained during session
- Grouping configuration available for quick reapplication
- Original data integrity maintained regardless of grouping

### Error Handling
- **Empty Groups**: Graceful handling when group criteria yields no results
- **Null Values**: Proper categorization of records with missing group fields
- **Large Groups**: Performance optimization for groups with many items
- **State Conflicts**: Resolution when grouping conflicts with filters or sorting

---

---

## Test Case ID: TC-UI-AGT-081
**Test Objective**: Verify table aggregation functionality for calculating sum, average, and count operations on project data  
**Business Process**: List Report Data Aggregation and Calculations  
**SAP Module**: A2A Agents Developer Portal - Projects List Report Aggregations  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-081
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface, Data Aggregation
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:1020-1317`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/ProjectsListReport.view.xml:335-340`
- **Functions Under Test**: `_calculateAggregations()`, `_performAggregation()`, `_formatAggregationResult()`, `onToggleAggregations()`

### Test Preconditions
1. **Data Requirements**: Projects with numeric fields (budget, agent count, success rate)
2. **User Authentication**: Valid user session with list report access
3. **Table State**: Projects table loaded with minimum 15 projects containing varied data
4. **Browser Support**: Modern browser supporting table footer calculations
5. **Initial State**: Aggregations disabled, standard table view active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Project Count | 20+ | Number | Test Data Set |
| Budget Range | $10K - $500K | Currency | Project Budgets |
| Agent Counts | 1-10 per project | Number | Agent Data |
| Success Rates | 0-100% | Percentage | Deployment Data |
| Null Values | Some projects | Mixed | Edge Case Testing |

### Test Procedure Steps

1. **Step 1 - Enable Aggregations**
   - Action: Click "Aggregations" button in toolbar
   - Expected: Aggregations activate with default calculations (Total Budget, Project Count), footer appears showing results
   - Verification: Footer displays with formatted aggregation values, button changes to emphasized state

2. **Step 2 - Total Budget Aggregation**
   - Action: Observe "Total Budget" aggregation in footer
   - Expected: Sum of all project budgets displayed in currency format (e.g., $2,345,678 USD)
   - Verification: Manual calculation matches displayed total, proper currency formatting with thousand separators

3. **Step 3 - Project Count Aggregation**
   - Action: Observe "Project Count" aggregation
   - Expected: Total number of projects in table displayed as integer
   - Verification: Count matches actual number of visible project rows (excluding any filtered items)

4. **Step 4 - Average Budget Calculation**
   - Action: Check if average budget calculation available (if implemented)
   - Expected: Average of all project budgets displayed in currency format
   - Verification: Average = Total Budget / Project Count, properly formatted

5. **Step 5 - Agent Count Aggregation**
   - Action: Observe total agents aggregation (if displayed)
   - Expected: Sum of all agents across all projects
   - Verification: Manual count of agents matches displayed total

6. **Step 6 - Success Rate Average**
   - Action: Check deployment success rate aggregation
   - Expected: Average success rate across all projects with deployments
   - Verification: Displayed as percentage (e.g., 87%), calculation excludes projects without deployments

7. **Step 7 - Aggregations with Filtering**
   - Action: Apply filter (e.g., status = ACTIVE), observe aggregation updates
   - Expected: Aggregations recalculate based only on filtered data
   - Verification: Totals and averages reflect only visible/filtered projects

8. **Step 8 - Aggregations with Grouping**
   - Action: Enable grouping by status, check if aggregations work with groups
   - Expected: Footer aggregations show totals across all groups, potential per-group aggregations
   - Verification: Group headers may show sub-aggregations, footer shows overall totals

9. **Step 9 - Null Value Handling**
   - Action: Include projects with null/empty budget values
   - Expected: Null values excluded from sum/average calculations, count still includes these projects
   - Verification: Calculations handle null gracefully without errors

10. **Step 10 - Real-time Updates**
    - Action: Change a project's budget value (if inline editing available)
    - Expected: Aggregations update immediately to reflect new values
    - Verification: Footer totals recalculate without manual refresh

11. **Step 11 - Disable Aggregations**
    - Action: Click "Aggregations" button again to toggle off
    - Expected: Footer disappears, button returns to normal state, toast confirms disabling
    - Verification: Table returns to standard view without footer

12. **Step 12 - Aggregation Persistence**
    - Action: Enable aggregations, navigate away and return
    - Expected: Aggregation state may persist based on user preferences
    - Verification: Check if aggregation settings maintained during session

### Expected Results
- **Calculation Accuracy Criteria**:
  - Sum aggregations mathematically correct
  - Averages calculated properly (sum/count)
  - Count operations accurate for visible records
  - Currency formatting with proper symbols and separators

- **Display Format Criteria**:
  - Currency values show with symbol and thousand separators
  - Percentages display with % symbol
  - Numbers formatted with appropriate precision
  - Icons and labels clearly identify each aggregation

- **Performance Criteria**:
  - Aggregations calculate within 500ms for up to 200 records
  - Real-time updates occur without lag
  - No performance impact on table scrolling/interaction
  - Memory efficient for large datasets

- **Integration Criteria**:
  - Works correctly with filtering
  - Compatible with grouping features
  - Updates when data changes
  - Handles all data types appropriately

### Post-Conditions
- Aggregation results accurate and properly formatted
- Footer visible when aggregations enabled
- Original data unchanged by aggregation calculations
- User preference for aggregation state may be saved

### Error Handling
- **Division by Zero**: Average calculations handle empty datasets gracefully
- **Null Values**: Proper exclusion from calculations without breaking functionality
- **Large Numbers**: Overflow handling for very large sums
- **Format Errors**: Graceful handling of non-numeric data in numeric fields

---

## Test Case ID: TC-UI-AGT-082
**Test Objective**: Verify chart toggle functionality in list report  
**Business Process**: Data Visualization and Analytics  
**SAP Module**: A2A Agents List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-082
- **Test Priority**: High (P2)
- **Test Type**: Functional, UI/UX
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/ProjectsListReport.view.xml:343-642`
- **Controller**: `a2a_agents/frontend/src/controller/ProjectsListReport.controller.js:1329-1856`
- **Functions Under Test**: `onToggleChartView()`, `_initializeCharts()`, `_updateChartData()`, `onExportChart()`

### Test Preconditions
1. **List Report Access**: User has access to Projects List Report
2. **Chart Libraries**: SAP Viz libraries loaded and available
3. **Data Available**: At least 10 projects with various statuses, priorities, and departments
4. **Browser Support**: Modern browser with SVG support
5. **Permissions**: User has view permissions for analytics

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Initial View | Table View | String | View State |
| Chart Types | pie, donut, bar, column, line | Array | Chart Config |
| Export Format | SVG | String | Export Config |
| Chart Tabs | status, priority, timeline, department | Array | Tab Config |
| Animation | Enabled | Boolean | User Preference |

### Test Procedure Steps
1. **Step 1 - Initial Chart Toggle Button State**
   - Action: Load Projects List Report in table view
   - Expected: Chart toggle button visible with default state
   - Verification: Button text shows "Charts", icon is business-objects-experience

2. **Step 2 - Toggle to Chart View**
   - Action: Click "Charts" button in toolbar
   - Expected: Table hides, chart panel appears with smooth transition
   - Verification: Button becomes emphasized, tooltip changes to "Hide chart visualization"

3. **Step 3 - Chart Initialization**
   - Action: Observe chart loading on first toggle
   - Expected: Four chart tabs appear with loading indicators
   - Verification: Status distribution chart loads as default selection

4. **Step 4 - Chart Data Rendering**
   - Action: Verify data visualization in status chart
   - Expected: Pie chart shows project distribution by status with colors
   - Verification: Legend shows all statuses, percentages visible on slices

5. **Step 5 - Tab Navigation**
   - Action: Click through Priority, Timeline, Department tabs
   - Expected: Each tab shows appropriate chart type with data
   - Verification: 
     - Priority: Bar chart with priority levels
     - Timeline: Line chart with monthly trends
     - Department: Donut chart with department distribution

6. **Step 6 - Chart Type Change**
   - Action: Select different chart type from dropdown (e.g., change pie to donut)
   - Expected: Chart morphs to new type with animation
   - Verification: Data labels and legend adjust to new chart type

7. **Step 7 - Chart Refresh**
   - Action: Click "Refresh Charts" button
   - Expected: Charts reload with latest data from table
   - Verification: Toast message "Charts refreshed" appears

8. **Step 8 - Chart Export**
   - Action: Click "Export Chart" button
   - Expected: Current chart downloads as SVG file
   - Verification: File downloads with timestamp in filename

9. **Step 9 - Toggle Back to Table**
   - Action: Click "Charts" button again
   - Expected: Chart panel hides, table reappears
   - Verification: Button returns to default state, table data intact

10. **Step 10 - Performance Check**
    - Action: Toggle rapidly between views
    - Expected: Smooth transitions without lag
    - Verification: No memory leaks, charts disposed properly

11. **Step 11 - Filter Integration**
    - Action: Apply filters then toggle to chart view
    - Expected: Charts reflect filtered data only
    - Verification: Chart totals match filtered record count

12. **Step 12 - Responsive Behavior**
    - Action: Resize browser window while in chart view
    - Expected: Charts resize responsively
    - Verification: Charts maintain aspect ratio and readability

### Expected Results
- **Chart Toggle Functionality**:
  - Toggle button switches between table and chart views seamlessly
  - Chart initialization occurs only once per session
  - All four chart types render correctly with proper data
  
- **Chart Interactions**:
  - Tab navigation works smoothly between chart types
  - Chart type changes apply instantly with animations
  - Data labels and legends display accurately
  - Hover tooltips show detailed information
  
- **Data Accuracy**:
  - Chart data matches table data exactly
  - Filters apply to both table and chart views
  - Aggregations calculate correctly for each dimension
  - Timeline shows chronological progression
  
- **Export Functionality**:
  - SVG export captures current chart state
  - Downloaded file opens correctly in image viewers
  - Filename includes chart type and timestamp

### Test Data Requirements
```javascript
// Sample projects for chart visualization
projects: [
    { status: "Active", priority: "High", department: "IT", createdDate: "2024-01-15" },
    { status: "Completed", priority: "Medium", department: "HR", createdDate: "2024-01-20" },
    { status: "Active", priority: "Critical", department: "IT", createdDate: "2024-02-01" },
    { status: "Failed", priority: "Low", department: "Finance", createdDate: "2024-02-15" },
    { status: "Active", priority: "High", department: "IT", createdDate: "2024-03-01" }
    // ... more test data
]
```

### Validation Rules
1. **Chart Rendering**: All charts must render within 2 seconds
2. **Data Integrity**: Chart totals must equal table row count
3. **Export Quality**: SVG exports must be at least 800x600 pixels
4. **Memory Management**: Charts must dispose properly on view change
5. **Accessibility**: Charts must have proper ARIA labels

### Integration Points
- **Smart Table**: Chart data sourced from table binding
- **Filter Bar**: Charts respect active filters
- **View Model**: Chart state persisted in view model
- **SAP Viz**: Utilizes SAP visualization framework

### Performance Criteria
- Chart initial load: < 2 seconds
- Chart type change: < 500ms
- Tab switch: < 300ms
- Export generation: < 1 second
- Memory usage: < 50MB increase

---

## Test Case ID: TC-UI-AGT-083
**Test Objective**: Test print preview functionality in list report  
**Business Process**: Report Printing and Export  
**SAP Module**: A2A Agents List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-083
- **Test Priority**: High (P2)
- **Test Type**: Functional, UI/UX
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/ProjectsListReport.view.xml:353-357`
- **Controller**: `a2a_agents/frontend/src/controller/ProjectsListReport.controller.js:1865-2597`
- **Functions Under Test**: `onPrintPreview()`, `_generatePrintHTML()`, `onPrint()`, `_updatePrintPreview()`

### Test Preconditions
1. **List Report Access**: User has access to Projects List Report
2. **Browser Support**: Modern browser with print capabilities
3. **Data Available**: At least 50 projects for pagination testing
4. **Print Settings**: Browser allows popup windows for print
5. **Screen Resolution**: Minimum 1280x720 for preview display

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Paper Sizes | A4, Letter, Legal, A3 | Array | Print Settings |
| Orientations | Portrait, Landscape | Array | Print Settings |
| Margins | Normal, Narrow, Wide, None | Array | Print Settings |
| Page Size | 25 (A4 Portrait) | Integer | Calculated |
| Scale Options | 50%, 75%, 100%, 125% | Array | Preview Settings |

### Test Procedure Steps
1. **Step 1 - Access Print Preview**
   - Action: Click "Print Preview" button in toolbar
   - Expected: Print preview dialog opens with loading indicator
   - Verification: Dialog shows print settings panel and preview area

2. **Step 2 - Default Settings Display**
   - Action: Observe initial print preview settings
   - Expected: A4, Portrait, Normal margins, all content included
   - Verification: Preview shows page 1 with header, KPIs, and table

3. **Step 3 - Paper Size Change**
   - Action: Change paper size from A4 to Letter
   - Expected: Preview updates with new page dimensions
   - Verification: Page size adjusts, item count per page changes

4. **Step 4 - Orientation Toggle**
   - Action: Switch from Portrait to Landscape
   - Expected: Preview rotates, more columns visible
   - Verification: Page width increases, fewer rows per page

5. **Step 5 - Margin Adjustment**
   - Action: Change margins from Normal to Narrow
   - Expected: Content area expands, more items fit
   - Verification: White space reduces, page count updates

6. **Step 6 - Content Selection**
   - Action: Uncheck "KPI Cards" and "Filters Summary"
   - Expected: Preview updates removing unchecked sections
   - Verification: Only header and table remain in preview

7. **Step 7 - Column Selection**
   - Action: Uncheck "Agents" and "Last Deployment" columns
   - Expected: Table in preview shows only selected columns
   - Verification: Table width adjusts, columns redistribute

8. **Step 8 - Page Navigation**
   - Action: Click next page button
   - Expected: Preview shows page 2 content
   - Verification: Different table rows visible, page indicator updates

9. **Step 9 - Zoom Controls**
   - Action: Click zoom in/out buttons and scale selector
   - Expected: Preview scales smoothly
   - Verification: Content remains readable at all zoom levels

10. **Step 10 - Print Range Selection**
    - Action: Select "Custom Range" and enter "1-3, 5"
    - Expected: Range input accepts page specification
    - Verification: Print will include only specified pages

11. **Step 11 - Print Execution**
    - Action: Click "Print" button
    - Expected: Browser print dialog opens with preview content
    - Verification: Print preview matches dialog preview

12. **Step 12 - PDF Export**
    - Action: Click "Export PDF" button
    - Expected: Information message about PDF implementation
    - Verification: Message explains PDF export requirements

### Expected Results
- **Print Preview Dialog**:
  - Opens quickly with responsive layout
  - Settings panel collapsible for more preview space
  - All settings immediately affect preview
  
- **Preview Rendering**:
  - Accurate page layout matching print output
  - Proper pagination based on paper size
  - Headers/footers positioned correctly
  - Table breaks cleanly between pages
  
- **Settings Functionality**:
  - Paper size affects page dimensions accurately
  - Orientation properly rotates content
  - Margins adjust content area as expected
  - Column selection dynamically updates table
  
- **Navigation Controls**:
  - Page buttons enable/disable appropriately
  - Current page indicator always accurate
  - Jump to first/last page works instantly
  - Preview updates smoothly on navigation

### Test Data Requirements
```javascript
// Print settings configuration
printSettings: {
    paperSizes: {
        A4: { width: 210, height: 297, unit: "mm" },
        Letter: { width: 8.5, height: 11, unit: "in" },
        Legal: { width: 8.5, height: 14, unit: "in" },
        A3: { width: 297, height: 420, unit: "mm" }
    },
    pageSizes: {
        A4: { portrait: 25, landscape: 15 },
        Letter: { portrait: 22, landscape: 14 }
    }
}
```

### Validation Rules
1. **Page Calculation**: Total pages = ceil(total items / page size)
2. **Content Fitting**: No content overflow outside page boundaries
3. **Print Accuracy**: Printed output matches preview exactly
4. **Scale Limits**: Zoom range 25% to 200%
5. **Performance**: Preview updates within 500ms

### Integration Points
- **Table Data**: Pulls current table items and filters
- **KPI Model**: Reads KPI values from view model
- **Filter Bar**: Extracts active filter conditions
- **Browser Print**: Integrates with window.print() API

### Performance Criteria
- Dialog open: < 1 second
- Preview update: < 500ms
- Page navigation: < 300ms
- Print window open: < 2 seconds
- Settings change response: < 200ms

### Accessibility Requirements
- Dialog fully keyboard navigable
- Screen reader announces page changes
- Focus management on dialog open/close
- High contrast mode supported
- Print output readable without color

### Error Handling
- **Popup Blocked**: Show instructions to allow popups
- **No Data**: Display message when table empty
- **Large Dataset**: Warn if > 1000 items
- **Print Failure**: Graceful error with retry option

---

## Test Case ID: TC-UI-AGT-084
**Test Objective**: Test export Excel functionality in list report  
**Business Process**: Data Export and Reporting  
**SAP Module**: A2A Agents List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-084
- **Test Priority**: High (P2)
- **Test Type**: Functional, Data Export
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/ProjectsListReport.view.xml:360-387`
- **Controller**: `a2a_agents/frontend/src/controller/ProjectsListReport.controller.js:2606-3045`
- **Functions Under Test**: `onExportExcel()`, `_getExportColumns()`, `_getExportData()`, `_prepareExportItem()`

### Test Preconditions
1. **List Report Access**: User has access to Projects List Report
2. **Export Library**: SAP UI5 export library loaded
3. **Data Available**: At least 100 projects for testing various scenarios
4. **Browser Support**: Modern browser with file download capability
5. **Excel Software**: Microsoft Excel or compatible viewer for verification

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Export Formats | Excel (.xlsx), CSV (.csv) | Array | Export Menu |
| Export Scopes | Visible, Selected, Filtered, All | Array | Export Options |
| Column Sets | Basic (10 cols), Extended (16 cols) | Array | Column Config |
| File Naming | Projects_Export_YYYY-MM-DD | String | Auto-generated |
| Max Records | 10000 | Integer | Performance Limit |

### Test Procedure Steps
1. **Step 1 - Access Export Menu**
   - Action: Click "Export" dropdown button in toolbar
   - Expected: Dropdown menu shows all export options
   - Verification: Six menu items visible with appropriate icons

2. **Step 2 - Basic Excel Export**
   - Action: Click "Export to Excel" menu item
   - Expected: Excel file downloads with current table data
   - Verification: File name includes today's date

3. **Step 3 - Verify Excel Content**
   - Action: Open downloaded Excel file
   - Expected: All visible table rows exported with headers
   - Verification: Data matches table display, formatting preserved

4. **Step 4 - Export Selected Items**
   - Action: Select 5 items, then click "Export Selected"
   - Expected: Only selected items included in export
   - Verification: Excel contains exactly 5 data rows plus header

5. **Step 5 - Export with Filters**
   - Action: Apply status filter, click "Export Filtered"
   - Expected: Export respects active filters
   - Verification: All exported items match filter criteria

6. **Step 6 - Export All Data**
   - Action: Click "Export All" and confirm dialog
   - Expected: Confirmation dialog, then full export with progress
   - Verification: Excel includes all database records

7. **Step 7 - Column Configuration**
   - Action: Export all to see extended columns
   - Expected: Additional columns included (timestamps, audit fields)
   - Verification: 16 columns total vs 10 in basic export

8. **Step 8 - Large Dataset Export**
   - Action: Export 1000+ items
   - Expected: Progress dialog shows, web worker processes
   - Verification: Export completes without browser freeze

9. **Step 9 - CSV Export**
   - Action: Click "Export to CSV"
   - Expected: CSV file downloads with proper formatting
   - Verification: Opens correctly in Excel, quotes handled

10. **Step 10 - Export Error Handling**
    - Action: Simulate export failure (disconnect network)
    - Expected: Error message displayed
    - Verification: Clear error description, no partial downloads

11. **Step 11 - Date/Number Formatting**
    - Action: Export and check date/number columns
    - Expected: Dates formatted as dd.mm.yyyy, numbers with decimals
    - Verification: Excel recognizes data types correctly

12. **Step 12 - Special Characters**
    - Action: Export data with special characters (", ', &, <, >)
    - Expected: All characters preserved and escaped properly
    - Verification: No data corruption in Excel

### Expected Results
- **Export Menu Functionality**:
  - All export options clearly labeled with icons
  - Menu items enable/disable based on context
  - Selected count shown when applicable
  
- **Excel Export Quality**:
  - Professional formatting with column headers
  - Appropriate column widths set
  - Data types preserved (dates, numbers, text)
  - No truncation or data loss
  
- **Performance Standards**:
  - Export starts within 1 second
  - Progress shown for large exports
  - Web worker prevents UI blocking
  - Memory efficient for large datasets
  
- **File Management**:
  - Automatic file naming with date
  - Proper file extension (.xlsx, .csv)
  - Browser download handling
  - No temporary files left behind

### Test Data Requirements
```javascript
// Export configuration
exportConfig: {
    columns: {
        basic: ["name", "description", "status", "priority", "department", 
               "projectManager", "startDate", "endDate", "budget", "currency"],
        extended: [...basic, "agentCount", "lastDeployment", "createdBy", 
                  "createdDate", "modifiedBy", "modifiedDate"]
    },
    formats: {
        date: "dd.mm.yyyy",
        dateTime: "dd.mm.yyyy hh:mm",
        number: "#,##0.00"
    },
    limits: {
        maxRows: 10000,
        maxFileSize: "50MB"
    }
}
```

### Validation Rules
1. **Data Integrity**: Exported data matches source exactly
2. **Format Preservation**: Numbers/dates maintain format
3. **Character Encoding**: UTF-8 support for international chars
4. **File Validity**: Generated files open without errors
5. **Performance**: Export completes within reasonable time

### Integration Points
- **SAP UI5 Export Library**: Uses sap.ui.export.Spreadsheet
- **Web Workers**: Offloads processing for performance
- **Table Binding**: Reads current table state
- **Filter Framework**: Respects active filters
- **Selection Model**: Tracks selected items

### Performance Criteria
- Small export (< 100 rows): < 1 second
- Medium export (100-1000 rows): < 3 seconds
- Large export (1000-5000 rows): < 10 seconds
- Progress dialog appears: > 500 rows
- Memory usage: < 200MB for 5000 rows

### Accessibility Requirements
- Export menu keyboard accessible
- Screen reader announces export progress
- Confirmation dialogs focus managed
- Error messages announced
- Alternative formats available (CSV)

### Error Scenarios
- **No Selection**: Warning when exporting selected with none
- **Empty Table**: Appropriate message for empty exports
- **Browser Block**: Instructions if download blocked
- **Memory Limit**: Graceful handling of very large exports
- **Network Error**: Clear error message with retry option

---

## Test Case ID: TC-UI-AGT-085
**Test Objective**: Verify export PDF functionality in list report  
**Business Process**: PDF Document Generation and Export  
**SAP Module**: A2A Agents List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-085
- **Test Priority**: High (P2)
- **Test Type**: Functional, Document Generation
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/frontend/src/view/fragments/PDFExportDialog.fragment.xml`
- **Controller**: `a2a_agents/frontend/src/controller/ProjectsListReport.controller.js:2636-3251`
- **Functions Under Test**: `onExportPDF()`, `_generatePDF()`, `_generatePDFHTML()`, `onExecutePDFExport()`

### Test Preconditions
1. **List Report Access**: User has access to Projects List Report
2. **PDF Library**: jsPDF library available or HTML5 print support
3. **Data Available**: At least 50 projects for comprehensive PDF
4. **Browser Support**: Modern browser with PDF capabilities
5. **Screen Resolution**: Minimum 1024x768 for dialog display

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------| 
| Page Sizes | A4, Letter, Legal, A3 | Array | PDF Settings |
| Orientations | Portrait, Landscape | Array | PDF Settings |
| Security Options | Encrypt, Allow Print, Allow Copy | Object | Advanced Settings |
| Content Sections | Cover, TOC, KPIs, Table, Charts | Array | Content Selection |
| Watermark | CONFIDENTIAL, DRAFT, FINAL | String | Advanced Options |

### Test Procedure Steps
1. **Step 1 - Open PDF Export Dialog**
   - Action: Click "Export to PDF" from export menu
   - Expected: PDF export settings dialog opens
   - Verification: All form fields populated with defaults

2. **Step 2 - Configure Document Properties**
   - Action: Enter custom title, author, subject, keywords
   - Expected: Fields accept text input
   - Verification: No character limit errors for reasonable inputs

3. **Step 3 - Select Page Settings**
   - Action: Change page size to Letter, orientation to Landscape
   - Expected: Settings update immediately
   - Verification: Preview reflects new dimensions

4. **Step 4 - Configure Content Sections**
   - Action: Check/uncheck various content options
   - Expected: Each option toggles independently
   - Verification: Related sub-options enable/disable appropriately

5. **Step 5 - Set Security Options**
   - Action: Enable encryption, disable copying
   - Expected: Security checkboxes respond correctly
   - Verification: Permission options only available when encrypted

6. **Step 6 - Add Watermark**
   - Action: Enable watermark, enter "CONFIDENTIAL"
   - Expected: Text field enables, accepts custom text
   - Verification: Preview shows watermark placement

7. **Step 7 - Execute PDF Generation**
   - Action: Click "Export PDF" button
   - Expected: Progress panel appears with status updates
   - Verification: Progress bar advances through stages

8. **Step 8 - Monitor Export Progress**
   - Action: Observe progress indicators
   - Expected: Status messages update at each stage
   - Verification: Percentage increases smoothly to 100%

9. **Step 9 - PDF Download**
   - Action: Wait for generation to complete
   - Expected: PDF file downloads automatically
   - Verification: File name matches document title

10. **Step 10 - Verify PDF Content**
    - Action: Open generated PDF file
    - Expected: All selected sections included
    - Verification: Cover page, TOC, data all present and formatted

11. **Step 11 - Preview Function**
    - Action: Click "Preview" button before export
    - Expected: Browser window opens with HTML preview
    - Verification: Preview matches final PDF layout

12. **Step 12 - Fallback HTML Export**
    - Action: Test when jsPDF unavailable
    - Expected: Falls back to print dialog method
    - Verification: Instructions shown for "Save as PDF"

### Expected Results
- **Dialog Functionality**:
  - All settings persist during session
  - Form validation prevents invalid configurations
  - Preview accurately represents final output
  
- **PDF Generation Process**:
  - Progress tracking smooth and accurate
  - Each stage completes without errors
  - Export time reasonable for data size
  - Memory usage stays within limits
  
- **Document Quality**:
  - Professional appearance with proper formatting
  - All sections render correctly
  - Page breaks positioned appropriately
  - Fonts embedded and readable
  
- **Advanced Features**:
  - Watermark appears on all pages
  - Security settings applied correctly
  - Compression reduces file size
  - Metadata properly embedded

### Test Data Requirements
```javascript
// PDF configuration options
pdfConfig: {
    metadata: {
        title: "Projects Report Q4 2024",
        author: "John Doe",
        subject: "Quarterly Project Analysis",
        keywords: "projects, Q4, analysis, agents"
    },
    layout: {
        pageSize: ["A4", "Letter", "Legal", "A3"],
        orientation: ["portrait", "landscape"],
        margins: ["normal", "narrow", "wide"]
    },
    content: {
        sections: ["cover", "toc", "kpis", "filters", "table", "charts"],
        watermark: ["CONFIDENTIAL", "DRAFT", "INTERNAL USE"]
    },
    security: {
        encryption: true,
        permissions: ["print", "copy", "modify"]
    }
}
```

### Validation Rules
1. **Title Required**: Document must have a title
2. **Page Size Valid**: Must be standard paper size
3. **Content Selection**: At least one section required
4. **File Size Limit**: PDF should not exceed 50MB
5. **Generation Time**: Complete within 30 seconds

### Integration Points
- **jsPDF Library**: Primary PDF generation engine
- **HTML5 Print**: Fallback for PDF creation
- **Table Data**: Pulls from current view state
- **KPI Model**: Reads dashboard metrics
- **User Model**: Gets author information

### Performance Criteria
- Dialog open: < 500ms
- Preview generation: < 2 seconds
- PDF generation start: < 1 second
- Small PDF (< 10 pages): < 5 seconds
- Large PDF (50+ pages): < 30 seconds

### Security Testing
- **Encryption**: Verify PDF requires password when encrypted
- **Permissions**: Test print/copy restrictions work
- **Watermark**: Ensure cannot be easily removed
- **Metadata**: Check no sensitive data exposed
- **File Access**: Confirm proper download permissions

### Browser Compatibility
- Chrome/Edge: Full jsPDF support
- Firefox: Full jsPDF support
- Safari: May require print dialog method
- IE11: Fallback to HTML method only

### Error Handling
- **Library Load Failure**: Graceful fallback to HTML
- **Generation Failure**: Clear error message
- **Download Blocked**: Instructions for user
- **Memory Exceeded**: Warning before attempting
- **Invalid Settings**: Validation messages

---

## Test Case ID: TC-UI-AGT-086
**Test Objective**: Verify CSV export functionality with advanced formatting options  
**Business Process**: Data Export and Formatting  
**SAP Module**: A2A Agents Developer Portal - List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-086
- **Test Priority**: High (P2)
- **Test Type**: Functional, Data Export
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:3256-3603`
- **Dialog Fragment**: `a2a_agents/backend/app/a2a/developerPortal/static/view/fragments/CSVExportDialog.fragment.xml`
- **Functions Under Test**: `onExportCSV()`, `onExecuteCSVExport()`, `_generateEnhancedCSV()`, `onCSVPreview()`

### Test Preconditions
1. **System State**: Projects list report view loaded with data
2. **User Access**: Read permissions on project data
3. **Browser Requirements**: Support for Blob API and downloads
4. **Data**: Multiple projects with various statuses and metadata
5. **UI State**: Projects table populated with at least 10 records

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Delimiter | Semicolon (;) | Selection | User Input |
| Text Qualifier | Double quotes (") | Selection | User Input |
| Encoding | UTF-8 with BOM | Selection | User Input |
| Date Format | EU (DD/MM/YYYY) | Selection | User Input |
| Include Headers | true | Checkbox | User Input |
| Row Limit | 100 | Number | User Input |

### Test Procedure Steps
1. **Step 1 - Open CSV Export Dialog**
   - Action: Click "Export" menu button and select "Export to CSV"
   - Expected: CSV Export Options dialog opens with default settings
   - Verification: All form controls are visible and properly initialized

2. **Step 2 - Configure Basic Settings**
   - Action: Change file name to "test_export_2024"
   - Expected: File name preview updates to "test_export_2024.csv"
   - Verification: Invalid characters are automatically replaced with underscores

3. **Step 3 - Select Delimiter Options**
   - Action: Select "Semicolon (;) - Excel friendly" as delimiter
   - Expected: Radio button selection changes to semicolon option
   - Verification: Preview reflects semicolon-separated format

4. **Step 4 - Configure Text Qualifier**
   - Action: Select "Single quotes (')" as text qualifier
   - Expected: Text qualifier radio button updates
   - Verification: Preview shows values wrapped in single quotes

5. **Step 5 - Set Character Encoding**
   - Action: Select "UTF-8 with BOM (Excel)" from encoding dropdown
   - Expected: Encoding selection updates to UTF-8-BOM
   - Verification: Export will include byte order mark for Excel

6. **Step 6 - Configure Content Options**
   - Action: Enable "Include row numbers" and "Export formatted values"
   - Expected: Checkboxes are selected
   - Verification: Preview shows row numbers and formatted values

7. **Step 7 - Select Data Range**
   - Action: Choose "Export filtered data" and set row limit to 50
   - Expected: Data range selection updates, row limit field accepts input
   - Verification: Only filtered records up to limit will be exported

8. **Step 8 - Configure Date Format**
   - Action: Select "EU Format (DD/MM/YYYY)" from date format dropdown
   - Expected: Date format selection updates
   - Verification: Dates in preview show DD/MM/YYYY format

9. **Step 9 - Preview Export**
   - Action: Click "Preview" button
   - Expected: Preview dialog shows first 10 rows with selected formatting
   - Verification: All formatting options are correctly applied in preview

10. **Step 10 - Execute Export**
    - Action: Click "Export" button
    - Expected: CSV file downloads with configured settings
    - Verification: File contains correctly formatted data

11. **Step 11 - Verify Metadata Inclusion**
    - Action: Enable "Include export metadata" and export again
    - Expected: CSV includes header rows with export date and record count
    - Verification: Metadata appears before column headers

12. **Step 12 - Test Summary Row**
    - Action: Enable "Add summary row" option and export
    - Expected: CSV includes summary row with totals
    - Verification: Numeric columns show sum, text shows record count

### Expected Results
- **Dialog Functionality**:
  - CSV export dialog opens with all options visible
  - File name validation works correctly
  - All delimiter options function properly
  - Text qualifier options apply correctly
  
- **Formatting Options**:
  - Character encoding applies correctly (BOM for Excel)
  - Line endings respect platform selection
  - Date formatting follows selected pattern
  - Special characters are properly escaped
  
- **Content Selection**:
  - Row numbers appear when enabled
  - Hidden columns excluded when unchecked
  - Formatted values used when selected
  - Data range filtering works correctly
  
- **Export Quality**:
  - Generated CSV opens correctly in Excel/LibreOffice
  - No data corruption or encoding issues
  - Delimiters and qualifiers consistent throughout
  - Summary rows calculate correctly

### Validation Criteria
- **File Validation**: Downloaded file has .csv extension
- **Content Validation**: Data matches source with applied formatting
- **Encoding Validation**: Special characters display correctly
- **Excel Compatibility**: File opens without import wizard in Excel
- **Performance**: Export completes within 5 seconds for 1000 rows

### Browser Compatibility
- Chrome/Edge: Full Blob API support
- Firefox: Full Blob API support  
- Safari: May require different download method
- IE11: Limited encoding options

### Error Handling
- **No Data**: Warning when no data to export
- **Invalid Settings**: Validation prevents export
- **Download Blocked**: Instructions for enabling downloads
- **Large Dataset**: Warning for exports over 10,000 rows
- **Encoding Issues**: Fallback to UTF-8 if encoding fails

---

## Test Case ID: TC-UI-AGT-087
**Test Objective**: Verify data integrity check functionality before export operations  
**Business Process**: Data Quality Assurance and Export Validation  
**SAP Module**: A2A Agents Developer Portal - List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-087
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Data Validation
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:3939-4241`
- **Dialog Fragment**: `a2a_agents/backend/app/a2a/developerPortal/static/view/fragments/DataIntegrityDialog.fragment.xml`
- **Functions Under Test**: `onVerifyDataIntegrity()`, `_performIntegrityCheck()`, `_checkRecordIntegrity()`, `onSaveIntegrityReport()`

### Test Preconditions
1. **System State**: Projects list report view loaded with diverse data
2. **User Access**: Read permissions on project data
3. **Data Conditions**: Mix of complete, incomplete, and malformed records
4. **Browser Requirements**: Support for async processing and downloads
5. **Test Data**: At least 100 records with various data quality issues

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Required Fields | name, status, priority | Array | Column Config |
| Number Fields | budget, agentCount | Array | Column Config |
| Date Fields | startDate, lastDeployment | Array | Column Config |
| Max Lengths | name: 50, description: 200 | Object | Validation Rules |
| Chunk Size | 100 | Number | Performance Config |

### Test Procedure Steps
1. **Step 1 - Initiate Integrity Check**
   - Action: Click "Verify Data" button in toolbar
   - Expected: Data Integrity Check dialog opens with progress indicator
   - Verification: Progress bar shows 0% with total record count

2. **Step 2 - Monitor Check Progress**
   - Action: Observe progress indicator during check
   - Expected: Progress updates in real-time showing "X / Y records"
   - Verification: Progress bar smoothly increases without freezing

3. **Step 3 - Verify Chunk Processing**
   - Action: Monitor browser performance during check
   - Expected: UI remains responsive, no browser warnings
   - Verification: Chunks process with 10ms delays between batches

4. **Step 4 - Review Error Detection**
   - Action: Wait for check completion and review errors panel
   - Expected: Errors panel shows missing required fields
   - Verification: Each error shows row number, column, type, and value

5. **Step 5 - Review Warning Detection**
   - Action: Expand warnings panel if present
   - Expected: Warnings show data type mismatches and truncation risks
   - Verification: Invalid dates, numbers, and oversized text flagged

6. **Step 6 - Verify Checksum Generation**
   - Action: Note the displayed data checksum value
   - Expected: Checksum displays as hexadecimal string
   - Verification: Same data produces consistent checksum

7. **Step 7 - Test Export with Errors**
   - Action: Try to export when errors exist
   - Expected: Export buttons disabled, error message shown
   - Verification: Cannot proceed until errors resolved

8. **Step 8 - Test Export with Warnings**
   - Action: Click export button with only warnings present
   - Expected: Confirmation dialog asks to proceed despite warnings
   - Verification: Can choose to continue or cancel export

9. **Step 9 - Save Integrity Report**
   - Action: Click "Save Report" button
   - Expected: Text file downloads with full integrity report
   - Verification: Report contains all errors, warnings, and metadata

10. **Step 10 - Test Validated Export**
    - Action: Enable "Export only validated records" and export
    - Expected: Export excludes records with errors
    - Verification: Exported file contains only clean records

11. **Step 11 - Test Report Inclusion**
    - Action: Enable "Include integrity report in export"
    - Expected: Export includes separate sheet/section with report
    - Verification: Integrity data appears in exported file

12. **Step 12 - Performance with Large Dataset**
    - Action: Run integrity check on 1000+ records
    - Expected: Check completes within reasonable time
    - Verification: No timeout errors or memory issues

### Expected Results
- **Progress Tracking**:
  - Real-time progress updates during check
  - UI remains responsive throughout process
  - Clear indication when check is complete
  - Accurate record count and percentage
  
- **Error Detection**:
  - Missing required fields correctly identified
  - Row and column information accurate
  - Error types clearly categorized
  - No false positives for valid data
  
- **Warning Detection**:
  - Data type mismatches properly flagged
  - Truncation risks identified for long values
  - Invalid date/number formats caught
  - Severity appropriately assigned
  
- **Checksum Validation**:
  - Consistent checksum for same dataset
  - Checksum updates when data changes
  - Hexadecimal format properly displayed
  - Included in integrity report

### Validation Criteria
- **Accuracy**: All data issues correctly identified
- **Performance**: Processes 1000 records in < 10 seconds
- **Completeness**: No valid issues missed by checker
- **Usability**: Clear error messages and guidance
- **Reliability**: Consistent results across runs

### Error Categories
- **Missing Required Field**: Critical error preventing export
- **Invalid Number Format**: Warning for non-numeric values
- **Invalid Date Format**: Warning for unparseable dates
- **Data Truncation Risk**: Warning when exceeding max length
- **Null Reference**: Error for required relationships

### Performance Benchmarks
- 100 records: < 1 second
- 1,000 records: < 10 seconds
- 10,000 records: < 60 seconds
- Progress updates: Every 100ms minimum

### Error Handling
- **Check Failure**: Clear error message with retry option
- **Memory Issues**: Warning before processing huge datasets
- **Export Blocked**: Explanation of why export is disabled
- **Report Generation**: Fallback if download fails
- **Browser Limits**: Chunking prevents stack overflow

---

## Test Case ID: TC-UI-AGT-088
**Test Objective**: Verify large dataset export functionality with progress tracking and performance optimization  
**Business Process**: High-Volume Data Export and Performance Management  
**SAP Module**: A2A Agents Developer Portal - List Report  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-088
- **Test Priority**: Critical (P1)
- **Test Type**: Performance, Functional
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/ProjectsListReport.controller.js:3608-3959`
- **Dialog Fragment**: `a2a_agents/backend/app/a2a/developerPortal/static/view/fragments/LargeExportDialog.fragment.xml`
- **Functions Under Test**: `onLargeExport()`, `_performLargeExport()`, `_processWithWebWorker()`, `_processSequentially()`

### Test Preconditions
1. **System State**: Projects list with 1000+ records available
2. **Browser Support**: Modern browser with Web Worker support preferred
3. **Memory**: Adequate browser memory (4GB+ recommended)
4. **User Access**: Export permissions on large datasets
5. **Test Data**: Generate test dataset with varying data types

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Dataset Size | 5000 | Number | Test Data |
| Chunk Size | 500 | Number | Performance Config |
| Export Formats | excel, csv, pdf | Array | Export Options |
| Worker Support | true/false | Boolean | Browser Feature |
| Memory Limit | 2GB | Number | Browser Constraint |

### Test Procedure Steps
1. **Step 1 - Initiate Large Export**
   - Action: Click Export menu > Large Export > Export All to CSV
   - Expected: Confirmation dialog appears warning about large export
   - Verification: Message shows "5000 records" and memory warning

2. **Step 2 - Confirm Large Export**
   - Action: Click "Yes" on confirmation dialog
   - Expected: Large Export Progress dialog opens immediately
   - Verification: Progress shows 0/5000 records, 0/10 chunks

3. **Step 3 - Monitor Web Worker Processing**
   - Action: Observe progress indicators during export
   - Expected: Parallel processing indicator shows if browser supports
   - Verification: "Using Web Worker" message visible

4. **Step 4 - Track Chunk Progress**
   - Action: Monitor chunk progress indicator
   - Expected: Chunks process incrementally (1/10, 2/10, etc.)
   - Verification: Each chunk takes ~1-2 seconds

5. **Step 5 - Verify Time Estimation**
   - Action: Check estimated time remaining display
   - Expected: Time estimate updates and becomes more accurate
   - Verification: Estimate decreases as export progresses

6. **Step 6 - Test UI Responsiveness**
   - Action: Try interacting with other UI elements during export
   - Expected: UI remains responsive, no freezing
   - Verification: Can scroll, click buttons during export

7. **Step 7 - Complete CSV Export**
   - Action: Wait for CSV export to complete
   - Expected: File downloads automatically, success message shown
   - Verification: CSV file contains all 5000 records with headers

8. **Step 8 - Test Excel Large Export**
   - Action: Initiate large export to Excel format
   - Expected: Excel export uses SAP library with chunking
   - Verification: Excel file properly formatted with all data

9. **Step 9 - Test PDF Large Export**
   - Action: Start large PDF export (limited to 2000 records)
   - Expected: Warning about PDF size limitations
   - Verification: PDF generates with proper pagination

10. **Step 10 - Cancel During Export**
    - Action: Start new export and click "Cancel Export" mid-process
    - Expected: Export stops immediately, dialog can be closed
    - Verification: No partial file downloaded

11. **Step 11 - Test Sequential Fallback**
    - Action: Disable Web Workers and retry large export
    - Expected: Falls back to sequential processing with warning
    - Verification: Export completes but takes longer

12. **Step 12 - Memory Stress Test**
    - Action: Export 10,000+ records if available
    - Expected: System handles without crashing
    - Verification: Memory usage monitored, no browser crash

### Expected Results
- **Progress Tracking**:
  - Real-time updates for both records and chunks
  - Accurate time estimation after initial chunks
  - Clear visual progress indicators
  - Chunk processing visible to user
  
- **Performance Optimization**:
  - Web Worker utilization when available
  - Chunked processing prevents UI freeze
  - Memory efficient for large datasets
  - Consistent chunk processing time
  
- **Export Quality**:
  - All records included in export
  - Data integrity maintained
  - Proper formatting preserved
  - No data truncation or loss
  
- **User Experience**:
  - Clear warnings for large exports
  - Ability to cancel at any time
  - UI remains responsive
  - Automatic download on completion

### Performance Benchmarks
| Dataset Size | CSV Export | Excel Export | PDF Export |
|-------------|------------|--------------|------------|
| 1,000 records | < 5 seconds | < 10 seconds | < 15 seconds |
| 5,000 records | < 20 seconds | < 30 seconds | < 45 seconds |
| 10,000 records | < 40 seconds | < 60 seconds | Limited/Warning |
| 50,000 records | < 3 minutes | < 5 minutes | Not Recommended |

### Browser Compatibility
- **Chrome/Edge**: Full Web Worker support, optimal performance
- **Firefox**: Full Web Worker support, good performance
- **Safari**: Web Worker support, may have memory limits
- **IE11**: No Web Worker, sequential processing only

### Memory Management
- **Chunk Size**: 500 records optimal for most systems
- **Memory Usage**: ~100MB per 10,000 records
- **Garbage Collection**: Chunks cleared after processing
- **URL Cleanup**: Object URLs revoked after download

### Error Handling
- **Out of Memory**: Graceful failure with clear message
- **Worker Failure**: Automatic fallback to sequential
- **Export Interruption**: Clean cancellation, no corruption
- **Download Blocked**: Browser-specific instructions provided
- **Format Errors**: Validation before processing begins

---

## Test Case ID: TC-UI-AGT-089
**Test Objective**: Verify agent name input validation and auto-generation functionality  
**Business Process**: Agent Configuration and Naming Standards  
**SAP Module**: A2A Agents Developer Portal - Agent Builder  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-089
- **Test Priority**: High (P2)
- **Test Type**: Functional, Input Validation
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:475-662`
- **View File**: `a2a_agents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:55-78`
- **Functions Under Test**: `onAgentNameChange()`, `_validateAgentName()`, `_generateAgentId()`, `onCheckUniqueness()`

### Test Preconditions
1. **System State**: Agent Builder view loaded in a project context
2. **User Access**: Developer role with agent creation permissions
3. **Project Context**: Valid project selected with agent creation enabled
4. **Network**: API endpoints for uniqueness checking available
5. **Browser**: JavaScript enabled with input validation support

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Valid Name | "Data Processing Agent" | String | User Input |
| Invalid Name | "123@#$%^" | String | User Input |
| Reserved Name | "system" | String | User Input |
| Long Name | "A" * 51 | String | User Input |
| Short Name | "AB" | String | User Input |
| Special Chars | "Agent-Name_123" | String | User Input |

### Test Procedure Steps
1. **Step 1 - Enter Valid Agent Name**
   - Action: Type "Data Processing Agent" in agent name field
   - Expected: Input shows green/no error state
   - Verification: Agent ID auto-generates as "data-processing-agent-[timestamp]"

2. **Step 2 - Test Minimum Length Validation**
   - Action: Clear field and type "AB" (2 characters)
   - Expected: Input shows error state with red border
   - Verification: Error message: "Agent name must be at least 3 characters long"

3. **Step 3 - Test Maximum Length Validation**
   - Action: Paste string with 51 characters
   - Expected: Input truncates at 50 characters
   - Verification: Error shown if attempting to exceed limit

4. **Step 4 - Test Special Character Validation**
   - Action: Type "Agent@#$%123"
   - Expected: Input shows error state
   - Verification: Error: "Agent name can only contain letters, numbers, spaces, hyphens, and underscores"

5. **Step 5 - Test Reserved Name Validation**
   - Action: Type "system" in agent name field
   - Expected: Input shows error state
   - Verification: Error: "This name is reserved and cannot be used"

6. **Step 6 - Test Leading/Trailing Spaces**
   - Action: Type "  Agent Name  " with spaces
   - Expected: Input shows error state
   - Verification: Error: "Agent name cannot start or end with spaces"

7. **Step 7 - Test Auto-ID Generation**
   - Action: Enter valid name "My Test Agent"
   - Expected: Agent ID field auto-populates
   - Verification: ID shows "my-test-agent-[timestamp]"

8. **Step 8 - Test Manual ID Entry**
   - Action: Clear ID field and type "custom-agent-id"
   - Expected: ID field accepts input with validation
   - Verification: No error if format is valid

9. **Step 9 - Test ID Format Validation**
   - Action: Type "123-invalid" in ID field
   - Expected: Error state shown
   - Verification: Error: "Agent ID must start with a letter..."

10. **Step 10 - Test Uniqueness Check**
    - Action: Click "Check Uniqueness" button
    - Expected: API call made, result displayed
    - Verification: Success message or warning about duplicates

11. **Step 11 - Test Real-time Validation**
    - Action: Type slowly character by character
    - Expected: Validation updates with each keystroke
    - Verification: Error states appear/disappear dynamically

12. **Step 12 - Test Save with Invalid Data**
    - Action: Try to save with invalid name/ID
    - Expected: Save blocked with error message
    - Verification: Detailed validation error shown

### Expected Results
- **Input Validation**:
  - Real-time validation feedback on keystroke
  - Clear error messages for each validation rule
  - Visual indicators (red border, error icon)
  - Value state text shows specific error
  
- **Auto-generation**:
  - ID generated from valid names automatically
  - Generated IDs follow naming conventions
  - Timestamp suffix ensures uniqueness
  - Special characters handled properly
  
- **Uniqueness Checking**:
  - API validates against existing agents
  - Clear feedback on name/ID availability
  - Separate validation for name vs ID
  - Loading state during check
  
- **User Experience**:
  - Intuitive validation messages
  - Helpful placeholder text
  - Maximum length enforced
  - Tab order preserved

### Validation Rules Summary
| Rule | Agent Name | Agent ID |
|------|------------|----------|
| Required | Yes | Yes |
| Min Length | 3 | 3 |
| Max Length | 50 | 30 |
| Start With | Any valid char | Letter only |
| Allowed Chars | a-z A-Z 0-9 space - _ | a-z A-Z 0-9 - _ |
| Reserved Words | system, admin, agent, default, test, demo | system, admin, default, test, demo, null, undefined |
| Spaces | Allowed (not leading/trailing) | Not allowed |

### Error States
- **Empty Field**: "Agent name/ID is required"
- **Too Short**: "Must be at least 3 characters long"
- **Too Long**: "Must not exceed X characters"
- **Invalid Format**: Specific format requirements
- **Reserved Word**: "This name/ID is reserved"
- **Not Unique**: "Already exists in project"

### Browser Compatibility
- Chrome/Edge: Full validation support
- Firefox: Full validation support
- Safari: Full validation support
- IE11: Basic validation, may lack some CSS

### Performance Criteria
- Validation response: < 100ms
- Auto-generation: Instant
- Uniqueness check: < 2 seconds
- No UI freezing during validation

---

## Test Case ID: TC-UI-AGT-090
**Test Objective**: Verify agent capability selection and configuration functionality  
**Business Process**: Agent Capability Configuration and Management  
**SAP Module**: A2A Agents Developer Portal - Agent Builder  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-090
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2a_agents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:249-709`
- **Dialog Fragment**: `a2a_agents/backend/app/a2a/developerPortal/static/view/fragments/CapabilitySelectionDialog.fragment.xml`
- **Functions Under Test**: `onAddSkill()`, `_loadAvailableCapabilities()`, `onCapabilitySelect()`, `onAddSelectedCapabilities()`

### Test Preconditions
1. **System State**: Agent Builder loaded with valid project context
2. **User Access**: Developer permissions for agent configuration
3. **Capability Library**: Available capabilities defined in system
4. **Template Data**: Agent templates with predefined capability sets
5. **UI State**: Skills tab accessible in agent configuration

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|------------|
| Category | "data" | String | User Selection |
| Search Query | "csv parser" | String | User Input |
| Capability IDs | ["csv-parser", "json-processor"] | Array | Selection |
| Complexity Filter | "Low" | String | User Filter |
| Quick Add Set | "Basic Data Agent" | String | Preset Option |

### Test Procedure Steps
1. **Step 1 - Open Capability Selection**
   - Action: Navigate to Skills tab and click "Add Skill" button
   - Expected: Capability Selection Dialog opens with category tabs
   - Verification: Dialog shows "Data Processing" category selected by default

2. **Step 2 - Browse Categories**
   - Action: Click through different capability categories (Communication, AI, etc.)
   - Expected: Available capabilities update based on selected category
   - Verification: Each category shows relevant capabilities with icons

3. **Step 3 - Search Capabilities**
   - Action: Enter "csv" in search field
   - Expected: List filters to show only CSV-related capabilities
   - Verification: "CSV File Parser" appears in results

4. **Step 4 - View Capability Details**
   - Action: Click information button on a capability
   - Expected: Details popup shows name, description, complexity, dependencies
   - Verification: All capability metadata displayed correctly

5. **Step 5 - Select Individual Capabilities**
   - Action: Check boxes for "CSV File Parser" and "JSON Data Processor"
   - Expected: Capabilities added to selection list
   - Verification: Selected count updates, capabilities appear in summary

6. **Step 6 - Test Complexity Indicators**
   - Action: Observe complexity badges (Low/Medium/High)
   - Expected: Color coding shows green/yellow/red for complexity levels
   - Verification: Visual indicators match complexity descriptions

7. **Step 7 - Check Dependencies**
   - Action: Select capability with dependencies (e.g., "Database Connector")
   - Expected: Dependencies listed clearly in capability details
   - Verification: Required dependencies shown in red text

8. **Step 8 - Use Quick Add Presets**
   - Action: Click "Basic Data Agent" quick add button
   - Expected: Predefined set of data capabilities auto-selected
   - Verification: CSV Parser, JSON Processor, Data Validator selected

9. **Step 9 - Clear Selection**
   - Action: Click "Clear Selection" button
   - Expected: All selected capabilities removed from list
   - Verification: Selection count returns to 0

10. **Step 10 - Add Selected Capabilities**
    - Action: Select capabilities and click "Add Selected" button
    - Expected: Dialog closes, capabilities added to agent skills list
    - Verification: Skills tab shows new capabilities with configuration options

11. **Step 11 - Test Duplicate Prevention**
    - Action: Try to add same capability again
    - Expected: System prevents duplicates
    - Verification: Warning message or capability already marked as added

12. **Step 12 - Verify Skill Configuration**
    - Action: Click on added skill in Skills tab
    - Expected: Skill configuration options available
    - Verification: Can configure parameters specific to capability

### Expected Results
- **Category Navigation**:
  - Smooth switching between capability categories
  - Category-specific capabilities displayed
  - Clear visual distinction between categories
  - Category descriptions and icons shown
  
- **Search and Filter**:
  - Real-time search filtering capabilities
  - Search works across name, description, and tags
  - Clear search results with highlighting
  - Easy search reset functionality
  
- **Capability Information**:
  - Complete capability metadata displayed
  - Complexity levels clearly indicated
  - Dependencies and tags visible
  - Detailed information accessible via popup
  
- **Selection Management**:
  - Multiple selection support
  - Clear visual feedback for selections
  - Selection summary with count
  - Easy selection clearing

### Capability Categories
| Category | Count | Example Capabilities |
|----------|-------|---------------------|
| Data Processing | 4+ | CSV Parser, JSON Processor, Data Validator |
| Communication | 3+ | MQTT Client, WebSocket Handler, Email Sender |
| Integration | 3+ | REST Client, Database Connector, File Watcher |
| AI & ML | 3+ | Text Classifier, Sentiment Analyzer, Anomaly Detector |
| Workflow | 2+ | Task Scheduler, State Machine |
| Monitoring | 2+ | Health Monitor, Performance Tracker |

### Quick Add Presets
- **Basic Data Agent**: CSV Parser + JSON Processor + Data Validator
- **Communication Hub**: MQTT Client + WebSocket Handler + Email Sender  
- **AI Assistant**: Text Classifier + Sentiment Analyzer + Anomaly Detector
- **Workflow Coordinator**: Task Scheduler + State Machine + Health Monitor

### Validation Criteria
- **Usability**: Intuitive capability browsing and selection
- **Performance**: Fast category switching and search filtering
- **Accuracy**: Correct capability metadata and dependencies
- **Reliability**: Consistent behavior across browser sessions
- **Integration**: Proper addition to agent skill set

### Error Handling
- **No Selection**: Warning when trying to add without selection
- **Search No Results**: Clear message when search yields no results
- **Category Load Error**: Fallback when category data fails to load
- **Duplicate Addition**: Prevention with appropriate user feedback
- **Dialog Close**: Proper cleanup of selection state

### Browser Compatibility
- Chrome/Edge: Full feature support
- Firefox: Full feature support
- Safari: Full feature support with minor styling differences
- IE11: Basic functionality, may lack some animations

---

## Test Case ID: TC-UI-AGT-091
**Test Objective**: Verify agent version management with semantic versioning support  
**Business Process**: Agent Version Control and Release Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-091
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Version Control
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/fragments/VersionManagementDialog.fragment.xml:1-235`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:onManageVersion()`
- **Functions Under Test**: `onManageVersion()`, `onVersionTypeChange()`, `onApplyVersion()`, `_calculateNextVersion()`, `_validateVersion()`, `_updateVersionHistory()`

### Test Preconditions
1. **Agent Project**: Valid agent project open in Agent Builder
2. **Version History**: At least one existing version in agent model
3. **User Permissions**: Developer role with version management rights
4. **Backend Services**: Version control service available and operational
5. **Model State**: Agent model contains version object with major/minor/patch values

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Current Version | 1.0.0 | String | Agent Model |
| Version Type | major/minor/patch | String | User Selection |
| Prerelease | alpha, beta, rc.1 | String | User Input |
| Build Metadata | build.123, commit.sha | String | User Input |
| Changelog Entry | Version update description | Text | User Input |
| Version Calculation | auto/manual | Boolean | User Toggle |

### Test Procedure Steps
1. **Step 1 - Version Dialog Access**
   - Action: Click "Manage Version" button in Agent Builder toolbar
   - Expected: Version Management Dialog opens with current version info displayed
   - Verification: Dialog title shows "Agent Version Management", current version visible

2. **Step 2 - Current Version Information**
   - Action: Review current version panel information
   - Expected: Current version (1.0.0), compatibility version, last updated date displayed
   - Verification: Version statistics show total version count, proper date formatting

3. **Step 3 - Version Type Selection**
   - Action: Select version type from dropdown (Major, Minor, Patch)
   - Expected: Version type description updates dynamically below selection
   - Verification: Description text explains semantic versioning rules for selected type

4. **Step 4 - Automatic Version Calculation**
   - Action: Keep "Auto" mode enabled, select "Minor" version type
   - Expected: Version preview automatically shows next minor version (1.1.0)
   - Verification: Preview updates instantly with correct calculated version

5. **Step 5 - Manual Version Input**
   - Action: Toggle to "Manual" mode, enter custom version numbers
   - Expected: Manual input fields appear with current version pre-populated
   - Verification: Can modify major, minor, patch numbers independently

6. **Step 6 - Prerelease Identifier**
   - Action: Add prerelease identifier "alpha.1" in optional field
   - Expected: Version preview updates to show "1.1.0-alpha.1"
   - Verification: Prerelease format validation and preview accuracy

7. **Step 7 - Build Metadata**
   - Action: Add build metadata "build.20240101" in optional field
   - Expected: Version preview shows "1.1.0-alpha.1+build.20240101"
   - Verification: Build metadata appears after '+' symbol in preview

8. **Step 8 - Changelog Entry**
   - Action: Enter required changelog description in text area
   - Expected: Character count updates, Apply button becomes enabled
   - Verification: Text area accepts up to 500 characters with counter

9. **Step 9 - Version History Review**
   - Action: Expand version history panel to review previous versions
   - Expected: Table displays all previous versions with dates, types, authors, changes
   - Verification: Version entries sorted chronologically, color-coded by type

10. **Step 10 - Apply New Version**
    - Action: Click "Apply Version" button to create new version
    - Expected: Version applied successfully, agent model updated with new version
    - Verification: Success message shown, version history updated, dialog remains open

11. **Step 11 - Version Validation**
    - Action: Attempt to create duplicate version or invalid format
    - Expected: Validation errors prevent version creation with descriptive messages
    - Verification: Error messages guide user to correct version conflicts

12. **Step 12 - Reset Form**
    - Action: Click "Reset Form" button to clear all inputs
    - Expected: All form fields reset to default values, version preview updated
    - Verification: Form returns to initial state with current version info

### Expected Results
- **Version Information Display**:
  - Current version displays correctly with proper formatting
  - Version statistics accurate (total versions, last updated)
  - Compatibility version shown for backward compatibility tracking
  - Version history table populated with complete historical data

- **Version Calculation Logic**:
  - Auto mode calculates next version correctly based on selected type
  - Major increment resets minor and patch to 0
  - Minor increment resets patch to 0, preserves major
  - Patch increment preserves major and minor values
  - Manual mode allows custom version number input

- **Semantic Versioning Compliance**:
  - Version format follows SemVer specification exactly
  - Prerelease identifiers properly formatted with hyphen prefix
  - Build metadata properly formatted with plus prefix
  - Version comparison logic works correctly for ordering
  - Validation prevents invalid version formats

- **User Interface Behavior**:
  - Dialog responsive and properly sized (800x600px)
  - Form validation provides immediate feedback
  - Apply button disabled until changelog entry provided
  - Reset functionality clears all user inputs
  - Guidelines panel explains semantic versioning rules

### Post-Execution Verification
1. **Model State Verification**:
   - Agent model contains updated version information
   - Version history array includes new version entry
   - Changelog updated with new entry and timestamp
   - Compatibility version updated if major version changed

2. **Persistence Validation**:
   - New version persisted to backend storage
   - Version metadata stored with complete information
   - Version history retrievable for future management
   - Agent deployment references correct version

3. **Integration Testing**:
   - Version information displayed correctly in other views
   - Build process uses new version for artifact generation
   - Deployment process references updated version
   - Version comparison works in dependency management

### Error Scenarios
- **Invalid Version Input**: Non-numeric values, negative numbers, excessive length
- **Duplicate Version**: Attempting to create version that already exists
- **Missing Changelog**: Attempting to apply version without required description
- **Network Failure**: Backend unavailable during version creation
- **Permission Denied**: User lacks rights to modify version information
- **Format Validation**: Invalid prerelease or build metadata formats

### Performance Criteria
- Version dialog opens within 500ms
- Version calculation/preview updates within 100ms of input change
- Version history loads within 1 second for up to 100 versions
- Version application completes within 2 seconds
- Form validation provides immediate feedback (<50ms)

### Security Considerations
- Version creation requires appropriate user permissions
- Changelog entries cannot contain malicious scripts
- Version metadata properly sanitized before storage
- Version history access restricted to authorized users
- Audit trail maintained for all version changes

### Accessibility Requirements
- All form elements properly labeled for screen readers
- Keyboard navigation supported throughout dialog
- High contrast mode compatibility for version status indicators
- Focus management when switching between auto/manual modes
- Error messages announced to assistive technologies

---

## Test Case ID: TC-UI-AGT-092
**Test Objective**: Verify agent icon upload and management functionality  
**Business Process**: Agent Visual Identity Configuration  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-092
- **Test Priority**: Medium (P2)
- **Test Type**: Functional, File Upload
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:108-146`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:onUploadIcon()`
- **Functions Under Test**: `onUploadIcon()`, `onChooseDefaultIcon()`, `onGenerateAIIcon()`, `onRemoveIcon()`, `_processIconFile()`, `_updateIconData()`

### Test Preconditions
1. **Agent Builder Access**: Agent Builder view loaded with agent configuration form
2. **File System Access**: Browser has file system access permissions
3. **Image Files**: Test image files available in various formats (PNG, JPG, SVG)
4. **Icon Preview**: Avatar control renders properly for icon display
5. **Agent Model**: Agent model initialized with icon object structure

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Valid PNG File | test_icon.png (256x256, 45KB) | File | Test Assets |
| Valid JPG File | agent_photo.jpg (512x512, 120KB) | File | Test Assets |
| Valid SVG File | vector_icon.svg (200x200, 8KB) | File | Test Assets |
| Oversized File | huge_image.png (5MB) | File | Test Assets |
| Invalid File | document.pdf (text document) | File | Test Assets |
| Non-square Image | banner.png (800x200) | File | Test Assets |

### Test Procedure Steps
1. **Step 1 - Icon Upload Button Access**
   - Action: Navigate to Agent Builder Basic Info tab
   - Expected: Agent Icon section visible with upload options
   - Verification: Upload Icon, Choose Default, Generate AI, Remove buttons present

2. **Step 2 - File Dialog Opening**
   - Action: Click "Upload Icon" button
   - Expected: Native file dialog opens with image file filters
   - Verification: Dialog accepts PNG, JPG, JPEG, SVG files only

3. **Step 3 - Valid Icon Upload**
   - Action: Select valid PNG file (256x256, 45KB) from file dialog
   - Expected: File uploads successfully, icon preview updates
   - Verification: Avatar shows uploaded image, file name and size displayed

4. **Step 4 - File Size Validation**
   - Action: Attempt to upload oversized file (>2MB)
   - Expected: Error message displays size limit violation
   - Verification: Upload rejected with "File size exceeds 2MB limit" message

5. **Step 5 - File Type Validation**
   - Action: Attempt to upload invalid file type (PDF document)
   - Expected: Error message displays format validation failure
   - Verification: Upload rejected with "Invalid file format" message

6. **Step 6 - Image Dimension Warnings**
   - Action: Upload non-square image (800x200 banner)
   - Expected: Warning displayed about aspect ratio, upload proceeds
   - Verification: Warning suggests using square images for best results

7. **Step 7 - Small Image Warning**
   - Action: Upload very small image (32x32 pixels)
   - Expected: Warning about image size, upload proceeds
   - Verification: Warning suggests minimum 64x64 pixel size

8. **Step 8 - Default Icon Selection**
   - Action: Click "Choose Default" button
   - Expected: Default Icon Dialog opens with categorized icons
   - Verification: Dialog shows AI/ML, Data, Communication, Integration categories

9. **Step 9 - Icon Category Filtering**
   - Action: Select different categories in default icon dialog
   - Expected: Icon display filters by selected category
   - Verification: Only relevant icons shown for each category

10. **Step 10 - Default Icon Selection**
    - Action: Click on a default icon (e.g., AI Assistant)
    - Expected: Icon selected, dialog closes, preview updates
    - Verification: Avatar displays selected SAP icon, metadata updated

11. **Step 11 - AI Icon Generation**
    - Action: Enter agent name "DataProcessor", click "Generate AI Icon"
    - Expected: Loading indicator shows, AI icon generated after delay
    - Verification: Custom SVG icon created and displayed in preview

12. **Step 12 - AI Generation Without Name**
    - Action: Clear agent name, click "Generate AI Icon"
    - Expected: Warning message requests agent name first
    - Verification: No generation occurs without agent name input

13. **Step 13 - Icon Removal**
    - Action: Upload icon, then click "Remove Icon" button
    - Expected: Confirmation dialog appears asking to remove icon
    - Verification: Avatar clears when confirmed, "No icon selected" shown

14. **Step 14 - Icon Persistence**
    - Action: Upload icon, navigate away from tab, return to Basic Info
    - Expected: Icon remains selected and visible in preview
    - Verification: Icon data persisted in agent model correctly

15. **Step 15 - Multiple Format Support**
    - Action: Test uploading PNG, JPG, and SVG files sequentially
    - Expected: All formats upload and display correctly
    - Verification: Each format shows proper preview and metadata

### Expected Results
- **Upload Functionality**:
  - File dialog opens with proper image filters
  - Valid images upload successfully within 3 seconds
  - Preview updates immediately after successful upload
  - File metadata (name, size, type) displayed accurately

- **Validation Mechanisms**:
  - File size validation enforces 2MB limit strictly
  - File type validation accepts only PNG, JPG, JPEG, SVG
  - Dimension warnings provided for non-square images
  - Size warnings for images smaller than 64x64 pixels

- **Default Icon System**:
  - Dialog loads with 20+ predefined icons across 5 categories
  - Category filtering works correctly for icon organization
  - Icon selection updates preview and closes dialog
  - SAP icon metadata properly stored in agent model

- **AI Generation Feature**:
  - Requires agent name for context-aware generation
  - Shows loading indicator during 2-second generation
  - Produces unique SVG icon based on agent properties
  - Generated icon includes agent name in filename

### Post-Execution Verification
1. **Agent Model State**:
   - Icon object contains src, name, size, type, lastModified
   - Icon data URL properly formatted for display
   - Metadata accurately reflects uploaded file properties
   - Icon persists through navigation and form interactions

2. **UI State Consistency**:
   - Avatar control displays icon correctly across all sizes
   - Remove button enabled only when icon present
   - File information updates reflect current selection
   - Button states properly reflect available actions

3. **Data Validation**:
   - File size calculations accurate (displayed in KB)
   - Image dimensions detected correctly for warnings
   - File type detection prevents malicious uploads
   - Generated filenames follow naming conventions

### Error Scenarios
- **File Access Denied**: Browser blocks file system access
- **Corrupted Image**: File appears valid but cannot be processed
- **Network Timeout**: AI generation service unavailable
- **Memory Limits**: Very large images cause browser memory issues
- **Format Edge Cases**: Uncommon image variants or corrupted headers
- **Dialog Interference**: Multiple dialogs open simultaneously

### Performance Criteria
- File dialog opens within 200ms of button click
- Image preview updates within 500ms of file selection
- File validation completes within 1 second
- AI icon generation completes within 3 seconds
- Dialog loading and rendering within 300ms

### Security Considerations
- File upload limited to image types only
- Base64 encoding used for secure data URL storage
- No server-side file storage in basic implementation
- Client-side validation prevents malicious file types
- Generated icons use safe SVG without external references

### Accessibility Requirements
- Upload button properly labeled for screen readers
- File dialog accessible via keyboard navigation
- Icon preview has alternative text describing current selection
- Error messages announced to assistive technologies
- High contrast support for icon visibility

### Browser Compatibility
- Chrome/Edge: Full feature support including drag-and-drop
- Firefox: Full feature support with standard file dialog
- Safari: Full feature support with minor styling differences
- IE11: Basic functionality, may lack some file validation

---

## Test Case ID: TC-UI-AGT-093
**Test Objective**: Verify comprehensive agent metadata save functionality  
**Business Process**: Agent Configuration and Metadata Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-093
- **Test Priority**: High (P1)
- **Test Type**: Functional, Data Persistence
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:255-413`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:_saveAgent()`
- **Functions Under Test**: `_saveAgent()`, `onUpdateTags()`, `onDeleteTag()`, `_validateEmail()`, `_autoFillMetadata()`

### Test Preconditions
1. **Agent Builder Access**: Agent Builder view loaded with all tabs accessible
2. **Form State**: Agent with basic info (name, ID, type) configured
3. **Metadata Tab**: Metadata tab contains all field groups and controls
4. **Backend Service**: Agent persistence service available and operational
5. **Validation Rules**: Email and required field validation active

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Agent Category | ai-ml | String | Dropdown Selection |
| Tags | ["AI", "Machine Learning", "Reactive"] | Array | MultiInput Tokens |
| Priority | high | String | Dropdown Selection |
| Author | John Developer | String | Text Input |
| Organization | AI Solutions Team | String | Text Input |
| Contact Email | john@example.com | String | Email Input |
| License | mit | String | Dropdown Selection |
| Performance Profile | moderate | String | Dropdown Selection |
| Memory Requirement | 512 | Number | Number Input |
| CPU Requirement | 2 | Number | Number Input |
| Storage Requirement | 200 | Number | Number Input |
| Documentation URL | https://docs.ai-agent.com | String | URL Input |
| Repository URL | https://github.com/team/ai-agent | String | URL Input |
| Release Notes | Enhanced ML capabilities | String | TextArea |

### Test Procedure Steps
1. **Step 1 - Access Metadata Tab**
   - Action: Click on "Metadata" tab in Agent Builder
   - Expected: Metadata form loads with all sections visible
   - Verification: Classification, Authoring, Technical, Documentation sections present

2. **Step 2 - Category Selection (Required)**
   - Action: Select "AI/Machine Learning" from Category dropdown
   - Expected: Category value updates in agent model
   - Verification: Required field validation passes, dropdown shows selection

3. **Step 3 - Tag Management**
   - Action: Enter "AI,Machine Learning,Reactive" in Tags field
   - Expected: Tags convert to tokens automatically
   - Verification: Three tokens created, each deletable, tags array updated

4. **Step 4 - Tag Deletion**
   - Action: Click delete button on "Reactive" tag token
   - Expected: Tag removed from tokens and model array
   - Verification: Only "AI" and "Machine Learning" tokens remain

5. **Step 5 - Priority and Authoring Info**
   - Action: Set priority to "High", fill author and organization fields
   - Expected: Values update in model, no validation errors
   - Verification: Form accepts all text input, model reflects changes

6. **Step 6 - Email Validation**
   - Action: Enter invalid email "invalid-email" in Contact Email field
   - Expected: Email validation triggers during save attempt
   - Verification: Validation error prevents save, descriptive message shown

7. **Step 7 - Email Correction**
   - Action: Correct email to "john@example.com"
   - Expected: Email validation passes
   - Verification: No validation errors, field shows valid state

8. **Step 8 - Technical Metadata**
   - Action: Set performance profile, check platforms, fill resource requirements
   - Expected: All technical fields update model correctly
   - Verification: Platform checkboxes update boolean values, numbers validate

9. **Step 9 - Resource Requirements**
   - Action: Enter memory (512), CPU (2), storage (200) requirements
   - Expected: Numeric inputs validate and update model
   - Verification: Number type validation, proper model updates

10. **Step 10 - Documentation URLs**
    - Action: Fill documentation and repository URL fields
    - Expected: URLs stored in model without validation
    - Verification: URLs saved as strings, no format validation required

11. **Step 11 - Release Notes**
    - Action: Enter multiline release notes in TextArea
    - Expected: Text area accepts multiline input
    - Verification: Release notes stored with line breaks preserved

12. **Step 12 - Timestamp Verification**
    - Action: Check Created, Last Modified timestamps in read-only fields
    - Expected: Timestamps display current dates in locale format
    - Verification: Created timestamp set, Last Modified shows current time

13. **Step 13 - Template Auto-Fill**
    - Action: Change template to "ML Analyzer" from Basic Info tab
    - Expected: Metadata category auto-fills to "ai-ml", relevant tags added
    - Verification: Category updates automatically, ML-related tags suggested

14. **Step 14 - Save with Complete Metadata**
    - Action: Click "Save Agent" button with all metadata fields populated
    - Expected: Validation passes, agent saves successfully with metadata
    - Verification: Success message shown, Last Modified timestamp updates

15. **Step 15 - Save Validation Failure**
    - Action: Clear required category field and attempt save
    - Expected: Save prevented with category validation error
    - Verification: Error message: "Please select an agent category in the Metadata tab"

16. **Step 16 - Deploy with Metadata**
    - Action: Use "Save and Deploy" with complete metadata
    - Expected: Save succeeds, deployment timestamp updates
    - Verification: Last Deployed timestamp set to current time

### Expected Results
- **Form Functionality**:
  - All metadata sections load and display correctly
  - Field groups organize related metadata logically
  - Input controls respond properly to user interaction
  - Required field validation enforces category selection

- **Tag Management System**:
  - MultiInput converts comma-separated text to tokens
  - Individual tags can be deleted via token delete buttons
  - Tag array in model updates with add/remove operations
  - Suggested tags appear from predefined list during input

- **Validation Mechanisms**:
  - Email validation uses standard regex pattern
  - Category selection required before save allowed
  - Numeric inputs validate type and range appropriately
  - URL fields accept any string format without validation

- **Auto-Fill Intelligence**:
  - Template changes trigger metadata category suggestions
  - Agent type and template influence suggested tags
  - Auto-fill respects existing user entries
  - Smart defaults provided based on agent configuration

### Post-Execution Verification
1. **Data Persistence**:
   - Complete metadata object saved to backend storage
   - All nested objects (platforms, resources) persist correctly
   - Tag array maintains proper structure and values
   - Timestamps update appropriately during save operations

2. **Model State Consistency**:
   - Metadata model reflects all user inputs accurately
   - Tag tokens synchronize with underlying data array
   - Resource requirements stored as numeric values
   - Platform selections stored as boolean flags

3. **Business Logic**:
   - Auto-fill logic works without overriding user choices
   - Template-based suggestions enhance user experience
   - Validation prevents incomplete agent configurations
   - Timestamp management tracks agent lifecycle accurately

### Error Scenarios
- **Invalid Email Format**: Various malformed email addresses
- **Missing Required Category**: Attempting save without category selection
- **Numeric Validation**: Non-numeric values in resource requirement fields
- **Template Conflicts**: Auto-fill behavior with existing user selections
- **Backend Failures**: Network issues during save operation
- **Large Text Input**: Extremely long text in release notes field

### Performance Criteria
- Metadata tab loads within 500ms
- Tag token creation responds within 100ms
- Form validation completes within 200ms
- Auto-fill suggestions apply within 300ms
- Save operation completes within 3 seconds

### Security Considerations
- Contact email stored securely without exposure
- Repository URLs validated for basic format safety
- User input sanitized before storage
- Metadata accessible only to authorized users
- No sensitive information logged during operations

### Accessibility Requirements
- All form sections properly labeled for screen readers
- Tab navigation works through all metadata fields
- Required field indicators visible to assistive technologies
- Error messages announced during validation failures
- High contrast support for all form elements

### Integration Testing
- Metadata displays correctly in agent listings
- Search and filter functionality uses metadata fields
- Export operations include complete metadata
- Version management preserves metadata history
- Deployment processes reference metadata correctly

---

## Test Case ID: TC-UI-AGT-097
**Test Objective**: Verify comprehensive agent build process with artifact generation  
**Business Process**: Agent Compilation and Deployment Preparation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-097
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Build Process
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:506-601`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:2011-2477`
- **Functions Under Test**: `onBuildAgent()`, `_startBuildProcess()`, `_executeBuildStep()`, `_validateBuildConfiguration()`, `_generateBuildCode()`, `_resolveDependencies()`, `_compileAgent()`, `_runBuildTests()`, `_generateArtifacts()`

### Test Preconditions
1. **Agent Configuration**: Complete agent with name, ID, type, and category configured
2. **Skills and Handlers**: At least one skill or handler defined for dependency resolution
3. **Build Tab Access**: Build Output tab accessible in right panel
4. **Progress Tracking**: Build progress indicators and status displays functional
5. **File System**: Browser supports file download for artifacts

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Agent Name | MLDataProcessor | String | Agent Model |
| Agent ID | ml_data_proc_001 | String | Agent Model |
| Agent Category | ai-ml | String | Metadata |
| Skills Count | 3 | Number | Skills Array |
| Code Language | python | String | View Model |
| Expected Duration | 6-18 seconds | Number | Build Process |
| Artifact Count | 5 | Number | Generated Files |
| Build Steps | 6 | Number | Process Stages |

### Test Procedure Steps

1. **Step 1 - Build Initiation from Header**
   - Action: Click "Build Agent" button in page header toolbar
   - Expected: Build process starts, view switches to Build Output tab
   - Verification: Tab selection changes, build status set to "building"

2. **Step 2 - Build Initiation from Tab**
   - Action: Navigate to Build Output tab and click "Build Agent" button
   - Expected: Build process starts with same functionality as header button
   - Verification: Both buttons trigger identical build process

3. **Step 3 - Configuration Validation Step**
   - Action: Monitor build output during Step 1 (Validating configuration)
   - Expected: Progress shows 0%, configuration checks logged with warnings
   - Verification: Agent name, ID, category validated; skills and description warnings shown

4. **Step 4 - Code Generation Step**
   - Action: Monitor build output during Step 2 (Generating source code)
   - Expected: Progress shows ~17%, Python code generated based on template
   - Verification: Code generated using agent configuration, character count logged

5. **Step 5 - Dependency Resolution Step**
   - Action: Monitor build output during Step 3 (Resolving dependencies)
   - Expected: Progress shows ~33%, dependencies listed based on agent skills
   - Verification: Base SDK, skill-specific packages (ML libraries) included

6. **Step 6 - Compilation Step**
   - Action: Monitor build output during Step 4 (Compiling agent)
   - Expected: Progress shows ~50%, code syntax validated
   - Verification: Line count processed, no compilation errors reported

7. **Step 7 - Testing Step**
   - Action: Monitor build output during Step 5 (Running tests)
   - Expected: Progress shows ~67%, unit and integration tests execute
   - Verification: Test results logged (PASSED), code quality warnings shown

8. **Step 8 - Artifact Generation Step**
   - Action: Monitor build output during Step 6 (Generating artifacts)
   - Expected: Progress shows ~83%, 5 deployment artifacts created
   - Verification: Python file, requirements.txt, Dockerfile, config YAML, ZIP package

9. **Step 9 - Build Completion**
   - Action: Wait for build process to complete
   - Expected: Progress reaches 100%, status changes to "success"
   - Verification: Success message, total duration displayed, artifacts panel visible

10. **Step 10 - Progress Tracking Validation**
    - Action: Observe progress indicator throughout build
    - Expected: Smooth progression from 0% to 100% across 6 steps
    - Verification: Each step shows correct percentage, step descriptions accurate

11. **Step 11 - Build Output Logging**
    - Action: Review complete build output log
    - Expected: Timestamped entries for all steps with visual indicators
    - Verification: Timestamps, checkmarks, warnings, error counts accurate

12. **Step 12 - Artifact List Display**
    - Action: Examine Generated Artifacts panel
    - Expected: 5 artifacts listed with names, descriptions, sizes, icons
    - Verification: Correct file extensions, realistic sizes, appropriate icons

13. **Step 13 - Download Artifacts Function**
    - Action: Click "Download Artifacts" button (enabled after successful build)
    - Expected: ZIP file download initiated with agent ID prefix
    - Verification: File download dialog, correct filename format

14. **Step 14 - Individual Artifact View**
    - Action: Click on individual artifact items in the list
    - Expected: Artifact details dialog shows name, description, size
    - Verification: Information dialog displays correctly for each file type

15. **Step 15 - Build Failure Scenario**
    - Action: Configure agent with invalid settings and attempt build
    - Expected: Build fails at validation step with specific error message
    - Verification: Status changes to "error", progress stops, error displayed

16. **Step 16 - Clean Build Function**
    - Action: Click "Clean Build" button and confirm
    - Expected: Build state resets and new build process starts
    - Verification: Output cleared, artifacts removed, fresh build initiated

17. **Step 17 - Clear Build Output**
    - Action: Click "Clear Build Output" button during or after build
    - Expected: Build log cleared but status and artifacts preserved
    - Verification: Output TextArea emptied, other build elements unchanged

### Expected Results
- **Build Process Criteria**:
  - Complete build process executes in 6-18 seconds
  - All 6 build steps complete sequentially without errors
  - Progress tracking accurately reflects current step and percentage
  - Build status changes from "idle" to "building" to "success"

- **Artifact Generation Criteria**:
  - Exactly 5 deployment artifacts generated
  - Python source file contains generated agent code
  - Requirements.txt lists appropriate dependencies for agent skills
  - Dockerfile provides container image definition
  - Configuration YAML includes agent metadata and settings
  - ZIP package contains all artifacts for deployment

- **User Interface Criteria**:
  - Build Output tab automatically selected when build starts
  - Progress indicator smoothly updates during build process
  - Build status summary shows completion time, warnings, errors
  - Artifact list displays with expandable panel
  - Download functionality works after successful build

- **Error Handling Criteria**:
  - Invalid configurations prevent build from starting
  - Build failures stop process and display specific error messages
  - Warning conditions logged but don't stop build
  - Clean build properly resets all build state

### Test Data Management
- **Build Artifacts**: Generated files should be realistic and downloadable
- **Progress Timing**: Each build step should take 1-3 seconds for realistic simulation
- **Dependencies**: Dependency resolution should reflect actual agent skill requirements
- **Error Scenarios**: Test both validation failures and compilation errors

### Performance Requirements
- **Build Duration**: Complete build process should finish within 18 seconds
- **Progress Updates**: Progress indicator should update smoothly without lag
- **Memory Usage**: Large build outputs should not cause browser performance issues
- **File Downloads**: Artifact downloads should initiate immediately when requested

### Security Considerations
- **Generated Code**: Ensure generated code doesn't contain security vulnerabilities
- **File Downloads**: Validate file types and sizes before download
- **Build Isolation**: Build process should not affect other agents or system state
- **Error Messages**: Build errors should not expose sensitive system information

### Accessibility Requirements
- **Progress Indicators**: Progress status announced to screen readers
- **Build Output**: Build log content accessible via keyboard navigation
- **Error States**: Build failures announced with appropriate ARIA labels
- **Download Actions**: Download buttons properly labeled for assistive technologies

### Integration Testing
- **Agent Configuration**: Build process uses complete agent configuration
- **Code Generation**: Generated artifacts match agent template and skills
- **Deployment Readiness**: Generated artifacts suitable for actual deployment
- **Version Management**: Build artifacts include version information
- **Error Recovery**: Failed builds can be retried with corrections

---

## Test Case ID: TC-UI-AGT-098
**Test Objective**: Verify agent deployment package creation and download functionality  
**Business Process**: Agent Package Generation and Distribution  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-098
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Package Management
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:2310-2477`
- **Secondary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:583-601`
- **Functions Under Test**: `_generateArtifacts()`, `onDownloadArtifacts()`, `onViewArtifact()`, `_appendBuildOutput()`

### Test Preconditions
1. **Agent Configuration**: Complete agent configuration with name, ID, category, and metadata
2. **Build Process**: Successful completion of all build steps (validation through testing)
3. **Build Status**: Build status set to "success" with all artifacts generated
4. **Browser Support**: Browser supports file download operations and Blob creation
5. **Package Generation**: Build artifacts array populated with 5 deployment files

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Agent Name | PackageTestAgent | String | Agent Model |
| Agent ID | pkg_test_agent_001 | String | Agent Model |
| Artifact Count | 5 | Number | Build Artifacts |
| Source File Size | 15.2 KB | String | Generated Artifact |
| Requirements Size | 0.8 KB | String | Generated Artifact |
| Dockerfile Size | 1.2 KB | String | Generated Artifact |
| Config Size | 2.1 KB | String | Generated Artifact |
| Package Size | 45.8 KB | String | Generated Artifact |

### Test Procedure Steps

1. **Step 1 - Prerequisites Validation**
   - Action: Verify agent has complete configuration and successful build
   - Expected: Agent name, ID, category configured; build status shows "success"
   - Verification: Build artifacts array contains exactly 5 files with valid metadata

2. **Step 2 - Artifact Generation Process**
   - Action: Execute build process to trigger `_generateArtifacts()` function
   - Expected: Function generates 5 deployment artifacts with correct naming
   - Verification: Python file named "{agent.id}.py", requirements.txt, Dockerfile, config YAML, ZIP package

3. **Step 3 - Artifact Metadata Validation**
   - Action: Examine generated artifact objects in build artifacts array
   - Expected: Each artifact has name, description, icon, and size properties
   - Verification: Source code (15.2 KB), requirements (0.8 KB), Dockerfile (1.2 KB), config (2.1 KB), package (45.8 KB)

4. **Step 4 - Build Output Logging**
   - Action: Review build output log during artifact generation
   - Expected: Each artifact generation logged with "Generated:" prefix and file size
   - Verification: Output shows "✅ Artifacts generation complete" at end

5. **Step 5 - Artifacts Panel Display**
   - Action: Examine Generated Artifacts panel after successful build
   - Expected: Panel visible with count "{/buildArtifacts/length}" and expandable list
   - Verification: Panel header shows correct count, list displays all artifacts

6. **Step 6 - Individual Artifact Display**
   - Action: Review each artifact in the list display
   - Expected: StandardListItem shows title, description, icon, info (size), and press handler
   - Verification: Correct icons (source-code, list, container, settings, zip-file)

7. **Step 7 - Artifact Details Dialog**
   - Action: Click on individual artifact items to trigger `onViewArtifact()`
   - Expected: MessageBox displays artifact name, description, and size
   - Verification: Information dialog shows correct details for each file type

8. **Step 8 - Download Button State**
   - Action: Verify "Download Artifacts" button enabled state after successful build
   - Expected: Button enabled when build status is "success", disabled otherwise
   - Verification: Button enabled binding `{= ${/buildStatus} === 'success' }`

9. **Step 9 - Package Download Initiation**
   - Action: Click "Download Artifacts" button to trigger `onDownloadArtifacts()`
   - Expected: ZIP file download initiated with agent ID as filename prefix
   - Verification: Browser download dialog appears with filename "{agent.id}-build-artifacts.txt"

10. **Step 10 - Download Content Validation**
    - Action: Examine downloaded file contents (simulated as text file)
    - Expected: File contains agent metadata including name, ID, and generation timestamp
    - Verification: Content includes agent name, ID, and ISO timestamp format

11. **Step 11 - Download Success Notification**
    - Action: Verify MessageToast appears after successful download
    - Expected: Toast message "Build artifacts downloaded" displays briefly
    - Verification: Success feedback provided to user

12. **Step 12 - Multiple Download Operations**
    - Action: Attempt multiple consecutive downloads of same artifacts
    - Expected: Each download creates new file with same content
    - Verification: No conflicts or caching issues with repeated downloads

13. **Step 13 - Package Creation Performance**
    - Action: Time the artifact generation process during build
    - Expected: Artifact generation completes within 2-3 seconds of build step
    - Verification: No significant delay in final build step completion

14. **Step 14 - Error Scenario - Missing Agent Data**
    - Action: Attempt artifact generation with incomplete agent configuration
    - Expected: Artifacts generated with default values where agent data missing
    - Verification: Function handles undefined properties gracefully

15. **Step 15 - Browser Compatibility**
    - Action: Test download functionality across different browser environments
    - Expected: Blob creation and download works in Chrome, Firefox, Safari
    - Verification: URL.createObjectURL and download link functionality consistent

16. **Step 16 - Memory Management**
    - Action: Verify URL object cleanup after download completion
    - Expected: `window.URL.revokeObjectURL()` called to prevent memory leaks
    - Verification: Temporary URLs properly revoked after download

17. **Step 17 - Artifact Reset on New Build**
    - Action: Initiate new build after successful artifact generation
    - Expected: Previous artifacts cleared and new ones generated
    - Verification: Build artifacts array reset during new build process

### Expected Results
- **Package Generation Criteria**:
  - Exactly 5 deployment artifacts created with consistent naming
  - Python source file contains agent ID in filename
  - Requirements file lists dependencies based on agent skills
  - Dockerfile provides standard container configuration
  - YAML config includes agent metadata and settings
  - ZIP package represents complete deployment bundle

- **File Metadata Criteria**:
  - Each artifact has realistic file size representation
  - Appropriate icons assigned (source-code, list, container, settings, zip-file)
  - Descriptions clearly identify file purpose and content
  - Names follow standard conventions for deployment artifacts

- **Download Functionality Criteria**:
  - Download button properly enabled/disabled based on build status
  - File download initiates immediately when button clicked
  - Downloaded file contains agent metadata and timestamp
  - Success notification provides user feedback
  - No memory leaks from temporary URL objects

- **User Interface Criteria**:
  - Generated Artifacts panel displays after successful build
  - Artifact count updates dynamically in panel header
  - Individual artifacts clickable for detail view
  - Detail dialogs show complete artifact information
  - Panel expandable/collapsible for space management

### Test Data Management
- **Agent Configuration**: Test with various agent types and skill combinations
- **File Sizes**: Verify realistic file size calculations for different agent complexities
- **Naming Conventions**: Ensure consistent file naming across agent types
- **Package Contents**: Validate that package represents deployable agent

### Performance Requirements
- **Generation Speed**: Artifact generation should complete within 3 seconds
- **Memory Usage**: Large packages should not cause browser performance issues
- **Download Performance**: Downloads should initiate within 500ms of button click
- **UI Responsiveness**: Interface should remain responsive during package operations

### Security Considerations
- **File Content**: Ensure generated files don't contain sensitive information
- **Download Security**: Validate file types and prevent malicious content
- **Agent Isolation**: Package generation should not access external resources
- **Content Sanitization**: Generated code should be properly escaped and validated

### Accessibility Requirements
- **Download Actions**: Download buttons properly labeled for screen readers
- **Artifact Lists**: List items accessible via keyboard navigation
- **Status Updates**: Package generation progress announced to assistive technologies
- **Error Handling**: Package errors communicated through accessible methods

### Integration Testing
- **Build Pipeline**: Package creation integrates seamlessly with build process
- **Deployment Ready**: Generated packages suitable for actual agent deployment
- **Version Consistency**: Package contents match agent configuration version
- **Cross-Component**: Package creation works with all agent types and templates
- **Export Compatibility**: Packages compatible with deployment environments

---

## Test Case ID: TC-UI-AGT-099
**Test Objective**: Verify local agent testing functionality and console interaction  
**Business Process**: Agent Testing and Validation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-099
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Local Testing
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:476-504`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:1252-1295`
- **Functions Under Test**: `onStartTest()`, `onStopTest()`, `onClearConsole()`, `onSendTestMessage()`

### Test Preconditions
1. **Agent Configuration**: Complete agent with name, ID, and configuration
2. **Test Console Access**: Test Console tab accessible in right panel
3. **Agent State**: Agent can be built or already in buildable state
4. **Console Interface**: Test output TextArea and message input functional
5. **Local Environment**: Local testing environment can simulate agent execution

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Agent Name | LocalTestAgent | String | Agent Model |
| Agent ID | local_test_001 | String | Agent Model |
| Test Message 1 | Hello Agent | String | User Input |
| Test Message 2 | Process Data | String | User Input |
| Test Message 3 | Status Check | String | User Input |
| Console Rows | 25 | Number | UI Configuration |
| Simulation Delay | 1000ms | Number | Controller |

### Test Procedure Steps

1. **Step 1 - Test Console Access**
   - Action: Navigate to Test Console tab in the right panel
   - Expected: Tab loads with Start Test, Stop Test, Clear buttons and console output area
   - Verification: Test console interface displays with proper layout and controls

2. **Step 2 - Initial Console State**
   - Action: Verify initial state of test console components
   - Expected: Test output TextArea empty, buttons enabled, message input available
   - Verification: Console ready for testing operations

3. **Step 3 - Start Agent Test**
   - Action: Click "Start Test" button to trigger `onStartTest()` function
   - Expected: Test initiation message appears, followed by agent initialization sequence
   - Verification: Console shows "Starting agent test...", then simulated startup messages

4. **Step 4 - Agent Initialization Sequence**
   - Action: Monitor console output during test startup (1-second delay)
   - Expected: Sequential messages for agent init, broker connection, readiness
   - Verification: Console displays:
     - "Agent initialized successfully"
     - "Connecting to message broker..."
     - "Connected to MQTT broker"
     - "Agent is ready to receive messages"

5. **Step 5 - Start Test Success Feedback**
   - Action: Verify MessageToast appears after test start
   - Expected: Toast message "Test started" displays briefly
   - Verification: User receives confirmation of successful test initiation

6. **Step 6 - Console Output Display**
   - Action: Examine test output TextArea after initialization
   - Expected: All startup messages visible in non-editable text area
   - Verification: Console shows complete initialization log with line breaks

7. **Step 7 - Send Test Message**
   - Action: Enter "Hello Agent" in message input and press Enter/submit
   - Expected: `onSendTestMessage()` processes input and simulates agent response
   - Verification: Console displays:
     - "> Hello Agent"
     - "< Processing message..."
     - "< Message processed successfully"

8. **Step 8 - Message Input Clearing**
   - Action: Verify input field cleared after message submission
   - Expected: Input field automatically emptied after sending message
   - Verification: Message input ready for next message

9. **Step 9 - Multiple Message Testing**
   - Action: Send additional test messages ("Process Data", "Status Check")
   - Expected: Each message processed with same response pattern
   - Verification: Console accumulates all message exchanges chronologically

10. **Step 10 - Console Scrolling**
    - Action: Send enough messages to exceed visible console area
    - Expected: Console scrolls to show latest messages while preserving history
    - Verification: All message history retained and accessible via scroll

11. **Step 11 - Clear Console Function**
    - Action: Click clear console button to trigger `onClearConsole()`
    - Expected: Console output cleared but functionality remains active
    - Verification: Test output TextArea emptied, agent continues running

12. **Step 12 - Post-Clear Message Testing**
    - Action: Send test message after clearing console
    - Expected: New messages appear in cleared console without startup sequence
    - Verification: Agent responds normally to new messages

13. **Step 13 - Stop Agent Test**
    - Action: Click "Stop Test" button to trigger `onStopTest()` function
    - Expected: Stop sequence logged and test session terminated
    - Verification: Console shows:
      - "Stopping agent test..."
      - "Test stopped"

14. **Step 14 - Stop Test Success Feedback**
    - Action: Verify MessageToast appears after test stop
    - Expected: Toast message "Test stopped" displays briefly
    - Verification: User receives confirmation of test termination

15. **Step 15 - Post-Stop Behavior**
    - Action: Attempt to send messages after stopping test
    - Expected: Messages still processed (simulated local testing continues)
    - Verification: Agent simulation continues even after "stop" for testing purposes

16. **Step 16 - Console State Persistence**
    - Action: Switch between tabs and return to Test Console
    - Expected: Console maintains current state and message history
    - Verification: Test output and interface state preserved during tab navigation

17. **Step 17 - Restart Test Cycle**
    - Action: Start test again after stopping previous session
    - Expected: New initialization sequence begins, previous messages retained
    - Verification: Agent starts fresh initialization while preserving console history

### Expected Results
- **Test Initiation Criteria**:
  - Start Test button initiates agent simulation successfully
  - Console displays realistic agent startup sequence
  - 1-second delay provides realistic initialization timing
  - Success feedback provided via MessageToast

- **Message Processing Criteria**:
  - Test messages processed with consistent response pattern
  - Input and output clearly distinguished (> for input, < for output)
  - Message input automatically cleared after submission
  - All message exchanges logged chronologically

- **Console Management Criteria**:
  - Console output accumulates properly without losing history
  - Clear function empties console while maintaining agent state
  - Console scrolls appropriately for long output
  - 25-row configuration provides adequate viewing area

- **Test Control Criteria**:
  - Stop Test terminates session with appropriate logging
  - Start/Stop cycle works correctly multiple times
  - Console state persists during tab navigation
  - All test controls remain functional throughout session

### Test Data Management
- **Message Simulation**: Test various message types and lengths
- **Console Capacity**: Test with large volumes of messages for scrolling
- **Timing Validation**: Verify startup delay and response timing
- **State Management**: Ensure proper state transitions between start/stop

### Performance Requirements
- **Startup Time**: Agent initialization should complete within 2 seconds
- **Message Response**: Test messages should process within 100ms
- **Console Rendering**: Large message histories should not cause UI lag
- **Memory Management**: Extended testing sessions should not cause memory leaks

### Security Considerations
- **Message Content**: Test messages should be properly escaped and sanitized
- **Local Execution**: Simulated testing should not execute actual agent code
- **Input Validation**: Message inputs should be validated for safety
- **Resource Isolation**: Local testing should not affect other system components

### Accessibility Requirements
- **Console Output**: Test console accessible via screen readers
- **Button Controls**: All test buttons properly labeled for assistive technologies
- **Message Input**: Input field accessible with appropriate labels
- **Status Updates**: Test progress announced to accessibility tools

### Integration Testing
- **Agent Configuration**: Local testing uses complete agent configuration
- **Build Integration**: Testing available after successful agent build
- **Console Coordination**: Test console integrates with other agent development tools
- **State Synchronization**: Test state coordinated with agent builder state
- **Error Handling**: Test failures handled gracefully with appropriate feedback

---

## Test Case ID: TC-UI-AGT-100
**Test Objective**: Verify test console interface functionality and user interaction patterns  
**Business Process**: Agent Testing Console Management  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-100
- **Test Priority**: High (P2)
- **Test Type**: Functional, User Interface
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:476-504`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:1252-1295`
- **Functions Under Test**: `onStartTest()`, `onStopTest()`, `onClearConsole()`, `onSendTestMessage()`, UI event handlers

### Test Preconditions
1. **Agent Builder Access**: Agent Builder interface loaded with test console tab
2. **Console Components**: All test console UI components rendered and functional
3. **Model Binding**: Test output model properties bound correctly to UI elements
4. **Event Handlers**: All button press and input submit events properly attached
5. **Tab Navigation**: Test Console tab accessible and selectable

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Console Height | 600px | String | VBox Configuration |
| Console Rows | 25 | Number | TextArea Configuration |
| Button Count | 3 | Number | Toolbar Buttons |
| Input Placeholder | {i18n>enterTestMessage} | String | i18n Binding |
| Output Binding | {/testOutput} | String | Model Property |
| Tab Icon | sap-icon://simulate | String | IconTabFilter |
| Tab Key | test | String | Tab Identifier |

### Test Procedure Steps

1. **Step 1 - Console Tab Access**
   - Action: Click on "Test Console" tab in right panel design tab bar
   - Expected: Tab switches to test console with simulate icon and "test" key
   - Verification: IconTabFilter displays with correct icon and tab becomes active

2. **Step 2 - Console Layout Verification**
   - Action: Examine test console layout structure and components
   - Expected: VBox with 600px height containing toolbar, text area, and input
   - Verification: Proper component hierarchy with correct spacing classes

3. **Step 3 - Toolbar Button Rendering**
   - Action: Verify all three toolbar buttons display correctly
   - Expected: Start Test (emphasized), Stop Test, Clear Console (with tooltip)
   - Verification: Buttons show correct icons, types, and press handlers assigned

4. **Step 4 - Start Test Button Functionality**
   - Action: Verify Start Test button configuration and icon
   - Expected: Button shows play icon, emphasized type, bound to onStartTest
   - Verification: Button renders with "sap-icon://play" and proper styling

5. **Step 5 - Stop Test Button Functionality**
   - Action: Verify Stop Test button configuration and icon
   - Expected: Button shows stop icon, default type, bound to onStopTest
   - Verification: Button renders with "sap-icon://stop" and standard styling

6. **Step 6 - Clear Console Button Functionality**
   - Action: Verify Clear Console button configuration and tooltip
   - Expected: Button shows clear icon, tooltip from i18n, bound to onClearConsole
   - Verification: Button renders with "sap-icon://clear-all" and tooltip text

7. **Step 7 - ToolbarSpacer Functionality**
   - Action: Verify toolbar spacer creates proper layout separation
   - Expected: Spacer pushes clear button to right side of toolbar
   - Verification: Proper spacing between action buttons and utility button

8. **Step 8 - Console TextArea Configuration**
   - Action: Examine console output TextArea properties and binding
   - Expected: 25 rows, non-editable, bound to {/testOutput}, full width
   - Verification: TextArea configured with correct attributes and model binding

9. **Step 9 - Console TextArea Styling**
   - Action: Verify TextArea styling and margin classes
   - Expected: "sapUiTinyMarginTop" class applied, 100% width, proper height
   - Verification: TextArea displays with correct CSS classes and dimensions

10. **Step 10 - Message Input Configuration**
    - Action: Examine message input field properties and binding
    - Expected: Placeholder from i18n, submit event bound to onSendTestMessage
    - Verification: Input shows placeholder text and submit handler attached

11. **Step 11 - Message Input Styling**
    - Action: Verify message input styling and positioning
    - Expected: "sapUiTinyMarginTop" class applied for consistent spacing
    - Verification: Input field positioned correctly below console output

12. **Step 12 - Console Height and Scrolling**
    - Action: Test console behavior with content exceeding visible area
    - Expected: TextArea scrolls appropriately for long output content
    - Verification: Scroll behavior works correctly within 25-row limit

13. **Step 13 - Model Binding Validation**
    - Action: Verify console output updates when {/testOutput} model property changes
    - Expected: UI reflects model changes immediately and accurately
    - Verification: Two-way binding works correctly for console content

14. **Step 14 - Responsive Behavior**
    - Action: Test console layout at different screen sizes and resolutions
    - Expected: Console maintains usability and layout integrity
    - Verification: Responsive design principles maintained across viewports

15. **Step 15 - Tab State Persistence**
    - Action: Switch between tabs and return to verify state preservation
    - Expected: Console maintains its state and content during navigation
    - Verification: Tab content preserved correctly during tab switching

16. **Step 16 - Accessibility Features**
    - Action: Test console with keyboard navigation and screen readers
    - Expected: All components accessible via keyboard and properly labeled
    - Verification: ARIA labels, focus management, and accessibility standards met

17. **Step 17 - Error State Handling**
    - Action: Test console behavior when model binding fails or errors occur
    - Expected: Console handles errors gracefully without breaking interface
    - Verification: Error conditions don't crash or corrupt console interface

### Expected Results
- **Interface Layout Criteria**:
  - Test Console tab displays with simulate icon and proper labeling
  - VBox container maintains 600px height with proper component arrangement
  - Toolbar contains all three buttons with correct spacing and alignment
  - TextArea and Input positioned correctly with appropriate margins

- **Component Configuration Criteria**:
  - Start Test button emphasized with play icon and proper event handler
  - Stop Test button standard with stop icon and proper event handler
  - Clear Console button with clear icon, tooltip, and proper event handler
  - TextArea non-editable, 25 rows, bound to testOutput model property

- **User Interaction Criteria**:
  - All buttons clickable and responsive to user interaction
  - Message input accepts text and triggers submit on Enter key
  - Console output displays content from model binding accurately
  - Tab navigation preserves console state and functionality

- **Visual Design Criteria**:
  - Consistent spacing using SAP UI5 margin classes
  - Proper button styling with appropriate types and icons
  - TextArea fills available width while maintaining readability
  - Overall layout follows SAP Fiori design guidelines

### Test Data Management
- **i18n Integration**: Verify all text resources loaded from internationalization files
- **Model Properties**: Test with various testOutput content lengths and formats
- **Event Binding**: Validate all event handlers properly attached and functional
- **CSS Classes**: Ensure all styling classes applied correctly

### Performance Requirements
- **Rendering Speed**: Console tab should load and render within 500ms
- **User Interaction**: Button clicks should respond within 100ms
- **Content Updates**: Model changes should reflect in UI within 50ms
- **Memory Usage**: Console should not cause memory leaks during extended use

### Security Considerations
- **Input Sanitization**: Message input should sanitize content appropriately
- **Content Escaping**: Console output should properly escape HTML/script content
- **Model Security**: testOutput model property should be protected from injection
- **Event Security**: Event handlers should validate input and prevent misuse

### Accessibility Requirements
- **Keyboard Navigation**: All interactive elements accessible via keyboard
- **Screen Readers**: Console content announced appropriately to assistive technologies
- **Focus Management**: Logical tab order and visible focus indicators
- **ARIA Labels**: Proper labeling for buttons and input fields

### Integration Testing
- **Model Integration**: Console integrates properly with agent builder model
- **Tab System**: Console works correctly within IconTabBar container
- **Event System**: All event handlers integrate with controller methods
- **i18n System**: Text resources loaded correctly from internationalization system
- **Styling System**: CSS classes and theming work correctly

---

## Test Case ID: TC-UI-AGT-101
**Test Objective**: Verify message injection functionality in agent testing environment  
**Business Process**: Agent Testing and Message Processing Validation  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-101
- **Test Priority**: High (P2)
- **Test Type**: Functional, Message Processing
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:500-504`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:1283-1295`
- **Functions Under Test**: `onSendTestMessage()`, message processing simulation, input validation

### Test Preconditions
1. **Agent Testing Active**: Agent test session started via "Start Test" button
2. **Test Console Available**: Test console tab accessible and displaying output
3. **Message Input Ready**: Message input field functional and accepting user input
4. **Agent Simulation Running**: Agent simulation environment active and responsive
5. **Console Output Binding**: Test output model properly bound to display area

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Test Message 1 | Hello World | String | User Input |
| Test Message 2 | {"type":"data","payload":"test"} | String | JSON Message |
| Test Message 3 | SYSTEM_STATUS_CHECK | String | Command Message |
| Long Message | Lorem ipsum... (500 chars) | String | Performance Test |
| Special Characters | !@#$%^&*()_+{}[]:";'<>?,./ | String | Validation Test |
| Empty Message | "" | String | Edge Case |
| Unicode Message | 你好世界 🌍 | String | Character Encoding |
| Input Placeholder | {i18n>enterTestMessage} | String | i18n Binding |

### Test Procedure Steps

1. **Step 1 - Message Input Availability**
   - Action: Verify message input field present in test console
   - Expected: Input field visible with placeholder text from i18n resources
   - Verification: Input shows "Enter test message" or equivalent localized text

2. **Step 2 - Input Field Configuration**
   - Action: Examine input field properties and event bindings
   - Expected: Input configured with submit event bound to onSendTestMessage
   - Verification: Pressing Enter triggers submit handler correctly

3. **Step 3 - Basic Message Injection**
   - Action: Type "Hello World" in message input and press Enter
   - Expected: onSendTestMessage processes input and simulates agent response
   - Verification: Console displays:
     - "> Hello World" (user message)
     - "< Processing message..." (agent processing)
     - "< Message processed successfully" (agent response)

4. **Step 4 - Input Field Clearing**
   - Action: Verify input field behavior after message submission
   - Expected: Input field automatically cleared via setValue("")
   - Verification: Input ready for next message without manual clearing

5. **Step 5 - JSON Message Injection**
   - Action: Send structured JSON message: {"type":"data","payload":"test"}
   - Expected: Agent processes JSON message using same response pattern
   - Verification: Console shows proper message formatting and processing flow

6. **Step 6 - Command Message Processing**
   - Action: Send command-style message: "SYSTEM_STATUS_CHECK"
   - Expected: Agent responds to command message with standard processing pattern
   - Verification: All message types processed uniformly by simulation

7. **Step 7 - Message Validation and Sanitization**
   - Action: Send message with special characters: !@#$%^&*()_+
   - Expected: Input properly sanitized and displayed without breaking console
   - Verification: Special characters handled safely in console output

8. **Step 8 - Long Message Handling**
   - Action: Send message with 500+ characters
   - Expected: Long message processed without truncation or error
   - Verification: Full message content preserved in console output

9. **Step 9 - Empty Message Handling**
   - Action: Submit empty message (just press Enter without typing)
   - Expected: onSendTestMessage validates for empty string and ignores
   - Verification: No console output generated for empty messages

10. **Step 10 - Unicode and International Characters**
    - Action: Send message with Unicode characters: "你好世界 🌍"
    - Expected: Unicode characters properly processed and displayed
    - Verification: International text and emoji render correctly in console

11. **Step 11 - Rapid Message Injection**
    - Action: Send multiple messages quickly in succession
    - Expected: All messages queued and processed in order
    - Verification: Console shows all messages with proper sequencing

12. **Step 12 - Message Processing Simulation**
    - Action: Observe agent processing simulation pattern
    - Expected: Each message shows three-line response pattern consistently
    - Verification: Processing indicators follow established pattern

13. **Step 13 - Console Output Formatting**
    - Action: Verify message formatting in console output
    - Expected: User messages prefixed with ">", agent responses with "<"
    - Verification: Clear visual distinction between user input and agent responses

14. **Step 14 - Message History Persistence**
    - Action: Send several messages and verify history retention
    - Expected: All message exchanges preserved in console history
    - Verification: Scroll through console to view complete message history

15. **Step 15 - Error Message Injection**
    - Action: Test behavior when message processing might fail
    - Expected: Error conditions handled gracefully without breaking interface
    - Verification: Agent simulation robust against various input conditions

16. **Step 16 - Message Context Preservation**
    - Action: Send messages while switching between tabs
    - Expected: Message context preserved during tab navigation
    - Verification: Console maintains message history across tab switches

17. **Step 17 - Performance with High Message Volume**
    - Action: Send 20+ messages to test console performance
    - Expected: Console remains responsive with large message volumes
    - Verification: No performance degradation or memory issues observed

### Expected Results
- **Message Processing Criteria**:
  - All message types processed using consistent three-line response pattern
  - User input clearly distinguished from agent responses using >, < prefixes
  - Input field automatically cleared after each message submission
  - Message processing simulation provides realistic agent behavior

- **Input Validation Criteria**:
  - Empty messages ignored and not processed
  - Special characters properly sanitized and safe for display
  - Unicode and international characters handled correctly
  - Long messages processed without truncation or errors

- **Console Integration Criteria**:
  - Message exchanges properly logged to console output
  - Console scrolling works correctly with message history
  - Message history preserved during tab navigation
  - All messages timestamped or sequenced appropriately

- **User Experience Criteria**:
  - Input field responsive and intuitive for message entry
  - Visual feedback provided for message processing
  - Console output readable and well-formatted
  - No UI freezing or lag during message injection

### Test Data Management
- **Message Variety**: Test with different message types, lengths, and formats
- **Character Encoding**: Verify proper handling of various character sets
- **Input Sanitization**: Ensure malicious input handled safely
- **Performance Data**: Monitor response times and resource usage

### Performance Requirements
- **Message Processing**: Each message should process within 200ms
- **Input Response**: Input field should respond to typing within 50ms
- **Console Update**: Console output should update within 100ms of submission
- **Memory Management**: Extended testing should not cause memory leaks

### Security Considerations
- **Input Sanitization**: All message input properly escaped and sanitized
- **XSS Prevention**: Console output safe from script injection attacks
- **Data Validation**: Message content validated before processing
- **Sandbox Isolation**: Message injection isolated within testing environment

### Accessibility Requirements
- **Keyboard Navigation**: Message input accessible via keyboard only
- **Screen Reader Support**: Console output announced to assistive technologies
- **Focus Management**: Input field maintains proper focus after message submission
- **ARIA Labels**: Appropriate labels for message input and processing states

### Integration Testing
- **Test Console Integration**: Message injection works within test console framework
- **Agent Model Integration**: Messages interact properly with agent configuration
- **Event System Integration**: Submit events properly bound and handled
- **Model Binding Integration**: Console output updates via proper model binding
- **i18n Integration**: Placeholder text properly localized

---

## Test Case ID: TC-UI-AGT-102
**Test Objective**: Verify comprehensive resource monitoring and system performance tracking functionality  
**Business Process**: System Performance and Resource Monitoring  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-102
- **Test Priority**: High (P2)
- **Test Type**: Functional, Performance Monitoring
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/monitoring.view.xml:1-393`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/monitoring.controller.js:1-100`
- **Functions Under Test**: `onRefreshMonitoring()`, `onCheckSystemHealth()`, `onCheckAgentStatus()`, `_loadMonitoringData()`, `_getMockMonitoringData()`

### Test Preconditions
1. **Monitoring Access**: Monitoring view accessible and properly loaded
2. **System Agents**: Multiple test agents running and available for monitoring
3. **Performance Data**: System performance metrics available and updating
4. **Resource Tracking**: CPU, memory, and system resource monitoring functional
5. **Dashboard Configuration**: Monitoring dashboard configured with appropriate cards and metrics

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| CPU Usage | 68% | Number | Mock Data |
| Memory Usage | 74% | Number | Mock Data |
| System Uptime | 15.2 days | Number | Mock Data |
| Total Requests | 24567 | Number | Mock Data |
| Average Response Time | 142ms | Number | Mock Data |
| Error Rate | 2.1% | Number | Mock Data |
| Active Agents | 3 | Number | Mock Data |
| View Modes | dashboard, agents, performance, logs | String | UI Configuration |

### Test Procedure Steps

1. **Step 1 - Monitoring Dashboard Access**
   - Action: Navigate to monitoring view and verify dashboard loads
   - Expected: DynamicPage displays with monitoring title and action buttons
   - Verification: Page loads with refresh, alerts, and export buttons visible

2. **Step 2 - Dashboard Cards Display**
   - Action: Examine system overview cards in dashboard view
   - Expected: Four numeric header cards showing uptime, requests, response time, error rate
   - Verification: Cards display correct values from mock data with appropriate states

3. **Step 3 - Resource Usage Monitoring**
   - Action: Verify CPU and Memory usage cards in resource section
   - Expected: Progress indicators showing CPU (68%) and Memory (74%) usage
   - Verification: Progress bars display with color-coded states (Success/Warning/Error)

4. **Step 4 - Refresh Monitoring Function**
   - Action: Click "Refresh" button to trigger onRefreshMonitoring
   - Expected: Monitoring data refreshes and displays updated values
   - Verification: _loadMonitoringData function called, data updated in view model

5. **Step 5 - System Health Check**
   - Action: Click "System Health" button to trigger onCheckSystemHealth
   - Expected: System health validation performed and status displayed
   - Verification: Health check function executes and provides feedback

6. **Step 6 - Agent Status Check**
   - Action: Click "Agent Status" button to trigger onCheckAgentStatus
   - Expected: Individual agent status verification performed
   - Verification: Agent status function executes and updates agent monitoring data

7. **Step 7 - View Mode Navigation**
   - Action: Switch between dashboard, agents, performance, and logs views
   - Expected: SegmentedButton selection changes content display accordingly
   - Verification: Each view mode shows appropriate content and hides others

8. **Step 8 - Agents View Functionality**
   - Action: Switch to agents view mode and examine agents table
   - Expected: Table displays running agents with status, uptime, and metrics
   - Verification: Three mock agents shown with proper data binding

9. **Step 9 - Performance Metrics Display**
   - Action: Switch to performance view to examine throughput and response time metrics
   - Expected: Performance cards show current, peak, average throughput and response times
   - Verification: Performance data displays from mock data with proper formatting

10. **Step 10 - Logs View Functionality**
    - Action: Switch to logs view and examine log entries table
    - Expected: Logs table displays with timestamp, level, component, message columns
    - Verification: Log entries show with proper sorting and growing functionality

11. **Step 11 - Live Logs Toggle**
    - Action: Toggle live logs switch in logs view
    - Expected: Switch state changes and affects log streaming behavior
    - Verification: onToggleLiveLogs function updates view model property

12. **Step 12 - Log Level Filtering**
    - Action: Change log level filter selection in logs view
    - Expected: Log entries filtered based on selected level (ERROR, WARN, INFO, DEBUG)
    - Verification: onLogLevelFilter function filters log display appropriately

13. **Step 13 - Agent Detail Actions**
    - Action: Click detail view, restart, and logs buttons for agents
    - Expected: Agent-specific actions trigger appropriate functions
    - Verification: onViewAgentDetails, onRestartAgent, onViewAgentLogs functions execute

14. **Step 14 - Resource Threshold Monitoring**
    - Action: Modify CPU/Memory usage values to test threshold states
    - Expected: Progress indicators change color based on usage thresholds
    - Verification: CPU >80% = Error, >60% = Warning; Memory >85% = Error, >70% = Warning

15. **Step 15 - Export Functionality**
    - Action: Click export buttons in various views
    - Expected: Export functions trigger for logs and monitoring data
    - Verification: onExportLogs and onExportSelected functions execute

16. **Step 16 - Search and Filter**
    - Action: Use search field and filter buttons in monitoring interface
    - Expected: Search and filter functionality works across monitoring data
    - Verification: onSearch, onOpenFilterDialog, onOpenSortDialog functions execute

17. **Step 17 - Empty State Handling**
    - Action: Test monitoring with no agents or logs available
    - Expected: Empty state displays with appropriate messaging and refresh option
    - Verification: Empty state VBox visible with no data message and refresh button

### Expected Results
- **Dashboard Display Criteria**:
  - System overview cards display key metrics (uptime, requests, response time, error rate)
  - Resource usage cards show CPU and memory with color-coded progress indicators
  - All numeric values formatted appropriately with units and states
  - Cards responsive and properly laid out in grid system

- **Agent Monitoring Criteria**:
  - Agent table displays all running agents with current status
  - Agent details include name, type, environment, uptime, request counts
  - Agent actions (view details, restart, view logs) functional
  - Agent status updates in real-time or on refresh

- **Performance Tracking Criteria**:
  - Performance metrics display throughput and response time data
  - Current, peak, and average values shown for throughput
  - Response time percentiles (P95, P99) and maximum values displayed
  - Performance data updates appropriately on refresh

- **Resource Monitoring Criteria**:
  - CPU and memory usage displayed with percentage values
  - Progress indicators show appropriate colors based on thresholds
  - Resource usage updates reflect current system state
  - Threshold warnings trigger at appropriate levels (60%, 70%, 80%, 85%)

### Test Data Management
- **Mock Data Integration**: Test uses comprehensive mock data when backend unavailable
- **Real-time Updates**: Monitoring data refreshes at appropriate intervals
- **Historical Data**: System maintains performance history for trending
- **Alert Thresholds**: Configurable thresholds for resource and performance alerts

### Performance Requirements
- **Dashboard Load**: Monitoring dashboard should load within 2 seconds
- **Data Refresh**: Monitoring data refresh should complete within 3 seconds
- **UI Responsiveness**: Interface should remain responsive during data updates
- **Memory Management**: Extended monitoring should not cause memory leaks

### Security Considerations
- **Data Access**: Monitoring data access properly authenticated and authorized
- **Log Security**: Log entries sanitized to prevent information disclosure
- **Agent Status**: Agent status information protected from unauthorized access
- **Export Security**: Export functionality validates user permissions

### Accessibility Requirements
- **Dashboard Navigation**: All monitoring views accessible via keyboard
- **Screen Reader Support**: Monitoring data announced to assistive technologies
- **Color Independence**: Status indicators not solely dependent on color
- **Focus Management**: Proper focus handling during view mode switches

### Integration Testing
- **Backend Integration**: Monitoring integrates with actual system metrics when available
- **Fallback Handling**: Graceful fallback to mock data when backend unavailable
- **Agent Integration**: Monitoring reflects actual agent status and performance
- **Alert Integration**: Monitoring alerts integrate with notification systems
- **Export Integration**: Export functionality integrates with file download systems

---

## Test Case ID: TC-UI-AGT-103

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Agent Name | MLDataProcessor | String | Agent Model |
| Agent ID | ml_data_proc_001 | String | Agent Model |
| Agent Category | ai-ml | String | Metadata |
| Skills Count | 3 | Number | Skills Array |
| Code Language | python | String | View Model |
| Expected Duration | 6-18 seconds | Number | Build Process |
| Artifact Count | 5 | Number | Generated Files |
| Build Steps | 6 | Number | Process Stages |

### Test Procedure Steps
1. **Step 1 - Build Initiation (Header)**
   - Action: Click "Build Agent" button in header toolbar
   - Expected: Build starts, switches to Build Output tab automatically
   - Verification: Tab selection changes, build status becomes "building"

2. **Step 2 - Build Initiation (Tab)**
   - Action: Navigate to Build Output tab, click "Build Agent" button
   - Expected: Build process starts with progress indicator
   - Verification: Progress bar appears, current step displays

3. **Step 3 - Configuration Validation**
   - Action: Monitor first build step "Validating configuration"
   - Expected: Configuration checked, warnings for missing description/skills
   - Verification: Build output shows validation results, warnings count updates

4. **Step 4 - Code Generation**
   - Action: Observe "Generating source code" step
   - Expected: Python code generated automatically if not present
   - Verification: Generated code appears in Code Preview tab, character count shown

5. **Step 5 - Dependency Resolution**
   - Action: Watch "Resolving dependencies" step execution
   - Expected: Dependencies identified based on skills (ML libraries for AI agents)
   - Verification: Dependencies list displayed in build output

6. **Step 6 - Compilation Process**
   - Action: Monitor "Compiling agent" step
   - Expected: Code syntax validation, line count processing
   - Verification: Compilation success message, no syntax errors

7. **Step 7 - Test Execution**
   - Action: Observe "Running tests" step
   - Expected: Unit tests, integration tests, security scans executed
   - Verification: All test results show "PASSED", warnings for code quality

8. **Step 8 - Artifact Generation**
   - Action: Watch "Generating artifacts" step completion
   - Expected: 5 deployment artifacts created (Python file, requirements, Dockerfile, config, zip)
   - Verification: Artifact list populated, file sizes displayed

9. **Step 9 - Build Success**
   - Action: Wait for build completion
   - Expected: Build status changes to "success", duration calculated
   - Verification: Success message, progress at 100%, artifacts downloadable

10. **Step 10 - Progress Tracking**
    - Action: Monitor progress indicator throughout build
    - Expected: Progress updates from 0% to 100% across 6 steps
    - Verification: Progress text updates, current step descriptions accurate

11. **Step 11 - Build Status Summary**
    - Action: Review build status panel after completion
    - Expected: Success status, duration, artifact count, warning/error counts
    - Verification: All metrics accurate, status indicators correct colors

12. **Step 12 - Artifact List**
    - Action: Expand "Generated Artifacts" panel
    - Expected: List shows 5 files with names, descriptions, icons, sizes
    - Verification: Correct file types, reasonable sizes, proper icons

13. **Step 13 - Artifact Details**
    - Action: Click on artifact in list (e.g., Python source file)
    - Expected: Artifact details dialog with name, description, size
    - Verification: Modal displays correct information for selected artifact

14. **Step 14 - Download Artifacts**
    - Action: Click "Download Artifacts" button
    - Expected: ZIP file download initiated with agent artifacts
    - Verification: File download starts, success message shown

15. **Step 15 - Clean Build**
    - Action: Click "Clean Build" button
    - Expected: Confirmation dialog, then build process restarts
    - Verification: Build state resets, new build process begins

16. **Step 16 - Build Failure Scenario**
    - Action: Trigger build with missing required category
    - Expected: Build fails at validation step with clear error
    - Verification: Error status, failed message, build stops early

17. **Step 17 - Clear Build Output**
    - Action: Click clear output button during or after build
    - Expected: Build output text area cleared
    - Verification: Output text removed, success message shown

### Expected Results
- **Build Process Flow**:
  - 6-step sequential process executes correctly
  - Each step completes within 1-3 seconds
  - Progress tracking accurate throughout process
  - Total build time between 6-18 seconds

- **Validation and Error Handling**:
  - Configuration validation prevents invalid builds
  - Missing required fields cause build failure
  - Syntax errors in code compilation detected
  - Clear error messages guide user to resolution

- **Code Generation and Compilation**:
  - Source code automatically generated if missing
  - Code compilation validates syntax and structure
  - Dependencies resolved based on agent skills
  - Language-specific build process (Python focus)

- **Artifact Management**:
  - 5 deployment artifacts generated consistently
  - Proper file names based on agent ID
  - Realistic file sizes and appropriate icons
  - Artifact details accessible via click

- **User Interface Behavior**:
  - Build tab automatically selected on build start
  - Progress indicators update in real-time
  - Build output streams continuously
  - Status panels show/hide based on build state

### Post-Execution Verification
1. **Build State Management**:
   - Build status accurately reflects current phase
   - Progress percentage matches actual completion
   - Error and warning counts correctly maintained
   - Duration calculated precisely

2. **Artifact Integrity**:
   - All 5 expected artifacts generated
   - File names follow agent ID conventions
   - Artifact metadata complete and accurate
   - Download functionality produces valid files

3. **Output Logging**:
   - Build steps logged with timestamps
   - Warning and error messages clearly marked
   - Output scrollable and readable
   - Log persistence throughout session

### Error Scenarios
- **Missing Configuration**: Build without agent name/ID
- **Invalid Category**: Build without required metadata category
- **Code Compilation Errors**: Syntax errors in generated code
- **Dependency Conflicts**: Incompatible skill-based dependencies
- **Build Process Interruption**: Browser/network issues during build
- **File System Errors**: Download permission issues

### Performance Criteria
- Build process completes within 20 seconds maximum
- Progress updates respond within 100ms
- Step transitions occur within 500ms
- Artifact generation completes within 3 seconds
- UI remains responsive during entire build process

### Security Considerations
- Generated code validated for security vulnerabilities
- Dependencies checked against known security issues
- Artifact downloads use secure blob creation
- No sensitive information exposed in build output
- Build process isolated from system resources

### Integration Points
- Generated code integrates with Code Preview tab
- Build status affects Save and Deploy functionality
- Artifacts usable for actual agent deployment
- Build warnings/errors logged for analytics
- Version management tracks build history

### Accessibility Requirements
- Progress indicators announce status to screen readers
- Build output readable with assistive technologies
- Keyboard navigation supported for all build controls
- High contrast support for status indicators
- Error messages properly announced

---

## Test Case ID: TC-UI-AGT-095
**Test Objective**: Verify build output display and logging functionality  
**Business Process**: Build Process Monitoring and Diagnostics  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-095
- **Test Priority**: High (P1)
- **Test Type**: Functional, Output Verification
- **Execution Method**: Manual/Automated
- **Risk Level**: Medium

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/view/agentBuilder.view.xml:575-581`
- **Controller**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js:_appendBuildOutput()`
- **Functions Under Test**: `_appendBuildOutput()`, `onClearBuildOutput()`, `_executeBuildStep()`, build logging system

### Test Preconditions
1. **Agent Configuration**: Valid agent configured for building
2. **Build Tab Access**: Build Output tab available and accessible
3. **Build Process**: Build process can be initiated and monitored
4. **Output Area**: Build output TextArea renders correctly
5. **Logging System**: Build steps generate appropriate log messages

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Build Steps | 6 | Number | Build Process |
| Log Messages | 20+ | Number | Build Execution |
| Timestamp Format | HH:MM:SS | String | Local Time |
| Output Length | 2000+ chars | Number | Text Content |
| Warning Messages | 2-5 | Number | Build Warnings |
| Error Messages | 0-3 | Number | Build Errors |

### Test Procedure Steps
1. **Step 1 - Initial Output State**
   - Action: Navigate to Build Output tab before any build
   - Expected: Build output TextArea is empty
   - Verification: TextArea shows no content, scroll position at top

2. **Step 2 - Build Initiation Logging**
   - Action: Start build process and observe initial logging
   - Expected: Build start message appears with timestamp
   - Verification: First log entry shows build initialization

3. **Step 3 - Step-by-Step Logging**
   - Action: Monitor output as each build step executes
   - Expected: Each step logs start message with timestamp format [HH:MM:SS]
   - Verification: 6 step entries appear sequentially with proper timestamps

4. **Step 4 - Detailed Step Output**
   - Action: Observe detailed output within each build step
   - Expected: Sub-messages appear indented under each main step
   - Verification: Configuration validation details, dependency lists, etc.

5. **Step 5 - Progress Indicators in Text**
   - Action: Watch for progress symbols in output (✅, ⚠️, ❌)
   - Expected: Success checkmarks, warning symbols, error symbols appear
   - Verification: Visual indicators match build step outcomes

6. **Step 6 - Warning Message Display**
   - Action: Build agent with missing description to trigger warnings
   - Expected: Warning messages with ⚠️ symbol appear in output
   - Verification: Warnings clearly marked and readable

7. **Step 7 - Auto-Scroll Behavior**
   - Action: Monitor TextArea scroll position during build
   - Expected: Output automatically scrolls to show latest messages
   - Verification: Latest content visible without manual scrolling

8. **Step 8 - Output Formatting**
   - Action: Review output text formatting and structure
   - Expected: Proper indentation, line breaks, consistent formatting
   - Verification: Hierarchical structure clear, readable layout

9. **Step 9 - Completion Summary**
   - Action: Observe final build completion messages
   - Expected: Success/failure summary with duration
   - Verification: Clear completion status with timing information

10. **Step 10 - Output Persistence**
    - Action: Switch tabs and return to Build Output
    - Expected: Complete build output remains visible
    - Verification: All log messages preserved across tab switches

11. **Step 11 - Clear Output Function**
    - Action: Click "Clear Build Output" button
    - Expected: TextArea content cleared, success message shown
    - Verification: Output area empty, confirmation message appears

12. **Step 12 - Multiple Build Outputs**
    - Action: Run build, then run another build without clearing
    - Expected: New build output appends to existing content
    - Verification: Both build sessions visible in output

13. **Step 13 - Long Output Handling**
    - Action: Generate very long build output (multiple builds)
    - Expected: TextArea handles large content without performance issues
    - Verification: Scrolling responsive, no UI freezing

14. **Step 14 - Copy Output Capability**
    - Action: Select and copy text from build output
    - Expected: Text selection and copy operations work normally
    - Verification: Output can be copied to clipboard

15. **Step 15 - Build Failure Output**
    - Action: Trigger build failure and observe error output
    - Expected: Error messages clearly displayed with ❌ symbols
    - Verification: Failure details readable and diagnostic

### Expected Results
- **Output Content Quality**:
  - All build steps logged with appropriate detail level
  - Timestamps accurate and consistently formatted
  - Visual indicators (✅, ⚠️, ❌) enhance readability
  - Hierarchical structure with proper indentation

- **Real-time Display**:
  - Output appears in real-time as build progresses
  - Auto-scroll keeps latest content visible
  - No significant delay between step execution and logging
  - Smooth text appending without flickering

- **Output Management**:
  - Clear function removes all content immediately
  - Output persists across tab navigation
  - Multiple builds append rather than replace
  - Large output volumes handled efficiently

- **User Experience**:
  - Text selectable and copyable for debugging
  - Readable formatting aids in troubleshooting
  - Clear distinction between informational and error messages
  - Proper contrast and font sizing

### Post-Execution Verification
1. **Content Accuracy**:
   - All executed build steps represented in output
   - Warning and error counts match actual issues
   - Timing information correlates with actual durations
   - Step completion status accurately reflected

2. **Formatting Consistency**:
   - Timestamp format uniform throughout output
   - Indentation levels consistent for sub-messages  
   - Visual symbols properly encoded and displayed
   - Line breaks appropriate for readability

3. **Functional Reliability**:
   - Clear function completely empties output
   - Output persistence reliable across navigation
   - No memory leaks with extensive output
   - Text selection and copy operations stable

### Error Scenarios
- **Output Truncation**: Extremely long build outputs
- **Character Encoding**: Special characters in output messages
- **Memory Limits**: Browser limits with massive output content
- **Concurrent Updates**: Rapid output updates causing display issues
- **Clear Function Failure**: Clear operation not completing
- **Copy Restrictions**: Browser limitations on text selection

### Performance Criteria
- Output messages appear within 50ms of generation
- Text appending operations complete within 10ms
- Auto-scroll updates within 100ms
- Clear function completes within 200ms
- Large output (50KB+) displays without lag

### Accessibility Requirements
- Build output readable by screen readers
- Text contrast meets WCAG 2.1 standards
- Keyboard navigation works within output area
- Error messages properly announced
- Visual symbols have text alternatives

### Integration Points
- Build output coordinates with progress indicators
- Warning/error counts sync with status displays
- Output content used for build result analysis
- Log messages support debugging and diagnostics
- Output can be exported or saved for records

### Browser Compatibility
- Chrome/Edge: Full text handling and selection support
- Firefox: Complete output formatting and scrolling
- Safari: Text display with minor font differences
- IE11: Basic functionality with reduced visual enhancements

---

## Test Case ID: TC-UI-AGT-096
**Test Objective**: Verify comprehensive error display and handling across agent builder  
**Business Process**: Error Management and User Guidance  
**SAP Module**: A2A Agents Developer Portal  

### Test Specification
- **Test Case Identifier**: TC-UI-AGT-096
- **Test Priority**: Critical (P1)
- **Test Type**: Functional, Error Handling
- **Execution Method**: Manual/Automated
- **Risk Level**: High

### Target Implementation
- **Primary File**: `a2aAgents/backend/app/a2a/developerPortal/static/controller/agentBuilder.controller.js` (Multiple error handling functions)
- **Controller**: Various error handling functions: `_validateAgentName()`, `_validateAgentId()`, `_validateEmail()`, build error handling
- **Functions Under Test**: All validation and error display functions, MessageBox.error(), MessageToast.show(), field validation states

### Test Preconditions
1. **Agent Builder Access**: Agent Builder view loaded and functional
2. **Form Controls**: All input fields and validation mechanisms active
3. **Error Display**: MessageBox and MessageToast components available
4. **Field States**: Input controls support error states and validation
5. **Network Simulation**: Ability to simulate network failures for testing

### Test Input Data
| Parameter | Value | Type | Source |
|-----------|--------|------|---------|
| Invalid Agent Name | "agent@#$%" | String | Test Input |
| Invalid Agent ID | "123-invalid-id" | String | Test Input |
| Invalid Email | "not-an-email" | String | Test Input |
| Missing Category | null | String | Empty Selection |
| Long Text Input | 1000+ characters | String | Generated |
| Special Characters | Unicode, emojis | String | Test Data |
| Network Timeout | 30 seconds | Number | Simulation |

### Test Procedure Steps
1. **Step 1 - Agent Name Validation Errors**
   - Action: Enter invalid agent name with special characters "@#$%"
   - Expected: Real-time validation error, field shows error state
   - Verification: Input field red border, error message descriptive

2. **Step 2 - Agent ID Format Errors**
   - Action: Enter agent ID starting with number "123invalid"
   - Expected: ID validation fails, proper error message displayed
   - Verification: Clear guidance on correct ID format provided

3. **Step 3 - Required Field Validation**
   - Action: Attempt to save agent with empty name field
   - Expected: Save prevented, error dialog with field requirement message
   - Verification: MessageBox.error appears with clear instruction

4. **Step 4 - Email Format Validation**
   - Action: Enter malformed email in contact field "not-email-format"
   - Expected: Email validation error during save attempt
   - Verification: Specific email format error message shown

5. **Step 5 - Category Selection Requirement**
   - Action: Try to build agent without selecting category in metadata
   - Expected: Build fails with category requirement error
   - Verification: Error message guides user to Metadata tab

6. **Step 6 - File Upload Errors**
   - Action: Attempt to upload 5MB image as agent icon
   - Expected: File size error prevents upload
   - Verification: Clear message about 2MB limit and solution

7. **Step 7 - Invalid File Type Upload**
   - Action: Try to upload PDF file as agent icon
   - Expected: File type validation error displayed
   - Verification: Error specifies acceptable formats (PNG, JPG, SVG)

8. **Step 8 - Build Process Errors**
   - Action: Trigger build failure by manipulating agent configuration
   - Expected: Build stops at error point with diagnostic information
   - Verification: Error location and resolution guidance provided

9. **Step 9 - Network Connection Errors**
   - Action: Simulate network failure during save operation
   - Expected: Network error handled gracefully with retry option
   - Verification: User-friendly error message, not technical details

10. **Step 10 - Version Management Errors**
    - Action: Attempt to create duplicate version number
    - Expected: Version conflict error with clear explanation
    - Verification: Error suggests alternative version numbers

11. **Step 11 - Capability Selection Errors**
    - Action: Select conflicting capabilities with dependency issues
    - Expected: Dependency conflict warning/error displayed
    - Verification: Clear explanation of capability conflicts

12. **Step 12 - Tag Input Validation**
    - Action: Enter extremely long tag name (200+ characters)
    - Expected: Tag length validation prevents addition
    - Verification: Character limit error with recommended length

13. **Step 13 - Resource Requirement Errors**
    - Action: Enter negative numbers in memory/CPU requirements
    - Expected: Numeric validation prevents negative values
    - Verification: Input field validation with appropriate constraints

14. **Step 14 - Template Loading Errors**
    - Action: Simulate template loading failure
    - Expected: Template error handled without breaking application
    - Verification: Fallback behavior or retry mechanism available

15. **Step 15 - Form State Recovery**
    - Action: Trigger error, then correct issue and retry operation
    - Expected: Error state clears, operation proceeds normally
    - Verification: Form validation state resets properly

16. **Step 16 - Multiple Simultaneous Errors**
    - Action: Create multiple validation errors (name, ID, email, category)
    - Expected: All errors reported clearly without overwhelming user
    - Verification: Error prioritization and clear resolution steps

17. **Step 17 - Critical Error Handling**
    - Action: Simulate critical system error during operation
    - Expected: Graceful degradation with user notification
    - Verification: Application remains stable, data not lost

### Expected Results
- **Error Message Quality**:
  - All error messages use clear, non-technical language
  - Specific guidance provided for resolving each error type
  - Error messages consistent in tone and format
  - Critical vs. warning errors properly differentiated

- **Visual Error Indicators**:
  - Form fields show error states with red borders/icons
  - Error messages positioned near relevant controls
  - Error symbols and colors follow SAP Fiori guidelines
  - Visual hierarchy guides user attention appropriately

- **Error Recovery**:
  - All error states can be cleared by correcting input
  - Form validation updates in real-time as errors resolved
  - No persistent error states after successful correction
  - Retry mechanisms available for transient errors

- **User Experience**:
  - Errors don't interrupt workflow unnecessarily
  - Multiple errors presented in digestible format
  - Error prevention where possible through input constraints
  - Help text and tooltips supplement error messages

### Post-Execution Verification
1. **Error State Management**:
   - Error states properly set and cleared
   - Validation triggers at appropriate times
   - No memory leaks from error handling
   - Error recovery mechanisms functional

2. **Message Consistency**:
   - Error message terminology consistent across application
   - Technical details appropriately hidden from users
   - Error codes available for support scenarios
   - Internationalization support for error messages

3. **System Stability**:
   - Application remains responsive during error conditions
   - No cascading errors from initial failures
   - Data integrity maintained during error scenarios
   - Session state preserved through error recovery

### Error Categories Tested
- **Validation Errors**: Format, length, character restrictions
- **Required Field Errors**: Missing mandatory information
- **Business Logic Errors**: Rule violations, conflicts
- **System Errors**: Network, server, resource limitations
- **File Operation Errors**: Upload, download, processing
- **Build Process Errors**: Compilation, dependency issues

### Error Display Methods
- **MessageBox.error**: Modal dialogs for critical errors
- **MessageToast**: Non-blocking notifications for warnings
- **Field Validation**: Inline error states and messages
- **Build Output**: Error logging in build console
- **Status Indicators**: Visual error states in UI components

### Recovery Mechanisms
- **Input Correction**: Real-time validation as user types
- **Retry Operations**: Automatic or manual retry for transient errors
- **Fallback Options**: Alternative paths when primary fails
- **State Restoration**: Return to valid state after error
- **Help Integration**: Context-sensitive help for error resolution

### Performance Criteria
- Error validation completes within 100ms
- Error messages display within 200ms
- Complex validation (uniqueness checks) within 2 seconds
- Error recovery operations complete within 500ms
- No UI blocking during error processing

### Accessibility Requirements
- Error messages announced to screen readers
- Error states detectable by assistive technologies
- High contrast support for error indicators
- Keyboard navigation works during error states
- Error messages readable at various zoom levels

### Security Considerations
- Error messages don't expose sensitive system information
- User input properly sanitized before validation
- Error logging doesn't include credentials or tokens
- Client-side validation complemented by server validation
- No SQL injection or XSS vulnerabilities in error handling

---

**Total Test Cases Completed**: 96  
**Next Document**: a2aAgentsBackend.md (TC-UI-AGT-097 onwards)