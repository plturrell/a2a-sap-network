/**
 * Workflow.view.xml Component Tests
 * Test Cases: TC-AN-137 to TC-AN-152
 * Coverage: Workflow Designer, Process Management, Execution Engine, Templates
 */

describe('Workflow.view.xml - Workflow Management', () => {
  beforeEach(() => {
    cy.visit('/workflows');
    cy.viewport(1280, 720);
    
    // Wait for workflow management page to load
    cy.get('[data-testid="workflow-management"]').should('be.visible');
  });

  describe('TC-AN-137 to TC-AN-141: Workflow Designer Tests', () => {
    it('TC-AN-137: Should verify workflow designer interface loads', () => {
      // Navigate to designer
      cy.get('[data-testid="create-workflow"]').click();
      
      // Verify designer components
      cy.get('[data-testid="workflow-designer"]').should('be.visible');
      cy.get('[data-testid="designer-canvas"]').should('be.visible');
      cy.get('[data-testid="component-palette"]').should('be.visible');
      cy.get('[data-testid="properties-panel"]').should('be.visible');
      cy.get('[data-testid="designer-toolbar"]').should('be.visible');
      
      // Check toolbar buttons
      cy.get('[data-testid="designer-toolbar"]').within(() => {
        cy.get('[data-testid="save-workflow"]').should('be.visible');
        cy.get('[data-testid="validate-workflow"]').should('be.visible');
        cy.get('[data-testid="test-workflow"]').should('be.visible');
        cy.get('[data-testid="undo-action"]').should('be.visible');
        cy.get('[data-testid="redo-action"]').should('be.visible');
        cy.get('[data-testid="zoom-fit"]').should('be.visible');
      });
      
      // Verify canvas grid
      cy.get('[data-testid="designer-canvas"]').should('have.class', 'grid-enabled');
      cy.get('[data-testid="canvas-grid"]').should('be.visible');
      
      // Check component palette categories
      cy.get('[data-testid="component-palette"]').within(() => {
        cy.get('[data-testid="category-triggers"]').should('be.visible');
        cy.get('[data-testid="category-actions"]').should('be.visible');
        cy.get('[data-testid="category-conditions"]').should('be.visible');
        cy.get('[data-testid="category-utilities"]').should('be.visible');
      });
    });

    it('TC-AN-138: Should test drag and drop workflow components', () => {
      cy.get('[data-testid="create-workflow"]').click();
      
      // Test dragging start trigger
      cy.get('[data-testid="component-start-trigger"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 100, 100);
      
      // Verify component placed
      cy.get('[data-testid="workflow-node"]').should('have.length', 1);
      cy.get('[data-testid="workflow-node"]').should('have.attr', 'data-type', 'start-trigger');
      
      // Test dragging action component
      cy.get('[data-testid="component-data-processing"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 300, 100);
      
      // Verify second component placed
      cy.get('[data-testid="workflow-node"]').should('have.length', 2);
      
      // Test dragging condition component
      cy.get('[data-testid="component-if-condition"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 500, 100);
      
      cy.get('[data-testid="workflow-node"]').should('have.length', 3);
      
      // Test invalid drop area
      cy.get('[data-testid="component-end-action"]').trigger('dragstart');
      cy.get('[data-testid="component-palette"]').trigger('drop');
      
      // Should not create node in invalid area
      cy.get('[data-testid="workflow-node"]').should('have.length', 3);
    });

    it('TC-AN-139: Should verify node connection functionality', () => {
      cy.get('[data-testid="create-workflow"]').click();
      
      // Add two nodes
      cy.get('[data-testid="component-start-trigger"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 100, 100);
      
      cy.get('[data-testid="component-data-processing"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 300, 100);
      
      // Connect nodes
      cy.get('[data-testid="workflow-node"]').first().find('[data-testid="output-port"]').trigger('mousedown');
      cy.get('[data-testid="workflow-node"]').last().find('[data-testid="input-port"]').trigger('mouseup');
      
      // Verify connection created
      cy.get('[data-testid="workflow-connection"]').should('have.length', 1);
      cy.get('[data-testid="workflow-connection"]').should('have.attr', 'data-source');
      cy.get('[data-testid="workflow-connection"]').should('have.attr', 'data-target');
      
      // Test connection validation
      cy.get('[data-testid="workflow-node"]').first().find('[data-testid="output-port"]').trigger('mousedown');
      cy.get('[data-testid="workflow-node"]').first().find('[data-testid="input-port"]').trigger('mouseup');
      
      // Should show validation error for self-connection
      cy.get('[data-testid="connection-error"]').should('be.visible');
      cy.get('[data-testid="workflow-connection"]').should('have.length', 1); // No new connection
      
      // Test connection deletion
      cy.get('[data-testid="workflow-connection"]').click();
      cy.get('[data-testid="delete-connection"]').click();
      cy.get('[data-testid="workflow-connection"]').should('have.length', 0);
    });

    it('TC-AN-140: Should test node configuration properties', () => {
      cy.get('[data-testid="create-workflow"]').click();
      
      // Add a data processing node
      cy.get('[data-testid="component-data-processing"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 200, 200);
      
      // Select the node
      cy.get('[data-testid="workflow-node"]').click();
      
      // Verify properties panel shows
      cy.get('[data-testid="properties-panel"]').should('be.visible');
      cy.get('[data-testid="node-properties"]').should('be.visible');
      
      // Check property fields
      cy.get('[data-testid="properties-panel"]').within(() => {
        cy.get('[data-testid="node-name"]').should('be.visible');
        cy.get('[data-testid="node-description"]').should('be.visible');
        cy.get('[data-testid="processing-type"]').should('be.visible');
        cy.get('[data-testid="input-schema"]').should('be.visible');
        cy.get('[data-testid="output-schema"]').should('be.visible');
        cy.get('[data-testid="error-handling"]').should('be.visible');
      });
      
      // Configure node properties
      cy.get('[data-testid="node-name"]').clear().type('Customer Data Processor');
      cy.get('[data-testid="node-description"]').type('Processes incoming customer data');
      cy.get('[data-testid="processing-type"]').select('Transformation');
      
      // Configure input schema
      cy.get('[data-testid="input-schema"]').within(() => {
        cy.get('[data-testid="add-field"]').click();
        cy.get('[data-testid="field-name"]').type('customerId');
        cy.get('[data-testid="field-type"]').select('String');
        cy.get('[data-testid="field-required"]').check();
        cy.get('[data-testid="save-field"]').click();
      });
      
      // Verify properties saved
      cy.get('[data-testid="workflow-node"]').should('contain', 'Customer Data Processor');
      cy.get('[data-testid="properties-saved-indicator"]').should('be.visible');
    });

    it('TC-AN-141: Should verify workflow validation', () => {
      cy.get('[data-testid="create-workflow"]').click();
      
      // Create incomplete workflow
      cy.get('[data-testid="component-data-processing"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 200, 200);
      
      // Validate workflow
      cy.get('[data-testid="validate-workflow"]').click();
      
      // Check validation errors
      cy.get('[data-testid="validation-results"]').should('be.visible');
      cy.get('[data-testid="validation-errors"]').should('contain', 'Missing start trigger');
      cy.get('[data-testid="validation-errors"]').should('contain', 'Unconnected nodes');
      
      // Fix validation issues
      cy.get('[data-testid="component-start-trigger"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 50, 200);
      
      // Connect nodes
      cy.get('[data-testid="workflow-node"]').first().find('[data-testid="output-port"]').trigger('mousedown');
      cy.get('[data-testid="workflow-node"]').last().find('[data-testid="input-port"]').trigger('mouseup');
      
      // Add end node
      cy.get('[data-testid="component-end-action"]').trigger('dragstart');
      cy.get('[data-testid="designer-canvas"]').trigger('drop', 350, 200);
      
      // Connect to end node
      cy.get('[data-testid="workflow-node"]').eq(1).find('[data-testid="output-port"]').trigger('mousedown');
      cy.get('[data-testid="workflow-node"]').last().find('[data-testid="input-port"]').trigger('mouseup');
      
      // Validate again
      cy.get('[data-testid="validate-workflow"]').click();
      cy.get('[data-testid="validation-success"]').should('be.visible');
      cy.get('[data-testid="validation-success"]').should('contain', 'Workflow is valid');
    });
  });

  describe('TC-AN-142 to TC-AN-146: Process Management Tests', () => {
    it('TC-AN-142: Should test workflow list display', () => {
      // Verify workflows table
      cy.get('[data-testid="workflows-table"]').should('be.visible');
      
      // Check table headers
      cy.get('[data-testid="workflows-table"]').within(() => {
        cy.get('[data-testid="header-name"]').should('contain', 'Name');
        cy.get('[data-testid="header-status"]').should('contain', 'Status');
        cy.get('[data-testid="header-version"]').should('contain', 'Version');
        cy.get('[data-testid="header-created"]').should('contain', 'Created');
        cy.get('[data-testid="header-actions"]').should('contain', 'Actions');
      });
      
      // Verify workflow rows
      cy.get('[data-testid="workflow-row"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="workflow-row"]').first().within(() => {
        cy.get('[data-testid="workflow-name"]').should('not.be.empty');
        cy.get('[data-testid="workflow-status"]').should('be.visible');
        cy.get('[data-testid="workflow-version"]').should('match', /v\d+\.\d+/);
        cy.get('[data-testid="workflow-created"]').should('match', /\d{4}-\d{2}-\d{2}/);
        cy.get('[data-testid="workflow-actions"]').should('be.visible');
      });
      
      // Test status indicators
      cy.get('[data-testid="workflow-status"]').each(($status) => {
        const statusText = $status.text();
        expect(['Active', 'Draft', 'Inactive', 'Archived']).to.include(statusText);
      });
    });

    it('TC-AN-143: Should verify workflow search and filtering', () => {
      // Test search functionality
      cy.get('[data-testid="workflow-search"]').should('be.visible');
      cy.get('[data-testid="workflow-search"]').type('customer');
      cy.get('[data-testid="search-workflows"]').click();
      
      // Verify search results
      cy.get('[data-testid="workflow-row"]').each(($row) => {
        cy.wrap($row).should('contain.text', 'customer');
      });
      
      // Test status filter
      cy.get('[data-testid="status-filter"]').select('Active');
      cy.get('[data-testid="workflow-row"]').each(($row) => {
        cy.wrap($row).find('[data-testid="workflow-status"]').should('contain', 'Active');
      });
      
      // Test category filter
      cy.get('[data-testid="category-filter"]').select('Data Processing');
      cy.get('[data-testid="filtered-results"]').should('be.visible');
      
      // Test date range filter
      cy.get('[data-testid="date-filter"]').click();
      cy.get('[data-testid="start-date"]').type('2024-01-01');
      cy.get('[data-testid="end-date"]').type('2024-01-31');
      cy.get('[data-testid="apply-date-filter"]').click();
      
      // Clear all filters
      cy.get('[data-testid="clear-filters"]').click();
      cy.get('[data-testid="workflow-search"]').should('have.value', '');
      cy.get('[data-testid="status-filter"]').should('have.value', 'all');
    });

    it('TC-AN-144: Should test workflow versioning', () => {
      // Select a workflow
      cy.get('[data-testid="workflow-row"]').first().click();
      
      // Open version history
      cy.get('[data-testid="version-history"]').click();
      
      // Verify version history panel
      cy.get('[data-testid="version-history-panel"]').should('be.visible');
      cy.get('[data-testid="version-list"]').should('be.visible');
      
      // Check version entries
      cy.get('[data-testid="version-entry"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="version-entry"]').first().within(() => {
        cy.get('[data-testid="version-number"]').should('be.visible');
        cy.get('[data-testid="version-date"]').should('be.visible');
        cy.get('[data-testid="version-author"]').should('be.visible');
        cy.get('[data-testid="version-changes"]').should('be.visible');
        cy.get('[data-testid="version-actions"]').should('be.visible');
      });
      
      // Test version comparison
      cy.get('[data-testid="version-entry"]').first().find('[data-testid="compare-version"]').click();
      cy.get('[data-testid="version-entry"]').eq(1).find('[data-testid="compare-version"]').click();
      
      cy.get('[data-testid="version-comparison"]').should('be.visible');
      cy.get('[data-testid="diff-viewer"]').should('be.visible');
      
      // Test version rollback
      cy.get('[data-testid="version-entry"]').eq(1).find('[data-testid="rollback-version"]').click();
      cy.get('[data-testid="rollback-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-rollback"]').click();
      
      cy.get('[data-testid="rollback-success"]').should('be.visible');
    });

    it('TC-AN-145: Should verify workflow deployment', () => {
      // Select an active workflow
      cy.get('[data-testid="workflow-row"]').contains('Active').click();
      
      // Deploy workflow
      cy.get('[data-testid="deploy-workflow"]').click();
      
      // Verify deployment dialog
      cy.get('[data-testid="deployment-dialog"]').should('be.visible');
      cy.get('[data-testid="deployment-dialog"]').within(() => {
        cy.get('[data-testid="deployment-environment"]').should('be.visible');
        cy.get('[data-testid="deployment-strategy"]').should('be.visible');
        cy.get('[data-testid="deployment-schedule"]').should('be.visible');
        cy.get('[data-testid="deployment-notifications"]').should('be.visible');
      });
      
      // Configure deployment
      cy.get('[data-testid="deployment-environment"]').select('Production');
      cy.get('[data-testid="deployment-strategy"]').select('Blue-Green');
      cy.get('[data-testid="deployment-schedule"]').select('Immediate');
      cy.get('[data-testid="notify-on-completion"]').check();
      
      // Start deployment
      cy.intercept('POST', '/api/workflows/*/deploy').as('deployWorkflow');
      cy.get('[data-testid="start-deployment"]').click();
      
      cy.wait('@deployWorkflow');
      
      // Verify deployment progress
      cy.get('[data-testid="deployment-progress"]').should('be.visible');
      cy.get('[data-testid="deployment-status"]').should('contain', 'Deploying');
      
      // Wait for completion
      cy.get('[data-testid="deployment-complete"]').should('be.visible');
      cy.get('[data-testid="deployment-status"]').should('contain', 'Deployed');
    });

    it('TC-AN-146: Should test workflow import/export', () => {
      // Test export workflow
      cy.get('[data-testid="workflow-row"]').first().click();
      cy.get('[data-testid="export-workflow"]').click();
      
      // Configure export options
      cy.get('[data-testid="export-options"]').should('be.visible');
      cy.get('[data-testid="include-version-history"]').check();
      cy.get('[data-testid="include-execution-logs"]').check();
      cy.get('[data-testid="export-format-json"]').click();
      
      cy.get('[data-testid="start-export"]').click();
      
      // Verify export download
      cy.get('[data-testid="export-complete"]').should('be.visible');
      cy.readFile('cypress/downloads/workflow-export.json').should('exist');
      
      // Test import workflow
      cy.get('[data-testid="import-workflow"]').click();
      
      // Upload workflow file
      cy.get('[data-testid="workflow-upload"]').selectFile('cypress/fixtures/sample-workflow.json');
      
      // Configure import options
      cy.get('[data-testid="import-options"]').within(() => {
        cy.get('[data-testid="import-name"]').type('Imported Workflow');
        cy.get('[data-testid="resolve-conflicts"]').select('Create New');
        cy.get('[data-testid="validate-on-import"]').check();
      });
      
      // Import workflow
      cy.intercept('POST', '/api/workflows/import').as('importWorkflow');
      cy.get('[data-testid="start-import"]').click();
      
      cy.wait('@importWorkflow');
      
      // Verify import success
      cy.get('[data-testid="import-success"]').should('be.visible');
      cy.get('[data-testid="workflows-table"]').should('contain', 'Imported Workflow');
    });
  });

  describe('TC-AN-147 to TC-AN-152: Execution Engine Tests', () => {
    it('TC-AN-147: Should test workflow execution monitoring', () => {
      // Navigate to executions tab
      cy.get('[data-testid="tab-executions"]').click();
      
      // Verify executions list
      cy.get('[data-testid="executions-table"]').should('be.visible');
      cy.get('[data-testid="execution-row"]').should('have.length.greaterThan', 0);
      
      // Check execution details
      cy.get('[data-testid="execution-row"]').first().within(() => {
        cy.get('[data-testid="execution-id"]').should('not.be.empty');
        cy.get('[data-testid="workflow-name"]').should('be.visible');
        cy.get('[data-testid="execution-status"]').should('be.visible');
        cy.get('[data-testid="start-time"]').should('be.visible');
        cy.get('[data-testid="duration"]').should('be.visible');
      });
      
      // Test execution details view
      cy.get('[data-testid="execution-row"]').first().click();
      cy.get('[data-testid="execution-details"]').should('be.visible');
      
      cy.get('[data-testid="execution-details"]').within(() => {
        cy.get('[data-testid="execution-timeline"]').should('be.visible');
        cy.get('[data-testid="step-execution-log"]').should('be.visible');
        cy.get('[data-testid="execution-metrics"]').should('be.visible');
        cy.get('[data-testid="error-details"]').should('exist');
      });
      
      // Test real-time updates for running executions
      cy.get('[data-testid="execution-row"]').contains('Running').then(($runningRow) => {
        if ($runningRow.length > 0) {
          cy.wrap($runningRow).click();
          cy.get('[data-testid="real-time-updates"]').should('be.visible');
          cy.get('[data-testid="progress-indicator"]').should('be.visible');
        }
      });
    });

    it('TC-AN-148: Should verify workflow step execution tracking', () => {
      // Select a completed execution
      cy.get('[data-testid="tab-executions"]').click();
      cy.get('[data-testid="execution-row"]').contains('Completed').first().click();
      
      // Verify step execution details
      cy.get('[data-testid="step-execution-log"]').should('be.visible');
      cy.get('[data-testid="execution-step"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="execution-step"]').first().within(() => {
        cy.get('[data-testid="step-name"]').should('be.visible');
        cy.get('[data-testid="step-status"]').should('be.visible');
        cy.get('[data-testid="step-start-time"]').should('be.visible');
        cy.get('[data-testid="step-duration"]').should('be.visible');
        cy.get('[data-testid="step-input"]').should('be.visible');
        cy.get('[data-testid="step-output"]').should('be.visible');
      });
      
      // Test step details expansion
      cy.get('[data-testid="execution-step"]').first().click();
      cy.get('[data-testid="step-details-expanded"]').should('be.visible');
      
      // Verify step input/output data
      cy.get('[data-testid="step-input-data"]').should('not.be.empty');
      cy.get('[data-testid="step-output-data"]').should('not.be.empty');
      
      // Test step retry functionality
      cy.get('[data-testid="execution-step"]').contains('Failed').then(($failedStep) => {
        if ($failedStep.length > 0) {
          cy.wrap($failedStep).click();
          cy.get('[data-testid="retry-step"]').should('be.visible');
          cy.get('[data-testid="retry-step"]').click();
          
          cy.get('[data-testid="retry-confirmation"]').should('be.visible');
          cy.get('[data-testid="confirm-retry"]').click();
          
          cy.get('[data-testid="step-retry-initiated"]').should('be.visible');
        }
      });
    });

    it('TC-AN-149: Should test workflow pause and resume', () => {
      // Find a running workflow execution
      cy.get('[data-testid="tab-executions"]').click();
      cy.get('[data-testid="execution-row"]').contains('Running').then(($runningRow) => {
        if ($runningRow.length > 0) {
          cy.wrap($runningRow).click();
          
          // Test pause functionality
          cy.get('[data-testid="pause-execution"]').click();
          cy.get('[data-testid="pause-confirmation"]').should('be.visible');
          cy.get('[data-testid="confirm-pause"]').click();
          
          // Verify execution paused
          cy.get('[data-testid="execution-status"]').should('contain', 'Paused');
          cy.get('[data-testid="pause-indicator"]').should('be.visible');
          
          // Test resume functionality
          cy.get('[data-testid="resume-execution"]').click();
          cy.get('[data-testid="resume-confirmation"]').should('be.visible');
          cy.get('[data-testid="confirm-resume"]').click();
          
          // Verify execution resumed
          cy.get('[data-testid="execution-status"]').should('contain', 'Running');
          cy.get('[data-testid="progress-indicator"]').should('be.visible');
        } else {
          // If no running execution, start a test workflow
          cy.get('[data-testid="tab-workflows"]').click();
          cy.get('[data-testid="workflow-row"]').first().find('[data-testid="start-workflow"]').click();
          cy.get('[data-testid="execution-started"]').should('be.visible');
        }
      });
    });

    it('TC-AN-150: Should verify error handling and retry logic', () => {
      // Navigate to failed executions
      cy.get('[data-testid="tab-executions"]').click();
      cy.get('[data-testid="status-filter"]').select('Failed');
      
      // Select a failed execution
      cy.get('[data-testid="execution-row"]').first().click();
      
      // Verify error details
      cy.get('[data-testid="error-details"]').should('be.visible');
      cy.get('[data-testid="error-details"]').within(() => {
        cy.get('[data-testid="error-message"]').should('not.be.empty');
        cy.get('[data-testid="error-type"]').should('be.visible');
        cy.get('[data-testid="error-timestamp"]').should('be.visible');
        cy.get('[data-testid="error-stack-trace"]').should('be.visible');
      });
      
      // Test retry with different parameters
      cy.get('[data-testid="retry-execution"]').click();
      
      cy.get('[data-testid="retry-options"]').should('be.visible');
      cy.get('[data-testid="retry-options"]').within(() => {
        cy.get('[data-testid="retry-from-failed"]').should('be.visible');
        cy.get('[data-testid="retry-from-beginning"]').should('be.visible');
        cy.get('[data-testid="modify-parameters"]').should('be.visible');
      });
      
      // Configure retry
      cy.get('[data-testid="retry-from-failed"]').click();
      cy.get('[data-testid="modify-parameters"]').check();
      
      // Modify retry parameters
      cy.get('[data-testid="parameter-editor"]').within(() => {
        cy.get('[data-testid="timeout-value"]').clear().type('300');
        cy.get('[data-testid="retry-attempts"]').clear().type('3');
      });
      
      // Start retry
      cy.intercept('POST', '/api/workflows/executions/*/retry').as('retryExecution');
      cy.get('[data-testid="start-retry"]').click();
      
      cy.wait('@retryExecution');
      cy.get('[data-testid="retry-started"]').should('be.visible');
    });

    it('TC-AN-151: Should test workflow performance metrics', () => {
      // Navigate to performance tab
      cy.get('[data-testid="tab-performance"]').click();
      
      // Verify performance dashboard
      cy.get('[data-testid="performance-dashboard"]').should('be.visible');
      
      // Check performance metrics
      cy.get('[data-testid="performance-metrics"]').within(() => {
        cy.get('[data-testid="avg-execution-time"]').should('be.visible');
        cy.get('[data-testid="success-rate"]').should('be.visible');
        cy.get('[data-testid="throughput-rate"]').should('be.visible');
        cy.get('[data-testid="error-rate"]').should('be.visible');
      });
      
      // Verify performance charts
      cy.get('[data-testid="execution-time-chart"]').should('be.visible');
      cy.get('[data-testid="throughput-chart"]').should('be.visible');
      cy.get('[data-testid="success-rate-trend"]').should('be.visible');
      
      // Test chart interactions
      cy.get('[data-testid="execution-time-chart"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
      
      // Test time range selection
      cy.get('[data-testid="performance-time-range"]').select('Last 7 days');
      cy.get('[data-testid="charts-loading"]').should('be.visible');
      cy.get('[data-testid="charts-loading"]').should('not.exist');
      
      // Test performance alerts
      cy.get('[data-testid="performance-alerts"]').should('be.visible');
      cy.get('[data-testid="alert-item"]').then(($alerts) => {
        if ($alerts.length > 0) {
          cy.wrap($alerts).first().should('contain', 'Performance');
        }
      });
    });

    it('TC-AN-152: Should verify workflow templates and library', () => {
      // Navigate to templates
      cy.get('[data-testid="tab-templates"]').click();
      
      // Verify template library
      cy.get('[data-testid="template-library"]').should('be.visible');
      cy.get('[data-testid="template-categories"]').should('be.visible');
      
      // Check template categories
      const categories = ['Data Processing', 'Integration', 'Monitoring', 'Automation'];
      categories.forEach(category => {
        cy.get('[data-testid="template-categories"]').should('contain', category);
      });
      
      // Test template selection
      cy.get('[data-testid="template-card"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="template-card"]').first().within(() => {
        cy.get('[data-testid="template-name"]').should('be.visible');
        cy.get('[data-testid="template-description"]').should('be.visible');
        cy.get('[data-testid="template-preview"]').should('be.visible');
        cy.get('[data-testid="use-template"]').should('be.visible');
      });
      
      // Test template preview
      cy.get('[data-testid="template-card"]').first().find('[data-testid="template-preview"]').click();
      cy.get('[data-testid="template-preview-modal"]').should('be.visible');
      cy.get('[data-testid="template-diagram"]').should('be.visible');
      cy.get('[data-testid="template-details"]').should('be.visible');
      
      // Close preview
      cy.get('[data-testid="close-preview"]').click();
      cy.get('[data-testid="template-preview-modal"]').should('not.exist');
      
      // Test create workflow from template
      cy.get('[data-testid="template-card"]').first().find('[data-testid="use-template"]').click();
      
      // Configure template parameters
      cy.get('[data-testid="template-configuration"]').should('be.visible');
      cy.get('[data-testid="template-configuration"]').within(() => {
        cy.get('[data-testid="workflow-name"]').type('My Custom Workflow');
        cy.get('[data-testid="workflow-description"]').type('Created from template');
      });
      
      // Create workflow from template
      cy.get('[data-testid="create-from-template"]').click();
      cy.get('[data-testid="workflow-created"]').should('be.visible');
      
      // Verify redirected to designer
      cy.get('[data-testid="workflow-designer"]').should('be.visible');
      cy.get('[data-testid="workflow-node"]').should('have.length.greaterThan', 0);
    });
  });
});