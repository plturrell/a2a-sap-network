/**
 * Operations.view.xml Component Tests
 * Test Cases: TC-AN-106 to TC-AN-120
 * Coverage: Operations Management, Workflow Execution, Monitoring, Scheduling, Performance
 */

describe('Operations.view.xml - Operations Management', () => {
  beforeEach(() => {
    cy.visit('/operations');
    cy.viewport(1280, 720);
    
    // Wait for operations dashboard to load
    cy.get('[data-testid="operations-dashboard"]').should('be.visible');
  });

  describe('TC-AN-106 to TC-AN-110: Operation Management Tests', () => {
    it('TC-AN-106: Should verify operations list display', () => {
      // Verify operations table loads
      cy.get('[data-testid="operations-table"]').should('be.visible');
      
      // Check table headers
      cy.get('[data-testid="operations-table"]').within(() => {
        cy.get('[data-testid="header-operation-id"]').should('be.visible');
        cy.get('[data-testid="header-operation-type"]').should('be.visible');
        cy.get('[data-testid="header-status"]').should('be.visible');
        cy.get('[data-testid="header-created"]').should('be.visible');
        cy.get('[data-testid="header-actions"]').should('be.visible');
      });
      
      // Verify operation entries
      cy.get('[data-testid="operation-row"]').should('have.length.greaterThan', 0);
      
      // Check operation data structure
      cy.get('[data-testid="operation-row"]').first().within(() => {
        cy.get('[data-testid="operation-id"]').should('not.be.empty');
        cy.get('[data-testid="operation-type"]').should('be.visible');
        cy.get('[data-testid="operation-status"]').should('be.visible');
        cy.get('[data-testid="operation-created"]').should('match', /\d{4}-\d{2}-\d{2}/);
      });
    });

    it('TC-AN-107: Should test create new operation dialog', () => {
      cy.get('[data-testid="create-operation-button"]').click();
      
      // Verify dialog opens
      cy.get('[data-testid="create-operation-dialog"]').should('be.visible');
      
      // Check form fields
      cy.get('[data-testid="create-operation-dialog"]').within(() => {
        cy.get('[data-testid="operation-name"]').should('be.visible');
        cy.get('[data-testid="operation-type-select"]').should('be.visible');
        cy.get('[data-testid="operation-description"]').should('be.visible');
        cy.get('[data-testid="priority-select"]').should('be.visible');
        cy.get('[data-testid="target-agents-select"]').should('be.visible');
        cy.get('[data-testid="scheduled-time"]').should('be.visible');
        cy.get('[data-testid="parameters-editor"]').should('be.visible');
      });
      
      // Test form validation
      cy.get('[data-testid="create-operation-submit"]').click();
      cy.get('[data-testid="name-error"]').should('be.visible');
      
      // Fill form with valid data
      cy.get('[data-testid="operation-name"]').type('Test Operation');
      cy.get('[data-testid="operation-type-select"]').select('Data Sync');
      cy.get('[data-testid="operation-description"]').type('Test operation for validation');
      cy.get('[data-testid="priority-select"]').select('Medium');
      cy.get('[data-testid="target-agents-select"]').select(['Agent1', 'Agent2']);
      
      // Submit form
      cy.intercept('POST', '/api/operations').as('createOperation');
      cy.get('[data-testid="create-operation-submit"]').click();
      
      // Verify operation created
      cy.wait('@createOperation');
      cy.get('[data-testid="create-success-message"]').should('be.visible');
      cy.get('[data-testid="create-operation-dialog"]').should('not.exist');
    });

    it('TC-AN-108: Should verify operation status filters', () => {
      cy.get('[data-testid="status-filter"]').should('be.visible');
      
      // Test different status filters
      const statusOptions = ['All', 'Running', 'Completed', 'Failed', 'Pending', 'Cancelled'];
      
      statusOptions.forEach(status => {
        cy.get('[data-testid="status-filter"]').select(status);
        
        // Verify filter applied
        cy.get('[data-testid="filter-applied-badge"]').should('contain', status);
        
        // Check filtered results
        if (status !== 'All') {
          cy.get('[data-testid="operation-row"]').each(($row) => {
            cy.wrap($row).find('[data-testid="operation-status"]')
              .should('contain', status);
          });
        }
      });
      
      // Test multiple filters
      cy.get('[data-testid="advanced-filters"]').click();
      cy.get('[data-testid="type-filter"]').select('Data Processing');
      cy.get('[data-testid="priority-filter"]').select('High');
      cy.get('[data-testid="apply-filters"]').click();
      
      // Verify combined filters
      cy.get('[data-testid="active-filters"]').should('contain', 'Data Processing');
      cy.get('[data-testid="active-filters"]').should('contain', 'High');
    });

    it('TC-AN-109: Should test operation search functionality', () => {
      cy.get('[data-testid="operations-search"]').should('be.visible');
      
      // Test search by operation name
      cy.get('[data-testid="operations-search"]').type('sync');
      cy.get('[data-testid="search-button"]').click();
      
      // Verify search results
      cy.get('[data-testid="operation-row"]').each(($row) => {
        cy.wrap($row).should('contain.text', 'sync');
      });
      
      // Test search by operation ID
      cy.get('[data-testid="operations-search"]').clear().type('OP-12345');
      cy.get('[data-testid="search-button"]').click();
      
      cy.get('[data-testid="operation-row"]').should('have.length', 1);
      cy.get('[data-testid="operation-id"]').should('contain', 'OP-12345');
      
      // Test search suggestions
      cy.get('[data-testid="operations-search"]').clear().type('dat');
      cy.get('[data-testid="search-suggestions"]').should('be.visible');
      cy.get('[data-testid="suggestion-item"]').should('have.length.greaterThan', 0);
      
      // Select suggestion
      cy.get('[data-testid="suggestion-item"]').first().click();
      cy.get('[data-testid="operations-search"]').should('not.have.value', 'dat');
      
      // Clear search
      cy.get('[data-testid="clear-search"]').click();
      cy.get('[data-testid="operations-search"]').should('have.value', '');
    });

    it('TC-AN-110: Should verify pagination controls', () => {
      // Mock large dataset
      cy.intercept('GET', '/api/operations*', {
        fixture: 'large-operations-list.json'
      }).as('getOperations');
      
      cy.reload();
      cy.wait('@getOperations');
      
      // Verify pagination appears
      cy.get('[data-testid="pagination"]').should('be.visible');
      
      // Check pagination info
      cy.get('[data-testid="pagination-info"]').should('contain', 'Showing 1-20 of');
      cy.get('[data-testid="total-operations"]').should('not.contain', '0');
      
      // Test next page
      cy.get('[data-testid="next-page"]').click();
      cy.get('[data-testid="pagination-info"]').should('contain', 'Showing 21-40 of');
      
      // Test previous page
      cy.get('[data-testid="previous-page"]').click();
      cy.get('[data-testid="pagination-info"]').should('contain', 'Showing 1-20 of');
      
      // Test page size selector
      cy.get('[data-testid="page-size-select"]').select('50');
      cy.get('[data-testid="pagination-info"]').should('contain', 'Showing 1-50 of');
      
      // Test direct page navigation
      cy.get('[data-testid="page-3"]').click();
      cy.get('[data-testid="current-page"]').should('contain', '3');
    });
  });

  describe('TC-AN-111 to TC-AN-115: Workflow Execution Tests', () => {
    it('TC-AN-111: Should test start operation execution', () => {
      // Select an operation
      cy.get('[data-testid="operation-row"]').first().click();
      
      // Start execution
      cy.get('[data-testid="start-operation"]').click();
      
      // Verify confirmation dialog
      cy.get('[data-testid="start-confirmation-dialog"]').should('be.visible');
      cy.get('[data-testid="start-confirmation-dialog"]').within(() => {
        cy.get('[data-testid="operation-details"]').should('be.visible');
        cy.get('[data-testid="target-agents-list"]').should('be.visible');
        cy.get('[data-testid="estimated-duration"]').should('be.visible');
      });
      
      // Confirm start
      cy.intercept('POST', '/api/operations/*/start').as('startOperation');
      cy.get('[data-testid="confirm-start"]').click();
      
      // Verify operation started
      cy.wait('@startOperation');
      cy.get('[data-testid="start-success-message"]').should('be.visible');
      
      // Check status update
      cy.get('[data-testid="operation-status"]').should('contain', 'Running');
      cy.get('[data-testid="execution-indicator"]').should('be.visible');
    });

    it('TC-AN-112: Should verify operation progress tracking', () => {
      // Select running operation
      cy.get('[data-testid="operation-row"]').contains('Running').click();
      
      // Open progress view
      cy.get('[data-testid="view-progress"]').click();
      
      // Verify progress panel
      cy.get('[data-testid="progress-panel"]').should('be.visible');
      cy.get('[data-testid="progress-panel"]').within(() => {
        cy.get('[data-testid="progress-bar"]').should('be.visible');
        cy.get('[data-testid="progress-percentage"]').should('match', /\d{1,3}%/);
        cy.get('[data-testid="current-step"]').should('be.visible');
        cy.get('[data-testid="steps-completed"]').should('be.visible');
        cy.get('[data-testid="estimated-completion"]').should('be.visible');
      });
      
      // Verify step details
      cy.get('[data-testid="step-details"]').should('be.visible');
      cy.get('[data-testid="step-item"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="step-item"]').first().within(() => {
        cy.get('[data-testid="step-name"]').should('be.visible');
        cy.get('[data-testid="step-status"]').should('be.visible');
        cy.get('[data-testid="step-duration"]').should('be.visible');
      });
      
      // Test auto-refresh
      cy.get('[data-testid="progress-percentage"]').invoke('text').then((initialProgress) => {
        cy.wait(3000);
        cy.get('[data-testid="progress-percentage"]').should('not.contain', initialProgress);
      });
    });

    it('TC-AN-113: Should test pause/resume functionality', () => {
      // Select running operation
      cy.get('[data-testid="operation-row"]').contains('Running').click();
      
      // Pause operation
      cy.get('[data-testid="pause-operation"]').click();
      
      // Verify pause confirmation
      cy.get('[data-testid="pause-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-pause"]').click();
      
      // Verify operation paused
      cy.intercept('POST', '/api/operations/*/pause').as('pauseOperation');
      cy.wait('@pauseOperation');
      
      cy.get('[data-testid="operation-status"]').should('contain', 'Paused');
      cy.get('[data-testid="pause-indicator"]').should('be.visible');
      
      // Resume operation
      cy.get('[data-testid="resume-operation"]').click();
      cy.get('[data-testid="confirm-resume"]').click();
      
      cy.intercept('POST', '/api/operations/*/resume').as('resumeOperation');
      cy.wait('@resumeOperation');
      
      // Verify operation resumed
      cy.get('[data-testid="operation-status"]').should('contain', 'Running');
      cy.get('[data-testid="execution-indicator"]').should('be.visible');
    });

    it('TC-AN-114: Should verify stop operation functionality', () => {
      // Select running operation
      cy.get('[data-testid="operation-row"]').contains('Running').click();
      
      // Stop operation
      cy.get('[data-testid="stop-operation"]').click();
      
      // Verify stop warning
      cy.get('[data-testid="stop-warning-dialog"]').should('be.visible');
      cy.get('[data-testid="stop-warning-dialog"]').within(() => {
        cy.get('[data-testid="warning-message"]').should('contain', 'This will stop the operation immediately');
        cy.get('[data-testid="impact-warning"]').should('be.visible');
        cy.get('[data-testid="confirm-checkbox"]').should('be.visible');
      });
      
      // Confirm stop
      cy.get('[data-testid="confirm-checkbox"]').check();
      cy.get('[data-testid="confirm-stop"]').click();
      
      cy.intercept('POST', '/api/operations/*/stop').as('stopOperation');
      cy.wait('@stopOperation');
      
      // Verify operation stopped
      cy.get('[data-testid="operation-status"]').should('contain', 'Stopped');
      cy.get('[data-testid="stop-reason"]').should('be.visible');
    });

    it('TC-AN-115: Should test operation retry mechanism', () => {
      // Select failed operation
      cy.get('[data-testid="operation-row"]').contains('Failed').click();
      
      // Retry operation
      cy.get('[data-testid="retry-operation"]').click();
      
      // Verify retry options
      cy.get('[data-testid="retry-options-dialog"]').should('be.visible');
      cy.get('[data-testid="retry-options-dialog"]').within(() => {
        cy.get('[data-testid="retry-type-full"]').should('be.visible');
        cy.get('[data-testid="retry-type-failed-steps"]').should('be.visible');
        cy.get('[data-testid="retry-parameters"]').should('be.visible');
      });
      
      // Select retry type
      cy.get('[data-testid="retry-type-failed-steps"]').click();
      
      // Configure retry parameters
      cy.get('[data-testid="max-retries"]').clear().type('3');
      cy.get('[data-testid="retry-delay"]').clear().type('30');
      
      // Start retry
      cy.intercept('POST', '/api/operations/*/retry').as('retryOperation');
      cy.get('[data-testid="start-retry"]').click();
      
      cy.wait('@retryOperation');
      
      // Verify retry started
      cy.get('[data-testid="retry-success-message"]').should('be.visible');
      cy.get('[data-testid="operation-status"]').should('contain', 'Retrying');
    });
  });

  describe('TC-AN-116 to TC-AN-120: Monitoring and Performance Tests', () => {
    it('TC-AN-116: Should verify operation logs display', () => {
      // Select operation
      cy.get('[data-testid="operation-row"]').first().click();
      
      // Open logs tab
      cy.get('[data-testid="tab-logs"]').click();
      
      // Verify logs viewer
      cy.get('[data-testid="logs-viewer"]').should('be.visible');
      cy.get('[data-testid="logs-content"]').should('be.visible');
      
      // Check log entries
      cy.get('[data-testid="log-entry"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="log-entry"]').first().within(() => {
        cy.get('[data-testid="log-timestamp"]').should('be.visible');
        cy.get('[data-testid="log-level"]').should('be.visible');
        cy.get('[data-testid="log-source"]').should('be.visible');
        cy.get('[data-testid="log-message"]').should('be.visible');
      });
      
      // Test log filtering
      cy.get('[data-testid="log-level-filter"]').select('ERROR');
      cy.get('[data-testid="log-entry"]').each(($entry) => {
        cy.wrap($entry).find('[data-testid="log-level"]').should('contain', 'ERROR');
      });
      
      // Test log search
      cy.get('[data-testid="log-search"]').type('connection');
      cy.get('[data-testid="log-entry"]').each(($entry) => {
        cy.wrap($entry).should('contain.text', 'connection');
      });
      
      // Test auto-refresh
      cy.get('[data-testid="auto-refresh-logs"]').check();
      cy.get('[data-testid="logs-streaming-indicator"]').should('be.visible');
    });

    it('TC-AN-117: Should test export operation data', () => {
      // Select operation
      cy.get('[data-testid="operation-row"]').first().click();
      
      // Open export menu
      cy.get('[data-testid="export-operation-data"]').click();
      
      // Verify export options
      cy.get('[data-testid="export-options-dialog"]').should('be.visible');
      cy.get('[data-testid="export-options-dialog"]').within(() => {
        cy.get('[data-testid="export-summary"]').should('be.visible');
        cy.get('[data-testid="export-logs"]').should('be.visible');
        cy.get('[data-testid="export-metrics"]').should('be.visible');
        cy.get('[data-testid="export-format-json"]').should('be.visible');
        cy.get('[data-testid="export-format-csv"]').should('be.visible');
        cy.get('[data-testid="export-format-pdf"]').should('be.visible');
      });
      
      // Configure export
      cy.get('[data-testid="export-summary"]').check();
      cy.get('[data-testid="export-logs"]').check();
      cy.get('[data-testid="export-format-json"]').click();
      
      // Start export
      cy.get('[data-testid="start-export"]').click();
      
      // Verify export progress
      cy.get('[data-testid="export-progress"]').should('be.visible');
      cy.get('[data-testid="export-complete"]').should('be.visible');
      
      // Verify download
      cy.readFile('cypress/downloads/operation-data.json').should('exist');
    });

    it('TC-AN-118: Should verify operations dashboard metrics', () => {
      // Navigate to operations dashboard
      cy.get('[data-testid="dashboard-tab"]').click();
      
      // Verify dashboard widgets
      cy.get('[data-testid="operations-dashboard"]').within(() => {
        cy.get('[data-testid="total-operations"]').should('be.visible');
        cy.get('[data-testid="running-operations"]').should('be.visible');
        cy.get('[data-testid="completed-operations"]').should('be.visible');
        cy.get('[data-testid="failed-operations"]').should('be.visible');
        cy.get('[data-testid="success-rate"]').should('be.visible');
      });
      
      // Verify charts
      cy.get('[data-testid="operations-timeline-chart"]').should('be.visible');
      cy.get('[data-testid="status-distribution-chart"]').should('be.visible');
      cy.get('[data-testid="performance-metrics-chart"]').should('be.visible');
      
      // Test chart interactions
      cy.get('[data-testid="operations-timeline-chart"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
      
      // Test time range selector
      cy.get('[data-testid="time-range-selector"]').select('Last 7 days');
      cy.get('[data-testid="charts-loading"]').should('be.visible');
      cy.get('[data-testid="charts-loading"]').should('not.exist');
    });

    it('TC-AN-119: Should test performance benchmarking', () => {
      // Navigate to performance tab
      cy.get('[data-testid="tab-performance"]').click();
      
      // Verify performance metrics
      cy.get('[data-testid="performance-metrics"]').should('be.visible');
      cy.get('[data-testid="performance-metrics"]').within(() => {
        cy.get('[data-testid="avg-execution-time"]').should('be.visible');
        cy.get('[data-testid="throughput-rate"]').should('be.visible');
        cy.get('[data-testid="resource-utilization"]').should('be.visible');
        cy.get('[data-testid="error-rate"]').should('be.visible');
      });
      
      // Test benchmark comparison
      cy.get('[data-testid="run-benchmark"]').click();
      
      // Configure benchmark
      cy.get('[data-testid="benchmark-config"]').within(() => {
        cy.get('[data-testid="operation-type"]').select('Data Processing');
        cy.get('[data-testid="sample-size"]').clear().type('100');
        cy.get('[data-testid="concurrent-executions"]').clear().type('5');
      });
      
      // Start benchmark
      cy.intercept('POST', '/api/operations/benchmark').as('runBenchmark');
      cy.get('[data-testid="start-benchmark"]').click();
      
      cy.wait('@runBenchmark');
      
      // Verify benchmark results
      cy.get('[data-testid="benchmark-results"]').should('be.visible');
      cy.get('[data-testid="benchmark-results"]').within(() => {
        cy.get('[data-testid="min-time"]').should('be.visible');
        cy.get('[data-testid="max-time"]').should('be.visible');
        cy.get('[data-testid="median-time"]').should('be.visible');
        cy.get('[data-testid="percentile-95"]').should('be.visible');
      });
    });

    it('TC-AN-120: Should verify scheduled operations management', () => {
      // Navigate to scheduled operations
      cy.get('[data-testid="tab-scheduled"]').click();
      
      // Verify scheduled operations list
      cy.get('[data-testid="scheduled-operations"]').should('be.visible');
      cy.get('[data-testid="scheduled-operation-item"]').should('have.length.greaterThan', 0);
      
      // Create new scheduled operation
      cy.get('[data-testid="create-schedule"]').click();
      
      // Configure schedule
      cy.get('[data-testid="schedule-dialog"]').within(() => {
        cy.get('[data-testid="schedule-name"]').type('Daily Data Sync');
        cy.get('[data-testid="operation-template"]').select('Data Synchronization');
        cy.get('[data-testid="schedule-type"]').select('Recurring');
        cy.get('[data-testid="cron-expression"]').type('0 2 * * *');
        cy.get('[data-testid="schedule-enabled"]').check();
      });
      
      // Save schedule
      cy.intercept('POST', '/api/operations/schedules').as('createSchedule');
      cy.get('[data-testid="save-schedule"]').click();
      
      cy.wait('@createSchedule');
      
      // Verify schedule created
      cy.get('[data-testid="schedule-success-message"]').should('be.visible');
      cy.get('[data-testid="scheduled-operation-item"]').should('contain', 'Daily Data Sync');
      
      // Test schedule management
      cy.get('[data-testid="scheduled-operation-item"]').first().within(() => {
        cy.get('[data-testid="next-run-time"]').should('be.visible');
        cy.get('[data-testid="last-run-status"]').should('be.visible');
        cy.get('[data-testid="schedule-actions"]').should('be.visible');
      });
      
      // Test disable/enable schedule
      cy.get('[data-testid="toggle-schedule"]').click();
      cy.get('[data-testid="schedule-status"]').should('contain', 'Disabled');
      
      cy.get('[data-testid="toggle-schedule"]').click();
      cy.get('[data-testid="schedule-status"]').should('contain', 'Enabled');
    });
  });
});