/**
 * Agents.view.xml Component Tests
 * Test Cases: TC-AN-037 to TC-AN-062
 * Coverage: Agent List, Registration, Actions, Search/Filter, Export
 */

describe('Agents.view.xml - Agent Management', () => {
  beforeEach(() => {
    cy.visit('/agents');
    cy.viewport(1280, 720);
    
    // Wait for agent list to load
    cy.get('[data-testid="agents-table"]').should('be.visible');
  });

  describe('TC-AN-037 to TC-AN-044: Agent List View', () => {
    it('TC-AN-037: Should display agent table with correct columns', () => {
      // Check table structure
      const expectedColumns = [
        'checkbox',
        'name',
        'type', 
        'status',
        'health',
        'last-active',
        'actions'
      ];
      
      expectedColumns.forEach(col => {
        cy.get(`[data-testid="column-${col}"]`).should('be.visible');
      });
    });

    it('TC-AN-038: Should implement default sorting', () => {
      // Default sort by name A-Z
      cy.get('[data-testid="column-name"]').should('have.attr', 'aria-sort', 'ascending');
      
      // Verify alphabetical order
      let previousName = '';
      cy.get('[data-testid^="agent-name-"]').each(($el) => {
        const currentName = $el.text();
        if (previousName) {
          expect(currentName.localeCompare(previousName)).to.be.greaterThan(-1);
        }
        previousName = currentName;
      });
    });

    it('TC-AN-039: Should handle column sorting', () => {
      // Sort by status
      cy.get('[data-testid="column-status"]').click();
      cy.get('[data-testid="column-status"]').should('have.attr', 'aria-sort', 'ascending');
      
      // Sort descending
      cy.get('[data-testid="column-status"]').click();
      cy.get('[data-testid="column-status"]').should('have.attr', 'aria-sort', 'descending');
      
      // Remove sort
      cy.get('[data-testid="column-status"]').click();
      cy.get('[data-testid="column-status"]').should('have.attr', 'aria-sort', 'none');
    });

    it('TC-AN-040: Should handle pagination', () => {
      // Check pagination controls
      cy.get('[data-testid="pagination"]').should('be.visible');
      cy.get('[data-testid="page-size-selector"]').should('contain', '25');
      
      // Change page size
      cy.get('[data-testid="page-size-selector"]').select('50');
      cy.get('[data-testid="agents-table"] tbody tr').should('have.length.lte', 50);
      
      // Navigate pages
      cy.get('[data-testid="page-next"]').click();
      cy.get('[data-testid="current-page"]').should('contain', '2');
    });

    it('TC-AN-041: Should select agents with checkboxes', () => {
      // Select individual
      cy.get('[data-testid="select-agent-1"]').check();
      cy.get('[data-testid="selected-count"]').should('contain', '1 selected');
      
      // Select all on page
      cy.get('[data-testid="select-all"]').check();
      cy.get('[data-testid="selected-count"]').should('contain', '25 selected');
      
      // Bulk actions should appear
      cy.get('[data-testid="bulk-actions"]').should('be.visible');
    });

    it('TC-AN-042: Should implement table virtualization for large datasets', () => {
      // Mock large dataset
      cy.intercept('GET', '/api/agents?limit=100', {
        fixture: 'large-agent-list.json' // 1000+ agents
      }).as('getLargeList');
      
      cy.get('[data-testid="page-size-selector"]').select('100');
      cy.wait('@getLargeList');
      
      // Check virtual scrolling
      cy.get('[data-testid="agents-table-container"]').scrollTo(0, 1000);
      
      // Verify performance
      cy.window().then((win) => {
        const entries = win.performance.getEntriesByType('measure');
        const renderTime = entries.find(e => e.name === 'table-render');
        expect(renderTime.duration).to.be.lessThan(100); // 100ms max
      });
    });

    it('TC-AN-043: Should display agent status indicators', () => {
      // Check status badges
      cy.get('[data-testid^="agent-status-"]').each(($status) => {
        const statusClass = $status.attr('class');
        expect(statusClass).to.match(/status-(active|inactive|error)/);
        
        // Verify colors
        if (statusClass.includes('active')) {
          cy.wrap($status).should('have.css', 'background-color', 'rgb(0, 128, 0)');
        }
      });
    });

    it('TC-AN-044: Should show agent health metrics', () => {
      cy.get('[data-testid^="agent-health-"]').each(($health) => {
        // Check health percentage
        const healthText = $health.text();
        expect(healthText).to.match(/\d+%/);
        
        // Check color coding
        const healthValue = parseInt(healthText);
        if (healthValue > 95) {
          cy.wrap($health).should('have.class', 'health-good');
        } else if (healthValue > 80) {
          cy.wrap($health).should('have.class', 'health-warning');
        } else {
          cy.wrap($health).should('have.class', 'health-critical');
        }
      });
    });
  });

  describe('TC-AN-045 to TC-AN-050: Agent Registration', () => {
    it('TC-AN-045: Should open registration wizard', () => {
      cy.get('[data-testid="register-new-agent"]').click();
      cy.get('[data-testid="registration-wizard"]').should('be.visible');
      
      // Check wizard steps
      cy.get('[data-testid="wizard-step-1"]').should('have.class', 'active');
      cy.get('[data-testid="wizard-step-2"]').should('exist');
      cy.get('[data-testid="wizard-step-3"]').should('exist');
    });

    it('TC-AN-046: Should validate registration form', () => {
      cy.get('[data-testid="register-new-agent"]').click();
      
      // Try to proceed without filling required fields
      cy.get('[data-testid="wizard-next"]').click();
      
      // Validation errors
      cy.get('[data-testid="error-agent-name"]').should('contain', 'Name is required');
      cy.get('[data-testid="error-agent-type"]').should('contain', 'Type is required');
      
      // Fill required fields
      cy.get('[data-testid="agent-name"]').type('Test Agent');
      cy.get('[data-testid="agent-type"]').select('data-processor');
      cy.get('[data-testid="agent-endpoint"]').type('http://localhost:8080');
      
      // Proceed
      cy.get('[data-testid="wizard-next"]').click();
      cy.get('[data-testid="wizard-step-2"]').should('have.class', 'active');
    });

    it('TC-AN-047: Should configure agent capabilities', () => {
      // Navigate to step 2
      cy.get('[data-testid="register-new-agent"]').click();
      cy.get('[data-testid="agent-name"]').type('Test Agent');
      cy.get('[data-testid="agent-type"]').select('data-processor');
      cy.get('[data-testid="wizard-next"]').click();
      
      // Select capabilities
      cy.get('[data-testid="capability-data-transform"]').check();
      cy.get('[data-testid="capability-real-time"]').check();
      cy.get('[data-testid="capability-batch-processing"]').check();
      
      // Add custom capability
      cy.get('[data-testid="add-custom-capability"]').click();
      cy.get('[data-testid="custom-capability-name"]').type('custom-analytics');
      cy.get('[data-testid="custom-capability-add"]').click();
      
      cy.get('[data-testid="selected-capabilities"]').should('contain', '4 capabilities');
    });

    it('TC-AN-048: Should review and submit registration', () => {
      // Complete wizard steps
      cy.get('[data-testid="register-new-agent"]').click();
      
      // Step 1
      cy.get('[data-testid="agent-name"]').type('Analytics Agent');
      cy.get('[data-testid="agent-type"]').select('analytics');
      cy.get('[data-testid="wizard-next"]').click();
      
      // Step 2
      cy.get('[data-testid="capability-analytics"]').check();
      cy.get('[data-testid="wizard-next"]').click();
      
      // Step 3 - Review
      cy.get('[data-testid="review-agent-name"]').should('contain', 'Analytics Agent');
      cy.get('[data-testid="review-agent-type"]').should('contain', 'analytics');
      cy.get('[data-testid="review-capabilities"]').should('contain', 'analytics');
      
      // Submit
      cy.intercept('POST', '/api/agents/register').as('registerAgent');
      cy.get('[data-testid="wizard-submit"]').click();
      
      cy.wait('@registerAgent');
      cy.get('[data-testid="registration-success"]').should('be.visible');
    });

    it('TC-AN-049: Should handle registration errors', () => {
      cy.intercept('POST', '/api/agents/register', {
        statusCode: 409,
        body: { error: 'Agent name already exists' }
      }).as('registerError');
      
      // Complete registration
      cy.get('[data-testid="register-new-agent"]').click();
      cy.get('[data-testid="agent-name"]').type('Existing Agent');
      cy.get('[data-testid="agent-type"]').select('data-processor');
      cy.get('[data-testid="wizard-next"]').click();
      cy.get('[data-testid="wizard-next"]').click();
      cy.get('[data-testid="wizard-submit"]').click();
      
      cy.wait('@registerError');
      cy.get('[data-testid="registration-error"]').should('contain', 'Agent name already exists');
    });

    it('TC-AN-050: Should validate agent endpoint connectivity', () => {
      cy.get('[data-testid="register-new-agent"]').click();
      
      cy.get('[data-testid="agent-endpoint"]').type('http://localhost:8080');
      cy.get('[data-testid="test-connection"]').click();
      
      // Loading state
      cy.get('[data-testid="connection-testing"]').should('be.visible');
      
      // Success
      cy.get('[data-testid="connection-success"]').should('be.visible');
      cy.get('[data-testid="connection-latency"]').should('contain', 'ms');
    });
  });

  describe('TC-AN-051 to TC-AN-056: Agent Actions', () => {
    it('TC-AN-051: Should start/stop individual agents', () => {
      // Find inactive agent
      cy.get('[data-testid="agent-status-inactive"]').first().parent('tr').within(() => {
        cy.get('[data-testid="action-start"]').click();
      });
      
      // Confirm action
      cy.get('[data-testid="confirm-start"]').click();
      
      // Verify status change
      cy.get('[data-testid="agent-status-active"]').should('exist');
      
      // Stop agent
      cy.get('[data-testid="agent-status-active"]').first().parent('tr').within(() => {
        cy.get('[data-testid="action-stop"]').click();
      });
      
      cy.get('[data-testid="confirm-stop"]').click();
    });

    it('TC-AN-052: Should edit agent configuration', () => {
      cy.get('[data-testid="action-edit"]').first().click();
      
      // Edit modal
      cy.get('[data-testid="edit-agent-modal"]').should('be.visible');
      
      // Modify configuration
      cy.get('[data-testid="edit-agent-name"]').clear().type('Updated Agent Name');
      cy.get('[data-testid="edit-agent-description"]').type('Updated description');
      
      // Save
      cy.get('[data-testid="save-agent-config"]').click();
      
      // Verify update
      cy.get('[data-testid="agent-name-1"]').should('contain', 'Updated Agent Name');
    });

    it('TC-AN-053: Should view agent details', () => {
      cy.get('[data-testid="action-view-details"]').first().click();
      
      // Navigate to detail page
      cy.url().should('match', /\/agents\/[a-f0-9-]+$/);
      cy.get('[data-testid="agent-detail-page"]').should('be.visible');
    });

    it('TC-AN-054: Should delete agent with confirmation', () => {
      cy.get('[data-testid="action-delete"]').first().click();
      
      // Confirmation dialog
      cy.get('[data-testid="delete-confirmation-dialog"]').should('be.visible');
      cy.get('[data-testid="delete-warning"]').should('contain', 'This action cannot be undone');
      
      // Type confirmation
      cy.get('[data-testid="delete-confirmation-input"]').type('DELETE');
      cy.get('[data-testid="confirm-delete"]').click();
      
      // Verify removal
      cy.get('[data-testid="delete-success"]').should('be.visible');
    });

    it('TC-AN-055: Should perform bulk operations', () => {
      // Select multiple agents
      cy.get('[data-testid^="select-agent-"]').slice(0, 3).check();
      
      // Bulk actions menu
      cy.get('[data-testid="bulk-actions"]').should('be.visible');
      cy.get('[data-testid="bulk-start"]').should('exist');
      cy.get('[data-testid="bulk-stop"]').should('exist');
      cy.get('[data-testid="bulk-delete"]').should('exist');
      
      // Bulk start
      cy.get('[data-testid="bulk-start"]').click();
      cy.get('[data-testid="bulk-confirm"]').click();
      
      // Progress indicator
      cy.get('[data-testid="bulk-progress"]').should('be.visible');
      cy.get('[data-testid="bulk-complete"]').should('contain', '3 agents started');
    });

    it('TC-AN-056: Should export selected agents', () => {
      // Select agents
      cy.get('[data-testid^="select-agent-"]').slice(0, 5).check();
      
      // Export action
      cy.get('[data-testid="bulk-export"]').click();
      
      // Export options
      cy.get('[data-testid="export-format-json"]').should('exist');
      cy.get('[data-testid="export-format-csv"]').should('exist');
      cy.get('[data-testid="export-format-yaml"]').should('exist');
      
      // Export as JSON
      cy.get('[data-testid="export-format-json"]').click();
      cy.get('[data-testid="export-selected"]').click();
      
      // Verify download
      cy.readFile('cypress/downloads/agents-export.json').then((content) => {
        expect(content.agents).to.have.length(5);
      });
    });
  });

  describe('TC-AN-057 to TC-AN-062: Search and Filter', () => {
    it('TC-AN-057: Should search agents by name', () => {
      cy.get('[data-testid="search-agents"]').type('data');
      
      // Debounce check
      cy.wait(300);
      
      // Results updated
      cy.get('[data-testid^="agent-name-"]').each(($el) => {
        cy.wrap($el).should('contain.text', 'data', { matchCase: false });
      });
      
      // Result count
      cy.get('[data-testid="search-results-count"]').should('exist');
    });

    it('TC-AN-058: Should search by agent ID', () => {
      // Get an agent ID
      cy.get('[data-testid^="agent-id-"]').first().invoke('text').then((id) => {
        cy.get('[data-testid="search-agents"]').clear().type(id);
        
        // Exact match for ID
        cy.get('[data-testid="agents-table"] tbody tr').should('have.length', 1);
        cy.get('[data-testid^="agent-id-"]').should('contain', id);
      });
    });

    it('TC-AN-059: Should filter by status', () => {
      cy.get('[data-testid="filter-status"]').click();
      cy.get('[data-testid="status-active"]').check();
      cy.get('[data-testid="apply-filters"]').click();
      
      // Only active agents shown
      cy.get('[data-testid^="agent-status-"]').each(($status) => {
        cy.wrap($status).should('have.class', 'status-active');
      });
    });

    it('TC-AN-060: Should filter by type', () => {
      cy.get('[data-testid="filter-type"]').select('data-processor');
      
      // Type column should only show selected type
      cy.get('[data-testid^="agent-type-"]').each(($type) => {
        cy.wrap($type).should('contain', 'data-processor');
      });
    });

    it('TC-AN-061: Should filter by capabilities', () => {
      cy.get('[data-testid="filter-capabilities"]').click();
      
      // Select multiple capabilities
      cy.get('[data-testid="cap-real-time"]').check();
      cy.get('[data-testid="cap-batch-processing"]').check();
      cy.get('[data-testid="apply-capability-filter"]').click();
      
      // Verify filtered results
      cy.get('[data-testid^="agent-capabilities-"]').each(($cap) => {
        const caps = $cap.text();
        expect(caps).to.satisfy((text) => 
          text.includes('real-time') || text.includes('batch-processing')
        );
      });
    });

    it('TC-AN-062: Should combine multiple filters', () => {
      // Search
      cy.get('[data-testid="search-agents"]').type('analytics');
      
      // Status filter
      cy.get('[data-testid="filter-status"]').click();
      cy.get('[data-testid="status-active"]').check();
      cy.get('[data-testid="apply-filters"]').click();
      
      // Type filter
      cy.get('[data-testid="filter-type"]').select('analytics');
      
      // Verify combined results
      cy.get('[data-testid="agents-table"] tbody tr').each(($row) => {
        cy.wrap($row).within(() => {
          cy.get('[data-testid^="agent-name-"]').should('contain.text', 'analytics', { matchCase: false });
          cy.get('[data-testid^="agent-status-"]').should('have.class', 'status-active');
          cy.get('[data-testid^="agent-type-"]').should('contain', 'analytics');
        });
      });
      
      // Clear all filters
      cy.get('[data-testid="clear-all-filters"]').click();
      cy.get('[data-testid="filter-active-count"]').should('contain', '0');
    });
  });
});