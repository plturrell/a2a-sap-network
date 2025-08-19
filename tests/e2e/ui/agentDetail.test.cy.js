/**
 * AgentDetail.view.xml Component Tests
 * Test Cases: TC-AN-063 to TC-AN-082
 * Coverage: Individual Agent Details, Configuration, Performance, Logs
 */

describe('AgentDetail.view.xml - Individual Agent Details', () => {
  const mockAgentId = 'agent-123';
  
  beforeEach(() => {
    cy.visit(`/agents/${mockAgentId}`);
    cy.viewport(1280, 720);
    
    // Wait for agent detail page to load
    cy.get('[data-testid="agent-detail-container"]').should('be.visible');
  });

  describe('TC-AN-063 to TC-AN-067: Agent Overview Tests', () => {
    it('TC-AN-063: Should verify agent details load correctly', () => {
      // Verify main agent info section loads
      cy.get('[data-testid="agent-overview"]').should('be.visible');
      
      // Check for essential elements
      cy.get('[data-testid="agent-name"]').should('not.be.empty');
      cy.get('[data-testid="agent-id"]').should('contain', mockAgentId);
      cy.get('[data-testid="agent-type"]').should('be.visible');
      
      // Verify loading completes within timeout
      cy.get('[data-testid="loading-spinner"]', { timeout: 5000 }).should('not.exist');
    });

    it('TC-AN-064: Should test agent metadata display', () => {
      cy.get('[data-testid="agent-metadata"]').within(() => {
        // Verify metadata fields
        cy.get('[data-testid="created-date"]').should('be.visible');
        cy.get('[data-testid="modified-date"]').should('be.visible');
        cy.get('[data-testid="version"]').should('match', /\d+\.\d+\.\d+/);
        cy.get('[data-testid="description"]').should('be.visible');
        cy.get('[data-testid="tags"]').should('exist');
        
        // Verify capabilities list
        cy.get('[data-testid="capabilities-list"]').should('be.visible');
        cy.get('[data-testid="capability-item"]').should('have.length.greaterThan', 0);
      });
    });

    it('TC-AN-065: Should verify status indicator accuracy', () => {
      // Check status indicator
      cy.get('[data-testid="agent-status-indicator"]').should('be.visible');
      
      // Verify status matches backend data
      cy.request(`/api/agents/${mockAgentId}/status`).then((response) => {
        const expectedStatus = response.body.status;
        
        cy.get('[data-testid="agent-status-indicator"]')
          .should('have.class', `status-${expectedStatus.toLowerCase()}`);
        
        cy.get('[data-testid="status-text"]')
          .should('contain', expectedStatus);
      });
      
      // Verify status colors
      cy.get('[data-testid="agent-status-indicator"]').then(($indicator) => {
        const statusClass = $indicator.attr('class');
        if (statusClass.includes('status-active')) {
          cy.wrap($indicator).should('have.css', 'color', 'rgb(0, 128, 0)');
        } else if (statusClass.includes('status-inactive')) {
          cy.wrap($indicator).should('have.css', 'color', 'rgb(128, 128, 128)');
        } else if (statusClass.includes('status-error')) {
          cy.wrap($indicator).should('have.css', 'color', 'rgb(220, 53, 69)');
        }
      });
    });

    it('TC-AN-066: Should test uptime counter', () => {
      cy.get('[data-testid="uptime-counter"]').should('be.visible');
      
      // Verify uptime format (e.g., "5d 10h 30m")
      cy.get('[data-testid="uptime-value"]')
        .invoke('text')
        .should('match', /\d+[dhms]\s*\d*[hms]?\s*\d*[ms]?/);
      
      // Verify uptime percentage if shown
      cy.get('[data-testid="uptime-percentage"]').then(($el) => {
        if ($el.length > 0) {
          cy.wrap($el).invoke('text').should('match', /\d{2}\.\d{2}%/);
        }
      });
      
      // Test uptime tooltip
      cy.get('[data-testid="uptime-counter"]').trigger('mouseenter');
      cy.get('[data-testid="uptime-tooltip"]').should('be.visible')
        .and('contain', 'Started at');
    });

    it('TC-AN-067: Should verify last heartbeat timestamp', () => {
      cy.get('[data-testid="last-heartbeat"]').should('be.visible');
      
      // Verify timestamp format
      cy.get('[data-testid="heartbeat-timestamp"]')
        .invoke('text')
        .should('match', /\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|\d+ (seconds?|minutes?) ago/);
      
      // Check heartbeat indicator color based on recency
      cy.get('[data-testid="heartbeat-indicator"]').then(($indicator) => {
        const timestamp = $indicator.attr('data-timestamp');
        const now = Date.now();
        const heartbeatTime = new Date(timestamp).getTime();
        const ageInMinutes = (now - heartbeatTime) / (1000 * 60);
        
        if (ageInMinutes < 5) {
          cy.wrap($indicator).should('have.class', 'heartbeat-fresh');
        } else if (ageInMinutes < 30) {
          cy.wrap($indicator).should('have.class', 'heartbeat-stale');
        } else {
          cy.wrap($indicator).should('have.class', 'heartbeat-old');
        }
      });
    });
  });

  describe('TC-AN-068 to TC-AN-072: Configuration Tests', () => {
    beforeEach(() => {
      cy.get('[data-testid="tab-configuration"]').click();
    });

    it('TC-AN-068: Should test configuration editor loads', () => {
      // Verify configuration editor is visible
      cy.get('[data-testid="configuration-editor"]').should('be.visible');
      
      // Check for editor components
      cy.get('[data-testid="config-format-toggle"]').should('be.visible');
      cy.get('[data-testid="config-content"]').should('be.visible');
      
      // Verify syntax highlighting is active
      cy.get('.cm-editor').should('exist'); // CodeMirror editor
      
      // Check for action buttons
      cy.get('[data-testid="save-config"]').should('be.visible');
      cy.get('[data-testid="reset-config"]').should('be.visible');
      cy.get('[data-testid="validate-config"]').should('be.visible');
    });

    it('TC-AN-069: Should verify JSON/YAML syntax validation', () => {
      // Test JSON validation
      cy.get('[data-testid="config-format-toggle"]').select('JSON');
      
      // Insert invalid JSON
      cy.get('[data-testid="config-content"] .cm-editor').type('{selectall}{del}{"invalid": json}');
      cy.get('[data-testid="validate-config"]').click();
      
      // Verify error display
      cy.get('[data-testid="validation-errors"]').should('be.visible');
      cy.get('[data-testid="error-message"]').should('contain', 'Invalid JSON syntax');
      
      // Test valid JSON
      cy.get('[data-testid="config-content"] .cm-editor').type('{selectall}{del}{"valid": "json", "test": true}');
      cy.get('[data-testid="validate-config"]').click();
      
      // Verify success
      cy.get('[data-testid="validation-success"]').should('be.visible');
      
      // Test YAML validation
      cy.get('[data-testid="config-format-toggle"]').select('YAML');
      
      // Insert invalid YAML
      cy.get('[data-testid="config-content"] .cm-editor').type('{selectall}{del}invalid:\n  - yaml: structure\n    missing: quote"');
      cy.get('[data-testid="validate-config"]').click();
      
      cy.get('[data-testid="validation-errors"]').should('be.visible');
    });

    it('TC-AN-070: Should test save configuration functionality', () => {
      // Modify configuration
      cy.get('[data-testid="config-content"] .cm-editor')
        .type('{selectall}{del}{"name": "updated-config", "enabled": true}');
      
      // Save configuration
      cy.intercept('PUT', `/api/agents/${mockAgentId}/config`).as('saveConfig');
      cy.get('[data-testid="save-config"]').click();
      
      // Verify save request
      cy.wait('@saveConfig');
      
      // Check success notification
      cy.get('[data-testid="save-success"]').should('be.visible')
        .and('contain', 'Configuration saved successfully');
      
      // Verify editor reflects saved state
      cy.get('[data-testid="unsaved-changes-indicator"]').should('not.exist');
    });

    it('TC-AN-071: Should verify configuration history', () => {
      cy.get('[data-testid="config-history-button"]').click();
      
      // Verify history panel opens
      cy.get('[data-testid="config-history-panel"]').should('be.visible');
      
      // Check history entries
      cy.get('[data-testid="history-entry"]').should('have.length.greaterThan', 0);
      
      // Verify history entry details
      cy.get('[data-testid="history-entry"]').first().within(() => {
        cy.get('[data-testid="history-timestamp"]').should('be.visible');
        cy.get('[data-testid="history-author"]').should('be.visible');
        cy.get('[data-testid="history-changes"]').should('be.visible');
        cy.get('[data-testid="view-diff"]').should('be.visible');
      });
      
      // Test diff view
      cy.get('[data-testid="history-entry"]').first().find('[data-testid="view-diff"]').click();
      cy.get('[data-testid="diff-viewer"]').should('be.visible');
    });

    it('TC-AN-072: Should test rollback to previous version', () => {
      cy.get('[data-testid="config-history-button"]').click();
      
      // Select previous version
      cy.get('[data-testid="history-entry"]').eq(1).within(() => {
        cy.get('[data-testid="rollback-button"]').click();
      });
      
      // Confirm rollback
      cy.get('[data-testid="confirm-rollback"]').click();
      
      // Verify rollback request
      cy.intercept('POST', `/api/agents/${mockAgentId}/config/rollback`).as('rollbackConfig');
      cy.wait('@rollbackConfig');
      
      // Check success notification
      cy.get('[data-testid="rollback-success"]').should('be.visible');
      
      // Verify configuration updated
      cy.get('[data-testid="config-content"]').should('not.be.empty');
    });
  });

  describe('TC-AN-073 to TC-AN-077: Performance Tests', () => {
    beforeEach(() => {
      cy.get('[data-testid="tab-performance"]').click();
    });

    it('TC-AN-073: Should verify CPU usage graph displays', () => {
      cy.get('[data-testid="cpu-usage-graph"]').should('be.visible');
      
      // Verify chart elements
      cy.get('[data-testid="cpu-usage-graph"] canvas').should('exist');
      cy.get('[data-testid="cpu-current-value"]').should('be.visible');
      cy.get('[data-testid="cpu-average-value"]').should('be.visible');
      
      // Check data points
      cy.get('[data-testid="cpu-usage-graph"]').then(($chart) => {
        // Verify chart has data
        cy.wrap($chart).find('canvas').should('have.attr', 'width').and('not.eq', '0');
      });
      
      // Test hover interactions
      cy.get('[data-testid="cpu-usage-graph"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
    });

    it('TC-AN-074: Should test memory usage chart', () => {
      cy.get('[data-testid="memory-usage-chart"]').should('be.visible');
      
      // Verify memory metrics
      cy.get('[data-testid="memory-current"]').invoke('text').should('match', /\d+(\.\d+)?\s*(MB|GB)/);
      cy.get('[data-testid="memory-max"]').invoke('text').should('match', /\d+(\.\d+)?\s*(MB|GB)/);
      cy.get('[data-testid="memory-percentage"]').invoke('text').should('match', /\d+%/);
      
      // Test memory chart visualization
      cy.get('[data-testid="memory-usage-chart"] .chart-area').should('exist');
    });

    it('TC-AN-075: Should verify request/response metrics', () => {
      cy.get('[data-testid="request-metrics"]').should('be.visible');
      
      // Verify request counters
      cy.get('[data-testid="total-requests"]').should('be.visible');
      cy.get('[data-testid="successful-requests"]').should('be.visible');
      cy.get('[data-testid="failed-requests"]').should('be.visible');
      
      // Verify response time metrics
      cy.get('[data-testid="avg-response-time"]').invoke('text').should('match', /\d+ms/);
      cy.get('[data-testid="min-response-time"]').should('be.visible');
      cy.get('[data-testid="max-response-time"]').should('be.visible');
      
      // Test requests per second chart
      cy.get('[data-testid="rps-chart"]').should('be.visible');
    });

    it('TC-AN-076: Should test time range selector', () => {
      cy.get('[data-testid="time-range-selector"]').should('be.visible');
      
      // Test predefined ranges
      const timeRanges = ['1h', '6h', '24h', '7d', '30d'];
      timeRanges.forEach(range => {
        cy.get(`[data-testid="range-${range}"]`).click();
        
        // Verify charts update
        cy.get('[data-testid="loading-charts"]').should('be.visible');
        cy.get('[data-testid="loading-charts"]').should('not.exist');
        
        // Verify selected state
        cy.get(`[data-testid="range-${range}"]`).should('have.class', 'selected');
      });
      
      // Test custom range
      cy.get('[data-testid="custom-range"]').click();
      cy.get('[data-testid="start-date"]').type('2024-01-01');
      cy.get('[data-testid="end-date"]').type('2024-01-02');
      cy.get('[data-testid="apply-custom-range"]').click();
      
      // Verify custom range applied
      cy.get('[data-testid="selected-range-display"]').should('contain', '2024-01-01 to 2024-01-02');
    });

    it('TC-AN-077: Should verify export metrics functionality', () => {
      cy.get('[data-testid="export-metrics"]').click();
      
      // Verify export options
      cy.get('[data-testid="export-format-csv"]').should('be.visible');
      cy.get('[data-testid="export-format-json"]').should('be.visible');
      cy.get('[data-testid="export-format-pdf"]').should('be.visible');
      
      // Test CSV export
      cy.get('[data-testid="export-format-csv"]').click();
      cy.get('[data-testid="export-download"]').click();
      
      // Verify download
      cy.readFile('cypress/downloads/agent-metrics.csv').should('exist');
    });
  });

  describe('TC-AN-078 to TC-AN-082: Logs Tests', () => {
    beforeEach(() => {
      cy.get('[data-testid="tab-logs"]').click();
    });

    it('TC-AN-078: Should test log viewer loads', () => {
      cy.get('[data-testid="log-viewer"]').should('be.visible');
      
      // Verify log viewer components
      cy.get('[data-testid="log-controls"]').should('be.visible');
      cy.get('[data-testid="log-content"]').should('be.visible');
      cy.get('[data-testid="log-scrollbar"]').should('be.visible');
      
      // Check for log entries
      cy.get('[data-testid="log-entry"]').should('have.length.greaterThan', 0);
      
      // Verify log entry structure
      cy.get('[data-testid="log-entry"]').first().within(() => {
        cy.get('[data-testid="log-timestamp"]').should('be.visible');
        cy.get('[data-testid="log-level"]').should('be.visible');
        cy.get('[data-testid="log-message"]').should('be.visible');
      });
    });

    it('TC-AN-079: Should verify real-time log streaming', () => {
      // Enable auto-refresh
      cy.get('[data-testid="auto-refresh-toggle"]').check();
      
      // Wait for new log entries
      cy.get('[data-testid="log-entry"]').its('length').then((initialCount) => {
        // Wait for streaming to add entries
        cy.wait(5000);
        
        cy.get('[data-testid="log-entry"]').should('have.length.greaterThan', initialCount);
      });
      
      // Verify streaming indicator
      cy.get('[data-testid="streaming-indicator"]').should('be.visible')
        .and('contain', 'Live');
    });

    it('TC-AN-080: Should test log level filtering', () => {
      cy.get('[data-testid="log-level-filter"]').should('be.visible');
      
      // Test different log levels
      const logLevels = ['ERROR', 'WARN', 'INFO', 'DEBUG'];
      
      logLevels.forEach(level => {
        // Select log level
        cy.get('[data-testid="log-level-filter"]').select(level);
        
        // Verify only selected level is shown
        cy.get('[data-testid="log-entry"]').each(($entry) => {
          cy.wrap($entry).find('[data-testid="log-level"]')
            .should('contain', level);
        });
      });
      
      // Test "All Levels" option
      cy.get('[data-testid="log-level-filter"]').select('ALL');
      
      // Verify multiple levels are shown
      const visibleLevels = new Set();
      cy.get('[data-testid="log-entry"] [data-testid="log-level"]').each(($level) => {
        visibleLevels.add($level.text());
      }).then(() => {
        expect(visibleLevels.size).to.be.greaterThan(1);
      });
    });

    it('TC-AN-081: Should verify search within logs', () => {
      cy.get('[data-testid="log-search"]').should('be.visible');
      
      // Perform search
      const searchTerm = 'error';
      cy.get('[data-testid="log-search"]').type(searchTerm);
      cy.get('[data-testid="search-button"]').click();
      
      // Verify search results
      cy.get('[data-testid="log-entry"]').each(($entry) => {
        cy.wrap($entry).find('[data-testid="log-message"]')
          .should('contain.text', searchTerm, { matchCase: false });
      });
      
      // Verify search highlighting
      cy.get('[data-testid="log-entry"] .search-highlight').should('exist');
      
      // Test search navigation
      cy.get('[data-testid="search-next"]').click();
      cy.get('[data-testid="search-previous"]').click();
      
      // Clear search
      cy.get('[data-testid="clear-search"]').click();
      cy.get('[data-testid="log-entry"] .search-highlight').should('not.exist');
    });

    it('TC-AN-082: Should test download logs functionality', () => {
      cy.get('[data-testid="download-logs"]').click();
      
      // Verify download options
      cy.get('[data-testid="download-options"]').should('be.visible');
      cy.get('[data-testid="download-current-view"]').should('be.visible');
      cy.get('[data-testid="download-date-range"]').should('be.visible');
      cy.get('[data-testid="download-all-logs"]').should('be.visible');
      
      // Test current view download
      cy.get('[data-testid="download-current-view"]').click();
      
      // Verify download initiated
      cy.get('[data-testid="download-progress"]').should('be.visible');
      
      // Check file download
      cy.readFile(`cypress/downloads/agent-${mockAgentId}-logs.txt`).should('exist');
      
      // Test date range download
      cy.get('[data-testid="download-logs"]').click();
      cy.get('[data-testid="download-date-range"]').click();
      
      cy.get('[data-testid="download-start-date"]').type('2024-01-01');
      cy.get('[data-testid="download-end-date"]').type('2024-01-02');
      cy.get('[data-testid="confirm-download"]').click();
      
      cy.get('[data-testid="download-success"]').should('be.visible');
    });
  });
});