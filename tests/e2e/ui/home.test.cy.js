/**
 * Home.view.xml Component Tests
 * Test Cases: TC-AN-018 to TC-AN-036
 * Coverage: Dashboard Loading, Widgets, Quick Actions, Data Refresh
 */

describe('Home.view.xml - Landing Page Dashboard', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.viewport(1280, 720);
    
    // Wait for dashboard to load
    cy.get('[data-testid="dashboard-container"]').should('be.visible');
  });

  describe('TC-AN-018 to TC-AN-022: Dashboard Loading', () => {
    it('TC-AN-018: Should load dashboard within performance threshold', () => {
      cy.visit('/', {
        onBeforeLoad: (win) => {
          win.performance.mark('dashboard-start');
        },
        onLoad: (win) => {
          win.performance.mark('dashboard-end');
          win.performance.measure('dashboard-load', 'dashboard-start', 'dashboard-end');
          const measure = win.performance.getEntriesByName('dashboard-load')[0];
          expect(measure.duration).to.be.lessThan(3000); // 3 seconds max
        }
      });
    });

    it('TC-AN-019: Should display loading indicators for widgets', () => {
      cy.intercept('GET', '/api/dashboard/widgets', { delay: 1000 }).as('getWidgets');
      cy.reload();
      
      // Check skeleton loaders
      cy.get('[data-testid="widget-skeleton"]').should('have.length.greaterThan', 0);
      
      // Wait for data
      cy.wait('@getWidgets');
      cy.get('[data-testid="widget-skeleton"]').should('not.exist');
    });

    it('TC-AN-020: Should handle widget loading errors gracefully', () => {
      cy.intercept('GET', '/api/dashboard/widgets/network-status', { 
        statusCode: 500 
      }).as('widgetError');
      
      cy.reload();
      cy.wait('@widgetError');
      
      // Error state should be shown
      cy.get('[data-testid="widget-error-network-status"]').should('be.visible');
      cy.get('[data-testid="widget-retry-network-status"]').should('exist');
    });

    it('TC-AN-021: Should retry failed widget loads', () => {
      cy.intercept('GET', '/api/dashboard/widgets/network-status', { 
        statusCode: 500 
      }).as('widgetError');
      
      cy.reload();
      cy.wait('@widgetError');
      
      // Mock successful response for retry
      cy.intercept('GET', '/api/dashboard/widgets/network-status', {
        statusCode: 200,
        body: { status: 'healthy', agents: 42 }
      }).as('widgetSuccess');
      
      cy.get('[data-testid="widget-retry-network-status"]').click();
      cy.wait('@widgetSuccess');
      
      cy.get('[data-testid="widget-content-network-status"]').should('be.visible');
    });

    it('TC-AN-022: Should maintain widget layout on refresh', () => {
      // Get initial positions
      const positions = {};
      cy.get('[data-testid^="widget-"]').each(($el) => {
        const id = $el.attr('data-testid');
        positions[id] = $el.position();
      });
      
      // Refresh
      cy.reload();
      
      // Verify positions maintained
      cy.get('[data-testid^="widget-"]').each(($el) => {
        const id = $el.attr('data-testid');
        const newPos = $el.position();
        expect(newPos.top).to.be.closeTo(positions[id].top, 10);
        expect(newPos.left).to.be.closeTo(positions[id].left, 10);
      });
    });
  });

  describe('TC-AN-023 to TC-AN-028: Dashboard Widgets', () => {
    it('TC-AN-023: Should display network status widget with real-time updates', () => {
      cy.get('[data-testid="widget-network-status"]').within(() => {
        // Check active agents count
        cy.get('[data-testid="active-agents-count"]').should('exist')
          .invoke('text').should('match', /\d+/);
        
        // Check health indicator
        cy.get('[data-testid="health-indicator"]').should('exist')
          .and('have.class', 'status-green');
        
        // Check uptime
        cy.get('[data-testid="uptime-percentage"]').should('exist')
          .invoke('text').should('match', /\d{2}\.\d{2}%/);
      });
      
      // Simulate real-time update
      cy.window().its('WebSocket').then((ws) => {
        ws.send(JSON.stringify({
          type: 'status-update',
          data: { activeAgents: 45 }
        }));
      });
      
      // Verify update within 5 seconds
      cy.get('[data-testid="active-agents-count"]', { timeout: 5000 })
        .should('contain', '45');
    });

    it('TC-AN-024: Should display transaction statistics', () => {
      cy.get('[data-testid="widget-transaction-stats"]').within(() => {
        // Volume counter
        cy.get('[data-testid="transaction-volume"]').should('exist');
        
        // Response time
        cy.get('[data-testid="avg-response-time"]')
          .invoke('text').should('match', /\d+ms avg/);
        
        // Success rate
        cy.get('[data-testid="success-rate"]')
          .invoke('text').should('match', /\d{2}\.\d%/);
        
        // Error count
        cy.get('[data-testid="error-count-24h"]').should('exist');
      });
    });

    it('TC-AN-025: Should render performance graphs correctly', () => {
      cy.get('[data-testid="widget-performance-graph"]').within(() => {
        // Check canvas exists
        cy.get('canvas').should('exist');
        
        // Verify data points
        cy.get('canvas').then(($canvas) => {
          const ctx = $canvas[0].getContext('2d');
          expect(ctx).to.not.be.null;
        });
        
        // Check legend
        cy.get('[data-testid="graph-legend"]').should('exist');
        cy.get('[data-testid="legend-item"]').should('have.length.greaterThan', 0);
      });
    });

    it('TC-AN-026: Should handle graph interactions', () => {
      cy.get('[data-testid="widget-performance-graph"]').within(() => {
        // Hover for tooltip
        cy.get('canvas').trigger('mousemove', 100, 100);
        cy.get('[data-testid="graph-tooltip"]').should('be.visible')
          .and('contain', 'Time:')
          .and('contain', 'Value:');
        
        // Toggle series
        cy.get('[data-testid="legend-item-cpu"]').click();
        cy.get('[data-testid="legend-item-cpu"]').should('have.class', 'disabled');
      });
    });

    it('TC-AN-027: Should auto-refresh widget data', () => {
      cy.intercept('GET', '/api/dashboard/widgets/network-status').as('statusRefresh');
      
      // Wait for auto-refresh (30 seconds for transaction stats)
      cy.wait(30000);
      cy.wait('@statusRefresh');
      
      // Verify data updated
      cy.get('[data-testid="last-updated"]').should('contain', 'Just now');
    });

    it('TC-AN-028: Should customize dashboard layout', () => {
      // Open customize mode
      cy.get('[data-testid="customize-dashboard"]').click();
      
      // Drag widget
      cy.get('[data-testid="widget-network-status"]')
        .trigger('mousedown', { button: 0 })
        .trigger('mousemove', { clientX: 400, clientY: 300 })
        .trigger('mouseup');
      
      // Save layout
      cy.get('[data-testid="save-layout"]').click();
      
      // Verify persisted
      cy.reload();
      cy.get('[data-testid="widget-network-status"]').then(($el) => {
        const pos = $el.position();
        expect(pos.left).to.be.greaterThan(350);
      });
    });
  });

  describe('TC-AN-029 to TC-AN-032: Quick Actions', () => {
    it('TC-AN-029: Should display quick action cards', () => {
      cy.get('[data-testid="quick-actions"]').within(() => {
        cy.get('[data-testid="action-register-agent"]').should('exist');
        cy.get('[data-testid="action-create-workflow"]').should('exist');
        cy.get('[data-testid="action-view-agents"]').should('exist');
        cy.get('[data-testid="action-system-settings"]').should('exist');
      });
    });

    it('TC-AN-030: Should navigate from quick actions', () => {
      // Register new agent
      cy.get('[data-testid="action-register-agent"]').click();
      cy.get('[data-testid="register-agent-wizard"]').should('be.visible');
      cy.get('[data-testid="wizard-cancel"]').click();
      
      // Create workflow
      cy.get('[data-testid="action-create-workflow"]').click();
      cy.url().should('include', '/workflows/designer');
      cy.go('back');
      
      // View all agents
      cy.get('[data-testid="action-view-agents"]').click();
      cy.url().should('include', '/agents');
      cy.go('back');
    });

    it('TC-AN-031: Should enforce permissions on quick actions', () => {
      // Mock non-admin user
      cy.window().then((win) => {
        win.localStorage.setItem('userRole', 'viewer');
      });
      cy.reload();
      
      // System settings should be disabled
      cy.get('[data-testid="action-system-settings"]')
        .should('have.attr', 'disabled');
      
      // Hover should show permission message
      cy.get('[data-testid="action-system-settings"]').trigger('mouseenter');
      cy.get('.tooltip').should('contain', 'Admin access required');
    });

    it('TC-AN-032: Should track quick action usage', () => {
      cy.intercept('POST', '/api/analytics/track').as('trackAction');
      
      cy.get('[data-testid="action-register-agent"]').click();
      
      cy.wait('@trackAction').then((interception) => {
        expect(interception.request.body).to.deep.include({
          event: 'quick_action_clicked',
          action: 'register_agent'
        });
      });
    });
  });

  describe('TC-AN-033 to TC-AN-036: Data Refresh', () => {
    it('TC-AN-033: Should refresh dashboard manually', () => {
      cy.intercept('GET', '/api/dashboard/refresh').as('manualRefresh');
      
      cy.get('[data-testid="refresh-dashboard"]').click();
      
      // Loading state
      cy.get('[data-testid="refresh-spinner"]').should('be.visible');
      
      cy.wait('@manualRefresh');
      
      // Updated timestamp
      cy.get('[data-testid="last-refresh"]')
        .should('contain', 'Updated just now');
    });

    it('TC-AN-034: Should export dashboard data', () => {
      cy.get('[data-testid="export-dashboard"]').click();
      
      // Export options
      cy.get('[data-testid="export-pdf"]').should('exist');
      cy.get('[data-testid="export-csv"]').should('exist');
      cy.get('[data-testid="export-png"]').should('exist');
      
      // Export as CSV
      cy.get('[data-testid="export-csv"]').click();
      
      // Verify download
      cy.readFile('cypress/downloads/dashboard-export.csv').should('exist');
    });

    it('TC-AN-035: Should handle time range selection', () => {
      cy.get('[data-testid="time-range-selector"]').click();
      
      // Preset options
      cy.get('[data-testid="range-last-hour"]').should('exist');
      cy.get('[data-testid="range-last-24h"]').should('exist');
      cy.get('[data-testid="range-last-7d"]').should('exist');
      cy.get('[data-testid="range-custom"]').should('exist');
      
      // Select last 7 days
      cy.get('[data-testid="range-last-7d"]').click();
      
      // Verify data refreshed
      cy.get('[data-testid="selected-range"]').should('contain', 'Last 7 days');
    });

    it('TC-AN-036: Should show recent activities feed', () => {
      cy.get('[data-testid="recent-activities"]').within(() => {
        // Check feed items
        cy.get('[data-testid="activity-item"]')
          .should('have.length', 10); // Latest 10
        
        // Verify timestamp format
        cy.get('[data-testid="activity-timestamp"]').first()
          .should('match', /\d+ (seconds?|minutes?|hours?) ago/);
        
        // Activity icons
        cy.get('[data-testid="activity-icon"]').should('exist');
        
        // View all link
        cy.get('[data-testid="view-all-activities"]').click();
      });
      
      cy.url().should('include', '/activities');
    });
  });
});