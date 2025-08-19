/**
 * Screen Reader Support Accessibility Tests
 * Based on testCases1Common.md - Screen Reader Support section
 */

describe('Screen Reader Support Tests', () => {
  beforeEach(() => {
    cy.visit('/');
    // Install axe-core for accessibility testing
    cy.injectAxe();
  });

  describe('Page Structure and Announcements', () => {
    it('should announce page title within 2 seconds of load', () => {
      // Verify page has a title
      cy.title().should('not.be.empty');
      
      // Check document title updates
      cy.document().its('title').should('exist');
      
      // Verify title is announced (check for live region)
      cy.get('[role="status"], [aria-live]').should('exist');
      
      // Page should have proper heading hierarchy
      cy.get('h1').should('have.length', 1);
    });

    it('should have proper landmark regions', () => {
      // Check all required landmarks exist
      const requiredLandmarks = ['banner', 'navigation', 'main', 'contentinfo'];
      
      requiredLandmarks.forEach(landmark => {
        cy.get(`[role="${landmark}"]`).should('exist').and('be.visible');
      });
      
      // Alternative HTML5 elements
      cy.get('header').should('exist');
      cy.get('nav').should('exist');
      cy.get('main').should('exist');
      cy.get('footer').should('exist');
    });

    it('should have skip navigation links', () => {
      // Check for skip links
      cy.get('a[href="#main"], a[href="#main-content"]').should('exist');
      
      // Verify skip link is first focusable element
      cy.get('body').tab();
      cy.focused().should('contain.text', 'Skip');
    });
  });

  describe('ARIA Labels and Descriptions', () => {
    it('should have descriptive ARIA labels for actions', () => {
      // Check buttons have proper labels
      cy.get('button').each(($button) => {
        if (!$button.text().trim()) {
          // Icon-only buttons must have aria-label
          cy.wrap($button).should('have.attr', 'aria-label');
          
          // Label should describe action, not appearance
          cy.wrap($button).then(($btn) => {
            const label = $btn.attr('aria-label');
            expect(label).to.not.match(/green|blue|red|button/i);
            expect(label).to.match(/submit|save|delete|cancel|open|close/i);
          });
        }
      });
    });

    it('should have proper labels for form inputs', () => {
      cy.visit('/forms/sample');
      
      cy.get('input, select, textarea').each(($input) => {
        const id = $input.attr('id');
        
        // Must have associated label
        if (id) {
          cy.get(`label[for="${id}"]`).should('exist');
        } else {
          // Or aria-label/aria-labelledby
          cy.wrap($input).should('satisfy', ($el) => {
            return $el.attr('aria-label') || $el.attr('aria-labelledby');
          });
        }
      });
    });

    it('should describe required fields', () => {
      cy.visit('/forms/sample');
      
      cy.get('input[required], select[required], textarea[required]').each(($field) => {
        // Should have aria-required
        cy.wrap($field).should('have.attr', 'aria-required', 'true');
        
        // Label should indicate required
        const id = $field.attr('id');
        if (id) {
          cy.get(`label[for="${id}"]`).should('contain', '*');
        }
      });
    });
  });

  describe('Live Regions and Dynamic Updates', () => {
    it('should announce updates within 500ms using live regions', () => {
      // Check for live regions
      cy.get('[aria-live="polite"], [aria-live="assertive"]').as('liveRegions');
      
      // Trigger an update
      cy.get('[data-testid="refresh-data"]').click();
      
      // Verify live region updated
      cy.get('@liveRegions').should('not.be.empty');
      
      // Check timing
      cy.get('@liveRegions', { timeout: 500 }).should('contain', 'Updated');
    });

    it('should announce form errors immediately', () => {
      cy.visit('/forms/sample');
      
      // Submit invalid form
      cy.get('form').submit();
      
      // Error should be in aria-live region
      cy.get('[aria-live="assertive"]').should('contain', 'error');
      
      // Individual field errors should be announced
      cy.get('input:invalid').each(($input) => {
        const id = $input.attr('id');
        cy.get(`[id="${id}-error"]`).should('exist');
        cy.wrap($input).should('have.attr', 'aria-describedby', `${id}-error`);
      });
    });

    it('should announce loading states', () => {
      cy.get('[data-testid="load-more"]').click();
      
      // Loading announcement
      cy.get('[aria-live]').should('contain', 'Loading');
      
      // Completion announcement
      cy.get('[aria-live]').should('contain', 'Loaded');
    });
  });

  describe('Images and Media', () => {
    it('should have alt text for informative images', () => {
      cy.get('img').each(($img) => {
        // Decorative images should have empty alt
        if ($img.attr('role') === 'presentation') {
          cy.wrap($img).should('have.attr', 'alt', '');
        } else {
          // Informative images need descriptive alt text
          cy.wrap($img).should('have.attr', 'alt').and('not.be.empty');
        }
      });
    });

    it('should have captions for videos', () => {
      cy.get('video').each(($video) => {
        // Check for track element
        cy.wrap($video).find('track[kind="captions"]').should('exist');
      });
    });
  });

  describe('Tables', () => {
    beforeEach(() => {
      cy.visit('/tables/data');
    });

    it('should have proper table headers', () => {
      // Tables should have caption or aria-label
      cy.get('table').should('satisfy', ($table) => {
        return $table.find('caption').length > 0 || $table.attr('aria-label');
      });
      
      // Check for th elements
      cy.get('table thead th').should('exist');
      
      // Data cells should reference headers
      cy.get('table tbody td').each(($td) => {
        cy.wrap($td).should('have.attr', 'headers');
      });
    });

    it('should announce sort state for sortable columns', () => {
      cy.get('th[aria-sort]').each(($th) => {
        // Click to sort
        cy.wrap($th).click();
        
        // Verify sort state updated
        cy.wrap($th).should('have.attr', 'aria-sort').and('be.oneOf', ['ascending', 'descending', 'none']);
      });
    });
  });

  describe('Modals and Dialogs', () => {
    it('should announce modal opening and have proper structure', () => {
      cy.get('[data-testid="open-modal"]').click();
      
      // Modal should have role="dialog"
      cy.get('[role="dialog"]').should('exist').and('be.visible');
      
      // Should have aria-labelledby pointing to title
      cy.get('[role="dialog"]').should('have.attr', 'aria-labelledby');
      
      // Title should exist
      cy.get('[role="dialog"]').then(($dialog) => {
        const labelId = $dialog.attr('aria-labelledby');
        cy.get(`#${labelId}`).should('exist').and('not.be.empty');
      });
      
      // Should have aria-describedby for description
      cy.get('[role="dialog"]').should('have.attr', 'aria-describedby');
    });
  });

  describe('Navigation', () => {
    it('should announce current page in navigation', () => {
      cy.get('nav a').each(($link) => {
        if ($link.attr('href') === window.location.pathname) {
          cy.wrap($link).should('have.attr', 'aria-current', 'page');
        }
      });
    });

    it('should have proper breadcrumb structure', () => {
      cy.visit('/agents/details/123');
      
      cy.get('nav[aria-label="Breadcrumb"]').should('exist');
      cy.get('nav[aria-label="Breadcrumb"] ol').should('exist');
      
      // Last item should have aria-current
      cy.get('nav[aria-label="Breadcrumb"] li:last-child').should('have.attr', 'aria-current', 'page');
    });
  });

  describe('Interactive Elements', () => {
    it('should announce button states', () => {
      // Toggle buttons
      cy.get('[aria-pressed]').each(($button) => {
        const pressed = $button.attr('aria-pressed');
        cy.wrap($button).click();
        cy.wrap($button).should('have.attr', 'aria-pressed', pressed === 'true' ? 'false' : 'true');
      });
      
      // Expanded/collapsed states
      cy.get('[aria-expanded]').each(($button) => {
        const expanded = $button.attr('aria-expanded');
        cy.wrap($button).click();
        cy.wrap($button).should('have.attr', 'aria-expanded', expanded === 'true' ? 'false' : 'true');
      });
    });

    it('should announce disabled states', () => {
      cy.get('button[disabled], input[disabled], select[disabled]').each(($el) => {
        cy.wrap($el).should('have.attr', 'aria-disabled', 'true');
      });
    });
  });

  describe('Error Messages', () => {
    beforeEach(() => {
      cy.visit('/forms/sample');
    });

    it('should associate error messages with form fields', () => {
      // Submit invalid form
      cy.get('form').submit();
      
      // Each invalid field should have associated error
      cy.get('input:invalid').each(($input) => {
        const errorId = $input.attr('aria-describedby');
        cy.get(`#${errorId}`).should('exist').and('be.visible');
        
        // Error should have role="alert"
        cy.get(`#${errorId}`).should('have.attr', 'role', 'alert');
      });
    });
  });

  describe('Progress Indicators', () => {
    it('should announce progress updates', () => {
      cy.visit('/upload');
      
      // Start upload
      cy.get('input[type="file"]').attachFile('test.pdf');
      cy.get('button[type="submit"]').click();
      
      // Progress bar should have proper ARIA
      cy.get('[role="progressbar"]').should('exist');
      cy.get('[role="progressbar"]').should('have.attr', 'aria-valuenow');
      cy.get('[role="progressbar"]').should('have.attr', 'aria-valuemin', '0');
      cy.get('[role="progressbar"]').should('have.attr', 'aria-valuemax', '100');
      
      // Should have label
      cy.get('[role="progressbar"]').should('have.attr', 'aria-label');
    });
  });

  describe('Automated Accessibility Checks', () => {
    it('should pass automated accessibility checks', () => {
      // Run axe-core checks
      cy.checkA11y(null, {
        rules: {
          // Focus on critical WCAG 2.1 Level A & AA rules
          'color-contrast': { enabled: true },
          'label': { enabled: true },
          'aria-required-attr': { enabled: true },
          'aria-valid-attr': { enabled: true },
          'button-name': { enabled: true },
          'duplicate-id': { enabled: true },
          'empty-heading': { enabled: true },
          'heading-order': { enabled: true },
          'html-has-lang': { enabled: true },
          'image-alt': { enabled: true },
          'link-name': { enabled: true },
          'list': { enabled: true },
          'region': { enabled: true }
        }
      });
    });

    it('should have no critical accessibility violations', () => {
      cy.checkA11y(null, {
        includedImpacts: ['critical', 'serious']
      }, (violations) => {
        // Log violations for debugging
        cy.task('table', violations);
      });
    });
  });
});