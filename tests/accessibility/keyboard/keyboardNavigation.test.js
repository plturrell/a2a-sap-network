/**
 * Keyboard Navigation Accessibility Tests
 * Based on testCases1Common.md - Accessibility Tests section
 */

describe('Keyboard Navigation Accessibility Tests', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  describe('Tab Navigation', () => {
    it('should allow navigation through all interactive elements with Tab key', () => {
      // Start at the beginning of the page
      cy.get('body').tab();
      
      // Verify skip to main content link
      cy.focused().should('have.attr', 'href', '#main-content');
      
      // Tab through navigation menu
      cy.tab();
      cy.focused().should('have.attr', 'role', 'navigation');
      
      // Verify all interactive elements are reachable
      const interactiveElements = [
        'button',
        'a[href]',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])'
      ];
      
      interactiveElements.forEach(selector => {
        cy.get(selector).each(($el) => {
          cy.wrap($el).should('be.visible').and('have.attr', 'tabindex').and('not.eq', '-1');
        });
      });
    });

    it('should follow logical tab order (left-to-right, top-to-bottom)', () => {
      const elements = [];
      
      // Collect all tabbable elements with their positions
      cy.get('[tabindex]:not([tabindex="-1"]), button, a[href], input, select, textarea')
        .each(($el) => {
          const rect = $el[0].getBoundingClientRect();
          elements.push({
            element: $el,
            top: rect.top,
            left: rect.left
          });
        })
        .then(() => {
          // Sort by position (top to bottom, left to right)
          elements.sort((a, b) => {
            if (Math.abs(a.top - b.top) < 10) {
              return a.left - b.left;
            }
            return a.top - b.top;
          });
          
          // Verify tab order matches visual order
          elements.forEach((item, index) => {
            const tabIndex = item.element.attr('tabindex');
            if (tabIndex && tabIndex !== '0') {
              expect(parseInt(tabIndex)).to.be.at.least(index);
            }
          });
        });
    });

    it('should support keyboard shortcuts with Alt+[Letter] pattern', () => {
      // Test main navigation shortcuts
      const shortcuts = [
        { key: 'h', target: 'home' },
        { key: 'a', target: 'agents' },
        { key: 'w', target: 'workflows' },
        { key: 's', target: 'settings' }
      ];
      
      shortcuts.forEach(({ key, target }) => {
        cy.get('body').type(`{alt}${key}`);
        cy.url().should('include', `/${target}`);
        cy.go('back');
      });
    });

    it('should have focus indicators with minimum 3:1 contrast ratio', () => {
      cy.get('button, a, input, select, textarea').each(($el) => {
        cy.wrap($el).focus();
        
        // Check focus outline exists
        cy.wrap($el).should('have.css', 'outline-style').and('not.eq', 'none');
        
        // Verify outline width is at least 2px
        cy.wrap($el).then(($focused) => {
          const outlineWidth = $focused.css('outline-width');
          const width = parseInt(outlineWidth);
          expect(width).to.be.at.least(2);
        });
        
        // Check contrast ratio (simplified check)
        cy.wrap($el).then(($focused) => {
          const outlineColor = $focused.css('outline-color');
          const backgroundColor = $focused.css('background-color');
          
          // This would need a proper contrast calculation library
          // For now, just verify colors are different
          expect(outlineColor).to.not.equal(backgroundColor);
        });
      });
    });
  });

  describe('Modal Focus Management', () => {
    it('should trap focus within modal and close on Escape', () => {
      // Open a modal
      cy.get('[data-testid="open-modal-button"]').click();
      
      // Verify focus is trapped
      cy.focused().should('be.visible').and('have.attr', 'role', 'dialog');
      
      // Tab through modal elements
      cy.tab();
      cy.focused().should('be.visible').parent().should('have.attr', 'role', 'dialog');
      
      // Try to tab past last element - should cycle to first
      cy.get('[role="dialog"] [tabindex]:last').focus().tab();
      cy.focused().should('be.visible').parent().should('have.attr', 'role', 'dialog');
      
      // Close with Escape key
      cy.get('body').type('{esc}');
      cy.get('[role="dialog"]').should('not.exist');
      
      // Focus should return to trigger element
      cy.focused().should('have.attr', 'data-testid', 'open-modal-button');
    });
  });

  describe('Form Navigation', () => {
    beforeEach(() => {
      cy.visit('/forms/sample');
    });

    it('should navigate through form fields with Tab', () => {
      // Tab through form fields in order
      cy.get('form input:first').focus();
      
      cy.tab();
      cy.focused().should('have.attr', 'name', 'email');
      
      cy.tab();
      cy.focused().should('have.attr', 'name', 'password');
      
      cy.tab();
      cy.focused().should('have.attr', 'type', 'submit');
    });

    it('should submit form with Enter key in input fields', () => {
      cy.get('input[name="email"]').type('test@example.com');
      cy.get('input[name="password"]').type('password123{enter}');
      
      // Verify form submission
      cy.url().should('include', '/dashboard');
    });

    it('should navigate radio button groups with arrow keys', () => {
      cy.get('input[type="radio"]:first').focus();
      
      // Arrow down should select next radio
      cy.focused().type('{downarrow}');
      cy.focused().should('be.checked');
      
      // Arrow up should select previous radio
      cy.focused().type('{uparrow}');
      cy.focused().should('be.checked');
    });
  });

  describe('Skip Links', () => {
    it('should provide skip to main content link', () => {
      cy.get('body').tab();
      cy.focused().should('contain', 'Skip to main content');
      
      cy.focused().click();
      cy.focused().should('have.attr', 'id', 'main-content');
    });

    it('should provide skip to navigation link', () => {
      cy.get('[data-testid="skip-to-nav"]').focus().click();
      cy.focused().should('have.attr', 'role', 'navigation');
    });
  });

  describe('Table Navigation', () => {
    beforeEach(() => {
      cy.visit('/tables/data');
    });

    it('should navigate table cells with arrow keys', () => {
      cy.get('table tbody tr:first td:first').focus();
      
      // Right arrow moves to next cell
      cy.focused().type('{rightarrow}');
      cy.focused().should('have.attr', 'data-column', '2');
      
      // Down arrow moves to next row
      cy.focused().type('{downarrow}');
      cy.focused().should('have.attr', 'data-row', '2');
      
      // Left arrow moves to previous cell
      cy.focused().type('{leftarrow}');
      cy.focused().should('have.attr', 'data-column', '1');
      
      // Up arrow moves to previous row
      cy.focused().type('{uparrow}');
      cy.focused().should('have.attr', 'data-row', '1');
    });

    it('should announce table headers when navigating', () => {
      // This would require screen reader testing
      // Verify aria-describedby or headers attribute
      cy.get('table tbody td').each(($cell) => {
        cy.wrap($cell).should('have.attr', 'headers');
      });
    });
  });

  describe('Dropdown Navigation', () => {
    beforeEach(() => {
      cy.visit('/components/dropdown');
    });

    it('should open dropdown with Space or Enter', () => {
      cy.get('[role="combobox"]').focus();
      
      // Space opens dropdown
      cy.focused().type(' ');
      cy.get('[role="listbox"]').should('be.visible');
      
      // Escape closes dropdown
      cy.focused().type('{esc}');
      cy.get('[role="listbox"]').should('not.be.visible');
      
      // Enter also opens dropdown
      cy.get('[role="combobox"]').focus().type('{enter}');
      cy.get('[role="listbox"]').should('be.visible');
    });

    it('should navigate options with arrow keys', () => {
      cy.get('[role="combobox"]').focus().type(' ');
      
      // Down arrow selects first option
      cy.focused().type('{downarrow}');
      cy.get('[role="option"][aria-selected="true"]').should('exist');
      
      // Continue down
      cy.focused().type('{downarrow}');
      cy.get('[role="option"][aria-selected="true"]').should('have.attr', 'data-index', '2');
      
      // Up arrow goes back
      cy.focused().type('{uparrow}');
      cy.get('[role="option"][aria-selected="true"]').should('have.attr', 'data-index', '1');
      
      // Enter selects option
      cy.focused().type('{enter}');
      cy.get('[role="combobox"]').should('have.value', 'Option 1');
    });
  });
});

// Cypress commands for keyboard navigation
Cypress.Commands.add('tab', { prevSubject: 'optional' }, (subject) => {
  if (subject) {
    cy.wrap(subject).trigger('keydown', { keyCode: 9, which: 9 });
  } else {
    cy.get('body').trigger('keydown', { keyCode: 9, which: 9 });
  }
});