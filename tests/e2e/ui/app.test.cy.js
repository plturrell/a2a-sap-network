/**
 * App.view.xml Component Tests
 * Test Cases: TC-AN-001 to TC-AN-017
 * Coverage: Navigation, User Profile, Global Features
 */

describe('App.view.xml - Main Application Shell', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.viewport(1280, 720);
  });

  describe('TC-AN-001 to TC-AN-005: Navigation Menu', () => {
    it('TC-AN-001: Should toggle navigation menu', () => {
      // Check initial state
      cy.get('[data-testid="nav-menu"]').should('be.visible');
      
      // Toggle menu
      cy.get('[data-testid="nav-toggle"]').click();
      cy.get('[data-testid="nav-menu"]').should('have.class', 'collapsed');
      
      // Toggle back
      cy.get('[data-testid="nav-toggle"]').click();
      cy.get('[data-testid="nav-menu"]').should('not.have.class', 'collapsed');
    });

    it('TC-AN-002: Should highlight active navigation item', () => {
      // Check home is active by default
      cy.get('[data-testid="nav-home"]').should('have.class', 'active');
      
      // Navigate to agents
      cy.get('[data-testid="nav-agents"]').click();
      cy.get('[data-testid="nav-agents"]').should('have.class', 'active');
      cy.get('[data-testid="nav-home"]').should('not.have.class', 'active');
    });

    it('TC-AN-003: Should navigate between all main sections', () => {
      const sections = [
        { nav: 'nav-home', url: '/', title: 'Home' },
        { nav: 'nav-agents', url: '/agents', title: 'Agents' },
        { nav: 'nav-operations', url: '/operations', title: 'Operations' },
        { nav: 'nav-analytics', url: '/analytics', title: 'Analytics' },
        { nav: 'nav-blockchain', url: '/blockchain', title: 'Blockchain' },
        { nav: 'nav-workflows', url: '/workflows', title: 'Workflows' }
      ];

      sections.forEach(({ nav, url, title }) => {
        cy.get(`[data-testid="${nav}"]`).click();
        cy.url().should('include', url);
        cy.get('h1').should('contain', title);
      });
    });

    it('TC-AN-004: Should show navigation tooltips on hover', () => {
      // Collapse menu first
      cy.get('[data-testid="nav-toggle"]').click();
      
      // Hover over icons
      cy.get('[data-testid="nav-home"]').trigger('mouseenter');
      cy.get('.tooltip').should('be.visible').and('contain', 'Home');
    });

    it('TC-AN-005: Should handle keyboard navigation', () => {
      // Focus on first nav item
      cy.get('[data-testid="nav-home"]').focus();
      
      // Navigate with arrow keys
      cy.focused().type('{downarrow}');
      cy.focused().should('have.attr', 'data-testid', 'nav-agents');
      
      // Enter to navigate
      cy.focused().type('{enter}');
      cy.url().should('include', '/agents');
    });
  });

  describe('TC-AN-006 to TC-AN-010: User Profile', () => {
    it('TC-AN-006: Should display user profile information', () => {
      cy.get('[data-testid="user-profile"]').should('be.visible');
      cy.get('[data-testid="user-avatar"]').should('exist');
      cy.get('[data-testid="user-name"]').should('not.be.empty');
    });

    it('TC-AN-007: Should open user profile dropdown', () => {
      cy.get('[data-testid="user-profile"]').click();
      cy.get('[data-testid="profile-dropdown"]').should('be.visible');
      
      // Check dropdown items
      cy.get('[data-testid="profile-settings"]').should('exist');
      cy.get('[data-testid="profile-preferences"]').should('exist');
      cy.get('[data-testid="profile-logout"]').should('exist');
    });

    it('TC-AN-008: Should navigate to user settings', () => {
      cy.get('[data-testid="user-profile"]').click();
      cy.get('[data-testid="profile-settings"]').click();
      cy.url().should('include', '/settings/profile');
    });

    it('TC-AN-009: Should handle logout', () => {
      cy.get('[data-testid="user-profile"]').click();
      cy.get('[data-testid="profile-logout"]').click();
      
      // Confirm logout
      cy.get('[data-testid="confirm-logout"]').click();
      cy.url().should('include', '/login');
    });

    it('TC-AN-010: Should close dropdown when clicking outside', () => {
      cy.get('[data-testid="user-profile"]').click();
      cy.get('[data-testid="profile-dropdown"]').should('be.visible');
      
      // Click outside
      cy.get('body').click(0, 0);
      cy.get('[data-testid="profile-dropdown"]').should('not.exist');
    });
  });

  describe('TC-AN-011 to TC-AN-014: Global Features', () => {
    it('TC-AN-011: Should handle global search', () => {
      cy.get('[data-testid="global-search"]').type('agent{enter}');
      cy.url().should('include', '/search?q=agent');
      cy.get('[data-testid="search-results"]').should('exist');
    });

    it('TC-AN-012: Should show notification bell with count', () => {
      cy.get('[data-testid="notification-bell"]').should('be.visible');
      cy.get('[data-testid="notification-count"]').should('exist');
    });

    it('TC-AN-013: Should open notification panel', () => {
      cy.get('[data-testid="notification-bell"]').click();
      cy.get('[data-testid="notification-panel"]').should('be.visible');
      
      // Check notification items
      cy.get('[data-testid="notification-item"]').should('have.length.greaterThan', 0);
    });

    it('TC-AN-014: Should toggle theme', () => {
      // Check initial theme
      cy.get('body').should('have.class', 'light-theme');
      
      // Toggle theme
      cy.get('[data-testid="theme-switcher"]').click();
      cy.get('body').should('have.class', 'dark-theme');
      
      // Toggle back
      cy.get('[data-testid="theme-switcher"]').click();
      cy.get('body').should('have.class', 'light-theme');
    });
  });

  describe('TC-AN-015 to TC-AN-017: Responsive Behavior', () => {
    it('TC-AN-015: Should adapt to mobile viewport', () => {
      cy.viewport('iphone-x');
      
      // Navigation should be hidden
      cy.get('[data-testid="nav-menu"]').should('not.be.visible');
      
      // Hamburger menu should be visible
      cy.get('[data-testid="mobile-menu-toggle"]').should('be.visible');
    });

    it('TC-AN-016: Should handle mobile navigation', () => {
      cy.viewport('iphone-x');
      
      // Open mobile menu
      cy.get('[data-testid="mobile-menu-toggle"]').click();
      cy.get('[data-testid="mobile-nav-overlay"]').should('be.visible');
      
      // Navigate
      cy.get('[data-testid="mobile-nav-agents"]').click();
      cy.url().should('include', '/agents');
      
      // Menu should close after navigation
      cy.get('[data-testid="mobile-nav-overlay"]').should('not.exist');
    });

    it('TC-AN-017: Should persist user preferences across sessions', () => {
      // Set dark theme
      cy.get('[data-testid="theme-switcher"]').click();
      
      // Collapse navigation
      cy.get('[data-testid="nav-toggle"]').click();
      
      // Reload page
      cy.reload();
      
      // Preferences should persist
      cy.get('body').should('have.class', 'dark-theme');
      cy.get('[data-testid="nav-menu"]').should('have.class', 'collapsed');
    });
  });
});