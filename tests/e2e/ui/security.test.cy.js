/**
 * Security.view.xml Component Tests
 * Test Cases: TC-AN-169 to TC-AN-184
 * Coverage: Security Dashboard, Access Control, Audit Logs, Threat Detection, Compliance
 */

describe('Security.view.xml - Security Management', () => {
  beforeEach(() => {
    cy.visit('/security');
    cy.viewport(1280, 720);
    
    // Wait for security management page to load
    cy.get('[data-testid="security-dashboard"]').should('be.visible');
  });

  describe('TC-AN-169 to TC-AN-173: Security Overview Tests', () => {
    it('TC-AN-169: Should verify security dashboard loads correctly', () => {
      // Verify main security dashboard components
      cy.get('[data-testid="security-overview"]').should('be.visible');
      cy.get('[data-testid="threat-level-indicator"]').should('be.visible');
      cy.get('[data-testid="security-metrics"]').should('be.visible');
      cy.get('[data-testid="recent-alerts"]').should('be.visible');
      
      // Check security status indicator
      cy.get('[data-testid="security-status"]').should('be.visible');
      cy.get('[data-testid="status-icon"]').should('exist');
      cy.get('[data-testid="status-text"]').should('not.be.empty');
      
      // Verify threat level display
      cy.get('[data-testid="threat-level-indicator"]').within(() => {
        cy.get('[data-testid="threat-level"]').should('be.visible');
        cy.get('[data-testid="threat-score"]').should('match', /\d+/);
        cy.get('[data-testid="threat-trend"]').should('be.visible');
      });
      
      // Check security metrics cards
      cy.get('[data-testid="security-metrics"]').within(() => {
        cy.get('[data-testid="active-sessions"]').should('be.visible');
        cy.get('[data-testid="failed-logins"]').should('be.visible');
        cy.get('[data-testid="blocked-ips"]').should('be.visible');
        cy.get('[data-testid="security-incidents"]').should('be.visible');
      });
      
      // Verify last updated timestamp
      cy.get('[data-testid="last-security-scan"]').should('be.visible');
      cy.get('[data-testid="scan-timestamp"]').should('match', /\d{4}-\d{2}-\d{2}/);
    });

    it('TC-AN-170: Should test security alerts monitoring', () => {
      // Verify alerts panel
      cy.get('[data-testid="security-alerts"]').should('be.visible');
      cy.get('[data-testid="alerts-header"]').should('contain', 'Security Alerts');
      
      // Check alert severity filtering
      cy.get('[data-testid="severity-filter"]').should('be.visible');
      cy.get('[data-testid="severity-filter"]').select('High');
      
      // Verify high severity alerts displayed
      cy.get('[data-testid="alert-item"]').each(($alert) => {
        cy.wrap($alert).find('[data-testid="alert-severity"]')
          .should('have.class', 'severity-high');
      });
      
      // Test alert details expansion
      cy.get('[data-testid="alert-item"]').first().click();
      cy.get('[data-testid="alert-details"]').should('be.visible');
      
      cy.get('[data-testid="alert-details"]').within(() => {
        cy.get('[data-testid="alert-timestamp"]').should('be.visible');
        cy.get('[data-testid="alert-source"]').should('be.visible');
        cy.get('[data-testid="alert-description"]').should('not.be.empty');
        cy.get('[data-testid="affected-resources"]').should('be.visible');
        cy.get('[data-testid="recommended-actions"]').should('be.visible');
      });
      
      // Test alert actions
      cy.get('[data-testid="alert-actions"]').within(() => {
        cy.get('[data-testid="acknowledge-alert"]').should('be.visible');
        cy.get('[data-testid="resolve-alert"]').should('be.visible');
        cy.get('[data-testid="escalate-alert"]').should('be.visible');
      });
      
      // Test acknowledge alert
      cy.get('[data-testid="acknowledge-alert"]').click();
      cy.get('[data-testid="acknowledgment-note"]').type('Investigating the issue');
      cy.get('[data-testid="confirm-acknowledge"]').click();
      
      cy.get('[data-testid="alert-acknowledged"]').should('be.visible');
    });

    it('TC-AN-171: Should verify threat detection status', () => {
      // Navigate to threat detection tab
      cy.get('[data-testid="tab-threat-detection"]').click();
      
      // Verify threat detection interface
      cy.get('[data-testid="threat-detection-dashboard"]').should('be.visible');
      cy.get('[data-testid="detection-engines"]').should('be.visible');
      
      // Check detection engine status
      cy.get('[data-testid="detection-engines"]').within(() => {
        cy.get('[data-testid="intrusion-detection"]').should('be.visible');
        cy.get('[data-testid="malware-scanner"]').should('be.visible');
        cy.get('[data-testid="anomaly-detector"]').should('be.visible');
        cy.get('[data-testid="behavioral-analysis"]').should('be.visible');
      });
      
      // Verify each engine status
      cy.get('[data-testid="detection-engine"]').each(($engine) => {
        cy.wrap($engine).within(() => {
          cy.get('[data-testid="engine-name"]').should('be.visible');
          cy.get('[data-testid="engine-status"]').should('be.visible');
          cy.get('[data-testid="last-scan"]').should('be.visible');
          cy.get('[data-testid="threats-found"]').should('be.visible');
        });
      });
      
      // Test manual security scan
      cy.get('[data-testid="run-security-scan"]').click();
      cy.get('[data-testid="scan-type-selector"]').select('Full System Scan');
      cy.get('[data-testid="start-scan"]').click();
      
      // Verify scan progress
      cy.get('[data-testid="scan-progress"]').should('be.visible');
      cy.get('[data-testid="scan-status"]').should('contain', 'Running');
      cy.get('[data-testid="progress-bar"]').should('be.visible');
      
      // Test scan results
      cy.get('[data-testid="scan-complete"]', { timeout: 30000 }).should('be.visible');
      cy.get('[data-testid="scan-results"]').should('be.visible');
      cy.get('[data-testid="threats-detected"]').should('be.visible');
    });

    it('TC-AN-172: Should test security policy configuration', () => {
      // Navigate to security policies
      cy.get('[data-testid="tab-policies"]').click();
      
      // Verify policies interface
      cy.get('[data-testid="security-policies"]').should('be.visible');
      cy.get('[data-testid="policy-categories"]').should('be.visible');
      
      // Check policy categories
      const policyCategories = [
        'Password Policy',
        'Access Control',
        'Session Management',
        'Data Protection',
        'Network Security'
      ];
      
      policyCategories.forEach(category => {
        cy.get('[data-testid="policy-categories"]').should('contain', category);
      });
      
      // Configure password policy
      cy.get('[data-testid="password-policy"]').click();
      cy.get('[data-testid="policy-settings"]').should('be.visible');
      
      cy.get('[data-testid="policy-settings"]').within(() => {
        cy.get('[data-testid="min-length"]').clear().type('12');
        cy.get('[data-testid="require-uppercase"]').check();
        cy.get('[data-testid="require-lowercase"]').check();
        cy.get('[data-testid="require-numbers"]').check();
        cy.get('[data-testid="require-symbols"]').check();
        cy.get('[data-testid="password-history"]').clear().type('5');
        cy.get('[data-testid="max-age"]').clear().type('90');
      });
      
      // Save policy changes
      cy.intercept('PUT', '/api/security/policies/password').as('updatePasswordPolicy');
      cy.get('[data-testid="save-policy"]').click();
      
      cy.wait('@updatePasswordPolicy');
      cy.get('[data-testid="policy-saved"]').should('be.visible');
      
      // Test policy validation
      cy.get('[data-testid="validate-policy"]').click();
      cy.get('[data-testid="validation-results"]').should('be.visible');
      cy.get('[data-testid="policy-compliance"]').should('be.visible');
    });

    it('TC-AN-173: Should verify vulnerability assessment', () => {
      // Navigate to vulnerability assessment
      cy.get('[data-testid="tab-vulnerabilities"]').click();
      
      // Verify vulnerability dashboard
      cy.get('[data-testid="vulnerability-dashboard"]').should('be.visible');
      cy.get('[data-testid="vulnerability-summary"]').should('be.visible');
      
      // Check vulnerability metrics
      cy.get('[data-testid="vulnerability-summary"]').within(() => {
        cy.get('[data-testid="critical-vulns"]').should('be.visible');
        cy.get('[data-testid="high-vulns"]').should('be.visible');
        cy.get('[data-testid="medium-vulns"]').should('be.visible');
        cy.get('[data-testid="low-vulns"]').should('be.visible');
      });
      
      // Verify vulnerability list
      cy.get('[data-testid="vulnerability-list"]').should('be.visible');
      cy.get('[data-testid="vuln-item"]').should('have.length.greaterThan', 0);
      
      // Check vulnerability details
      cy.get('[data-testid="vuln-item"]').first().within(() => {
        cy.get('[data-testid="vuln-id"]').should('be.visible');
        cy.get('[data-testid="vuln-severity"]').should('be.visible');
        cy.get('[data-testid="vuln-component"]').should('be.visible');
        cy.get('[data-testid="vuln-status"]').should('be.visible');
        cy.get('[data-testid="discovery-date"]').should('be.visible');
      });
      
      // Test vulnerability details modal
      cy.get('[data-testid="vuln-item"]').first().click();
      cy.get('[data-testid="vulnerability-details"]').should('be.visible');
      
      cy.get('[data-testid="vulnerability-details"]').within(() => {
        cy.get('[data-testid="cve-id"]').should('be.visible');
        cy.get('[data-testid="cvss-score"]').should('be.visible');
        cy.get('[data-testid="description"]').should('not.be.empty');
        cy.get('[data-testid="affected-versions"]').should('be.visible');
        cy.get('[data-testid="remediation"]').should('be.visible');
      });
      
      // Test vulnerability remediation
      cy.get('[data-testid="mark-resolved"]').click();
      cy.get('[data-testid="resolution-notes"]').type('Applied security patch');
      cy.get('[data-testid="confirm-resolution"]').click();
      
      cy.get('[data-testid="vulnerability-resolved"]').should('be.visible');
    });
  });

  describe('TC-AN-174 to TC-AN-178: Access Control Tests', () => {
    it('TC-AN-174: Should test user access management', () => {
      // Navigate to access control
      cy.get('[data-testid="tab-access-control"]').click();
      
      // Verify access control interface
      cy.get('[data-testid="access-control-dashboard"]').should('be.visible');
      cy.get('[data-testid="users-table"]').should('be.visible');
      
      // Check users table headers
      cy.get('[data-testid="users-table"]').within(() => {
        cy.get('[data-testid="header-username"]').should('contain', 'Username');
        cy.get('[data-testid="header-role"]').should('contain', 'Role');
        cy.get('[data-testid="header-status"]').should('contain', 'Status');
        cy.get('[data-testid="header-last-login"]').should('contain', 'Last Login');
        cy.get('[data-testid="header-actions"]').should('contain', 'Actions');
      });
      
      // Verify user entries
      cy.get('[data-testid="user-row"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="user-row"]').first().within(() => {
        cy.get('[data-testid="username"]').should('not.be.empty');
        cy.get('[data-testid="user-role"]').should('be.visible');
        cy.get('[data-testid="user-status"]').should('be.visible');
        cy.get('[data-testid="last-login"]').should('be.visible');
      });
      
      // Test user permissions management
      cy.get('[data-testid="user-row"]').first().find('[data-testid="manage-permissions"]').click();
      
      cy.get('[data-testid="permissions-modal"]').should('be.visible');
      cy.get('[data-testid="permissions-modal"]').within(() => {
        cy.get('[data-testid="current-permissions"]').should('be.visible');
        cy.get('[data-testid="available-permissions"]').should('be.visible');
        cy.get('[data-testid="permission-groups"]').should('be.visible');
      });
      
      // Modify permissions
      cy.get('[data-testid="permission-item"]').first().find('[data-testid="grant-permission"]').click();
      cy.get('[data-testid="permission-granted"]').should('be.visible');
      
      // Save permission changes
      cy.get('[data-testid="save-permissions"]').click();
      cy.get('[data-testid="permissions-updated"]').should('be.visible');
      
      // Test user deactivation
      cy.get('[data-testid="user-row"]').first().find('[data-testid="deactivate-user"]').click();
      cy.get('[data-testid="deactivation-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-deactivation"]').click();
      
      cy.get('[data-testid="user-deactivated"]').should('be.visible');
    });

    it('TC-AN-175: Should verify role-based permissions', () => {
      // Navigate to roles tab
      cy.get('[data-testid="tab-roles"]').click();
      
      // Verify roles interface
      cy.get('[data-testid="roles-management"]').should('be.visible');
      cy.get('[data-testid="roles-list"]').should('be.visible');
      
      // Check predefined roles
      const defaultRoles = ['Administrator', 'Operator', 'Analyst', 'Viewer'];
      defaultRoles.forEach(role => {
        cy.get('[data-testid="roles-list"]').should('contain', role);
      });
      
      // Create new custom role
      cy.get('[data-testid="create-role"]').click();
      
      cy.get('[data-testid="role-creation-modal"]').within(() => {
        cy.get('[data-testid="role-name"]').type('Security Auditor');
        cy.get('[data-testid="role-description"]').type('Read-only access to security data');
        cy.get('[data-testid="role-level"]').select('Custom');
      });
      
      // Configure role permissions
      cy.get('[data-testid="permission-categories"]').should('be.visible');
      
      // Grant specific permissions
      cy.get('[data-testid="category-security"]').within(() => {
        cy.get('[data-testid="perm-view-logs"]').check();
        cy.get('[data-testid="perm-view-alerts"]').check();
        cy.get('[data-testid="perm-generate-reports"]').check();
      });
      
      cy.get('[data-testid="category-users"]').within(() => {
        cy.get('[data-testid="perm-view-users"]').check();
      });
      
      // Save new role
      cy.intercept('POST', '/api/security/roles').as('createRole');
      cy.get('[data-testid="save-role"]').click();
      
      cy.wait('@createRole');
      cy.get('[data-testid="role-created"]').should('be.visible');
      
      // Verify role in list
      cy.get('[data-testid="roles-list"]').should('contain', 'Security Auditor');
      
      // Test role assignment
      cy.get('[data-testid="assign-role"]').click();
      cy.get('[data-testid="user-selector"]').select('testuser@company.com');
      cy.get('[data-testid="role-selector"]').select('Security Auditor');
      cy.get('[data-testid="confirm-assignment"]').click();
      
      cy.get('[data-testid="role-assigned"]').should('be.visible');
    });

    it('TC-AN-176: Should test session management', () => {
      // Navigate to sessions tab
      cy.get('[data-testid="tab-sessions"]').click();
      
      // Verify session management interface
      cy.get('[data-testid="session-management"]').should('be.visible');
      cy.get('[data-testid="active-sessions"]').should('be.visible');
      
      // Check active sessions table
      cy.get('[data-testid="sessions-table"]').within(() => {
        cy.get('[data-testid="header-user"]').should('contain', 'User');
        cy.get('[data-testid="header-ip-address"]').should('contain', 'IP Address');
        cy.get('[data-testid="header-start-time"]').should('contain', 'Started');
        cy.get('[data-testid="header-last-activity"]').should('contain', 'Last Activity');
        cy.get('[data-testid="header-status"]').should('contain', 'Status');
      });
      
      // Verify session entries
      cy.get('[data-testid="session-row"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="session-row"]').first().within(() => {
        cy.get('[data-testid="session-user"]').should('not.be.empty');
        cy.get('[data-testid="ip-address"]').should('match', /\d+\.\d+\.\d+\.\d+/);
        cy.get('[data-testid="start-time"]').should('be.visible');
        cy.get('[data-testid="last-activity"]').should('be.visible');
        cy.get('[data-testid="session-status"]').should('be.visible');
      });
      
      // Test session termination
      cy.get('[data-testid="session-row"]').first().find('[data-testid="terminate-session"]').click();
      cy.get('[data-testid="termination-confirmation"]').should('be.visible');
      cy.get('[data-testid="termination-reason"]').select('Administrative Action');
      cy.get('[data-testid="confirm-termination"]').click();
      
      cy.get('[data-testid="session-terminated"]').should('be.visible');
      
      // Test bulk session management
      cy.get('[data-testid="session-row"]').each(($row, index) => {
        if (index < 2) {
          cy.wrap($row).find('[data-testid="select-session"]').check();
        }
      });
      
      cy.get('[data-testid="bulk-actions"]').select('Terminate Selected');
      cy.get('[data-testid="execute-bulk-action"]').click();
      cy.get('[data-testid="bulk-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-bulk-action"]').click();
      
      // Test session timeout configuration
      cy.get('[data-testid="session-settings"]').click();
      cy.get('[data-testid="session-config"]').within(() => {
        cy.get('[data-testid="idle-timeout"]').clear().type('30');
        cy.get('[data-testid="max-session-duration"]').clear().type('480');
        cy.get('[data-testid="concurrent-sessions"]').clear().type('3');
        cy.get('[data-testid="force-logout-on-ip-change"]').check();
      });
      
      cy.get('[data-testid="save-session-config"]').click();
      cy.get('[data-testid="session-config-saved"]').should('be.visible');
    });

    it('TC-AN-177: Should verify IP address whitelisting', () => {
      // Navigate to IP management
      cy.get('[data-testid="tab-ip-management"]').click();
      
      // Verify IP management interface
      cy.get('[data-testid="ip-management"]').should('be.visible');
      cy.get('[data-testid="whitelist-section"]').should('be.visible');
      cy.get('[data-testid="blacklist-section"]').should('be.visible');
      
      // Check current whitelist
      cy.get('[data-testid="whitelist-entries"]').should('be.visible');
      cy.get('[data-testid="whitelist-item"]').then(($items) => {
        if ($items.length > 0) {
          cy.wrap($items).first().within(() => {
            cy.get('[data-testid="ip-address"]').should('be.visible');
            cy.get('[data-testid="ip-description"]').should('be.visible');
            cy.get('[data-testid="added-date"]').should('be.visible');
          });
        }
      });
      
      // Add new IP to whitelist
      cy.get('[data-testid="add-whitelist-ip"]').click();
      
      cy.get('[data-testid="ip-entry-form"]').within(() => {
        cy.get('[data-testid="ip-address-input"]').type('192.168.1.100');
        cy.get('[data-testid="ip-description-input"]').type('Development Server');
        cy.get('[data-testid="ip-category"]').select('Internal');
        cy.get('[data-testid="expiration-date"]').type('2024-12-31');
      });
      
      cy.intercept('POST', '/api/security/whitelist').as('addWhitelistIP');
      cy.get('[data-testid="save-whitelist-entry"]').click();
      
      cy.wait('@addWhitelistIP');
      cy.get('[data-testid="ip-whitelisted"]').should('be.visible');
      
      // Test IP range whitelisting
      cy.get('[data-testid="add-ip-range"]').click();
      cy.get('[data-testid="ip-range-start"]').type('10.0.0.1');
      cy.get('[data-testid="ip-range-end"]').type('10.0.0.255');
      cy.get('[data-testid="range-description"]').type('Corporate Network');
      cy.get('[data-testid="save-ip-range"]').click();
      
      cy.get('[data-testid="ip-range-added"]').should('be.visible');
      
      // Test blacklist management
      cy.get('[data-testid="blacklist-section"]').click();
      cy.get('[data-testid="add-blacklist-ip"]').click();
      
      cy.get('[data-testid="blacklist-ip-input"]').type('198.51.100.1');
      cy.get('[data-testid="blacklist-reason"]').select('Suspicious Activity');
      cy.get('[data-testid="auto-expire"]').check();
      cy.get('[data-testid="expire-duration"]').select('7 days');
      
      cy.get('[data-testid="save-blacklist-entry"]').click();
      cy.get('[data-testid="ip-blacklisted"]').should('be.visible');
      
      // Test IP verification
      cy.get('[data-testid="verify-ip-access"]').click();
      cy.get('[data-testid="test-ip-input"]').type('192.168.1.100');
      cy.get('[data-testid="check-access"]').click();
      
      cy.get('[data-testid="access-result"]').should('be.visible');
      cy.get('[data-testid="access-status"]').should('contain', 'Allowed');
    });

    it('TC-AN-178: Should test multi-factor authentication settings', () => {
      // Navigate to MFA settings
      cy.get('[data-testid="tab-mfa"]').click();
      
      // Verify MFA interface
      cy.get('[data-testid="mfa-settings"]').should('be.visible');
      cy.get('[data-testid="mfa-status"]').should('be.visible');
      
      // Check MFA methods
      cy.get('[data-testid="available-mfa-methods"]').should('be.visible');
      cy.get('[data-testid="mfa-methods"]').within(() => {
        cy.get('[data-testid="method-totp"]').should('be.visible');
        cy.get('[data-testid="method-sms"]').should('be.visible');
        cy.get('[data-testid="method-email"]').should('be.visible');
        cy.get('[data-testid="method-hardware-key"]').should('be.visible');
      });
      
      // Configure TOTP
      cy.get('[data-testid="configure-totp"]').click();
      
      cy.get('[data-testid="totp-setup"]').should('be.visible');
      cy.get('[data-testid="qr-code"]').should('be.visible');
      cy.get('[data-testid="manual-key"]').should('not.be.empty');
      
      // Verify TOTP code
      cy.get('[data-testid="verification-code"]').type('123456');
      cy.get('[data-testid="verify-totp"]').click();
      
      // Mock successful verification
      cy.intercept('POST', '/api/security/mfa/verify-totp', { statusCode: 200 }).as('verifyTOTP');
      cy.wait('@verifyTOTP');
      
      cy.get('[data-testid="totp-configured"]').should('be.visible');
      
      // Configure MFA policy
      cy.get('[data-testid="mfa-policy-settings"]').click();
      
      cy.get('[data-testid="mfa-policy"]').within(() => {
        cy.get('[data-testid="enforce-mfa"]').check();
        cy.get('[data-testid="mfa-required-roles"]').should('be.visible');
        cy.get('[data-testid="role-admin"]').check();
        cy.get('[data-testid="role-operator"]').check();
        cy.get('[data-testid="grace-period"]').clear().type('3');
        cy.get('[data-testid="backup-codes"]').check();
      });
      
      cy.get('[data-testid="save-mfa-policy"]').click();
      cy.get('[data-testid="mfa-policy-saved"]').should('be.visible');
      
      // Test backup codes generation
      cy.get('[data-testid="generate-backup-codes"]').click();
      cy.get('[data-testid="backup-codes-modal"]').should('be.visible');
      cy.get('[data-testid="backup-code"]').should('have.length', 10);
      
      cy.get('[data-testid="download-codes"]').click();
      cy.get('[data-testid="codes-downloaded"]').should('be.visible');
    });
  });

  describe('TC-AN-179 to TC-AN-184: Compliance and Audit Tests', () => {
    it('TC-AN-179: Should test audit log viewing', () => {
      // Navigate to audit logs
      cy.get('[data-testid="tab-audit-logs"]').click();
      
      // Verify audit logs interface
      cy.get('[data-testid="audit-logs"]').should('be.visible');
      cy.get('[data-testid="logs-table"]').should('be.visible');
      
      // Check audit logs table headers
      cy.get('[data-testid="logs-table"]').within(() => {
        cy.get('[data-testid="header-timestamp"]').should('contain', 'Timestamp');
        cy.get('[data-testid="header-user"]').should('contain', 'User');
        cy.get('[data-testid="header-action"]').should('contain', 'Action');
        cy.get('[data-testid="header-resource"]').should('contain', 'Resource');
        cy.get('[data-testid="header-result"]').should('contain', 'Result');
        cy.get('[data-testid="header-ip-address"]').should('contain', 'IP Address');
      });
      
      // Verify audit log entries
      cy.get('[data-testid="audit-log-row"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="audit-log-row"]').first().within(() => {
        cy.get('[data-testid="log-timestamp"]').should('be.visible');
        cy.get('[data-testid="log-user"]').should('not.be.empty');
        cy.get('[data-testid="log-action"]').should('be.visible');
        cy.get('[data-testid="log-resource"]').should('be.visible');
        cy.get('[data-testid="log-result"]').should('be.visible');
        cy.get('[data-testid="log-ip"]').should('match', /\d+\.\d+\.\d+\.\d+/);
      });
      
      // Test audit log filtering
      cy.get('[data-testid="log-filters"]').should('be.visible');
      
      // Filter by user
      cy.get('[data-testid="user-filter"]').type('admin@company.com');
      cy.get('[data-testid="apply-user-filter"]').click();
      
      cy.get('[data-testid="audit-log-row"]').each(($row) => {
        cy.wrap($row).find('[data-testid="log-user"]').should('contain', 'admin');
      });
      
      // Filter by action type
      cy.get('[data-testid="action-filter"]').select('Authentication');
      cy.get('[data-testid="apply-action-filter"]').click();
      
      cy.get('[data-testid="audit-log-row"]').each(($row) => {
        cy.wrap($row).find('[data-testid="log-action"]').should('contain', 'login');
      });
      
      // Filter by date range
      cy.get('[data-testid="date-range-filter"]').click();
      cy.get('[data-testid="start-date"]').type('2024-01-01');
      cy.get('[data-testid="end-date"]').type('2024-01-31');
      cy.get('[data-testid="apply-date-filter"]').click();
      
      // Test audit log details
      cy.get('[data-testid="audit-log-row"]').first().click();
      cy.get('[data-testid="log-details-modal"]').should('be.visible');
      
      cy.get('[data-testid="log-details-modal"]').within(() => {
        cy.get('[data-testid="full-timestamp"]').should('be.visible');
        cy.get('[data-testid="session-id"]').should('be.visible');
        cy.get('[data-testid="user-agent"]').should('be.visible');
        cy.get('[data-testid="request-details"]').should('be.visible');
        cy.get('[data-testid="response-details"]').should('be.visible');
      });
    });

    it('TC-AN-180: Should verify compliance reporting', () => {
      // Navigate to compliance tab
      cy.get('[data-testid="tab-compliance"]').click();
      
      // Verify compliance dashboard
      cy.get('[data-testid="compliance-dashboard"]').should('be.visible');
      cy.get('[data-testid="compliance-frameworks"]').should('be.visible');
      
      // Check supported compliance frameworks
      const frameworks = ['SOC 2', 'ISO 27001', 'GDPR', 'HIPAA', 'PCI DSS'];
      frameworks.forEach(framework => {
        cy.get('[data-testid="compliance-frameworks"]').should('contain', framework);
      });
      
      // Check compliance status
      cy.get('[data-testid="framework-status"]').each(($status) => {
        cy.wrap($status).within(() => {
          cy.get('[data-testid="framework-name"]').should('be.visible');
          cy.get('[data-testid="compliance-score"]').should('match', /\d+%/);
          cy.get('[data-testid="last-assessment"]').should('be.visible');
          cy.get('[data-testid="next-review"]').should('be.visible');
        });
      });
      
      // Generate compliance report
      cy.get('[data-testid="generate-report"]').click();
      
      cy.get('[data-testid="report-configuration"]').within(() => {
        cy.get('[data-testid="framework-selector"]').select('SOC 2');
        cy.get('[data-testid="report-period"]').select('Annual');
        cy.get('[data-testid="include-evidence"]').check();
        cy.get('[data-testid="include-recommendations"]').check();
        cy.get('[data-testid="report-format"]').select('PDF');
      });
      
      cy.intercept('POST', '/api/security/compliance/generate-report').as('generateComplianceReport');
      cy.get('[data-testid="generate-compliance-report"]').click();
      
      cy.wait('@generateComplianceReport');
      
      // Verify report generation
      cy.get('[data-testid="report-generation-progress"]').should('be.visible');
      cy.get('[data-testid="report-generated"]').should('be.visible');
      cy.get('[data-testid="download-report"]').should('be.visible');
      
      // Test compliance gap analysis
      cy.get('[data-testid="gap-analysis"]').click();
      cy.get('[data-testid="gap-analysis-results"]').should('be.visible');
      
      cy.get('[data-testid="compliance-gaps"]').should('be.visible');
      cy.get('[data-testid="gap-item"]').first().within(() => {
        cy.get('[data-testid="control-id"]').should('be.visible');
        cy.get('[data-testid="gap-severity"]').should('be.visible');
        cy.get('[data-testid="remediation-plan"]').should('be.visible');
        cy.get('[data-testid="target-date"]').should('be.visible');
      });
    });

    it('TC-AN-181: Should test data privacy controls', () => {
      // Navigate to data privacy
      cy.get('[data-testid="tab-data-privacy"]').click();
      
      // Verify data privacy interface
      cy.get('[data-testid="data-privacy-dashboard"]').should('be.visible');
      cy.get('[data-testid="privacy-controls"]').should('be.visible');
      
      // Check data classification
      cy.get('[data-testid="data-classification"]').should('be.visible');
      cy.get('[data-testid="classification-levels"]').within(() => {
        cy.get('[data-testid="public-data"]').should('be.visible');
        cy.get('[data-testid="internal-data"]').should('be.visible');
        cy.get('[data-testid="confidential-data"]').should('be.visible');
        cy.get('[data-testid="restricted-data"]').should('be.visible');
      });
      
      // Test data retention policies
      cy.get('[data-testid="retention-policies"]').click();
      
      cy.get('[data-testid="retention-rules"]').should('be.visible');
      cy.get('[data-testid="add-retention-rule"]').click();
      
      cy.get('[data-testid="retention-rule-form"]').within(() => {
        cy.get('[data-testid="rule-name"]').type('User Activity Logs');
        cy.get('[data-testid="data-type"]').select('Log Data');
        cy.get('[data-testid="retention-period"]').clear().type('365');
        cy.get('[data-testid="retention-unit"]').select('Days');
        cy.get('[data-testid="auto-delete"]').check();
      });
      
      cy.get('[data-testid="save-retention-rule"]').click();
      cy.get('[data-testid="retention-rule-saved"]').should('be.visible');
      
      // Test data subject requests
      cy.get('[data-testid="data-subject-requests"]').click();
      
      cy.get('[data-testid="dsr-dashboard"]').should('be.visible');
      cy.get('[data-testid="pending-requests"]').should('be.visible');
      
      // Process data access request
      cy.get('[data-testid="new-dsr"]').click();
      
      cy.get('[data-testid="dsr-form"]').within(() => {
        cy.get('[data-testid="request-type"]').select('Data Access');
        cy.get('[data-testid="subject-email"]').type('user@example.com');
        cy.get('[data-testid="request-scope"]').select('All Personal Data');
        cy.get('[data-testid="verification-method"]').select('Email Verification');
      });
      
      cy.get('[data-testid="submit-dsr"]').click();
      cy.get('[data-testid="dsr-submitted"]').should('be.visible');
      
      // Test data anonymization
      cy.get('[data-testid="data-anonymization"]').click();
      
      cy.get('[data-testid="anonymization-tools"]').should('be.visible');
      cy.get('[data-testid="select-dataset"]').select('User Analytics Data');
      cy.get('[data-testid="anonymization-method"]').select('K-Anonymity');
      cy.get('[data-testid="k-value"]').clear().type('5');
      
      cy.get('[data-testid="preview-anonymization"]').click();
      cy.get('[data-testid="anonymization-preview"]').should('be.visible');
      
      cy.get('[data-testid="apply-anonymization"]').click();
      cy.get('[data-testid="anonymization-complete"]').should('be.visible');
    });

    it('TC-AN-182: Should verify security incident management', () => {
      // Navigate to incidents tab
      cy.get('[data-testid="tab-incidents"]').click();
      
      // Verify incident management interface
      cy.get('[data-testid="incident-management"]').should('be.visible');
      cy.get('[data-testid="incidents-dashboard"]').should('be.visible');
      
      // Check incident metrics
      cy.get('[data-testid="incident-metrics"]').within(() => {
        cy.get('[data-testid="open-incidents"]').should('be.visible');
        cy.get('[data-testid="critical-incidents"]').should('be.visible');
        cy.get('[data-testid="avg-response-time"]').should('be.visible');
        cy.get('[data-testid="resolved-incidents"]').should('be.visible');
      });
      
      // Verify incidents list
      cy.get('[data-testid="incidents-list"]').should('be.visible');
      cy.get('[data-testid="incident-item"]').then(($incidents) => {
        if ($incidents.length > 0) {
          cy.wrap($incidents).first().within(() => {
            cy.get('[data-testid="incident-id"]').should('be.visible');
            cy.get('[data-testid="incident-severity"]').should('be.visible');
            cy.get('[data-testid="incident-status"]').should('be.visible');
            cy.get('[data-testid="incident-title"]').should('be.visible');
            cy.get('[data-testid="assigned-to"]').should('be.visible');
          });
        }
      });
      
      // Create new security incident
      cy.get('[data-testid="create-incident"]').click();
      
      cy.get('[data-testid="incident-form"]').within(() => {
        cy.get('[data-testid="incident-title"]').type('Suspicious Login Activity');
        cy.get('[data-testid="incident-type"]').select('Authentication');
        cy.get('[data-testid="incident-severity"]').select('High');
        cy.get('[data-testid="incident-description"]').type('Multiple failed login attempts from unusual IP addresses');
        cy.get('[data-testid="affected-systems"]').type('Authentication Service, User Database');
        cy.get('[data-testid="assign-to"]').select('security-team@company.com');
      });
      
      cy.intercept('POST', '/api/security/incidents').as('createIncident');
      cy.get('[data-testid="create-security-incident"]').click();
      
      cy.wait('@createIncident');
      cy.get('[data-testid="incident-created"]').should('be.visible');
      
      // Test incident workflow
      cy.get('[data-testid="incident-item"]').first().click();
      cy.get('[data-testid="incident-details"]').should('be.visible');
      
      // Add investigation notes
      cy.get('[data-testid="add-note"]').click();
      cy.get('[data-testid="investigation-note"]').type('Analyzing login patterns and IP geolocation');
      cy.get('[data-testid="note-type"]').select('Investigation');
      cy.get('[data-testid="save-note"]').click();
      
      cy.get('[data-testid="note-added"]').should('be.visible');
      
      // Update incident status
      cy.get('[data-testid="update-status"]').click();
      cy.get('[data-testid="new-status"]').select('In Progress');
      cy.get('[data-testid="status-comment"]').type('Investigation started');
      cy.get('[data-testid="confirm-status-update"]').click();
      
      cy.get('[data-testid="status-updated"]').should('be.visible');
      
      // Test incident escalation
      cy.get('[data-testid="escalate-incident"]').click();
      cy.get('[data-testid="escalation-level"]').select('Management');
      cy.get('[data-testid="escalation-reason"]').type('Potential data breach risk');
      cy.get('[data-testid="confirm-escalation"]').click();
      
      cy.get('[data-testid="incident-escalated"]').should('be.visible');
    });

    it('TC-AN-183: Should test security metrics dashboard', () => {
      // Navigate to security metrics
      cy.get('[data-testid="tab-metrics"]').click();
      
      // Verify security metrics dashboard
      cy.get('[data-testid="security-metrics-dashboard"]').should('be.visible');
      cy.get('[data-testid="metrics-overview"]').should('be.visible');
      
      // Check key security metrics
      cy.get('[data-testid="security-kpis"]').within(() => {
        cy.get('[data-testid="security-score"]').should('be.visible');
        cy.get('[data-testid="threat-detection-rate"]').should('be.visible');
        cy.get('[data-testid="incident-response-time"]').should('be.visible');
        cy.get('[data-testid="vulnerability-remediation"]').should('be.visible');
      });
      
      // Verify security trend charts
      cy.get('[data-testid="security-trends"]').should('be.visible');
      cy.get('[data-testid="threats-over-time"]').should('be.visible');
      cy.get('[data-testid="incidents-trend"]').should('be.visible');
      cy.get('[data-testid="compliance-score-trend"]').should('be.visible');
      
      // Test chart interactions
      cy.get('[data-testid="threats-over-time"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
      
      // Test metrics time range selection
      cy.get('[data-testid="metrics-timerange"]').select('Last 90 days');
      cy.get('[data-testid="metrics-loading"]').should('be.visible');
      cy.get('[data-testid="metrics-loading"]').should('not.exist');
      
      // Check risk assessment metrics
      cy.get('[data-testid="risk-assessment"]').should('be.visible');
      cy.get('[data-testid="risk-level-distribution"]').should('be.visible');
      cy.get('[data-testid="top-risks"]').should('be.visible');
      
      // Test metrics export
      cy.get('[data-testid="export-metrics"]').click();
      
      cy.get('[data-testid="export-options"]').within(() => {
        cy.get('[data-testid="export-format-pdf"]').click();
        cy.get('[data-testid="include-charts"]').check();
        cy.get('[data-testid="include-trends"]').check();
        cy.get('[data-testid="include-recommendations"]').check();
      });
      
      cy.get('[data-testid="generate-metrics-report"]').click();
      cy.get('[data-testid="metrics-report-generated"]').should('be.visible');
      
      // Test automated reporting
      cy.get('[data-testid="automated-reporting"]').click();
      
      cy.get('[data-testid="report-schedule"]').within(() => {
        cy.get('[data-testid="report-frequency"]').select('Weekly');
        cy.get('[data-testid="report-day"]').select('Monday');
        cy.get('[data-testid="recipients"]').type('security-team@company.com');
        cy.get('[data-testid="report-format"]').select('PDF + Excel');
      });
      
      cy.get('[data-testid="save-schedule"]').click();
      cy.get('[data-testid="schedule-saved"]').should('be.visible');
    });

    it('TC-AN-184: Should verify security configuration backup', () => {
      // Navigate to backup settings
      cy.get('[data-testid="tab-backup"]').click();
      
      // Verify backup interface
      cy.get('[data-testid="security-backup"]').should('be.visible');
      cy.get('[data-testid="backup-status"]').should('be.visible');
      
      // Check backup configuration
      cy.get('[data-testid="backup-config"]').within(() => {
        cy.get('[data-testid="backup-frequency"]').should('be.visible');
        cy.get('[data-testid="backup-location"]').should('be.visible');
        cy.get('[data-testid="retention-period"]').should('be.visible');
        cy.get('[data-testid="encryption-status"]').should('be.visible');
      });
      
      // View backup history
      cy.get('[data-testid="backup-history"]').should('be.visible');
      cy.get('[data-testid="backup-item"]').should('have.length.greaterThan', 0);
      
      cy.get('[data-testid="backup-item"]').first().within(() => {
        cy.get('[data-testid="backup-date"]').should('be.visible');
        cy.get('[data-testid="backup-size"]').should('be.visible');
        cy.get('[data-testid="backup-status"]').should('contain', 'Success');
        cy.get('[data-testid="backup-type"]').should('be.visible');
      });
      
      // Create manual backup
      cy.get('[data-testid="create-manual-backup"]').click();
      
      cy.get('[data-testid="backup-options"]').within(() => {
        cy.get('[data-testid="include-policies"]').check();
        cy.get('[data-testid="include-roles"]').check();
        cy.get('[data-testid="include-users"]').check();
        cy.get('[data-testid="include-audit-config"]').check();
        cy.get('[data-testid="backup-encryption"]').check();
        cy.get('[data-testid="backup-description"]').type('Pre-upgrade security backup');
      });
      
      cy.intercept('POST', '/api/security/backup').as('createBackup');
      cy.get('[data-testid="start-backup"]').click();
      
      cy.wait('@createBackup');
      
      // Verify backup progress
      cy.get('[data-testid="backup-in-progress"]').should('be.visible');
      cy.get('[data-testid="backup-progress-bar"]').should('be.visible');
      
      // Test backup completion
      cy.get('[data-testid="backup-complete"]', { timeout: 30000 }).should('be.visible');
      cy.get('[data-testid="backup-success-message"]').should('be.visible');
      
      // Test backup restore functionality
      cy.get('[data-testid="backup-item"]').first().find('[data-testid="restore-backup"]').click();
      
      cy.get('[data-testid="restore-confirmation"]').should('be.visible');
      cy.get('[data-testid="restore-warning"]').should('contain', 'This will overwrite current settings');
      
      cy.get('[data-testid="restore-options"]').within(() => {
        cy.get('[data-testid="restore-policies"]').check();
        cy.get('[data-testid="restore-roles"]').check();
        cy.get('[data-testid="backup-current-config"]').check();
      });
      
      cy.get('[data-testid="confirm-restore"]').click();
      cy.get('[data-testid="restore-initiated"]').should('be.visible');
      
      // Test automated backup schedule
      cy.get('[data-testid="backup-schedule"]').click();
      
      cy.get('[data-testid="schedule-settings"]').within(() => {
        cy.get('[data-testid="enable-automated"]').check();
        cy.get('[data-testid="backup-frequency"]').select('Daily');
        cy.get('[data-testid="backup-time"]').type('02:00');
        cy.get('[data-testid="retention-days"]').clear().type('30');
      });
      
      cy.get('[data-testid="save-backup-schedule"]').click();
      cy.get('[data-testid="schedule-updated"]').should('be.visible');
    });
  });
});