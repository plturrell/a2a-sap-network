/**
 * Blockchain.view.xml Component Tests
 * Test Cases: TC-AN-153 to TC-AN-168
 * Coverage: Blockchain Integration, Smart Contracts, Transaction Management, Network Monitoring
 */

describe('Blockchain.view.xml - Blockchain Management', () => {
  beforeEach(() => {
    cy.visit('/blockchain');
    cy.viewport(1280, 720);
    
    // Wait for blockchain management page to load
    cy.get('[data-testid="blockchain-management"]').should('be.visible');
  });

  describe('TC-AN-153 to TC-AN-157: Blockchain Network Tests', () => {
    it('TC-AN-153: Should verify blockchain network status display', () => {
      // Verify main network status components
      cy.get('[data-testid="network-status-panel"]').should('be.visible');
      
      // Check network connection status
      cy.get('[data-testid="network-connection"]').should('be.visible');
      cy.get('[data-testid="connection-indicator"]').should('have.class', 'connected');
      cy.get('[data-testid="network-name"]').should('not.be.empty');
      cy.get('[data-testid="chain-id"]').should('match', /\d+/);
      
      // Verify node information
      cy.get('[data-testid="node-info"]').within(() => {
        cy.get('[data-testid="node-version"]').should('be.visible');
        cy.get('[data-testid="peer-count"]').should('match', /\d+/);
        cy.get('[data-testid="sync-status"]').should('be.visible');
        cy.get('[data-testid="block-height"]').should('match', /\d+/);
      });
      
      // Check network statistics
      cy.get('[data-testid="network-stats"]').within(() => {
        cy.get('[data-testid="gas-price"]').should('be.visible');
        cy.get('[data-testid="transaction-count"]').should('be.visible');
        cy.get('[data-testid="block-time"]').should('match', /\d+(\.\d+)?\s*s/);
        cy.get('[data-testid="difficulty"]').should('be.visible');
      });
      
      // Verify last update timestamp
      cy.get('[data-testid="last-updated"]').should('be.visible');
      cy.get('[data-testid="auto-refresh-indicator"]').should('be.visible');
    });

    it('TC-AN-154: Should test blockchain network switching', () => {
      // Open network selector
      cy.get('[data-testid="network-selector"]').click();
      
      // Verify available networks
      cy.get('[data-testid="network-options"]').should('be.visible');
      cy.get('[data-testid="network-option"]').should('have.length.greaterThan', 1);
      
      // Check network options
      const expectedNetworks = ['Mainnet', 'Testnet', 'Local'];
      expectedNetworks.forEach(network => {
        cy.get('[data-testid="network-options"]').should('contain', network);
      });
      
      // Switch to testnet
      cy.get('[data-testid="network-option"]').contains('Testnet').click();
      
      // Verify network switch confirmation
      cy.get('[data-testid="network-switch-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-network-switch"]').click();
      
      // Wait for network connection
      cy.get('[data-testid="network-switching"]').should('be.visible');
      cy.get('[data-testid="network-switching"]').should('not.exist');
      
      // Verify switched to testnet
      cy.get('[data-testid="network-name"]').should('contain', 'Testnet');
      cy.get('[data-testid="connection-indicator"]').should('have.class', 'connected');
      
      // Verify network-specific data updated
      cy.get('[data-testid="chain-id"]').should('not.be.empty');
      cy.get('[data-testid="block-height"]').should('match', /\d+/);
    });

    it('TC-AN-155: Should verify account wallet integration', () => {
      // Check wallet connection status
      cy.get('[data-testid="wallet-connection"]').should('be.visible');
      
      // If wallet not connected, connect it
      cy.get('[data-testid="wallet-status"]').then(($status) => {
        if ($status.text().includes('Not Connected')) {
          cy.get('[data-testid="connect-wallet"]').click();
          
          // Select wallet type
          cy.get('[data-testid="wallet-selection"]').should('be.visible');
          cy.get('[data-testid="metamask-option"]').click();
          
          // Mock wallet connection
          cy.window().then((win) => {
            win.ethereum = {
              request: cy.stub().resolves(['0x1234567890123456789012345678901234567890']),
              on: cy.stub()
            };
          });
          
          cy.get('[data-testid="wallet-connected"]').should('be.visible');
        }
      });
      
      // Verify wallet information
      cy.get('[data-testid="wallet-info"]').within(() => {
        cy.get('[data-testid="wallet-address"]').should('match', /0x[a-fA-F0-9]{40}/);
        cy.get('[data-testid="wallet-balance"]').should('be.visible');
        cy.get('[data-testid="wallet-type"]').should('be.visible');
      });
      
      // Test account switching
      cy.get('[data-testid="account-selector"]').click();
      cy.get('[data-testid="available-accounts"]').should('be.visible');
      
      // Test wallet disconnection
      cy.get('[data-testid="wallet-menu"]').click();
      cy.get('[data-testid="disconnect-wallet"]').click();
      cy.get('[data-testid="wallet-status"]').should('contain', 'Not Connected');
    });

    it('TC-AN-156: Should test gas price monitoring', () => {
      // Verify gas price display
      cy.get('[data-testid="gas-price-monitor"]').should('be.visible');
      
      // Check current gas prices
      cy.get('[data-testid="gas-price-monitor"]').within(() => {
        cy.get('[data-testid="slow-gas-price"]').should('be.visible');
        cy.get('[data-testid="standard-gas-price"]').should('be.visible');
        cy.get('[data-testid="fast-gas-price"]').should('be.visible');
        cy.get('[data-testid="gas-unit"]').should('contain', 'gwei');
      });
      
      // Test gas price history chart
      cy.get('[data-testid="gas-price-chart"]').should('be.visible');
      cy.get('[data-testid="gas-price-chart"] canvas').should('exist');
      
      // Test chart interactions
      cy.get('[data-testid="gas-price-chart"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="gas-tooltip"]').should('be.visible');
      
      // Test time range selection
      cy.get('[data-testid="gas-chart-timerange"]').select('24h');
      cy.get('[data-testid="chart-loading"]').should('be.visible');
      cy.get('[data-testid="chart-loading"]').should('not.exist');
      
      // Test gas price alerts
      cy.get('[data-testid="gas-price-alerts"]').click();
      
      cy.get('[data-testid="alert-configuration"]').within(() => {
        cy.get('[data-testid="high-gas-threshold"]').type('50');
        cy.get('[data-testid="low-gas-threshold"]').type('10');
        cy.get('[data-testid="enable-notifications"]').check();
      });
      
      cy.get('[data-testid="save-gas-alerts"]').click();
      cy.get('[data-testid="alerts-saved"]').should('be.visible');
    });

    it('TC-AN-157: Should verify block explorer integration', () => {
      // Navigate to block explorer tab
      cy.get('[data-testid="tab-explorer"]').click();
      
      // Verify block explorer interface
      cy.get('[data-testid="block-explorer"]').should('be.visible');
      cy.get('[data-testid="explorer-search"]').should('be.visible');
      cy.get('[data-testid="latest-blocks"]').should('be.visible');
      cy.get('[data-testid="latest-transactions"]').should('be.visible');
      
      // Test search functionality
      cy.get('[data-testid="explorer-search"]').type('0x1234567890123456789012345678901234567890');
      cy.get('[data-testid="search-type-address"]').click();
      cy.get('[data-testid="execute-search"]').click();
      
      // Verify search results
      cy.get('[data-testid="search-results"]').should('be.visible');
      cy.get('[data-testid="address-details"]').should('be.visible');
      
      // Test block details
      cy.get('[data-testid="latest-blocks"]').find('[data-testid="block-item"]').first().click();
      cy.get('[data-testid="block-details"]').should('be.visible');
      
      cy.get('[data-testid="block-details"]').within(() => {
        cy.get('[data-testid="block-number"]').should('be.visible');
        cy.get('[data-testid="block-hash"]').should('be.visible');
        cy.get('[data-testid="parent-hash"]').should('be.visible');
        cy.get('[data-testid="timestamp"]').should('be.visible');
        cy.get('[data-testid="gas-used"]').should('be.visible');
        cy.get('[data-testid="transaction-count"]').should('be.visible');
      });
      
      // Test transaction details
      cy.get('[data-testid="latest-transactions"]').find('[data-testid="tx-item"]').first().click();
      cy.get('[data-testid="transaction-details"]').should('be.visible');
      
      cy.get('[data-testid="transaction-details"]').within(() => {
        cy.get('[data-testid="tx-hash"]').should('be.visible');
        cy.get('[data-testid="from-address"]').should('be.visible');
        cy.get('[data-testid="to-address"]').should('be.visible');
        cy.get('[data-testid="tx-value"]').should('be.visible');
        cy.get('[data-testid="gas-price"]').should('be.visible');
        cy.get('[data-testid="tx-status"]').should('be.visible');
      });
    });
  });

  describe('TC-AN-158 to TC-AN-162: Smart Contract Tests', () => {
    it('TC-AN-158: Should test smart contract deployment', () => {
      // Navigate to contracts tab
      cy.get('[data-testid="tab-contracts"]').click();
      
      // Open contract deployment
      cy.get('[data-testid="deploy-contract"]').click();
      
      // Verify deployment interface
      cy.get('[data-testid="contract-deployment"]').should('be.visible');
      cy.get('[data-testid="contract-deployment"]').within(() => {
        cy.get('[data-testid="contract-source"]').should('be.visible');
        cy.get('[data-testid="contract-name"]').should('be.visible');
        cy.get('[data-testid="constructor-params"]').should('be.visible');
        cy.get('[data-testid="gas-estimate"]').should('be.visible');
        cy.get('[data-testid="deployment-options"]').should('be.visible');
      });
      
      // Upload contract source
      cy.get('[data-testid="upload-source"]').selectFile('cypress/fixtures/sample-contract.sol');
      
      // Verify contract compilation
      cy.get('[data-testid="compile-contract"]').click();
      cy.get('[data-testid="compilation-progress"]').should('be.visible');
      cy.get('[data-testid="compilation-success"]').should('be.visible');
      
      // Check compiled contract details
      cy.get('[data-testid="compiled-contracts"]').should('be.visible');
      cy.get('[data-testid="contract-bytecode"]').should('not.be.empty');
      cy.get('[data-testid="contract-abi"]').should('not.be.empty');
      
      // Configure deployment parameters
      cy.get('[data-testid="contract-name"]').select('SampleContract');
      cy.get('[data-testid="constructor-param-0"]').type('Initial Value');
      cy.get('[data-testid="gas-limit"]').clear().type('2000000');
      
      // Deploy contract
      cy.intercept('POST', '/api/blockchain/contracts/deploy').as('deployContract');
      cy.get('[data-testid="deploy-button"]').click();
      
      // Verify deployment confirmation
      cy.get('[data-testid="deployment-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-deployment"]').click();
      
      cy.wait('@deployContract');
      
      // Check deployment success
      cy.get('[data-testid="deployment-success"]').should('be.visible');
      cy.get('[data-testid="contract-address"]').should('match', /0x[a-fA-F0-9]{40}/);
      cy.get('[data-testid="deployment-tx-hash"]').should('be.visible');
    });

    it('TC-AN-159: Should verify contract interaction interface', () => {
      // Select deployed contract
      cy.get('[data-testid="deployed-contracts"]').should('be.visible');
      cy.get('[data-testid="contract-item"]').first().click();
      
      // Verify contract interaction panel
      cy.get('[data-testid="contract-interaction"]').should('be.visible');
      cy.get('[data-testid="contract-functions"]').should('be.visible');
      
      // Check read functions
      cy.get('[data-testid="read-functions"]').within(() => {
        cy.get('[data-testid="function-item"]').should('have.length.greaterThan', 0);
      });
      
      // Test read function call
      cy.get('[data-testid="read-functions"]').find('[data-testid="function-item"]').first().within(() => {
        cy.get('[data-testid="function-name"]').should('be.visible');
        cy.get('[data-testid="call-function"]').click();
      });
      
      // Verify function result
      cy.get('[data-testid="function-result"]').should('be.visible');
      cy.get('[data-testid="result-value"]').should('not.be.empty');
      
      // Check write functions
      cy.get('[data-testid="write-functions"]').within(() => {
        cy.get('[data-testid="function-item"]').should('have.length.greaterThan', 0);
      });
      
      // Test write function call
      cy.get('[data-testid="write-functions"]').find('[data-testid="function-item"]').first().within(() => {
        cy.get('[data-testid="function-params"]').then(($params) => {
          if ($params.length > 0) {
            cy.get('[data-testid="param-input"]').first().type('Test Value');
          }
        });
        cy.get('[data-testid="call-function"]').click();
      });
      
      // Verify transaction confirmation
      cy.get('[data-testid="tx-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-transaction"]').click();
      
      // Check transaction submission
      cy.get('[data-testid="tx-submitted"]').should('be.visible');
      cy.get('[data-testid="tx-hash"]').should('be.visible');
    });

    it('TC-AN-160: Should test contract event monitoring', () => {
      // Navigate to events tab
      cy.get('[data-testid="tab-events"]').click();
      
      // Verify events monitoring interface
      cy.get('[data-testid="events-monitor"]').should('be.visible');
      cy.get('[data-testid="event-filters"]').should('be.visible');
      cy.get('[data-testid="events-list"]').should('be.visible');
      
      // Configure event filters
      cy.get('[data-testid="contract-filter"]').select('All Contracts');
      cy.get('[data-testid="event-type-filter"]').select('All Events');
      cy.get('[data-testid="block-range-filter"]').within(() => {
        cy.get('[data-testid="from-block"]').type('latest-1000');
        cy.get('[data-testid="to-block"]').type('latest');
      });
      
      // Apply filters
      cy.get('[data-testid="apply-filters"]').click();
      cy.get('[data-testid="events-loading"]').should('be.visible');
      cy.get('[data-testid="events-loading"]').should('not.exist');
      
      // Verify event entries
      cy.get('[data-testid="event-item"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="event-item"]').first().within(() => {
        cy.get('[data-testid="event-name"]').should('be.visible');
        cy.get('[data-testid="contract-address"]').should('be.visible');
        cy.get('[data-testid="transaction-hash"]').should('be.visible');
        cy.get('[data-testid="block-number"]').should('be.visible');
        cy.get('[data-testid="event-data"]').should('be.visible');
      });
      
      // Test event details expansion
      cy.get('[data-testid="event-item"]').first().click();
      cy.get('[data-testid="event-details-expanded"]').should('be.visible');
      
      // Test real-time event monitoring
      cy.get('[data-testid="real-time-monitoring"]').check();
      cy.get('[data-testid="live-events-indicator"]').should('be.visible');
      
      // Test event export
      cy.get('[data-testid="export-events"]').click();
      cy.get('[data-testid="export-format-csv"]').click();
      cy.get('[data-testid="start-export"]').click();
      
      cy.get('[data-testid="export-complete"]').should('be.visible');
    });

    it('TC-AN-161: Should verify contract source verification', () => {
      // Select contract for verification
      cy.get('[data-testid="deployed-contracts"]').find('[data-testid="contract-item"]').first().click();
      cy.get('[data-testid="verify-contract"]').click();
      
      // Verify source verification interface
      cy.get('[data-testid="source-verification"]').should('be.visible');
      cy.get('[data-testid="verification-form"]').within(() => {
        cy.get('[data-testid="contract-address"]').should('not.be.empty');
        cy.get('[data-testid="contract-name-input"]').should('be.visible');
        cy.get('[data-testid="compiler-version"]').should('be.visible');
        cy.get('[data-testid="optimization-enabled"]').should('be.visible');
        cy.get('[data-testid="source-code-input"]').should('be.visible');
      });
      
      // Fill verification form
      cy.get('[data-testid="contract-name-input"]').type('SampleContract');
      cy.get('[data-testid="compiler-version"]').select('v0.8.19+commit.7dd6d404');
      cy.get('[data-testid="optimization-enabled"]').check();
      cy.get('[data-testid="optimization-runs"]').clear().type('200');
      
      // Upload source code
      cy.get('[data-testid="upload-source-files"]').selectFile([
        'cypress/fixtures/SampleContract.sol',
        'cypress/fixtures/interfaces/IERC20.sol'
      ]);
      
      // Submit verification
      cy.intercept('POST', '/api/blockchain/contracts/verify').as('verifyContract');
      cy.get('[data-testid="submit-verification"]').click();
      
      cy.wait('@verifyContract');
      
      // Check verification result
      cy.get('[data-testid="verification-result"]').should('be.visible');
      cy.get('[data-testid="verification-status"]').should('contain', 'Verified');
      cy.get('[data-testid="verified-source-code"]').should('be.visible');
      
      // Test verified contract features
      cy.get('[data-testid="verified-badge"]').should('be.visible');
      cy.get('[data-testid="source-code-tab"]').should('be.visible');
    });

    it('TC-AN-162: Should test contract upgrade management', () => {
      // Navigate to upgradeable contracts
      cy.get('[data-testid="upgradeable-contracts"]').click();
      
      // Select proxy contract
      cy.get('[data-testid="proxy-contract-item"]').first().click();
      
      // Verify upgrade interface
      cy.get('[data-testid="contract-upgrade"]').should('be.visible');
      cy.get('[data-testid="current-implementation"]').should('be.visible');
      cy.get('[data-testid="upgrade-options"]').should('be.visible');
      
      // Check current implementation details
      cy.get('[data-testid="current-implementation"]').within(() => {
        cy.get('[data-testid="implementation-address"]').should('be.visible');
        cy.get('[data-testid="implementation-version"]').should('be.visible');
        cy.get('[data-testid="upgrade-history"]').should('be.visible');
      });
      
      // Prepare new implementation
      cy.get('[data-testid="new-implementation"]').click();
      cy.get('[data-testid="upload-new-contract"]').selectFile('cypress/fixtures/ContractV2.sol');
      
      // Compile new version
      cy.get('[data-testid="compile-new-version"]').click();
      cy.get('[data-testid="compilation-success"]').should('be.visible');
      
      // Configure upgrade parameters
      cy.get('[data-testid="upgrade-type"]').select('UUPS');
      cy.get('[data-testid="initialization-data"]').type('0x');
      cy.get('[data-testid="timelock-delay"]').clear().type('86400');
      
      // Propose upgrade
      cy.intercept('POST', '/api/blockchain/contracts/propose-upgrade').as('proposeUpgrade');
      cy.get('[data-testid="propose-upgrade"]').click();
      
      cy.wait('@proposeUpgrade');
      
      // Verify upgrade proposal
      cy.get('[data-testid="upgrade-proposed"]').should('be.visible');
      cy.get('[data-testid="proposal-id"]').should('be.visible');
      cy.get('[data-testid="execution-time"]').should('be.visible');
      
      // Test upgrade execution (after timelock)
      cy.get('[data-testid="execute-upgrade"]').should('be.disabled');
      cy.get('[data-testid="timelock-remaining"]').should('be.visible');
    });
  });

  describe('TC-AN-163 to TC-AN-168: Transaction Management Tests', () => {
    it('TC-AN-163: Should test transaction history display', () => {
      // Navigate to transactions tab
      cy.get('[data-testid="tab-transactions"]').click();
      
      // Verify transaction history interface
      cy.get('[data-testid="transaction-history"]').should('be.visible');
      cy.get('[data-testid="transactions-table"]').should('be.visible');
      
      // Check table headers
      cy.get('[data-testid="transactions-table"]').within(() => {
        cy.get('[data-testid="header-hash"]').should('contain', 'Hash');
        cy.get('[data-testid="header-from"]').should('contain', 'From');
        cy.get('[data-testid="header-to"]').should('contain', 'To');
        cy.get('[data-testid="header-value"]').should('contain', 'Value');
        cy.get('[data-testid="header-status"]').should('contain', 'Status');
        cy.get('[data-testid="header-timestamp"]').should('contain', 'Time');
      });
      
      // Verify transaction entries
      cy.get('[data-testid="transaction-row"]').should('have.length.greaterThan', 0);
      cy.get('[data-testid="transaction-row"]').first().within(() => {
        cy.get('[data-testid="tx-hash"]').should('match', /0x[a-fA-F0-9]{64}/);
        cy.get('[data-testid="from-address"]').should('match', /0x[a-fA-F0-9]{40}/);
        cy.get('[data-testid="to-address"]').should('be.visible');
        cy.get('[data-testid="tx-value"]').should('be.visible');
        cy.get('[data-testid="tx-status"]').should('be.visible');
        cy.get('[data-testid="tx-timestamp"]').should('be.visible');
      });
      
      // Test transaction status filtering
      cy.get('[data-testid="status-filter"]').select('Confirmed');
      cy.get('[data-testid="transaction-row"]').each(($row) => {
        cy.wrap($row).find('[data-testid="tx-status"]').should('contain', 'Confirmed');
      });
      
      // Test transaction details view
      cy.get('[data-testid="transaction-row"]').first().click();
      cy.get('[data-testid="transaction-details"]').should('be.visible');
    });

    it('TC-AN-164: Should verify transaction details modal', () => {
      // Select a transaction
      cy.get('[data-testid="tab-transactions"]').click();
      cy.get('[data-testid="transaction-row"]').first().click();
      
      // Verify transaction details modal
      cy.get('[data-testid="transaction-details-modal"]').should('be.visible');
      
      // Check transaction overview
      cy.get('[data-testid="tx-overview"]').within(() => {
        cy.get('[data-testid="tx-hash-full"]').should('be.visible');
        cy.get('[data-testid="tx-status-badge"]').should('be.visible');
        cy.get('[data-testid="block-number"]').should('be.visible');
        cy.get('[data-testid="confirmations"]').should('be.visible');
        cy.get('[data-testid="tx-fee"]').should('be.visible');
      });
      
      // Check transaction participants
      cy.get('[data-testid="tx-participants"]').within(() => {
        cy.get('[data-testid="from-address-full"]').should('be.visible');
        cy.get('[data-testid="to-address-full"]').should('be.visible');
        cy.get('[data-testid="value-transferred"]').should('be.visible');
      });
      
      // Check gas information
      cy.get('[data-testid="gas-info"]').within(() => {
        cy.get('[data-testid="gas-limit"]').should('be.visible');
        cy.get('[data-testid="gas-used"]').should('be.visible');
        cy.get('[data-testid="gas-price"]').should('be.visible');
        cy.get('[data-testid="gas-fee"]').should('be.visible');
      });
      
      // Test transaction input data
      cy.get('[data-testid="tx-input-data"]').should('be.visible');
      cy.get('[data-testid="input-data-decoded"]').then(($decoded) => {
        if ($decoded.length > 0) {
          cy.get('[data-testid="method-name"]').should('be.visible');
          cy.get('[data-testid="method-params"]').should('be.visible');
        }
      });
      
      // Test external links
      cy.get('[data-testid="external-links"]').within(() => {
        cy.get('[data-testid="etherscan-link"]').should('have.attr', 'href');
        cy.get('[data-testid="copy-tx-hash"]').click();
        cy.get('[data-testid="copy-success"]').should('be.visible');
      });
    });

    it('TC-AN-165: Should test transaction submission interface', () => {
      // Open transaction submission
      cy.get('[data-testid="send-transaction"]').click();
      
      // Verify transaction form
      cy.get('[data-testid="transaction-form"]').should('be.visible');
      cy.get('[data-testid="transaction-form"]').within(() => {
        cy.get('[data-testid="to-address"]').should('be.visible');
        cy.get('[data-testid="tx-value"]').should('be.visible');
        cy.get('[data-testid="gas-settings"]').should('be.visible');
        cy.get('[data-testid="data-input"]').should('be.visible');
      });
      
      // Fill transaction details
      cy.get('[data-testid="to-address"]').type('0x742d35Cc6481C467c3bdc5d7e7c9e74Bbb11E5e7');
      cy.get('[data-testid="tx-value"]').type('0.1');
      cy.get('[data-testid="value-unit"]').select('ETH');
      
      // Configure gas settings
      cy.get('[data-testid="gas-settings"]').within(() => {
        cy.get('[data-testid="gas-price"]').clear().type('20');
        cy.get('[data-testid="gas-limit"]').clear().type('21000');
      });
      
      // Preview transaction
      cy.get('[data-testid="preview-transaction"]').click();
      
      // Verify transaction preview
      cy.get('[data-testid="tx-preview"]').should('be.visible');
      cy.get('[data-testid="tx-preview"]').within(() => {
        cy.get('[data-testid="preview-to"]').should('contain', '0x742d35Cc');
        cy.get('[data-testid="preview-value"]').should('contain', '0.1 ETH');
        cy.get('[data-testid="preview-fee"]').should('be.visible');
        cy.get('[data-testid="preview-total"]').should('be.visible');
      });
      
      // Submit transaction
      cy.intercept('POST', '/api/blockchain/transactions/send').as('sendTransaction');
      cy.get('[data-testid="confirm-send"]').click();
      
      // Verify wallet confirmation
      cy.get('[data-testid="wallet-confirmation"]').should('be.visible');
      cy.get('[data-testid="confirm-in-wallet"]').click();
      
      cy.wait('@sendTransaction');
      
      // Check transaction submitted
      cy.get('[data-testid="tx-submitted"]').should('be.visible');
      cy.get('[data-testid="submitted-tx-hash"]').should('be.visible');
    });

    it('TC-AN-166: Should verify transaction status tracking', () => {
      // Submit a test transaction first (from previous test)
      cy.get('[data-testid="submitted-tx-hash"]').invoke('text').then((txHash) => {
        // Navigate to transaction tracking
        cy.get('[data-testid="track-transaction"]').click();
        cy.get('[data-testid="tx-hash-input"]').type(txHash);
        cy.get('[data-testid="start-tracking"]').click();
        
        // Verify tracking interface
        cy.get('[data-testid="tx-tracking"]').should('be.visible');
        cy.get('[data-testid="tracking-status"]').should('be.visible');
        
        // Check tracking stages
        cy.get('[data-testid="tracking-stages"]').within(() => {
          cy.get('[data-testid="stage-submitted"]').should('have.class', 'completed');
          cy.get('[data-testid="stage-pending"]').should('be.visible');
          cy.get('[data-testid="stage-confirmed"]').should('be.visible');
        });
        
        // Test real-time updates
        cy.get('[data-testid="auto-refresh"]').should('be.checked');
        cy.get('[data-testid="refresh-interval"]').should('contain', '5 seconds');
        
        // Test manual refresh
        cy.get('[data-testid="manual-refresh"]').click();
        cy.get('[data-testid="last-checked"]').should('be.visible');
        
        // Check confirmation progress
        cy.get('[data-testid="confirmations-count"]').should('be.visible');
        cy.get('[data-testid="confirmations-progress"]').should('be.visible');
        
        // Test notification settings
        cy.get('[data-testid="notification-settings"]').click();
        cy.get('[data-testid="notify-on-confirmation"]').check();
        cy.get('[data-testid="notify-on-failure"]').check();
        cy.get('[data-testid="save-notifications"]').click();
      });
    });

    it('TC-AN-167: Should test batch transaction operations', () => {
      // Open batch operations
      cy.get('[data-testid="batch-operations"]').click();
      
      // Verify batch interface
      cy.get('[data-testid="batch-transaction-builder"]').should('be.visible');
      cy.get('[data-testid="transaction-list"]').should('be.visible');
      
      // Add multiple transactions
      cy.get('[data-testid="add-transaction"]').click();
      
      // Configure first transaction
      cy.get('[data-testid="batch-tx-0"]').within(() => {
        cy.get('[data-testid="to-address"]').type('0x742d35Cc6481C467c3bdc5d7e7c9e74Bbb11E5e7');
        cy.get('[data-testid="value"]').type('0.05');
        cy.get('[data-testid="tx-type"]').select('Transfer');
      });
      
      // Add second transaction
      cy.get('[data-testid="add-transaction"]').click();
      cy.get('[data-testid="batch-tx-1"]').within(() => {
        cy.get('[data-testid="to-address"]').type('0x8ba1f109551bD432803012645Hac136c4ce45De');
        cy.get('[data-testid="value"]').type('0.03');
        cy.get('[data-testid="tx-type"]').select('Transfer');
      });
      
      // Configure batch settings
      cy.get('[data-testid="batch-settings"]').within(() => {
        cy.get('[data-testid="execution-order"]').select('Sequential');
        cy.get('[data-testid="failure-handling"]').select('Stop on Failure');
        cy.get('[data-testid="gas-optimization"]').check();
      });
      
      // Preview batch
      cy.get('[data-testid="preview-batch"]').click();
      
      // Verify batch preview
      cy.get('[data-testid="batch-preview"]').should('be.visible');
      cy.get('[data-testid="total-transactions"]').should('contain', '2');
      cy.get('[data-testid="total-value"]').should('contain', '0.08 ETH');
      cy.get('[data-testid="estimated-gas"]').should('be.visible');
      
      // Execute batch
      cy.intercept('POST', '/api/blockchain/transactions/batch').as('executeBatch');
      cy.get('[data-testid="execute-batch"]').click();
      
      cy.wait('@executeBatch');
      
      // Verify batch execution
      cy.get('[data-testid="batch-executing"]').should('be.visible');
      cy.get('[data-testid="batch-progress"]').should('be.visible');
      cy.get('[data-testid="executed-transactions"]').should('be.visible');
    });

    it('TC-AN-168: Should verify transaction analytics dashboard', () => {
      // Navigate to analytics tab
      cy.get('[data-testid="tab-analytics"]').click();
      
      // Verify analytics dashboard
      cy.get('[data-testid="tx-analytics-dashboard"]').should('be.visible');
      
      // Check analytics metrics
      cy.get('[data-testid="analytics-metrics"]').within(() => {
        cy.get('[data-testid="total-transactions"]').should('be.visible');
        cy.get('[data-testid="successful-rate"]').should('be.visible');
        cy.get('[data-testid="avg-gas-used"]').should('be.visible');
        cy.get('[data-testid="total-gas-spent"]').should('be.visible');
      });
      
      // Verify transaction volume chart
      cy.get('[data-testid="tx-volume-chart"]').should('be.visible');
      cy.get('[data-testid="tx-volume-chart"] canvas').should('exist');
      
      // Test chart interactions
      cy.get('[data-testid="tx-volume-chart"] canvas').trigger('mousemove', 100, 50);
      cy.get('[data-testid="chart-tooltip"]').should('be.visible');
      
      // Check gas usage analytics
      cy.get('[data-testid="gas-analytics"]').should('be.visible');
      cy.get('[data-testid="gas-usage-chart"]').should('be.visible');
      cy.get('[data-testid="gas-price-trend"]').should('be.visible');
      
      // Test time range filtering
      cy.get('[data-testid="analytics-timerange"]').select('Last 30 days');
      cy.get('[data-testid="analytics-loading"]').should('be.visible');
      cy.get('[data-testid="analytics-loading"]').should('not.exist');
      
      // Check transaction type distribution
      cy.get('[data-testid="tx-type-distribution"]').should('be.visible');
      cy.get('[data-testid="distribution-chart"]').should('be.visible');
      
      // Test export analytics data
      cy.get('[data-testid="export-analytics"]').click();
      cy.get('[data-testid="export-format-csv"]').click();
      cy.get('[data-testid="start-export"]').click();
      
      cy.get('[data-testid="export-complete"]').should('be.visible');
      cy.readFile('cypress/downloads/transaction-analytics.csv').should('exist');
      
      // Test analytics alerts
      cy.get('[data-testid="analytics-alerts"]').click();
      cy.get('[data-testid="alert-high-gas"]').check();
      cy.get('[data-testid="alert-threshold"]').clear().type('50');
      cy.get('[data-testid="save-alerts"]').click();
      
      cy.get('[data-testid="alerts-configured"]').should('be.visible');
    });
  });
});