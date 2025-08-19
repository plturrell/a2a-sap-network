/**
 * Jest test setup for SAP CAP application
 */

// Set test environment
process.env.NODE_ENV = 'test';
process.env.CDS_ENV = 'test';

// Mock console methods to reduce test output noise
global.console = {
  ...console,
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn()
};

// Set longer timeout for integration tests
jest.setTimeout(30000);

// Global test utilities
global.testUtils = {
  generateTestAgent: () => ({
    address: `0x${Math.random().toString(16).substr(2, 40)}`,
    name: `Test Agent ${Math.random().toString(36).substr(2, 9)}`,
    endpoint: 'https://test-agent.a2a.network',
    reputation: 100,
    isActive: true
  }),
  
  generateTestService: (providerId) => ({
    provider_ID: providerId,
    name: `Test Service ${Math.random().toString(36).substr(2, 9)}`,
    description: 'Test service description',
    category: 'COMPUTATION',
    pricePerCall: Math.random() * 10,
    currency: 'EUR',
    isActive: true
  }),
  
  generateTestWorkflow: (ownerId) => ({
    name: `Test Workflow ${Math.random().toString(36).substr(2, 9)}`,
    description: 'Test workflow description',
    definition: JSON.stringify({
      steps: [
        { action: 'test', agent: 'agent1' },
        { action: 'validate', agent: 'agent2' }
      ]
    }),
    owner_ID: ownerId,
    isActive: true
  })
};

// Clean up after tests
afterAll(async () => {
  // Close database connections
  const cds = require('@sap/cds');
  if (cds.db) {
    await cds.db.disconnect();
  }
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error('Unhandled promise rejection:', err);
  process.exit(1);
});