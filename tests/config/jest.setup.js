// A2A Enterprise Test Setup
// Global test setup and configuration

// Set test environment variables
process.env.NODE_ENV = 'test';
process.env.LOG_LEVEL = 'error';

// Global test timeout
jest.setTimeout(30000);

// Mock console methods in test environment
global.console = {
  ...console,
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Global test utilities
global.testUtils = {
  // Mock data generators
  generateMockAgent: () => ({
    id: `test-agent-${  Math.random().toString(36).substr(2, 9)}`,
    name: 'Test Agent',
    type: 'reasoning',
    status: 'active',
    capabilities: ['search', 'analysis'],
    createdAt: new Date().toISOString()
  }),
  
  generateMockUser: () => ({
    id: `test-user-${  Math.random().toString(36).substr(2, 9)}`,
    username: 'testuser',
    email: 'test@example.com',
    role: 'user'
  }),
  
  // Test data cleanup
  cleanupTestData: async () => {
    // Clean up any test data that persists between tests
    // Implementation depends on your data storage
  },
  
  // Mock HTTP responses
  mockHttpResponse: (status = 200, data = {}) => ({
    status,
    data,
    headers: { 'content-type': 'application/json' }
  })
};

// Global mocks for external dependencies
jest.mock('axios', () => ({
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} })),
  put: jest.fn(() => Promise.resolve({ data: {} })),
  delete: jest.fn(() => Promise.resolve({ data: {} }))
}));

// Mock blockchain connections for testing
jest.mock('web3', () => {
  return jest.fn().mockImplementation(() => ({
    eth: {
      getAccounts: jest.fn(() => Promise.resolve(['0x123...'])),
      getBalance: jest.fn(() => Promise.resolve('1000000000000000000')),
      sendTransaction: jest.fn(() => Promise.resolve({ hash: '0xabc...' }))
    }
  }));
});

// Setup and teardown hooks
beforeEach(() => {
  // Clear all mocks before each test
  jest.clearAllMocks();
});

afterEach(async () => {
  // Clean up after each test
  await global.testUtils.cleanupTestData();
});

// Unhandled promise rejection handler
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  throw reason;
});

module.exports = {};