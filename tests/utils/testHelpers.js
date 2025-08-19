// A2A Enterprise Test Helpers
// Common utilities for all test types

const crypto = require('crypto');

/**
 * Test data generators
 */
const generators = {
  // Generate unique test ID
  uniqueId: () => crypto.randomBytes(16).toString('hex'),
  
  // Generate mock agent data
  mockAgent: (overrides = {}) => ({
    id: generators.uniqueId(),
    name: 'Test Agent',
    type: 'reasoning',
    status: 'active',
    capabilities: ['search', 'analysis'],
    metadata: {
      version: '1.0.0',
      author: 'test-system',
      created: new Date().toISOString()
    },
    ...overrides
  }),
  
  // Generate mock user data  
  mockUser: (overrides = {}) => ({
    id: generators.uniqueId(),
    username: 'testuser',
    email: 'test@example.com',
    role: 'user',
    permissions: ['read'],
    ...overrides
  }),
  
  // Generate mock blockchain transaction
  mockTransaction: (overrides = {}) => ({
    hash: '0x' + generators.uniqueId(),
    from: '0x' + generators.uniqueId(),
    to: '0x' + generators.uniqueId(),
    value: '1000000000000000000',
    gasUsed: 21000,
    status: 'success',
    timestamp: new Date().toISOString(),
    ...overrides
  })
};

/**
 * Test database utilities
 */
const database = {
  // Reset test database to clean state
  reset: async () => {
    // Implementation depends on your database setup
    console.log('Resetting test database...');
  },
  
  // Seed test data
  seed: async (data = {}) => {
    console.log('Seeding test database with:', data);
    // Implementation depends on your database setup
  },
  
  // Clean up test data
  cleanup: async () => {
    console.log('Cleaning up test database...');
    // Implementation depends on your database setup
  }
};

/**
 * API testing utilities
 */
const api = {
  // Mock successful response
  mockSuccess: (data = {}, status = 200) => ({
    status,
    data,
    headers: { 'content-type': 'application/json' }
  }),
  
  // Mock error response
  mockError: (message = 'Test error', status = 500) => ({
    status,
    data: { error: message },
    headers: { 'content-type': 'application/json' }
  }),
  
  // Create authenticated headers
  authHeaders: (token = 'test-token') => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  })
};

/**
 * Async test utilities
 */
const async = {
  // Wait for specified time
  wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
  
  // Wait for condition to be true
  waitFor: async (condition, timeout = 5000, interval = 100) => {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      if (await condition()) {
        return true;
      }
      await async.wait(interval);
    }
    throw new Error(`Condition not met within ${timeout}ms`);
  },
  
  // Retry function with exponential backoff
  retry: async (fn, maxAttempts = 3, delay = 1000) => {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxAttempts) throw error;
        await async.wait(delay * Math.pow(2, attempt - 1));
      }
    }
  }
};

/**
 * Test environment utilities
 */
const environment = {
  // Check if running in CI
  isCI: () => process.env.CI === 'true',
  
  // Get test environment
  getEnv: () => process.env.NODE_ENV || 'test',
  
  // Set test environment variables
  setTestVars: (vars = {}) => {
    Object.entries(vars).forEach(([key, value]) => {
      process.env[key] = value;
    });
  },
  
  // Restore environment variables
  restoreVars: (originalEnv = {}) => {
    Object.entries(originalEnv).forEach(([key, value]) => {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    });
  }
};

module.exports = {
  generators,
  database,
  api,
  async,
  environment
};