// A2A Enterprise Test Configuration
// Unified Jest configuration following SAP enterprise standards

"use strict";

module.exports = {
  // Base configuration
  preset: '@sap/cds-jest',
  testEnvironment: 'node',
  testTimeout: 30000,
  
  // Test discovery patterns
  projects: [
    {
      displayName: 'Unit Tests',
      testMatch: ['<rootDir>/tests/unit/**/*.test.js'],
      collectCoverageFrom: [
        'a2aNetwork/srv/**/*.js',
        'a2aNetwork/app/**/*.js',
        'a2aNetwork/common/**/*.js',
        'a2aAgents/**/*.py',
        '!**/node_modules/**',
        '!**/test/**',
        '!**/tests/**',
        '!**/*.test.*',
        '!**/*.spec.*'
      ]
    },
    {
      displayName: 'Integration Tests',
      testMatch: ['<rootDir>/tests/integration/**/*.test.js'],
      testTimeout: 60000,
      collectCoverageFrom: [
        'a2aNetwork/srv/**/*.js',
        'a2aAgents/**/*.py'
      ]
    },
    {
      displayName: 'Contract Tests',
      testMatch: ['<rootDir>/tests/contracts/**/*.test.js'],
      testEnvironment: 'node'
    }
  ],
  
  // Global test patterns (fallback)
  testMatch: [
    '<rootDir>/tests/**/*.test.js',
    '<rootDir>/tests/**/*.spec.js'
  ],
  
  // Ignore patterns
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/build/',
    '/coverage/',
    '/gen/',
    '/lib/',
    '/docs/'
  ],
  
  // Setup and configuration
  setupFilesAfterEnv: [
    '<rootDir>/tests/config/jest.setup.js'
  ],
  
  // Module resolution
  moduleNameMapper: {
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^@a2aNetwork/(.*)$': '<rootDir>/a2aNetwork/$1',
    '^@a2aAgents/(.*)$': '<rootDir>/a2aAgents/$1',
    '^@common/(.*)$': '<rootDir>/common/$1',
    '^@/(.*)$': '<rootDir>/$1'
  },
  
  moduleDirectories: [
    'node_modules',
    '<rootDir>/tests',
    '<rootDir>/a2aNetwork',
    '<rootDir>/a2aAgents',
    '<rootDir>/common'
  ],
  
  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: '<rootDir>/coverage',
  coverageReporters: [
    'text',
    'text-summary', 
    'lcov',
    'html',
    'json',
    'cobertura'
  ],
  
  // Enterprise coverage thresholds
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  
  // Reporting
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: './test-results',
      outputName: 'junit.xml',
      ancestorSeparator: ' › ',
      uniqueOutputName: 'false',
      suiteNameTemplate: '{displayName}: {filepath}',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}'
    }],
    ['jest-html-reporters', {
      publicPath: './test-results',
      filename: 'report.html',
      expand: true,
      hideIcon: false
    }]
  ],
  
  // Enhanced Jest options
  verbose: true,
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,
  errorOnDeprecated: true,
  detectOpenHandles: true,
  forceExit: true,
  
  // Global setup/teardown
  globalSetup: '<rootDir>/tests/config/globalSetup.js',
  globalTeardown: '<rootDir>/tests/config/globalTeardown.js',
  
  // Watch mode configuration
  watchman: true,
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ],
  
  // Performance and optimization
  maxWorkers: '50%',
  
  // Transform configuration
  transform: {
    '^.+\\.js$': 'babel-jest',
    '^.+\\.ts$': 'ts-jest'
  },
  
  // File extensions to consider
  moduleFileExtensions: [
    'js',
    'ts',
    'json',
    'node'
  ],
  
  // Globals for SAP environments
  globals: {
    'cds': {
      'env': {
        'production': false,
        'test': true
      }
    }
  }
};