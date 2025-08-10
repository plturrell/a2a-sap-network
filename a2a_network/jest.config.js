module.exports = {
  preset: '@sap/cds-jest',
  testEnvironment: 'node',
  testTimeout: 20000,
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  collectCoverageFrom: [
    'srv/**/*.js',
    'app/**/*.js',
    '!**/node_modules/**',
    '!**/gen/**',
    '!**/dist/**'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  testMatch: [
    '**/test/**/*.test.js',
    '**/tests/**/*.test.js',
    '**/__tests__/**/*.js'
  ],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/srv/$1',
    '^@db/(.*)$': '<rootDir>/db/$1',
    '^@app/(.*)$': '<rootDir>/app/$1'
  },
  setupFilesAfterEnv: ['<rootDir>/test/setup.js'],
  verbose: true,
  testResultsProcessor: 'jest-sonar-reporter',
  globals: {
    'cds': {
      'env': {
        'production': false
      }
    }
  }
};