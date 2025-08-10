/**
 * Jest Configuration for SAP CAP Application
 * Enterprise-grade testing configuration with 90% coverage requirement
 */

module.exports = {
    testEnvironment: 'node',
    roots: ['<rootDir>/srv', '<rootDir>/test'],
    testMatch: [
        '**/__tests__/**/*.js',
        '**/*.test.js',
        '**/*.spec.js'
    ],
    collectCoverageFrom: [
        'srv/**/*.js',
        '!srv/node_modules/**',
        '!srv/**/*.test.js',
        '!srv/**/*.spec.js',
        '!srv/coverage/**'
    ],
    coverageThreshold: {
        global: {
            branches: 90,
            functions: 90,
            lines: 90,
            statements: 90
        }
    },
    coverageReporters: [
        'text',
        'lcov',
        'html',
        'json-summary'
    ],
    coverageDirectory: 'coverage',
    moduleFileExtensions: ['js', 'json'],
    transform: {
        '^.+\\.js$': 'babel-jest'
    },
    setupFilesAfterEnv: ['<rootDir>/test/setup.js'],
    testTimeout: 30000,
    verbose: true,
    bail: false,
    errorOnDeprecated: true,
    reporters: [
        'default',
        ['jest-junit', {
            outputDirectory: './test-results',
            outputName: 'junit.xml',
            classNameTemplate: '{classname}',
            titleTemplate: '{title}',
            ancestorSeparator: ' › ',
            usePathForSuiteName: true
        }]
    ]
};