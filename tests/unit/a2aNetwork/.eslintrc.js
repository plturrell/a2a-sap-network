/**
 * ESLint configuration for test files
 * Test files have different patterns and requirements
 */

module.exports = {
    env: {
        node: true,
        mocha: true,
        jest: true
    },
    globals: {
        before: 'readonly',
        after: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        assert: 'readonly',
        sinon: 'readonly'
    },
    rules: {
        // Test files often have unused variables for setup
        'no-unused-vars': 'off',
        
        // Test globals are defined by test framework
        'no-undef': 'off',
        
        // Test files may have inner function declarations
        'no-inner-declarations': 'off',
        
        // Allow prototype builtins in tests
        'no-prototype-builtins': 'off',
        
        // Test assertions might look like constant conditions
        'no-constant-condition': 'off',
        
        // Allow shadow variables in tests
        'no-shadow': 'off'
    }
};