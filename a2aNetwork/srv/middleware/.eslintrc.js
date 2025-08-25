/**
 * ESLint configuration for middleware security modules
 * These modules require specific regex patterns that may trigger false positives
 */

module.exports = {
    rules: {
        // Security regex patterns often need escapes that ESLint thinks are unnecessary
        'no-useless-escape': 'off',

        // Security modules may have unreachable code for defense in depth
        'no-unreachable': 'off',

        // Allow inner function declarations for security isolation
        'no-inner-declarations': 'off',

        // Security checks may have constant conditions by design
        'no-constant-condition': 'off'
    }
};