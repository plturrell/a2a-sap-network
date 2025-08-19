/**
 * ESLint configuration for SAP UI5/Fiori applications
 * Extends base configuration with UI5-specific rules
 */

module.exports = {
    extends: ["eslint:recommended"],
    env: {
        browser: true,
        jquery: true
    },
    globals: {
        sap: "readonly",
        jQuery: "readonly",
        QUnit: "readonly",
        sinon: "readonly",
        module: "readonly",
        require: "readonly",
        io: "readonly"
    },
    rules: {
        // UI5 specific relaxations
        "no-unused-vars": ["error", {
            "args": "none",
            "varsIgnorePattern": "^_",
            "argsIgnorePattern": "^_|oEvent|oResponse|sTxHash"
        }],
        "max-len": ["error", {
            "code": 120,
            "ignoreStrings": true,
            "ignoreTemplateLiterals": true,
            "ignoreRegExpLiterals": true
        }],

        // Relax for UI5 patterns
        "no-undef": "off", // UI5 global namespaces
        "radix": "error",
        "no-var": "error",
        "prefer-const": "error",

        // Allow some UI5 specific patterns
        "no-new": "off", // UI5 sometimes uses new for side effects
        "no-case-declarations": "off", // Common in switch statements
        "no-prototype-builtins": "off", // hasOwnProperty is common in UI5
        "no-useless-escape": "off", // RegExp patterns may need escapes
        "no-inner-declarations": "off", // UI5 patterns sometimes need this
        "no-shadow": "off", // UI5 variable shadowing is common
        "brace-style": "off", // UI5 formatting style
        "no-unreachable": "error", // Keep important errors
        "no-async-promise-executor": "error", // Keep important errors
        "no-constant-condition": "error", // Keep important errors
        "no-useless-catch": "error" // Keep important errors
    },
    overrides: [
        {
            files: ["test/**/*.js", "**/*.qunit.js"],
            env: {
                qunit: true,
                mocha: true
            },
            globals: {
                before: "readonly",
                after: "readonly",
                beforeEach: "readonly",
                afterEach: "readonly",
                describe: "readonly",
                it: "readonly",
                expect: "readonly",
                assert: "readonly"
            },
            rules: {
                "no-unused-vars": "off", // Test files often have unused vars
                "no-undef": "off" // Test globals
            }
        }
    ]
};