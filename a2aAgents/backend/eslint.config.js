"use strict";

import js from '@eslint/js';

export default [
    js.configs.recommended,
    {
        languageOptions: {
            ecmaVersion: 2021,
            sourceType: "module",
            globals: {
                // Browser globals
                console: "readonly",
                window: "readonly",
                document: "readonly",
                navigator: "readonly",
                location: "readonly",
                fetch: "readonly",
                XMLHttpRequest: "readonly",
                
                // Node.js globals
                require: "readonly",
                module: "readonly",
                exports: "readonly",
                process: "readonly",
                global: "readonly",
                __dirname: "readonly",
                __filename: "readonly",
                Buffer: "readonly",
                
                // Timers
                setTimeout: "readonly",
                clearTimeout: "readonly",
                setInterval: "readonly",
                clearInterval: "readonly",
                setImmediate: "readonly",
                clearImmediate: "readonly",
                
                // Common libraries
                jQuery: "readonly",
                $: "readonly",
                _: "readonly",
                moment: "readonly",
                
                // SAP UI5 globals
                sap: "readonly",
                QUnit: "readonly",
                sinon: "readonly",
                URI: "readonly"
            }
        },
        rules: {
            // ES6+ enforcement
            "no-var": "error",
            "prefer-const": ["error", {
                destructuring: "any",
                ignoreReadBeforeAssign: false
            }],
            "prefer-arrow-callback": "error",
            "prefer-template": "error",
            "prefer-spread": "error",
            "prefer-rest-params": "error",
            
            // Code quality
            "no-unused-vars": ["error", {
                argsIgnorePattern: "^_",
                varsIgnorePattern: "^_"
            }],
            "no-undef": "error",
            "no-console": ["warn", {
                allow: ["warn", "error"]
            }],
            "eqeqeq": ["error", "always"],
            "curly": ["error", "all"],
            "brace-style": ["error", "1tbs"],
            
            // Security
            "no-eval": "error",
            "no-implied-eval": "error",
            "no-new-func": "error",
            "no-script-url": "error",
            
            // Best practices
            "no-debugger": "error",
            "no-alert": "warn",
            "no-duplicate-imports": "error",
            "no-empty": "warn",
            "no-extra-semi": "error",
            "no-unreachable": "error",
            "valid-typeof": "error",
            "no-throw-literal": "error",
            "require-await": "error",
            "no-return-await": "error"
        }
    },
    {
        // Special rules for test files
        files: ["**/*.test.js", "**/*.spec.js", "**/test/**/*.js"],
        rules: {
            "no-console": "off",
            "no-unused-expressions": "off"
        }
    },
    {
        // SAP UI5 specific files
        files: ["**/webapp/**/*.js", "**/controller/**/*.js", "**/view/**/*.js"],
        languageOptions: {
            globals: {
                // Additional SAP UI5 globals
                "sap.ui": "readonly",
                "sap.m": "readonly",
                "sap.ui.core": "readonly",
                "sap.ui.model": "readonly",
                "sap.ui.layout": "readonly"
            }
        }
    }
];