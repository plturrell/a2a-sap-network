module.exports = {
    extends: ["eslint:recommended"],
    env: {
        node: true,
        es2022: true,
        jest: true
    },
    parserOptions: {
        ecmaVersion: 2022,
        sourceType: "module"
    },
    globals: {
        // SAP CAP globals
        'SELECT': 'readonly',
        'INSERT': 'readonly', 
        'UPDATE': 'readonly',
        'DELETE': 'readonly',
        'CREATE': 'readonly',
        'DROP': 'readonly',
        'cds': 'readonly',
        
        // SAP UI5 globals
        'sap': 'readonly',
        'jQuery': 'readonly',
        '$': 'readonly',
        
        // Web3 and blockchain globals
        'Web3': 'readonly',
        'web3': 'readonly',
        'blockchainClient': 'readonly',
        'BlockchainEventClient': 'readonly',
        
        // Custom A2A globals
        'callAgent2Backend': 'readonly',
        'axios': 'readonly',
        'activeIntervals': 'writable',
        'eventsToProcess': 'writable'
    },
    rules: {
        "no-console": "warn",
        "no-unused-vars": ["error", { 
            "argsIgnorePattern": "^_",
            "varsIgnorePattern": "^_"
        }],
        "no-undef": "error",
        "prefer-const": ["error", {
            "destructuring": "any",
            "ignoreReadBeforeAssign": false
        }],
        "no-useless-catch": "warn",
        "no-inner-declarations": "warn"
    },
    overrides: [
        {
            // UI5 controller files
            files: ["**/webapp/**/*.js", "**/controller/**/*.js"],
            globals: {
                'sap': 'readonly'
            },
            rules: {
                "no-undef": "off" // UI5 has complex global patterns
            }
        },
        {
            // Service implementation files
            files: ["srv/**/*service.js", "srv/**/*Service.js"],
            globals: {
                'SELECT': 'readonly',
                'INSERT': 'readonly', 
                'UPDATE': 'readonly',
                'DELETE': 'readonly'
            }
        }
    ]
};