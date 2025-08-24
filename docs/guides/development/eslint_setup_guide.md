# ESLint Configuration Guide

## Available Templates

### 1. SAP UI5 Projects (`eslintrc.sapui5.json`)
Optimized for SAP UI5 applications with jQuery and ES6+ features.
- Includes all SAP global variables
- Enforces modern JavaScript practices
- Allows console.warn and console.error

### 2. Node.js Backend (`eslintrc.nodejs.json`)
For server-side Node.js applications.
- Allows console statements
- Enforces async/await best practices
- Module-based configuration

### 3. React Applications (`eslintrc.react.json`)
For React/JSX projects.
- Includes React-specific rules
- JSX support
- React Hooks linting

## Usage

1. Copy the appropriate template to your project root
2. Rename to `.eslintrc.json`
3. Install required dependencies:
   ```bash
   npm install --save-dev eslint
   # For React projects:
   npm install --save-dev eslint-plugin-react eslint-plugin-react-hooks
   ```
4. Add to package.json scripts:
   ```json
   "scripts": {
     "lint": "eslint .",
     "lint:fix": "eslint . --fix"
   }
   ```

## Customization

Modify rules based on your team's preferences:
- `"off"` or `0` - Turn off the rule
- `"warn"` or `1` - Warning (doesn't affect exit code)
- `"error"` or `2` - Error (exit code 1)
