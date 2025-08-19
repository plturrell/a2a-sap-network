const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.js',
    specPattern: 'tests/e2e/**/*.cy.{js,jsx,ts,tsx}',
    excludeSpecPattern: [
      'tests/e2e/**/legacy/**',
      'tests/e2e/**/temp/**'
    ],
    videosFolder: 'cypress/videos',
    screenshotsFolder: 'cypress/screenshots',
    viewportWidth: 1280,
    viewportHeight: 720,
    defaultCommandTimeout: 10000,
    requestTimeout: 10000,
    responseTimeout: 10000,
    video: true,
    screenshotOnRunFailure: true,
    
    // Test retries for stability
    retries: {
      runMode: 2,
      openMode: 0
    },
    
    // Environment variables for different test environments
    env: {
      apiUrl: 'http://localhost:8000/api',
      a2aNetworkUrl: 'http://localhost:3001',
      a2aAgentsUrl: 'http://localhost:3002'
    },
    
    setupNodeEvents(on, config) {
      // Implement node event listeners here
      on('task', {
        log(message) {
          console.log(message);
          return null;
        },
        table(message) {
          console.table(message);
          return null;
        }
      });
      
      // Code coverage
      require('@cypress/code-coverage/task')(on, config);
      
      return config;
    }
  },
  
  component: {
    devServer: {
      framework: 'react',
      bundler: 'webpack'
    },
    specPattern: 'tests/components/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/component.js'
  }
});