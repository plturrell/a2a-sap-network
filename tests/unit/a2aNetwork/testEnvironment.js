/**
 * Comprehensive Test Environment Setup for A2A Launchpad Common Components
 * Provides complete browser API mocking and test utilities
 */

const { JSDOM } = require('jsdom');
const crypto = require('crypto');

class TestEnvironmentSetup {
    constructor() {
        this.dom = null;
        this.mockStorage = new Map();
        this.mockAlerts = [];
        this.mockConsoleOutput = [];
    }

    /**
     * Initialize comprehensive browser environment for testing
     */
    setupBrowserEnvironment() {
        // Create JSDOM with comprehensive options
        this.dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
            url: 'http://localhost:4004',
            pretendToBeVisual: true,
            resources: 'usable',
            runScripts: 'dangerously'
        });

        // Setup global objects (handle read-only properties)
        global.window = this.dom.window;
        global.document = this.dom.window.document;
        
        // Handle navigator separately due to Node.js restrictions
        if (!global.navigator) {
            try {
                global.navigator = this.dom.window.navigator;
            } catch (e) {
                // Navigator might be read-only in some Node.js versions
                Object.defineProperty(global, 'navigator', {
                    value: this.dom.window.navigator,
                    writable: true,
                    configurable: true
                });
            }
        }
        
        global.location = this.dom.window.location;

        // Mock HTML5 storage APIs
        this.setupStorageAPIs();
        
        // Mock browser APIs
        this.setupBrowserAPIs();
        
        // Setup event handling
        this.setupEventHandling();
        
        // Mock performance APIs
        this.setupPerformanceAPIs();
        
        // Setup URL APIs
        this.setupURLAPIs();

        console.log('✅ Browser environment setup complete');
    }

    /**
     * Setup storage APIs (localStorage, sessionStorage)
     */
    setupStorageAPIs() {
        const createMockStorage = () => ({
            store: new Map(),
            getItem: function(key) {
                return this.store.get(key) || null;
            },
            setItem: function(key, value) {
                this.store.set(key, String(value));
            },
            removeItem: function(key) {
                this.store.delete(key);
            },
            clear: function() {
                this.store.clear();
            },
            get length() {
                return this.store.size;
            },
            key: function(index) {
                const keys = Array.from(this.store.keys());
                return keys[index] || null;
            }
        });

        global.localStorage = createMockStorage();
        global.sessionStorage = createMockStorage();
        global.window.localStorage = global.localStorage;
        global.window.sessionStorage = global.sessionStorage;
    }

    /**
     * Setup browser APIs (alert, confirm, prompt, etc.)
     */
    setupBrowserAPIs() {
        // Mock dialog APIs
        global.alert = (message) => {
            this.mockAlerts.push({ type: 'alert', message, timestamp: Date.now() });
            console.log(`[MOCK ALERT]: ${message}`);
        };

        global.confirm = (message) => {
            this.mockAlerts.push({ type: 'confirm', message, timestamp: Date.now() });
            console.log(`[MOCK CONFIRM]: ${message}`);
            return true; // Default to confirm for tests
        };

        global.prompt = (message, defaultValue = '') => {
            this.mockAlerts.push({ type: 'prompt', message, defaultValue, timestamp: Date.now() });
            console.log(`[MOCK PROMPT]: ${message}`);
            return defaultValue;
        };

        global.window.alert = global.alert;
        global.window.confirm = global.confirm;
        global.window.prompt = global.prompt;

        // Mock fetch API
        global.fetch = require('node-fetch');
        global.window.fetch = global.fetch;

        // Mock crypto API (handle read-only global)
        const cryptoMock = {
            ...crypto,
            getRandomValues: (array) => {
                return crypto.randomFillSync(array);
            },
            randomUUID: () => crypto.randomUUID()
        };
        
        if (!global.crypto) {
            try {
                global.crypto = cryptoMock;
            } catch (e) {
                Object.defineProperty(global, 'crypto', {
                    value: cryptoMock,
                    writable: true,
                    configurable: true
                });
            }
        }
        global.window.crypto = global.crypto;
    }

    /**
     * Setup event handling and timers
     */
    setupEventHandling() {
        // Mock history API with proper event handling
        const originalPushState = this.dom.window.history.pushState;
        const originalReplaceState = this.dom.window.history.replaceState;

        this.dom.window.history.pushState = function(state, title, url) {
            originalPushState.call(this, state, title, url);
            
            // Trigger popstate event for navigation testing
            const event = new this.dom.window.Event('pushstate');
            event.state = state;
            global.window.dispatchEvent(event);
        };

        this.dom.window.history.replaceState = function(state, title, url) {
            originalReplaceState.call(this, state, title, url);
            
            const event = new this.dom.window.Event('replacestate');
            event.state = state;
            global.window.dispatchEvent(event);
        };

        // Mock setTimeout/setInterval if needed
        global.setTimeout = this.dom.window.setTimeout;
        global.setInterval = this.dom.window.setInterval;
        global.clearTimeout = this.dom.window.clearTimeout;
        global.clearInterval = this.dom.window.clearInterval;
    }

    /**
     * Setup performance measurement APIs
     */
    setupPerformanceAPIs() {
        global.performance = {
            now: () => Date.now(),
            mark: (name) => console.log(`[PERF MARK]: ${name}`),
            measure: (name, start, end) => console.log(`[PERF MEASURE]: ${name} (${start} -> ${end})`),
            getEntriesByType: () => [],
            getEntriesByName: () => []
        };
        global.window.performance = global.performance;
    }

    /**
     * Setup URL APIs for proper URL handling
     */
    setupURLAPIs() {
        // Ensure URL constructor is available
        global.URL = this.dom.window.URL;
        global.URLSearchParams = this.dom.window.URLSearchParams;

        // Mock location with better URL handling
        const mockLocation = {
            href: 'http://localhost:4004/',
            protocol: 'http:',
            host: 'localhost:4004',
            hostname: 'localhost',
            port: '4004',
            pathname: '/',
            search: '',
            hash: '',
            origin: 'http://localhost:4004',
            
            assign: function(url) {
                this.href = url;
                console.log(`[LOCATION ASSIGN]: ${url}`);
            },
            
            replace: function(url) {
                this.href = url;
                console.log(`[LOCATION REPLACE]: ${url}`);
            },
            
            reload: function() {
                console.log('[LOCATION RELOAD]');
            },

            toString: function() {
                return this.href;
            }
        };

        // Make location writable for navigation tests (handle existing property)
        try {
            if (global.window.location) {
                // Try to update existing location
                Object.assign(global.window.location, mockLocation);
            } else {
                Object.defineProperty(global.window, 'location', {
                    value: mockLocation,
                    writable: true,
                    configurable: true
                });
            }
        } catch (e) {
            // Fallback: Create a separate mock location
            global.mockLocation = mockLocation;
            console.log('Using global.mockLocation due to property restrictions');
        }

        global.location = global.window.location;
    }

    /**
     * Setup comprehensive error handling
     */
    setupErrorHandling() {
        // Global error handler
        global.window.addEventListener('error', (event) => {
            console.log(`[WINDOW ERROR]: ${event.message} at ${event.filename}:${event.lineno}`);
        });

        // Unhandled promise rejection handler
        global.window.addEventListener('unhandledrejection', (event) => {
            console.log(`[UNHANDLED PROMISE REJECTION]: ${event.reason}`);
        });

        // Console capture for testing
        const originalConsole = { ...console };
        console.log = (...args) => {
            this.mockConsoleOutput.push({ level: 'log', args, timestamp: Date.now() });
            originalConsole.log(...args);
        };
    }

    /**
     * Reset environment between tests
     */
    reset() {
        // Clear storage
        if (global.localStorage) global.localStorage.clear();
        if (global.sessionStorage) global.sessionStorage.clear();
        
        // Clear mock data
        this.mockAlerts = [];
        this.mockConsoleOutput = [];
        
        // Reset URL safely
        try {
            if (global.location && typeof global.location === 'object') {
                global.location.href = 'http://localhost:4004/';
                global.location.hash = '';
            } else if (global.mockLocation) {
                global.mockLocation.href = 'http://localhost:4004/';
                global.mockLocation.hash = '';
            }
        } catch (e) {
            // Location might be read-only, ignore reset error
            console.log('Location reset skipped due to restrictions');
        }
        
        console.log('🔄 Test environment reset');
    }

    /**
     * Get mock data for assertions
     */
    getMockData() {
        return {
            alerts: [...this.mockAlerts],
            consoleOutput: [...this.mockConsoleOutput],
            localStorage: global.localStorage ? Object.fromEntries(global.localStorage.store) : {},
            sessionStorage: global.sessionStorage ? Object.fromEntries(global.sessionStorage.store) : {}
        };
    }

    /**
     * Cleanup environment
     */
    cleanup() {
        if (this.dom) {
            this.dom.window.close();
        }
        
        // Remove globals
        delete global.window;
        delete global.document;
        delete global.navigator;
        delete global.location;
        delete global.localStorage;
        delete global.sessionStorage;
        delete global.alert;
        delete global.confirm;
        delete global.prompt;
        delete global.fetch;
        delete global.crypto;
        delete global.performance;
        delete global.URL;
        delete global.URLSearchParams;
        
        console.log('🧹 Test environment cleaned up');
    }
}

module.exports = TestEnvironmentSetup;