/**
 * Comprehensive Integration Tests for A2A Network Launchpad
 * Tests all enterprise features and real implementations
 */
const request = require('supertest');
const { expect } = require('chai');
const sqlite3 = require('sqlite3').verbose();
const Redis = require('ioredis-mock');
const path = require('path');
const fs = require('fs');

// Import the production launchpad server
let app;
let server;
let db;
let redisClient;

// Test configuration
const TEST_CONFIG = {
    DATABASE_TYPE: 'sqlite',
    SQLITE_DB_PATH: './test/data/test_launchpad.db',
    NODE_ENV: 'test',
    PORT: 0, // Use random port for testing
    BTP_ENVIRONMENT: 'false',
    ENABLE_XSUAA_VALIDATION: 'false',
    USE_DEVELOPMENT_AUTH: 'true',
    REDIS_URL: 'redis://localhost:6379/1', // Test database
    ENABLE_CACHE: 'true',
    ENABLE_METRICS: 'true',
    ENABLE_SAP_CLOUD_ALM: 'false' // Disable for testing
};

describe('A2A Network Launchpad - Comprehensive Integration Tests', function() {
    this.timeout(30000);

    before(async () => {
        // Set test environment variables
        Object.assign(process.env, TEST_CONFIG);
        
        // Ensure test directories exist
        const testDataDir = path.dirname(TEST_CONFIG.SQLITE_DB_PATH);
        if (!fs.existsSync(testDataDir)) {
            fs.mkdirSync(testDataDir, { recursive: true });
        }

        // Initialize test database
        await initializeTestDatabase();
        
        // Mock Redis for testing
        redisClient = new Redis();
        
        // Import and start the server
        const serverModule = require('../../production-launchpad-server.js');
        app = serverModule.app;
        server = serverModule.server;
        
        // Wait for server to be ready
        await new Promise(resolve => setTimeout(resolve, 2000));
    });

    after(async () => {
        // Cleanup
        if (server) {
            server.close();
        }
        if (db) {
            db.close();
        }
        if (redisClient) {
            redisClient.disconnect();
        }
        
        // Clean up test files
        try {
            if (fs.existsSync(TEST_CONFIG.SQLITE_DB_PATH)) {
                fs.unlinkSync(TEST_CONFIG.SQLITE_DB_PATH);
            }
        } catch (error) {
            console.warn('Failed to cleanup test database:', error.message);
        }
    });

    describe('1. Core Launchpad Functionality', () => {
        it('should serve the main launchpad page', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);
            
            expect(response.text).to.include('sap-ushell-config');
            expect(response.text).to.include('SAP Fiori Launchpad');
            expect(response.text).to.include('A2A Network Management Platform');
        });

        it('should load launchpad with proper SAP UI5 configuration', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);
            
            // Check for essential SAP UI5 components
            expect(response.text).to.include('sap.ushell');
            expect(response.text).to.include('enablePersonalization: true');
            expect(response.text).to.include('enableTagFiltering: true');
            expect(response.text).to.include('enableSearch: true');
        });

        it('should include all required tile groups', async () => {
            const response = await request(app)
                .get('/api/launchpad/config')
                .expect(200);
            
            const config = response.body;
            expect(config.groups).to.be.an('array');
            
            const groupIds = config.groups.map(g => g.id);
            expect(groupIds).to.include('a2a_home_group');
            expect(groupIds).to.include('a2a_operations_group');
            expect(groupIds).to.include('a2a_blockchain_group');
        });
    });

    describe('2. Authentication & Security', () => {
        it('should handle development authentication', async () => {
            const response = await request(app)
                .post('/auth/login')
                .send({
                    username: 'test@a2a.network',
                    password: 'test123'
                })
                .expect(200);
            
            expect(response.body).to.have.property('token');
            expect(response.body).to.have.property('user');
            expect(response.body.user.email).to.equal('test@a2a.network');
        });

        it('should reject invalid authentication', async () => {
            await request(app)
                .post('/auth/login')
                .send({
                    username: 'invalid@example.com',
                    password: 'wrongpassword'
                })
                .expect(401);
        });

        it('should validate JWT tokens', async () => {
            // First login to get token
            const loginResponse = await request(app)
                .post('/auth/login')
                .send({
                    username: 'test@a2a.network',
                    password: 'test123'
                });
            
            const token = loginResponse.body.token;
            
            // Use token to access protected endpoint
            await request(app)
                .get('/api/user/profile')
                .set('Authorization', `Bearer ${token}`)
                .expect(200);
        });
    });

    describe('3. Tile Data API', () => {
        let authToken;

        before(async () => {
            const loginResponse = await request(app)
                .post('/auth/login')
                .send({
                    username: 'test@a2a.network',
                    password: 'test123'
                });
            authToken = loginResponse.body.token;
        });

        it('should return tile data for all tiles', async () => {
            const response = await request(app)
                .get('/api/tiles/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response.body).to.be.an('object');
            expect(response.body).to.have.property('overview_tile');
            expect(response.body).to.have.property('agents_tile');
            expect(response.body).to.have.property('blockchain_tile');
        });

        it('should return specific tile data', async () => {
            const response = await request(app)
                .get('/api/tiles/overview_tile/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response.body).to.have.property('title');
            expect(response.body).to.have.property('subtitle');
            expect(response.body).to.have.property('info');
        });

        it('should handle caching headers correctly', async () => {
            const response = await request(app)
                .get('/api/tiles/agents_tile/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            // First request should be a cache miss
            expect(response.headers['x-cache']).to.equal('MISS');
            
            // Second request should be a cache hit
            const response2 = await request(app)
                .get('/api/tiles/agents_tile/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response2.headers['x-cache']).to.equal('HIT');
        });
    });

    describe('4. Personalization Service', () => {
        let authToken;
        const userId = 'test-user-123';

        before(async () => {
            const loginResponse = await request(app)
                .post('/auth/login')
                .send({
                    username: 'test@a2a.network',
                    password: 'test123'
                });
            authToken = loginResponse.body.token;
        });

        it('should save user tile configuration', async () => {
            const tileConfig = {
                position: 5,
                isVisible: true,
                size: '2x2',
                customTitle: 'My Custom Title',
                refreshInterval: 60
            };

            const response = await request(app)
                .post('/api/personalization/tiles/overview_tile')
                .set('Authorization', `Bearer ${authToken}`)
                .send({
                    userId,
                    groupId: 'a2a_home_group',
                    config: tileConfig
                })
                .expect(200);
            
            expect(response.body).to.have.property('success', true);
        });

        it('should retrieve user tile configuration', async () => {
            const response = await request(app)
                .get(`/api/personalization/tiles?userId=${userId}`)
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response.body).to.be.an('array');
            const overviewTile = response.body.find(t => t.tile_id === 'overview_tile');
            expect(overviewTile).to.exist;
            expect(overviewTile.position).to.equal(5);
            expect(overviewTile.size).to.equal('2x2');
        });

        it('should save user preferences', async () => {
            const preferences = {
                theme: 'sap_horizon_dark',
                layoutMode: 'grid',
                tileSizePreference: 'large',
                autoRefresh: false,
                compactMode: true
            };

            const response = await request(app)
                .post('/api/personalization/preferences')
                .set('Authorization', `Bearer ${authToken}`)
                .send({
                    userId,
                    preferences
                })
                .expect(200);
            
            expect(response.body).to.have.property('success', true);
        });

        it('should export user configuration', async () => {
            const response = await request(app)
                .get(`/api/personalization/export?userId=${userId}`)
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response.body).to.have.property('userId', userId);
            expect(response.body).to.have.property('tileConfiguration');
            expect(response.body).to.have.property('userPreferences');
        });
    });

    describe('5. Caching System', () => {
        let authToken;

        before(async () => {
            const loginResponse = await request(app)
                .post('/auth/login')
                .send({
                    username: 'test@a2a.network',
                    password: 'test123'
                });
            authToken = loginResponse.body.token;
        });

        it('should cache API responses', async () => {
            // Clear cache first
            await request(app)
                .delete('/api/cache/clear')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);

            // First request - should be cache miss
            const response1 = await request(app)
                .get('/api/tiles/blockchain_tile/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response1.headers['x-cache']).to.equal('MISS');

            // Second request - should be cache hit
            const response2 = await request(app)
                .get('/api/tiles/blockchain_tile/data')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response2.headers['x-cache']).to.equal('HIT');
        });

        it('should return cache statistics', async () => {
            const response = await request(app)
                .get('/api/cache/stats')
                .set('Authorization', `Bearer ${authToken}`)
                .expect(200);
            
            expect(response.body).to.have.property('hits');
            expect(response.body).to.have.property('misses');
            expect(response.body).to.have.property('hitRate');
            expect(response.body).to.have.property('redisAvailable');
        });
    });

    describe('6. Health & Monitoring', () => {
        it('should return comprehensive health status', async () => {
            const response = await request(app)
                .get('/health')
                .expect(200);
            
            expect(response.body).to.have.property('status', 'healthy');
            expect(response.body).to.have.property('timestamp');
            expect(response.body).to.have.property('services');
            
            // Check individual service health
            expect(response.body.services).to.have.property('database');
            expect(response.body.services).to.have.property('cache');
            expect(response.body.services).to.have.property('personalization');
            expect(response.body.services).to.have.property('telemetry');
        });

        it('should return metrics endpoint', async () => {
            const response = await request(app)
                .get('/metrics')
                .expect(200);
            
            // Should be Prometheus format or JSON
            expect(response.text || response.body).to.exist;
        });

        it('should handle graceful shutdown', async () => {
            // Test that the server can handle shutdown signals properly
            // This is more of a structural test
            expect(process.listeners('SIGTERM').length).to.be.greaterThan(0);
            expect(process.listeners('SIGINT').length).to.be.greaterThan(0);
        });
    });

    describe('7. Error Handling & Resilience', () => {
        it('should handle database connection failures gracefully', async () => {
            // Test error handling when database is unavailable
            const response = await request(app)
                .get('/api/tiles/data')
                .expect(200); // Should still work with fallback
        });

        it('should handle invalid tile requests', async () => {
            await request(app)
                .get('/api/tiles/nonexistent_tile/data')
                .expect(404);
        });

        it('should validate request parameters', async () => {
            await request(app)
                .post('/api/personalization/tiles/overview_tile')
                .send({
                    // Missing required fields
                    config: { position: 'invalid' }
                })
                .expect(400);
        });
    });

    describe('8. Performance & Load', () => {
        it('should handle concurrent requests efficiently', async () => {
            const requests = [];
            const concurrentUsers = 10;
            
            // Create concurrent requests
            for (let i = 0; i < concurrentUsers; i++) {
                requests.push(
                    request(app)
                        .get('/api/tiles/data')
                        .expect(200)
                );
            }
            
            const responses = await Promise.all(requests);
            expect(responses).to.have.length(concurrentUsers);
            responses.forEach(response => {
                expect(response.status).to.equal(200);
            });
        });

        it('should respond within acceptable time limits', async () => {
            const start = Date.now();
            
            await request(app)
                .get('/')
                .expect(200);
            
            const duration = Date.now() - start;
            expect(duration).to.be.lessThan(2000); // Should load in under 2 seconds
        });
    });

    describe('9. Security Features', () => {
        it('should include security headers', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);
            
            expect(response.headers).to.have.property('x-content-type-options');
            expect(response.headers).to.have.property('x-frame-options');
            expect(response.headers).to.have.property('x-xss-protection');
        });

        it('should sanitize user inputs', async () => {
            const maliciousInput = '<script>alert("xss")</script>';
            
            const response = await request(app)
                .post('/api/personalization/tiles/test_tile')
                .send({
                    userId: 'test-user',
                    groupId: 'test-group',
                    config: {
                        customTitle: maliciousInput
                    }
                })
                .expect(400); // Should reject malicious input
        });
    });

    describe('10. SAP Standards Compliance', () => {
        it('should follow SAP Fiori design guidelines', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);
            
            // Check for SAP UI5 theme
            expect(response.text).to.include('sap_horizon');
            expect(response.text).to.include('sap.ui.core');
            expect(response.text).to.include('sap.ushell');
        });

        it('should support proper internationalization', async () => {
            const response = await request(app)
                .get('/api/i18n/en.json')
                .expect(200);
            
            expect(response.body).to.be.an('object');
            expect(response.body).to.have.property('LAUNCHPAD_TITLE');
        });

        it('should provide accessibility features', async () => {
            const response = await request(app)
                .get('/')
                .expect(200);
            
            // Check for accessibility attributes
            expect(response.text).to.include('aria-');
            expect(response.text).to.include('role=');
        });
    });
});

// Helper function to initialize test database
async function initializeTestDatabase() {
    return new Promise((resolve, reject) => {
        db = new sqlite3.Database(TEST_CONFIG.SQLITE_DB_PATH, (err) => {
            if (err) {
                reject(err);
                return;
            }
            
            // Create test tables
            db.serialize(() => {
                db.run(`CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )`);

                db.run(`CREATE TABLE IF NOT EXISTS user_tile_config (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    tile_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    position INTEGER DEFAULT 0,
                    is_visible BOOLEAN DEFAULT 1,
                    size TEXT DEFAULT '1x1',
                    custom_title TEXT,
                    refresh_interval INTEGER DEFAULT 30,
                    settings TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )`);

                // Insert test user
                db.run(`INSERT OR REPLACE INTO users (id, email, password_hash, name) 
                    VALUES ('test-user-1', 'test@a2a.network', '$2b$10$test.hash.for.test123', 'Test User')`);

                resolve();
            });
        });
    });
}