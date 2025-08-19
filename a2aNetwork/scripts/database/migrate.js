#!/usr/bin/env node

/**
 * Database Migration Runner for A2A Network
 * Handles schema creation, updates, and versioning
 */

const fs = require('fs').promises;
const path = require('path');
const cds = require('@sap/cds');

class DatabaseMigrator {
    constructor() {
        this.migrationsDir = path.join(__dirname, '../migrations');
        this.db = null;
    }

    async initialize() {
        try {
            log.debug('🔌 Connecting to database...');
            
            // Connect to database based on environment
            const env = process.env.NODE_ENV || 'development';
            
            if (env === 'production') {
                // Use HANA connection for production
                this.db = await cds.connect.to('db');
                log.debug('✅ Connected to SAP HANA Cloud');
            } else {
                // Use SQLite for development/testing
                this.db = await cds.connect.to('db', {
                    kind: 'sqlite',
                    credentials: { url: './data/a2a-network.db' }
                });
                log.debug('✅ Connected to SQLite database');
            }

            // Ensure migrations table exists
            await this.ensureMigrationsTable();
            
        } catch (error) {
            console.error('❌ Database connection failed:', error);
            throw error;
        }
    }

    async ensureMigrationsTable() {
        const createMigrationsTable = `
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum VARCHAR(64),
                execution_time INTEGER
            )
        `;
        
        await this.db.run(createMigrationsTable);
        log.info('📋 Migration tracking table ready');
    }

    async getAppliedMigrations() {
        const result = await this.db.run(
            'SELECT version FROM schema_migrations ORDER BY version'
        );
        return result.map(row => row.version);
    }

    async getAllMigrations() {
        try {
            const files = await fs.readdir(this.migrationsDir);
            return files
                .filter(file => file.endsWith('.js'))
                .sort()
                .map(file => ({
                    version: path.basename(file, '.js'),
                    path: path.join(this.migrationsDir, file)
                }));
        } catch (error) {
            log.debug('📁 Creating migrations directory...');
            await fs.mkdir(this.migrationsDir, { recursive: true });
            return [];
        }
    }

    async runMigration(migration) {
        const startTime = Date.now();
        
        try {
            log.debug(`🚀 Running migration: ${migration.version}`);
            
            const migrationModule = require(migration.path);
            
            // Execute the up migration
            if (typeof migrationModule.up === 'function') {
                await migrationModule.up(this.db);
            } else {
                throw new Error(`Migration ${migration.version} missing 'up' function`);
            }
            
            const executionTime = Date.now() - startTime;
            
            // Record the migration as applied
            await this.db.run(
                'INSERT INTO schema_migrations (version, execution_time) VALUES (?, ?)',
                [migration.version, executionTime]
            );
            
            log.debug(`✅ Migration ${migration.version} completed in ${executionTime}ms`);
            
        } catch (error) {
            console.error(`❌ Migration ${migration.version} failed:`, error);
            throw error;
        }
    }

    async migrate() {
        log.debug('🏗️  Starting database migration...');
        
        const appliedMigrations = await this.getAppliedMigrations();
        const allMigrations = await this.getAllMigrations();
        
        const pendingMigrations = allMigrations.filter(
            migration => !appliedMigrations.includes(migration.version)
        );
        
        if (pendingMigrations.length === 0) {
            log.debug('✨ Database is up to date');
            return;
        }
        
        log.debug(`📊 Found ${pendingMigrations.length} pending migration(s)`);
        
        for (const migration of pendingMigrations) {
            await this.runMigration(migration);
        }
        
        log.debug('🎉 All migrations completed successfully');
    }

    async rollback(targetVersion) {
        log.debug(`🔄 Rolling back to version: ${targetVersion}`);
        
        const appliedMigrations = await this.getAppliedMigrations();
        const migrationsToRollback = appliedMigrations
            .filter(version => version > targetVersion)
            .reverse(); // Rollback in reverse order
        
        for (const version of migrationsToRollback) {
            const migrationPath = path.join(this.migrationsDir, `${version}.js`);
            
            try {
                const migrationModule = require(migrationPath);
                
                if (typeof migrationModule.down === 'function') {
                    log.debug(`🔄 Rolling back: ${version}`);
                    await migrationModule.down(this.db);
                    
                    // Remove from migrations table
                    await this.db.run(
                        'DELETE FROM schema_migrations WHERE version = ?',
                        [version]
                    );
                    
                    log.debug(`✅ Rolled back: ${version}`);
                } else {
                    console.warn(`⚠️  Migration ${version} has no 'down' function`);
                }
            } catch (error) {
                console.error(`❌ Rollback failed for ${version}:`, error);
                throw error;
            }
        }
        
        log.debug('🎉 Rollback completed');
    }

    async status() {
        log.debug('📊 Migration Status:');
        
        const appliedMigrations = await this.getAppliedMigrations();
        const allMigrations = await this.getAllMigrations();
        
        if (allMigrations.length === 0) {
            log.debug('  No migrations found');
            return;
        }
        
        for (const migration of allMigrations) {
            const isApplied = appliedMigrations.includes(migration.version);
            const status = isApplied ? '✅ Applied' : '⏳ Pending';
            log.debug(`  ${migration.version}: ${status}`);
        }
        
        const pending = allMigrations.length - appliedMigrations.length;
        log.debug(`\n📈 Summary: ${appliedMigrations.length} applied, ${pending} pending`);
    }

    async disconnect() {
        if (this.db) {
            await this.db.disconnect();
            log.debug('🔌 Database disconnected');
        }
    }
}

// CLI Interface
async function main() {
    const migrator = new DatabaseMigrator();
    
    try {
        await migrator.initialize();
        
        const command = process.argv[2];
        const argument = process.argv[3];
        
        switch (command) {
            case 'up':
            case 'migrate':
                await migrator.migrate();
                break;
                
            case 'down':
            case 'rollback':
                if (!argument) {
                    console.error('❌ Please specify target version for rollback');
                    process.exit(1);
                }
                await migrator.rollback(argument);
                break;
                
            case 'status':
                await migrator.status();
                break;
                
            default:
                log.debug(`
A2A Network Database Migrator

Usage:
  node migrate.js migrate    - Run all pending migrations
  node migrate.js rollback <version> - Rollback to specific version
  node migrate.js status     - Show migration status

Examples:
  node migrate.js migrate
  node migrate.js rollback 001_initial_schema
  node migrate.js status
                `);
                break;
        }
        
    } catch (error) {
        console.error('💥 Migration failed:', error);
        process.exit(1);
    } finally {
        await migrator.disconnect();
    }
}

if (require.main === module) {
    main();
}

module.exports = DatabaseMigrator;