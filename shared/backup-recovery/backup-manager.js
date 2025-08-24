/**
 * Comprehensive Backup and Disaster Recovery Manager for A2A Platform
 * Handles automated backups, point-in-time recovery, and disaster recovery orchestration
 */

const fs = require('fs').promises;
const path = require('path');
const { spawn, exec } = require('child_process');
const { promisify } = require('util');
const crypto = require('crypto');
const EventEmitter = require('events');

const execAsync = promisify(exec);

class A2ABackupManager extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Backup settings
            backupDir: options.backupDir || '/backups',
            retentionDays: options.retentionDays || 30,
            compressionLevel: options.compressionLevel || 6,
            encryptionEnabled: options.encryptionEnabled || true,
            encryptionKey: options.encryptionKey || process.env.BACKUP_ENCRYPTION_KEY,
            
            // Backup types
            enableDatabaseBackup: options.enableDatabaseBackup !== false,
            enableFileBackup: options.enableFileBackup !== false,
            enableConfigBackup: options.enableConfigBackup !== false,
            enableRedisBackup: options.enableRedisBackup !== false,
            
            // Schedule settings
            fullBackupCron: options.fullBackupCron || '0 2 * * 0', // Sunday 2 AM
            incrementalBackupCron: options.incrementalBackupCron || '0 2 * * 1-6', // Mon-Sat 2 AM
            
            // Database settings
            databases: options.databases || [
                {
                    type: 'postgresql',
                    host: process.env.DB_HOST || 'localhost',
                    port: process.env.DB_PORT || 5432,
                    database: process.env.DB_NAME || 'a2a',
                    username: process.env.DB_USER || 'postgres',
                    password: process.env.DB_PASSWORD,
                    connectionString: process.env.DATABASE_URL
                }
            ],
            
            // Redis settings
            redis: {
                host: options.redis?.host || process.env.REDIS_HOST || 'localhost',
                port: options.redis?.port || process.env.REDIS_PORT || 6379,
                password: options.redis?.password || process.env.REDIS_PASSWORD,
                databases: options.redis?.databases || [0, 1, 2]
            },
            
            // File backup paths
            filePaths: options.filePaths || [
                '/app/data',
                '/app/uploads',
                '/app/logs',
                '/etc/a2a'
            ],
            
            // Cloud storage settings
            cloudStorage: {
                enabled: options.cloudStorage?.enabled || false,
                provider: options.cloudStorage?.provider || 's3', // s3, gcs, azure
                bucket: options.cloudStorage?.bucket,
                credentials: options.cloudStorage?.credentials,
                region: options.cloudStorage?.region
            },
            
            // Monitoring
            healthCheckInterval: options.healthCheckInterval || 300000, // 5 minutes
            
            ...options
        };
        
        this.backupHistory = new Map(); // backup_id -> backup_info
        this.runningBackups = new Map(); // backup_id -> backup_process
        this.backupStats = {
            totalBackups: 0,
            successfulBackups: 0,
            failedBackups: 0,
            totalDataBackedUp: 0,
            lastFullBackup: null,
            lastIncrementalBackup: null
        };
        
        this.isInitialized = false;
        this.healthCheckInterval = null;
    }
    
    async initialize() {
        if (this.isInitialized) {
            return this;
        }
        
        try {
            // Ensure backup directory exists
            await fs.mkdir(this.config.backupDir, { recursive: true });
            
            // Validate encryption key
            if (this.config.encryptionEnabled && !this.config.encryptionKey) {
                throw new Error('Encryption is enabled but no encryption key provided');
            }
            
            // Test database connections
            await this.testDatabaseConnections();
            
            // Test Redis connection
            if (this.config.enableRedisBackup) {
                await this.testRedisConnection();
            }
            
            // Initialize cloud storage
            if (this.config.cloudStorage.enabled) {
                await this.initializeCloudStorage();
            }
            
            // Load backup history
            await this.loadBackupHistory();
            
            // Start health monitoring
            this.startHealthMonitoring();
            
            this.isInitialized = true;
            this.emit('backup:manager:initialized');
            
            console.log('A2A Backup Manager initialized successfully');
            return this;
            
        } catch (error) {
            console.error('Failed to initialize Backup Manager:', error);
            throw error;
        }
    }
    
    // Main backup methods
    async createFullBackup(options = {}) {
        const backupId = this.generateBackupId('full');
        const backupInfo = {
            id: backupId,
            type: 'full',
            startTime: new Date(),
            status: 'running',
            components: [],
            metadata: {
                version: process.env.A2A_VERSION || '1.0.0',
                environment: process.env.NODE_ENV || 'production',
                initiatedBy: options.initiatedBy || 'automatic',
                ...options.metadata
            }
        };
        
        this.backupHistory.set(backupId, backupInfo);
        this.emit('backup:started', backupInfo);
        
        try {
            const backupDir = path.join(this.config.backupDir, backupId);
            await fs.mkdir(backupDir, { recursive: true });
            
            // Backup databases
            if (this.config.enableDatabaseBackup) {
                const dbBackup = await this.backupDatabases(backupId, backupDir);
                backupInfo.components.push(dbBackup);
            }
            
            // Backup Redis
            if (this.config.enableRedisBackup) {
                const redisBackup = await this.backupRedis(backupId, backupDir);
                backupInfo.components.push(redisBackup);
            }
            
            // Backup files
            if (this.config.enableFileBackup) {
                const fileBackup = await this.backupFiles(backupId, backupDir);
                backupInfo.components.push(fileBackup);
            }
            
            // Backup configurations
            if (this.config.enableConfigBackup) {
                const configBackup = await this.backupConfigurations(backupId, backupDir);
                backupInfo.components.push(configBackup);
            }
            
            // Create backup manifest
            await this.createBackupManifest(backupId, backupDir, backupInfo);
            
            // Compress and encrypt backup
            const archivePath = await this.compressAndEncryptBackup(backupId, backupDir);
            
            // Upload to cloud storage
            if (this.config.cloudStorage.enabled) {
                await this.uploadToCloud(backupId, archivePath);
            }
            
            // Update backup info
            backupInfo.endTime = new Date();
            backupInfo.duration = backupInfo.endTime - backupInfo.startTime;
            backupInfo.status = 'completed';
            backupInfo.size = await this.getFileSize(archivePath);
            backupInfo.archivePath = archivePath;
            
            // Update stats
            this.backupStats.totalBackups++;
            this.backupStats.successfulBackups++;
            this.backupStats.totalDataBackedUp += backupInfo.size;
            this.backupStats.lastFullBackup = new Date();
            
            this.emit('backup:completed', backupInfo);
            console.log(`Full backup completed: ${backupId} (${this.formatBytes(backupInfo.size)})`);
            
            return backupInfo;
            
        } catch (error) {
            backupInfo.status = 'failed';
            backupInfo.error = error.message;
            backupInfo.endTime = new Date();
            
            this.backupStats.failedBackups++;
            
            this.emit('backup:failed', { backupInfo, error });
            console.error(`Full backup failed: ${backupId}`, error);
            
            throw error;
        }
    }
    
    async createIncrementalBackup(options = {}) {
        const lastFullBackup = this.findLastSuccessfulBackup('full');
        if (!lastFullBackup) {
            console.log('No full backup found, creating full backup instead');
            return await this.createFullBackup(options);
        }
        
        const backupId = this.generateBackupId('incremental');
        const backupInfo = {
            id: backupId,
            type: 'incremental',
            baseBackupId: lastFullBackup.id,
            startTime: new Date(),
            status: 'running',
            components: [],
            metadata: {
                version: process.env.A2A_VERSION || '1.0.0',
                environment: process.env.NODE_ENV || 'production',
                initiatedBy: options.initiatedBy || 'automatic',
                ...options.metadata
            }
        };
        
        this.backupHistory.set(backupId, backupInfo);
        this.emit('backup:started', backupInfo);
        
        try {
            const backupDir = path.join(this.config.backupDir, backupId);
            await fs.mkdir(backupDir, { recursive: true });
            
            const sinceTime = lastFullBackup.endTime;
            
            // Incremental database backup
            if (this.config.enableDatabaseBackup) {
                const dbBackup = await this.backupDatabasesIncremental(backupId, backupDir, sinceTime);
                backupInfo.components.push(dbBackup);
            }
            
            // Incremental file backup
            if (this.config.enableFileBackup) {
                const fileBackup = await this.backupFilesIncremental(backupId, backupDir, sinceTime);
                backupInfo.components.push(fileBackup);
            }
            
            // Create backup manifest
            await this.createBackupManifest(backupId, backupDir, backupInfo);
            
            // Compress and encrypt backup
            const archivePath = await this.compressAndEncryptBackup(backupId, backupDir);
            
            // Upload to cloud storage
            if (this.config.cloudStorage.enabled) {
                await this.uploadToCloud(backupId, archivePath);
            }
            
            // Update backup info
            backupInfo.endTime = new Date();
            backupInfo.duration = backupInfo.endTime - backupInfo.startTime;
            backupInfo.status = 'completed';
            backupInfo.size = await this.getFileSize(archivePath);
            backupInfo.archivePath = archivePath;
            
            // Update stats
            this.backupStats.totalBackups++;
            this.backupStats.successfulBackups++;
            this.backupStats.totalDataBackedUp += backupInfo.size;
            this.backupStats.lastIncrementalBackup = new Date();
            
            this.emit('backup:completed', backupInfo);
            console.log(`Incremental backup completed: ${backupId} (${this.formatBytes(backupInfo.size)})`);
            
            return backupInfo;
            
        } catch (error) {
            backupInfo.status = 'failed';
            backupInfo.error = error.message;
            backupInfo.endTime = new Date();
            
            this.backupStats.failedBackups++;
            
            this.emit('backup:failed', { backupInfo, error });
            console.error(`Incremental backup failed: ${backupId}`, error);
            
            throw error;
        }
    }
    
    // Database backup methods
    async backupDatabases(backupId, backupDir) {
        const dbBackupInfo = {
            type: 'database',
            startTime: new Date(),
            databases: []
        };
        
        for (const dbConfig of this.config.databases) {
            try {
                let backupFile;
                
                switch (dbConfig.type) {
                    case 'postgresql':
                        backupFile = await this.backupPostgreSQL(dbConfig, backupId, backupDir);
                        break;
                    case 'mysql':
                        backupFile = await this.backupMySQL(dbConfig, backupId, backupDir);
                        break;
                    case 'mongodb':
                        backupFile = await this.backupMongoDB(dbConfig, backupId, backupDir);
                        break;
                    default:
                        throw new Error(`Unsupported database type: ${dbConfig.type}`);
                }
                
                dbBackupInfo.databases.push({
                    name: dbConfig.database,
                    type: dbConfig.type,
                    file: backupFile,
                    size: await this.getFileSize(backupFile),
                    status: 'success'
                });
                
            } catch (error) {
                console.error(`Database backup failed for ${dbConfig.database}:`, error);
                dbBackupInfo.databases.push({
                    name: dbConfig.database,
                    type: dbConfig.type,
                    status: 'failed',
                    error: error.message
                });
            }
        }
        
        dbBackupInfo.endTime = new Date();
        dbBackupInfo.duration = dbBackupInfo.endTime - dbBackupInfo.startTime;
        
        return dbBackupInfo;
    }
    
    async backupPostgreSQL(dbConfig, backupId, backupDir) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupFile = path.join(backupDir, `postgres_${dbConfig.database}_${timestamp}.sql`);
        
        const env = {
            ...process.env,
            PGPASSWORD: dbConfig.password
        };
        
        const command = [
            'pg_dump',
            '-h', dbConfig.host,
            '-p', dbConfig.port.toString(),
            '-U', dbConfig.username,
            '-d', dbConfig.database,
            '--verbose',
            '--no-owner',
            '--no-privileges',
            '-f', backupFile
        ];
        
        await this.executeCommand('pg_dump', command, { env });
        
        console.log(`PostgreSQL backup completed: ${backupFile}`);
        return backupFile;
    }
    
    async backupMySQL(dbConfig, backupId, backupDir) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupFile = path.join(backupDir, `mysql_${dbConfig.database}_${timestamp}.sql`);
        
        const command = [
            'mysqldump',
            '-h', dbConfig.host,
            '-P', dbConfig.port.toString(),
            '-u', dbConfig.username,
            `-p${dbConfig.password}`,
            '--single-transaction',
            '--routines',
            '--triggers',
            dbConfig.database
        ];
        
        const { stdout } = await this.executeCommand('mysqldump', command);
        await fs.writeFile(backupFile, stdout);
        
        console.log(`MySQL backup completed: ${backupFile}`);
        return backupFile;
    }
    
    async backupMongoDB(dbConfig, backupId, backupDir) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const backupDir2 = path.join(backupDir, `mongodb_${dbConfig.database}_${timestamp}`);
        
        await fs.mkdir(backupDir2, { recursive: true });
        
        const command = [
            'mongodump',
            '--host', `${dbConfig.host}:${dbConfig.port}`,
            '--db', dbConfig.database,
            '--out', backupDir2
        ];
        
        if (dbConfig.username) {
            command.push('--username', dbConfig.username);
            command.push('--password', dbConfig.password);
        }
        
        await this.executeCommand('mongodump', command);
        
        console.log(`MongoDB backup completed: ${backupDir2}`);
        return backupDir2;
    }
    
    // Redis backup
    async backupRedis(backupId, backupDir) {
        const redisBackupInfo = {
            type: 'redis',
            startTime: new Date(),
            databases: []
        };
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        
        for (const dbIndex of this.config.redis.databases) {
            try {
                const backupFile = path.join(backupDir, `redis_db${dbIndex}_${timestamp}.rdb`);
                
                // Use redis-cli to create backup
                const command = [
                    'redis-cli',
                    '-h', this.config.redis.host,
                    '-p', this.config.redis.port.toString(),
                    '-n', dbIndex.toString()
                ];
                
                if (this.config.redis.password) {
                    command.push('-a', this.config.redis.password);
                }
                
                command.push('--rdb', backupFile);
                
                await this.executeCommand('redis-cli', command);
                
                redisBackupInfo.databases.push({
                    index: dbIndex,
                    file: backupFile,
                    size: await this.getFileSize(backupFile),
                    status: 'success'
                });
                
            } catch (error) {
                console.error(`Redis backup failed for database ${dbIndex}:`, error);
                redisBackupInfo.databases.push({
                    index: dbIndex,
                    status: 'failed',
                    error: error.message
                });
            }
        }
        
        redisBackupInfo.endTime = new Date();
        redisBackupInfo.duration = redisBackupInfo.endTime - redisBackupInfo.startTime;
        
        return redisBackupInfo;
    }
    
    // File backup
    async backupFiles(backupId, backupDir) {
        const fileBackupInfo = {
            type: 'files',
            startTime: new Date(),
            paths: []
        };
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        
        for (const filePath of this.config.filePaths) {
            try {
                const pathExists = await this.pathExists(filePath);
                if (!pathExists) {
                    console.warn(`Backup path does not exist: ${filePath}`);
                    continue;
                }
                
                const pathName = path.basename(filePath) || 'root';
                const backupFile = path.join(backupDir, `files_${pathName}_${timestamp}.tar.gz`);
                
                // Create compressed archive
                const command = [
                    'tar',
                    '-czf', backupFile,
                    '-C', path.dirname(filePath),
                    path.basename(filePath)
                ];
                
                await this.executeCommand('tar', command);
                
                fileBackupInfo.paths.push({
                    path: filePath,
                    file: backupFile,
                    size: await this.getFileSize(backupFile),
                    status: 'success'
                });
                
            } catch (error) {
                console.error(`File backup failed for ${filePath}:`, error);
                fileBackupInfo.paths.push({
                    path: filePath,
                    status: 'failed',
                    error: error.message
                });
            }
        }
        
        fileBackupInfo.endTime = new Date();
        fileBackupInfo.duration = fileBackupInfo.endTime - fileBackupInfo.startTime;
        
        return fileBackupInfo;
    }
    
    // Configuration backup
    async backupConfigurations(backupId, backupDir) {
        const configBackupInfo = {
            type: 'configuration',
            startTime: new Date(),
            configs: []
        };
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const configDir = path.join(backupDir, 'configurations');
        await fs.mkdir(configDir, { recursive: true });
        
        // Backup environment variables
        const envFile = path.join(configDir, `environment_${timestamp}.json`);
        const envData = {
            NODE_ENV: process.env.NODE_ENV,
            A2A_VERSION: process.env.A2A_VERSION,
            // Add other important environment variables (excluding secrets)
            timestamp: new Date().toISOString()
        };
        
        await fs.writeFile(envFile, JSON.stringify(envData, null, 2));
        
        configBackupInfo.configs.push({
            type: 'environment',
            file: envFile,
            size: await this.getFileSize(envFile),
            status: 'success'
        });
        
        // Backup Docker configurations
        try {
            const dockerFiles = [
                'docker-compose.yml',
                'docker-compose.prod.yml',
                'Dockerfile'
            ];
            
            for (const dockerFile of dockerFiles) {
                const sourcePath = path.join(process.cwd(), dockerFile);
                const destPath = path.join(configDir, dockerFile);
                
                if (await this.pathExists(sourcePath)) {
                    await fs.copyFile(sourcePath, destPath);
                    
                    configBackupInfo.configs.push({
                        type: 'docker',
                        name: dockerFile,
                        file: destPath,
                        size: await this.getFileSize(destPath),
                        status: 'success'
                    });
                }
            }
        } catch (error) {
            console.error('Docker configuration backup failed:', error);
            configBackupInfo.configs.push({
                type: 'docker',
                status: 'failed',
                error: error.message
            });
        }
        
        configBackupInfo.endTime = new Date();
        configBackupInfo.duration = configBackupInfo.endTime - configBackupInfo.startTime;
        
        return configBackupInfo;
    }
    
    // Backup compression and encryption
    async compressAndEncryptBackup(backupId, backupDir) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        let archivePath = path.join(this.config.backupDir, `${backupId}_${timestamp}.tar.gz`);
        
        // Create compressed archive
        const command = [
            'tar',
            '-czf', archivePath,
            '-C', this.config.backupDir,
            backupId
        ];
        
        await this.executeCommand('tar', command);
        
        // Encrypt if enabled
        if (this.config.encryptionEnabled) {
            const encryptedPath = `${archivePath}.enc`;
            await this.encryptFile(archivePath, encryptedPath);
            
            // Remove unencrypted archive
            await fs.unlink(archivePath);
            archivePath = encryptedPath;
        }
        
        // Remove temporary backup directory
        await this.removeDirectory(backupDir);
        
        return archivePath;
    }
    
    async encryptFile(inputPath, outputPath) {
        const algorithm = 'aes-256-gcm';
        const key = crypto.scryptSync(this.config.encryptionKey, 'salt', 32);
        const iv = crypto.randomBytes(16);
        
        const cipher = crypto.createCipher(algorithm, key);
        
        const input = await fs.readFile(inputPath);
        const encrypted = Buffer.concat([iv, cipher.update(input), cipher.final()]);
        
        await fs.writeFile(outputPath, encrypted);
    }
    
    // Recovery methods
    async listBackups(options = {}) {
        const backups = Array.from(this.backupHistory.values());
        
        if (options.type) {
            return backups.filter(backup => backup.type === options.type);
        }
        
        if (options.since) {
            const sinceDate = new Date(options.since);
            return backups.filter(backup => backup.startTime >= sinceDate);
        }
        
        return backups.sort((a, b) => b.startTime - a.startTime);
    }
    
    async restoreBackup(backupId, options = {}) {
        const backupInfo = this.backupHistory.get(backupId);
        if (!backupInfo) {
            throw new Error(`Backup not found: ${backupId}`);
        }
        
        if (backupInfo.status !== 'completed') {
            throw new Error(`Cannot restore incomplete backup: ${backupId}`);
        }
        
        const restoreId = this.generateRestoreId();
        const restoreInfo = {
            id: restoreId,
            backupId,
            startTime: new Date(),
            status: 'running',
            options,
            components: []
        };
        
        this.emit('restore:started', restoreInfo);
        
        try {
            // Extract and decrypt backup
            const extractDir = await this.extractBackup(backupInfo.archivePath, restoreId);
            
            // Restore components based on options
            if (options.restoreDatabases !== false) {
                const dbRestore = await this.restoreDatabases(extractDir, options);
                restoreInfo.components.push(dbRestore);
            }
            
            if (options.restoreRedis !== false) {
                const redisRestore = await this.restoreRedis(extractDir, options);
                restoreInfo.components.push(redisRestore);
            }
            
            if (options.restoreFiles !== false) {
                const fileRestore = await this.restoreFiles(extractDir, options);
                restoreInfo.components.push(fileRestore);
            }
            
            // Cleanup
            await this.removeDirectory(extractDir);
            
            restoreInfo.endTime = new Date();
            restoreInfo.duration = restoreInfo.endTime - restoreInfo.startTime;
            restoreInfo.status = 'completed';
            
            this.emit('restore:completed', restoreInfo);
            console.log(`Restore completed: ${restoreId}`);
            
            return restoreInfo;
            
        } catch (error) {
            restoreInfo.status = 'failed';
            restoreInfo.error = error.message;
            restoreInfo.endTime = new Date();
            
            this.emit('restore:failed', { restoreInfo, error });
            console.error(`Restore failed: ${restoreId}`, error);
            
            throw error;
        }
    }
    
    // Utility methods
    generateBackupId(type) {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const random = crypto.randomBytes(4).toString('hex');
        return `${type}_${timestamp}_${random}`;
    }
    
    generateRestoreId() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const random = crypto.randomBytes(4).toString('hex');
        return `restore_${timestamp}_${random}`;
    }
    
    async executeCommand(command, args, options = {}) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                ...options
            });
            
            let stdout = '';
            let stderr = '';
            
            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            
            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ stdout, stderr });
                } else {
                    reject(new Error(`Command failed with code ${code}: ${stderr}`));
                }
            });
            
            child.on('error', reject);
        });
    }
    
    async pathExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }
    
    async getFileSize(filePath) {
        try {
            const stats = await fs.stat(filePath);
            return stats.size;
        } catch {
            return 0;
        }
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))  } ${  sizes[i]}`;
    }
    
    findLastSuccessfulBackup(type) {
        const backups = Array.from(this.backupHistory.values())
            .filter(backup => backup.type === type && backup.status === 'completed')
            .sort((a, b) => b.endTime - a.endTime);
        
        return backups[0] || null;
    }
    
    // Health monitoring
    startHealthMonitoring() {
        this.healthCheckInterval = setInterval(async () => {
            await this.performHealthCheck();
        }, this.config.healthCheckInterval);
    }
    
    async performHealthCheck() {
        const health = {
            timestamp: new Date(),
            status: 'healthy',
            components: {}
        };
        
        try {
            // Check backup directory
            const backupDirStats = await fs.stat(this.config.backupDir);
            health.components.backupDirectory = {
                status: 'healthy',
                path: this.config.backupDir,
                exists: true,
                writable: true
            };
        } catch (error) {
            health.components.backupDirectory = {
                status: 'unhealthy',
                error: error.message
            };
            health.status = 'degraded';
        }
        
        // Check database connections
        try {
            await this.testDatabaseConnections();
            health.components.databases = { status: 'healthy' };
        } catch (error) {
            health.components.databases = {
                status: 'unhealthy',
                error: error.message
            };
            health.status = 'degraded';
        }
        
        this.emit('health:check', health);
        
        if (health.status !== 'healthy') {
            this.emit('health:warning', health);
        }
    }
    
    async testDatabaseConnections() {
        for (const dbConfig of this.config.databases) {
            if (dbConfig.type === 'postgresql') {
                const { stdout } = await this.executeCommand('pg_isready', [
                    '-h', dbConfig.host,
                    '-p', dbConfig.port.toString(),
                    '-U', dbConfig.username,
                    '-d', dbConfig.database
                ]);
                
                if (!stdout.includes('accepting connections')) {
                    throw new Error(`PostgreSQL connection failed: ${dbConfig.database}`);
                }
            }
        }
    }
    
    async testRedisConnection() {
        const command = [
            'redis-cli',
            '-h', this.config.redis.host,
            '-p', this.config.redis.port.toString(),
            'ping'
        ];
        
        if (this.config.redis.password) {
            command.push('-a', this.config.redis.password);
        }
        
        const { stdout } = await this.executeCommand('redis-cli', command);
        
        if (!stdout.includes('PONG')) {
            throw new Error('Redis connection failed');
        }
    }
    
    // Cleanup methods
    async cleanupOldBackups() {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - this.config.retentionDays);
        
        let deletedCount = 0;
        let freedSpace = 0;
        
        for (const [backupId, backupInfo] of this.backupHistory) {
            if (backupInfo.endTime && backupInfo.endTime < cutoffDate) {
                try {
                    if (backupInfo.archivePath && await this.pathExists(backupInfo.archivePath)) {
                        const fileSize = await this.getFileSize(backupInfo.archivePath);
                        await fs.unlink(backupInfo.archivePath);
                        freedSpace += fileSize;
                    }
                    
                    this.backupHistory.delete(backupId);
                    deletedCount++;
                    
                    this.emit('backup:deleted', { backupId, backupInfo });
                    
                } catch (error) {
                    console.error(`Failed to delete backup ${backupId}:`, error);
                }
            }
        }
        
        console.log(`Cleanup completed: ${deletedCount} backups deleted, ${this.formatBytes(freedSpace)} freed`);
        return { deletedCount, freedSpace };
    }
    
    async removeDirectory(dirPath) {
        try {
            await fs.rm(dirPath, { recursive: true, force: true });
        } catch (error) {
            console.warn(`Failed to remove directory ${dirPath}:`, error);
        }
    }
    
    // Backup history persistence
    async loadBackupHistory() {
        const historyFile = path.join(this.config.backupDir, 'backup_history.json');
        
        try {
            const data = await fs.readFile(historyFile, 'utf8');
            const history = JSON.parse(data);
            
            for (const backup of history) {
                this.backupHistory.set(backup.id, {
                    ...backup,
                    startTime: new Date(backup.startTime),
                    endTime: backup.endTime ? new Date(backup.endTime) : null
                });
            }
            
            console.log(`Loaded ${history.length} backup records from history`);
            
        } catch (error) {
            console.log('No backup history file found, starting fresh');
        }
    }
    
    async saveBackupHistory() {
        const historyFile = path.join(this.config.backupDir, 'backup_history.json');
        const history = Array.from(this.backupHistory.values());
        
        await fs.writeFile(historyFile, JSON.stringify(history, null, 2));
    }
    
    // Statistics and reporting
    getBackupStats() {
        return {
            ...this.backupStats,
            totalStorageUsed: this.calculateTotalStorageUsed(),
            oldestBackup: this.findOldestBackup(),
            newestBackup: this.findNewestBackup(),
            backupFrequency: this.calculateBackupFrequency()
        };
    }
    
    calculateTotalStorageUsed() {
        let total = 0;
        for (const backup of this.backupHistory.values()) {
            if (backup.size) {
                total += backup.size;
            }
        }
        return total;
    }
    
    findOldestBackup() {
        const backups = Array.from(this.backupHistory.values())
            .filter(b => b.status === 'completed')
            .sort((a, b) => a.startTime - b.startTime);
        
        return backups[0] || null;
    }
    
    findNewestBackup() {
        const backups = Array.from(this.backupHistory.values())
            .filter(b => b.status === 'completed')
            .sort((a, b) => b.startTime - a.startTime);
        
        return backups[0] || null;
    }
    
    calculateBackupFrequency() {
        const backups = Array.from(this.backupHistory.values())
            .filter(b => b.status === 'completed')
            .sort((a, b) => a.startTime - b.startTime);
        
        if (backups.length < 2) return 0;
        
        const oldest = backups[0].startTime;
        const newest = backups[backups.length - 1].startTime;
        const daysSpan = (newest - oldest) / (1000 * 60 * 60 * 24);
        
        return backups.length / Math.max(daysSpan, 1);
    }
    
    // Shutdown
    async shutdown() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        // Save backup history
        await this.saveBackupHistory();
        
        // Cancel running backups
        for (const [backupId, backupProcess] of this.runningBackups) {
            try {
                backupProcess.kill();
                this.emit('backup:cancelled', { backupId });
            } catch (error) {
                console.error(`Failed to cancel backup ${backupId}:`, error);
            }
        }
        
        this.emit('backup:manager:shutdown');
        console.log('A2A Backup Manager shutdown completed');
    }
}

// Factory function
function createBackupManager(options = {}) {
    return new A2ABackupManager(options);
}

module.exports = {
    A2ABackupManager,
    createBackupManager
};