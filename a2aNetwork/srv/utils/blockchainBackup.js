/**
 * @fileoverview Blockchain Data Backup and Recovery System
 * @description Comprehensive backup solution for blockchain data and state
 * @module blockchain-backup
 */

const cds = require('@sap/cds');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { EventEmitter } = require('events');

/**
 * Blockchain backup and recovery manager
 */
class BlockchainBackupManager extends EventEmitter {
    constructor(web3Provider, options = {}) {
        super();
        this.web3 = web3Provider;
        this.backupDir = options.backupDir || path.join(process.cwd(), 'blockchain-backups');
        this.encryptionKey = options.encryptionKey || this.generateEncryptionKey();
        this.compressionEnabled = options.compression !== false;
        this.retentionDays = options.retentionDays || 30;
        this.backupSchedule = options.backupSchedule || '0 2 * * *'; // 2 AM daily
        
        this.contracts = new Map();
        this.backupHistory = [];
        this.isBackupRunning = false;
    }

    /**
     * Initialize backup manager
     */
    async initialize() {
        try {
            // Create backup directory
            await fs.mkdir(this.backupDir, { recursive: true });
            
            // Load backup history
            await this.loadBackupHistory();
            
            // Start scheduled backups if in production
            if (process.env.NODE_ENV === 'production') {
                this.scheduleBackups();
            }
            
            cds.log('blockchain-backup').info('Backup manager initialized', {
                backupDir: this.backupDir,
                retentionDays: this.retentionDays
            });
            
        } catch (error) {
            cds.log('blockchain-backup').error('Failed to initialize backup manager:', error);
            throw error;
        }
    }

    /**
     * Register contract for backup
     */
    registerContract(name, address, abi, startBlock = 0) {
        this.contracts.set(name, {
            address,
            abi,
            startBlock,
            lastBackupBlock: startBlock,
            contract: new this.web3.eth.Contract(abi, address)
        });
        
        cds.log('blockchain-backup').info('Contract registered for backup', {
            name,
            address,
            startBlock
        });
    }

    /**
     * Perform full backup of all registered contracts
     */
    async performFullBackup() {
        if (this.isBackupRunning) {
            cds.log('blockchain-backup').warn('Backup already in progress');
            return null;
        }

        this.isBackupRunning = true;
        const backupId = this.generateBackupId();
        
        try {
            const currentBlock = await this.web3.eth.getBlockNumber();
            const backupData = {
                id: backupId,
                timestamp: new Date().toISOString(),
                blockNumber: currentBlock,
                networkId: await this.web3.eth.net.getId(),
                contracts: {},
                metadata: {
                    version: '1.0',
                    format: 'json',
                    compressed: this.compressionEnabled,
                    encrypted: true
                }
            };

            // Backup each registered contract
            for (const [contractName, contractInfo] of this.contracts.entries()) {
                cds.log('blockchain-backup').info('Backing up contract', { contractName });
                
                const contractBackup = await this.backupContract(
                    contractName,
                    contractInfo,
                    currentBlock
                );
                
                backupData.contracts[contractName] = contractBackup;
                
                // Update last backup block
                contractInfo.lastBackupBlock = currentBlock;
            }

            // Save backup to file
            const backupFile = await this.saveBackupToFile(backupData);
            
            // Update backup history
            this.backupHistory.push({
                id: backupId,
                timestamp: backupData.timestamp,
                blockNumber: currentBlock,
                file: backupFile,
                size: await this.getFileSize(backupFile),
                contracts: Object.keys(backupData.contracts)
            });

            // Cleanup old backups
            await this.cleanupOldBackups();

            // Save updated history
            await this.saveBackupHistory();

            this.emit('backupCompleted', {
                id: backupId,
                blockNumber: currentBlock,
                file: backupFile
            });

            cds.log('blockchain-backup').info('Full backup completed', {
                backupId,
                blockNumber: currentBlock,
                file: backupFile
            });

            return backupId;

        } catch (error) {
            cds.log('blockchain-backup').error('Backup failed:', error);
            this.emit('backupFailed', { error: error.message });
            throw error;
        } finally {
            this.isBackupRunning = false;
        }
    }

    /**
     * Backup specific contract data
     */
    async backupContract(contractName, contractInfo, toBlock) {
        const { contract, address, lastBackupBlock } = contractInfo;
        
        const contractBackup = {
            address,
            fromBlock: lastBackupBlock,
            toBlock,
            events: {},
            state: {},
            transactions: []
        };

        try {
            // Get all events from last backup to current block
            const allEvents = await contract.getPastEvents('allEvents', {
                fromBlock: lastBackupBlock,
                toBlock: toBlock
            });

            // Organize events by type
            for (const event of allEvents) {
                const eventType = event.event;
                if (!contractBackup.events[eventType]) {
                    contractBackup.events[eventType] = [];
                }
                
                contractBackup.events[eventType].push({
                    blockNumber: event.blockNumber,
                    transactionHash: event.transactionHash,
                    logIndex: event.logIndex,
                    returnValues: event.returnValues,
                    signature: event.signature
                });
            }

            // Backup current state (public variables)
            contractBackup.state = await this.backupContractState(contract);

            // Get contract transactions
            contractBackup.transactions = await this.getContractTransactions(
                address,
                lastBackupBlock,
                toBlock
            );

            return contractBackup;

        } catch (error) {
            cds.log('blockchain-backup').error('Contract backup failed', {
                contractName,
                error: error.message
            });
            throw error;
        }
    }

    /**
     * Backup contract state (public variables)
     */
    async backupContractState(contract) {
        const state = {};
        
        try {
            // This would read all public state variables
            // Implementation depends on specific contract ABI
            
            // Example for common variables
            const commonMethods = [
                'owner',
                'totalSupply',
                'name',
                'symbol',
                'decimals'
            ];

            for (const method of commonMethods) {
                try {
                    if (contract.methods[method]) {
                        state[method] = await contract.methods[method]().call();
                    }
                } catch (error) {
                    // Method doesn't exist or call failed
                    continue;
                }
            }

        } catch (error) {
            cds.log('blockchain-backup').warn('State backup partial failure:', error);
        }

        return state;
    }

    /**
     * Get all transactions for a contract
     */
    async getContractTransactions(address, fromBlock, toBlock) {
        const transactions = [];
        
        try {
            // Get all blocks in range and filter transactions
            for (let blockNum = fromBlock; blockNum <= toBlock; blockNum++) {
                const block = await this.web3.eth.getBlock(blockNum, true);
                
                if (block && block.transactions) {
                    for (const tx of block.transactions) {
                        if (tx.to && tx.to.toLowerCase() === address.toLowerCase()) {
                            transactions.push({
                                hash: tx.hash,
                                blockNumber: blockNum,
                                from: tx.from,
                                to: tx.to,
                                value: tx.value,
                                gasUsed: tx.gas,
                                gasPrice: tx.gasPrice,
                                input: tx.input,
                                timestamp: block.timestamp
                            });
                        }
                    }
                }
            }

        } catch (error) {
            cds.log('blockchain-backup').warn('Transaction backup failed:', error);
        }

        return transactions;
    }

    /**
     * Save backup data to encrypted file
     */
    async saveBackupToFile(backupData) {
        const filename = `backup-${backupData.id}.json`;
        const filepath = path.join(this.backupDir, filename);
        
        try {
            let data = JSON.stringify(backupData, null, 2);
            
            // Compress if enabled
            if (this.compressionEnabled) {
                const zlib = require('zlib');
                data = zlib.gzipSync(data);
            }
            
            // Encrypt data
            const encryptedData = this.encryptData(data);
            
            await fs.writeFile(filepath, encryptedData);
            
            return filepath;

        } catch (error) {
            cds.log('blockchain-backup').error('Failed to save backup file:', error);
            throw error;
        }
    }

    /**
     * Load and decrypt backup file
     */
    async loadBackupFromFile(filepath) {
        try {
            const encryptedData = await fs.readFile(filepath);
            
            // Decrypt data
            let data = this.decryptData(encryptedData);
            
            // Decompress if needed
            if (this.compressionEnabled) {
                const zlib = require('zlib');
                data = zlib.gunzipSync(data);
            }
            
            return JSON.parse(data.toString());

        } catch (error) {
            cds.log('blockchain-backup').error('Failed to load backup file:', error);
            throw error;
        }
    }

    /**
     * Restore from backup
     */
    async restoreFromBackup(backupId) {
        try {
            const backupInfo = this.backupHistory.find(b => b.id === backupId);
            if (!backupInfo) {
                throw new Error(`Backup not found: ${backupId}`);
            }

            const backupData = await this.loadBackupFromFile(backupInfo.file);
            
            cds.log('blockchain-backup').info('Starting restore process', {
                backupId,
                blockNumber: backupData.blockNumber
            });

            // Verify backup integrity
            await this.verifyBackupIntegrity(backupData);

            // Restore contract states (this would depend on specific requirements)
            for (const [contractName, contractBackup] of Object.entries(backupData.contracts)) {
                await this.restoreContractData(contractName, contractBackup);
            }

            this.emit('restoreCompleted', {
                backupId,
                blockNumber: backupData.blockNumber
            });

            cds.log('blockchain-backup').info('Restore completed', { backupId });

        } catch (error) {
            cds.log('blockchain-backup').error('Restore failed:', error);
            this.emit('restoreFailed', { backupId, error: error.message });
            throw error;
        }
    }

    /**
     * Verify backup integrity
     */
    async verifyBackupIntegrity(backupData) {
        // Verify network ID matches
        const currentNetworkId = await this.web3.eth.net.getId();
        if (backupData.networkId !== currentNetworkId) {
            throw new Error('Network ID mismatch');
        }

        // Verify all required contracts are present
        for (const contractName of this.contracts.keys()) {
            if (!backupData.contracts[contractName]) {
                throw new Error(`Missing contract data: ${contractName}`);
            }
        }

        // Additional integrity checks
        for (const [contractName, contractData] of Object.entries(backupData.contracts)) {
            if (!contractData.address || !contractData.events) {
                throw new Error(`Invalid contract data: ${contractName}`);
            }
        }
    }

    /**
     * Restore contract data (implementation depends on requirements)
     */
    async restoreContractData(contractName, contractBackup) {
        // This would implement the actual restoration logic
        // For most cases, this might involve:
        // 1. Validating current on-chain state
        // 2. Identifying differences
        // 3. Recreating missing data in a local database
        // 4. Generating reports on state differences
        
        cds.log('blockchain-backup').info('Restoring contract data', {
            contractName,
            events: Object.keys(contractBackup.events).length,
            transactions: contractBackup.transactions.length
        });
    }

    /**
     * Generate unique backup ID
     */
    generateBackupId() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const random = crypto.randomBytes(4).toString('hex');
        return `${timestamp}-${random}`;
    }

    /**
     * Generate encryption key
     */
    generateEncryptionKey() {
        if (process.env.BACKUP_ENCRYPTION_KEY) {
            return Buffer.from(process.env.BACKUP_ENCRYPTION_KEY, 'hex');
        }
        
        if (process.env.NODE_ENV === 'production') {
            throw new Error('BACKUP_ENCRYPTION_KEY environment variable required in production');
        }
        
        // Development only
        cds.log('blockchain-backup').warn('Using temporary encryption key for development');
        return crypto.randomBytes(32);
    }

    /**
     * Encrypt data
     */
    encryptData(data) {
        const iv = crypto.randomBytes(16);
        const cipher = crypto.createCipherGCM('aes-256-gcm', this.encryptionKey, iv);
        
        let encrypted = cipher.update(data, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        
        const authTag = cipher.getAuthTag();
        
        return JSON.stringify({
            iv: iv.toString('hex'),
            authTag: authTag.toString('hex'),
            data: encrypted
        });
    }

    /**
     * Decrypt data
     */
    decryptData(encryptedData) {
        const parsed = JSON.parse(encryptedData);
        const iv = Buffer.from(parsed.iv, 'hex');
        const authTag = Buffer.from(parsed.authTag, 'hex');
        
        const decipher = crypto.createDecipherGCM('aes-256-gcm', this.encryptionKey, iv);
        decipher.setAuthTag(authTag);
        
        let decrypted = decipher.update(parsed.data, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        
        return Buffer.from(decrypted, 'utf8');
    }

    /**
     * Get file size
     */
    async getFileSize(filepath) {
        try {
            const stats = await fs.stat(filepath);
            return stats.size;
        } catch {
            return 0;
        }
    }

    /**
     * Load backup history
     */
    async loadBackupHistory() {
        const historyFile = path.join(this.backupDir, 'backup-history.json');
        
        try {
            const data = await fs.readFile(historyFile, 'utf8');
            this.backupHistory = JSON.parse(data);
        } catch (error) {
            // No history file exists yet
            this.backupHistory = [];
        }
    }

    /**
     * Save backup history
     */
    async saveBackupHistory() {
        const historyFile = path.join(this.backupDir, 'backup-history.json');
        
        try {
            await fs.writeFile(
                historyFile,
                JSON.stringify(this.backupHistory, null, 2)
            );
        } catch (error) {
            cds.log('blockchain-backup').error('Failed to save backup history:', error);
        }
    }

    /**
     * Cleanup old backups based on retention policy
     */
    async cleanupOldBackups() {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - this.retentionDays);
        
        const backupsToDelete = this.backupHistory.filter(
            backup => new Date(backup.timestamp) < cutoffDate
        );

        for (const backup of backupsToDelete) {
            try {
                await fs.unlink(backup.file);
                
                // Remove from history
                const index = this.backupHistory.indexOf(backup);
                if (index > -1) {
                    this.backupHistory.splice(index, 1);
                }
                
                cds.log('blockchain-backup').info('Old backup deleted', {
                    id: backup.id,
                    file: backup.file
                });
                
            } catch (error) {
                cds.log('blockchain-backup').warn('Failed to delete old backup:', error);
            }
        }
    }

    /**
     * Schedule automatic backups
     */
    scheduleBackups() {
        const cron = require('node-cron');
        
        cron.schedule(this.backupSchedule, async () => {
            try {
                await this.performFullBackup();
            } catch (error) {
                cds.log('blockchain-backup').error('Scheduled backup failed:', error);
            }
        });
        
        cds.log('blockchain-backup').info('Backup schedule configured', {
            schedule: this.backupSchedule
        });
    }

    /**
     * Get backup statistics
     */
    getBackupStats() {
        const totalSize = this.backupHistory.reduce((sum, backup) => sum + backup.size, 0);
        const avgSize = this.backupHistory.length > 0 ? totalSize / this.backupHistory.length : 0;
        
        return {
            totalBackups: this.backupHistory.length,
            totalSize: totalSize,
            averageSize: avgSize,
            oldestBackup: this.backupHistory.length > 0 ? 
                Math.min(...this.backupHistory.map(b => new Date(b.timestamp))) : null,
            newestBackup: this.backupHistory.length > 0 ? 
                Math.max(...this.backupHistory.map(b => new Date(b.timestamp))) : null,
            registeredContracts: this.contracts.size
        };
    }

    /**
     * List available backups
     */
    listBackups() {
        return this.backupHistory.map(backup => ({
            id: backup.id,
            timestamp: backup.timestamp,
            blockNumber: backup.blockNumber,
            size: backup.size,
            contracts: backup.contracts
        }));
    }
}

module.exports = {
    BlockchainBackupManager
};