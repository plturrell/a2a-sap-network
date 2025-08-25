const cds = require('@sap/cds');
const goalAssignmentIntegration = require('./goal-assignment-integration');

/**
 * Goal Sync Scheduler
 * Periodically syncs goals from A2A orchestrator to CAP database
 */
module.exports = class GoalSyncScheduler {

    constructor() {
        this.LOG = cds.log('goal-sync-scheduler');
        this.syncInterval = null;
        this.SYNC_INTERVAL_MS = process.env.GOAL_SYNC_INTERVAL || 5 * 60 * 1000; // 5 minutes default
    }

    /**
     * Start the scheduled sync
     */
    start() {
        this.LOG.info('Starting goal sync scheduler', { interval: this.SYNC_INTERVAL_MS });

        // Run initial sync
        this.syncGoals();

        // Schedule periodic syncs
        this.syncInterval = setInterval(() => {
            this.syncGoals();
        }, this.SYNC_INTERVAL_MS);

        // Handle graceful shutdown
        process.on('SIGTERM', () => this.stop());
        process.on('SIGINT', () => this.stop());
    }

    /**
     * Stop the scheduled sync
     */
    stop() {
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
            this.LOG.info('Goal sync scheduler stopped');
        }
    }

    /**
     * Perform goal synchronization
     */
    async syncGoals() {
        try {
            this.LOG.info('Starting scheduled goal sync');
            const startTime = Date.now();

            // Sync all agent goals
            const result = await goalAssignmentIntegration.syncAllAgentGoals();

            // Create collaborative goals if they don't exist
            await this._ensureCollaborativeGoals();

            // Log sync results
            const duration = Date.now() - startTime;
            this.LOG.info('Goal sync completed', {
                duration: `${duration}ms`,
                result: result
            });

            // Emit sync event for UI updates
            if (cds.emit) {
                cds.emit('goal:sync:completed', {
                    timestamp: new Date(),
                    duration: duration,
                    result: result
                });
            }

        } catch (error) {
            this.LOG.error('Goal sync failed', { error: error.message, stack: error.stack });

            // Emit error event
            if (cds.emit) {
                cds.emit('goal:sync:error', {
                    timestamp: new Date(),
                    error: error.message
                });
            }
        }
    }

    /**
     * Ensure collaborative goals exist
     */
    async _ensureCollaborativeGoals() {
        try {
            const srv = await cds.connect.to('GoalManagementService');

            // Check if collaborative goals already exist
            const existingGoals = await srv.run(
                cds.ql.SELECT.from('CollaborativeGoals').limit(1)
            );

            if (existingGoals.length === 0) {
                this.LOG.info('Creating initial collaborative goals');
                await goalAssignmentIntegration.createCollaborativeGoals();
            }

        } catch (error) {
            this.LOG.warn('Failed to ensure collaborative goals', { error: error.message });
        }
    }

    /**
     * Get sync status
     */
    getStatus() {
        return {
            running: this.syncInterval !== null,
            interval: this.SYNC_INTERVAL_MS,
            nextSync: this.syncInterval ? new Date(Date.now() + this.SYNC_INTERVAL_MS) : null
        };
    }

    /**
     * Trigger manual sync
     */
    async triggerManualSync() {
        this.LOG.info('Manual goal sync triggered');
        return await this.syncGoals();
    }
}

// Create and export singleton instance
const scheduler = new GoalSyncScheduler();

// Auto-start if not in test environment
if (process.env.NODE_ENV !== 'test') {
    // Wait for CDS to be ready
    cds.on('served', () => {
        scheduler.start();
    });
}

module.exports = scheduler;