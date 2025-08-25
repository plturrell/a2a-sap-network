/**
 * @fileoverview Graceful Shutdown Middleware
 * @since 1.0.0
 * @module graceful-shutdown
 *
 * Implements graceful shutdown patterns for the A2A Network server
 * to prevent data corruption and ensure clean resource cleanup
 */

const cds = require('@sap/cds');
const { Server } = require('socket.io');
const { BlockchainEventServer, BlockchainEventClient } = require('./blockchain-event-adapter');

/**
 * Graceful shutdown manager
 */
class GracefulShutdownManager {
  constructor() {
    this.isShuttingDown = false;
    this.connections = new Set();
    this.servers = new Set();
    this.services = new Set();
    this.shutdownTimeout = 30000; // 30 seconds
    this.log = cds.log('graceful-shutdown');
  }

  /**
   * Register a server for graceful shutdown
   * @param {Object} server - Server instance
   */
  registerServer(server) {
    this.servers.add(server);

    // Track connections
    server.on('blockchain-connection', (connection) => {
      this.connections.add(connection);
      connection.on('close', () => {
        this.connections.delete(connection);
      });
    });
  }

  /**
   * Register a service for graceful shutdown
   * @param {Object} service - Service instance with cleanup method
   */
  registerService(service) {
    this.services.add(service);
  }

  /**
   * Initialize graceful shutdown handlers
   */
  initialize() {
    // Handle termination signals
    process.on('SIGTERM', () => this.shutdown('SIGTERM'));
    process.on('SIGINT', () => this.shutdown('SIGINT'));
    process.on('SIGUSR2', () => this.shutdown('SIGUSR2')); // nodemon restart

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      this.log.error('Uncaught exception:', error);
      this.shutdown('uncaughtException', 1);
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason, promise) => {
      this.log.error('Unhandled promise rejection:', reason);
      this.log.error('Promise:', promise);
      this.shutdown('unhandledRejection', 1);
    });

    this.log.info('Graceful shutdown handlers initialized');
  }

  /**
   * Perform graceful shutdown
   * @param {string} signal - Shutdown signal
   * @param {number} exitCode - Exit code (default: 0)
   */
  async shutdown(signal, exitCode = 0) {
    if (this.isShuttingDown) {
      this.log.warn('Shutdown already in progress, forcing exit');
      process.exit(exitCode);
      return;
    }

    this.isShuttingDown = true;
    this.log.info(`Received ${signal}, starting graceful shutdown...`);

    // Set a timeout to force exit if graceful shutdown takes too long
    const forceExitTimer = setTimeout(() => {
      this.log.error('Graceful shutdown timeout, forcing exit');
      process.exit(1);
    }, this.shutdownTimeout);

    try {
      // Stop accepting new connections
      this.log.info('Stopping servers from accepting new connections');
      for (const server of this.servers) {
        if (server && typeof server.close === 'function') {
          await new Promise((resolve) => {
            server.close(() => {
              this.log.debug('Server closed');
              resolve();
            });
          });
        }
      }

      // Close existing connections gracefully
      this.log.info(`Closing ${this.connections.size} active connections`);
      for (const connection of this.connections) {
        if (connection && !connection.destroyed) {
          connection.end();
          // Force close after a short delay
          setTimeout(() => {
            if (!connection.destroyed) {
              connection.destroy();
            }
          }, 5000);
        }
      }

      // Cleanup services
      this.log.info('Cleaning up services');
      for (const service of this.services) {
        if (service && typeof service.cleanup === 'function') {
          try {
            await service.cleanup();
            this.log.debug('Service cleaned up successfully');
          } catch (error) {
            this.log.error('Error cleaning up service:', error);
          }
        }
      }

      // Close database connections
      this.log.info('Closing database connections');
      try {
        await cds.disconnect();
        this.log.debug('Database connections closed');
      } catch (error) {
        this.log.error('Error closing database connections:', error);
      }

      // Clear the force exit timer
      clearTimeout(forceExitTimer);

      this.log.info('Graceful shutdown completed successfully');
      process.exit(exitCode);

    } catch (error) {
      this.log.error('Error during graceful shutdown:', error);
      clearTimeout(forceExitTimer);
      process.exit(1);
    }
  }

  /**
   * Create a safe restart function
   * @returns {Function} Safe restart function
   */
  createSafeRestart() {
    return async () => {
      if (this.isShuttingDown) {
        throw new Error('System is already shutting down');
      }

      this.log.info('Safe restart initiated');

      // Perform graceful shutdown without exit
      await this.shutdown('restart', 0);
    };
  }

  /**
   * Check if system is shutting down
   * @returns {boolean} True if shutting down
   */
  isShutdownInProgress() {
    return this.isShuttingDown;
  }
}

// Global instance
const shutdownManager = new GracefulShutdownManager();

/**
 * Middleware to check shutdown status
 */
function shutdownMiddleware(req, res, next) {
  if (shutdownManager.isShutdownInProgress()) {
    return res.status(503).json({
      error: 'Service unavailable - server is shutting down',
      code: 'SHUTDOWN_IN_PROGRESS'
    });
  }
  next();
}

/**
 * Initialize graceful shutdown for the application
 * @param {Object} app - Express application
 * @param {Object} server - HTTP server
 * @param {Object} io - Socket.IO server (optional)
 */
function initializeGracefulShutdown(app, server, io) {
  shutdownManager.initialize();

  if (server) {
    shutdownManager.registerServer(server);
  }

  if (io) {
    shutdownManager.registerService({
      cleanup: async () => {
        return new Promise((resolve) => {
          io.close(() => {
            shutdownManager.log.debug('Socket.IO server closed');
            resolve();
          });
        });
      }
    });
  }

  // Add shutdown middleware to all routes
  app.use(shutdownMiddleware);

  shutdownManager.log.info('Graceful shutdown initialized for application');
}

module.exports = {
  GracefulShutdownManager,
  shutdownManager,
  shutdownMiddleware,
  initializeGracefulShutdown
};