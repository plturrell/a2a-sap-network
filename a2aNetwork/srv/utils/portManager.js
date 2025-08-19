/**
 * @fileoverview Dynamic Port Management for A2A Network Services
 * @description Handles port allocation and conflicts for WebSocket services
 */

const net = require('net');
const cds = require('@sap/cds');

class PortManager {
    constructor() {
        this.log = cds.log('port-manager');
        this.allocatedPorts = new Map();
        this.defaultPorts = {
            notifications: 4005,
            realtime: 4006,
            websocket: 4007,
            blockchain: 8545,
            redis: 6379
        };
    }

    /**
     * Check if a port is available
     */
    async isPortAvailable(port) {
        return new Promise((resolve) => {
            const server = net.createServer();
            
            server.listen(port, () => {
                server.once('close', () => {
                    resolve(true);
                });
                server.close();
            });
            
            server.on('error', () => {
                resolve(false);
            });
        });
    }

    /**
     * Find an available port starting from the preferred port
     */
    async findAvailablePort(serviceName, preferredPort = null, maxAttempts = 10) {
        const startPort = preferredPort || this.defaultPorts[serviceName] || 4005;
        
        for (let i = 0; i < maxAttempts; i++) {
            const port = startPort + i;
            const available = await this.isPortAvailable(port);
            
            if (available) {
                this.allocatedPorts.set(serviceName, port);
                this.log.info(`✅ Allocated port ${port} for service: ${serviceName}`);
                return port;
            }
            
            this.log.debug(`Port ${port} already in use, trying next...`);
        }
        
        throw new Error(`Could not find available port for service: ${serviceName} (tried ${startPort}-${startPort + maxAttempts - 1})`);
    }

    /**
     * Get allocated port for a service
     */
    getPort(serviceName) {
        return this.allocatedPorts.get(serviceName);
    }

    /**
     * Release a port allocation
     */
    releasePort(serviceName) {
        const port = this.allocatedPorts.get(serviceName);
        if (port) {
            this.allocatedPorts.delete(serviceName);
            this.log.info(`Released port ${port} for service: ${serviceName}`);
        }
    }

    /**
     * Kill processes using specific ports (development only)
     */
    async killPortProcesses(ports, force = false) {
        if (process.env.NODE_ENV === 'production' && !force) {
            this.log.warn('Refusing to kill processes in production mode');
            return;
        }

        const { exec } = require('child_process');
        const portsArray = Array.isArray(ports) ? ports : [ports];
        
        for (const port of portsArray) {
            try {
                // Kill processes on macOS/Linux
                await new Promise((resolve) => {
                    exec(`lsof -ti:${port} | xargs kill -9 2>/dev/null || true`, (error) => {
                        if (!error) {
                            this.log.info(`🧹 Killed processes on port ${port}`);
                        }
                        resolve();
                    });
                });
                
                // Wait a moment for cleanup
                await new Promise(resolve => setTimeout(resolve, 100));
                
            } catch (error) {
                this.log.debug(`No processes to kill on port ${port}`);
            }
        }
    }

    /**
     * Smart port allocation with conflict resolution
     */
    async allocatePortSafely(serviceName, preferredPort = null, killConflicts = false) {
        try {
            // First, try the preferred port
            const targetPort = preferredPort || this.defaultPorts[serviceName];
            
            if (targetPort && await this.isPortAvailable(targetPort)) {
                this.allocatedPorts.set(serviceName, targetPort);
                this.log.info(`✅ Allocated preferred port ${targetPort} for service: ${serviceName}`);
                return targetPort;
            }

            // If development mode and killConflicts is enabled, try to kill conflicting processes
            if (killConflicts && process.env.NODE_ENV === 'development') {
                this.log.info(`🔧 Attempting to free port ${targetPort} for ${serviceName}...`);
                await this.killPortProcesses(targetPort);
                
                // Check if port is now available
                if (await this.isPortAvailable(targetPort)) {
                    this.allocatedPorts.set(serviceName, targetPort);
                    this.log.info(`✅ Freed and allocated port ${targetPort} for service: ${serviceName}`);
                    return targetPort;
                }
            }

            // Fallback to finding any available port
            this.log.warn(`Port ${targetPort} unavailable for ${serviceName}, finding alternative...`);
            return await this.findAvailablePort(serviceName, targetPort);

        } catch (error) {
            this.log.error(`Failed to allocate port for ${serviceName}:`, error.message);
            
            // Last resort: disable the service in development
            if (process.env.NODE_ENV === 'development') {
                this.log.warn(`⚠️  Service ${serviceName} will be disabled due to port allocation failure`);
                return null;
            }
            
            throw error;
        }
    }

    /**
     * Get all allocated ports info
     */
    getPortInfo() {
        const info = {};
        for (const [service, port] of this.allocatedPorts.entries()) {
            info[service] = port;
        }
        return info;
    }
}

// Singleton instance
const portManager = new PortManager();

module.exports = {
    PortManager,
    portManager,
    
    // Helper functions
    findAvailablePort: (serviceName, preferredPort) => portManager.findAvailablePort(serviceName, preferredPort),
    allocatePortSafely: (serviceName, preferredPort, killConflicts) => portManager.allocatePortSafely(serviceName, preferredPort, killConflicts),
    isPortAvailable: (port) => portManager.isPortAvailable(port),
    killPortProcesses: (ports) => portManager.killPortProcesses(ports),
    getPortInfo: () => portManager.getPortInfo(),
    releasePort: (serviceName) => portManager.releasePort(serviceName)
};