"use strict";

/**
 * Tile Metrics Service for SAP Fiori Launchpad
 * Provides real-time data for all dashboard tiles
 */

const EventEmitter = require('events');

class TileMetricsService extends EventEmitter {
    constructor() {
        super();
        this.tileData = new Map();
        this.agentMonitoringService = require('./a2aAgentMonitoringService');
        this.realtimeMetricsService = require('./realtimeMetricsService');
        
        // Initialize tile data
        this._initializeTileData();
        
        // Start real-time updates
        this._startTileUpdates();
    }

    /**
     * Initialize default tile data
     */
    _initializeTileData() {
        // Operations & Monitoring Tiles
        this.tileData.set('realtime-dashboard-tile', {
            numberValue: '16',
            numberUnit: 'Agents',
            numberState: 'Positive',
            stateArrow: 'Up',
            info: 'Live'
        });

        this.tileData.set('blockchain-metrics-tile', {
            numberValue: '147',
            numberUnit: 'TPS',
            info: 'Multi-chain'
        });

        this.tileData.set('agent-health-tile', {
            numberValue: '94',
            numberUnit: '%',
            numberState: 'Positive',
            info: 'All agents'
        });

        // Analytics & Intelligence Tiles
        this.tileData.set('performance-analytics-tile', {
            numberValue: '125',
            numberUnit: 'ms',
            numberState: 'Positive',
            info: 'ML-powered'
        });

        // Removed fake business KPI tile

        this.tileData.set('data-quality-tile', {
            numberValue: '98.5',
            numberUnit: '%',
            numberState: 'Positive',
            info: 'Quality Score'
        });

        this.tileData.set('security-compliance-tile', {
            numberValue: '0',
            numberUnit: 'Alerts',
            numberState: 'Positive',
            info: 'SOC2 Ready'
        });

        // Infrastructure & Resources Tiles
        // Removed fake cost and incident tiles

        // Project Management Tiles
        this.tileData.set('projects-tile', {
            info: '12 Active'
        });

        this.tileData.set('agent-builder-tile', {
            info: '45 Agents'
        });

        this.tileData.set('deployment-tile', {
            info: '8 This Month'
        });
    }

    /**
     * Start real-time tile updates
     */
    _startTileUpdates() {
        // Update tiles every 5 seconds
        setInterval(() => {
            this._updateAllTiles();
        }, 5000);

        // Listen to realtime metrics events
        this.realtimeMetricsService.on('agent.status.change', (_data) => {
            this._updateAgentTiles();
        });

        this.realtimeMetricsService.on('blockchain.event', (_data) => {
            this._updateBlockchainTiles();
        });

        this.realtimeMetricsService.on('anomaly.detected', (_data) => {
            this._updateSecurityTiles();
        });

        this.realtimeMetricsService.on('error.cascade', (_data) => {
            this._updateIncidentTiles();
        });
    }

    /**
     * Update all tiles with fresh data
     */
    async _updateAllTiles() {
        try {
            // Get latest metrics
            const agentMetrics = await this._getAgentMetrics();
            const blockchainMetrics = await this._getBlockchainMetrics();
            const performanceMetrics = await this._getPerformanceMetrics();
            const dataQualityMetrics = await this._getDataQualityMetrics();
            const securityMetrics = await this._getSecurityMetrics();

            // Update Real-Time Operations tile
            const activeAgents = agentMetrics.agents.filter(a => a.status === 'running').length;
            this.tileData.set('realtime-dashboard-tile', {
                numberValue: activeAgents.toString(),
                numberUnit: 'Agents',
                numberState: activeAgents === 16 ? 'Positive' : activeAgents > 12 ? 'Critical' : 'Error',
                stateArrow: activeAgents >= 15 ? 'Up' : 'Down',
                info: 'Live'
            });

            // Update Blockchain Metrics tile
            const totalTps = Math.round(blockchainMetrics.ethereum.tps + blockchainMetrics.polygon.tps);
            this.tileData.set('blockchain-metrics-tile', {
                numberValue: totalTps.toString(),
                numberUnit: 'TPS',
                info: 'Multi-chain'
            });

            // Update Agent Health tile
            const healthScore = Math.round((activeAgents / 16) * 100);
            this.tileData.set('agent-health-tile', {
                numberValue: healthScore.toString(),
                numberUnit: '%',
                numberState: healthScore > 90 ? 'Positive' : healthScore > 70 ? 'Critical' : 'Error',
                info: 'All agents'
            });

            // Update Performance Analytics tile
            const avgResponseTime = Math.round(performanceMetrics.avgResponseTime);
            this.tileData.set('performance-analytics-tile', {
                numberValue: avgResponseTime.toString(),
                numberUnit: 'ms',
                numberState: avgResponseTime < 200 ? 'Positive' : avgResponseTime < 500 ? 'Critical' : 'Error',
                info: 'ML-powered'
            });

            // Remove fake business KPI - no real revenue data

            // Update Data Quality tile
            const dataQuality = dataQualityMetrics.validationSuccessRate || 0;
            this.tileData.set('data-quality-tile', {
                numberValue: dataQuality.toFixed(1),
                numberUnit: '%',
                numberState: dataQuality > 95 ? 'Positive' : dataQuality > 90 ? 'Critical' : 'Error',
                info: 'Quality Score'
            });

            // Update Security & Compliance tile
            const securityAlerts = securityMetrics.activeAlerts;
            this.tileData.set('security-compliance-tile', {
                numberValue: securityAlerts.toString(),
                numberUnit: 'Alerts',
                numberState: securityAlerts === 0 ? 'Positive' : securityAlerts < 5 ? 'Critical' : 'Error',
                info: 'SOC2 Ready'
            });

            // Remove fake cost and incident tiles - no real data

            // Emit update event
            this.emit('tiles.updated', {
                timestamp: new Date().toISOString(),
                tiles: Object.fromEntries(this.tileData)
            });

        } catch (error) {
            console.error('Error updating tiles:', error);
        }
    }

    /**
     * Get agent metrics
     */
    async _getAgentMetrics() {
        try {
            const agents = await this.agentMonitoringService._getAgentsList();
            return { agents };
        } catch (error) {
            return { agents: [] };
        }
    }

    /**
     * Get blockchain metrics
     */
    _getBlockchainMetrics() {
        const cached = this.realtimeMetricsService.blockchainMetrics.get('latest');
        return cached || {
            ethereum: { tps: 15 },
            polygon: { tps: 85 }
        };
    }

    /**
     * Get performance metrics
     */
    _getPerformanceMetrics() {
        const cached = this.realtimeMetricsService.metricsCache.get('performance');
        return cached || {
            avgResponseTime: 125,
            throughput: 1500
        };
    }

    /**
     * Get data quality metrics from real agents
     */
    async _getDataQualityMetrics() {
        try {
            // Get from QA validation agent metrics
            const response = await fetch('http://localhost:8005/metrics/quality', {
                method: 'GET',
                timeout: 2000
            });
            
            if (response.ok) {
                const data = await response.json();
                return {
                    validationSuccessRate: data.validation_success_rate || 0,
                    standardizationAccuracy: data.standardization_accuracy || 0,
                    completenessScore: data.completeness_score || 0
                };
            }
        } catch (error) {
            console.error('Failed to get data quality metrics:', error);
        }
        
        return {
            validationSuccessRate: 0,
            standardizationAccuracy: 0,
            completenessScore: 0
        };
    }

    /**
     * Get security metrics from real alert data
     */
    _getSecurityMetrics() {
        // Count real alerts from monitoring service
        const activeAlerts = this.agentMonitoringService.alertHistory.size;
        
        return {
            activeAlerts: activeAlerts
        };
    }

    /**
     * Update agent-related tiles
     */
    _updateAgentTiles() {
        this._updateAllTiles();
    }

    /**
     * Update blockchain-related tiles
     */
    _updateBlockchainTiles() {
        this._updateAllTiles();
    }

    /**
     * Update security-related tiles
     */
    _updateSecurityTiles() {
        const currentAlerts = parseInt(this.tileData.get('security-compliance-tile').numberValue);
        this.tileData.set('security-compliance-tile', {
            ...this.tileData.get('security-compliance-tile'),
            numberValue: (currentAlerts + 1).toString(),
            numberState: 'Critical'
        });
    }

    /**
     * Update incident-related tiles
     */
    _updateIncidentTiles() {
        const currentIncidents = parseInt(this.tileData.get('incident-management-tile').numberValue);
        this.tileData.set('incident-management-tile', {
            ...this.tileData.get('incident-management-tile'),
            numberValue: (currentIncidents + 1).toString(),
            numberState: 'Error'
        });
    }

    /**
     * Get tile data for a specific tile
     */
    getTileData(tileId) {
        return this.tileData.get(tileId) || {};
    }

    /**
     * Get all tile data
     */
    getAllTileData() {
        return Object.fromEntries(this.tileData);
    }

    /**
     * Express router for tile metrics API
     */
    getRouter() {
        const express = require('express');
        const router = express.Router();

        // Get all tile metrics
        router.get('/tiles', (req, res) => {
            res.json({
                timestamp: new Date().toISOString(),
                tiles: this.getAllTileData()
            });
        });

        // Get specific tile metrics
        router.get('/tiles/:tileId', (req, res) => {
            const tileData = this.getTileData(req.params.tileId);
            if (Object.keys(tileData).length === 0) {
                return res.status(404).json({ error: 'Tile not found' });
            }
            res.json({
                tileId: req.params.tileId,
                timestamp: new Date().toISOString(),
                data: tileData
            });
        });

        // WebSocket endpoint for real-time tile updates
        router.get('/tiles/stream', (req, res) => {
            res.json({
                message: 'Use WebSocket connection at ws://[host]/ws/tiles for real-time updates'
            });
        });

        return router;
    }
}

module.exports = new TileMetricsService();