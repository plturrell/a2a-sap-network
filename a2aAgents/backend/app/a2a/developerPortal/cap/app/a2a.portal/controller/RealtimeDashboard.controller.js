sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
  'use strict';
  /* global WebSocket */

  return Controller.extend('a2a.portal.controller.RealtimeDashboard', {

    onInit: function () {
      // Initialize view model with comprehensive metrics
      const oViewModel = new JSONModel({
        autoRefresh: true,
        systemHealth: {
          score: 95,
          trend: 'Up'
        },
        agents: {
          total: 16,
          active: 15,
          degraded: 1,
          offline: 0
        },
        blockchain: {
          totalTps: 0,
          tpsHistory: [],
          networks: []
        },
        errors: {
          rate: 0.5,
          total: 0,
          recent: []
        },
        performance: {
          responseTimeData: [],
          throughputData: []
        },
        business: {
          transactionVolume: {},
          revenue: {},
          slaCompliance: {}
        },
        alerts: {
          critical: 0,
          warning: 0,
          info: 0
        },
        eventStream: [],
        communicationMatrix: {}
      });
      this.getView().setModel(oViewModel, 'view');

      // Initialize WebSocket connection for real-time data
      this._initializeWebSocket();

      // Start polling for fallback
      this._startPolling();

      // Initialize D3.js visualizations
      this._initializeVisualizations();
    },

    /**
         * Initialize WebSocket connection for real-time metrics
         */
    _initializeWebSocket: function () {
      const sProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const sHost = window.location.host;
      const sUrl = `${sProtocol  }//${  sHost  }/ws/metrics`;

      try {
        this.websocket = new WebSocket(sUrl);
                
        this.websocket.onopen = function () {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('WebSocket connected for real-time metrics');
                    
          // Subscribe to all channels
          this.websocket.send(JSON.stringify({
            type: 'subscribe',
            channels: ['all']
          }));
                    
          MessageToast.show('Connected to real-time data stream');
        }.bind(this);

        this.websocket.onmessage = function (event) {
          this._handleWebSocketMessage(event);
        }.bind(this);

        this.websocket.onerror = function (error) {
          console.error('WebSocket error:', error);
          MessageToast.show('Real-time connection error - falling back to polling');
        };

        this.websocket.onclose = function () {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('WebSocket disconnected');
          // Attempt reconnection after 5 seconds
          setTimeout(this._initializeWebSocket.bind(this), 5000);
        }.bind(this);

      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        // Fall back to polling
      }
    },

    /**
         * Handle incoming WebSocket messages
         */
    _handleWebSocketMessage: function (event) {
      try {
        const data = JSON.parse(event.data);
        const _oViewModel = this.getView().getModel('view');

        switch (data.type) {
        case 'agent_heartbeats':
          this._updateAgentHeartbeats(data.data);
          break;
                    
        case 'blockchain_metrics':
          this._updateBlockchainMetrics(data.data);
          break;
                    
        case 'performance_analytics':
          this._updatePerformanceMetrics(data.data);
          break;
                    
        case 'communication_patterns':
          this._updateCommunicationMatrix(data.data);
          break;
                    
        case 'business_metrics':
          this._updateBusinessMetrics(data.data);
          break;
                    
        case 'agent_status_update':
          this._handleAgentStatusChange(data.data);
          break;
                    
        case 'performance_anomaly':
          this._handlePerformanceAnomaly(data.data);
          break;
                    
        case 'blockchain_event':
          this._handleBlockchainEvent(data.data);
          break;
        }

      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    },

    /**
         * Update agent heartbeat data
         */
    _updateAgentHeartbeats: function (heartbeats) {
      const oViewModel = this.getView().getModel('view');
            
      // Calculate active agents
      const activeAgents = heartbeats.filter((hb) => {
        return hb.status === 'healthy';
      }).length;

      oViewModel.setProperty('/agents/active', activeAgents);
      oViewModel.setProperty('/agents/degraded', heartbeats.length - activeAgents);
            
      // Calculate system health score
      const healthScore = Math.round((activeAgents / heartbeats.length) * 100);
      oViewModel.setProperty('/systemHealth/score', healthScore);
            
      // Update trend
      const previousScore = oViewModel.getProperty('/systemHealth/score');
      oViewModel.setProperty('/systemHealth/trend', healthScore >= previousScore ? 'Up' : 'Down');
    },

    /**
         * Update blockchain metrics
         */
    _updateBlockchainMetrics: function (metrics) {
      const oViewModel = this.getView().getModel('view');
            
      // Calculate total TPS
      let totalTps = 0;
      const networks = [];
            
      Object.keys(metrics).forEach((network) => {
        if (network !== 'a2aContracts' && network !== 'crossChainBridge') {
          const networkData = metrics[network];
          totalTps += networkData.tps || 0;
                    
          networks.push({
            name: network.charAt(0).toUpperCase() + network.slice(1),
            icon: 'sap-icon://chain-link',
            blockHeight: networkData.blockHeight,
            gasPrice: Math.round(networkData.gasPrice),
            tps: Math.round(networkData.tps),
            congestion: Math.round(networkData.networkCongestion || 0),
            status: networkData.networkCongestion > 80 ? 'Congested' : 'Healthy',
            statusState: networkData.networkCongestion > 80 ? 'Error' : 'Success'
          });
        }
      });

      oViewModel.setProperty('/blockchain/totalTps', Math.round(totalTps));
      oViewModel.setProperty('/blockchain/networks', networks);
            
      // Update TPS history for chart
      let tpsHistory = oViewModel.getProperty('/blockchain/tpsHistory') || [];
      tpsHistory.push({
        x: tpsHistory.length,
        y: Math.round(totalTps)
      });
            
      // Keep only last 20 points
      if (tpsHistory.length > 20) {
        tpsHistory = tpsHistory.slice(-20);
      }
            
      oViewModel.setProperty('/blockchain/tpsHistory', tpsHistory);
    },

    /**
         * Update performance metrics
         */
    _updatePerformanceMetrics: function (analytics) {
      const oViewModel = this.getView().getModel('view');
            
      // Update error rate
      oViewModel.setProperty('/errors/rate', Math.round(analytics.systemPerformance.errorRate * 10) / 10);
            
      // Update response time data
      let responseTimeData = oViewModel.getProperty('/performance/responseTimeData') || [];
      responseTimeData.push({
        timestamp: new Date().toISOString(),
        avgResponseTime: Math.round(analytics.systemPerformance.avgResponseTime),
        p95: Math.round(analytics.systemPerformance.p95ResponseTime),
        p99: Math.round(analytics.systemPerformance.p99ResponseTime)
      });
            
      // Keep only last 30 points
      if (responseTimeData.length > 30) {
        responseTimeData = responseTimeData.slice(-30);
      }
            
      oViewModel.setProperty('/performance/responseTimeData', responseTimeData);
            
      // Update throughput data
      let throughputData = oViewModel.getProperty('/performance/throughputData') || [];
      throughputData.push({
        timestamp: new Date().toISOString(),
        throughput: Math.round(analytics.systemPerformance.throughput)
      });
            
      if (throughputData.length > 30) {
        throughputData = throughputData.slice(-30);
      }
            
      oViewModel.setProperty('/performance/throughputData', throughputData);
    },

    /**
         * Update communication matrix visualization
         */
    _updateCommunicationMatrix: function (data) {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/communicationMatrix', data);
            
      // Update D3.js visualization
      this._renderCommunicationMatrix(data);
    },

    /**
         * Update business metrics
         */
    _updateBusinessMetrics: function (metrics) {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/business', metrics);
    },

    /**
         * Handle agent status change events
         */
    _handleAgentStatusChange: function (data) {
      const oViewModel = this.getView().getModel('view');
      let eventStream = oViewModel.getProperty('/eventStream') || [];
            
      eventStream.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        severity: data.currentStatus === 'degraded' ? 'warning' : 'info',
        title: 'Agent Status Change',
        description: `${data.agentId  } changed from ${  data.previousStatus  } to ${  data.currentStatus}`,
        details: data
      });
            
      // Keep only last 100 events
      if (eventStream.length > 100) {
        eventStream = eventStream.slice(0, 100);
      }
            
      oViewModel.setProperty('/eventStream', eventStream);
            
      // Update alert counts
      if (data.currentStatus === 'degraded') {
        const warnings = oViewModel.getProperty('/alerts/warning') || 0;
        oViewModel.setProperty('/alerts/warning', warnings + 1);
      }
    },

    /**
         * Handle performance anomaly events
         */
    _handlePerformanceAnomaly: function (anomaly) {
      const oViewModel = this.getView().getModel('view');
      const eventStream = oViewModel.getProperty('/eventStream') || [];
            
      eventStream.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        severity: anomaly.severity,
        title: 'Performance Anomaly Detected',
        description: anomaly.message,
        details: anomaly
      });
            
      oViewModel.setProperty('/eventStream', eventStream.slice(0, 100));
            
      // Update alert counts
      const alertType = anomaly.severity === 'critical' ? 'critical' : 'warning';
      const count = oViewModel.getProperty(`/alerts/${  alertType}`) || 0;
      oViewModel.setProperty(`/alerts/${  alertType}`, count + 1);
            
      // Show notification for critical anomalies
      if (anomaly.severity === 'critical') {
        MessageBox.error(anomaly.message, {
          title: 'Critical Performance Issue'
        });
      }
    },

    /**
         * Handle blockchain events
         */
    _handleBlockchainEvent: function (event) {
      const oViewModel = this.getView().getModel('view');
      const eventStream = oViewModel.getProperty('/eventStream') || [];
            
      eventStream.unshift({
        id: Date.now(),
        timestamp: new Date().toISOString(),
        severity: 'info',
        title: 'Blockchain Event',
        description: `${event.type  } on ${  event.network}`,
        details: event
      });
            
      oViewModel.setProperty('/eventStream', eventStream.slice(0, 100));
    },

    /**
         * Initialize D3.js visualizations
         */
    _initializeVisualizations: function () {
      // Initialize after view is rendered
      setTimeout(() => {
        this._initializeCommunicationMatrix();
      }, 1000);
    },

    /**
         * Initialize communication matrix D3.js visualization
         */
    _initializeCommunicationMatrix: function () {
      // This would contain D3.js code to create a Sankey diagram
      // For now, we'll use a placeholder
      const container = this.byId('communicationMatrixContainer');
      if (container) {
        container.addContent(new sap.m.Text({
          text: 'Agent communication flow visualization will be rendered here using D3.js Sankey diagram'
        }));
      }
    },

    /**
         * Render communication matrix using D3.js
         */
    _renderCommunicationMatrix: function (data) {
      // D3.js rendering logic would go here
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Rendering communication matrix with data:', data);
    },

    /**
         * Start polling for data (fallback when WebSocket unavailable)
         */
    _startPolling: function () {
      // Poll every 5 seconds if WebSocket is not connected
      this.pollingInterval = setInterval(() => {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
          this._loadDashboardData();
        }
      }, 5000);
    },

    /**
         * Load dashboard data via REST API
         */
    _loadDashboardData: function () {
      // Implementation for REST API fallback
      jQuery.ajax({
        url: '/srv/monitoring/a2a-agents/data',
        method: 'GET',
        success: function (_data) {
          // Update view model with data
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Loaded dashboard data via REST API');
        }.bind(this),
        error: function (error) {
          console.error('Failed to load dashboard data:', error);
        }
      });
    },

    /**
         * Format timestamp for display
         */
    formatTimestamp: function (timestamp) {
      if (!timestamp) {
        return '';
      }
            
      const date = new Date(timestamp);
      const now = new Date();
      const diff = now - date;
            
      if (diff < 60000) { // Less than 1 minute
        return `${Math.floor(diff / 1000)  }s ago`;
      } else if (diff < 3600000) { // Less than 1 hour
        return `${Math.floor(diff / 60000)  }m ago`;
      } else if (diff < 86400000) { // Less than 1 day
        return `${Math.floor(diff / 3600000)  }h ago`;
      } else {
        return DateFormat.getDateTimeInstance({
          style: 'short'
        }).format(date);
      }
    },

    /**
         * Toggle full screen mode
         */
    onToggleFullScreen: function () {
      const elem = document.documentElement;
      if (!document.fullscreenElement) {
        elem.requestFullscreen().catch((_err) => {
          MessageToast.show('Failed to enter fullscreen mode');
        });
      } else {
        document.exitFullscreen();
      }
    },

    /**
         * Export dashboard report
         */
    onExportReport: function () {
      MessageBox.information(
        'Dashboard report export will generate a comprehensive PDF report with all current metrics, charts, and analytics.',
        {
          title: 'Export Report',
          actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show('Generating report...');
              // Report generation logic would go here
            }
          }
        }
      );
    },

    /**
         * Handle event item press
         */
    onEventPress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const oEventData = oContext.getObject();
            
      MessageBox.information(
        JSON.stringify(oEventData.details, null, 2),
        {
          title: oEventData.title,
          styleClass: 'sapUiSizeCompact'
        }
      );
    },

    /**
         * Cleanup on exit
         */
    onExit: function () {
      if (this.websocket) {
        this.websocket.close();
      }
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval);
      }
    }
  });
});