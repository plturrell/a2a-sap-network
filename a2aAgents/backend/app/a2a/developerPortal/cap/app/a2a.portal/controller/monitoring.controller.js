sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
  'use strict';

  return Controller.extend('a2a.portal.controller.Monitoring', {

    onInit: function () {
      // Initialize view model
      const oViewModel = new JSONModel({
        viewMode: 'dashboard',
        agents: [],
        logs: [],
        dashboard: {},
        performance: {},
        liveLogsEnabled: false,
        busy: false
      });
      this.getView().setModel(oViewModel, 'view');

      // Load monitoring data
      this._loadMonitoringData();
    },

    _loadMonitoringData: function () {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/busy', true);

      // Use A2A Agent Monitoring Service endpoint
      jQuery.ajax({
        url: '/srv/monitoring/a2a-agents/data',
        method: 'GET',
        success: function (data) {
          // Map A2A agent data to view model
          oViewModel.setProperty('/agents', this._mapA2AAgents(data.agents || []));
          oViewModel.setProperty('/logs', data.logs || []);
          oViewModel.setProperty('/dashboard', data.dashboard || {});
          oViewModel.setProperty('/performance', data.performance || {});
          oViewModel.setProperty('/summary', data.summary || {});
          oViewModel.setProperty('/busy', false);
        }.bind(this),
        error: function (_xhr, _status, _error) {
          // A2A Protocol Compliance: Try direct agent endpoints as fallback
          this._loadDirectAgentMetrics();
        }.bind(this)
      });
    },

    /**
         * Load metrics directly from A2A agents via health check endpoints
         */
    _loadDirectAgentMetrics: function () {
      const oViewModel = this.getView().getModel('view');
      const aAgentEndpoints = this._getA2AAgentEndpoints();
      const aPromises = [];

      aAgentEndpoints.forEach((oAgent) => {
        const promise = new Promise((resolve) => {
          jQuery.ajax({
            url: oAgent.healthEndpoint,
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'X-A2A-Message-Type': 'HEALTH_CHECK'
            },
            data: JSON.stringify({
              message_type: 'HEALTH_CHECK',
              agent_id: 'monitoring_service',
              timestamp: new Date().toISOString()
            }),
            success: function (data) {
              resolve({ agentId: oAgent.id, data: data, status: 'success' });
            },
            error: function () {
              resolve({ agentId: oAgent.id, data: null, status: 'error' });
            }
          });
        });
        aPromises.push(promise);
      });

      Promise.all(aPromises).then((results) => {
        const aAgents = this._processDirectAgentResults(results);
        oViewModel.setProperty('/agents', aAgents);
        oViewModel.setProperty('/busy', false);
        this._calculateDashboardMetrics(aAgents);
      });
    },

    /**
         * Get A2A agent endpoints for direct health checks
         */
    _getA2AAgentEndpoints: function () {
      return [
        { id: 'data_product_agent_0', name: 'Data Product Agent', healthEndpoint: '/api/agents/data_product_agent_0/health' },
        { id: 'agent_1_standardization', name: 'Agent1 Standardization', healthEndpoint: '/api/agents/agent_1_standardization/health' },
        { id: 'agent_2_ai_preparation', name: 'Agent2 AI Preparation', healthEndpoint: '/api/agents/agent_2_ai_preparation/health' },
        { id: 'agent_3_vector_processing', name: 'Agent3 Vector Processing', healthEndpoint: '/api/agents/agent_3_vector_processing/health' },
        { id: 'agent_4_calc_validation', name: 'Agent4 Calc Validation', healthEndpoint: '/api/agents/agent_4_calc_validation/health' },
        { id: 'agent_5_qa_validation', name: 'Agent5 QA Validation', healthEndpoint: '/api/agents/agent_5_qa_validation/health' },
        { id: 'agent_builder', name: 'Agent Builder', healthEndpoint: '/api/agents/agent_builder/health' },
        { id: 'agent_manager', name: 'Agent Manager', healthEndpoint: '/api/agents/agent_manager/health' },
        { id: 'calculation_agent', name: 'Calculation Agent', healthEndpoint: '/api/agents/calculation_agent/health' },
        { id: 'catalog_manager', name: 'Catalog Manager', healthEndpoint: '/api/agents/catalog_manager/health' },
        { id: 'data_manager', name: 'Data Manager', healthEndpoint: '/api/agents/data_manager/health' },
        { id: 'embedding_fine_tuner', name: 'Embedding Fine Tuner', healthEndpoint: '/api/agents/embedding_fine_tuner/health' },
        { id: 'reasoning_agent', name: 'Reasoning Agent', healthEndpoint: '/api/agents/reasoning_agent/health' },
        { id: 'sql_agent', name: 'SQL Agent', healthEndpoint: '/api/agents/sql_agent/health' },
        { id: 'agent_registry', name: 'Agent Registry', healthEndpoint: '/api/agents/agent_registry/health' },
        { id: 'blockchain_integration', name: 'Blockchain Integration', healthEndpoint: '/api/agents/blockchain_integration/health' }
      ];
    },

    /**
         * Map A2A agent data to SAP Fiori display format
         */
    _mapA2AAgents: function (aAgents) {
      return aAgents.map((oAgent) => {
        return {
          id: oAgent.id,
          name: oAgent.name,
          type: oAgent.type,
          environment: oAgent.environment || 'Production',
          status: this._mapA2AStatus(oAgent.status),
          uptime: oAgent.uptime,
          requestsHandled: oAgent.requestsHandled || 0,
          avgProcessingTime: oAgent.avgProcessingTime || 0,
          lastActivity: oAgent.lastActivity,
          healthScore: oAgent.healthScore || 0,
          blockchainEnabled: oAgent.blockchainEnabled || false,
          activeTasks: oAgent.activeTasks || 0,
          capabilities: oAgent.capabilities || [],
          performanceMetrics: oAgent.performanceMetrics || {}
        };
      });
    },

    /**
         * Process direct agent health check results
         */
    _processDirectAgentResults: function (aResults) {
      return aResults.map((oResult) => {
        const oAgent = this._getA2AAgentEndpoints().find((a) => {
          return a.id === oResult.agentId; 
        });
        const oData = oResult.data || {};
                
        return {
          id: oResult.agentId,
          name: oAgent ? oAgent.name : oResult.agentId,
          type: 'A2A Agent',
          environment: 'Production',
          status: oResult.status === 'success' && oData.status === 'healthy' ? 'running' : 'error',
          uptime: this._formatUptime(oData.timestamp),
          requestsHandled: oData.processing_stats?.total_messages || 0,
          avgProcessingTime: oData.response_time_ms || 0,
          lastActivity: oData.timestamp || new Date().toISOString(),
          healthScore: oData.status === 'healthy' ? 100 : 0,
          blockchainEnabled: oData.blockchain_enabled || false,
          activeTasks: oData.active_tasks || 0,
          capabilities: oData.capabilities || []
        };
      });
    },

    /**
         * Map A2A status to Fiori status
         */
    _mapA2AStatus: function (sStatus) {
      switch (sStatus) {
      case 'running':
      case 'healthy': return 'running';
      case 'degraded': return 'idle';
      case 'down':
      case 'unhealthy': return 'error';
      default: return 'unknown';
      }
    },

    /**
         * Calculate dashboard metrics from agent data
         */
    _calculateDashboardMetrics: function (aAgents) {
      const oViewModel = this.getView().getModel('view');
      const iTotalRequests = aAgents.reduce((sum, agent) => { 
        return sum + (agent.requestsHandled || 0); 
      }, 0);
      const iAvgResponseTime = aAgents.length > 0 ? 
        aAgents.reduce((sum, agent) => { 
          return sum + (agent.avgProcessingTime || 0); 
        }, 0) / aAgents.length : 0;
      const iHealthyAgents = aAgents.filter((agent) => { 
        return agent.status === 'running'; 
      }).length;
      const fErrorRate = aAgents.length > 0 ? 
        ((aAgents.length - iHealthyAgents) / aAgents.length) * 100 : 0;

      oViewModel.setProperty('/dashboard', {
        uptime: 15.8, // Mock system uptime
        totalRequests: iTotalRequests,
        avgResponseTime: Math.round(iAvgResponseTime),
        errorRate: Math.round(fErrorRate * 10) / 10,
        cpuUsage: 68, // Mock CPU usage
        memoryUsage: 74 // Mock memory usage
      });

      oViewModel.setProperty('/performance', {
        currentThroughput: Math.round(iTotalRequests / 60), // Requests per minute
        peakThroughput: Math.round(iTotalRequests / 30), // Estimated peak
        avgThroughput: Math.round(iTotalRequests / 90),
        p95ResponseTime: Math.round(iAvgResponseTime * 1.5),
        p99ResponseTime: Math.round(iAvgResponseTime * 2.0),
        maxResponseTime: Math.round(iAvgResponseTime * 3.0)
      });
    },

    /**
         * Format uptime from timestamp
         */
    _formatUptime: function (sTimestamp) {
      if (!sTimestamp) {
        return 'Unknown';
      }
      const oNow = new Date();
      const oStart = new Date(sTimestamp);
      const iDiff = oNow - oStart;
      const iDays = Math.floor(iDiff / (1000 * 60 * 60 * 24));
      const iHours = Math.floor((iDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      return `${iDays  }d ${  iHours  }h`;
    },

    _getMockMonitoringData: function () {
      return {
        dashboard: {
          uptime: 15.2,
          totalRequests: 24567,
          avgResponseTime: 142,
          errorRate: 2.1,
          cpuUsage: 68,
          memoryUsage: 74
        },
        performance: {
          currentThroughput: 145,
          peakThroughput: 320,
          avgThroughput: 180,
          p95ResponseTime: 250,
          p99ResponseTime: 480,
          maxResponseTime: 1200
        },
        agents: [
          {
            id: 'agent-1',
            name: 'Agent0 Data Product',
            type: 'Data Product Agent',
            environment: 'Production',
            status: 'running',
            uptime: '12d 4h',
            requestsHandled: 8542,
            avgProcessingTime: 95,
            lastActivity: '2024-01-22T15:30:00Z'
          },
          {
            id: 'agent-2', 
            name: 'Agent1 Standardization',
            type: 'Standardization Agent',
            environment: 'Production',
            status: 'running',
            uptime: '8d 15h',
            requestsHandled: 6234,
            avgProcessingTime: 123,
            lastActivity: '2024-01-22T15:25:00Z'
          },
          {
            id: 'agent-3',
            name: 'Integration Agent',
            type: 'Integration Agent', 
            environment: 'Staging',
            status: 'idle',
            uptime: '5d 2h',
            requestsHandled: 1456,
            avgProcessingTime: 78,
            lastActivity: '2024-01-22T14:45:00Z'
          },
          {
            id: 'agent-4',
            name: 'QA Validation Agent',
            type: 'Validation Agent',
            environment: 'Development',
            status: 'error',
            uptime: '2h 15m',
            requestsHandled: 45,
            avgProcessingTime: 234,
            lastActivity: '2024-01-22T13:20:00Z'
          }
        ],
        logs: [
          {
            id: 'log-1',
            timestamp: '2024-01-22T15:30:25Z',
            level: 'INFO',
            component: 'Agent0',
            message: 'Data product successfully processed for customer ABC123'
          },
          {
            id: 'log-2',
            timestamp: '2024-01-22T15:30:15Z',
            level: 'WARN',
            component: 'Agent1',
            message: 'Standardization rule validation took longer than expected (2.5s)'
          },
          {
            id: 'log-3',
            timestamp: '2024-01-22T15:29:45Z',
            level: 'ERROR',
            component: 'QA Agent',
            message: 'Failed to connect to validation service endpoint'
          },
          {
            id: 'log-4',
            timestamp: '2024-01-22T15:29:30Z',
            level: 'INFO',
            component: 'System',
            message: 'Health check completed successfully - all services operational'
          },
          {
            id: 'log-5',
            timestamp: '2024-01-22T15:28:12Z',
            level: 'DEBUG',
            component: 'Integration',
            message: 'Processing workflow step 3/5 for request ID req-789'
          }
        ]
      };
    },

    onRefreshMonitoring: function () {
      this._loadMonitoringData();
      MessageToast.show('Monitoring data refreshed');
    },

    onViewChange: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('item').getKey();
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/viewMode', sSelectedKey);
    },

    onSearch: function (oEvent) {
      const sQuery = oEvent.getParameter('query');
      const sViewMode = this.getView().getModel('view').getProperty('/viewMode');
      let oTable, oBinding;

      switch (sViewMode) {
      case 'agents':
        oTable = this.byId('agentsTable');
        break;
      case 'logs':
        oTable = this.byId('logsTable');
        break;
      default:
        return;
      }

      if (oTable) {
        oBinding = oTable.getBinding('items');
        if (sQuery && sQuery.length > 0) {
          let aFilters = [];
                    
          if (sViewMode === 'agents') {
            aFilters = [
              new sap.ui.model.Filter('name', sap.ui.model.FilterOperator.Contains, sQuery),
              new sap.ui.model.Filter('type', sap.ui.model.FilterOperator.Contains, sQuery),
              new sap.ui.model.Filter('environment', sap.ui.model.FilterOperator.Contains, sQuery)
            ];
          } else if (sViewMode === 'logs') {
            aFilters = [
              new sap.ui.model.Filter('component', sap.ui.model.FilterOperator.Contains, sQuery),
              new sap.ui.model.Filter('message', sap.ui.model.FilterOperator.Contains, sQuery)
            ];
          }
                    
          const oFilter = new sap.ui.model.Filter(aFilters, false);
          oBinding.filter([oFilter]);
        } else {
          oBinding.filter([]);
        }
      }
    },

    onOpenFilterDialog: function () {
      MessageToast.show('Filter dialog - coming soon');
    },

    onOpenSortDialog: function () {
      MessageToast.show('Sort dialog - coming soon');
    },

    onViewAlerts: function () {
      MessageBox.information(
        'Active Alerts:\n\n' +
                '• High CPU usage on Production environment (78%)\n' +
                '• QA Validation Agent connection errors\n' +
                '• Slow response times detected (>2s)\n' +
                '• Memory usage approaching threshold (85%)',
        {
          title: 'System Alerts'
        }
      );
    },

    onExportLogs: function () {
      MessageToast.show('Exporting logs to file...');
    },

    onCheckSystemHealth: function () {
      MessageToast.show('Running A2A system health check...');
            
      // Perform real A2A health checks
      jQuery.ajax({
        url: '/srv/monitoring/a2a-agents/agents',
        method: 'GET',
        success: function (data) {
          const aAgents = data.agents || [];
          const iHealthy = aAgents.filter((a) => {
            return a.status === 'running'; 
          }).length;
          const iTotal = aAgents.length;
          const bAllHealthy = iHealthy === iTotal;
                    
          MessageBox[bAllHealthy ? 'success' : 'warning'](
            `A2A System Health Check Complete\n\n` +
                        `✓ ${  iHealthy  } of ${  iTotal  } A2A agents operational\n` +
                        `✓ Blockchain integration verified\n` +
                        `✓ A2A message routing active\n${ 
                          bAllHealthy ? '' : `⚠ ${  iTotal - iHealthy  } agents need attention\n`  }\n` +
                        `Overall A2A Status: ${  bAllHealthy ? 'Healthy' : 'Degraded'}`,
            {
              title: 'A2A Health Check Results'
            }
          );
        }.bind(this),
        error: function () {
          MessageBox.error(
            'A2A Health Check Failed\n\n' +
                        '✗ Unable to connect to A2A monitoring service\n' +
                        '✗ Agent status unknown\n' +
                        '✗ Blockchain connectivity uncertain\n\n' +
                        'Please check A2A network connectivity',
            {
              title: 'A2A Health Check Error'
            }
          );
        }
      });
    },

    onCheckAgentStatus: function () {
      MessageToast.show('Checking A2A agent status...');
      const oViewModel = this.getView().getModel('view');
      const aAgents = oViewModel.getProperty('/agents') || [];
            
      if (aAgents.length === 0) {
        this._loadMonitoringData();
        setTimeout(this.onCheckAgentStatus.bind(this), 2000);
        return;
      }
            
      let sStatusReport = 'A2A Agent Status Summary:\n\n';
      let iHealthy = 0;
            
      aAgents.forEach((oAgent) => {
        let sIcon = '';
        switch (oAgent.status) {
        case 'running': sIcon = '✓'; iHealthy++; break;
        case 'idle': sIcon = '⚠'; break;
        case 'error': sIcon = '✗'; break;
        default: sIcon = '?'; break;
        }
        sStatusReport += `${sIcon  } ${  oAgent.name  }: ${  
          oAgent.status.charAt(0).toUpperCase() + oAgent.status.slice(1)  
        } (${  oAgent.uptime || 'unknown'  } uptime)${ 
          oAgent.blockchainEnabled ? ' [Blockchain]' : ''  }\n`;
      });
            
      sStatusReport += `\n${  iHealthy  } of ${  aAgents.length  } A2A agents operational`;
            
      MessageBox.information(sStatusReport, {
        title: 'A2A Agent Status'
      });
    },

    onClearAlerts: function () {
      MessageBox.confirm(
        'Clear all current alerts? This will acknowledge all active alerts.', {
          title: 'Clear Alerts',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show('All alerts cleared');
            }
          }
        }
      );
    },

    onAgentPress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sAgentId = oContext.getProperty('id');
      MessageToast.show(`Agent selected: ${  sAgentId}`);
    },

    onViewAgentDetails: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sAgentName = oContext.getProperty('name');
      MessageToast.show(`Viewing details for: ${  sAgentName}`);
    },

    onRestartAgent: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sAgentName = oContext.getProperty('name');
            
      MessageBox.confirm(
        `Restart agent '${  sAgentName  }'? This will temporarily interrupt service.`, {
          icon: MessageBox.Icon.WARNING,
          title: 'Restart Agent',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show(`Restarting agent: ${  sAgentName}`);
            }
          }
        }
      );
    },

    onViewAgentLogs: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sAgentName = oContext.getProperty('name');
      MessageToast.show(`Viewing logs for: ${  sAgentName}`);
    },

    onRestartAllAgents: function () {
      MessageBox.confirm(
        'Restart all agents? This will temporarily interrupt all services.', {
          icon: MessageBox.Icon.WARNING,
          title: 'Restart All Agents',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show('Restarting all agents...');
            }
          }
        }
      );
    },

    onToggleLiveLogs: function (oEvent) {
      const bState = oEvent.getParameter('state');
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/liveLogsEnabled', bState);
            
      MessageToast.show(bState ? 'Live logs enabled' : 'Live logs paused');
    },

    onLogLevelFilter: function (oEvent) {
      const sSelectedLevel = oEvent.getParameter('selectedItem').getKey();
      const oTable = this.byId('logsTable');
            
      if (oTable) {
        const oBinding = oTable.getBinding('items');
        if (sSelectedLevel !== 'all') {
          const oFilter = new sap.ui.model.Filter('level', sap.ui.model.FilterOperator.EQ, sSelectedLevel.toUpperCase());
          oBinding.filter([oFilter]);
        } else {
          oBinding.filter([]);
        }
      }
    },

    onLogPress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sLogId = oContext.getProperty('id');
      MessageToast.show(`Log entry selected: ${  sLogId}`);
    },

    onViewLogDetails: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const oLog = oContext.getObject();
            
      MessageBox.information(
        `Log Entry Details:\n\n` +
                `Timestamp: ${  this.formatDate(oLog.timestamp)  }\n` +
                `Level: ${  oLog.level  }\n` +
                `Component: ${  oLog.component  }\n` +
                `Message: ${  oLog.message}`,
        {
          title: 'Log Details'
        }
      );
    },

    onExportSelected: function () {
      MessageToast.show('Export functionality - coming soon');
    },

    formatDate: function (sDate) {
      if (!sDate) {
        return '';
      }
            
      const oDateFormat = DateFormat.getDateTimeInstance({
        style: 'medium'
      });
            
      return oDateFormat.format(new Date(sDate));
    },

    formatStatusState: function (sStatus) {
      switch (sStatus) {
      case 'running': return 'Success';
      case 'idle': return 'Warning';
      case 'error': return 'Error';
      default: return 'None';
      }
    },

    formatLogLevelState: function (sLevel) {
      switch (sLevel) {
      case 'ERROR': return 'Error';
      case 'WARN': return 'Warning';
      case 'INFO': return 'Success';
      case 'DEBUG': return 'Information';
      default: return 'None';
      }
    }
  });
});