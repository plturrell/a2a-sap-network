import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Grid,
  Paper,
  Tabs,
  Tab,
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Badge,
  Chip,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Computer as AgentsIcon,
  NetworkCheck as NetworkIcon,
  Article as LogsIcon,
  Assessment as MetricsIcon,
  Warning as AlertsIcon,
  Speed as PerformanceIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

import { AgentMonitor } from './AgentMonitor';
import { NetworkVisualizer } from './NetworkVisualizer';
import { LogViewer } from './LogViewer';
import { MetricsDashboard } from './MetricsDashboard';
import { AlertsPanel } from './AlertsPanel';
import { PerformanceProfiler } from './PerformanceProfiler';
import { SystemOverview } from './SystemOverview';

import { useSocket } from '../hooks/useSocket';
import { useApiClient } from '../hooks/useApiClient';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [systemStatus, setSystemStatus] = useState('healthy');
  const [alertCount, setAlertCount] = useState(0);
  const [agentCount, setAgentCount] = useState(0);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const socket = useSocket();
  const apiClient = useApiClient();

  useEffect(() => {
    // Subscribe to real-time updates
    if (socket) {
      socket.on('system:status', handleSystemStatus);
      socket.on('alerts:count', handleAlertCount);
      socket.on('agents:count', handleAgentCount);
      socket.on('notification', handleNotification);

      // Initial data fetch
      fetchInitialData();

      return () => {
        socket.off('system:status', handleSystemStatus);
        socket.off('alerts:count', handleAlertCount);
        socket.off('agents:count', handleAgentCount);
        socket.off('notification', handleNotification);
      };
    }
  }, [socket]);

  const handleSystemStatus = (status) => {
    setSystemStatus(status);
  };

  const handleAlertCount = (count) => {
    setAlertCount(count);
  };

  const handleAgentCount = (count) => {
    setAgentCount(count);
  };

  const handleNotification = (notification) => {
    setSnackbar({
      open: true,
      message: notification.message,
      severity: notification.severity || 'info'
    });
  };

  const fetchInitialData = async () => {
    try {
      const [alertsResponse, agentsResponse] = await Promise.all([
        apiClient.get('/api/alerts'),
        apiClient.get('/api/agents')
      ]);

      setAlertCount(alertsResponse.data.filter(alert => alert.status === 'active').length);
      setAgentCount(agentsResponse.data.length);
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleRefresh = () => {
    window.location.reload();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  const tabs = [
    { label: 'Overview', icon: <DashboardIcon />, component: SystemOverview },
    { label: 'Agents', icon: <AgentsIcon />, component: AgentMonitor },
    { label: 'Network', icon: <NetworkIcon />, component: NetworkVisualizer },
    { label: 'Logs', icon: <LogsIcon />, component: LogViewer },
    { label: 'Metrics', icon: <MetricsIcon />, component: MetricsDashboard },
    { label: 'Alerts', icon: <AlertsIcon />, component: AlertsPanel },
    { label: 'Performance', icon: <PerformanceIcon />, component: PerformanceProfiler }
  ];

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* App Bar */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            A2A Debug Dashboard
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={`${agentCount} Agents`}
              color="primary"
              variant="outlined"
              size="small"
            />
            
            <Badge badgeContent={alertCount} color="error">
              <AlertsIcon />
            </Badge>
            
            <Chip
              label={systemStatus}
              color={getStatusColor(systemStatus)}
              size="small"
            />
            
            <IconButton color="inherit" onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
            
            <IconButton color="inherit">
              <SettingsIcon />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Status Bar */}
      {systemStatus !== 'healthy' && (
        <Alert severity={systemStatus === 'degraded' ? 'warning' : 'error'} sx={{ mb: 2 }}>
          System status: {systemStatus}. Some features may be limited.
        </Alert>
      )}

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 2 }}>
        <Paper sx={{ width: '100%' }}>
          {/* Tab Navigation */}
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            {tabs.map((tab, index) => (
              <Tab
                key={index}
                label={tab.label}
                icon={tab.icon}
                iconPosition="start"
                id={`dashboard-tab-${index}`}
                aria-controls={`dashboard-tabpanel-${index}`}
              />
            ))}
          </Tabs>

          {/* Tab Content */}
          {tabs.map((tab, index) => (
            <TabPanel key={index} value={activeTab} index={index}>
              <tab.component />
            </TabPanel>
          ))}
        </Paper>
      </Container>

      {/* Notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default Dashboard;