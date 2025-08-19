import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Chip,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Avatar,
  ListItemIcon,
  ListItemText,
  Tooltip,
  LinearProgress
} from '@mui/material';
import {
  Computer as AgentIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RestartIcon,
  Bug as DebugIcon,
  MoreVert as MoreIcon,
  Speed as MetricsIcon,
  Settings as ConfigIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  CheckCircle as HealthyIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

import { useSocket } from '../hooks/useSocket';
import { useApiClient } from '../hooks/useApiClient';

const AgentStatusChip = ({ status }) => {
  const getStatusProps = (status) => {
    switch (status) {
      case 'running':
        return { color: 'success', icon: <HealthyIcon fontSize="small" /> };
      case 'stopped':
        return { color: 'default', icon: <StopIcon fontSize="small" /> };
      case 'error':
        return { color: 'error', icon: <ErrorIcon fontSize="small" /> };
      case 'degraded':
        return { color: 'warning', icon: <WarningIcon fontSize="small" /> };
      default:
        return { color: 'default', icon: <InfoIcon fontSize="small" /> };
    }
  };

  const { color, icon } = getStatusProps(status);

  return (
    <Chip
      label={status}
      color={color}
      size="small"
      icon={icon}
      variant="outlined"
    />
  );
};

const AgentCard = ({ agent, onAction }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleAction = async (action) => {
    setLoading(true);
    await onAction(agent.id, action);
    setLoading(false);
    handleMenuClose();
  };

  const getAgentTypeColor = (type) => {
    const colors = {
      'data-processor': '#2196F3',
      'ai-ml': '#9C27B0',
      'storage': '#4CAF50',
      'orchestrator': '#FF9800',
      'analytics': '#E91E63',
      'custom': '#795548'
    };
    return colors[type] || '#607D8B';
  };

  return (
    <Card sx={{ height: '100%', position: 'relative' }}>
      {loading && (
        <LinearProgress
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 1
          }}
        />
      )}
      
      <CardHeader
        avatar={
          <Avatar
            sx={{
              bgcolor: getAgentTypeColor(agent.type),
              width: 40,
              height: 40
            }}
          >
            <AgentIcon />
          </Avatar>
        }
        title={
          <Typography variant="h6" component="div">
            {agent.name}
          </Typography>
        }
        subheader={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={agent.type}
              size="small"
              variant="outlined"
            />
            <AgentStatusChip status={agent.status} />
          </Box>
        }
        action={
          <IconButton onClick={handleMenuOpen}>
            <MoreIcon />
          </IconButton>
        }
      />
      
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Version
            </Typography>
            <Typography variant="body1">
              {agent.version || 'Unknown'}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Uptime
            </Typography>
            <Typography variant="body1">
              {agent.uptime || 'N/A'}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              CPU
            </Typography>
            <Typography variant="body1">
              {agent.metrics?.cpu ? `${agent.metrics.cpu.toFixed(1)}%` : 'N/A'}
            </Typography>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="body2" color="text.secondary">
              Memory
            </Typography>
            <Typography variant="body1">
              {agent.metrics?.memory ? `${agent.metrics.memory.toFixed(1)}MB` : 'N/A'}
            </Typography>
          </Grid>
        </Grid>

        {agent.capabilities && agent.capabilities.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Capabilities
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {agent.capabilities.map((capability, index) => (
                <Chip
                  key={index}
                  label={capability}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}

        {agent.lastError && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="error" gutterBottom>
              Last Error
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {agent.lastError}
            </Typography>
          </Box>
        )}
      </CardContent>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleAction('restart')}>
          <ListItemIcon>
            <RestartIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Restart</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => handleAction('stop')}>
          <ListItemIcon>
            <StopIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Stop</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => handleAction('debug')}>
          <ListItemIcon>
            <DebugIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Enable Debug</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => handleAction('metrics')}>
          <ListItemIcon>
            <MetricsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>View Metrics</ListItemText>
        </MenuItem>
        
        <MenuItem onClick={() => handleAction('config')}>
          <ListItemIcon>
            <ConfigIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit Config</ListItemText>
        </MenuItem>
      </Menu>
    </Card>
  );
};

export function AgentMonitor() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [viewMode, setViewMode] = useState('grid');

  const socket = useSocket();
  const apiClient = useApiClient();

  useEffect(() => {
    fetchAgents();

    if (socket) {
      socket.emit('subscribe:agents');
      socket.on('agents:updated', handleAgentsUpdate);
      socket.on('agent:status-changed', handleAgentStatusChange);
      socket.on('agent:action-result', handleActionResult);
      socket.on('agent:action-error', handleActionError);

      return () => {
        socket.off('agents:updated', handleAgentsUpdate);
        socket.off('agent:status-changed', handleAgentStatusChange);
        socket.off('agent:action-result', handleActionResult);
        socket.off('agent:action-error', handleActionError);
      };
    }
  }, [socket]);

  const fetchAgents = async () => {
    try {
      setLoading(true);
      const response = await apiClient.get('/api/agents');
      setAgents(response.data);
    } catch (error) {
      console.error('Error fetching agents:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAgentsUpdate = (updatedAgents) => {
    setAgents(updatedAgents);
  };

  const handleAgentStatusChange = (update) => {
    setAgents(prev => prev.map(agent => 
      agent.id === update.agentId 
        ? { ...agent, status: update.status, metrics: update.metrics }
        : agent
    ));
  };

  const handleActionResult = (result) => {
    console.log('Agent action result:', result);
    // Refresh agents to get updated status
    fetchAgents();
  };

  const handleActionError = (error) => {
    console.error('Agent action error:', error);
  };

  const handleAgentAction = async (agentId, action) => {
    try {
      switch (action) {
        case 'restart':
          socket.emit('agent:restart', agentId);
          break;
        case 'stop':
          socket.emit('agent:stop', agentId);
          break;
        case 'debug':
          socket.emit('agent:debug', agentId);
          break;
        case 'metrics':
          // Navigate to metrics view for this agent
          setSelectedAgent(agents.find(a => a.id === agentId));
          setDialogOpen(true);
          break;
        case 'config':
          // Open config editor
          setSelectedAgent(agents.find(a => a.id === agentId));
          setDialogOpen(true);
          break;
        default:
          console.warn('Unknown action:', action);
      }
    } catch (error) {
      console.error('Error executing agent action:', error);
    }
  };

  const getStatusCounts = () => {
    const counts = {
      running: 0,
      stopped: 0,
      error: 0,
      degraded: 0
    };
    
    agents.forEach(agent => {
      counts[agent.status] = (counts[agent.status] || 0) + 1;
    });
    
    return counts;
  };

  const statusCounts = getStatusCounts();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <LinearProgress sx={{ width: '50%' }} />
      </Box>
    );
  }

  return (
    <Box>
      {/* Status Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Agents
              </Typography>
              <Typography variant="h4">
                {agents.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Running
              </Typography>
              <Typography variant="h4" color="success.main">
                {statusCounts.running || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Stopped
              </Typography>
              <Typography variant="h4" color="text.secondary">
                {statusCounts.stopped || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Issues
              </Typography>
              <Typography variant="h4" color="error.main">
                {(statusCounts.error || 0) + (statusCounts.degraded || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5">
          Agents
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            onClick={fetchAgents}
            startIcon={<RestartIcon />}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Agents Grid */}
      {viewMode === 'grid' ? (
        <Grid container spacing={3}>
          {agents.map((agent) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={agent.id}>
              <AgentCard agent={agent} onAction={handleAgentAction} />
            </Grid>
          ))}
        </Grid>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Version</TableCell>
                <TableCell>CPU</TableCell>
                <TableCell>Memory</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {agents.map((agent) => (
                <TableRow key={agent.id}>
                  <TableCell>{agent.name}</TableCell>
                  <TableCell>
                    <Chip label={agent.type} size="small" variant="outlined" />
                  </TableCell>
                  <TableCell>
                    <AgentStatusChip status={agent.status} />
                  </TableCell>
                  <TableCell>{agent.version || 'Unknown'}</TableCell>
                  <TableCell>
                    {agent.metrics?.cpu ? `${agent.metrics.cpu.toFixed(1)}%` : 'N/A'}
                  </TableCell>
                  <TableCell>
                    {agent.metrics?.memory ? `${agent.metrics.memory.toFixed(1)}MB` : 'N/A'}
                  </TableCell>
                  <TableCell>
                    <IconButton
                      onClick={() => handleAgentAction(agent.id, 'restart')}
                      size="small"
                    >
                      <RestartIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Agent Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Agent Details: {selectedAgent?.name}
        </DialogTitle>
        <DialogContent>
          {selectedAgent && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Configuration
              </Typography>
              <TextField
                multiline
                rows={10}
                fullWidth
                value={JSON.stringify(selectedAgent, null, 2)}
                variant="outlined"
                InputProps={{
                  readOnly: true,
                }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default AgentMonitor;