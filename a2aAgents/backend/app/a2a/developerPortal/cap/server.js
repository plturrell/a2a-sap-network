'use strict';

const cds = require('@sap/cds');
const express = require('express');
const path = require('path');

/**
 * CAP Server with UI5 Static File Serving
 * Serves both OData services and UI5 application
 */

// Configure static file serving for UI5 application
cds.on('bootstrap', (app) => {
  // Serve UI5 static files from the proper CAP app directory
  const staticPath = path.join(__dirname, 'app', 'a2a.portal');
  // eslint-disable-next-line no-console
  // eslint-disable-next-line no-console
  console.log('Serving UI5 static files from CAP app directory:', staticPath);

  // Serve static files with proper MIME types and CORS headers
  app.use('/', express.static(staticPath, {
    setHeaders: (res, path) => {
      // Set CORS headers
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
            
      // Set proper MIME types
      if (path.endsWith('.js')) {
        res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      } else if (path.endsWith('.json')) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
      } else if (path.endsWith('.css')) {
        res.setHeader('Content-Type', 'text/css; charset=utf-8');
      } else if (path.endsWith('.html')) {
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
      } else if (path.endsWith('.properties')) {
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      } else if (path.endsWith('.xml')) {
        res.setHeader('Content-Type', 'application/xml; charset=utf-8');
      }
            
      // Disable X-Content-Type-Options for development
      res.removeHeader('X-Content-Type-Options');
    }
  }));

  // Special handling for a2a.portal namespace resources
  app.get('/a2a/portal/*', (req, res) => {
    const resourcePath = req.path.replace('/a2a/portal/', '');
    const filePath = path.join(staticPath, resourcePath);
        
    // Set proper headers
    if (resourcePath.endsWith('.js')) {
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    } else if (resourcePath.endsWith('.json')) {
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
    }
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    res.sendFile(filePath);
  });

  // SAP Flex Services - Handle UI5 flexibility services
  app.get('/sap/bc/lrep/flex/data/:appId', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Return empty flex data for development
    res.json({
      changes: [],
      contexts: [],
      settings: {
        isKeyUser: false,
        isAtoAvailable: false,
        isProductiveSystem: false
      }
    });
  });

  app.get('/sap/bc/lrep/flex/settings', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Return flex settings for development
    res.json({
      isKeyUser: false,
      isAtoAvailable: false,
      isProductiveSystem: false,
      versioning: {
        CUSTOMER: false,
        VENDOR: false
      }
    });
  });

  // API endpoints for projects
  app.get('/api/projects', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Return sample projects data
    res.json({
      projects: [
        {
          id: 'project-1',
          name: 'Sample A2A Project',
          description: 'A sample project for demonstration',
          status: 'active',
          lastModified: new Date().toISOString(),
          agents: 3,
          workflows: 2
        },
        {
          id: 'project-2',
          name: 'Enterprise Integration',
          description: 'Enterprise system integration project',
          status: 'development',
          lastModified: new Date().toISOString(),
          agents: 5,
          workflows: 4
        }
      ],
      total: 2
    });
  });

  // Handle OPTIONS requests for CORS
  app.options('*', (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.sendStatus(200);
  });

  // Authentication Session API endpoints
  app.get('/api/auth/sessions', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Return sample session data for development
    res.json({
      sessions: [
        {
          id: 'session-1',
          deviceType: 'Desktop',
          browser: 'Chrome 120.0',
          location: 'San Francisco, CA',
          loginTime: new Date(Date.now() - 3600000).toISOString(),
          lastActivity: new Date(Date.now() - 300000).toISOString(),
          ipAddress: '192.168.1.100',
          current: true
        },
        {
          id: 'session-2',
          deviceType: 'Mobile',
          browser: 'Safari 17.0',
          location: 'New York, NY',
          loginTime: new Date(Date.now() - 7200000).toISOString(),
          lastActivity: new Date(Date.now() - 1800000).toISOString(),
          ipAddress: '10.0.0.50',
          current: false
        }
      ],
      statistics: {
        totalSessions: 2,
        activeSessions: 1,
        totalLogins: 15,
        lastLogin: new Date(Date.now() - 3600000).toISOString()
      }
    });
  });

  app.delete('/api/auth/sessions/:sessionId', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    const sessionId = req.params.sessionId;
    // Simulate session termination
    res.json({
      success: true,
      message: `Session ${sessionId} terminated successfully`,
      sessionId: sessionId
    });
  });

  app.post('/api/auth/sessions/terminate-all', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
    // Simulate terminating all sessions except current
    res.json({
      success: true,
      message: 'All other sessions terminated successfully',
      terminatedCount: 1,
      currentSessionPreserved: true
    });
  });

  // Notification API endpoints
  app.get('/api/notifications', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    // Return sample notifications data
    const notifications = [
      {
        id: 'notif_000001',
        user_id: 'DEV_USER_001',
        title: 'Welcome to A2A Agents',
        message: 'Your developer portal is ready! Explore projects, build agents, and deploy workflows.',
        type: 'success',
        priority: 'medium',
        status: 'unread',
        created_at: new Date(Date.now() - 3600000).toISOString(),
        updated_at: new Date(Date.now() - 3600000).toISOString(),
        source: 'system',
        category: 'welcome',
        metadata: {},
        actions: [
          {
            id: 'explore_projects',
            label: 'Explore Projects',
            action_type: 'navigate',
            target: '#/projects',
            style: 'primary'
          }
        ]
      },
      {
        id: 'notif_000002',
        user_id: 'DEV_USER_001',
        title: 'Project Build Complete',
        message: "Your 'Enterprise Integration' project has been successfully built and is ready for deployment.",
        type: 'success',
        priority: 'high',
        status: 'unread',
        created_at: new Date(Date.now() - 1800000).toISOString(),
        updated_at: new Date(Date.now() - 1800000).toISOString(),
        source: 'build_system',
        category: 'project',
        metadata: { project_id: 'project-2', build_id: 'build-123' },
        actions: [
          {
            id: 'view_project',
            label: 'View Project',
            action_type: 'navigate',
            target: '#/projects/project-2',
            style: 'primary'
          },
          {
            id: 'deploy_now',
            label: 'Deploy Now',
            action_type: 'api_call',
            target: '/api/projects/project-2/deploy',
            style: 'success'
          }
        ]
      },
      {
        id: 'notif_000003',
        user_id: 'DEV_USER_001',
        title: 'Security Alert',
        message: 'Unusual login activity detected from a new location. Please verify this was you.',
        type: 'warning',
        priority: 'high',
        status: 'unread',
        created_at: new Date(Date.now() - 900000).toISOString(),
        updated_at: new Date(Date.now() - 900000).toISOString(),
        source: 'security_monitor',
        category: 'security',
        metadata: { login_location: 'New York, NY', ip_address: '192.168.1.200' },
        actions: [
          {
            id: 'verify_login',
            label: 'Verify Login',
            action_type: 'navigate',
            target: '#/profile',
            style: 'warning'
          }
        ]
      },
      {
        id: 'notif_000004',
        user_id: 'DEV_USER_001',
        title: 'Agent Training Complete',
        message: 'Your data standardization agent has completed training with 94% accuracy.',
        type: 'info',
        priority: 'medium',
        status: 'read',
        created_at: new Date(Date.now() - 7200000).toISOString(),
        updated_at: new Date(Date.now() - 3600000).toISOString(),
        read_at: new Date(Date.now() - 3600000).toISOString(),
        source: 'agent_trainer',
        category: 'agent',
        metadata: { agent_id: 'agent-001', accuracy: 0.94 },
        actions: [
          {
            id: 'view_metrics',
            label: 'View Metrics',
            action_type: 'navigate',
            target: '#/agents/agent-001/metrics',
            style: 'primary'
          }
        ]
      },
      {
        id: 'notif_000005',
        user_id: 'DEV_USER_001',
        title: 'Scheduled Maintenance',
        message: 'System maintenance is scheduled for tonight at 2:00 AM UTC. Services may be temporarily unavailable.',
        type: 'system',
        priority: 'low',
        status: 'unread',
        created_at: new Date(Date.now() - 300000).toISOString(),
        updated_at: new Date(Date.now() - 300000).toISOString(),
        expires_at: new Date(Date.now() + 86400000).toISOString(),
        source: 'system_admin',
        category: 'maintenance',
        metadata: { maintenance_window: '2025-08-08T02:00:00Z' },
        actions: []
      }
    ];
        
    // Filter and paginate based on query parameters
    const status = req.query.status;
    const type = req.query.type;
    const limit = parseInt(req.query.limit) || 20;
    const offset = parseInt(req.query.offset) || 0;
        
    let filteredNotifications = notifications;
        
    if (status) {
      filteredNotifications = filteredNotifications.filter(n => n.status === status);
    }
        
    if (type) {
      filteredNotifications = filteredNotifications.filter(n => n.type === type);
    }
        
    const paginatedNotifications = filteredNotifications.slice(offset, offset + limit);
    const unreadCount = notifications.filter(n => n.status === 'unread').length;
        
    res.json({
      notifications: paginatedNotifications,
      total: filteredNotifications.length,
      unread_count: unreadCount,
      has_more: filteredNotifications.length > offset + limit
    });
  });
    
  app.get('/api/notifications/stats', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    // Return notification statistics
    res.json({
      total: 5,
      unread: 4,
      read: 1,
      dismissed: 0,
      critical: 0,
      high: 2,
      medium: 2,
      low: 1
    });
  });
    
  app.patch('/api/notifications/:id/read', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    const notificationId = req.params.id;
    res.json({
      success: true,
      message: `Notification ${notificationId} marked as read`
    });
  });
    
  app.patch('/api/notifications/:id/dismiss', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    const notificationId = req.params.id;
    res.json({
      success: true,
      message: `Notification ${notificationId} dismissed`
    });
  });
    
  app.post('/api/notifications/mark-all-read', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    res.json({
      success: true,
      message: 'All notifications marked as read',
      processed_count: 4
    });
  });
    
  app.delete('/api/notifications/:id', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Access-Control-Allow-Origin', '*');
        
    const notificationId = req.params.id;
    res.json({
      success: true,
      message: `Notification ${notificationId} deleted`
    });
  });

  // Serve index.html at root
  app.get('/', (req, res) => {
    res.sendFile(path.join(staticPath, 'index.html'));
  });
    
  // Explicit routes for critical UI5 resources
  app.get('/model/models.js', (req, res) => {
    res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.sendFile(path.join(staticPath, 'model', 'models.js'));
  });
    
  app.get('/Component.js', (req, res) => {
    res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.sendFile(path.join(staticPath, 'Component.js'));
  });
    
  // Serve manifest.json
  app.get('/manifest.json', (req, res) => {
    res.sendFile(path.join(staticPath, 'manifest.json'));
  });
    
  // Serve all controller files
  app.get('/controller/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'controller', req.params.file));
  });
    
  // Serve all view files
  app.get('/view/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'view', req.params.file));
  });
    
  // Serve all view fragments
  app.get('/view/fragments/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'view', 'fragments', req.params.file));
  });
    
  // Serve all service files
  app.get('/services/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'services', req.params.file));
  });
    
  // Serve all i18n files
  app.get('/i18n/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'i18n', req.params.file));
  });
    
  // Serve all CSS files
  app.get('/css/:file', (req, res) => {
    res.sendFile(path.join(staticPath, 'css', req.params.file));
  });
    
  // Serve launchpad files
  app.get('/launchpad/:path(*)', (req, res) => {
    res.sendFile(path.join(staticPath, 'launchpad', req.params.path));
  });
    
     
    
  // eslint-disable-next-line no-console
    
     
    
  // eslint-disable-next-line no-console
  console.log('UI5 static file routes configured successfully');
});

// Start the CAP server
module.exports = cds.server;
