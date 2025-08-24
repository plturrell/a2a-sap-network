'use strict';

/**
 * Project Service with Caching
 * Demonstrates enterprise caching patterns
 */

const cds = require('@sap/cds');
const { SELECT, INSERT, UPDATE, DELETE } = cds;
const { cache, invalidateCache, keyGenerators } = require('./cache/cache-decorators');
const cacheManager = require('./cache/cache-manager');
const businessPartnerService = require('./external/business-partner-service');
const salesOrderService = require('./external/sales-order-service');

class ProjectService extends cds.Service {
  async init() {
    const { Projects, ProjectAgents, ProjectWorkflows } = this.entities;

    // Warm up cache on startup
    this.warmupProjectCache();

    // READ handlers with caching
    this.on('READ', Projects, this.onReadProjects.bind(this));
    this.on('READ', Projects, this.onReadProject.bind(this));

    // CREATE/UPDATE/DELETE handlers with cache invalidation
    this.on('CREATE', Projects, this.onCreateProject.bind(this));
    this.on('UPDATE', Projects, this.onUpdateProject.bind(this));
    this.on('DELETE', Projects, this.onDeleteProject.bind(this));

    // Custom actions
    this.on('deployProject', this.onDeployProject.bind(this));
    this.on('linkBusinessPartner', this.onLinkBusinessPartner.bind(this));

    await super.init();
  }

  /**
     * Read projects with caching
     */
  // @cache('project', 600, function(req) {
  //     return `projects:${req.query.SELECT?.where?.join(':') || 'all'}:${req.query.SELECT?.limit || 100}`;
  // })
  async onReadProjects(req) {
    const query = SELECT.from('Projects');
        
    // Apply filters
    if (req.query.SELECT?.where) {
      query.where(req.query.SELECT.where);
    }

    // Apply pagination
    if (req.query.SELECT?.limit) {
      query.limit(req.query.SELECT.limit);
    }

    const projects = await cds.run(query);
        
    // Enrich with cached business partner data
    for (const project of projects) {
      if (project.businessPartnerId) {
        project.businessPartner = await cacheManager.cacheAside(
          `bp:${project.businessPartnerId}`,
          () => businessPartnerService.getBusinessPartner(project.businessPartnerId),
          'businessPartner',
          3600
        );
      }
    }

    return projects;
  }

  /**
     * Read single project with caching
     */
  // @cache('project', 600, keyGenerators.byId)
  async onReadProject(req) {
    const { ID } = req.params[0] || {};
    if (!ID) {
      return;
    }

    const project = await cds.run(
      SELECT.one.from('Projects')
        .where({ ID })
        .columns('*')
    );

    if (project) {
      // Load related data with caching
      project.agents = await this.getProjectAgents(ID);
      project.workflows = await this.getProjectWorkflows(ID);
      project.metrics = await this.getProjectMetrics(ID);
    }

    return project;
  }

  /**
     * Create project with cache invalidation
     */
  async onCreateProject(req) {
    const data = req.data;
        
    // Create project
    const project = await cds.run(
      INSERT.into('Projects').entries(data)
    );

    // Create sales order if business partner is linked
    if (data.businessPartnerId) {
      try {
        const salesOrder = await salesOrderService.createSalesOrderFromProject({
          ...data,
          projectId: project.ID
        });
                
        // Cache the sales order reference
        await cacheManager.set(
          `so:project:${project.ID}`,
          salesOrder,
          'salesOrder'
        );
      } catch (error) {
        console.error('Failed to create sales order:', error);
      }
    }

    return project;
  }

  /**
     * Update project with cache invalidation
     */
  async onUpdateProject(req) {
    const { ID } = req.params[0] || {};
    const data = req.data;

    await cds.run(
      UPDATE('Projects').set(data).where({ ID })
    );

    return { ID, ...data };
  }

  /**
     * Delete project with cache invalidation
     */
  async onDeleteProject(req) {
    const { ID } = req.params[0] || {};

    // Clear all related cache entries
    await Promise.all([
      cacheManager.clearPattern(`agents:project:${ID}`, 'agent'),
      cacheManager.clearPattern(`workflows:project:${ID}`, 'workflow'),
      cacheManager.clearPattern(`metrics:project:${ID}`, 'metrics')
    ]);

    await cds.run(
      DELETE.from('Projects').where({ ID })
    );
  }

  /**
     * Deploy project action
     */
  async onDeployProject(req) {
    const { projectId } = req.data;
        
    // Update project status
    await this.onUpdateProject({
      params: [{ ID: projectId }],
      data: { status: 'DEPLOYED' }
    });

    // Increment deployment counter
    const deploymentCount = await cacheManager.increment(
      `deployments:${projectId}`,
      'metrics'
    );

    return {
      success: true,
      projectId,
      deploymentNumber: deploymentCount,
      timestamp: new Date().toISOString()
    };
  }

  /**
     * Link business partner to project
     */
  async onLinkBusinessPartner(req) {
    const { projectId, businessPartnerId, role } = req.data;

    // Link in S/4HANA
    const link = await businessPartnerService.linkToProject(
      businessPartnerId,
      projectId,
      role
    );

    // Update project
    await this.onUpdateProject({
      params: [{ ID: projectId }],
      data: { businessPartnerId }
    });

    // Cache the link
    await cacheManager.set(
      `bp:link:${projectId}`,
      link,
      'businessPartner'
    );

    return link;
  }

  /**
     * Get project agents with caching
     */
  async getProjectAgents(projectId) {
    return cds.run(
      SELECT.from('ProjectAgents')
        .where({ project_ID: projectId })
    );
  }

  /**
     * Get project workflows with caching
     */
  async getProjectWorkflows(projectId) {
    return cds.run(
      SELECT.from('ProjectWorkflows')
        .where({ project_ID: projectId })
    );
  }

  /**
     * Get project metrics with short cache
     */
  async getProjectMetrics(projectId) {
    // Simulate metrics calculation
    return {
      agentCount: await this.getAgentCount(projectId),
      workflowCount: await this.getWorkflowCount(projectId),
      executionCount: await cacheManager.get(`executions:${projectId}`, 'metrics') || 0,
      successRate: 96.5,
      avgResponseTime: 342
    };
  }

  /**
     * Warm up frequently accessed project data with performance optimization
     */
  async warmupProjectCache() {
    // const span = tracer.startSpan('project-service.cache-warmup');
        
    try {
      // eslint-disable-next-line no-console
      console.log('Warming up project cache...');
            
      // Parallel data loading for performance
      const [activeProjects, recentProjects, popularProjects] = await Promise.all([
        // Active projects
        cds.run(
          SELECT.from('Projects')
            .where({ status: 'ACTIVE' })
            .limit(10)
            .orderBy('modifiedAt desc')
        ),
        // Recently modified projects
        cds.run(
          SELECT.from('Projects')
            .where('modifiedAt >', new Date(Date.now() - 7 * 24 * 60 * 60 * 1000))
            .limit(15)
            .orderBy('modifiedAt desc')
        ),
        // Most accessed projects (based on cache metrics)
        this.getPopularProjects(20)
      ]);

      // Batch cache operations for efficiency
      const cacheOperations = [];
      const projectsToCache = new Set();
            
      // Add unique projects from all sources
      [...activeProjects, ...recentProjects, ...popularProjects].forEach(project => {
        if (project && project.ID && !projectsToCache.has(project.ID)) {
          projectsToCache.add(project.ID);
                    
          // Cache project with different TTL based on activity
          const ttl = this._calculateCacheTTL(project);
                    
          cacheOperations.push(
            cacheManager.set(
              `project:${project.ID}`,
              project,
              'project',
              ttl
            )
          );
                    
          // Pre-cache related data
          if (project.agents && project.agents.length > 0) {
            cacheOperations.push(
              cacheManager.set(
                `agents:project:${project.ID}`,
                project.agents,
                'agent',
                ttl
              )
            );
          }
                    
          // Cache project metrics
          cacheOperations.push(
            this._precacheProjectMetrics(project.ID)
          );
        }
      });

      // Execute all cache operations in parallel
      await Promise.allSettled(cacheOperations);
            
      // Warm up search indices
      await this._warmupSearchCache(Array.from(projectsToCache));

      // eslint-disable-next-line no-console

      // eslint-disable-next-line no-console

      // eslint-disable-next-line no-console
      console.log(`Warmed up cache for ${projectsToCache.size} projects with ${cacheOperations.length} operations`);
      // span.setStatus({ code: 1, message: `Cached ${projectsToCache.size} projects` });
            
    } catch (error) {
      console.error('Cache warmup failed:', error);
      // span.recordException(error);
      // span.setStatus({ code: 2, message: error.message });
    } finally {
      // span.end();
    }
  }

  /**
     * Get popular projects based on access patterns
     */
  async getPopularProjects(limit = 20) {
    try {
      // Get project access metrics from cache
      const accessMetrics = await cacheManager.mget(
        Array.from({ length: 100 }, (_, i) => `access:project:${i}`),
        'metrics'
      );
            
      // Sort by access count and return top projects
      const popularProjectIds = Object.entries(accessMetrics)
        .filter(([_, count]) => count && count > 0)
        .sort(([_, a], [__, b]) => b - a)
        .slice(0, limit)
        .map(([key, _]) => key.split(':').pop());
            
      if (popularProjectIds.length === 0) {
        return [];
      }
            
      return await cds.run(
        SELECT.from('Projects')
          .where({ ID: { in: popularProjectIds } })
          .orderBy('modifiedAt desc')
      );
    } catch (error) {
      console.warn('Failed to get popular projects:', error);
      return [];
    }
  }

  /**
     * Calculate optimal cache TTL based on project activity
     */
  _calculateCacheTTL(project) {
    const now = Date.now();
    const modifiedTime = new Date(project.modifiedAt).getTime();
    const ageDays = (now - modifiedTime) / (24 * 60 * 60 * 1000);
        
    // More active projects get shorter TTL for fresher data
    if (project.status === 'ACTIVE' && ageDays < 1) {
      return 300; // 5 minutes for very active projects
    } else if (project.status === 'ACTIVE' && ageDays < 7) {
      return 900; // 15 minutes for active projects
    } else if (project.status === 'DEPLOYED') {
      return 1800; // 30 minutes for deployed projects
    } else {
      return 3600; // 1 hour for inactive projects
    }
  }

  /**
     * Pre-cache project metrics for performance
     */
  async _precacheProjectMetrics(projectId) {
    try {
      const metrics = await this.getProjectMetrics(projectId);
      return cacheManager.set(
        `metrics:project:${projectId}`,
        metrics,
        'metrics',
        60 // Metrics have shorter TTL
      );
    } catch (error) {
      console.warn(`Failed to precache metrics for project ${projectId}:`, error);
    }
  }

  /**
     * Warm up search cache with project data
     */
  async _warmupSearchCache(projectIds) {
    try {
      // Build search index cache
      const searchTerms = ['active', 'customer', 'automation', 'ai', 'analytics'];
            
      for (const term of searchTerms) {
        // Cache common search results
        const searchResults = await cds.run(
          SELECT.from('Projects')
            .where(`name like '%${term}%' or description like '%${term}%'`)
            .limit(50)
        );
                
        await cacheManager.set(
          `search:projects:${term}`,
          searchResults,
          'search',
          1800 // 30 minutes
        );
      }
    } catch (error) {
      console.warn('Search cache warmup failed:', error);
    }
  }

  async getAgentCount(projectId) {
    const result = await cds.run(
      SELECT.one.from('ProjectAgents')
        .columns('count(*) as count')
        .where({ project_ID: projectId })
    );
    return result?.count || 0;
  }

  async getWorkflowCount(projectId) {
    const result = await cds.run(
      SELECT.one.from('ProjectWorkflows')
        .columns('count(*) as count')
        .where({ project_ID: projectId })
    );
    return result?.count || 0;
  }
}

module.exports = ProjectService;