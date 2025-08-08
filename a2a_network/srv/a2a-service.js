const cds = require('@sap/cds')
const LOG = cds.log('a2a-service')
const { inspect } = require('util')
const transactionCoordinator = require('./lib/transaction-coordinator')
const BaseService = require('./lib/base-service')

// Internal state symbols following SAP patterns
const $blockchain = Symbol('blockchain')
const $initialized = Symbol('initialized')
const $agents = Symbol('agents')
const $services = Symbol('services')

/**
 * A2A Network Service - Enterprise Agent Coordination
 * 
 * Following SAP internal code patterns and standards
 */
class A2AService extends BaseService {

  async initializeService() {
    const { Agents, Services, Capabilities, Messages, Workflows, WorkflowExecutions } = this.entities

    // Connect to blockchain service
    this[$blockchain] = this.blockchain

    // Input validation handlers
    this.before('CREATE', Agents, req => this._validateAgent(req))
    this.before('UPDATE', Agents, req => this._validateAgentUpdate(req))
    this.before('CREATE', Services, req => this._validateService(req))
    this.before('UPDATE', Services, req => this._validateServiceUpdate(req))

    // Business logic handlers
    this.after('CREATE', Agents, (data, req) => this._afterAgentCreate(data, req))
    this.after('CREATE', Services, (data, req) => this._afterServiceCreate(data, req))
    this.after('UPDATE', Agents, (data, req) => this._afterAgentUpdate(data, req))

    // Action handlers
    this.on('updateReputation', Agents, req => this._updateReputation(req))
    this.on('execute', Workflows, req => this._executeWorkflow(req))
    this.on('getNetworkHealth', req => this._getNetworkHealth(req))
    this.on('searchAgents', req => this._searchAgents(req))
    this.on('matchCapabilities', req => this._matchCapabilities(req))
    this.on('calculateReputation', req => this._calculateReputation(req))
    this.on('deployContract', req => this._deployContract(req))
    this.on('syncBlockchain', req => this._syncBlockchain(req))

    this[$initialized] = true
  }

  _validateAgent(req) {
    this.validation.validateAgent(req.data, req);
    LOG.debug('Agent validation passed', { name: req.data.name });
  }

  async _validateAgentUpdate(req) {
    const { data, params } = req;
    const agentId = params[0];

    // Check if agent exists
    const existing = await SELECT.one.from(this.entities.Agents).where({ ID: agentId });
    if (!existing) {
      req.error(404, 'AGENT_NOT_FOUND', `Agent ${agentId} not found`);
    }

    // Validate update data
    this.validation.validateAgentUpdate(data, req);
  }

  async _validateService(req) {
    const { data } = req;

    // Use shared validation
    this.validation.validateService(data, req);

    // Additional business logic validation
    if (data.provider_ID) {
      const provider = await SELECT.one.from(this.entities.Agents).where({ ID: data.provider_ID });
      if (!provider) {
        req.error(400, 'PROVIDER_NOT_FOUND', `Provider agent ${data.provider_ID} not found`);
      }
    }
  }

  async _validateServiceUpdate(req) {
    const { data, params } = req;
    const serviceId = params[0];

    // Check if service exists
    const existing = await SELECT.one.from(this.entities.Services).where({ ID: serviceId });
    if (!existing) {
      req.error(404, 'SERVICE_NOT_FOUND', `Service ${serviceId} not found`);
    }

    // Validate update data
    this.validation.validateServiceUpdate(data, req);
  }

  async _afterAgentCreate(data, req) {
    try {
      // Auto-register on blockchain if requested using distributed transaction
      if (req.data.registerOnBlockchain && this[$blockchain]) {
        const result = await transactionCoordinator.executeSaga('agent-registration', {
          agentId: data.ID,
          name: data.name,
          endpoint: data.endpoint,
          capabilities: data.capabilities
        }, {
          correlationId: req.id || 'agent-create-' + data.ID
        })
        
        LOG.info('Agent registration saga completed', { 
          agentId: data.ID, 
          transactionId: result.transactionId 
        })
      }

      // Emit business event
      await this.emit('AgentCreated', {
        agentId: data.ID,
        name: data.name,
        address: data.address,
        timestamp: new Date().toISOString()
      })

    } catch (error) {
      LOG.error('Post-creation processing failed', { agentId: data.ID, error: error.message })
      // Don't fail the transaction - agent was created successfully
    }
  }

  async _afterServiceCreate(data, req) {
    try {
      // Auto-list on marketplace if requested
      if (req.data.listOnMarketplace && this[$blockchain]) {
        const txHash = await this._listServiceOnMarketplace(data)
        LOG.info('Service listed on marketplace', { serviceId: data.ID, txHash })
      }

      // Emit business event
      await this.emit('ServiceCreated', {
        serviceId: data.ID,
        name: data.name,
        providerId: data.provider_ID,
        timestamp: new Date().toISOString()
      })

    } catch (error) {
      LOG.error('Post-creation processing failed', { serviceId: data.ID, error: error.message })
    }
  }

  async _afterAgentUpdate(data, req) {
    try {
      await this.emit('AgentUpdated', {
        agentId: data.ID,
        changes: Object.keys(req.data),
        timestamp: new Date().toISOString()
      })
    } catch (error) {
      LOG.error('Post-update processing failed', { agentId: data.ID, error: error.message })
    }
  }

  async _updateReputation(req) {
    const { ID } = req.params[0]
    const { score } = req.data

    if (typeof score !== 'number' || score < 0 || score > 200) {
      req.error(400, 'INVALID_REPUTATION_SCORE', 'Reputation score must be between 0 and 200')
    }

    const agent = await SELECT.one.from(this.entities.Agents).where({ ID })
    if (!agent) {
      req.error(404, 'AGENT_NOT_FOUND', `Agent ${ID} not found`)
    }

    const oldScore = agent.reputation

    // Update in database
    await UPDATE(this.entities.Agents).set({ reputation: score }).where({ ID })

    // Emit business event
    await this.emit('ReputationUpdated', {
      agentId: ID,
      oldScore,
      newScore: score,
      reason: 'Manual update',
      timestamp: new Date().toISOString()
    })

    LOG.info('Reputation updated', { agentId: ID, oldScore, newScore: score })

    return {
      success: true,
      agentId: ID,
      oldScore,
      newScore: score
    }
  }

  async _executeWorkflow(req) {
    const span = tracing.startSpan('execute-workflow')
    
    try {
      const { ID } = req.params[0]
      const { parameters } = req.data

      span.setTags({
        'workflow.id': ID,
        'parameters.count': Object.keys(parameters || {}).length
      })

      const workflow = await SELECT.one.from(this.entities.Workflows).where({ ID })
      if (!workflow) {
        span.setStatus('ERROR')
        req.error(404, 'WORKFLOW_NOT_FOUND', `Workflow ${ID} not found`)
      }

      // Validate workflow definition
      let definition
      try {
        definition = JSON.parse(workflow.definition)
        span.setTag('workflow.steps', definition.steps?.length || 0)
      } catch (error) {
        span.logError(error)
        req.error(400, 'INVALID_WORKFLOW_DEFINITION', 'Invalid workflow definition JSON')
      }

      // Create execution record
      const execution = await INSERT.into(this.entities.WorkflowExecutions).entries({
        workflow_ID: ID,
        status: 'running',
        startedAt: new Date(),
        parameters: JSON.stringify(parameters || {})
      })

      span.setTag('execution.id', execution.ID)
      LOG.info('Workflow execution started', { workflowId: ID, executionId: execution.ID })

      // Execute asynchronously
      setImmediate(() => {
        this._executeWorkflowAsync(execution.ID, definition, parameters || {})
          .catch(error => LOG.error('Workflow execution failed', { executionId: execution.ID, error: error.message }))
      })

      span.finish()
      return execution.ID
      
    } catch (error) {
      span.logError(error)
      span.finish()
      throw error
    }
  }

  async _getNetworkHealth() {
    try {
      const stats = await SELECT.one.from('NetworkStats').orderBy('validFrom desc')

      const totalAgents = stats?.totalAgents || 0
      const activeAgents = stats?.activeAgents || 0
      const activeRatio = totalAgents > 0 ? activeAgents / totalAgents : 0

      const pendingMessages = await SELECT.count.from(this.entities.Messages).where({ status: 'pending' })
      const failedMessages = await SELECT.count.from(this.entities.Messages).where({ status: 'failed' })
      const totalMessages = await SELECT.count.from(this.entities.Messages)
      const failureRate = totalMessages > 0 ? failedMessages / totalMessages : 0

      let status = 'healthy'
      if (activeRatio < 0.5 || failureRate > 0.1) status = 'degraded'
      if (activeRatio < 0.2 || failureRate > 0.3) status = 'critical'

      const healthData = {
        status,
        metrics: {
          activeAgentRatio: activeRatio,
          avgTransactionTime: stats?.avgTransactionTime || 0,
          pendingMessages,
          failureRate
        },
        timestamp: new Date().toISOString()
      }

      LOG.debug('Network health calculated', healthData)
      return JSON.stringify(healthData)

    } catch (error) {
      LOG.error('Health calculation failed', error)
      req.error(503, 'HEALTH_CALCULATION_FAILED', 'Failed to calculate network health')
    }
  }

  async _searchAgents(req) {
    const span = tracing.startSpan('search-agents')
    
    try {
      const { capabilities, minReputation, maxPrice } = req.data

      span.setTags({
        'search.capabilities': capabilities?.length || 0,
        'search.minReputation': minReputation || 0,
        'search.maxPrice': maxPrice || 'unlimited'
      })

      let query = SELECT.from(this.entities.Agents).where({ isActive: true })

      if (minReputation) {
        query = query.and({ reputation: { '>=': minReputation } })
      }

      const agents = await query
      span.setTag('agents.found', agents.length)

      // Filter by capabilities if specified
      if (capabilities?.length > 0) {
        const filtered = await this._filterAgentsByCapabilities(agents, capabilities)
        span.setTag('agents.filtered', filtered.length)
        span.finish()
        return JSON.stringify(filtered.map(a => a.ID))
      }

      span.finish()
      return JSON.stringify(agents.map(a => a.ID))

    } catch (error) {
      span.logError(error)
      span.finish()
      LOG.error('Agent search failed', { error: error.message, criteria: req.data })
      return JSON.stringify([])
    }
  }

  async _matchCapabilities(req) {
    const span = tracing.startSpan('match-capabilities')
    
    try {
      const { requirements } = req.data

      if (!requirements?.length) {
        span.setTag('requirements.empty', true)
        span.finish()
        return JSON.stringify([])
      }

      span.setTags({
        'requirements.count': requirements.length,
        'requirements.list': requirements.join(',')
      })

      // Use blockchain capability matcher if available
      if (this[$blockchain]) {
        const instrumentation = tracing.instrumentServiceCall('BlockchainService', 'matchCapabilities')
        try {
          const matches = await this[$blockchain].send('matchCapabilities', { requirements })
          instrumentation.finish(matches)
          span.setTag('matches.count', matches?.length || 0)
          span.finish()
          return JSON.stringify(matches)
        } catch (error) {
          instrumentation.finish(null, error)
          throw error
        }
      }

      // Fallback to local matching
      span.setTag('matching.method', 'local')
      const result = await this._localCapabilityMatching(requirements)
      span.finish()
      return result

    } catch (error) {
      span.logError(error)
      span.finish()
      LOG.error('Capability matching failed', { error: error.message, requirements: req.data.requirements })
      return JSON.stringify([])
    }
  }

  async _calculateReputation(req) {
    const { agentAddress } = req.data

    if (!agentAddress) {
      req.error(400, 'AGENT_ADDRESS_REQUIRED', 'Agent address is required')
    }

    try {
      // Get reputation from blockchain if available
      if (this[$blockchain]) {
        const reputation = await this[$blockchain].send('calculateReputation', { agentAddress })
        return JSON.stringify(reputation)
      }

      // Fallback to local calculation
      const agent = await SELECT.one.from(this.entities.Agents).where({ address: agentAddress })
      return JSON.stringify({
        reputationScore: agent?.reputation || 100,
        trustScore: 1.0,
        source: 'database'
      })

    } catch (error) {
      LOG.error('Reputation calculation failed', { agentAddress, error: error.message })
      req.error(503, 'REPUTATION_CALCULATION_FAILED', 'Failed to calculate reputation')
    }
  }

  async _deployContract(req) {
    const { contractType, parameters } = req.data

    if (!this[$blockchain]) {
      req.error(503, 'BLOCKCHAIN_SERVICE_UNAVAILABLE', 'Blockchain service not available')
    }

    try {
      const result = await this[$blockchain].send('deployContract', {
        contractType,
        parameters
      })

      LOG.info('Contract deployed', { contractType, result })

      return JSON.stringify({
        address: result,
        transactionHash: result,
        gasUsed: 500000,
        contractType
      })

    } catch (error) {
      LOG.error('Contract deployment failed', { contractType, error: error.message })
      req.error(503, 'CONTRACT_DEPLOYMENT_FAILED', 'Failed to deploy contract')
    }
  }

  async _syncBlockchain() {
    if (!this[$blockchain]) {
      req.error(503, 'BLOCKCHAIN_SERVICE_UNAVAILABLE', 'Blockchain service not available')
    }

    try {
      const result = await this[$blockchain].send('syncBlockchain')
      LOG.info('Blockchain sync completed', result)
      return JSON.stringify(result)

    } catch (error) {
      LOG.error('Blockchain sync failed', error)
      return JSON.stringify({
        synced: 0,
        pending: 0,
        failed: 1,
        error: error.message
      })
    }
  }

  // Private helper methods following SAP patterns

  async _registerAgentOnBlockchain(agent) {
    if (!this[$blockchain]) return null

    try {
      return await this[$blockchain].send('registerAgent', {
        agentId: agent.ID,
        address: agent.address,
        name: agent.name,
        endpoint: agent.endpoint || ''
      })
    } catch (error) {
      LOG.error('Blockchain agent registration failed', { agentId: agent.ID, error: error.message })
      throw error
    }
  }

  async _listServiceOnMarketplace(service) {
    if (!this[$blockchain]) return null

    try {
      return await this[$blockchain].send('listService', {
        serviceId: service.ID,
        name: service.name,
        description: service.description || '',
        pricePerCall: service.pricePerCall,
        minReputation: service.minReputation || 0
      })
    } catch (error) {
      LOG.error('Blockchain service listing failed', { serviceId: service.ID, error: error.message })
      throw error
    }
  }

  async _executeWorkflowAsync(executionId, definition, parameters) {
    try {
      const steps = definition.steps || []

      for (let i = 0; i < steps.length; i++) {
        const step = steps[i]

        const stepRecord = await INSERT.into(this.entities.WorkflowSteps || 'WorkflowSteps').entries({
          execution_ID: executionId,
          stepNumber: i + 1,
          agentAddress: step.agent,
          action: step.action,
          input: JSON.stringify(step.input || {}),
          status: 'running',
          startedAt: new Date()
        })

        try {
          const result = await this._executeWorkflowStep(step, parameters)

          await UPDATE(this.entities.WorkflowSteps || 'WorkflowSteps').set({
            output: JSON.stringify(result),
            status: 'completed',
            completedAt: new Date()
          }).where({ ID: stepRecord.ID })

          parameters = { ...parameters, [`step${i}`]: result }

        } catch (error) {
          await UPDATE(this.entities.WorkflowSteps || 'WorkflowSteps').set({
            status: 'failed',
            output: JSON.stringify({ error: error.message }),
            completedAt: new Date()
          }).where({ ID: stepRecord.ID })

          throw error
        }
      }

      // Mark execution as completed
      await UPDATE(this.entities.WorkflowExecutions).set({
        status: 'completed',
        completedAt: new Date()
      }).where({ ID: executionId })

      await this.emit('WorkflowCompleted', {
        executionId,
        status: 'completed',
        timestamp: new Date().toISOString()
      })

    } catch (error) {
      await UPDATE(this.entities.WorkflowExecutions).set({
        status: 'failed',
        error: error.message,
        completedAt: new Date()
      }).where({ ID: executionId })

      LOG.error('Workflow execution failed', { executionId, error: error.message })
    }
  }

  async _executeWorkflowStep(step, parameters) {
    const agent = await SELECT.one.from(this.entities.Agents).where({ address: step.agent })

    if (!agent) {
      throw new Error(`Agent ${step.agent} not found`)
    }

    if (!agent.isActive) {
      throw new Error(`Agent ${agent.name} is not active`)
    }

    // Execute step based on agent endpoint
    if (agent.endpoint) {
      return await this._executeRemoteStep(agent, step, parameters)
    }

    // No endpoint available - this is a real business constraint
    throw new Error(`Agent ${agent.name} has no endpoint configured`)
  }

  async _executeRemoteStep(agent, step, parameters) {
    const span = tracing.startSpan('execute-remote-step')
    
    try {
      span.setTags({
        'agent.name': agent.name,
        'agent.endpoint': agent.endpoint,
        'step.action': step.action,
        'http.method': 'POST'
      })

      const fetch = (await import('node-fetch')).default
      const headers = {
        'Content-Type': 'application/json',
        ...tracing.injectTraceHeaders()
      }
      
      const response = await fetch(agent.endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          action: step.action,
          input: step.input,
          parameters: parameters
        }),
        timeout: 30000
      })

      span.setTag('http.status_code', response.status)

      if (!response.ok) {
        span.setStatus('ERROR')
        throw new Error(`Agent returned ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      span.setTag('response.success', true)
      span.finish()
      
      return {
        success: true,
        data: result,
        timestamp: new Date().toISOString(),
        agent: agent.name
      }

    } catch (error) {
      span.logError(error)
      span.finish()
      LOG.error('Remote step execution failed', {
        agent: agent.name,
        step: step.action,
        error: error.message
      })
      throw error
    }
  }

  async _filterAgentsByCapabilities(agents, capabilities) {
    const filtered = []

    for (const agent of agents) {
      const agentCaps = await SELECT.from('AgentCapabilities')
        .where({ agent_ID: agent.ID })
        .columns('capability_ID')

      const capNames = await SELECT.from(this.entities.Capabilities)
        .where({ ID: { in: agentCaps.map(c => c.capability_ID) } })
        .columns('name')

      const hasAllCaps = capabilities.every(cap =>
        capNames.some(c => c.name === cap)
      )

      if (hasAllCaps) {
        filtered.push(agent)
      }
    }

    return filtered
  }

  async _localCapabilityMatching(requirements) {
    try {
      const capabilities = await SELECT.from(this.entities.Capabilities)
        .where({ name: { in: requirements } })

      if (capabilities.length === 0) {
        return JSON.stringify([])
      }

      const capabilityIds = capabilities.map(c => c.ID)

      const agentCaps = await SELECT.from('AgentCapabilities')
        .where({ capability_ID: { in: capabilityIds } })

      const agentCapCount = {}
      agentCaps.forEach(ac => {
        agentCapCount[ac.agent_ID] = (agentCapCount[ac.agent_ID] || 0) + 1
      })

      const matchingAgentIds = Object.keys(agentCapCount)
        .filter(agentId => agentCapCount[agentId] === requirements.length)

      if (matchingAgentIds.length === 0) {
        return JSON.stringify([])
      }

      const agents = await SELECT.from(this.entities.Agents)
        .where({ ID: { in: matchingAgentIds }, isActive: true })
        .orderBy({ reputation: 'desc' })

      return JSON.stringify(agents.map(a => a.ID))

    } catch (error) {
      LOG.error('Local capability matching failed', error)
      return JSON.stringify([])
    }
  }

  // Utility methods following SAP patterns

  _isValidEthereumAddress(address) {
    return /^0x[a-fA-F0-9]{40}$/.test(address)
  }

  _isValidURL(url) {
    try {
      new URL(url)
      return true
    } catch {
      return false
    }
  }
}

module.exports = A2AService