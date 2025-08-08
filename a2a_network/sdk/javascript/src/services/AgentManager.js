const { ethers } = require('ethers');
const { A2AError, ErrorCode } = require('../utils/errors');
const { validateAddress, validateAgentParams } = require('../utils/validation');

/**
 * Agent management service for A2A Network
 */
class AgentManager {
    constructor(client) {
        this.client = client;
    }

    /**
     * Register a new agent on the network
     * @param {Object} params - Agent registration parameters
     * @param {string} params.name - Agent name
     * @param {string} params.description - Agent description
     * @param {string} params.endpoint - Agent endpoint URL
     * @param {Object} params.capabilities - Agent capabilities
     * @param {string} [params.metadata] - Optional metadata
     * @returns {Promise<Object>} Registration result
     */
    async register(params) {
        const validation = validateAgentParams(params);
        if (!validation.isValid) {
            throw new A2AError(ErrorCode.INVALID_PARAMS, validation.errors?.join(', '));
        }

        try {
            const contract = this.client.getContract('AgentRegistry');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required for registration');
            }

            // Prepare capabilities array
            const capabilitiesArray = Object.entries(params.capabilities).map(([key, value]) => ({
                name: key,
                enabled: value,
                metadata: ''
            }));

            // Calculate registration fee
            const registrationFee = await contract.getRegistrationFee();

            // Submit registration transaction
            const tx = await contract.registerAgent(
                params.name,
                params.description,
                params.endpoint,
                capabilitiesArray,
                params.metadata || '{}',
                { value: registrationFee }
            );

            const receipt = await tx.wait();
            
            // Extract agent ID from events
            const event = receipt.events?.find(e => e.event === 'AgentRegistered');
            const agentId = event?.args?.agentId;

            if (!agentId) {
                throw new A2AError(ErrorCode.REGISTRATION_FAILED, 'Failed to get agent ID from registration');
            }

            return {
                transactionHash: tx.hash,
                agentId: agentId.toString()
            };

        } catch (error) {
            if (error instanceof A2AError) throw error;
            throw new A2AError(ErrorCode.REGISTRATION_FAILED, error.message);
        }
    }

    /**
     * Update agent information
     * @param {string} agentId - Agent ID
     * @param {Object} params - Update parameters
     * @returns {Promise<Object>} Update result
     */
    async update(agentId, params) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required for update');
            }

            // Verify agent ownership
            const agent = await this.getAgent(agentId);
            const signerAddress = await signer.getAddress();
            
            if (agent.owner.toLowerCase() !== signerAddress.toLowerCase()) {
                throw new A2AError(ErrorCode.UNAUTHORIZED, 'Not authorized to update this agent');
            }

            const tx = await contract.updateAgent(
                agentId,
                params.name || agent.name,
                params.description || agent.description,
                params.endpoint || agent.endpoint,
                params.metadata || agent.metadata
            );

            await tx.wait();

            return { transactionHash: tx.hash };

        } catch (error) {
            if (error instanceof A2AError) throw error;
            throw new A2AError(ErrorCode.UPDATE_FAILED, error.message);
        }
    }

    /**
     * Get agent information by ID
     * @param {string} agentId - Agent ID
     * @returns {Promise<Object>} Agent information
     */
    async getAgent(agentId) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const result = await contract.getAgent(agentId);

            return {
                id: agentId,
                owner: result.owner,
                name: result.name,
                description: result.description,
                endpoint: result.endpoint,
                isActive: result.isActive,
                registrationDate: new Date(result.registrationDate.toNumber() * 1000),
                lastActive: new Date(result.lastActive.toNumber() * 1000),
                messageCount: result.messageCount.toNumber(),
                capabilities: this.parseCapabilities(result.capabilities),
                metadata: result.metadata
            };

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch agent: ${error.message}`);
        }
    }

    /**
     * Get agent profile with reputation and performance metrics
     * @param {string} agentId - Agent ID
     * @returns {Promise<Object>} Agent profile
     */
    async getAgentProfile(agentId) {
        try {
            const [agent, reputation] = await Promise.all([
                this.getAgent(agentId),
                this.client.reputation.getReputation(agentId)
            ]);

            const reputationContract = this.client.getContract('AIAgentMatcher');
            const profileData = await reputationContract.getAIAgentProfile(agent.owner);

            return {
                ...agent,
                reputation: {
                    score: reputation.score,
                    rank: reputation.rank,
                    totalTasks: profileData.totalTasksCompleted.toNumber(),
                    successRate: profileData.taskSuccessRate.toNumber() / 100,
                    avgResponseTime: profileData.avgResponseTime.toNumber(),
                    totalEarnings: ethers.utils.formatEther(profileData.totalEarnings)
                },
                performance: {
                    quality: profileData.performanceMetrics[0],
                    speed: profileData.performanceMetrics[1],
                    reliability: profileData.performanceMetrics[2],
                    innovation: profileData.performanceMetrics[3]
                },
                skills: profileData.skillTags.map(tag => ethers.utils.parseBytes32String(tag))
            };

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch agent profile: ${error.message}`);
        }
    }

    /**
     * Search agents by criteria
     * @param {Object} criteria - Search criteria
     * @param {string[]} [criteria.skills] - Required skills
     * @param {number} [criteria.minReputation] - Minimum reputation
     * @param {number} [criteria.maxResponseTime] - Maximum response time
     * @param {string} [criteria.region] - Geographic region
     * @param {number} [criteria.limit] - Result limit
     * @param {number} [criteria.offset] - Result offset
     * @returns {Promise<Object>} Search results
     */
    async searchAgents(criteria = {}) {
        try {
            const contract = this.client.getContract('AIAgentMatcher');
            
            if (criteria.skills && criteria.skills.length > 0) {
                // Convert skills to bytes32
                const skillBytes = criteria.skills.map(skill => 
                    ethers.utils.formatBytes32String(skill)
                );

                const result = await contract.getTopAgentsBySkills(
                    skillBytes,
                    criteria.limit || 10
                );

                const agents = await Promise.all(
                    result.agents.map(async (address, index) => {
                        try {
                            // Find agent ID by owner address
                            const registryContract = this.client.getContract('AgentRegistry');
                            const agentIds = await registryContract.getAgentsByOwner(address);
                            
                            if (agentIds.length === 0) return null;
                            
                            const agent = await this.getAgent(agentIds[0].toString());
                            return {
                                ...agent,
                                matchScore: result.scores[index].toNumber()
                            };
                        } catch {
                            return null;
                        }
                    })
                );

                const validAgents = agents.filter(agent => agent !== null);

                return {
                    agents: validAgents,
                    total: validAgents.length
                };
            }

            // Fallback to general search
            return await this.getAllAgents(criteria.limit, criteria.offset);

        } catch (error) {
            throw new A2AError(ErrorCode.SEARCH_FAILED, `Failed to search agents: ${error.message}`);
        }
    }

    /**
     * Get all registered agents (paginated)
     * @param {number} [limit=50] - Result limit
     * @param {number} [offset=0] - Result offset
     * @returns {Promise<Object>} All agents
     */
    async getAllAgents(limit = 50, offset = 0) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const totalAgents = await contract.getTotalAgents();
            
            const agentIds = [];
            const end = Math.min(offset + limit, totalAgents.toNumber());
            
            for (let i = offset; i < end; i++) {
                agentIds.push(i.toString());
            }

            const agents = await Promise.all(
                agentIds.map(async (id) => {
                    try {
                        return await this.getAgent(id);
                    } catch {
                        return null;
                    }
                })
            );

            const validAgents = agents.filter(agent => agent !== null);

            return {
                agents: validAgents,
                total: totalAgents.toNumber()
            };

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch agents: ${error.message}`);
        }
    }

    /**
     * Get agents owned by address
     * @param {string} ownerAddress - Owner address
     * @returns {Promise<Object[]>} Owner's agents
     */
    async getAgentsByOwner(ownerAddress) {
        if (!validateAddress(ownerAddress)) {
            throw new A2AError(ErrorCode.INVALID_ADDRESS, 'Invalid owner address');
        }

        try {
            const contract = this.client.getContract('AgentRegistry');
            const agentIds = await contract.getAgentsByOwner(ownerAddress);

            const agents = await Promise.all(
                agentIds.map(async (id) => {
                    try {
                        return await this.getAgent(id.toString());
                    } catch {
                        return null;
                    }
                })
            );

            return agents.filter(agent => agent !== null);

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch owner agents: ${error.message}`);
        }
    }

    /**
     * Update agent status (active/inactive)
     * @param {string} agentId - Agent ID
     * @param {boolean} isActive - Active status
     * @returns {Promise<Object>} Transaction result
     */
    async setStatus(agentId, isActive) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required');
            }

            const tx = await contract.setAgentStatus(agentId, isActive);
            await tx.wait();

            return { transactionHash: tx.hash };

        } catch (error) {
            throw new A2AError(ErrorCode.STATUS_UPDATE_FAILED, error.message);
        }
    }

    /**
     * Deregister agent
     * @param {string} agentId - Agent ID
     * @returns {Promise<Object>} Transaction result
     */
    async deregister(agentId) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const signer = this.client.getSigner();
            
            if (!signer) {
                throw new A2AError(ErrorCode.NO_SIGNER, 'Signer required');
            }

            // Verify ownership
            const agent = await this.getAgent(agentId);
            const signerAddress = await signer.getAddress();
            
            if (agent.owner.toLowerCase() !== signerAddress.toLowerCase()) {
                throw new A2AError(ErrorCode.UNAUTHORIZED, 'Not authorized to deregister this agent');
            }

            const tx = await contract.deregisterAgent(agentId);
            await tx.wait();

            return { transactionHash: tx.hash };

        } catch (error) {
            if (error instanceof A2AError) throw error;
            throw new A2AError(ErrorCode.DEREGISTRATION_FAILED, error.message);
        }
    }

    /**
     * Get agent statistics
     * @param {string} agentId - Agent ID
     * @returns {Promise<Object>} Agent statistics
     */
    async getStatistics(agentId) {
        try {
            const reputationContract = this.client.getContract('AIAgentMatcher');
            const agent = await this.getAgent(agentId);
            const profileData = await reputationContract.getAIAgentProfile(agent.owner);

            return {
                totalMessages: agent.messageCount,
                successfulTasks: profileData.totalTasksCompleted.toNumber(),
                failedTasks: 0, // Calculate from success rate
                avgResponseTime: profileData.avgResponseTime.toNumber(),
                uptime: 99.5, // Calculate from last active
                earnings: ethers.utils.formatEther(profileData.totalEarnings)
            };

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch statistics: ${error.message}`);
        }
    }

    /**
     * Subscribe to agent events
     * @param {Function} callback - Event callback
     * @returns {Promise<string>} Subscription ID
     */
    async subscribeToEvents(callback) {
        return this.client.subscribe('AgentRegistry', '*', callback);
    }

    /**
     * Subscribe to specific agent events
     * @param {string} agentId - Agent ID
     * @param {Function} callback - Event callback
     * @returns {Promise<string>} Subscription ID
     */
    async subscribeToAgent(agentId, callback) {
        return this.client.subscribe('AgentRegistry', 'AgentUpdated', (id, ...args) => {
            if (id === agentId) {
                callback({ type: 'AgentUpdated', agentId: id, data: args });
            }
        });
    }

    /**
     * Parse capabilities from contract format
     * @private
     */
    parseCapabilities(capabilities) {
        const caps = {};
        
        capabilities.forEach((cap) => {
            caps[cap.name] = cap.enabled;
        });

        return caps;
    }

    /**
     * Estimate gas for agent registration
     * @param {Object} params - Registration parameters
     * @returns {Promise<ethers.BigNumber>} Gas estimate
     */
    async estimateRegistrationGas(params) {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const capabilitiesArray = Object.entries(params.capabilities).map(([key, value]) => ({
                name: key,
                enabled: value,
                metadata: ''
            }));

            const registrationFee = await contract.getRegistrationFee();

            return await contract.estimateGas.registerAgent(
                params.name,
                params.description,
                params.endpoint,
                capabilitiesArray,
                params.metadata || '{}',
                { value: registrationFee }
            );

        } catch (error) {
            throw new A2AError(ErrorCode.ESTIMATION_FAILED, `Failed to estimate gas: ${error.message}`);
        }
    }
}

module.exports = AgentManager;