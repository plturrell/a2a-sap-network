import { ethers } from 'ethers';
import { A2AClient } from '../client/A2AClient';
import { 
    Agent, 
    AgentProfile, 
    AgentStatus, 
    AgentCapabilities,
    AgentRegistrationParams,
    AgentUpdateParams
} from '../types/Agent';
import { A2AError, ErrorCode } from '../utils/errors';
import { validateAddress, validateAgentParams } from '../utils/validation';

/**
 * Agent management service for A2A Network
 */
export class AgentManager {
    constructor(private client: A2AClient) {}

    /**
     * Register a new agent on the network
     */
    async register(params: AgentRegistrationParams): Promise<{ transactionHash: string; agentId: string }> {
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
            const event = receipt.events?.find((e: any) => e.event === 'AgentRegistered');
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
     */
    async update(agentId: string, params: AgentUpdateParams): Promise<{ transactionHash: string }> {
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
     */
    async getAgent(agentId: string): Promise<Agent> {
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
     */
    async getAgentProfile(agentId: string): Promise<AgentProfile> {
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
                    successRate: profileData.taskSuccessRate.toNumber() / 100, // Convert from basis points
                    avgResponseTime: profileData.avgResponseTime.toNumber(),
                    totalEarnings: ethers.utils.formatEther(profileData.totalEarnings)
                },
                performance: {
                    quality: profileData.performanceMetrics[0],
                    speed: profileData.performanceMetrics[1],
                    reliability: profileData.performanceMetrics[2],
                    innovation: profileData.performanceMetrics[3]
                },
                skills: profileData.skillTags.map((tag: any) => ethers.utils.parseBytes32String(tag))
            };

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch agent profile: ${error.message}`);
        }
    }

    /**
     * Search agents by criteria
     */
    async searchAgents(criteria: {
        skills?: string[];
        minReputation?: number;
        maxResponseTime?: number;
        region?: string;
        limit?: number;
        offset?: number;
    }): Promise<{ agents: Agent[]; total: number }> {
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
                    result.agents.map(async (address: string, index: number) => {
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

                const validAgents = agents.filter((agent): agent is Agent => agent !== null);

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
     */
    async getAllAgents(limit = 50, offset = 0): Promise<{ agents: Agent[]; total: number }> {
        try {
            const contract = this.client.getContract('AgentRegistry');
            const totalAgents = await contract.getTotalAgents();
            
            const agentIds: string[] = [];
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

            const validAgents = agents.filter((agent): agent is Agent => agent !== null);

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
     */
    async getAgentsByOwner(ownerAddress: string): Promise<Agent[]> {
        if (!validateAddress(ownerAddress)) {
            throw new A2AError(ErrorCode.INVALID_ADDRESS, 'Invalid owner address');
        }

        try {
            const contract = this.client.getContract('AgentRegistry');
            const agentIds = await contract.getAgentsByOwner(ownerAddress);

            const agents = await Promise.all(
                agentIds.map(async (id: ethers.BigNumber) => {
                    try {
                        return await this.getAgent(id.toString());
                    } catch {
                        return null;
                    }
                })
            );

            return agents.filter((agent): agent is Agent => agent !== null);

        } catch (error) {
            throw new A2AError(ErrorCode.FETCH_FAILED, `Failed to fetch owner agents: ${error.message}`);
        }
    }

    /**
     * Update agent status (active/inactive)
     */
    async setStatus(agentId: string, isActive: boolean): Promise<{ transactionHash: string }> {
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
     */
    async deregister(agentId: string): Promise<{ transactionHash: string }> {
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
     */
    async getStatistics(agentId: string): Promise<{
        totalMessages: number;
        successfulTasks: number;
        failedTasks: number;
        avgResponseTime: number;
        uptime: number;
        earnings: string;
    }> {
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
     */
    async subscribeToEvents(callback: (event: any) => void): Promise<string> {
        return this.client.subscribe('AgentRegistry', '*', callback);
    }

    /**
     * Subscribe to specific agent events
     */
    async subscribeToAgent(agentId: string, callback: (event: any) => void): Promise<string> {
        return this.client.subscribe('AgentRegistry', 'AgentUpdated', (id: string, ...args: any[]) => {
            if (id === agentId) {
                callback({ type: 'AgentUpdated', agentId: id, data: args });
            }
        });
    }

    /**
     * Parse capabilities from contract format
     */
    private parseCapabilities(capabilities: any[]): AgentCapabilities {
        const caps: AgentCapabilities = {};
        
        capabilities.forEach((cap: any) => {
            caps[cap.name] = cap.enabled;
        });

        return caps;
    }

    /**
     * Estimate gas for agent registration
     */
    async estimateRegistrationGas(params: AgentRegistrationParams): Promise<ethers.BigNumber> {
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