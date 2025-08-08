// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IAgentRegistry
 * @notice Interface for the Agent Registry contract
 * @dev Used by other contracts to interact with agent registration and reputation
 */
interface IAgentRegistry {
    /**
     * @notice Check if an address is a registered agent
     * @param agent Address to check
     * @return bool True if registered, false otherwise
     */
    function isRegistered(address agent) external view returns (bool);
    
    /**
     * @notice Get the reputation score of an agent
     * @param agent Address of the agent
     * @return uint256 Reputation score (0-200)
     */
    function getReputation(address agent) external view returns (uint256);
    
    /**
     * @notice Set the reputation score of an agent (restricted access)
     * @param agent Address of the agent
     * @param reputation New reputation score
     */
    function setReputation(address agent, uint256 reputation) external;
    
    /**
     * @notice Get agent details
     * @param agent Address of the agent
     * @return name Agent name
     * @return endpoint Agent endpoint URL
     * @return reputation Agent reputation score
     * @return active Agent active status
     */
    function getAgent(address agent) external view returns (
        string memory name,
        string memory endpoint,
        uint256 reputation,
        bool active
    );
    
    /**
     * @notice Get total number of registered agents
     * @return uint256 Total agent count
     */
    function getAgentCount() external view returns (uint256);
    
    /**
     * @notice Check if an agent is currently active
     * @param agent Address to check
     * @return bool True if active, false otherwise
     */
    function isActive(address agent) external view returns (bool);
}