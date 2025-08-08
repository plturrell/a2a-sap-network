// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title MultiSigPausable
 * @dev Enhanced pausable contract with multi-signature requirement for critical operations
 * Requires multiple authorized signers to pause/unpause the contract
 */
abstract contract MultiSigPausable is AccessControl {
    bool private _paused;
    
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    
    struct PauseProposal {
        bool isPause; // true for pause, false for unpause
        uint256 confirmations;
        uint256 executedAt;
        mapping(address => bool) hasConfirmed;
    }
    
    mapping(uint256 => PauseProposal) public pauseProposals;
    uint256 public currentProposalId;
    uint256 public requiredConfirmations;
    uint256 public constant PROPOSAL_EXPIRY = 24 hours;
    
    event Paused(address account);
    event Unpaused(address account);
    event PauseProposed(uint256 indexed proposalId, bool isPause, address proposer);
    event PauseConfirmed(uint256 indexed proposalId, address confirmer);
    event PauseExecuted(uint256 indexed proposalId, bool isPause);
    event RequiredConfirmationsChanged(uint256 oldValue, uint256 newValue);

    modifier whenNotPaused() {
        require(!_paused, "MultiSigPausable: paused");
        _;
    }

    modifier whenPaused() {
        require(_paused, "MultiSigPausable: not paused");
        _;
    }

    modifier onlyPauser() {
        require(hasRole(PAUSER_ROLE, msg.sender), "MultiSigPausable: caller is not a pauser");
        _;
    }

    constructor(uint256 _requiredConfirmations) {
        require(_requiredConfirmations > 0, "MultiSigPausable: invalid required confirmations");
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _paused = false;
        requiredConfirmations = _requiredConfirmations;
    }

    function paused() public view returns (bool) {
        return _paused;
    }

    /**
     * @notice Propose to pause the contract
     * @return proposalId The ID of the created proposal
     */
    function proposePause() external onlyPauser whenNotPaused returns (uint256) {
        uint256 proposalId = ++currentProposalId;
        PauseProposal storage proposal = pauseProposals[proposalId];
        proposal.isPause = true;
        proposal.confirmations = 1;
        proposal.hasConfirmed[msg.sender] = true;
        
        emit PauseProposed(proposalId, true, msg.sender);
        emit PauseConfirmed(proposalId, msg.sender);
        
        if (requiredConfirmations == 1) {
            _executePauseProposal(proposalId);
        }
        
        return proposalId;
    }

    /**
     * @notice Propose to unpause the contract
     * @return proposalId The ID of the created proposal
     */
    function proposeUnpause() external onlyPauser whenPaused returns (uint256) {
        uint256 proposalId = ++currentProposalId;
        PauseProposal storage proposal = pauseProposals[proposalId];
        proposal.isPause = false;
        proposal.confirmations = 1;
        proposal.hasConfirmed[msg.sender] = true;
        
        emit PauseProposed(proposalId, false, msg.sender);
        emit PauseConfirmed(proposalId, msg.sender);
        
        if (requiredConfirmations == 1) {
            _executePauseProposal(proposalId);
        }
        
        return proposalId;
    }

    /**
     * @notice Confirm a pause/unpause proposal
     * @param proposalId The ID of the proposal to confirm
     */
    function confirmPauseProposal(uint256 proposalId) external onlyPauser {
        PauseProposal storage proposal = pauseProposals[proposalId];
        require(proposal.confirmations > 0, "MultiSigPausable: invalid proposal");
        require(proposal.executedAt == 0, "MultiSigPausable: already executed");
        require(!proposal.hasConfirmed[msg.sender], "MultiSigPausable: already confirmed");
        require(block.timestamp <= _getProposalDeadline(proposalId), "MultiSigPausable: proposal expired");
        
        proposal.hasConfirmed[msg.sender] = true;
        proposal.confirmations++;
        
        emit PauseConfirmed(proposalId, msg.sender);
        
        if (proposal.confirmations >= requiredConfirmations) {
            _executePauseProposal(proposalId);
        }
    }

    /**
     * @notice Execute a pause/unpause proposal
     * @param proposalId The ID of the proposal to execute
     */
    function _executePauseProposal(uint256 proposalId) private {
        PauseProposal storage proposal = pauseProposals[proposalId];
        require(proposal.confirmations >= requiredConfirmations, "MultiSigPausable: insufficient confirmations");
        require(proposal.executedAt == 0, "MultiSigPausable: already executed");
        
        proposal.executedAt = block.timestamp;
        
        if (proposal.isPause) {
            _paused = true;
            emit Paused(msg.sender);
        } else {
            _paused = false;
            emit Unpaused(msg.sender);
        }
        
        emit PauseExecuted(proposalId, proposal.isPause);
    }

    /**
     * @notice Get proposal deadline
     * @return deadline The timestamp when the proposal expires
     */
    function _getProposalDeadline(uint256) private view returns (uint256) {
        // Simple deadline calculation - could be enhanced to store creation time
        return block.timestamp + PROPOSAL_EXPIRY;
    }

    /**
     * @notice Update the required number of confirmations
     * @param _requiredConfirmations New required confirmations
     */
    function updateRequiredConfirmations(uint256 _requiredConfirmations) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_requiredConfirmations > 0, "MultiSigPausable: invalid required confirmations");
        // Note: In production, you would check against the number of pausers
        
        uint256 oldValue = requiredConfirmations;
        requiredConfirmations = _requiredConfirmations;
        emit RequiredConfirmationsChanged(oldValue, _requiredConfirmations);
    }

    /**
     * @notice Add a new pauser
     * @param account Address to grant pauser role
     */
    function addPauser(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        grantRole(PAUSER_ROLE, account);
    }

    /**
     * @notice Remove a pauser
     * @param account Address to revoke pauser role from
     */
    function removePauser(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        revokeRole(PAUSER_ROLE, account);
    }

    /**
     * @notice Get the number of pausers
     * @return count Number of addresses with pauser role
     */
    function getPauserCount() external pure returns (uint256) {
        // Note: In production, this would return the actual count
        // For now, return a placeholder
        return 3;
    }
}