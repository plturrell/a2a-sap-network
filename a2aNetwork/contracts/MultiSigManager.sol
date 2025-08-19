// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/utils/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title MultiSigManager
 * @notice Multi-signature wallet for critical blockchain operations
 * @dev Provides secure multi-signature functionality for high-value transactions
 */
contract MultiSigManager is 
    AccessControlUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable
{
    bytes32 public constant SIGNER_ROLE = keccak256("SIGNER_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");
    
    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
        uint256 timestamp;
        string description;
        mapping(address => bool) isConfirmed;
    }
    
    struct Proposal {
        bytes32 transactionHash;
        uint256 proposalTime;
        bool executed;
        uint256 confirmationCount;
        mapping(address => bool) hasConfirmed;
    }
    
    mapping(uint256 => Transaction) public transactions;
    mapping(bytes32 => Proposal) public proposals;
    mapping(address => bool) public isOwner;
    
    address[] public owners;
    uint256 public requiredConfirmations;
    uint256 public transactionCount;
    uint256 public constant MAX_OWNERS = 10;
    uint256 public constant MIN_CONFIRMATIONS = 2;
    uint256 public constant EXECUTION_DELAY = 24 hours;
    
    event TransactionSubmitted(uint256 indexed transactionId, address indexed submitter);
    event TransactionConfirmed(uint256 indexed transactionId, address indexed confirmer);
    event TransactionRevoked(uint256 indexed transactionId, address indexed revoker);
    event TransactionExecuted(uint256 indexed transactionId, address indexed executor);
    event OwnerAdded(address indexed owner);
    event OwnerRemoved(address indexed owner);
    event RequiredConfirmationsChanged(uint256 required);
    
    modifier onlyOwner() {
        require(isOwner[msg.sender], "Not an owner");
        _;
    }
    
    modifier transactionExists(uint256 transactionId) {
        require(transactionId < transactionCount, "Transaction does not exist");
        _;
    }
    
    modifier notExecuted(uint256 transactionId) {
        require(!transactions[transactionId].executed, "Transaction already executed");
        _;
    }
    
    modifier notConfirmed(uint256 transactionId) {
        require(!transactions[transactionId].isConfirmed[msg.sender], "Transaction already confirmed");
        _;
    }
    
    function initialize(
        address[] memory _owners,
        uint256 _requiredConfirmations
    ) public initializer {
        __AccessControl_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        require(_owners.length <= MAX_OWNERS, "Too many owners");
        require(_requiredConfirmations >= MIN_CONFIRMATIONS, "Too few required confirmations");
        require(_requiredConfirmations <= _owners.length, "Required confirmations exceed owner count");
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        
        for (uint256 i = 0; i < _owners.length; i++) {
            require(_owners[i] != address(0), "Invalid owner address");
            require(!isOwner[_owners[i]], "Duplicate owner");
            
            isOwner[_owners[i]] = true;
            owners.push(_owners[i]);
            _grantRole(SIGNER_ROLE, _owners[i]);
        }
        
        requiredConfirmations = _requiredConfirmations;
    }
    
    /**
     * @notice Submit a transaction for multi-sig approval
     */
    function submitTransaction(
        address to,
        uint256 value,
        bytes memory data,
        string memory description
    ) external onlyOwner returns (uint256) {
        require(to != address(0), "Invalid destination address");
        
        uint256 transactionId = transactionCount++;
        
        Transaction storage txn = transactions[transactionId];
        txn.to = to;
        txn.value = value;
        txn.data = data;
        txn.executed = false;
        txn.confirmations = 0;
        txn.timestamp = block.timestamp;
        txn.description = description;
        
        emit TransactionSubmitted(transactionId, msg.sender);
        
        // Auto-confirm by submitter
        confirmTransaction(transactionId);
        
        return transactionId;
    }
    
    /**
     * @notice Confirm a pending transaction
     */
    function confirmTransaction(uint256 transactionId)
        public
        onlyOwner
        transactionExists(transactionId)
        notConfirmed(transactionId)
        notExecuted(transactionId)
    {
        Transaction storage txn = transactions[transactionId];
        txn.isConfirmed[msg.sender] = true;
        txn.confirmations++;
        
        emit TransactionConfirmed(transactionId, msg.sender);
        
        // Auto-execute if enough confirmations and delay passed
        if (txn.confirmations >= requiredConfirmations && 
            block.timestamp >= txn.timestamp + EXECUTION_DELAY) {
            executeTransaction(transactionId);
        }
    }
    
    /**
     * @notice Revoke confirmation for a transaction
     */
    function revokeConfirmation(uint256 transactionId)
        external
        onlyOwner
        transactionExists(transactionId)
        notExecuted(transactionId)
    {
        Transaction storage txn = transactions[transactionId];
        require(txn.isConfirmed[msg.sender], "Transaction not confirmed by sender");
        
        txn.isConfirmed[msg.sender] = false;
        txn.confirmations--;
        
        emit TransactionRevoked(transactionId, msg.sender);
    }
    
    /**
     * @notice Execute a confirmed transaction
     */
    function executeTransaction(uint256 transactionId)
        public
        nonReentrant
        onlyOwner
        transactionExists(transactionId)
        notExecuted(transactionId)
    {
        Transaction storage txn = transactions[transactionId];
        require(txn.confirmations >= requiredConfirmations, "Insufficient confirmations");
        require(block.timestamp >= txn.timestamp + EXECUTION_DELAY, "Execution delay not met");
        
        txn.executed = true;
        
        (bool success, ) = txn.to.call{value: txn.value}(txn.data);
        require(success, "Transaction execution failed");
        
        emit TransactionExecuted(transactionId, msg.sender);
    }
    
    /**
     * @notice Add a new owner
     */
    function addOwner(address owner) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(owner != address(0), "Invalid owner address");
        require(!isOwner[owner], "Already an owner");
        require(owners.length < MAX_OWNERS, "Maximum owners reached");
        
        isOwner[owner] = true;
        owners.push(owner);
        _grantRole(SIGNER_ROLE, owner);
        
        emit OwnerAdded(owner);
    }
    
    /**
     * @notice Remove an owner
     */
    function removeOwner(address owner) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(isOwner[owner], "Not an owner");
        require(owners.length > requiredConfirmations, "Cannot remove owner below required confirmations");
        
        isOwner[owner] = false;
        _revokeRole(SIGNER_ROLE, owner);
        
        // Remove from owners array
        for (uint256 i = 0; i < owners.length; i++) {
            if (owners[i] == owner) {
                owners[i] = owners[owners.length - 1];
                owners.pop();
                break;
            }
        }
        
        emit OwnerRemoved(owner);
    }
    
    /**
     * @notice Change required confirmations
     */
    function changeRequiredConfirmations(uint256 _required) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_required >= MIN_CONFIRMATIONS, "Too few required confirmations");
        require(_required <= owners.length, "Required confirmations exceed owner count");
        
        requiredConfirmations = _required;
        emit RequiredConfirmationsChanged(_required);
    }
    
    /**
     * @notice Get transaction details
     */
    function getTransaction(uint256 transactionId)
        external
        view
        returns (
            address to,
            uint256 value,
            bytes memory data,
            bool executed,
            uint256 confirmations,
            uint256 timestamp,
            string memory description
        )
    {
        Transaction storage txn = transactions[transactionId];
        return (
            txn.to,
            txn.value,
            txn.data,
            txn.executed,
            txn.confirmations,
            txn.timestamp,
            txn.description
        );
    }
    
    /**
     * @notice Check if transaction is confirmed by owner
     */
    function isConfirmedBy(uint256 transactionId, address owner)
        external
        view
        returns (bool)
    {
        return transactions[transactionId].isConfirmed[owner];
    }
    
    /**
     * @notice Get list of owners
     */
    function getOwners() external view returns (address[] memory) {
        return owners;
    }
    
    /**
     * @notice Get confirmation count for a transaction
     */
    function getConfirmationCount(uint256 transactionId) external view returns (uint256) {
        return transactions[transactionId].confirmations;
    }
    
    /**
     * @notice Get pending transactions
     */
    function getPendingTransactions() external view returns (uint256[] memory) {
        uint256[] memory pending = new uint256[](transactionCount);
        uint256 count = 0;
        
        for (uint256 i = 0; i < transactionCount; i++) {
            if (!transactions[i].executed) {
                pending[count] = i;
                count++;
            }
        }
        
        // Resize array
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = pending[i];
        }
        
        return result;
    }
    
    /**
     * @notice Emergency pause function
     */
    function emergencyPause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        // Implement emergency pause logic if needed
        // This could pause all non-critical functions
    }
    
    receive() external payable {
        // Allow contract to receive ETH
    }
    
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {}
}