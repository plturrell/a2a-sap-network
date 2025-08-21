#!/usr/bin/env python3
"""
Nonce Management for Blockchain Transactions
Ensures proper transaction ordering and prevents nonce conflicts
"""

import asyncio
import logging
from typing import Dict, Optional
from threading import Lock
import time

logger = logging.getLogger(__name__)

class NonceManager:
    """
    Thread-safe nonce manager for blockchain transactions
    Prevents nonce conflicts and handles transaction failures
    """
    
    def __init__(self, web3_client):
        self.web3 = web3_client
        self.local_nonces: Dict[str, int] = {}
        self.locks: Dict[str, Lock] = {}
        self.pending_transactions: Dict[str, set] = {}
        self.last_sync: Dict[str, float] = {}
        self.sync_interval = 60  # Sync with blockchain every 60 seconds
        
    def _get_lock(self, address: str) -> Lock:
        """Get or create lock for address"""
        if address not in self.locks:
            self.locks[address] = Lock()
        return self.locks[address]
    
    async def get_nonce(self, address: str) -> int:
        """
        Get next available nonce for address
        Thread-safe and handles pending transactions
        """
        lock = self._get_lock(address)
        
        with lock:
            current_time = time.time()
            
            # Check if we need to sync with blockchain
            if (address not in self.last_sync or 
                current_time - self.last_sync[address] > self.sync_interval):
                await self._sync_nonce_with_blockchain(address)
                self.last_sync[address] = current_time
            
            # Initialize if not exists
            if address not in self.local_nonces:
                self.local_nonces[address] = await self._get_blockchain_nonce(address)
                self.pending_transactions[address] = set()
            
            # Get next nonce
            nonce = self.local_nonces[address]
            
            # Increment local nonce
            self.local_nonces[address] += 1
            
            # Track as pending
            if address not in self.pending_transactions:
                self.pending_transactions[address] = set()
            self.pending_transactions[address].add(nonce)
            
            logger.debug(f"Assigned nonce {nonce} to address {address}")
            return nonce
    
    async def _get_blockchain_nonce(self, address: str) -> int:
        """Get current nonce from blockchain"""
        try:
            # Get nonce including pending transactions
            nonce = self.web3.eth.get_transaction_count(address, 'pending')
            logger.debug(f"Blockchain nonce for {address}: {nonce}")
            return nonce
        except Exception as e:
            logger.error(f"Failed to get blockchain nonce for {address}: {e}")
            # Fallback to confirmed transactions only
            return self.web3.eth.get_transaction_count(address, 'latest')
    
    async def _sync_nonce_with_blockchain(self, address: str):
        """Sync local nonce with blockchain state"""
        try:
            blockchain_nonce = await self._get_blockchain_nonce(address)
            
            if address in self.local_nonces:
                local_nonce = self.local_nonces[address]
                
                # If blockchain is ahead, we missed some transactions
                if blockchain_nonce > local_nonce:
                    logger.warning(f"Blockchain nonce ({blockchain_nonce}) ahead of local ({local_nonce}) for {address}")
                    self.local_nonces[address] = blockchain_nonce
                    
                    # Clear old pending transactions
                    if address in self.pending_transactions:
                        old_pending = self.pending_transactions[address].copy()
                        self.pending_transactions[address] = {n for n in old_pending if n >= blockchain_nonce}
                        
                        if old_pending != self.pending_transactions[address]:
                            logger.info(f"Cleaned up confirmed transactions for {address}")
                
                # If local is significantly ahead, something is wrong
                elif local_nonce - blockchain_nonce > 10:
                    logger.error(f"Local nonce ({local_nonce}) too far ahead of blockchain ({blockchain_nonce}) for {address}")
                    await self.reset_nonce(address)
            else:
                self.local_nonces[address] = blockchain_nonce
                
        except Exception as e:
            logger.error(f"Failed to sync nonce for {address}: {e}")
    
    async def confirm_transaction(self, address: str, nonce: int, tx_hash: str):
        """
        Confirm transaction was mined successfully
        Removes nonce from pending list
        """
        lock = self._get_lock(address)
        
        with lock:
            if address in self.pending_transactions:
                self.pending_transactions[address].discard(nonce)
                logger.debug(f"Confirmed transaction {tx_hash} with nonce {nonce} for {address}")
    
    async def handle_transaction_failure(self, address: str, nonce: int, error: str):
        """
        Handle failed transaction
        May need to reset nonce depending on error type
        """
        lock = self._get_lock(address)
        
        with lock:
            # Remove from pending
            if address in self.pending_transactions:
                self.pending_transactions[address].discard(nonce)
            
            # Check if nonce error
            if 'nonce' in error.lower() or 'replacement transaction underpriced' in error.lower():
                logger.warning(f"Nonce-related error for {address}: {error}")
                await self.reset_nonce(address)
            else:
                logger.debug(f"Transaction failed for {address} with nonce {nonce}: {error}")
    
    async def reset_nonce(self, address: str):
        """
        Reset nonce for address - sync with blockchain
        Use when transactions are stuck or nonces are out of sync
        """
        lock = self._get_lock(address)
        
        with lock:
            try:
                # Get fresh nonce from blockchain
                blockchain_nonce = await self._get_blockchain_nonce(address)
                self.local_nonces[address] = blockchain_nonce
                
                # Clear pending transactions
                if address in self.pending_transactions:
                    self.pending_transactions[address].clear()
                
                # Reset sync timer
                self.last_sync[address] = time.time()
                
                logger.info(f"Reset nonce for {address} to {blockchain_nonce}")
                
            except Exception as e:
                logger.error(f"Failed to reset nonce for {address}: {e}")
                raise
    
    def get_pending_count(self, address: str) -> int:
        """Get number of pending transactions for address"""
        if address in self.pending_transactions:
            return len(self.pending_transactions[address])
        return 0
    
    def get_status(self, address: str) -> dict:
        """Get status information for address"""
        return {
            'local_nonce': self.local_nonces.get(address, 0),
            'pending_count': self.get_pending_count(address),
            'pending_nonces': list(self.pending_transactions.get(address, set())),
            'last_sync': self.last_sync.get(address, 0)
        }
    
    async def recover_stuck_transactions(self, address: str):
        """
        Attempt to recover from stuck transactions
        Checks for stale pending transactions and resubmits or cancels them
        """
        lock = self._get_lock(address)
        
        with lock:
            if address not in self.pending_transactions:
                return
            
            current_time = time.time()
            stale_threshold = 300  # 5 minutes
            
            # Get current blockchain nonce
            blockchain_nonce = await self._get_blockchain_nonce(address)
            
            # Find stale transactions
            stale_nonces = []
            for nonce in self.pending_transactions[address]:
                # If nonce is less than blockchain nonce, it should have been confirmed
                if nonce < blockchain_nonce:
                    stale_nonces.append(nonce)
            
            # Clean up stale transactions
            for nonce in stale_nonces:
                self.pending_transactions[address].discard(nonce)
                logger.info(f"Cleaned up stale transaction with nonce {nonce} for {address}")
            
            # Check if we need to reset
            if len(stale_nonces) > 0:
                await self._sync_nonce_with_blockchain(address)


class TransactionManager:
    """
    High-level transaction manager that integrates nonce management
    with transaction submission and monitoring
    """
    
    def __init__(self, web3_client, nonce_manager: NonceManager):
        self.web3 = web3_client
        self.nonce_manager = nonce_manager
        self.pending_transactions: Dict[str, dict] = {}
        
    async def submit_transaction(self, transaction, account, max_retries: int = 3):
        """
        Submit transaction with proper nonce management
        Handles retries and error recovery
        """
        address = account.address
        
        for attempt in range(max_retries):
            try:
                # Get nonce
                nonce = await self.nonce_manager.get_nonce(address)
                
                # Update transaction with nonce
                transaction['nonce'] = nonce
                
                # Sign transaction
                signed_txn = account.sign_transaction(transaction)
                
                # Submit to blockchain
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.raw_transaction)
                
                # Track transaction
                self.pending_transactions[tx_hash.hex()] = {
                    'address': address,
                    'nonce': nonce,
                    'timestamp': time.time(),
                    'attempt': attempt + 1
                }
                
                logger.info(f"Submitted transaction {tx_hash.hex()} with nonce {nonce} (attempt {attempt + 1})")
                
                return tx_hash
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Transaction submission failed (attempt {attempt + 1}): {error_msg}")
                
                # Handle nonce-related errors
                await self.nonce_manager.handle_transaction_failure(address, nonce, error_msg)
                
                # If last attempt, raise the error
                if attempt == max_retries - 1:
                    raise
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def wait_for_confirmation(self, tx_hash: str, timeout: int = 120):
        """
        Wait for transaction confirmation with timeout
        Updates nonce manager when transaction is confirmed
        """
        tx_hash_str = tx_hash.hex() if hasattr(tx_hash, 'hex') else str(tx_hash)
        
        try:
            # Wait for receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            
            # Update nonce manager
            if tx_hash_str in self.pending_transactions:
                tx_info = self.pending_transactions[tx_hash_str]
                await self.nonce_manager.confirm_transaction(
                    tx_info['address'], 
                    tx_info['nonce'], 
                    tx_hash_str
                )
                del self.pending_transactions[tx_hash_str]
            
            logger.info(f"Transaction {tx_hash_str} confirmed in block {receipt.blockNumber}")
            return receipt
            
        except Exception as e:
            logger.error(f"Transaction {tx_hash_str} confirmation failed: {e}")
            
            # Handle failure
            if tx_hash_str in self.pending_transactions:
                tx_info = self.pending_transactions[tx_hash_str]
                await self.nonce_manager.handle_transaction_failure(
                    tx_info['address'], 
                    tx_info['nonce'], 
                    str(e)
                )
                del self.pending_transactions[tx_hash_str]
            
            raise
    
    def get_pending_transactions(self) -> dict:
        """Get all pending transactions"""
        return self.pending_transactions.copy()
    
    async def cleanup_stale_transactions(self):
        """Clean up stale pending transactions"""
        current_time = time.time()
        stale_threshold = 600  # 10 minutes
        
        stale_txs = []
        for tx_hash, tx_info in self.pending_transactions.items():
            if current_time - tx_info['timestamp'] > stale_threshold:
                stale_txs.append(tx_hash)
        
        for tx_hash in stale_txs:
            tx_info = self.pending_transactions[tx_hash]
            logger.warning(f"Cleaning up stale transaction {tx_hash}")
            
            await self.nonce_manager.handle_transaction_failure(
                tx_info['address'], 
                tx_info['nonce'], 
                "Transaction timeout"
            )
            del self.pending_transactions[tx_hash]