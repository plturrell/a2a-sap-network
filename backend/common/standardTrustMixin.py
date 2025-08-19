"""
Standard trust system mixin for A2A agents.
Provides consistent blockchain trust integration across all agents.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from trustSystem.trustIntegration import get_trust_system
from config.agentConfig import config
from common.errorHandling import with_circuit_breaker, with_retry

logger = logging.getLogger(__name__)


class TrustSystemMixin:
    """
    Mixin class that provides standardized trust system integration
    for all A2A agents. Ensures consistent blockchain trust operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize trust system mixin."""
        super().__init__(*args, **kwargs)
        
        # Trust system components
        self._trust_system = None
        self._trust_initialized = False
        self._trust_score_cache = {}
        self._trust_verification_cache = {}
        
        # Trust metrics
        self._trust_operations_count = 0
        self._trust_failures_count = 0
        self._last_trust_sync = None
        
        # Agent trust configuration
        self._required_trust_score = 0.7  # Minimum trust score for interactions
        self._trust_cache_ttl = 300  # 5 minutes cache TTL
        self._auto_trust_update = True  # Automatically update trust scores
    
    async def initialize_trust_system(
        self,
        agent_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialize blockchain trust system for this agent.
        
        Args:
            agent_metadata: Additional metadata for agent registration
            
        Returns:
            True if initialization successful
        """
        if self._trust_initialized:
            return True
        
        try:
            # Get trust system instance
            self._trust_system = get_trust_system()
            
            # Prepare agent metadata
            metadata = {
                'agent_id': getattr(self, 'agent_id', 'unknown'),
                'agent_type': getattr(self, 'agent_type', 'unknown'),
                'capabilities': getattr(self, 'capabilities', []),
                'version': getattr(self, 'version', '1.0.0'),
                'initialized_at': datetime.now().isoformat()
            }
            
            if agent_metadata:
                metadata.update(agent_metadata)
            
            # Initialize agent trust on blockchain
            success = await self._initialize_agent_trust_with_retry(
                self.agent_id, metadata
            )
            
            if success:
                self._trust_initialized = True
                self._last_trust_sync = datetime.now()
                logger.info(f"Trust system initialized for agent {self.agent_id}")
                
                # Setup periodic trust score updates
                if self._auto_trust_update:
                    asyncio.create_task(self._periodic_trust_updates())
                
                return True
            else:
                logger.error(f"Failed to initialize trust for agent {self.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Trust system initialization error: {e}")
            return False
    
    @with_circuit_breaker("trust_init", failure_threshold=3, recovery_timeout=60)
    @with_retry(max_retries=3, initial_delay=2.0)
    async def _initialize_agent_trust_with_retry(
        self,
        agent_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Initialize agent trust with retry logic."""
        return self._trust_system.initialize_agent_trust(agent_id, metadata)
    
    async def sign_message(
        self,
        message: Dict[str, Any],
        private_key: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Sign a message using the agent's private key.
        
        Args:
            message: Message to sign
            private_key: Optional private key (uses agent's key if not provided)
            
        Returns:
            Signature data
        """
        if not self._trust_initialized:
            await self.initialize_trust_system()
        
        try:
            signature_data = await self._sign_message_with_circuit_breaker(
                message, self.agent_id, private_key
            )
            
            self._trust_operations_count += 1
            return signature_data
            
        except Exception as e:
            self._trust_failures_count += 1
            logger.error(f"Message signing failed: {e}")
            raise
    
    @with_circuit_breaker("message_signing", failure_threshold=5, recovery_timeout=30)
    async def _sign_message_with_circuit_breaker(
        self,
        message: Dict[str, Any],
        agent_id: str,
        private_key: Optional[str]
    ) -> Dict[str, str]:
        """Sign message with circuit breaker protection."""
        return self._trust_system.sign_a2a_message(message, agent_id, private_key)
    
    async def verify_message(
        self,
        message: Dict[str, Any],
        signature_data: Dict[str, str]
    ) -> bool:
        """
        Verify a signed message.
        
        Args:
            message: Original message
            signature_data: Signature data to verify
            
        Returns:
            True if signature is valid
        """
        if not self._trust_initialized:
            await self.initialize_trust_system()
        
        # Check verification cache first
        cache_key = self._generate_verification_cache_key(message, signature_data)
        if cache_key in self._trust_verification_cache:
            cached_result, cached_time = self._trust_verification_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self._trust_cache_ttl):
                return cached_result
        
        try:
            is_valid = await self._verify_message_with_circuit_breaker(
                message, signature_data
            )
            
            # Cache verification result
            self._trust_verification_cache[cache_key] = (is_valid, datetime.now())
            self._cleanup_verification_cache()
            
            self._trust_operations_count += 1
            return is_valid
            
        except Exception as e:
            self._trust_failures_count += 1
            logger.error(f"Message verification failed: {e}")
            return False
    
    @with_circuit_breaker("message_verification", failure_threshold=5, recovery_timeout=30)
    async def _verify_message_with_circuit_breaker(
        self,
        message: Dict[str, Any],
        signature_data: Dict[str, str]
    ) -> bool:
        """Verify message with circuit breaker protection."""
        return self._trust_system.verify_a2a_message(message, signature_data)
    
    async def get_trust_score(
        self,
        peer_agent_id: str,
        force_refresh: bool = False
    ) -> float:
        """
        Get trust score for a peer agent.
        
        Args:
            peer_agent_id: ID of peer agent
            force_refresh: Force refresh from blockchain
            
        Returns:
            Trust score (0.0 - 1.0)
        """
        if not self._trust_initialized:
            await self.initialize_trust_system()
        
        # Check cache first
        if not force_refresh and peer_agent_id in self._trust_score_cache:
            cached_score, cached_time = self._trust_score_cache[peer_agent_id]
            if datetime.now() - cached_time < timedelta(seconds=self._trust_cache_ttl):
                return cached_score
        
        try:
            # Get trust score from blockchain
            trust_score = await self._get_trust_score_from_blockchain(peer_agent_id)
            
            # Cache the result
            self._trust_score_cache[peer_agent_id] = (trust_score, datetime.now())
            self._cleanup_trust_cache()
            
            return trust_score
            
        except Exception as e:
            logger.error(f"Failed to get trust score for {peer_agent_id}: {e}")
            # Return cached value if available, otherwise default
            if peer_agent_id in self._trust_score_cache:
                return self._trust_score_cache[peer_agent_id][0]
            return 0.5  # Default neutral trust
    
    @with_circuit_breaker("trust_score_query", failure_threshold=3, recovery_timeout=45)
    async def _get_trust_score_from_blockchain(self, peer_agent_id: str) -> float:
        """Get trust score from blockchain with circuit breaker."""
        # This would query the blockchain trust registry
        # For now, return a default implementation
        return 0.8  # Mock trust score
    
    async def update_trust_score(
        self,
        peer_agent_id: str,
        score: float,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update trust score for a peer agent.
        
        Args:
            peer_agent_id: ID of peer agent
            score: New trust score (0.0 - 1.0)
            reason: Optional reason for trust update
            
        Returns:
            True if update successful
        """
        if not self._trust_initialized:
            await self.initialize_trust_system()
        
        if not 0.0 <= score <= 1.0:
            raise ValueError("Trust score must be between 0.0 and 1.0")
        
        try:
            # Convert to 0-100 scale for blockchain storage
            blockchain_score = int(score * 100)
            
            success = await self._update_trust_score_with_circuit_breaker(
                self.agent_id, peer_agent_id, blockchain_score
            )
            
            if success:
                # Update local cache
                self._trust_score_cache[peer_agent_id] = (score, datetime.now())
                
                logger.info(
                    f"Updated trust score: {self.agent_id} → {peer_agent_id} = {score:.2f}"
                    + (f" (reason: {reason})" if reason else "")
                )
                
                return True
            else:
                logger.error(f"Failed to update trust score for {peer_agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Trust score update error: {e}")
            return False
    
    @with_circuit_breaker("trust_score_update", failure_threshold=3, recovery_timeout=45)
    async def _update_trust_score_with_circuit_breaker(
        self,
        from_agent: str,
        to_agent: str,
        score: int
    ) -> bool:
        """Update trust score with circuit breaker protection."""
        return self._trust_system.update_trust_score(from_agent, to_agent, score)
    
    async def is_agent_trusted(
        self,
        peer_agent_id: str,
        minimum_score: Optional[float] = None
    ) -> bool:
        """
        Check if a peer agent is trusted.
        
        Args:
            peer_agent_id: ID of peer agent
            minimum_score: Minimum required trust score
            
        Returns:
            True if agent is trusted
        """
        required_score = minimum_score or self._required_trust_score
        current_score = await self.get_trust_score(peer_agent_id)
        
        return current_score >= required_score
    
    async def get_trusted_agents(
        self,
        minimum_score: Optional[float] = None
    ) -> List[str]:
        """
        Get list of trusted peer agents.
        
        Args:
            minimum_score: Minimum required trust score
            
        Returns:
            List of trusted agent IDs
        """
        required_score = minimum_score or self._required_trust_score
        trusted_agents = []
        
        # Check cached trust scores
        for agent_id, (score, _) in self._trust_score_cache.items():
            if score >= required_score:
                trusted_agents.append(agent_id)
        
        return trusted_agents
    
    async def _periodic_trust_updates(self):
        """Periodic task to update trust scores based on interactions."""
        while self._trust_initialized:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if datetime.now() - self._last_trust_sync > timedelta(hours=6):
                    await self._sync_trust_scores()
                    self._last_trust_sync = datetime.now()
                    
            except Exception as e:
                logger.error(f"Periodic trust update error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _sync_trust_scores(self):
        """Synchronize trust scores with blockchain."""
        logger.info(f"Syncing trust scores for agent {self.agent_id}")
        
        # Clear cache to force refresh
        self._trust_score_cache.clear()
        
        # Sync with known peers
        known_peers = getattr(self, 'known_peers', [])
        for peer_id in known_peers:
            try:
                await self.get_trust_score(peer_id, force_refresh=True)
            except Exception as e:
                logger.warning(f"Failed to sync trust score for {peer_id}: {e}")
    
    def _generate_verification_cache_key(
        self,
        message: Dict[str, Any],
        signature_data: Dict[str, str]
    ) -> str:
        """Generate cache key for message verification."""
        import hashlib
        import json
        
        key_data = {
            'message_hash': signature_data.get('message_hash', ''),
            'signature': signature_data.get('signature', ''),
            'signer_address': signature_data.get('signer_address', '')
        }
        
        return hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
    
    def _cleanup_trust_cache(self):
        """Clean up expired trust score cache entries."""
        if len(self._trust_score_cache) <= 100:
            return
        
        cutoff_time = datetime.now() - timedelta(seconds=self._trust_cache_ttl)
        expired_keys = [
            key for key, (_, cached_time) in self._trust_score_cache.items()
            if cached_time < cutoff_time
        ]
        
        for key in expired_keys:
            del self._trust_score_cache[key]
    
    def _cleanup_verification_cache(self):
        """Clean up expired verification cache entries."""
        if len(self._trust_verification_cache) <= 100:
            return
        
        cutoff_time = datetime.now() - timedelta(seconds=self._trust_cache_ttl)
        expired_keys = [
            key for key, (_, cached_time) in self._trust_verification_cache.items()
            if cached_time < cutoff_time
        ]
        
        for key in expired_keys:
            del self._trust_verification_cache[key]
    
    def get_trust_metrics(self) -> Dict[str, Any]:
        """Get trust system metrics for monitoring."""
        return {
            'trust_initialized': self._trust_initialized,
            'trust_operations_count': self._trust_operations_count,
            'trust_failures_count': self._trust_failures_count,
            'trust_success_rate': (
                (self._trust_operations_count - self._trust_failures_count) /
                max(self._trust_operations_count, 1)
            ),
            'cached_trust_scores': len(self._trust_score_cache),
            'cached_verifications': len(self._trust_verification_cache),
            'last_trust_sync': self._last_trust_sync.isoformat() if self._last_trust_sync else None,
            'required_trust_score': self._required_trust_score
        }
    
    async def configure_trust_settings(
        self,
        required_trust_score: Optional[float] = None,
        cache_ttl: Optional[int] = None,
        auto_trust_update: Optional[bool] = None
    ):
        """
        Configure trust system settings.
        
        Args:
            required_trust_score: Minimum trust score for interactions
            cache_ttl: Cache TTL in seconds
            auto_trust_update: Enable automatic trust updates
        """
        if required_trust_score is not None:
            if not 0.0 <= required_trust_score <= 1.0:
                raise ValueError("Required trust score must be between 0.0 and 1.0")
            self._required_trust_score = required_trust_score
        
        if cache_ttl is not None:
            if cache_ttl < 60:
                raise ValueError("Cache TTL must be at least 60 seconds")
            self._trust_cache_ttl = cache_ttl
        
        if auto_trust_update is not None:
            self._auto_trust_update = auto_trust_update
        
        logger.info(f"Trust settings updated for agent {self.agent_id}")


class TrustedAgentMixin(TrustSystemMixin):
    """
    Enhanced trust mixin with additional features for high-trust agents.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize trusted agent mixin."""
        super().__init__(*args, **kwargs)
        
        # Enhanced trust features
        self._trust_threshold_high = 0.9  # High trust threshold
        self._trust_threshold_low = 0.3   # Low trust threshold
        self._reputation_tracking = True
        self._trust_delegation_enabled = False
    
    async def evaluate_agent_reputation(
        self,
        peer_agent_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate comprehensive reputation of a peer agent.
        
        Args:
            peer_agent_id: ID of peer agent
            
        Returns:
            Reputation evaluation
        """
        trust_score = await self.get_trust_score(peer_agent_id)
        
        # Classify trust level
        if trust_score >= self._trust_threshold_high:
            trust_level = "high"
        elif trust_score >= self._required_trust_score:
            trust_level = "medium"
        elif trust_score >= self._trust_threshold_low:
            trust_level = "low"
        else:
            trust_level = "untrusted"
        
        return {
            'agent_id': peer_agent_id,
            'trust_score': trust_score,
            'trust_level': trust_level,
            'evaluation_time': datetime.now().isoformat(),
            'recommendations': self._generate_trust_recommendations(trust_score),
            'interaction_history': await self._get_interaction_history(peer_agent_id)
        }
    
    def _generate_trust_recommendations(self, trust_score: float) -> List[str]:
        """Generate trust-based recommendations."""
        recommendations = []
        
        if trust_score >= self._trust_threshold_high:
            recommendations.extend([
                "Safe for critical operations",
                "Enable delegation if supported",
                "Consider for leadership roles"
            ])
        elif trust_score >= self._required_trust_score:
            recommendations.extend([
                "Safe for normal operations",
                "Monitor performance",
                "Build more interaction history"
            ])
        elif trust_score >= self._trust_threshold_low:
            recommendations.extend([
                "Limit to low-risk operations",
                "Require additional verification",
                "Increase monitoring frequency"
            ])
        else:
            recommendations.extend([
                "Avoid direct interactions",
                "Require manual approval",
                "Investigate trust issues"
            ])
        
        return recommendations
    
    async def _get_interaction_history(self, peer_agent_id: str) -> Dict[str, Any]:
        """Get interaction history with peer agent."""
        # This would query interaction logs
        return {
            'total_interactions': 0,
            'successful_interactions': 0,
            'failed_interactions': 0,
            'last_interaction': None,
            'interaction_trend': 'stable'
        }