"""
Agent Delegation Smart Contracts
Defines authorized delegation relationships and capabilities between agents
"""

import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass

from .smartContractTrust import SmartContractTrust, get_trust_contract

logger = logging.getLogger(__name__)


class DelegationAction(str, Enum):
    """Types of actions that can be delegated"""
    DATA_STORAGE = "data_storage"
    DATA_RETRIEVAL = "data_retrieval" 
    DATA_ARCHIVAL = "data_archival"
    METADATA_REGISTRATION = "metadata_registration"
    METADATA_ENHANCEMENT = "metadata_enhancement"
    QUALITY_ASSESSMENT = "quality_assessment"
    SEARCH_OPERATIONS = "search_operations"
    STATUS_REPORTING = "status_reporting"
    AI_CONSULTATION = "ai_consultation"


class DelegationScope(str, Enum):
    """Scope of delegation permissions"""
    FULL = "full"  # Complete authority
    LIMITED = "limited"  # Specific operations only
    READ_ONLY = "read_only"  # Query only, no modifications
    EMERGENCY = "emergency"  # Emergency operations only


@dataclass
class DelegationRule:
    """Defines a delegation rule between agents"""
    delegator_id: str  # Agent delegating authority
    delegatee_id: str  # Agent receiving authority
    actions: Set[DelegationAction]  # What actions are delegated
    scope: DelegationScope  # Level of authority
    conditions: Dict[str, Any]  # Conditions for delegation
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def is_valid(self) -> bool:
        """Check if delegation rule is currently valid"""
        if not self.active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def can_perform_action(self, action: DelegationAction, context: Dict[str, Any] = None) -> bool:
        """Check if delegatee can perform specific action"""
        if not self.is_valid():
            return False
        
        if action not in self.actions:
            return False
        
        # Check conditions if specified
        if self.conditions and context:
            return self._check_conditions(context)
        
        return True
    
    def _check_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate delegation conditions"""
        for condition_key, condition_value in self.conditions.items():
            if condition_key == "min_trust_score":
                trust_score = context.get("trust_score", 0.0)
                if trust_score < condition_value:
                    return False
            elif condition_key == "required_capability":
                capabilities = context.get("capabilities", [])
                if condition_value not in capabilities:
                    return False
            elif condition_key == "time_window":
                current_hour = datetime.utcnow().hour
                if not (condition_value["start"] <= current_hour <= condition_value["end"]):
                    return False
        
        return True


class AgentDelegationContract:
    """Smart contract managing agent delegation relationships"""
    
    def __init__(self, contract_id: str = None):
        self.contract_id = contract_id or f"delegation_{uuid4().hex[:8]}"
        self.trust_contract = get_trust_contract()
        self.delegation_rules: Dict[str, DelegationRule] = {}
        self.delegation_history: List[Dict[str, Any]] = []
        
        # Initialize standard delegation relationships
        self._initialize_standard_delegations()
        
        logger.info(f"✅ Agent Delegation Contract initialized: {self.contract_id}")
    
    def _initialize_standard_delegations(self):
        """Initialize standard delegation relationships between agents"""
        # Agent 0 (Data Product Registration) delegations
        self.create_delegation(
            delegator_id="data_product_agent_0",
            delegatee_id="data_manager_agent",
            actions={DelegationAction.DATA_STORAGE, DelegationAction.DATA_ARCHIVAL},
            scope=DelegationScope.FULL,
            conditions={"min_trust_score": 0.7}
        )
        
        self.create_delegation(
            delegator_id="data_product_agent_0", 
            delegatee_id="catalog_manager_agent",
            actions={DelegationAction.METADATA_REGISTRATION, DelegationAction.METADATA_ENHANCEMENT},
            scope=DelegationScope.FULL,
            conditions={"min_trust_score": 0.7}
        )
        
        # Agent 1 (Data Standardization) delegations
        self.create_delegation(
            delegator_id="financial_standardization_agent_1",
            delegatee_id="data_manager_agent", 
            actions={DelegationAction.DATA_RETRIEVAL, DelegationAction.DATA_STORAGE},
            scope=DelegationScope.FULL,
            conditions={"min_trust_score": 0.7}
        )
        
        self.create_delegation(
            delegator_id="financial_standardization_agent_1",
            delegatee_id="catalog_manager_agent",
            actions={DelegationAction.SEARCH_OPERATIONS, DelegationAction.QUALITY_ASSESSMENT},
            scope=DelegationScope.FULL,
            conditions={"min_trust_score": 0.7}
        )
        
        # Data Manager can delegate to Catalog Manager for metadata operations
        self.create_delegation(
            delegator_id="data_manager_agent",
            delegatee_id="catalog_manager_agent",
            actions={DelegationAction.METADATA_REGISTRATION},
            scope=DelegationScope.LIMITED,
            conditions={"min_trust_score": 0.8}
        )
        
        # Catalog Manager can delegate to Data Manager for data operations
        self.create_delegation(
            delegator_id="catalog_manager_agent",
            delegatee_id="data_manager_agent",
            actions={DelegationAction.DATA_RETRIEVAL},
            scope=DelegationScope.READ_ONLY,
            conditions={"min_trust_score": 0.8}
        )
        
        # All agents can request AI consultation from each other
        for agent_id in ["data_product_agent_0", "financial_standardization_agent_1", 
                        "data_manager_agent", "catalog_manager_agent"]:
            for other_agent_id in ["data_product_agent_0", "financial_standardization_agent_1",
                                  "data_manager_agent", "catalog_manager_agent"]:
                if agent_id != other_agent_id:
                    self.create_delegation(
                        delegator_id=agent_id,
                        delegatee_id=other_agent_id,
                        actions={DelegationAction.AI_CONSULTATION, DelegationAction.STATUS_REPORTING},
                        scope=DelegationScope.READ_ONLY,
                        conditions={"min_trust_score": 0.5}
                    )
    
    def create_delegation(
        self,
        delegator_id: str,
        delegatee_id: str, 
        actions: Set[DelegationAction],
        scope: DelegationScope,
        conditions: Dict[str, Any] = None,
        duration_hours: int = None
    ) -> str:
        """Create a new delegation rule"""
        try:
            # Verify both agents exist in trust contract
            if delegator_id not in self.trust_contract.agents:
                raise ValueError(f"Delegator agent {delegator_id} not registered")
            if delegatee_id not in self.trust_contract.agents:
                raise ValueError(f"Delegatee agent {delegatee_id} not registered")
            
            # Check trust relationship with stricter validation
            trust_score = self.trust_contract.get_trust_score(delegator_id, delegatee_id)
            min_trust_for_delegation = 0.6  # Higher threshold for delegation creation
            
            if trust_score < min_trust_for_delegation:
                raise ValueError(f"Insufficient trust between agents for delegation: {trust_score} < {min_trust_for_delegation}")
            
            # Check if agents are verified for high-privilege delegations
            delegator_identity = self.trust_contract.agents[delegator_id]
            delegatee_identity = self.trust_contract.agents[delegatee_id]
            
            high_privilege_actions = {DelegationAction.DATA_STORAGE, DelegationAction.DATA_ARCHIVAL, DelegationAction.METADATA_REGISTRATION}
            if actions.intersection(high_privilege_actions) and not (delegator_identity.verified and delegatee_identity.verified):
                raise ValueError("Both agents must be verified for high-privilege delegations")
            
            # Create delegation rule
            expires_at = None
            if duration_hours:
                expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
            
            rule_id = f"delegation_{uuid4().hex[:8]}"
            delegation_rule = DelegationRule(
                delegator_id=delegator_id,
                delegatee_id=delegatee_id,
                actions=actions,
                scope=scope,
                conditions=conditions or {},
                expires_at=expires_at
            )
            
            self.delegation_rules[rule_id] = delegation_rule
            
            # Record in history
            self.delegation_history.append({
                "rule_id": rule_id,
                "action": "created",
                "delegator": delegator_id,
                "delegatee": delegatee_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"✅ Delegation created: {delegator_id} → {delegatee_id} ({len(actions)} actions)")
            return rule_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create delegation: {e}")
            raise
    
    def can_delegate(
        self,
        delegator_id: str,
        delegatee_id: str,
        action: DelegationAction,
        context: Dict[str, Any] = None
    ) -> bool:
        """Check if delegation is authorized"""
        try:
            # Find applicable delegation rule
            for rule in self.delegation_rules.values():
                if (rule.delegator_id == delegator_id and 
                    rule.delegatee_id == delegatee_id and
                    rule.can_perform_action(action, context)):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking delegation: {e}")
            return False
    
    def get_delegations_for_agent(self, agent_id: str) -> Dict[str, List[DelegationRule]]:
        """Get all delegations involving an agent"""
        try:
            delegated_to = []  # Actions delegated TO this agent
            delegated_from = []  # Actions delegated FROM this agent
            
            for rule in self.delegation_rules.values():
                if rule.delegatee_id == agent_id and rule.is_valid():
                    delegated_to.append(rule)
                elif rule.delegator_id == agent_id and rule.is_valid():
                    delegated_from.append(rule)
            
            return {
                "delegated_to": delegated_to,
                "delegated_from": delegated_from
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting delegations: {e}")
            return {"delegated_to": [], "delegated_from": []}
    
    def revoke_delegation(self, rule_id: str, revoker_id: str) -> bool:
        """Revoke a delegation rule"""
        try:
            if rule_id not in self.delegation_rules:
                return False
            
            rule = self.delegation_rules[rule_id]
            
            # Only delegator can revoke
            if rule.delegator_id != revoker_id:
                return False
            
            rule.active = False
            
            # Record revocation
            self.delegation_history.append({
                "rule_id": rule_id,
                "action": "revoked",
                "revoker": revoker_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"✅ Delegation revoked: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error revoking delegation: {e}")
            return False
    
    def record_delegation_use(
        self,
        delegator_id: str,
        delegatee_id: str,
        action: DelegationAction,
        success: bool,
        context: Dict[str, Any] = None
    ):
        """Record use of delegation for audit and trust scoring"""
        try:
            usage_record = {
                "delegator": delegator_id,
                "delegatee": delegatee_id,
                "action": action.value,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context or {}
            }
            
            self.delegation_history.append(usage_record)
            
            # Update trust scores based on delegation success with validation
            if success:
                # Successful delegation increases mutual trust slowly
                current_trust = self.trust_contract.get_trust_score(delegator_id, delegatee_id)
                
                # Validate trust increase is reasonable
                trust_increase = 0.005  # Much smaller increase to prevent gaming
                new_trust = min(0.95, current_trust + trust_increase)  # Cap below 1.0 to maintain security
                
                if delegator_id in self.trust_contract.trust_relationships and delegatee_id in self.trust_contract.trust_relationships[delegator_id]:
                    # Only update if both agents exist and relationship is valid
                    if new_trust > current_trust:  # Sanity check
                        self.trust_contract.trust_relationships[delegator_id][delegatee_id] = new_trust
                        logger.debug(f"Trust increased: {delegator_id} → {delegatee_id}: {current_trust:.3f} → {new_trust:.3f}")
            else:
                # Failed delegation decreases trust significantly
                current_trust = self.trust_contract.get_trust_score(delegator_id, delegatee_id)
                
                # Larger trust decrease for failures to discourage bad behavior
                trust_decrease = 0.1
                new_trust = max(0.0, current_trust - trust_decrease)
                
                if delegator_id in self.trust_contract.trust_relationships and delegatee_id in self.trust_contract.trust_relationships[delegator_id]:
                    self.trust_contract.trust_relationships[delegator_id][delegatee_id] = new_trust
                    
                logger.warning(f"⚠️ Failed delegation recorded: {delegator_id} → {delegatee_id} ({action}) - Trust: {current_trust:.3f} → {new_trust:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error recording delegation use: {e}")
    
    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation contract statistics"""
        try:
            total_rules = len(self.delegation_rules)
            active_rules = sum(1 for rule in self.delegation_rules.values() if rule.is_valid())
            
            # Count recent usage
            recent_usage = [
                record for record in self.delegation_history
                if "action" in record and record.get("action") not in ["created", "revoked"]
                and (datetime.utcnow() - datetime.fromisoformat(record["timestamp"])).total_seconds() < 3600
            ]
            
            success_rate = (
                sum(1 for record in recent_usage if record.get("success", False)) / len(recent_usage)
                if recent_usage else 0
            )
            
            return {
                "contract_id": self.contract_id,
                "total_delegation_rules": total_rules,
                "active_delegation_rules": active_rules,
                "recent_delegations": len(recent_usage),
                "success_rate": round(success_rate, 3),
                "total_history_records": len(self.delegation_history)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting delegation stats: {e}")
            return {"error": str(e)}


# Global delegation contract instance
_delegation_contract = None

def get_delegation_contract() -> AgentDelegationContract:
    """Get or create the global delegation contract"""
    global _delegation_contract
    if _delegation_contract is None:
        _delegation_contract = AgentDelegationContract()
    return _delegation_contract


def can_agent_delegate(
    delegator_id: str,
    delegatee_id: str,
    action: DelegationAction,
    context: Dict[str, Any] = None
) -> bool:
    """Check if agent can delegate specific action"""
    contract = get_delegation_contract()
    return contract.can_delegate(delegator_id, delegatee_id, action, context)


def create_delegation_contract(
    delegator_id: str,
    delegatee_id: str,
    actions: List[str],
    scope: DelegationScope = DelegationScope.LIMITED,
    conditions: Dict[str, Any] = None,
    duration_hours: int = None,
) -> str:
    """Public helper to create a delegation contract via the global instance.

    This keeps Agent-level code decoupled from the underlying contract implementation.
    Returns the rule/contract ID created by AgentDelegationContract.create_delegation.
    """
    contract = get_delegation_contract()

    # Convert action strings to DelegationAction enum values
    action_set: Set[DelegationAction] = set()
    for act in actions:
        try:
            action_set.add(DelegationAction(act))
        except ValueError:
            # Ignore unknown actions, they will be validated inside contract as well
            continue

    rule_id = contract.create_delegation(
        delegator_id=delegator_id,
        delegatee_id=delegatee_id,
        actions=action_set,
        scope=scope,
        conditions=conditions or {},
        duration_hours=duration_hours,
    )
    return rule_id


def record_delegation_usage(
    delegator_id: str,
    delegatee_id: str,
    action: DelegationAction,
    success: bool,
    context: Dict[str, Any] = None
):
    """Record delegation usage for audit"""
    contract = get_delegation_contract()
    contract.record_delegation_use(delegator_id, delegatee_id, action, success, context)