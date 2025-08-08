"""
Trust-Based Help Request Validator for A2A System
Validates help requests according to smart contract trust framework and delegation authorities
Ensures agents can only ask for help on skills they DON'T have according to trust contract
"""

import logging
from typing import Dict, List, Set, Optional, Any
from enum import Enum

from ..security.delegation_contracts import get_delegation_contract, DelegationAction

logger = logging.getLogger(__name__)


class HelpRequestType(str, Enum):
    """Types of help requests that can be made"""
    SKILL_GUIDANCE = "skill_guidance"  # How to use a skill the requester doesn't have
    STATUS_INQUIRY = "status_inquiry"  # What did you do? How did you do it?
    CAPABILITY_QUESTION = "capability_question"  # What can you do?
    TROUBLESHOOTING = "troubleshooting"  # Help with problems in requester's domain


class TrustBasedHelpValidator:
    """Validates help requests against smart contract trust boundaries and delegation authorities"""
    
    def __init__(self):
        # Delay delegation contract initialization to avoid circular dependency
        self.delegation_contract = None
        
        # Define skill categories and mappings
        self.skill_categories = {
            # Data Processing Skills
            "data_processing": {
                "dublin-core-extraction", "cds-csn-generation", "ord-descriptor-creation-with-dublin-core",
                "catalog-registration-enhanced", "metadata-quality-assessment"
            },
            
            # Data Standardization Skills  
            "data_standardization": {
                "location-standardization", "account-standardization", "product-standardization",
                "book-standardization", "measure-standardization", "batch-standardization"
            },
            
            # Data Management Skills
            "data_management": {
                "crud-operations", "file-management", "database-management", 
                "dual-storage", "service-levels"
            },
            
            # Metadata Management Skills
            "metadata_management": {
                "metadata-registration", "metadata-enhancement", "quality-assessment",
                "search-operations", "ord-repository"
            },
            
            # Infrastructure Skills
            "infrastructure": {
                "smart-delegation", "a2a-orchestration", "ai-advisor"
            }
        }
        
        # Map delegation actions to skill categories
        self.delegation_to_skills = {
            DelegationAction.DATA_STORAGE: "data_management",
            DelegationAction.DATA_RETRIEVAL: "data_management", 
            DelegationAction.DATA_ARCHIVAL: "data_management",
            DelegationAction.METADATA_REGISTRATION: "metadata_management",
            DelegationAction.METADATA_ENHANCEMENT: "metadata_management",
            DelegationAction.QUALITY_ASSESSMENT: "metadata_management",
            DelegationAction.SEARCH_OPERATIONS: "metadata_management",
            DelegationAction.AI_CONSULTATION: "infrastructure",
            DelegationAction.STATUS_REPORTING: "infrastructure"
        }
        
        # Define agent skill profiles
        self.agent_skills = {
            "data_product_agent_0": {
                "dublin-core-extraction", "cds-csn-generation", "ord-descriptor-creation-with-dublin-core",
                "catalog-registration-enhanced", "metadata-quality-assessment", "a2a-orchestration", 
                "smart-delegation", "ai-advisor"
            },
            "financial_standardization_agent_1": {
                "location-standardization", "account-standardization", "product-standardization",
                "book-standardization", "measure-standardization", "batch-standardization", "ai-advisor"
            },
            "data_manager_agent": {
                "crud-operations", "file-management", "database-management", 
                "dual-storage", "service-levels", "ai-advisor"
            },
            "catalog_manager_agent": {
                "metadata-registration", "metadata-enhancement", "quality-assessment",
                "search-operations", "ord-repository", "ai-advisor"
            }
        }
    
    def validate_help_request(
        self, 
        requesting_agent_id: str,
        target_agent_id: str,
        help_type: HelpRequestType,
        requested_skill: Optional[str] = None,
        problem_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate if a help request is within proper scope boundaries
        
        Returns:
            {
                "valid": bool,
                "reason": str,
                "allowed_help_types": List[str],
                "skill_compatibility": Dict[str, Any]
            }
        """
        
        try:
            # Get agent skill profiles
            requester_skills = self.agent_skills.get(requesting_agent_id, set())
            target_skills = self.agent_skills.get(target_agent_id, set())
            
            if not requester_skills or not target_skills:
                return {
                    "valid": False,
                    "reason": "Unknown agent in help request",
                    "allowed_help_types": [],
                    "skill_compatibility": {}
                }
            
            # Validate based on help request type
            if help_type == HelpRequestType.SKILL_GUIDANCE:
                return self._validate_skill_guidance(
                    requesting_agent_id, target_agent_id, 
                    requester_skills, target_skills, requested_skill
                )
                
            elif help_type == HelpRequestType.STATUS_INQUIRY:
                return self._validate_status_inquiry(
                    requesting_agent_id, target_agent_id,
                    requester_skills, target_skills
                )
                
            elif help_type == HelpRequestType.CAPABILITY_QUESTION:
                return self._validate_capability_question(
                    requesting_agent_id, target_agent_id,
                    requester_skills, target_skills
                )
                
            elif help_type == HelpRequestType.TROUBLESHOOTING:
                return self._validate_troubleshooting(
                    requesting_agent_id, target_agent_id,
                    requester_skills, target_skills, problem_context
                )
            
            else:
                return {
                    "valid": False,
                    "reason": f"Unknown help request type: {help_type}",
                    "allowed_help_types": [t.value for t in HelpRequestType],
                    "skill_compatibility": {}
                }
                
        except Exception as e:
            logger.error(f"Error validating help request: {e}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}",
                "allowed_help_types": [],
                "skill_compatibility": {}
            }
    
    def _validate_skill_guidance(
        self, 
        requesting_agent_id: str,
        target_agent_id: str,
        requester_skills: Set[str],
        target_skills: Set[str],
        requested_skill: Optional[str]
    ) -> Dict[str, Any]:
        """Validate skill guidance request - requester can only ask about skills they DON'T have"""
        
        if not requested_skill:
            return {
                "valid": False,
                "reason": "Skill guidance requires specific skill to be specified",
                "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value, HelpRequestType.CAPABILITY_QUESTION.value],
                "skill_compatibility": {
                    "requester_has_skill": False,
                    "target_has_skill": False
                }
            }
        
        # Check if requester already has this skill
        if requested_skill in requester_skills:
            return {
                "valid": False,
                "reason": f"Cannot ask for help on skill '{requested_skill}' - you already have this skill",
                "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value, HelpRequestType.CAPABILITY_QUESTION.value],
                "skill_compatibility": {
                    "requester_has_skill": True,
                    "target_has_skill": requested_skill in target_skills,
                    "violation": "requesting_own_skill"
                }
            }
        
        # Check if target has this skill
        if requested_skill not in target_skills:
            return {
                "valid": False,
                "reason": f"Target agent does not have skill '{requested_skill}' to provide help",
                "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value, HelpRequestType.CAPABILITY_QUESTION.value],
                "skill_compatibility": {
                    "requester_has_skill": False,
                    "target_has_skill": False,
                    "violation": "target_lacks_skill"
                }
            }
        
        # Check delegation authority
        delegation_valid = self._check_delegation_authority(requesting_agent_id, target_agent_id, requested_skill)
        
        return {
            "valid": True,
            "reason": f"Valid skill guidance request for '{requested_skill}'",
            "allowed_help_types": [HelpRequestType.SKILL_GUIDANCE.value],
            "skill_compatibility": {
                "requester_has_skill": False,
                "target_has_skill": True,
                "delegation_authorized": delegation_valid,
                "skill_category": self._get_skill_category(requested_skill)
            }
        }
    
    def _validate_status_inquiry(
        self,
        requesting_agent_id: str,
        target_agent_id: str,
        requester_skills: Set[str],
        target_skills: Set[str]
    ) -> Dict[str, Any]:
        """Validate status inquiry - agents can ask 'what did you do?' regardless of skills"""
        
        # Status inquiries are always allowed - agents can ask about others' work
        return {
            "valid": True,
            "reason": "Status inquiries are always permitted",
            "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value],
            "skill_compatibility": {
                "inquiry_type": "status_report",
                "delegation_required": False
            }
        }
    
    def _validate_capability_question(
        self,
        requesting_agent_id: str,
        target_agent_id: str,
        requester_skills: Set[str],
        target_skills: Set[str]
    ) -> Dict[str, Any]:
        """Validate capability question - agents can ask 'what can you do?' regardless of skills"""
        
        # Capability questions are always allowed
        return {
            "valid": True,
            "reason": "Capability questions are always permitted",
            "allowed_help_types": [HelpRequestType.CAPABILITY_QUESTION.value],
            "skill_compatibility": {
                "inquiry_type": "capability_discovery",
                "delegation_required": False
            }
        }
    
    def _validate_troubleshooting(
        self,
        requesting_agent_id: str,
        target_agent_id: str,
        requester_skills: Set[str],
        target_skills: Set[str],
        problem_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate troubleshooting request - must be about requester's own skills"""
        
        if not problem_context:
            return {
                "valid": False,
                "reason": "Troubleshooting requests require problem context",
                "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value, HelpRequestType.CAPABILITY_QUESTION.value],
                "skill_compatibility": {}
            }
        
        problem_skill = problem_context.get("skill")
        if not problem_skill:
            return {
                "valid": False,
                "reason": "Troubleshooting requests require specific skill context",
                "allowed_help_types": [HelpRequestType.STATUS_INQUIRY.value, HelpRequestType.CAPABILITY_QUESTION.value],
                "skill_compatibility": {}
            }
        
        # Requester must have the skill they're asking for help with
        if problem_skill not in requester_skills:
            return {
                "valid": False,
                "reason": f"Cannot ask for troubleshooting help on skill '{problem_skill}' - you don't have this skill",
                "allowed_help_types": [HelpRequestType.SKILL_GUIDANCE.value],
                "skill_compatibility": {
                    "requester_has_skill": False,
                    "violation": "troubleshooting_without_skill"
                }
            }
        
        return {
            "valid": True,
            "reason": f"Valid troubleshooting request for owned skill '{problem_skill}'",
            "allowed_help_types": [HelpRequestType.TROUBLESHOOTING.value],
            "skill_compatibility": {
                "requester_has_skill": True,
                "problem_skill": problem_skill,
                "help_type": "troubleshooting"
            }
        }
    
    def _check_delegation_authority(
        self, 
        requesting_agent_id: str, 
        target_agent_id: str, 
        skill: str
    ) -> bool:
        """Check if there's delegation authority for the skill"""
        
        try:
            # Initialize delegation contract if not already done
            if self.delegation_contract is None:
                self.delegation_contract = get_delegation_contract()
            
            # Map skill to delegation action
            skill_category = self._get_skill_category(skill)
            
            # Check if there's an existing delegation relationship
            for delegation_rule in self.delegation_contract.delegation_rules.values():
                if (delegation_rule.delegator_id == requesting_agent_id and 
                    delegation_rule.delegatee_id == target_agent_id and
                    delegation_rule.is_valid()):
                    
                    # Check if the delegation covers the skill category
                    for action in delegation_rule.actions:
                        if self.delegation_to_skills.get(action) == skill_category:
                            return True
        except Exception as e:
            logger.warning(f"Could not check delegation authority: {e}")
            # Return True as fallback to allow help requests
            return True
        
        return False
    
    def _get_skill_category(self, skill: str) -> Optional[str]:
        """Get the category for a specific skill"""
        for category, skills in self.skill_categories.items():
            if skill in skills:
                return category
        return None
    
    def get_allowed_help_requests(
        self, 
        requesting_agent_id: str,
        target_agent_id: str
    ) -> Dict[str, Any]:
        """Get all allowed help request types between two agents"""
        
        requester_skills = self.agent_skills.get(requesting_agent_id, set())
        target_skills = self.agent_skills.get(target_agent_id, set())
        
        # Skills requester can ask for guidance on (skills they don't have that target has)
        can_request_guidance = target_skills - requester_skills
        
        # Skills requester can ask for troubleshooting help on (skills they have)
        can_request_troubleshooting = requester_skills.intersection(target_skills)
        
        return {
            "status_inquiry": True,  # Always allowed
            "capability_question": True,  # Always allowed
            "skill_guidance": {
                "allowed": len(can_request_guidance) > 0,
                "available_skills": list(can_request_guidance)
            },
            "troubleshooting": {
                "allowed": len(can_request_troubleshooting) > 0,
                "available_skills": list(can_request_troubleshooting)
            },
            "agent_skills": {
                "requester": list(requester_skills),
                "target": list(target_skills)
            }
        }


# Global validator instance
_skill_scope_validator = None

def get_trust_based_help_validator() -> TrustBasedHelpValidator:
    """Get global trust-based help validator instance"""
    global _skill_scope_validator
    if _skill_scope_validator is None:
        _skill_scope_validator = TrustBasedHelpValidator()
    return _skill_scope_validator