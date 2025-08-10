"""
SDK API - Interface for A2A SDK utilities and helpers
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)


class SdkAPI:
    """API interface for A2A SDK utilities and helpers"""
    
    def __init__(self):
        """Initialize SDK API"""
        logger.info("SDK API initialized")
    
    def create_agent_id(self, prefix: str = "agent") -> str:
        """
        Generate unique agent ID
        
        Args:
            prefix: Prefix for the agent ID
            
        Returns:
            Unique agent identifier
        """
        unique_id = str(uuid.uuid4()).replace('-', '')[:12]
        return f"{prefix}_{unique_id}"
    
    def create_message_id(self) -> str:
        """Generate unique message ID"""
        return str(uuid.uuid4())
    
    def create_context_id(self) -> str:
        """Generate unique context ID for workflow tracking"""
        return f"ctx_{str(uuid.uuid4()).replace('-', '')[:16]}"
    
    def validate_agent_card(self, agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate agent card structure and required fields
        
        Args:
            agent_card: Agent registration data
            
        Returns:
            Validation result with success status and errors
        """
        errors = []
        required_fields = [
            "agent_id", "name", "description", "version", "base_url"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in agent_card:
                errors.append(f"Missing required field: {field}")
            elif not agent_card[field]:
                errors.append(f"Empty required field: {field}")
        
        # Validate agent_id format
        if "agent_id" in agent_card:
            agent_id = agent_card["agent_id"]
            if not isinstance(agent_id, str) or len(agent_id) < 3:
                errors.append("agent_id must be a string with at least 3 characters")
        
        # Validate version format
        if "version" in agent_card:
            version = agent_card["version"]
            if not isinstance(version, str):
                errors.append("version must be a string")
        
        # Validate base_url format
        if "base_url" in agent_card:
            base_url = agent_card["base_url"]
            if not isinstance(base_url, str) or not base_url.startswith(('http://', 'https://')):
                errors.append("base_url must be a valid HTTP/HTTPS URL")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def create_success_response(self, data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """
        Create standardized success response
        
        Args:
            data: Response data
            message: Success message
            
        Returns:
            Standardized success response
        """
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if data is not None:
            response["data"] = data
            
        return response
    
    def create_error_response(self, error_code: int, error_message: str, 
                            details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized error response
        
        Args:
            error_code: HTTP error code
            error_message: Error description
            details: Additional error details
            
        Returns:
            Standardized error response
        """
        response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": error_message
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            response["error"]["details"] = details
            
        return response
    
    def format_dublin_core_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metadata according to Dublin Core standards
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Dublin Core compliant metadata
        """
        dublin_core = {}
        
        # Map common fields to Dublin Core elements
        field_mapping = {
            "title": "title",
            "creator": "creator",
            "subject": "subject",
            "description": "description", 
            "publisher": "publisher",
            "contributor": "contributor",
            "date": "date",
            "type": "type",
            "format": "format",
            "identifier": "identifier",
            "source": "source",
            "language": "language",
            "relation": "relation",
            "coverage": "coverage",
            "rights": "rights"
        }
        
        for original_key, dc_key in field_mapping.items():
            if original_key in metadata:
                dublin_core[dc_key] = metadata[original_key]
        
        # Ensure required fields have defaults
        if "date" not in dublin_core:
            dublin_core["date"] = datetime.utcnow().isoformat()
            
        if "identifier" not in dublin_core:
            dublin_core["identifier"] = f"dp-{str(uuid.uuid4()).replace('-', '')[:12]}"
            
        if "type" not in dublin_core:
            dublin_core["type"] = "Dataset"
        
        return dublin_core
    
    def create_ord_descriptor(self, dublin_core: Dict[str, Any],
                            additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create ORD (Open Resource Discovery) descriptor from Dublin Core metadata
        
        Args:
            dublin_core: Dublin Core metadata
            additional_metadata: Additional ORD-specific metadata
            
        Returns:
            ORD descriptor
        """
        descriptor = {
            "title": dublin_core.get("title", ""),
            "shortDescription": dublin_core.get("description", "")[:250],
            "description": dublin_core.get("description", ""),
            "version": "1.0.0",
            "releaseStatus": "active",
            "visibility": "internal"
        }
        
        # Add subjects as tags
        if "subject" in dublin_core:
            subjects = dublin_core["subject"]
            if isinstance(subjects, list):
                descriptor["tags"] = subjects
            elif isinstance(subjects, str):
                descriptor["tags"] = [subjects]
        
        # Add metadata labels
        descriptor["labels"] = {
            "dublin-core-compliant": ["true"],
            "data-type": ["structured"]
        }
        
        # Add documentation labels
        descriptor["documentationLabels"] = {
            "Creator": str(dublin_core.get("creator", "A2A System")),
            "Language": str(dublin_core.get("language", "en")),
            "Rights": str(dublin_core.get("rights", "Internal Use"))
        }
        
        # Merge additional metadata
        if additional_metadata:
            descriptor.update(additional_metadata)
        
        return descriptor
    
    def extract_capabilities_from_skills(self, skills: List[Dict[str, Any]]) -> List[str]:
        """
        Extract capability list from skill definitions
        
        Args:
            skills: List of skill definitions
            
        Returns:
            List of unique capabilities
        """
        capabilities = set()
        
        for skill in skills:
            if "capabilities" in skill:
                skill_caps = skill["capabilities"]
                if isinstance(skill_caps, list):
                    capabilities.update(skill_caps)
                elif isinstance(skill_caps, str):
                    capabilities.add(skill_caps)
        
        return sorted(list(capabilities))
    
    def create_agent_card_from_base(self, agent_base) -> Dict[str, Any]:
        """
        Create agent card from A2AAgentBase instance
        
        Args:
            agent_base: A2AAgentBase instance
            
        Returns:
            Agent card suitable for registry
        """
        try:
            # Extract skills and capabilities
            skills = []
            capabilities = set()
            
            for skill_name, skill_def in agent_base.skills.items():
                skill_info = {
                    "name": skill_name,
                    "description": skill_def.description,
                    "capabilities": skill_def.capabilities or [],
                    "domain": getattr(skill_def, 'domain', 'general')
                }
                skills.append(skill_info)
                capabilities.update(skill_def.capabilities or [])
            
            # Create agent card
            agent_card = {
                "agent_id": agent_base.agent_id,
                "name": agent_base.name,
                "description": agent_base.description,
                "version": agent_base.version,
                "base_url": agent_base.base_url,
                "capabilities": sorted(list(capabilities)),
                "skills": skills,
                "status": "active",
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "sdk_version": getattr(agent_base, 'sdk_version', '1.0.0')
                }
            }
            
            return agent_card
            
        except Exception as e:
            logger.error(f"Error creating agent card from base: {e}")
            return {
                "agent_id": getattr(agent_base, 'agent_id', 'unknown'),
                "name": getattr(agent_base, 'name', 'Unknown Agent'),
                "description": getattr(agent_base, 'description', ''),
                "version": getattr(agent_base, 'version', '1.0.0'),
                "base_url": getattr(agent_base, 'base_url', ''),
                "capabilities": [],
                "skills": [],
                "status": "active",
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "error": f"Incomplete agent card: {str(e)}"
                }
            }