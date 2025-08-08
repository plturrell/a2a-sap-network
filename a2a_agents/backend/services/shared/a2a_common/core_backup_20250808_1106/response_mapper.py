"""
Platform API Response Mapping and Verification
Maps platform-specific responses to standardized formats
"""

import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from .data_validation import ValidationResult

logger = logging.getLogger(__name__)


class ResponseMappingError(Exception):
    """Raised when response mapping fails"""
    pass


class PlatformResponseMapper(ABC):
    """Abstract base class for platform response mappers"""
    
    @abstractmethod
    def map_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map successful platform response to standard format"""
        pass
    
    @abstractmethod
    def map_error_response(self, status_code: int, response: Any) -> Dict[str, Any]:
        """Map error response to standard format"""
        pass
    
    @abstractmethod
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate platform response structure"""
        pass


class SAPDatasphereResponseMapper(PlatformResponseMapper):
    """Response mapper for SAP Datasphere"""
    
    def map_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map Datasphere response to standard format"""
        mapped = {
            "status": "success",
            "platform": "sap_datasphere",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        # Map based on response type
        if "space" in response and "asset" in response:
            # Asset creation/update response
            mapped["data"] = {
                "entity_id": response.get("asset", {}).get("id"),
                "entity_type": response.get("asset", {}).get("type"),
                "space": response.get("space"),
                "version": response.get("asset", {}).get("version"),
                "status": response.get("status", "active")
            }
        elif "ordDocument" in response:
            # ORD document response
            mapped["data"] = {
                "document_id": response.get("ordDocument", {}).get("id"),
                "document_url": response.get("ordDocument", {}).get("url"),
                "validation_status": response.get("validationStatus", "valid")
            }
        elif "catalogEntry" in response:
            # Catalog entry response
            entry = response["catalogEntry"]
            mapped["data"] = {
                "catalog_id": entry.get("id"),
                "name": entry.get("name"),
                "type": entry.get("type"),
                "metadata": entry.get("metadata", {})
            }
        else:
            # Generic response
            mapped["data"] = response
        
        return mapped
    
    def map_error_response(self, status_code: int, response: Any) -> Dict[str, Any]:
        """Map Datasphere error response"""
        error_data = {
            "status": "error",
            "platform": "sap_datasphere",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "code": status_code,
                "message": "Unknown error"
            }
        }
        
        if isinstance(response, dict):
            # SAP standard error format
            if "error" in response:
                error = response["error"]
                error_data["error"]["message"] = error.get("message", "Unknown error")
                error_data["error"]["details"] = error.get("details", [])
                error_data["error"]["correlation_id"] = error.get("correlationId")
            elif "message" in response:
                error_data["error"]["message"] = response["message"]
        elif isinstance(response, str):
            error_data["error"]["message"] = response
        
        # Add specific handling for common errors
        if status_code == 401:
            error_data["error"]["type"] = "authentication_error"
            error_data["error"]["action"] = "refresh_token"
        elif status_code == 403:
            error_data["error"]["type"] = "authorization_error"
            error_data["error"]["action"] = "check_permissions"
        elif status_code == 429:
            error_data["error"]["type"] = "rate_limit_error"
            error_data["error"]["retry_after"] = response.get("Retry-After", 60)
        
        return error_data
    
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate Datasphere response structure"""
        result = ValidationResult(True)
        
        # Check for required fields based on response type
        if "space" in response and "asset" in response:
            # Asset response validation
            asset = response.get("asset", {})
            if not asset.get("id"):
                result.add_error("Asset response missing 'id'")
            if not asset.get("type"):
                result.add_error("Asset response missing 'type'")
        elif "ordDocument" in response:
            # ORD document validation
            doc = response.get("ordDocument", {})
            if not doc.get("id"):
                result.add_error("ORD document missing 'id'")
        
        return result


class DatabricksUnityResponseMapper(PlatformResponseMapper):
    """Response mapper for Databricks Unity Catalog"""
    
    def map_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map Unity Catalog response to standard format"""
        mapped = {
            "status": "success",
            "platform": "databricks_unity",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        # Map based on response type
        if "table_type" in response:
            # Table response
            mapped["data"] = {
                "entity_id": response.get("full_name"),
                "entity_type": "table",
                "catalog": response.get("catalog_name"),
                "schema": response.get("schema_name"),
                "name": response.get("name"),
                "table_type": response.get("table_type"),
                "data_source_format": response.get("data_source_format"),
                "created_at": response.get("created_at"),
                "updated_at": response.get("updated_at")
            }
        elif "catalogs" in response:
            # Catalog list response
            mapped["data"] = {
                "catalogs": [
                    {
                        "name": cat.get("name"),
                        "owner": cat.get("owner"),
                        "comment": cat.get("comment")
                    }
                    for cat in response.get("catalogs", [])
                ]
            }
        elif "metastore_id" in response:
            # Metastore response
            mapped["data"] = {
                "metastore_id": response.get("metastore_id"),
                "name": response.get("name"),
                "storage_root": response.get("storage_root"),
                "owner": response.get("owner")
            }
        else:
            # Generic response
            mapped["data"] = response
        
        return mapped
    
    def map_error_response(self, status_code: int, response: Any) -> Dict[str, Any]:
        """Map Unity Catalog error response"""
        error_data = {
            "status": "error",
            "platform": "databricks_unity",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "code": status_code,
                "message": "Unknown error"
            }
        }
        
        if isinstance(response, dict):
            # Databricks error format
            error_data["error"]["message"] = response.get("message", "Unknown error")
            error_data["error"]["error_code"] = response.get("error_code")
            
            if "details" in response:
                error_data["error"]["details"] = response["details"]
        
        # Specific error handling
        if status_code == 404:
            error_data["error"]["type"] = "not_found"
            error_data["error"]["action"] = "check_resource_exists"
        elif status_code == 409:
            error_data["error"]["type"] = "conflict"
            error_data["error"]["action"] = "resolve_conflict"
        
        return error_data
    
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate Unity Catalog response"""
        result = ValidationResult(True)
        
        if "table_type" in response:
            # Table validation
            if not response.get("full_name"):
                result.add_error("Table response missing 'full_name'")
            if not response.get("catalog_name"):
                result.add_error("Table response missing 'catalog_name'")
        
        return result


class SAPHANAResponseMapper(PlatformResponseMapper):
    """Response mapper for SAP HANA Cloud"""
    
    def map_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map HANA response to standard format"""
        mapped = {
            "status": "success",
            "platform": "sap_hana",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        if "container" in response:
            # HDI container response
            mapped["data"] = {
                "container_id": response.get("container", {}).get("id"),
                "container_name": response.get("container", {}).get("name"),
                "deployment_id": response.get("deploymentId"),
                "status": response.get("status", "deployed")
            }
        elif "artifacts" in response:
            # Artifacts deployment response
            mapped["data"] = {
                "deployed_artifacts": response.get("artifacts", []),
                "deployment_time": response.get("deploymentTime"),
                "warnings": response.get("warnings", [])
            }
        else:
            mapped["data"] = response
        
        return mapped
    
    def map_error_response(self, status_code: int, response: Any) -> Dict[str, Any]:
        """Map HANA error response"""
        error_data = {
            "status": "error",
            "platform": "sap_hana",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "code": status_code,
                "message": "Unknown error"
            }
        }
        
        if isinstance(response, dict):
            if "error" in response:
                error_data["error"]["message"] = response["error"].get("message", "Unknown error")
                error_data["error"]["code"] = response["error"].get("code")
        
        return error_data
    
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate HANA response"""
        result = ValidationResult(True)
        
        if "container" in response:
            container = response.get("container", {})
            if not container.get("id"):
                result.add_error("Container response missing 'id'")
        
        return result


class ClouderaAtlasResponseMapper(PlatformResponseMapper):
    """Response mapper for Cloudera Atlas"""
    
    def map_success_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Map Atlas response to standard format"""
        mapped = {
            "status": "success",
            "platform": "cloudera_atlas",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        if "entities" in response:
            # Entity creation response
            entities = response.get("entities", [])
            mapped["data"] = {
                "created_entities": len(entities),
                "entity_guids": [e.get("guid") for e in entities],
                "entity_types": [e.get("typeName") for e in entities]
            }
        elif "entity" in response:
            # Single entity response
            entity = response["entity"]
            mapped["data"] = {
                "entity_guid": entity.get("guid"),
                "entity_type": entity.get("typeName"),
                "qualified_name": entity.get("attributes", {}).get("qualifiedName"),
                "status": entity.get("status", "ACTIVE")
            }
        else:
            mapped["data"] = response
        
        return mapped
    
    def map_error_response(self, status_code: int, response: Any) -> Dict[str, Any]:
        """Map Atlas error response"""
        error_data = {
            "status": "error",
            "platform": "cloudera_atlas",
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "code": status_code,
                "message": "Unknown error"
            }
        }
        
        if isinstance(response, dict):
            error_data["error"]["message"] = response.get("errorMessage", "Unknown error")
            error_data["error"]["error_code"] = response.get("errorCode")
        
        return error_data
    
    def validate_response(self, response: Dict[str, Any]) -> ValidationResult:
        """Validate Atlas response"""
        result = ValidationResult(True)
        
        if "entities" in response:
            entities = response.get("entities", [])
            for i, entity in enumerate(entities):
                if not entity.get("guid"):
                    result.add_error(f"Entity {i} missing 'guid'")
        
        return result


class ResponseMapperRegistry:
    """Registry for platform response mappers"""
    
    def __init__(self):
        self.mappers: Dict[str, PlatformResponseMapper] = {
            "sap_datasphere": SAPDatasphereResponseMapper(),
            "databricks": DatabricksUnityResponseMapper(),
            "sap_hana": SAPHANAResponseMapper(),
            "cloudera": ClouderaAtlasResponseMapper()
        }
    
    def get_mapper(self, platform_type: str) -> Optional[PlatformResponseMapper]:
        """Get mapper for platform type"""
        return self.mappers.get(platform_type)
    
    def map_response(self, platform_type: str, response: Dict[str, Any], 
                    is_error: bool = False, status_code: int = 200) -> Dict[str, Any]:
        """Map platform response using appropriate mapper"""
        mapper = self.get_mapper(platform_type)
        if not mapper:
            logger.warning(f"No mapper found for platform: {platform_type}")
            return {"status": "unmapped", "data": response}
        
        try:
            if is_error:
                return mapper.map_error_response(status_code, response)
            else:
                # Validate first
                validation = mapper.validate_response(response)
                if not validation.is_valid:
                    logger.warning(f"Response validation failed: {validation.errors}")
                
                return mapper.map_success_response(response)
        except Exception as e:
            logger.error(f"Response mapping failed: {e}")
            raise ResponseMappingError(f"Failed to map {platform_type} response: {e}")
    
    def validate_response(self, platform_type: str, 
                         response: Dict[str, Any]) -> ValidationResult:
        """Validate platform response"""
        mapper = self.get_mapper(platform_type)
        if not mapper:
            result = ValidationResult(False)
            result.add_error(f"No mapper found for platform: {platform_type}")
            return result
        
        return mapper.validate_response(response)


# Global mapper registry
_mapper_registry = None

def get_response_mapper_registry() -> ResponseMapperRegistry:
    """Get global response mapper registry"""
    global _mapper_registry
    if _mapper_registry is None:
        _mapper_registry = ResponseMapperRegistry()
    return _mapper_registry