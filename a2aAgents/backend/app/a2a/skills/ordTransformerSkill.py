"""
ORD/CSN Transformation Skill for A2A Agent
Implements SAP ORD document generation and CSN conversion
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from jsonschema import validate, ValidationError

from ..core.a2aTypes import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class ORDTransformerSkill:
    """
    Skill for transforming catalog data to SAP ORD format
    Maintains A2A protocol compliance
    """

    def __init__(self, system_namespace: str, system_type: str):
        self.system_namespace = system_namespace
        self.system_type = system_type
        self.ord_version = "1.9"

    def transform_to_ord(self, catalog_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform catalog metadata to ORD document format"""
        ord_document = self._initialize_ord_document()

        # Add data product as API resource
        api_ord_id = self._add_data_product_api(ord_document, catalog_metadata)

        # Add entity types if present
        if "entities" in catalog_metadata:
            self._add_entity_types(ord_document, catalog_metadata["entities"])

        # Add events if present
        if "events" in catalog_metadata:
            self._add_events(ord_document, catalog_metadata["events"])

        return ord_document

    def _initialize_ord_document(self) -> Dict[str, Any]:
        """Initialize ORD document structure"""
        return {
            "openResourceDiscovery": self.ord_version,
            "description": f"ORD document for {self.system_type}",
            "schema": "https://github.com/SAP/open-resource-discovery/spec-v1",
            "policyLevel": "sap:core:v1",
            "products": [{
                "ordId": f"{self.system_namespace}:product:{self.system_type}:v1",
                "title": f"{self.system_type} Product",
                "shortDescription": "Data product catalog",
                "vendor": self.system_namespace
            }],
            "packages": [],
            "apis": [],
            "events": [],
            "entityTypes": []
        }

    def _add_data_product_api(self, ord_document: Dict[str, Any],
                             catalog_metadata: Dict[str, Any]) -> str:
        """Add data product as API resource"""
        api_ord_id = f"{self.system_namespace}:api:{catalog_metadata['name']}:v1"

        api_resource = {
            "ordId": api_ord_id,
            "title": catalog_metadata.get("title", catalog_metadata["name"]),
            "shortDescription": catalog_metadata.get("description", "Data product API"),
            "systemInstanceAware": True,
            "apiProtocol": "odata-v4",
            "visibility": "public",
            "releaseStatus": "active",
            "resourceDefinitions": [{
                "type": "openapi-v3",
                "mediaType": "application/json",
                "url": f"/api/catalog/{catalog_metadata['name']}/openapi.json",
                "accessStrategies": [{"type": "open"}]
            }],
            "lastUpdate": datetime.utcnow().isoformat() + "Z"
        }

        ord_document["apis"].append(api_resource)
        return api_ord_id

    def _add_entity_types(self, ord_document: Dict[str, Any], entities: Dict[str, Any]):
        """Add entity types to ORD document"""
        for entity_name, entity_def in entities.items():
            entity_type = {
                "ordId": f"{self.system_namespace}:entityType:{entity_name}:v1",
                "localId": entity_name,
                "title": entity_def.get("title", entity_name),
                "shortDescription": entity_def.get("description", f"{entity_name} entity"),
                "visibility": "public",
                "releaseStatus": "active",
                "lastUpdate": datetime.utcnow().isoformat() + "Z"
            }
            ord_document["entityTypes"].append(entity_type)

    def _add_events(self, ord_document: Dict[str, Any], events: List[Dict[str, Any]]):
        """Add events to ORD document"""
        for event in events:
            ord_event = {
                "ordId": f"{self.system_namespace}:event:{event['name']}:v1",
                "title": event.get("title", event["name"]),
                "shortDescription": event.get("description", "Event"),
                "visibility": "public",
                "releaseStatus": "active",
                "lastUpdate": datetime.utcnow().isoformat() + "Z"
            }
            ord_document["events"].append(ord_event)

    def generate_ord_config(self, base_url: str) -> Dict[str, Any]:
        """Generate .well-known/open-resource-discovery configuration"""
        return {
            "openResourceDiscovery": self.ord_version,
            "baseUrl": base_url,
            "documents": [{
                "url": f"{base_url}/open-resource-discovery/v1/documents/ord-document.json",
                "systemInstanceAware": False,
                "accessStrategies": [{"type": "open"}]
            }]
        }

    def handle_a2a_ord_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle A2A messages for ORD transformation"""
        for part in message.parts:
            if part.kind == "ord_transform_request":
                catalog_data = part.data.get("catalog_metadata", {})
                ord_document = self.transform_to_ord(catalog_data)

                return A2AMessage(
                    role=MessageRole.AGENT,
                    contextId=message.contextId,
                    parts=[
                        MessagePart(
                            kind="ord_transform_response",
                            data={
                                "ord_document": ord_document,
                                "version": self.ord_version,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    ]
                )

        return None

    def get_skill_info(self) -> Dict[str, Any]:
        """Get skill information"""
        return {
            "id": "ord-transformer",
            "name": "ORD Transformer",
            "description": "Transforms catalog metadata to SAP ORD format",
            "capabilities": ["ord_generation", "ord_validation"],
            "version": self.ord_version,
            "tags": ["sap", "ord", "transformation"]
        }


class CSNTransformerSkill:
    """
    Skill for transforming catalog data to CAP CSN format
    """

    def __init__(self):
        self.csn_version = "1.0"
        self.type_mapping = {
            "string": "cds.String",
            "integer": "cds.Integer",
            "decimal": "cds.Decimal",
            "boolean": "cds.Boolean",
            "timestamp": "cds.Timestamp",
            "date": "cds.Date",
            "binary": "cds.Binary"
        }

    def transform_to_csn(self, catalog_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform catalog data to CSN format"""
        csn_document = {
            "csnInteropEffective": "1.0",
            "$version": "2.0",
            "meta": {
                "document": {
                    "version": "1.0.0",
                    "generator": "a2A-to-CSN Transformer",
                    "created": datetime.utcnow().isoformat()
                }
            },
            "definitions": {}
        }

        # Convert entities
        for entity_name, entity_def in catalog_data.get("entities", {}).items():
            csn_entity = self._convert_entity(entity_name, entity_def)
            csn_document["definitions"][entity_name] = csn_entity

        # Add services if present
        if "services" in catalog_data:
            self._add_services(csn_document, catalog_data["services"])

        return csn_document

    def _convert_entity(self, name: str, entity_def: Dict[str, Any]) -> Dict[str, Any]:
        """Convert single entity to CSN format"""
        csn_entity = {
            "kind": "entity",
            "@EndUserText.label": entity_def.get("label", name),
            "elements": {}
        }

        # Add annotations if present
        if "annotations" in entity_def:
            for ann_key, ann_value in entity_def["annotations"].items():
                csn_entity[f"@{ann_key}"] = ann_value

        # Convert fields
        for field_name, field_def in entity_def.get("fields", {}).items():
            csn_element = {
                "type": self.type_mapping.get(field_def["type"], "cds.String")
            }

            if field_def.get("key"):
                csn_element["key"] = True

            if field_def.get("length"):
                csn_element["length"] = field_def["length"]

            if field_def.get("nullable") is False:
                csn_element["notNull"] = True

            if field_def.get("default"):
                csn_element["default"] = {"val": field_def["default"]}

            csn_entity["elements"][field_name] = csn_element

        return csn_entity

    def _add_services(self, csn_document: Dict[str, Any], services: Dict[str, Any]):
        """Add service definitions to CSN document"""
        for service_name, service_def in services.items():
            csn_service = {
                "kind": "service",
                "@path": service_def.get("path", f"/{service_name}")
            }
            csn_document["definitions"][service_name] = csn_service

    def validate_csn(self, csn_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CSN document structure"""
        # Basic validation - in production, use full CSN schema
        required_fields = ["csnInteropEffective", "$version", "definitions"]
        missing = [f for f in required_fields if f not in csn_data]

        if missing:
            return {"valid": False, "errors": f"Missing required fields: {missing}"}

        return {"valid": True}

    def handle_a2a_csn_request(self, message: A2AMessage) -> Optional[A2AMessage]:
        """Handle A2A messages for CSN transformation"""
        for part in message.parts:
            if part.kind == "csn_transform_request":
                catalog_data = part.data.get("catalog_metadata", {})
                csn_document = self.transform_to_csn(catalog_data)
                validation = self.validate_csn(csn_document)

                return A2AMessage(
                    role=MessageRole.AGENT,
                    contextId=message.contextId,
                    parts=[
                        MessagePart(
                            kind="csn_transform_response",
                            data={
                                "csn_document": csn_document,
                                "validation": validation,
                                "version": self.csn_version,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    ]
                )

        return None

    def get_skill_info(self) -> Dict[str, Any]:
        """Get skill information"""
        return {
            "id": "csn-transformer",
            "name": "CSN Transformer",
            "description": "Transforms catalog metadata to CAP CSN format",
            "capabilities": ["csn_generation", "csn_validation"],
            "version": self.csn_version,
            "tags": ["sap", "cap", "csn", "transformation"]
        }
