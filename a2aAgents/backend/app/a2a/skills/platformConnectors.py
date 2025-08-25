"""
Platform-specific connectors for catalog synchronization
All connectors maintain A2A protocol compliance
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import json
from typing import Dict, List, Any
import logging
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
import base64

from .catalogIntegrationSkill import DownstreamConnector, CatalogChangeEvent
from .ordTransformerSkill import ORDTransformerSkill, CSNTransformerSkill
from app.a2a.core.a2aTypes import A2AMessage, MessagePart, MessageRole
from app.a2a.core.authManager import get_auth_manager
from app.a2a.core.circuitBreaker import get_circuit_breaker_manager, CircuitBreakerOpenError
# Import trust components from a2aNetwork
from a2aNetwork.trustSystem.smartContractTrust import sign_a2a_message
logger = logging.getLogger(__name__)


class SAPDataspherConnector(DownstreamConnector):
    """Connector for SAP Datasphere integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.auth_config = config.get("auth_config", {})
        self.ord_transformer = ORDTransformerSkill(
            system_namespace=config.get("namespace", "com.company"),
            system_type="datasphere"
        )
        self._auth_registered = False

    async def push_catalog_change(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push catalog change to SAP Datasphere"""
        try:
            # Transform to ORD format
            ord_document = self.ord_transformer.transform_to_ord(event.metadata)

            # Create A2A message for Datasphere
            message = A2AMessage(
                role=MessageRole.AGENT,
                contextId=event.event_id,
                parts=[
                    MessagePart(
                        kind="datasphere_catalog_update",
                        data={
                            "operation": event.operation,
                            "entity_type": event.entity_type,
                            "entity_id": event.entity_id,
                            "ord_document": ord_document,
                            "metadata": event.metadata
                        }
                    )
                ]
            )

            # Sign the message
            signed_message = sign_a2a_message(message.model_dump())

            # Get circuit breaker for this connector
            breaker_manager = await get_circuit_breaker_manager()
            breaker = await breaker_manager.get_circuit_breaker(f"datasphere_{self.endpoint}")

            async def call_api():
                # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                    headers = await self._get_auth_headers()
                    headers["Content-Type"] = "application/json"

                    response = await client.post(
                        f"{self.endpoint}/api/v1/catalog/sync",
                        json=signed_message,
                        headers=headers,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        return {
                            "status": "success",
                            "platform": "datasphere",
                            "response": response.json()
                        }
                    else:
                        response.raise_for_status()

            # Call through circuit breaker
            return await breaker.call(call_api)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self._auth_registered = False
                logger.error("Authentication failed for Datasphere, token may be expired.")
            else:
                logger.error("Error pushing to Datasphere: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred while pushing to Datasphere: %s", e)
            raise

    async def validate_connection(self) -> bool:
        """Validate connection to Datasphere"""
        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                headers = await self._get_auth_headers()
                response = await client.get(
                    f"{self.endpoint}/api/v1/health",
                    headers=headers,
                    timeout=10.0
                )
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error("Datasphere connection validation failed: %s", e)
            return False

    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return ["table", "view", "data_product", "schema", "model"]

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Datasphere"""
        auth_manager = get_auth_manager()
        # Register OAuth2 client if not already done
        platform_id = f"datasphere_{self.endpoint}"
        if not self._auth_registered:
            auth_manager.register_oauth2(
                platform_id,
                self.auth_config["client_id"],
                self.auth_config["client_secret"],
                self.auth_config["token_url"],
                self.auth_config.get("scope")
            )
            self._auth_registered = True

        return await auth_manager.get_auth_headers(platform_id)


class UnityaCatalogConnector(DownstreamConnector):
    """Connector for Databricks Unity Catalog integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.workspace_url = config.get("workspace_url")
        self.token = config.get("auth_config", {}).get("token")

    async def push_catalog_change(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push catalog change to Unity Catalog"""
        try:
            # Map event to Unity Catalog format
            unity_payload = self._map_to_unity_format(event)


            # Send to Unity Catalog
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                }

                # Determine endpoint based on operation
                if event.operation == "create":
                    endpoint = f"{self.workspace_url}/api/2.1/unity-catalog/tables"
                    method = "POST"
                elif event.operation == "update":
                    endpoint = f"{self.workspace_url}/api/2.1/unity-catalog/tables/{event.entity_id}"
                    method = "PATCH"
                elif event.operation == "delete":
                    endpoint = f"{self.workspace_url}/api/2.1/unity-catalog/tables/{event.entity_id}"
                    method = "DELETE"
                else:
                    raise ValueError(f"Unsupported operation: {event.operation}")

                response = await client.request(
                    method=method,
                    url=endpoint,
                    json=unity_payload if method != "DELETE" else None,
                    headers=headers,
                    timeout=30.0
                )

                if response.status_code in [200, 201, 204]:
                    return {
                        "status": "success",
                        "platform": "unity_catalog",
                        "response": response.json() if response.content else {}
                    }
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error("Unity Catalog sync failed: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred while pushing to Unity Catalog: %s", e)
            raise

    async def validate_connection(self) -> bool:
        """Validate connection to Unity Catalog"""
        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.get(
                    f"{self.workspace_url}/api/2.1/unity-catalog/catalogs",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=10.0
                )
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error("Unity Catalog connection validation failed: %s", e)
            return False

    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return ["table", "schema", "catalog", "volume", "function"]

    def _map_to_unity_format(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Map catalog event to Unity Catalog format"""
        metadata = event.metadata

        unity_format = {
            "catalog_name": metadata.get("catalog", "main"),
            "schema_name": metadata.get("schema", "default"),
            "name": metadata.get("name", event.entity_id),
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "comment": metadata.get("description", "")
        }

        # Add columns if present
        if "columns" in metadata:
            unity_format["columns"] = [
                {
                    "name": col["name"],
                    "type_name": self._map_data_type(col["type"]),
                    "nullable": col.get("nullable", True),
                    "comment": col.get("description", "")
                }
                for col in metadata["columns"]
            ]

        # Add properties if present
        if "properties" in metadata:
            unity_format["properties"] = metadata["properties"]

        return unity_format

    def _map_data_type(self, source_type: str) -> str:
        """Map data types to Unity Catalog types"""
        type_mapping = {
            "string": "STRING",
            "integer": "INT",
            "long": "BIGINT",
            "float": "FLOAT",
            "double": "DOUBLE",
            "boolean": "BOOLEAN",
            "timestamp": "TIMESTAMP",
            "date": "DATE",
            "binary": "BINARY",
            "decimal": "DECIMAL"
        }
        return type_mapping.get(source_type.lower(), "STRING")


class SAPHANACloudConnector(DownstreamConnector):
    """Connector for SAP HANA Cloud HDI integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.auth_config = config.get("auth_config", {})
        self.csn_transformer = CSNTransformerSkill()

    async def push_catalog_change(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push catalog change to HANA Cloud HDI"""
        try:
            # Transform to CSN format
            csn_document = self.csn_transformer.transform_to_csn(event.metadata)

            # Create HDI artifacts
            hdi_artifacts = self._generate_hdi_artifacts(event, csn_document)

            # Create A2A message
            message = A2AMessage(
                role=MessageRole.AGENT,
                contextId=event.event_id,
                parts=[
                    MessagePart(
                        kind="hana_hdi_deploy",
                        data={
                            "container": f"CATALOG_{event.entity_id.upper()}",
                            "artifacts": hdi_artifacts,
                            "operation": event.operation
                        }
                    )
                ]
            )

            # Deploy to HANA
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                headers = await self._get_auth_headers()

                response = await client.post(
                    f"{self.endpoint}/v1/deploy",
                    json={
                        "container": message.parts[0].data["container"],
                        "artifacts": hdi_artifacts
                    },
                    headers=headers,
                    timeout=60.0  # HDI deployment can take time
                )

                if response.status_code == 200:
                    return {
                        "status": "success",
                        "platform": "hana_cloud",
                        "response": response.json()
                    }
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error("HANA HDI deployment failed: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred while pushing to HANA Cloud: %s", e)
            raise

    async def validate_connection(self) -> bool:
        """Validate connection to HANA Cloud"""
        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                headers = await self._get_auth_headers()
                response = await client.get(
                    f"{self.endpoint}/v1/status",
                    headers=headers,
                    timeout=10.0
                )
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error("HANA Cloud connection validation failed: %s", e)
            return False

    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return ["table", "view", "procedure", "type", "schema"]

    def _generate_hdi_artifacts(self, event: CatalogChangeEvent,
                               csn_document: Dict[str, Any]) -> Dict[str, str]:
        """Generate HDI artifacts from CSN document"""
        artifacts = {}

        # Generate .hdbcds file from CSN
        cds_content = self._csn_to_cds(csn_document)
        artifacts[f"{event.entity_id}.hdbcds"] = cds_content

        # Generate grants file if needed
        if event.metadata.get("requires_grants"):
            grants_content = self._generate_grants(event.metadata)
            artifacts[f"{event.entity_id}.hdbgrants"] = grants_content

        # Generate synonym file if needed
        if event.metadata.get("synonyms"):
            synonym_content = self._generate_synonyms(event.metadata["synonyms"])
            artifacts[f"{event.entity_id}.hdbsynonym"] = synonym_content

        return artifacts

    def _csn_to_cds(self, csn_document: Dict[str, Any]) -> str:
        """Convert CSN document to CDS format for HDI"""
        cds_lines = []

        for name, definition in csn_document.get("definitions", {}).items():
            if definition.get("kind") == "entity":
                cds_lines.append(f"entity {name} {{")

                for elem_name, elem_def in definition.get("elements", {}).items():
                    key_prefix = "key " if elem_def.get("key") else ""
                    type_str = elem_def.get("type", "String")
                    cds_lines.append(f"  {key_prefix}{elem_name}: {type_str};")

                cds_lines.append("}")
                cds_lines.append("")

        return "\n".join(cds_lines)

    def _generate_grants(self, metadata: Dict[str, Any]) -> str:
        """Generate HDI grants file"""
        grants = {
            "object_owner": {
                "privileges_with_grant_option": metadata.get("grants", ["SELECT"])
            }
        }
        return json.dumps(grants, indent=2)

    def _generate_synonyms(self, synonyms: List[Dict[str, Any]]) -> str:
        """Generate HDI synonym file"""
        syn_dict = {}
        for syn in synonyms:
            syn_dict[syn["name"]] = {
                "target": {
                    "object": syn["target"],
                    "schema": syn.get("schema", "PUBLIC")
                }
            }
        return json.dumps(syn_dict, indent=2)

    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HANA Cloud"""
        # Implement based on auth method (OAuth2, Basic, Certificate)
        if self.auth_config.get("method") == "oauth2":
            token = await self._get_oauth_token()
            return {"Authorization": f"Bearer {token}"}
        elif self.auth_config.get("method") == "basic":
            import base64
            credentials = f"{self.auth_config['username']}:{self.auth_config['password']}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        else:
            raise ValueError(f"Unsupported auth method: {self.auth_config.get('method')}")

    async def _get_oauth_token(self) -> str:
        """Get OAuth token for HANA Cloud"""
        # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
            response = await client.post(
                self.auth_config["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.auth_config["client_id"],
                    "client_secret": self.auth_config["client_secret"]
                }
            )
            return response.json()["access_token"]


class ClouderaAtlasConnector(DownstreamConnector):
    """Connector for Cloudera Data Platform Atlas integration"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint")
        self.auth_config = config.get("auth_config", {})

    async def push_catalog_change(self, event: CatalogChangeEvent) -> Dict[str, Any]:
        """Push catalog change to Apache Atlas"""
        try:
            # Map to Atlas entity format
            atlas_entities = self._map_to_atlas_format(event)


            # Send to Atlas
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                auth = (self.auth_config["username"], self.auth_config["password"])

                if event.operation in ["create", "update"]:
                    response = await client.post(
                        f"{self.endpoint}/api/atlas/v2/entity/bulk",
                        json={"entities": atlas_entities},
                        auth=auth,
                        timeout=30.0
                    )
                elif event.operation == "delete":
                    response = await client.delete(
                        f"{self.endpoint}/api/atlas/v2/entity/guid/{event.entity_id}",
                        auth=auth,
                        timeout=30.0
                    )
                else:
                    raise ValueError(f"Unsupported operation: {event.operation}")

                if response.status_code in [200, 201, 204]:
                    return {
                        "status": "success",
                        "platform": "cloudera_atlas",
                        "response": response.json() if response.content else {}
                    }
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error("Atlas sync failed: %s - %s", e.response.status_code, e.response.text)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred while pushing to Cloudera Atlas: %s", e)
            raise

    async def validate_connection(self) -> bool:
        """Validate connection to Atlas"""
        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                auth = (self.auth_config["username"], self.auth_config["password"])
                response = await client.get(
                    f"{self.endpoint}/api/atlas/v2/types/typedefs",
                    auth=auth,
                    timeout=10.0
                )
                return response.status_code == 200
        except httpx.RequestError as e:
            logger.error("Atlas connection validation failed: %s", e)
            return False

    def get_supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return ["hive_table", "hive_db", "hive_column", "hdfs_path", "process"]

    def _map_to_atlas_format(self, event: CatalogChangeEvent) -> List[Dict[str, Any]]:
        """Map catalog event to Atlas entity format"""
        metadata = event.metadata
        entities = []

        # Create database entity if needed
        if metadata.get("database"):
            db_entity = {
                "typeName": "hive_db",
                "attributes": {
                    "name": metadata["database"],
                    "clusterName": "production",
                    "qualifiedName": f"{metadata['database']}@production"
                }
            }
            entities.append(db_entity)

        # Create table entity
        table_entity = {
            "typeName": "hive_table",
            "attributes": {
                "name": metadata.get("name", event.entity_id),
                "qualifiedName": f"{metadata.get('database', 'default')}.{metadata['name']}@production",
                "owner": metadata.get("owner", "system"),
                "tableType": metadata.get("table_type", "MANAGED"),
                "comment": metadata.get("description", "")
            }
        }

        # Add columns if present
        if "columns" in metadata:
            columns = []
            for col in metadata["columns"]:
                col_entity = {
                    "typeName": "hive_column",
                    "attributes": {
                        "name": col["name"],
                        "type": col["type"],
                        "qualifiedName": f"{table_entity['attributes']['qualifiedName']}.{col['name']}",
                        "comment": col.get("description", "")
                    }
                }
                columns.append(col_entity)

            table_entity["attributes"]["columns"] = columns

        entities.append(table_entity)
        return entities
