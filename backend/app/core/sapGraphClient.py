"""
import time
SAP Graph API Client for unified data access across SAP systems
Provides centralized access to business data from multiple SAP sources
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import httpx
from dataclasses import dataclass, asdict
from enum import Enum
from app.core.loggingConfig import get_logger, LogCategory

from app.core.config import get_settings

settings = get_settings()
logger = get_logger(__name__, LogCategory.AGENT)


class GraphEntityType(Enum):
    """SAP Graph entity types for unified data access"""
    BUSINESS_PARTNER = "BusinessPartner"
    CUSTOMER = "Customer"
    SUPPLIER = "Supplier"
    COST_CENTER = "CostCenter"
    GENERAL_LEDGER_ACCOUNT = "GeneralLedgerAccount"
    FINANCIAL_STATEMENT = "FinancialStatement"
    SALES_ORDER = "SalesOrder"
    PURCHASE_ORDER = "PurchaseOrder"
    MATERIAL = "Material"
    EMPLOYEE = "Employee"
    PROFIT_CENTER = "ProfitCenter"
    COMPANY_CODE = "CompanyCode"


@dataclass
class GraphQuery:
    """Graph query configuration"""
    entity_type: str
    select_fields: List[str] = None
    filter_conditions: List[str] = None
    expand_relations: List[str] = None
    top: int = None
    skip: int = None
    order_by: List[str] = None
    search: str = None


@dataclass
class GraphBatch:
    """Batch request for multiple Graph queries"""
    requests: List[Dict[str, Any]]
    atomic: bool = False


class SAPGraphClient:
    """
    SAP Graph API Client for unified data access
    Integrates with multiple SAP systems through Graph API
    """
    
    def __init__(self):
        self.base_url = settings.SAP_GRAPH_URL
        self.client_id = settings.SAP_GRAPH_CLIENT_ID
        self.client_secret = settings.SAP_GRAPH_CLIENT_SECRET
        self.tenant_id = settings.SAP_GRAPH_TENANT_ID
        self.token_url = f"{settings.SAP_GRAPH_AUTH_URL}/oauth/token"
        
        self._access_token = None
        self._token_expires_at = None
        self._http_client = None
        
        # Entity configurations
        self.entity_configs = {
            GraphEntityType.BUSINESS_PARTNER: {
                "endpoint": "BusinessPartner",
                "key_field": "BusinessPartner",
                "searchable_fields": ["BusinessPartnerName", "SearchTerm1"]
            },
            GraphEntityType.CUSTOMER: {
                "endpoint": "Customer",
                "key_field": "Customer",
                "searchable_fields": ["CustomerName"]
            },
            GraphEntityType.COST_CENTER: {
                "endpoint": "CostCenter",
                "key_field": "CostCenter",
                "searchable_fields": ["CostCenterName"]
            },
            GraphEntityType.FINANCIAL_STATEMENT: {
                "endpoint": "FinancialStatement",
                "key_field": "CompanyCode",
                "searchable_fields": ["CompanyCodeName"]
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        await self._ensure_token()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._http_client:
            await self._http_client.aclose()
    
    async def _ensure_token(self):
        """Ensure valid access token"""
        if (not self._access_token or 
            not self._token_expires_at or 
            datetime.utcnow() >= self._token_expires_at):
            await self._refresh_token()
    
    async def _refresh_token(self):
        """Refresh OAuth2 access token"""
        try:
            response = await self._http_client.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": "graph:read graph:write"
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in - 60)
            
            logger.info("SAP Graph access token refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh SAP Graph token: {e}")
            raise
    
    def _build_query_url(self, query: GraphQuery) -> str:
        """Build OData query URL"""
        config = self.entity_configs.get(
            GraphEntityType(query.entity_type),
            {"endpoint": query.entity_type}
        )
        
        url = f"{self.base_url}/v1/{config['endpoint']}"
        params = []
        
        # $select
        if query.select_fields:
            params.append(f"$select={','.join(query.select_fields)}")
        
        # $filter
        if query.filter_conditions:
            filter_expr = " and ".join(query.filter_conditions)
            params.append(f"$filter={filter_expr}")
        
        # $expand
        if query.expand_relations:
            params.append(f"$expand={','.join(query.expand_relations)}")
        
        # $top
        if query.top:
            params.append(f"$top={query.top}")
        
        # $skip
        if query.skip:
            params.append(f"$skip={query.skip}")
        
        # $orderby
        if query.order_by:
            params.append(f"$orderby={','.join(query.order_by)}")
        
        # $search
        if query.search:
            params.append(f"$search={query.search}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    async def query_entities(self, query: GraphQuery) -> Dict[str, Any]:
        """Query entities using SAP Graph API"""
        await self._ensure_token()
        
        url = self._build_query_url(query)
        
        try:
            response = await self._http_client.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"Successfully queried {query.entity_type}: {len(result.get('value', []))} records")
            
            return {
                "entity_type": query.entity_type,
                "data": result.get("value", []),
                "total_count": result.get("@odata.count"),
                "next_link": result.get("@odata.nextLink"),
                "query_time": datetime.utcnow().isoformat()
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"SAP Graph API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error querying SAP Graph: {e}")
            raise
    
    async def get_entity_by_id(self, entity_type: str, entity_id: str, 
                              expand_relations: List[str] = None) -> Dict[str, Any]:
        """Get single entity by ID"""
        await self._ensure_token()
        
        config = self.entity_configs.get(
            GraphEntityType(entity_type),
            {"endpoint": entity_type}
        )
        
        url = f"{self.base_url}/v1/{config['endpoint']}('{entity_id}')"
        
        if expand_relations:
            url += f"?$expand={','.join(expand_relations)}"
        
        try:
            response = await self._http_client.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"SAP Graph API error: {e.response.status_code} - {e.response.text}")
            raise
    
    async def batch_query(self, batch: GraphBatch) -> List[Dict[str, Any]]:
        """Execute batch requests"""
        await self._ensure_token()
        
        batch_payload = {
            "requests": batch.requests
        }
        
        if batch.atomic:
            batch_payload["atomicityGroup"] = "group1"
        
        try:
            response = await self._http_client.post(
                f"{self.base_url}/v1/$batch",
                json=batch_payload,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("responses", [])
            
        except Exception as e:
            logger.error(f"Error executing batch query: {e}")
            raise
    
    async def search_business_partners(self, search_term: str, 
                                     top: int = 20) -> List[Dict[str, Any]]:
        """Search business partners with unified approach"""
        query = GraphQuery(
            entity_type=GraphEntityType.BUSINESS_PARTNER.value,
            search=f'"{search_term}"',
            select_fields=[
                "BusinessPartner",
                "BusinessPartnerName", 
                "BusinessPartnerCategory",
                "BusinessPartnerType",
                "SearchTerm1",
                "TelephoneNumber1",
                "EmailAddress"
            ],
            top=top
        )
        
        result = await self.query_entities(query)
        return result.get("data", [])
    
    async def get_financial_data_unified(self, company_code: str, 
                                       fiscal_year: str) -> Dict[str, Any]:
        """Get unified financial data from multiple sources"""
        
        # Create batch request for multiple financial entities
        batch_requests = [
            {
                "id": "financial_statements",
                "method": "GET",
                "url": f"/FinancialStatement?$filter=CompanyCode eq '{company_code}' and FiscalYear eq '{fiscal_year}'"
            },
            {
                "id": "cost_centers",
                "method": "GET", 
                "url": f"/CostCenter?$filter=CompanyCode eq '{company_code}'"
            },
            {
                "id": "profit_centers",
                "method": "GET",
                "url": f"/ProfitCenter?$filter=CompanyCode eq '{company_code}'"
            },
            {
                "id": "gl_accounts",
                "method": "GET",
                "url": f"/GeneralLedgerAccount?$filter=CompanyCode eq '{company_code}'"
            }
        ]
        
        batch = GraphBatch(requests=batch_requests)
        responses = await self.batch_query(batch)
        
        # Combine responses
        unified_data = {
            "company_code": company_code,
            "fiscal_year": fiscal_year,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for response in responses:
            if response.get("status") == 200:
                response_id = response.get("id")
                body = response.get("body", {})
                unified_data[response_id] = body.get("value", [])
        
        return unified_data
    
    async def get_master_data_references(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get reference master data for data standardization"""
        
        reference_queries = [
            GraphQuery(
                entity_type=GraphEntityType.COMPANY_CODE.value,
                select_fields=["CompanyCode", "CompanyCodeName", "Country", "Currency"]
            ),
            GraphQuery(
                entity_type=GraphEntityType.COST_CENTER.value,
                select_fields=["CostCenter", "CostCenterName", "CompanyCode", "CostCenterCategory"],
                top=1000
            ),
            GraphQuery(
                entity_type=GraphEntityType.MATERIAL.value,
                select_fields=["Material", "MaterialName", "MaterialType", "BaseUnitOfMeasure"],
                top=1000
            )
        ]
        
        master_data = {}
        
        for query in reference_queries:
            try:
                result = await self.query_entities(query)
                master_data[query.entity_type.lower()] = result.get("data", [])
            except Exception as e:
                logger.warning(f"Failed to fetch {query.entity_type}: {e}")
                master_data[query.entity_type.lower()] = []
        
        return master_data


class GraphDataIntegration:
    """Integration layer for A2A platform with SAP Graph"""
    
    def __init__(self):
        self.graph_client = SAPGraphClient()
        self.cache_ttl = 3600  # 1 hour
    
    async def enrich_agent0_data(self, data_product: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich Agent 0 data registration with SAP Graph master data"""
        
        enriched_data = data_product.copy()
        
        async with self.graph_client as client:
            try:
                # Extract potential business partner references
                if "customer_id" in data_product:
                    customer = await client.get_entity_by_id(
                        GraphEntityType.CUSTOMER.value,
                        data_product["customer_id"]
                    )
                    if customer:
                        enriched_data["customer_details"] = customer
                
                # Extract company code information
                if "company_code" in data_product:
                    company = await client.get_entity_by_id(
                        GraphEntityType.COMPANY_CODE.value,
                        data_product["company_code"]
                    )
                    if company:
                        enriched_data["company_details"] = company
                
                # Add master data context
                master_data = await client.get_master_data_references()
                enriched_data["master_data_context"] = master_data
                
            except Exception as e:
                logger.warning(f"Failed to enrich data with SAP Graph: {e}")
        
        return enriched_data
    
    async def validate_agent1_references(self, standardized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Agent 1 standardized data against SAP master data"""
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "enrichments": {}
        }
        
        async with self.graph_client as client:
            try:
                # Validate company codes
                if "company_codes" in standardized_data:
                    for company_code in standardized_data["company_codes"]:
                        company = await client.get_entity_by_id(
                            GraphEntityType.COMPANY_CODE.value,
                            company_code
                        )
                        if not company:
                            validation_result["warnings"].append(
                                f"Invalid company code: {company_code}"
                            )
                        else:
                            validation_result["enrichments"][f"company_{company_code}"] = company
                
                # Validate business partners
                if "business_partners" in standardized_data:
                    for bp_id in standardized_data["business_partners"]:
                        bp = await client.get_entity_by_id(
                            GraphEntityType.BUSINESS_PARTNER.value,
                            bp_id
                        )
                        if not bp:
                            validation_result["warnings"].append(
                                f"Invalid business partner: {bp_id}"
                            )
                            validation_result["valid"] = False
                        else:
                            validation_result["enrichments"][f"bp_{bp_id}"] = bp
                            
            except Exception as e:
                logger.error(f"Error validating with SAP Graph: {e}")
                validation_result["valid"] = False
                validation_result["warnings"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def get_contextual_data_for_ai(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual business data for Agent 2 AI preparation"""
        
        contextual_data = {
            "business_context": {},
            "master_data_context": {},
            "relationship_context": {}
        }
        
        async with self.graph_client as client:
            try:
                # Get business context if company code is available
                if "company_code" in data_context:
                    financial_data = await client.get_financial_data_unified(
                        data_context["company_code"],
                        data_context.get("fiscal_year", str(datetime.now().year))
                    )
                    contextual_data["business_context"] = financial_data
                
                # Get master data context
                master_data = await client.get_master_data_references()
                contextual_data["master_data_context"] = master_data
                
                # Search for related business partners if criteria provided
                if "search_term" in data_context:
                    related_partners = await client.search_business_partners(
                        data_context["search_term"]
                    )
                    contextual_data["relationship_context"]["related_partners"] = related_partners
                
            except Exception as e:
                logger.warning(f"Failed to get contextual data: {e}")
        
        return contextual_data