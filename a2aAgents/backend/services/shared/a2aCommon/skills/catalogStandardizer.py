import subprocess
import json
import os
from typing import Dict, List, Any, Optional
import logging
import asyncio
# Safe import for grok client
from ...clients.grokClient import GrokClient, create_grok_client
except ImportError:
    GrokClient = None
    create_grok_client = None

logger = logging.getLogger(__name__)


class CatalogStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/catalog_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None
        
    async def standardize(self, catalog_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize catalog metadata using AI enrichment and validation
        """
        try:
            # Prepare input data
            if isinstance(catalog_data, dict) and "raw_value" in catalog_data:
                # Single catalog string
                input_data = {
                    "catalog_name": catalog_data.get("raw_value", ""),
                    "description": catalog_data.get("description", ""),
                    "owner": catalog_data.get("owner", ""),
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = {
                    "catalog_name": catalog_data.get("catalog_name", ""),
                    "description": catalog_data.get("description", ""),
                    "owner": catalog_data.get("owner", ""),
                    "schema_version": catalog_data.get("schema_version", ""),
                    "created_date": catalog_data.get("created_date", ""),
                    "last_modified": catalog_data.get("last_modified", ""),
                    "_row_number": catalog_data.get("_row_number", 1)
                }
            
            # Initial standardization
            initial_standardized = {
                "name": input_data.get("catalog_name", ""),
                "catalog_type": self._infer_catalog_type(input_data),
                "domain": self._infer_domain(input_data),
                "business_area": "Unknown",
                "data_classification": self._infer_classification(input_data),
                "access_level": "Unknown",
                "retention_policy": "Unknown",
                "quality_level": "Unknown",
                "compliance_framework": "Unknown",
                "governance_tier": "Unknown",
                "schema_format": "Unknown",
                "integration_pattern": "Unknown",
                "update_frequency": "Unknown",
                "description": input_data.get("description", ""),
                "owner": input_data.get("owner", ""),
                "schema_version": input_data.get("schema_version", ""),
                "created_date": input_data.get("created_date", ""),
                "last_modified": input_data.get("last_modified", ""),
                "generated_catalog_id": self._generate_catalog_id(input_data)
            }
            
            # Calculate initial completeness
            completeness = self._calculate_completeness(initial_standardized)
            
            # If completeness is low, try to enrich with Grok
            if completeness < 0.8 and self._should_use_grok():
                enriched = await self._enrich_with_grok(initial_standardized, catalog_data)
                if enriched:
                    # Update with enriched data
                    for key, value in enriched.items():
                        if value and (not initial_standardized.get(key) or initial_standardized.get(key) == "Unknown"):
                            initial_standardized[key] = value
                    completeness = self._calculate_completeness(initial_standardized)
            
            return {
                "original": catalog_data,
                "standardized": initial_standardized,
                "confidence": self._calculate_confidence(initial_standardized),
                "completeness": completeness,
                "metadata": {
                    "is_valid_catalog": initial_standardized.get("catalog_type") != "Unknown",
                    "needs_review": completeness < 0.6,
                    "validation_issues": self._validate_catalog(initial_standardized),
                    "enriched_with_ai": "enriched" in locals()
                }
            }
            
        except Exception as e:
            logger.error(f"Error standardizing catalog: {str(e)}")
            return {
                "original": catalog_data,
                "standardized": None,
                "confidence": 0.0,
                "completeness": 0.0,
                "error": str(e)
            }
    
    def _infer_catalog_type(self, data: Dict) -> str:
        """
        Infer catalog type from name and description
        """
        name = data.get("catalog_name", "").lower()
        desc = data.get("description", "").lower()
        
        # Check for common catalog patterns
        if any(term in name for term in ["master", "reference", "dim", "dimension"]):
            return "Reference Data"
        elif any(term in name for term in ["fact", "transaction", "event", "activity"]):
            return "Transactional Data"
        elif any(term in name for term in ["staging", "raw", "landing", "inbound"]):
            return "Staging Data"
        elif any(term in name for term in ["mart", "warehouse", "dwh", "analytical"]):
            return "Analytical Data"
        elif any(term in name for term in ["metadata", "schema", "catalog"]):
            return "Metadata Catalog"
        elif any(term in desc for term in ["api", "service", "endpoint"]):
            return "API Catalog"
        else:
            return "Unknown"
    
    def _infer_domain(self, data: Dict) -> str:
        """
        Infer business domain from catalog name and description
        """
        name = data.get("catalog_name", "").lower()
        desc = data.get("description", "").lower()
        text = f"{name} {desc}"
        
        domain_keywords = {
            "Financial": ["finance", "accounting", "revenue", "cost", "budget", "gl", "ledger", "ifrs", "gaap"],
            "Customer": ["customer", "client", "crm", "contact", "account", "party"],
            "Product": ["product", "item", "catalog", "inventory", "sku", "material"],
            "Sales": ["sales", "order", "quote", "opportunity", "pipeline"],
            "Operations": ["operations", "process", "workflow", "task", "activity"],
            "HR": ["employee", "hr", "human", "payroll", "personnel", "staff"],
            "Regulatory": ["regulatory", "compliance", "audit", "sox", "basel", "mifid"],
            "Risk": ["risk", "credit", "market", "operational", "var", "stress"],
            "Technology": ["system", "application", "infrastructure", "platform", "service"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return "Unknown"
    
    def _infer_classification(self, data: Dict) -> str:
        """
        Infer data classification level
        """
        name = data.get("catalog_name", "").lower()
        desc = data.get("description", "").lower()
        owner = data.get("owner", "").lower()
        text = f"{name} {desc} {owner}"
        
        if any(term in text for term in ["confidential", "restricted", "sensitive", "private", "pii", "gdpr"]):
            return "Confidential"
        elif any(term in text for term in ["internal", "proprietary", "business"]):
            return "Internal"
        elif any(term in text for term in ["public", "external", "open", "shared"]):
            return "Public"
        else:
            return "Internal"  # Default assumption
    
    def _generate_catalog_id(self, data: Dict) -> str:
        """
        Generate standardized catalog ID
        """
        name = data.get("catalog_name", "").replace(" ", "_").replace("-", "_")
        owner = data.get("owner", "").split("@")[0] if "@" in data.get("owner", "") else data.get("owner", "")
        
        # Create a simplified ID
        catalog_id = f"CAT_{name.upper()[:20]}"
        if owner:
            catalog_id += f"_{owner.upper()[:10]}"
        
        return catalog_id
    
    def _validate_catalog(self, catalog: Dict) -> List[str]:
        """
        Validate catalog data and return list of issues
        """
        issues = []
        
        if not catalog.get("name"):
            issues.append("Missing catalog name")
        
        if catalog.get("catalog_type") == "Unknown":
            issues.append("Could not determine catalog type")
        
        if catalog.get("domain") == "Unknown":
            issues.append("Could not determine business domain")
        
        if not catalog.get("owner"):
            issues.append("Missing data owner")
        
        if not catalog.get("description"):
            issues.append("Missing catalog description")
        
        return issues
    
    def _calculate_confidence(self, standardized_data: Dict) -> float:
        """
        Calculate confidence score for standardized catalog
        """
        score = 0.0
        max_score = 6.0
        
        # Check if valid catalog type was found
        if standardized_data.get("catalog_type") != "Unknown":
            score += 1.0
        
        # Check if domain was identified
        if standardized_data.get("domain") != "Unknown":
            score += 1.0
        
        # Check if classification was identified
        if standardized_data.get("data_classification") != "Unknown":
            score += 1.0
        
        # Check if business area was identified
        if standardized_data.get("business_area") != "Unknown":
            score += 1.0
        
        # Check if governance tier was identified
        if standardized_data.get("governance_tier") != "Unknown":
            score += 1.0
        
        # Check if essential metadata is present
        if standardized_data.get("name") and standardized_data.get("owner"):
            score += 1.0
        
        return min(1.0, score / max_score)
    
    def _calculate_completeness(self, standardized: Dict) -> float:
        """
        Calculate completeness score (0-1) for standardized catalog data
        """
        fields = [
            ("name", 1.0),
            ("catalog_type", 1.0),
            ("domain", 1.0),
            ("business_area", 0.8),
            ("data_classification", 1.0),
            ("access_level", 0.6),
            ("retention_policy", 0.4),
            ("quality_level", 0.6),
            ("compliance_framework", 0.8),
            ("governance_tier", 0.8),
            ("schema_format", 0.4),
            ("integration_pattern", 0.4),
            ("update_frequency", 0.6),
            ("description", 0.8),
            ("owner", 1.0)
        ]
        
        total_weight = sum(weight for _, weight in fields)
        score = 0.0
        
        for field, weight in fields:
            value = standardized.get(field)
            if value and value != "Unknown":
                score += weight
        
        return min(1.0, score / total_weight)
    
    def _should_use_grok(self) -> bool:
        """
        Check if Grok API is available and should be used
        """
        return os.getenv('XAI_API_KEY') is not None
    
    async def _enrich_with_grok(self, standardized: Dict, original: Dict) -> Optional[Dict]:
        """
        Use Grok API to enrich missing catalog data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()
            
            # Build prompt for Grok
            catalog_name = standardized.get('name', '')
            current_type = standardized.get('catalog_type', 'Unknown')
            current_domain = standardized.get('domain', 'Unknown')
            description = standardized.get('description', '')
            owner = standardized.get('owner', '')
            
            prompt = f"""Given this data catalog information:
            
Catalog Name: {catalog_name}
Current Type: {current_type}
Current Domain: {current_domain}
Description: {description}
Owner: {owner}

Please provide the following missing information for this data catalog in JSON format:
1. catalog_type: Type of catalog ("Reference Data", "Transactional Data", "Analytical Data", "Staging Data", "Metadata Catalog", "API Catalog")
2. domain: Business domain ("Financial", "Customer", "Product", "Sales", "Operations", "HR", "Regulatory", "Risk", "Technology")
3. business_area: Specific business area within domain
4. data_classification: Security level ("Public", "Internal", "Confidential", "Restricted")
5. access_level: Who can access ("Public", "Team", "Department", "Organization", "Restricted")
6. retention_policy: How long data is kept ("30 days", "1 year", "7 years", "Indefinite")
7. quality_level: Data quality assessment ("High", "Medium", "Low")
8. compliance_framework: Applicable regulations ("GDPR", "SOX", "Basel III", "IFRS", "Internal", "None")
9. governance_tier: Governance level ("Tier 1", "Tier 2", "Tier 3", "Unclassified")
10. schema_format: Data format ("JSON", "Avro", "Parquet", "CSV", "XML", "Database")
11. integration_pattern: How data is integrated ("Batch", "Stream", "API", "Event-Driven", "Manual")
12. update_frequency: How often updated ("Real-time", "Hourly", "Daily", "Weekly", "Monthly", "On-Demand")

Return ONLY valid JSON without any markdown formatting."""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700
            )
            
            if response.content:
                # Parse JSON response
                import re
                # Remove any markdown code blocks if present
                json_str = re.sub(r'```json\s*|\s*```', '', response.content)
                enriched_data = json.loads(json_str)
                
                # Validate and return only the fields we need
                result = {}
                
                # Map enriched fields
                field_mappings = [
                    ("catalog_type", str),
                    ("domain", str),
                    ("business_area", str),
                    ("data_classification", str),
                    ("access_level", str),
                    ("retention_policy", str),
                    ("quality_level", str),
                    ("compliance_framework", str),
                    ("governance_tier", str),
                    ("schema_format", str),
                    ("integration_pattern", str),
                    ("update_frequency", str)
                ]
                
                for field, field_type in field_mappings:
                    if field in enriched_data and enriched_data[field]:
                        try:
                            result[field] = field_type(enriched_data[field])
                        except (ValueError, TypeError):
                            pass
                    
                return result if result else None
                
        except Exception as e:
            logger.warning(f"Failed to enrich catalog with Grok: {str(e)}")
        
        return None


# For compatibility with async/await pattern in Python < 3.7
if not hasattr(asyncio, 'create_subprocess_exec'):
    import subprocess


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
    
    async def create_subprocess_exec(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(args, **kwargs)
        )
    
    asyncio.create_subprocess_exec = create_subprocess_exec