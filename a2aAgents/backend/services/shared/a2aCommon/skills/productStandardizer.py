import subprocess
import json
import os
from typing import Dict, List, Any, Optional
import logging
import asyncio
# Safe import for grok client
try:
    from ...clients.grokClient import GrokClient, create_grok_client
except ImportError:
    GrokClient = None
    create_grok_client = None

logger = logging.getLogger(__name__)


class ProductStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/product_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None
        
    async def standardize(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize product data using the JavaScript standardizer and enrich with Grok
        """
        try:
            # Prepare input data
            if isinstance(product_data, dict) and "raw_value" in product_data:
                # Single product string
                raw = product_data.get("raw_value", "")
                parts = raw.split(" → ") if " → " in raw else [raw, "", "", ""]
                input_data = {
                    "Product (L0)": parts[0] if len(parts) > 0 else "",
                    "Product (L1)": parts[1] if len(parts) > 1 else "",
                    "Product (L2)": parts[2] if len(parts) > 2 else "",
                    "Product (L3)": parts[3] if len(parts) > 3 else "",
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = product_data
            
            # Call JavaScript standardizer
            result = await self._call_js_standardizer([input_data])
            
            if result and len(result) > 0:
                standardized = result[0]
                
                # Extract key fields from JS standardizer
                initial_standardized = {
                    "hierarchy_path": standardized.get("hierarchy_path"),
                    "product_code": standardized.get("generated_product_code"),
                    "product_category": standardized.get("product_category"),
                    "product_family": standardized.get("product_family"),
                    "basel_category": standardized.get("basel_category"),
                    "regulatory_treatment": standardized.get("regulatory_treatment"),
                    "clean_names": {
                        "L0": standardized.get("L0_clean_name"),
                        "L1": standardized.get("L1_clean_name"),
                        "L2": standardized.get("L2_clean_name"),
                        "L3": standardized.get("L3_clean_name")
                    }
                }
                
                # Calculate initial completeness
                completeness = self._calculate_completeness(initial_standardized)
                
                # If completeness is low, try to enrich with Grok (up to 3 passes)
                enrichment_passes = 0
                max_passes = 3
                
                while completeness < 0.8 and self._should_use_grok() and enrichment_passes < max_passes:
                    enrichment_passes += 1
                    logger.info(f"Product enrichment pass {enrichment_passes} - completeness: {completeness:.2f}")
                    
                    # Get missing fields for targeted enrichment
                    missing_fields = self._get_missing_fields(initial_standardized)
                    
                    if not missing_fields:
                        break
                    
                    enriched = await self._enrich_with_grok(initial_standardized, product_data, missing_fields, enrichment_passes)
                    if enriched:
                        # Update only missing fields
                        fields_updated = 0
                        for key, value in enriched.items():
                            if value and (not initial_standardized.get(key) or initial_standardized.get(key) == "Unknown"):
                                initial_standardized[key] = value
                                fields_updated += 1
                        
                        # Recalculate completeness
                        new_completeness = self._calculate_completeness(initial_standardized)
                        
                        # If no improvement, stop trying
                        if new_completeness <= completeness or fields_updated == 0:
                            logger.info(f"No improvement in pass {enrichment_passes}, stopping enrichment")
                            break
                            
                        completeness = new_completeness
                    else:
                        # If enrichment failed, don't try again
                        break
                
                return {
                    "original": product_data,
                    "standardized": initial_standardized,
                    "confidence": self._calculate_confidence(standardized),
                    "completeness": completeness,
                    "metadata": {
                        "is_complete": completeness >= 0.8,
                        "validation_issues": standardized.get("validation_issues"),
                        "standardization_quality": standardized.get("standardization_quality"),
                        "enriched_with_ai": enrichment_passes > 0,
                        "enrichment_passes": enrichment_passes
                    }
                }
            else:
                raise ValueError("No standardization result returned")
                
        except Exception as e:
            logger.error(f"Error standardizing product: {str(e)}")
            return {
                "original": product_data,
                "standardized": None,
                "confidence": 0.0,
                "completeness": 0.0,
                "error": str(e)
            }
    
    async def _call_js_standardizer(self, data: List[Dict]) -> List[Dict]:
        """
        Call the JavaScript standardizer via Node.js subprocess
        """
        # Use absolute path to ensure it's found
        absolute_js_path = os.path.abspath(self.js_file_path)
        
        # Create a temporary Node.js wrapper script
        wrapper_script = f"""
        const ProductStandardizer = require('{absolute_js_path}');
        
        // Create standardizer with custom logger that writes to stderr
        const standardizer = new ProductStandardizer({{
            logger: {{
                info: (msg) => console.error(`[INFO] ${{msg}}`),
                warn: (msg) => console.error(`[WARN] ${{msg}}`),
                error: (msg) => console.error(`[ERROR] ${{msg}}`),
                debug: (msg) => console.error(`[DEBUG] ${{msg}}`)
            }}
        }});
        
        async function run() {{
            try {{
                const input = JSON.parse(process.argv[2]);
                const result = await standardizer.standardizeDataset(input);
                // Only the actual result goes to stdout
                console.log(JSON.stringify(result));
            }} catch (error) {{
                console.error('Error:', error);
                process.exit(1);
            }}
        }}
        
        run();
        """
        
        # Write wrapper to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(wrapper_script)
            wrapper_path = f.name
        
        try:
            # Run Node.js subprocess
            process = await asyncio.create_subprocess_exec(
                'node', wrapper_path, json.dumps(data),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Node.js error: {stderr.decode()}")
            
            # Debug: log the output
            output = stdout.decode()
            if not output.strip():
                raise RuntimeError(f"No output from Node.js")
            
            return json.loads(output)
            
        finally:
            # Clean up temporary file
            os.unlink(wrapper_path)
    
    def _calculate_confidence(self, standardized_data: Dict) -> float:
        """
        Calculate confidence score for standardized product
        """
        score = 0.0
        max_score = 5.0
        
        # Check product classification
        if standardized_data.get("product_category") and standardized_data.get("product_category") != "Unknown":
            score += 1.0
        
        # Check Basel classification
        if standardized_data.get("basel_category") and standardized_data.get("basel_category") != "Unknown":
            score += 1.0
        
        # Check regulatory treatment
        if standardized_data.get("regulatory_treatment") and standardized_data.get("regulatory_treatment") != "Unknown":
            score += 1.0
        
        # Check if hierarchy is valid
        if not standardized_data.get("validation_issues"):
            score += 1.0
        
        # Check standardization quality
        quality = standardized_data.get("standardization_quality", "")
        if quality == "Excellent":
            score += 1.0
        elif quality == "Good":
            score += 0.6
        elif quality == "Fair":
            score += 0.3
        
        return min(1.0, score / max_score)
    
    def _calculate_completeness(self, standardized: Dict) -> float:
        """
        Calculate completeness score (0-1) for standardized data
        """
        fields = [
            "hierarchy_path",
            "product_code", 
            "product_category",
            "product_family",
            "basel_category",
            "regulatory_treatment"
        ]
        
        filled = sum(1 for field in fields if standardized.get(field) and standardized[field] != "Unknown")
        
        # Check clean names
        if standardized.get("clean_names"):
            names_filled = sum(1 for level in ["L0", "L1", "L2", "L3"] 
                             if standardized["clean_names"].get(level))
            filled += names_filled * 0.25  # Each name counts as 0.25
        
        return min(1.0, filled / (len(fields) + 1))  # +1 for clean names
    
    def _should_use_grok(self) -> bool:
        """
        Check if Grok API is available and should be used
        """
        return os.getenv('XAI_API_KEY') is not None
    
    def _get_missing_fields(self, standardized: Dict) -> List[str]:
        """
        Get list of missing or unknown fields that need enrichment
        """
        missing = []
        
        # Check each field
        fields_to_check = [
            "product_code",
            "product_category", 
            "product_family",
            "basel_category",
            "regulatory_treatment",
            "risk_classification",
            "accounting_treatment",
            "trading_book_eligibility"
        ]
        
        for field in fields_to_check:
            value = standardized.get(field)
            if not value or value == "Unknown":
                missing.append(field)
        
        return missing
    
    async def _enrich_with_grok(self, standardized: Dict, original: Dict, missing_fields: List[str] = None, pass_number: int = 1) -> Optional[Dict]:
        """
        Use Grok API to enrich missing product data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()
            
            # Use missing fields if provided, otherwise check all fields
            if missing_fields is None:
                missing_fields = self._get_missing_fields(standardized)
            
            # Build field descriptions
            field_descriptions = {
                "product_code": "A standardized product code",
                "product_category": 'The specific financial product category (e.g., "Derivatives", "Loans", "Deposits")',
                "product_family": 'The product family (e.g., "Interest Rate Derivatives", "Corporate Lending")',
                "basel_category": "The Basel regulatory category",
                "regulatory_treatment": 'The regulatory treatment (e.g., "Trading Book", "Banking Book")',
                "risk_classification": "The risk classification",
                "accounting_treatment": "IFRS/GAAP accounting classification", 
                "trading_book_eligibility": 'Whether eligible for trading book ("Yes", "No", "Conditional")',
                "typical_maturity": "Typical maturity profile",
                "target_market": "Target market segment"
            }
            
            # Build targeted prompt
            missing_items = []
            for field in missing_fields:
                if field in field_descriptions:
                    missing_items.append(f"{field}: {field_descriptions[field]}")
            
            # Add context hints for later passes
            context_hint = ""
            if pass_number > 1:
                context_hint = f"\n\nThis is enrichment pass {pass_number}. Focus on providing the most critical missing fields based on standard banking product classifications."
            
            # Build prompt for Grok
            prompt = f"""Given this financial product information:
            
Product Hierarchy: {standardized.get('hierarchy_path', 'Unknown')}
Current Category: {standardized.get('product_category', 'Unknown')}
Current Family: {standardized.get('product_family', 'Unknown')}

I specifically need the following missing information in JSON format:
{chr(10).join(f'{i+1}. {item}' for i, item in enumerate(missing_items))}
{context_hint}

Important: Only include the fields I've asked for above. Return ONLY valid JSON without any markdown formatting."""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            if response.content:
                # Parse JSON response
                import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                # Remove any markdown code blocks if present
                json_str = re.sub(r'```json\s*|\s*```', '', response.content)
                enriched_data = json.loads(json_str)
                
                return {
                    k: v for k, v in enriched_data.items() 
                    if v and v != "Unknown" and k in [
                        "product_category", "product_family", "basel_category",
                        "regulatory_treatment", "risk_classification", 
                        "typical_maturity", "target_market"
                    ]
                }
                
        except Exception as e:
            logger.warning(f"Failed to enrich with Grok: {str(e)}")
        
        return None