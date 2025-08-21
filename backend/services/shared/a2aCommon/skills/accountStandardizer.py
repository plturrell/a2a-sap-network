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


class AccountStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/account_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None
        
    async def standardize(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize account data using the JavaScript standardizer
        """
        try:
            # Prepare input data
            if isinstance(account_data, dict) and "raw_value" in account_data:
                # Single account string - parse into hierarchy
                raw = account_data.get("raw_value", "")
                parts = raw.split(" → ") if " → " in raw else [raw, "", "", ""]
                input_data = {
                    "Account (L0)": parts[0] if len(parts) > 0 else "",
                    "Account (L1)": parts[1] if len(parts) > 1 else "",
                    "Account (L2)": parts[2] if len(parts) > 2 else "",
                    "Account (L3)": parts[3] if len(parts) > 3 else "",
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = account_data
            
            # Call JavaScript standardizer
            result = await self._call_js_standardizer([input_data])
            
            if result and len(result) > 0:
                standardized = result[0]
                
                # Extract initial standardized data
                initial_standardized = {
                    "hierarchy_path": standardized.get("hierarchy_path"),
                    "gl_account_code": standardized.get("generated_account_code"),
                    "account_type": standardized.get("account_type", "Unknown"),
                    "account_subtype": standardized.get("account_subtype", "Unknown"),
                    "account_category": standardized.get("account_category", "Unknown"),
                    "ifrs9_classification": standardized.get("ifrs9_classification", "Unknown"),
                    "basel_classification": standardized.get("basel_classification", "Unknown"),
                    "regulatory_treatment": standardized.get("regulatory_treatment", "Unknown"),
                    "is_balance_sheet": standardized.get("is_balance_sheet"),
                    "is_income_statement": standardized.get("is_income_statement"),
                    "account_number": account_data.get("accountNumber", ""),
                    "account_description": account_data.get("accountDescription", ""),
                    "cost_center": account_data.get("costCenter", "")
                }
                
                # Calculate initial completeness
                completeness = self._calculate_completeness(initial_standardized)
                
                # If completeness is low, try to enrich with Grok (up to 3 passes)
                enrichment_passes = 0
                max_passes = 3
                
                while completeness < 0.8 and self._should_use_grok() and enrichment_passes < max_passes:
                    enrichment_passes += 1
                    logger.info(f"Account enrichment pass {enrichment_passes} - completeness: {completeness:.2f}")
                    
                    # Get missing fields for targeted enrichment
                    missing_fields = self._get_missing_fields(initial_standardized)
                    
                    if not missing_fields:
                        break
                    
                    enriched = await self._enrich_with_grok(initial_standardized, account_data, missing_fields, enrichment_passes)
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
                    "original": account_data,
                    "standardized": initial_standardized,
                    "confidence": self._calculate_confidence(standardized),
                    "completeness": completeness,
                    "metadata": {
                        "has_currency_code": standardized.get("has_currency_code", False),
                        "has_regional_code": standardized.get("has_regional_code", False),
                        "hierarchy_validation_issues": standardized.get("hierarchy_validation_issues"),
                        "standardization_quality": standardized.get("standardization_quality"),
                        "enriched_with_ai": enrichment_passes > 0,
                        "enrichment_passes": enrichment_passes
                    }
                }
            else:
                raise ValueError("No standardization result returned")
                
        except Exception as e:
            logger.error(f"Error standardizing account: {str(e)}")
            return {
                "original": account_data,
                "standardized": None,
                "confidence": 0.0,
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
        const AccountStandardizer = require('{absolute_js_path}');
        
        // Create standardizer with custom logger that writes to stderr
        const standardizer = new AccountStandardizer({{
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
                raise RuntimeError(f"No output from Node.js. Stderr: {stderr.decode()}")
            
            return json.loads(output)
            
        finally:
            # Clean up temporary file
            os.unlink(wrapper_path)
    
    def _calculate_confidence(self, standardized_data: Dict) -> float:
        """
        Calculate confidence score for standardized account
        """
        score = 0.0
        max_score = 5.0
        
        # Check account classification
        if standardized_data.get("account_type") and standardized_data.get("account_type") != "Unknown":
            score += 1.0
        
        # Check IFRS9 classification
        if standardized_data.get("ifrs9_classification") and standardized_data.get("ifrs9_classification") != "Unknown":
            score += 1.0
        
        # Check Basel classification  
        if standardized_data.get("basel_classification") and standardized_data.get("basel_classification") != "Unknown":
            score += 1.0
        
        # Check if hierarchy is valid
        if not standardized_data.get("hierarchy_validation_issues"):
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
        Calculate completeness score (0-1) for standardized account data
        """
        fields = [
            ("gl_account_code", 1.0),
            ("account_type", 1.0),
            ("account_subtype", 0.5),
            ("account_category", 1.0),
            ("ifrs9_classification", 1.0),
            ("basel_classification", 1.0),
            ("regulatory_treatment", 0.5),
            ("hierarchy_path", 0.5)
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
    
    def _get_missing_fields(self, standardized: Dict) -> List[str]:
        """
        Get list of missing or unknown fields that need enrichment
        """
        missing = []
        
        # Check each field
        fields_to_check = [
            "gl_account_code",
            "account_type",
            "account_subtype",
            "account_category",
            "ifrs9_classification",
            "basel_classification",
            "regulatory_treatment"
        ]
        
        for field in fields_to_check:
            value = standardized.get(field)
            if not value or value == "Unknown":
                missing.append(field)
        
        return missing
    
    async def _enrich_with_grok(self, standardized: Dict, original: Dict, missing_fields: List[str] = None, pass_number: int = 1) -> Optional[Dict]:
        """
        Use Grok API to enrich missing account data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()
            
            # Use missing fields if provided, otherwise check all fields
            if missing_fields is None:
                missing_fields = self._get_missing_fields(standardized)
            
            # Build field descriptions
            field_descriptions = {
                "gl_account_code": 'A standardized GL account code (e.g., "1001", "4001")',
                "account_type": 'Type of account ("Asset", "Liability", "Equity", "Revenue", "Expense")',
                "account_subtype": 'More specific classification (e.g., "Current Asset", "Fixed Asset", "Operating Revenue")',
                "account_category": 'Business category (e.g., "Cash and Cash Equivalents", "Trade Receivables", "Trading Revenue")',
                "ifrs9_classification": 'IFRS 9 classification if applicable ("Amortized Cost", "FVTPL", "FVOCI", "N/A")',
                "basel_classification": 'Basel regulatory classification ("Banking Book", "Trading Book", "N/A")',
                "regulatory_treatment": "How this account is treated for regulatory reporting",
                "is_balance_sheet": "Boolean - true if this is a balance sheet account",
                "is_income_statement": "Boolean - true if this is an income statement account",
                "typical_balance": 'Normal balance side ("Debit" or "Credit")'
            }
            
            # Build targeted prompt
            missing_items = []
            for field in missing_fields:
                if field in field_descriptions:
                    missing_items.append(f"{field}: {field_descriptions[field]}")
            
            # Add context hints for later passes
            context_hint = ""
            if pass_number > 1:
                context_hint = f"\n\nThis is enrichment pass {pass_number}. Use standard accounting principles and typical GL account mappings to determine the missing fields."
            
            # Build prompt for Grok
            account_num = standardized.get('account_number', '')
            account_desc = standardized.get('account_description', '')
            cost_center = standardized.get('cost_center', '')
            current_type = standardized.get('account_type', 'Unknown')
            
            prompt = f"""Given this financial account information:
            
Account Number: {account_num}
Account Description: {account_desc}
Cost Center: {cost_center}
Current Type: {current_type}

I specifically need the following missing information in JSON format:
{chr(10).join(f'{i+1}. {item}' for i, item in enumerate(missing_items))}
{context_hint}

Important: Only include the fields I've asked for above. Return ONLY valid JSON without any markdown formatting."""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600
            )
            
            if response.content:
                # Parse JSON response
                import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                # Remove any markdown code blocks if present
                json_str = re.sub(r'```json\s*|\s*```', '', response.content)
                enriched_data = json.loads(json_str)
                
                # Validate and return only the fields we need
                result = {}
                
                # Map enriched fields
                field_mappings = [
                    ("gl_account_code", str),
                    ("account_type", str),
                    ("account_subtype", str), 
                    ("account_category", str),
                    ("ifrs9_classification", str),
                    ("basel_classification", str),
                    ("regulatory_treatment", str),
                    ("is_balance_sheet", bool),
                    ("is_income_statement", bool),
                    ("typical_balance", str)
                ]
                
                for field, field_type in field_mappings:
                    if field in enriched_data and enriched_data[field] is not None:
                        try:
                            result[field] = field_type(enriched_data[field])
                        except (ValueError, TypeError):
                            pass
                    
                return result if result else None
                
        except Exception as e:
            logger.warning(f"Failed to enrich account with Grok: {str(e)}")
        
        return None