import subprocess
import json
import os
from typing import Dict, List, Any, Optional
import logging
import asyncio
from ...clients.grokClient import GrokClient, create_grok_client

logger = logging.getLogger(__name__)


class MeasureStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/measure_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None
        
    async def standardize(self, measure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize measure data using the JavaScript standardizer
        """
        try:
            # Prepare input data
            if isinstance(measure_data, dict) and "raw_value" in measure_data:
                # Single measure string
                input_data = {
                    "measureType": measure_data.get("raw_value", ""),
                    "Version": "",
                    "Currency": "",
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = {
                    "measureType": measure_data.get("measureType", ""),
                    "Version": measure_data.get("Version", ""),
                    "Currency": measure_data.get("Currency", ""),
                    "_row_number": measure_data.get("_row_number", 1)
                }
            
            # Call JavaScript standardizer via Node.js
            result = await self._call_js_standardizer([input_data])
            
            if result and len(result) > 0:
                standardized = result[0]
                
                # Extract initial standardized data
                initial_standardized = {
                    "name": standardized.get("clean_measure_type", ""),
                    "measure_type": standardized.get("measure_type", "Unknown"),
                    "category": standardized.get("category", "Unknown"),
                    "subcategory": standardized.get("subcategory", "Unknown"),
                    "version_period": standardized.get("clean_version", ""),
                    "version_period_type": standardized.get("version_type", "Unknown"),
                    "currency_treatment": standardized.get("currency_standardized", "Unknown"),
                    "currency_type": standardized.get("currency_category", "Unknown"),
                    "data_source": standardized.get("data_source", "Unknown"),
                    "timing": standardized.get("timing", "Unknown"),
                    "accuracy": standardized.get("accuracy", "Unknown"),
                    "regulatory_usage": standardized.get("regulatory_usage", "Unknown"),
                    "audit_requirement": standardized.get("audit_requirement", "Unknown"),
                    "generated_measure_code": standardized.get("generated_measure_code")
                }
                
                # Calculate initial completeness
                completeness = self._calculate_completeness(initial_standardized)
                
                # If completeness is low, try to enrich with Grok
                if completeness < 0.8 and self._should_use_grok():
                    enriched = await self._enrich_with_grok(initial_standardized, measure_data)
                    if enriched:
                        # Update with enriched data
                        for key, value in enriched.items():
                            if value and (not initial_standardized.get(key) or initial_standardized.get(key) == "Unknown"):
                                initial_standardized[key] = value
                        completeness = self._calculate_completeness(initial_standardized)
                
                return {
                    "original": measure_data,
                    "standardized": initial_standardized,
                    "confidence": self._calculate_confidence(standardized),
                    "completeness": completeness,
                    "metadata": {
                        "is_valid_measure": standardized.get("measure_type") != "Unknown",
                        "needs_review": completeness < 0.6,
                        "validation_issues": standardized.get("validation_issues"),
                        "enriched_with_ai": "enriched" in locals()
                    }
                }
            else:
                raise ValueError("No standardization result returned")
                
        except Exception as e:
            logger.error(f"Error standardizing measure: {str(e)}")
            return {
                "original": measure_data,
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
        const MeasureStandardizer = require('{absolute_js_path}');
        
        // Override console.log for the standardizer to redirect logs to stderr
        const originalLog = console.log;
        const originalInfo = console.info;
        const originalWarn = console.warn;
        
        // Create standardizer with custom logger that writes to stderr
        const standardizer = new MeasureStandardizer({{
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
                originalLog(JSON.stringify(result));
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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Node.js error: {stderr.decode()}")
            
            # Debug: log the output
            output = stdout.decode()
            stderr_output = stderr.decode()
            
            # Filter out INFO/WARN logs from stderr
            if stderr_output and not output.strip():
                # Check if stderr contains the actual output (sometimes Node.js logs go there)
                lines = stderr_output.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('['):
                        output = line
                        break
            
            if not output.strip():
                logger.error(f"No output from Node.js. Stdout: '{output}', Stderr: '{stderr_output}'")
                raise RuntimeError(f"No output from Node.js")
            
            return json.loads(output)
            
        finally:
            # Clean up temporary file
            os.unlink(wrapper_path)
    
    def _calculate_confidence(self, standardized_data: Dict) -> float:
        """
        Calculate confidence score for standardized measure
        """
        score = 0.0
        max_score = 6.0
        
        # Check if valid measure type was found
        if standardized_data.get("measure_type") != "Unknown":
            score += 1.0
        
        # Check if category was identified
        if standardized_data.get("category") != "Unknown":
            score += 1.0
        
        # Check if version/period was identified
        if standardized_data.get("version_type") != "Unknown":
            score += 1.0
        
        # Check if currency treatment was identified
        if standardized_data.get("currency_type") != "Unknown":
            score += 1.0
        
        # Check if regulatory usage was identified
        if standardized_data.get("regulatory_usage") != "Unknown":
            score += 1.0
        
        # Check standardization quality
        quality = standardized_data.get("standardization_quality", "")
        if quality == "Excellent":
            score += 1.0
        elif quality == "Good":
            score += 0.7
        elif quality == "Fair":
            score += 0.4
        
        return min(1.0, score / max_score)
    
    def _calculate_completeness(self, standardized: Dict) -> float:
        """
        Calculate completeness score (0-1) for standardized measure data
        """
        fields = [
            ("name", 1.0),
            ("measure_type", 1.0),
            ("category", 1.0),
            ("subcategory", 0.5),
            ("version_period", 0.5),
            ("version_period_type", 0.5),
            ("currency_treatment", 1.0),
            ("currency_type", 0.5),
            ("data_source", 0.5),
            ("timing", 0.5),
            ("accuracy", 0.5),
            ("regulatory_usage", 1.0),
            ("audit_requirement", 0.5)
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
        Use Grok API to enrich missing measure data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()
            
            # Build prompt for Grok
            measure_name = standardized.get('name', '')
            current_measure_type = standardized.get('measure_type', 'Unknown')
            current_category = standardized.get('category', 'Unknown')
            version = standardized.get('version_period', '')
            currency = standardized.get('currency_treatment', '')
            
            prompt = f"""Given this financial measurement information:
            
Measure Name: {measure_name}
Current Measure Type: {current_measure_type}
Current Category: {current_category}
Version/Period: {version}
Currency Treatment: {currency}

Please provide the following missing information for this financial measurement configuration in JSON format:
1. measure_type: Type of measurement ("Actual", "Budget", "Forecast", "Plan", "Variance", "Target", "Benchmark")
2. category: High-level category ("Historical", "Planning", "Analysis", "Comparison")
3. subcategory: More specific classification
4. version_period_type: Period classification ("Month-to-Date", "Year-to-Date", "Quarter-to-Date", "Period-to-Date", "Point-in-Time")
5. currency_type: Currency treatment type ("Reporting Currency", "Local Currency", "Constant FX", "Transaction Currency")
6. data_source: Where this data comes from ("General Ledger", "Planning System", "Calculated", "Market Data")
7. timing: When this measure is available ("Pre-Event", "Post-Event", "Real-Time")
8. accuracy: Expected accuracy level ("High", "Medium", "Low")
9. regulatory_usage: How used in regulatory reporting ("Primary", "Supporting", "Internal", "None")
10. audit_requirement: Audit requirements ("Full Audit", "Review", "None")
11. reporting_framework: Applicable framework ("IFRS", "US GAAP", "Basel III", "Internal")
12. calculation_method: How this measure is calculated or sourced

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
                    ("measure_type", str),
                    ("category", str),
                    ("subcategory", str),
                    ("version_period_type", str),
                    ("currency_type", str),
                    ("data_source", str),
                    ("timing", str),
                    ("accuracy", str),
                    ("regulatory_usage", str),
                    ("audit_requirement", str),
                    ("reporting_framework", str),
                    ("calculation_method", str)
                ]
                
                for field, field_type in field_mappings:
                    if field in enriched_data and enriched_data[field]:
                        try:
                            result[field] = field_type(enriched_data[field])
                        except (ValueError, TypeError):
                            pass
                    
                return result if result else None
                
        except Exception as e:
            logger.warning(f"Failed to enrich measure with Grok: {str(e)}")
        
        return None


# For compatibility with async/await pattern in Python < 3.7
if not hasattr(asyncio, 'create_subprocess_exec'):
    import subprocess
    
    async def create_subprocess_exec(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.Popen(args, **kwargs)
        )
    
    asyncio.create_subprocess_exec = create_subprocess_exec