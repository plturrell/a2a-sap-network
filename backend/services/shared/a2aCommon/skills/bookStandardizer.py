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


class BookStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/book_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None
        
    async def standardize(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize book data using the JavaScript standardizer
        """
        try:
            # Prepare input data
            if isinstance(book_data, dict) and "raw_value" in book_data:
                # Single book string
                input_data = {
                    "Books": book_data.get("raw_value", ""),
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = {
                    "Books": book_data.get("Book (L3)", "") or book_data.get("Book (L2)", "") or book_data.get("Books", ""),
                    "_row_number": book_data.get("_row_number", 1)
                }
            
            # Call JavaScript standardizer via Node.js
            result = await self._call_js_standardizer([input_data])
            
            if result and len(result) > 0:
                standardized = result[0]
                
                # Extract initial standardized data
                initial_standardized = {
                    "name": standardized.get("clean_book_name", ""),
                    "base_entity": standardized.get("base_entity"),
                    "entity_type": standardized.get("entity_type", "Unknown"),
                    "book_type": standardized.get("book_type", "Unknown"),
                    "book_subtype": standardized.get("book_subtype", "Unknown"),
                    "book_purpose": standardized.get("book_purpose", "Unknown"),
                    "regulatory_scope": standardized.get("regulatory_scope", "Unknown"),
                    "consolidation_method": standardized.get("consolidation_method", "Unknown"),
                    "is_adjustment_book": standardized.get("is_adjustment_book", False),
                    "is_main_entity": standardized.get("is_main_entity", False),
                    "generated_book_code": standardized.get("generated_book_code")
                }
                
                # Calculate initial completeness
                completeness = self._calculate_completeness(initial_standardized)
                
                # If completeness is low, try to enrich with Grok
                if completeness < 0.8 and self._should_use_grok():
                    enriched = await self._enrich_with_grok(initial_standardized, book_data)
                    if enriched:
                        # Update with enriched data
                        for key, value in enriched.items():
                            if value and (not initial_standardized.get(key) or initial_standardized.get(key) == "Unknown"):
                                initial_standardized[key] = value
                        completeness = self._calculate_completeness(initial_standardized)
                
                return {
                    "original": book_data,
                    "standardized": initial_standardized,
                    "confidence": self._calculate_confidence(standardized),
                    "completeness": completeness,
                    "metadata": {
                        "is_valid_book": standardized.get("book_type") != "Unknown",
                        "needs_review": completeness < 0.6,
                        "validation_issues": standardized.get("validation_issues"),
                        "enriched_with_ai": "enriched" in locals()
                    }
                }
            else:
                raise ValueError("No standardization result returned")
                
        except Exception as e:
            logger.error(f"Error standardizing book: {str(e)}")
            return {
                "original": book_data,
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
        const BookStandardizer = require('{absolute_js_path}');
        
        // Override console.log for the standardizer to redirect logs to stderr
        const originalLog = console.log;
        const originalInfo = console.info;
        const originalWarn = console.warn;
        
        // Create standardizer with custom logger that writes to stderr
        const standardizer = new BookStandardizer({{
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
        Calculate confidence score for standardized book
        """
        score = 0.0
        max_score = 5.0
        
        # Check if valid book was found
        if standardized_data.get("book_type") != "Unknown":
            score += 1.0
        
        # Check if entity was identified
        if standardized_data.get("base_entity"):
            score += 1.0
        
        # Check if entity type was classified
        if standardized_data.get("entity_type") != "Unknown":
            score += 1.0
        
        # Check if consolidation method was identified
        if standardized_data.get("consolidation_method") != "Unknown":
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
        Calculate completeness score (0-1) for standardized book data
        """
        fields = [
            ("name", 1.0),
            ("base_entity", 1.0),
            ("entity_type", 1.0),
            ("book_type", 1.0),
            ("book_subtype", 0.5),
            ("book_purpose", 0.5),
            ("regulatory_scope", 1.0),
            ("consolidation_method", 1.0),
            ("generated_book_code", 0.5)
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
        Use Grok API to enrich missing book data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()
            
            # Build prompt for Grok
            book_name = standardized.get('name', '')
            current_entity_type = standardized.get('entity_type', 'Unknown')
            current_book_type = standardized.get('book_type', 'Unknown')
            
            prompt = f"""Given this financial book information:
            
Book Name: {book_name}
Current Entity Type: {current_entity_type}
Current Book Type: {current_book_type}

Please provide the following missing information for this financial reporting book in JSON format:
1. base_entity: The base legal entity name (e.g., "New Financial", "Investment Bank LLC")
2. entity_type: The type of entity ("Legal Entity", "Subsidiary", "Branch", "SPV")
3. book_type: The book classification ("Legal Entity", "Consolidation Adjustment", "Operational Adjustment", "Presentation Adjustment")
4. book_subtype: More specific classification
5. book_purpose: The purpose of this book
6. regulatory_scope: Regulatory reporting scope ("Full Reporting", "Local Reporting", "Group Reporting", "Internal Control")
7. consolidation_method: How this book is consolidated ("Full Consolidation", "Equity Method", "Adjustment Entry", "Elimination")
8. legal_entity_identifier: LEI code if applicable
9. jurisdiction: Primary jurisdiction
10. reporting_currency: Primary reporting currency (3-letter ISO code)

Return ONLY valid JSON without any markdown formatting."""

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600
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
                    ("base_entity", str),
                    ("entity_type", str),
                    ("book_type", str),
                    ("book_subtype", str),
                    ("book_purpose", str),
                    ("regulatory_scope", str),
                    ("consolidation_method", str),
                    ("legal_entity_identifier", str),
                    ("jurisdiction", str),
                    ("reporting_currency", str)
                ]
                
                for field, field_type in field_mappings:
                    if field in enriched_data and enriched_data[field]:
                        try:
                            result[field] = field_type(enriched_data[field])
                        except (ValueError, TypeError):
                            pass
                    
                return result if result else None
                
        except Exception as e:
            logger.warning(f"Failed to enrich book with Grok: {str(e)}")
        
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