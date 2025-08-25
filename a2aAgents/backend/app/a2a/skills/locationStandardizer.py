import subprocess
import json
import os
from typing import Dict, List, Any, Optional
import logging
import asyncio
from app.clients.grokClient import GrokClient, create_grok_client

logger = logging.getLogger(__name__)


class LocationStandardizer:
    def __init__(self):
        self.version = "1.0.0"
        # Fix the path to scripts directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        self.js_file_path = os.path.join(project_root, "scripts/build/location_standardization.js")
        # Initialize Grok client for enrichment
        self.grok_client = None

    async def standardize(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize location data using the JavaScript standardizer
        """
        try:
            # Prepare input data
            if isinstance(location_data, dict) and "raw_value" in location_data:
                # Single location string
                input_data = {
                    "Location (L0)": location_data.get("raw_value", ""),
                    "Location (L1)": "",
                    "Location (L2)": "",
                    "Location (L3)": "",
                    "Location (L4)": "",
                    "_row_number": 1
                }
            else:
                # Already structured data
                input_data = location_data

            # Call JavaScript standardizer via Node.js
            result = await self._call_js_standardizer([input_data])

            if result and len(result) > 0:
                standardized = result[0]

                # Extract initial standardized data
                initial_standardized = {
                    "name": standardized.get("L4_clean_name") or standardized.get("L3_clean_name") or standardized.get("L2_clean_name"),
                    "country": standardized.get("L1_clean_name") if standardized.get("L1_type") == "country" else None,
                    "iso2": standardized.get("L1_iso2") or standardized.get("L2_iso2"),
                    "iso3": standardized.get("L1_iso3") or standardized.get("L2_iso3"),
                    "region": standardized.get("L0_clean_name") if standardized.get("L0_type") == "region" else None,
                    "subregion": None,  # Not provided by JS standardizer
                    "coordinates": {
                        "latitude": standardized.get("primary_latitude"),
                        "longitude": standardized.get("primary_longitude")
                    },
                    "hierarchy_path": standardized.get("hierarchy_path")
                }

                # Calculate initial completeness
                completeness = self._calculate_completeness(initial_standardized)

                # If completeness is low, try to enrich with Grok (up to 3 passes)
                enrichment_passes = 0
                max_passes = 3

                while completeness < 0.8 and self._should_use_grok() and enrichment_passes < max_passes:
                    enrichment_passes += 1
                    logger.info(f"Location enrichment pass {enrichment_passes} - completeness: {completeness:.2f}")

                    # Get missing fields for targeted enrichment
                    missing_fields = self._get_missing_fields(initial_standardized)

                    if not missing_fields:
                        break

                    enriched = await self._enrich_with_grok(initial_standardized, location_data, missing_fields, enrichment_passes)
                    if enriched:
                        # Update with enriched data
                        fields_updated = 0
                        if enriched.get("iso2") and not initial_standardized["iso2"]:
                            initial_standardized["iso2"] = enriched["iso2"]
                            fields_updated += 1
                        if enriched.get("iso3") and not initial_standardized["iso3"]:
                            initial_standardized["iso3"] = enriched["iso3"]
                            fields_updated += 1
                        if enriched.get("coordinates") and not initial_standardized["coordinates"]["latitude"]:
                            initial_standardized["coordinates"] = enriched["coordinates"]
                            fields_updated += 1
                        if enriched.get("region") and not initial_standardized["region"]:
                            initial_standardized["region"] = enriched["region"]
                            fields_updated += 1
                        if enriched.get("subregion") and not initial_standardized["subregion"]:
                            initial_standardized["subregion"] = enriched["subregion"]
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
                    "original": location_data,
                    "standardized": initial_standardized,
                    "confidence": self._calculate_confidence(standardized),
                    "completeness": completeness,
                    "metadata": {
                        "is_valid_location": standardized.get("L4_type") != "unknown" or standardized.get("L3_type") != "unknown",
                        "needs_review": completeness < 0.6,
                        "validation_issues": standardized.get("validation_issues"),
                        "enriched_with_ai": enrichment_passes > 0,
                        "enrichment_passes": enrichment_passes
                    }
                }
            else:
                raise ValueError("No standardization result returned")

        except Exception as e:
            logger.error(f"Error standardizing location: {str(e)}")
            return {
                "original": location_data,
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
        const LocationStandardizer = require('{absolute_js_path}');

        // Override console.log for the standardizer to redirect logs to stderr
        const originalLog = console.log;
        const originalInfo = console.info;
        const originalWarn = console.warn;

        // Create standardizer with custom logger that writes to stderr
        const standardizer = new LocationStandardizer({{
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
        Calculate confidence score for standardized location
        """
        score = 0.0
        max_score = 5.0

        # Check if valid location was found
        if standardized_data.get("is_valid_location"):
            score += 1.0

        # Check if ISO codes were found
        if standardized_data.get("iso2_code"):
            score += 1.0
        if standardized_data.get("iso3_code"):
            score += 0.5

        # Check if coordinates were found
        if standardized_data.get("latitude") and standardized_data.get("longitude"):
            score += 1.0

        # Check if UN region data was found
        if standardized_data.get("un_region"):
            score += 0.5
        if standardized_data.get("un_subregion"):
            score += 0.5

        # Check standardization quality
        quality = standardized_data.get("standardization_quality", "")
        if quality == "Excellent":
            score += 0.5
        elif quality == "Good":
            score += 0.3

        return min(1.0, score / max_score)

    def _calculate_completeness(self, standardized: Dict) -> float:
        """
        Calculate completeness score (0-1) for standardized location data
        """
        fields = [
            ("name", 1.0),
            ("country", 1.5),
            ("iso2", 1.0),
            ("iso3", 0.5),
            ("region", 0.5),
            ("subregion", 0.5),
            ("coordinates.latitude", 1.0),
            ("coordinates.longitude", 1.0)
        ]

        total_weight = sum(weight for _, weight in fields)
        score = 0.0

        for field, weight in fields:
            if "." in field:
                # Handle nested fields
                parts = field.split(".")
                value = standardized
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else None
                if value:
                    score += weight
            else:
                if standardized.get(field):
                    score += weight

        return min(1.0, score / total_weight)

    def _should_use_grok(self) -> bool:
        """
        Check if Grok API is available and should be used
        """
        return os.getenv('XAI_API_KEY') is not None

    def _get_missing_fields(self, standardized: Dict) -> List[str]:
        """
        Get list of missing fields that need enrichment
        """
        missing = []

        # Check each field
        if not standardized.get("iso2"):
            missing.append("iso2")
        if not standardized.get("iso3"):
            missing.append("iso3")
        if not standardized.get("region"):
            missing.append("region")
        if not standardized.get("subregion"):
            missing.append("subregion")

        # Check coordinates
        coords = standardized.get("coordinates", {})
        if not coords or not coords.get("latitude") or not coords.get("longitude"):
            missing.append("coordinates")

        return missing

    async def _enrich_with_grok(self, standardized: Dict, original: Dict, missing_fields: List[str] = None, pass_number: int = 1) -> Optional[Dict]:
        """
        Use Grok API to enrich missing location data
        """
        try:
            if not self.grok_client:
                self.grok_client = create_grok_client()

            # Build prompt for Grok
            location_name = standardized.get('name', '')
            hierarchy = standardized.get('hierarchy_path', '')
            country = standardized.get('country', 'Unknown')

            # Use missing fields if provided, otherwise check all fields
            if missing_fields is None:
                missing_fields = self._get_missing_fields(standardized)

            # Build targeted prompt based on what's missing
            missing_items = []
            if "iso2" in missing_fields:
                missing_items.append("iso2: The ISO 3166-1 alpha-2 code (2 letters)")
            if "iso3" in missing_fields:
                missing_items.append("iso3: The ISO 3166-1 alpha-3 code (3 letters)")
            if "coordinates" in missing_fields:
                missing_items.append("coordinates: Object with latitude and longitude (decimal degrees)")
            if "region" in missing_fields:
                missing_items.append("region: The UN M49 region name")
            if "subregion" in missing_fields:
                missing_items.append("subregion: The UN M49 subregion name")

            # Add context hints for later passes
            context_hint = ""
            if pass_number > 1:
                context_hint = f"\n\nThis is enrichment pass {pass_number}. Previous attempts may have failed to find some information. Please try alternative sources or make educated estimates based on the location hierarchy."

            prompt = f"""Given this location information:
Location Name: {location_name}
Hierarchy: {hierarchy}
Country: {country}

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
                # Remove any markdown code blocks if present
                json_str = re.sub(r'```json\s*|\s*```', '', response.content)
                enriched_data = json.loads(json_str)

                # Validate and return only the fields we need
                result = {}

                if enriched_data.get("iso2") and len(enriched_data["iso2"]) == 2:
                    result["iso2"] = enriched_data["iso2"].upper()

                if enriched_data.get("iso3") and len(enriched_data["iso3"]) == 3:
                    result["iso3"] = enriched_data["iso3"].upper()

                if enriched_data.get("coordinates"):
                    coords = enriched_data["coordinates"]
                    if isinstance(coords, dict) and "latitude" in coords and "longitude" in coords:
                        try:
                            lat = float(coords["latitude"])
                            lng = float(coords["longitude"])
                            if -90 <= lat <= 90 and -180 <= lng <= 180:
                                result["coordinates"] = {"latitude": lat, "longitude": lng}
                        except (ValueError, TypeError):
                            pass

                if enriched_data.get("region"):
                    result["region"] = enriched_data["region"]

                if enriched_data.get("subregion"):
                    result["subregion"] = enriched_data["subregion"]

                return result if result else None

        except Exception as e:
            logger.warning(f"Failed to enrich location with Grok: {str(e)}")

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
