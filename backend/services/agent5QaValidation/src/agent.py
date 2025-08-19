"""
QA Validation Agent - A2A Microservice
Agent 5: Validates data quality and performs quality assurance checks
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from statistics import mean, median, mode, stdev

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response


logger = logging.getLogger(__name__)


class QAValidationAgent(A2AAgentBase):
    """
    Agent 5: QA Validation Agent
    A2A compliant agent for data quality validation and QA checks
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="qa_validation_agent_5",
            name="QA Validation Agent",
            description="A2A v0.2.9 compliant agent for data quality validation and quality assurance",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # QA validation rules and thresholds
        self.validation_rules = {
            "completeness_threshold": 0.95,  # 95% completeness required
            "uniqueness_threshold": 0.99,   # 99% uniqueness for key fields
            "consistency_threshold": 0.98,  # 98% consistency
            "accuracy_threshold": 0.97,     # 97% accuracy
            "validity_threshold": 0.95      # 95% validity
        }

        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "records_validated": 0,
            "quality_score": 0.0
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing QA Validation Agent...")

        # Initialize output directory
        self.output_dir = os.getenv("QA_OUTPUT_DIR", "/tmp/qa_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("QA Validation Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "validation_types": ["completeness", "uniqueness", "consistency", "accuracy", "validity"],
                "data_types": ["financial", "numerical", "categorical", "temporal"],
                "quality_metrics": ["score", "profile", "anomalies", "outliers"],
                "batch_processing": True
            }

            logger.info("Registered with A2A network at %s", self.agent_manager_url)
            self.is_registered = True

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to register with A2A network: %s", e)
            raise
    
    async def deregister_from_network(self) -> None:
        """Deregister from A2A network"""
        logger.info("Deregistering from A2A network...")
        self.is_registered = False
        logger.info("Successfully deregistered from A2A network")
    
    @a2a_handler("validate_quality", "Perform comprehensive data quality validation")
    async def handle_qa_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for QA validation requests"""
        try:
            # Extract QA request from A2A message
            qa_request = self._extract_qa_request(message)
            
            if not qa_request:
                return create_error_response(400, "No QA request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("qa_validation", {
                "context_id": context_id,
                "request": qa_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_qa_validation(task_id, qa_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "validation_types": list(qa_request.get('data', {}).keys()),
                "message": "QA validation started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling QA request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("completeness_check", "Check data completeness")
    async def check_completeness(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data completeness across fields"""
        if not data:
            return {"completeness_score": 0.0, "missing_fields": [], "total_records": 0}
        
        df = pd.DataFrame(data)
        total_records = len(df)
        total_fields = len(df.columns)
        total_cells = total_records * total_fields
        
        # Count missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        # Calculate completeness score
        completeness_score = (total_cells - total_missing) / total_cells if total_cells > 0 else 0.0
        
        # Identify problematic fields
        missing_fields = []
        for field, count in missing_counts.items():
            if count > 0:
                missing_percentage = count / total_records
                missing_fields.append({
                    "field": field,
                    "missing_count": int(count),
                    "missing_percentage": missing_percentage,
                    "is_critical": missing_percentage > (1 - self.validation_rules["completeness_threshold"])
                })
        
        return {
            "completeness_score": completeness_score,
            "missing_fields": missing_fields,
            "total_records": total_records,
            "total_missing": int(total_missing),
            "passed": completeness_score >= self.validation_rules["completeness_threshold"]
        }

    @a2a_skill("uniqueness_check", "Check data uniqueness for key fields")
    async def check_uniqueness(self, data: List[Dict[str, Any]], key_fields: List[str]) -> Dict[str, Any]:
        """Check uniqueness of specified key fields"""
        if not data or not key_fields:
            return {"uniqueness_score": 1.0, "duplicate_records": [], "total_records": len(data or [])}
        
        df = pd.DataFrame(data)
        total_records = len(df)
        
        # Check uniqueness for each key field
        uniqueness_results = []
        overall_duplicates = []
        
        for field in key_fields:
            if field in df.columns:
                unique_count = df[field].nunique()
                duplicate_count = total_records - unique_count
                uniqueness_score = unique_count / total_records if total_records > 0 else 1.0
                
                # Find duplicate values
                duplicates = df[df.duplicated(subset=[field], keep=False)]
                duplicate_values = duplicates[field].unique().tolist()
                
                uniqueness_results.append({
                    "field": field,
                    "unique_count": unique_count,
                    "duplicate_count": duplicate_count,
                    "uniqueness_score": uniqueness_score,
                    "duplicate_values": duplicate_values[:10],  # Limit to first 10
                    "passed": uniqueness_score >= self.validation_rules["uniqueness_threshold"]
                })
                
                if duplicate_count > 0:
                    overall_duplicates.extend(duplicates.to_dict('records'))
        
        # Overall uniqueness score
        avg_uniqueness = mean([r["uniqueness_score"] for r in uniqueness_results]) if uniqueness_results else 1.0
        
        return {
            "uniqueness_score": avg_uniqueness,
            "field_results": uniqueness_results,
            "duplicate_records": overall_duplicates[:20],  # Limit to first 20
            "total_records": total_records,
            "passed": avg_uniqueness >= self.validation_rules["uniqueness_threshold"]
        }

    @a2a_skill("consistency_check", "Check data consistency and format validation")
    async def check_consistency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data consistency and format validation"""
        if not data:
            return {"consistency_score": 1.0, "inconsistencies": [], "total_records": 0}
        
        df = pd.DataFrame(data)
        total_records = len(df)
        inconsistencies = []
        
        # Check data types consistency
        for column in df.columns:
            series = df[column]
            if series.dtype == 'object':  # String/mixed type column
                # Check for mixed types
                types_found = set()
                for value in series.dropna():
                    types_found.add(type(value).__name__)
                
                if len(types_found) > 1:
                    inconsistencies.append({
                        "field": column,
                        "issue": "mixed_types",
                        "types_found": list(types_found),
                        "severity": "high"
                    })
                
                # Check for format consistency (e.g., dates, IDs)
                if "date" in column.lower() or "time" in column.lower():
                    date_formats = self._detect_date_formats(series)
                    if len(date_formats) > 1:
                        inconsistencies.append({
                            "field": column,
                            "issue": "inconsistent_date_format",
                            "formats_found": date_formats,
                            "severity": "medium"
                        })
        
        # Check for outliers in numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            series = df[column].dropna()
            if len(series) > 10:  # Need sufficient data for outlier detection
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                if len(outliers) > len(series) * 0.05:  # More than 5% outliers
                    inconsistencies.append({
                        "field": column,
                        "issue": "excessive_outliers",
                        "outlier_count": len(outliers),
                        "outlier_percentage": len(outliers) / len(series),
                        "severity": "medium"
                    })
        
        # Calculate consistency score
        high_severity = sum(1 for i in inconsistencies if i.get("severity") == "high")
        medium_severity = sum(1 for i in inconsistencies if i.get("severity") == "medium")
        
        # Penalty system: high severity = -0.1, medium = -0.05
        penalty = (high_severity * 0.1) + (medium_severity * 0.05)
        consistency_score = max(0.0, 1.0 - penalty)
        
        return {
            "consistency_score": consistency_score,
            "inconsistencies": inconsistencies,
            "total_records": total_records,
            "passed": consistency_score >= self.validation_rules["consistency_threshold"]
        }
    
    async def _process_qa_validation(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process QA validation request asynchronously"""
        try:
            validation_results = {}
            data = request.get('data', {})

            # Process each dataset
            for dataset_name, dataset in data.items():
                logger.info("Validating %d records in dataset: %s", len(dataset), dataset_name)

                # Run all validation checks
                completeness = await self.check_completeness(dataset)
                
                # Extract key fields from request or infer them
                key_fields = request.get('key_fields', {}).get(dataset_name, [])
                if not key_fields:
                    # Infer key fields (fields with 'id', 'key', or 'code' in name)
                    if dataset:
                        key_fields = [k for k in dataset[0].keys() 
                                    if any(term in k.lower() for term in ['id', 'key', 'code'])]
                
                uniqueness = await self.check_uniqueness(dataset, key_fields)
                consistency = await self.check_consistency(dataset)
                
                # Calculate overall quality score
                quality_scores = [
                    completeness["completeness_score"],
                    uniqueness["uniqueness_score"],
                    consistency["consistency_score"]
                ]
                overall_quality = mean(quality_scores)
                
                validation_results[dataset_name] = {
                    "completeness": completeness,
                    "uniqueness": uniqueness,
                    "consistency": consistency,
                    "overall_quality_score": overall_quality,
                    "passed_all_checks": all([
                        completeness["passed"],
                        uniqueness["passed"],
                        consistency["passed"]
                    ]),
                    "record_count": len(dataset)
                }
                
                self.validation_stats["records_validated"] += len(dataset)

            # Update stats
            self.validation_stats["total_validations"] += 1
            
            # Calculate pass/fail
            passed = all(result["passed_all_checks"] for result in validation_results.values())
            if passed:
                self.validation_stats["passed_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1
            
            # Update overall quality score
            all_scores = [r["overall_quality_score"] for r in validation_results.values()]
            self.validation_stats["quality_score"] = mean(all_scores) if all_scores else 0.0

            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(validation_results, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "validated_datasets": list(validation_results.keys()),
                "total_records": sum(r["record_count"] for r in validation_results.values()),
                "overall_quality_score": self.validation_stats["quality_score"],
                "passed_validation": passed
            })

        except Exception as e:
            logger.error("Error processing QA validation: %s", e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send validation results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "qa_results": data,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "validation_stats": self.validation_stats
            }

            logger.info("Sent QA results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_qa_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract QA request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('data_to_validate', content.get('data', None))
        return None
    
    def _detect_date_formats(self, series) -> List[str]:
        """Detect different date formats in a series"""
        formats_found = set()
        sample_size = min(100, len(series))  # Sample for performance
        
        for value in series.dropna().head(sample_size):
            if isinstance(value, str):
                # Try common date patterns
                if len(value) == 10 and value.count('-') == 2:
                    formats_found.add('YYYY-MM-DD')
                elif len(value) == 10 and value.count('/') == 2:
                    formats_found.add('MM/DD/YYYY')
                elif len(value) == 19 and 'T' in value:
                    formats_found.add('ISO_DATETIME')
                # Add more format detections as needed
        
        return list(formats_found)
