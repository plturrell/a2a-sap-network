"""
Calculation Validation Agent - A2A Microservice
Agent 4: Validates financial calculations and computational accuracy
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import math
import numpy as np
from decimal import Decimal, getcontext

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response


logger = logging.getLogger(__name__)

# Set decimal precision for financial calculations
getcontext().prec = 28


class CalculationValidationAgent(A2AAgentBase):
    """
    Agent 4: Calculation Validation Agent
    A2A compliant agent for validating financial calculations
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="calculation_validation_agent_4",
            name="Calculation Validation Agent",
            description="A2A v0.2.9 compliant agent for validating financial calculations and computational accuracy",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # Validation configuration
        self.tolerance_config = {
            "decimal_precision": 10,
            "percentage_tolerance": 0.001,  # 0.1%
            "absolute_tolerance": 0.01
        }

        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "calculations_validated": 0,
            "accuracy_rate": 0.0
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing Calculation Validation Agent...")

        # Initialize output directory
        self.output_dir = os.getenv("VALIDATION_OUTPUT_DIR", "/tmp/validation_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("Calculation Validation Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "validation_types": ["arithmetic", "formula", "balance", "variance", "statistical"],
                "precision_levels": ["high", "medium", "standard"],
                "calculation_formats": ["decimal", "float", "percentage"],
                "batch_processing": True
            }

            # Send registration to Agent Manager
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
    
    @a2a_handler("validate_calculations", "Validate financial calculations for accuracy")
    async def handle_validation_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for calculation validation requests"""
        try:
            # Extract validation request from A2A message
            validation_request = self._extract_validation_request(message)
            
            if not validation_request:
                return create_error_response(400, "No validation request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("calculation_validation", {
                "context_id": context_id,
                "request": validation_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_validation(task_id, validation_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "validation_types": list(validation_request.get('calculations', {}).keys()),
                "message": "Calculation validation started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling validation request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("arithmetic_validation", "Validate basic arithmetic calculations")
    async def validate_arithmetic(self, calculations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate basic arithmetic operations"""
        results = []
        
        for calc in calculations:
            try:
                operation = calc.get('operation')
                operands = calc.get('operands', [])
                expected_result = calc.get('expected_result')
                
                # Perform calculation
                if operation == 'add':
                    actual_result = sum(Decimal(str(op)) for op in operands)
                elif operation == 'subtract':
                    actual_result = Decimal(str(operands[0]))
                    for op in operands[1:]:
                        actual_result -= Decimal(str(op))
                elif operation == 'multiply':
                    actual_result = Decimal(str(operands[0]))
                    for op in operands[1:]:
                        actual_result *= Decimal(str(op))
                elif operation == 'divide':
                    actual_result = Decimal(str(operands[0]))
                    for op in operands[1:]:
                        if Decimal(str(op)) == 0:
                            raise ValueError("Division by zero")
                        actual_result /= Decimal(str(op))
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                # Validate result
                expected_decimal = Decimal(str(expected_result))
                is_valid = abs(actual_result - expected_decimal) <= Decimal(str(self.tolerance_config["absolute_tolerance"]))
                
                results.append({
                    "calculation_id": calc.get('id'),
                    "operation": operation,
                    "expected_result": str(expected_result),
                    "actual_result": str(actual_result),
                    "is_valid": is_valid,
                    "difference": str(abs(actual_result - expected_decimal)),
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                results.append({
                    "calculation_id": calc.get('id'),
                    "operation": calc.get('operation'),
                    "is_valid": False,
                    "error": str(e),
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
        
        return results

    @a2a_skill("balance_validation", "Validate financial balance equations")
    async def validate_balances(self, balance_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate balance equations (Assets = Liabilities + Equity)"""
        results = []
        
        for balance in balance_checks:
            try:
                assets = Decimal(str(balance.get('assets', 0)))
                liabilities = Decimal(str(balance.get('liabilities', 0)))
                equity = Decimal(str(balance.get('equity', 0)))
                
                # Calculate balance
                left_side = assets
                right_side = liabilities + equity
                difference = abs(left_side - right_side)
                
                is_balanced = difference <= Decimal(str(self.tolerance_config["absolute_tolerance"]))
                
                results.append({
                    "balance_id": balance.get('id'),
                    "assets": str(assets),
                    "liabilities": str(liabilities),
                    "equity": str(equity),
                    "difference": str(difference),
                    "is_balanced": is_balanced,
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                results.append({
                    "balance_id": balance.get('id'),
                    "is_balanced": False,
                    "error": str(e),
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
        
        return results

    @a2a_skill("variance_validation", "Validate variance calculations")
    async def validate_variance(self, variance_checks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate variance calculations"""
        results = []
        
        for variance in variance_checks:
            try:
                actual_values = [float(v) for v in variance.get('actual', [])]
                expected_values = [float(v) for v in variance.get('expected', [])]
                expected_variance = variance.get('expected_variance')
                
                if len(actual_values) != len(expected_values):
                    raise ValueError("Actual and expected arrays must have same length")
                
                # Calculate variance
                differences = [a - e for a, e in zip(actual_values, expected_values)]
                calculated_variance = np.var(differences, ddof=1) if len(differences) > 1 else 0
                
                # Validate
                if expected_variance is not None:
                    tolerance = abs(float(expected_variance)) * self.tolerance_config["percentage_tolerance"]
                    is_valid = abs(calculated_variance - float(expected_variance)) <= tolerance
                else:
                    is_valid = True
                
                results.append({
                    "variance_id": variance.get('id'),
                    "calculated_variance": calculated_variance,
                    "expected_variance": expected_variance,
                    "is_valid": is_valid,
                    "sample_size": len(actual_values),
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                results.append({
                    "variance_id": variance.get('id'),
                    "is_valid": False,
                    "error": str(e),
                    "validation_timestamp": datetime.utcnow().isoformat()
                })
        
        return results
    
    async def _process_validation(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process validation request asynchronously"""
        try:
            validation_results = {}
            calculations = request.get('calculations', {})

            # Process each validation type
            for validation_type, data in calculations.items():
                logger.info("Validating %d %s calculations", len(data), validation_type)

                # Use appropriate skill
                if validation_type == "arithmetic":
                    result = await self.validate_arithmetic(data)
                elif validation_type == "balance":
                    result = await self.validate_balances(data)
                elif validation_type == "variance":
                    result = await self.validate_variance(data)
                else:
                    logger.warning("Unknown validation type: %s", validation_type)
                    continue

                validation_results[validation_type] = result
                self.validation_stats["calculations_validated"] += len(result)

            # Update stats
            self.validation_stats["total_validations"] += 1
            
            # Calculate success rate
            all_validations = []
            for results in validation_results.values():
                all_validations.extend([r.get('is_valid', False) for r in results])
            
            if all_validations:
                success_count = sum(all_validations)
                self.validation_stats["successful_validations"] += success_count
                self.validation_stats["failed_validations"] += len(all_validations) - success_count
                self.validation_stats["accuracy_rate"] = success_count / len(all_validations)

            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(validation_results, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "validated_types": list(validation_results.keys()),
                "total_calculations": sum(len(data) for data in validation_results.values()),
                "accuracy_rate": self.validation_stats["accuracy_rate"]
            })

        except Exception as e:
            logger.error("Error processing validation: %s", e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send validation results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "validation_results": data,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "validation_stats": self.validation_stats
            }

            logger.info("Sent validation results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_validation_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract validation request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('calculations_to_validate', content.get('calculations', None))
        return None
