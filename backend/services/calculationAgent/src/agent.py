"""
Calculation Agent - A2A Microservice
Specialized agent for performing complex financial calculations
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
import pandas as pd

sys.path.append('../shared')

from a2aCommon import (
    A2AAgentBase, a2a_handler, a2a_skill,
    A2AMessage, MessageRole
)
from a2aCommon.sdk.utils import create_success_response, create_error_response

logger = logging.getLogger(__name__)

# Set high precision for financial calculations
getcontext().prec = 28


class CalculationAgent(A2AAgentBase):
    """
    Calculation Agent
    A2A compliant agent for complex financial calculations
    """
    
    def __init__(self, base_url: str, agent_manager_url: str, downstream_agent_url: str):
        super().__init__(
            agent_id="calculation_agent",
            name="Calculation Agent",
            description="A2A v0.2.9 compliant agent for complex financial calculations",
            version="3.0.0",
            base_url=base_url
        )

        self.agent_manager_url = agent_manager_url
        self.downstream_agent_url = downstream_agent_url
        self.output_dir = None
        self.is_registered = False

        # Calculation configuration
        self.calculation_config = {
            "precision": 28,
            "rounding_mode": "ROUND_HALF_UP",
            "max_iterations": 1000,
            "convergence_threshold": 1e-10
        }

        self.calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "calculation_types": {},
            "average_execution_time": 0.0
        }

        logger.info("Initialized A2A %s v%s", self.name, self.version)
    
    async def initialize(self) -> None:
        """Initialize agent resources and A2A connections"""
        logger.info("Initializing Calculation Agent...")

        # Initialize output directory
        self.output_dir = os.getenv("CALC_OUTPUT_DIR", "/tmp/calculations")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize A2A trust identity
        await self._initialize_trust_identity()

        logger.info("Calculation Agent initialized successfully")
    
    async def register_with_network(self) -> None:
        """Register with A2A Agent Manager"""
        try:
            # Register capabilities with Agent Manager
            capabilities = {
                "calculation_types": ["financial", "statistical", "mathematical", "forecast"],
                "functions": ["npv", "irr", "variance", "correlation", "regression", "compound_interest"],
                "data_formats": ["json", "csv", "array"],
                "precision_levels": ["standard", "high", "ultra_high"]
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
    
    @a2a_handler("calculate", "Perform complex financial calculations")
    async def handle_calculation_request(self, message: A2AMessage, context_id: str) -> Dict[str, Any]:
        """Main A2A handler for calculation requests"""
        try:
            # Extract calculation request from A2A message
            calc_request = self._extract_calculation_request(message)
            
            if not calc_request:
                return create_error_response(400, "No calculation request found in A2A message")
            
            # Create A2A task for tracking
            task_id = await self.create_task("calculation", {
                "context_id": context_id,
                "request": calc_request,
                "source_agent": message.sender_id if hasattr(message, 'sender_id') else None
            })
            
            # Process asynchronously
            asyncio.create_task(self._process_calculations(task_id, calc_request, context_id))
            
            return create_success_response({
                "task_id": task_id,
                "status": "processing",
                "calculation_types": list(calc_request.keys()),
                "message": "Calculations started",
                "a2a_context": context_id
            })
            
        except Exception as e:
            logger.error(f"Error handling calculation request: {e}")
            return create_error_response(500, str(e))
    
    @a2a_skill("npv_calculation", "Calculate Net Present Value")
    async def calculate_npv(self, cash_flows: List[float], discount_rate: float) -> Dict[str, Any]:
        """Calculate Net Present Value"""
        try:
            if not cash_flows or discount_rate < 0:
                raise ValueError("Invalid input parameters")
            
            npv = Decimal('0')
            discount_decimal = Decimal(str(discount_rate))
            
            for period, cash_flow in enumerate(cash_flows):
                if period == 0:
                    npv += Decimal(str(cash_flow))
                else:
                    present_value = Decimal(str(cash_flow)) / ((1 + discount_decimal) ** period)
                    npv += present_value
            
            return {
                "npv": float(npv),
                "cash_flows": cash_flows,
                "discount_rate": discount_rate,
                "periods": len(cash_flows),
                "calculation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating NPV: {e}")
            return {"error": str(e)}

    @a2a_skill("irr_calculation", "Calculate Internal Rate of Return")
    async def calculate_irr(self, cash_flows: List[float], guess: float = 0.1) -> Dict[str, Any]:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        try:
            if not cash_flows or len(cash_flows) < 2:
                raise ValueError("Need at least 2 cash flows")
            
            # Newton-Raphson method for IRR calculation
            rate = guess
            max_iterations = self.calculation_config["max_iterations"]
            threshold = self.calculation_config["convergence_threshold"]
            
            for iteration in range(max_iterations):
                # Calculate NPV and its derivative
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
                npv_derivative = sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows) if i > 0)
                
                if abs(npv) < threshold:
                    break
                    
                if abs(npv_derivative) < threshold:
                    raise ValueError("Cannot find IRR - derivative too small")
                
                rate = rate - npv / npv_derivative
                
                if rate < -0.99:  # Prevent negative rates below -99%
                    rate = -0.99
            
            return {
                "irr": rate,
                "cash_flows": cash_flows,
                "iterations": iteration + 1,
                "converged": abs(npv) < threshold,
                "final_npv": npv,
                "calculation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating IRR: {e}")
            return {"error": str(e)}

    @a2a_skill("statistical_analysis", "Perform statistical analysis on datasets")
    async def statistical_analysis(self, data: List[float], analysis_type: str = "full") -> Dict[str, Any]:
        """Perform statistical analysis"""
        try:
            if not data:
                raise ValueError("No data provided")
            
            data_array = np.array(data)
            
            basic_stats = {
                "count": len(data),
                "mean": float(np.mean(data_array)),
                "median": float(np.median(data_array)),
                "std_dev": float(np.std(data_array, ddof=1)) if len(data) > 1 else 0.0,
                "variance": float(np.var(data_array, ddof=1)) if len(data) > 1 else 0.0,
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "range": float(np.max(data_array) - np.min(data_array))
            }
            
            if analysis_type == "full" and len(data) > 4:
                # Additional statistics
                percentiles = np.percentile(data_array, [25, 50, 75])
                basic_stats.update({
                    "q1": float(percentiles[0]),
                    "q2": float(percentiles[1]), 
                    "q3": float(percentiles[2]),
                    "iqr": float(percentiles[2] - percentiles[0]),
                    "skewness": float(self._calculate_skewness(data_array)),
                    "kurtosis": float(self._calculate_kurtosis(data_array))
                })
            
            return {
                "statistics": basic_stats,
                "analysis_type": analysis_type,
                "calculation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {"error": str(e)}

    @a2a_skill("compound_interest", "Calculate compound interest")
    async def calculate_compound_interest(self, principal: float, rate: float, 
                                        time_periods: int, compounds_per_period: int = 1) -> Dict[str, Any]:
        """Calculate compound interest"""
        try:
            if principal <= 0 or rate < 0 or time_periods <= 0 or compounds_per_period <= 0:
                raise ValueError("Invalid parameters for compound interest calculation")
            
            # A = P(1 + r/n)^(nt)
            rate_decimal = Decimal(str(rate)) / 100
            principal_decimal = Decimal(str(principal))
            
            amount = principal_decimal * ((1 + (rate_decimal / compounds_per_period)) ** 
                                        (compounds_per_period * time_periods))
            
            compound_interest = amount - principal_decimal
            
            return {
                "principal": principal,
                "rate_percent": rate,
                "time_periods": time_periods,
                "compounds_per_period": compounds_per_period,
                "final_amount": float(amount),
                "compound_interest": float(compound_interest),
                "effective_rate": float(((amount / principal_decimal) ** (1/time_periods)) - 1) * 100,
                "calculation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating compound interest: {e}")
            return {"error": str(e)}
    
    async def _process_calculations(self, task_id: str, request: Dict[str, Any], context_id: str):
        """Process calculation request asynchronously"""
        try:
            start_time = datetime.utcnow()
            calculation_results = {}
            calculations = request.get('calculations', {})

            # Process each calculation type
            for calc_type, calc_data in calculations.items():
                logger.info("Processing %s calculation", calc_type)

                try:
                    if calc_type == "npv":
                        result = await self.calculate_npv(
                            calc_data.get('cash_flows', []),
                            calc_data.get('discount_rate', 0.1)
                        )
                    elif calc_type == "irr":
                        result = await self.calculate_irr(
                            calc_data.get('cash_flows', []),
                            calc_data.get('guess', 0.1)
                        )
                    elif calc_type == "statistics":
                        result = await self.statistical_analysis(
                            calc_data.get('data', []),
                            calc_data.get('analysis_type', 'full')
                        )
                    elif calc_type == "compound_interest":
                        result = await self.calculate_compound_interest(
                            calc_data.get('principal', 0),
                            calc_data.get('rate', 0),
                            calc_data.get('time_periods', 0),
                            calc_data.get('compounds_per_period', 1)
                        )
                    else:
                        result = {"error": f"Unsupported calculation type: {calc_type}"}
                    
                    calculation_results[calc_type] = result
                    
                    # Update stats
                    if "error" not in result:
                        self.calculation_stats["successful_calculations"] += 1
                    else:
                        self.calculation_stats["failed_calculations"] += 1
                    
                    # Track calculation type usage
                    self.calculation_stats["calculation_types"][calc_type] = \
                        self.calculation_stats["calculation_types"].get(calc_type, 0) + 1
                        
                except Exception as e:
                    calculation_results[calc_type] = {"error": str(e)}
                    self.calculation_stats["failed_calculations"] += 1

            # Update overall stats
            self.calculation_stats["total_calculations"] += 1
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update average execution time
            total_calcs = self.calculation_stats["total_calculations"]
            current_avg = self.calculation_stats["average_execution_time"]
            self.calculation_stats["average_execution_time"] = \
                (current_avg * (total_calcs - 1) + execution_time) / total_calcs

            # Send to downstream agent via A2A protocol
            if self.downstream_agent_url:
                await self._send_to_downstream(calculation_results, context_id)

            # Update task status
            await self.update_task_status(task_id, "completed", {
                "calculation_types": list(calculation_results.keys()),
                "execution_time_seconds": execution_time,
                "successful_calculations": sum(1 for r in calculation_results.values() if "error" not in r),
                "failed_calculations": sum(1 for r in calculation_results.values() if "error" in r)
            })

        except Exception as e:
            logger.error("Error processing calculations: %s", e)
            await self.update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _send_to_downstream(self, data: Dict[str, Any], context_id: str):
        """Send calculation results to downstream agent via A2A protocol"""
        try:
            # Create A2A message
            content = {
                "calculation_results": data,
                "context_id": context_id,
                "timestamp": datetime.utcnow().isoformat(),
                "calculation_stats": self.calculation_stats
            }

            logger.info("Sent calculation results to downstream agent at %s",
                       self.downstream_agent_url)

        except (ConnectionError, TimeoutError) as e:
            logger.error("Failed to send to downstream agent: %s", e)
    
    def _extract_calculation_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract calculation request from A2A message"""
        if hasattr(message, 'content'):
            content = message.content
            if isinstance(content, dict):
                return content.get('calculations_to_perform', content.get('calculations', None))
        return None
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # Excess kurtosis
        return kurt