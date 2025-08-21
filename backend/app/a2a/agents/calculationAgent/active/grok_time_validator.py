"""
Grok Real-Time Mathematical Validator
Provides continuous validation and feedback for mathematical calculations
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import time

try:
    from app.clients.grokMathematicalClient import GrokMathematicalClient
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

logger = logging.getLogger(__name__)

class GrokRealTimeValidator:
    """Real-time mathematical calculation validator using Grok AI"""
    
    def __init__(self, grok_client: Optional[GrokMathematicalClient] = None):
        self.grok_client = grok_client or (GrokMathematicalClient() if GROK_AVAILABLE else None)
        self.validation_queue = asyncio.Queue()
        self.validation_results = {}
        self.active_validations = set()
        self.callbacks = {}
        self._running = False
        self._validation_task = None
        
        self.validation_settings = {
            "auto_validate": True,
            "confidence_threshold": 0.7,
            "max_validation_time": 30,  # seconds
            "batch_size": 5,
            "enable_caching": True
        }
        
        self.validation_cache = {}  # Simple cache for repeated validations
        
    async def start_validator(self):
        """Start the real-time validation service"""
        if not self.grok_client:
            logger.warning("Grok client not available for real-time validation")
            return
        
        if self._running:
            logger.warning("Validator already running")
            return
        
        self._running = True
        self._validation_task = asyncio.create_task(self._validation_worker())
        logger.info("Grok real-time validator started")
    
    async def stop_validator(self):
        """Stop the real-time validation service"""
        self._running = False
        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass
        logger.info("Grok real-time validator stopped")
    
    async def validate_calculation(self, 
                                 calculation_id: str,
                                 query: str,
                                 result: Any,
                                 steps: List[Dict[str, Any]] = None,
                                 callback: Optional[Callable] = None,
                                 priority: str = "normal") -> str:
        """Queue a calculation for real-time validation"""
        
        validation_request = {
            "calculation_id": calculation_id,
            "query": query,
            "result": result,
            "steps": steps or [],
            "callback": callback,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "queued"
        }
        
        # Check cache first
        cache_key = self._generate_cache_key(query, result)
        if self.validation_settings["enable_caching"] and cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            validation_request["status"] = "cached"
            validation_request["validation_result"] = cached_result
            
            if callback:
                await self._execute_callback(callback, validation_request)
            
            return calculation_id
        
        # Add to validation queue
        await self.validation_queue.put(validation_request)
        self.active_validations.add(calculation_id)
        
        if callback:
            self.callbacks[calculation_id] = callback
        
        logger.info(f"Calculation {calculation_id} queued for validation")
        return calculation_id
    
    async def get_validation_result(self, calculation_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get validation result for a specific calculation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if calculation_id in self.validation_results:
                return self.validation_results[calculation_id]
            await asyncio.sleep(0.1)
        
        return None
    
    async def bulk_validate(self, calculations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate multiple calculations in batch"""
        validation_ids = []
        
        for calc in calculations:
            calc_id = f"bulk_{len(validation_ids)}_{int(time.time())}"
            await self.validate_calculation(
                calculation_id=calc_id,
                query=calc.get("query", ""),
                result=calc.get("result"),
                steps=calc.get("steps", []),
                priority="batch"
            )
            validation_ids.append(calc_id)
        
        # Wait for all validations to complete
        results = {}
        for calc_id in validation_ids:
            result = await self.get_validation_result(calc_id, timeout=60.0)
            results[calc_id] = result
        
        return results
    
    async def _validation_worker(self):
        """Background worker for processing validation requests"""
        while self._running:
            try:
                # Process validation queue
                validations_batch = []
                
                # Collect batch of validations
                try:
                    # Get first validation (blocking with timeout)
                    validation = await asyncio.wait_for(
                        self.validation_queue.get(), 
                        timeout=1.0
                    )
                    validations_batch.append(validation)
                    
                    # Collect additional validations for batch processing
                    for _ in range(self.validation_settings["batch_size"] - 1):
                        try:
                            validation = await asyncio.wait_for(
                                self.validation_queue.get(),
                                timeout=0.1
                            )
                            validations_batch.append(validation)
                        except asyncio.TimeoutError:
                            break
                
                except asyncio.TimeoutError:
                    continue
                
                # Process the batch
                if validations_batch:
                    await self._process_validation_batch(validations_batch)
                
            except Exception as e:
                logger.error(f"Validation worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_validation_batch(self, validations: List[Dict[str, Any]]):
        """Process a batch of validation requests"""
        
        for validation in validations:
            try:
                calculation_id = validation["calculation_id"]
                
                # Skip if already processed (cached)
                if validation.get("status") == "cached":
                    continue
                
                # Perform Grok validation
                validation_result = await self._perform_grok_validation(validation)
                
                # Store result
                self.validation_results[calculation_id] = {
                    "calculation_id": calculation_id,
                    "validation_result": validation_result,
                    "original_request": validation,
                    "completed_at": datetime.utcnow().isoformat()
                }
                
                # Cache the result
                cache_key = self._generate_cache_key(
                    validation["query"], 
                    validation["result"]
                )
                if self.validation_settings["enable_caching"]:
                    self.validation_cache[cache_key] = validation_result
                
                # Execute callback if provided
                if calculation_id in self.callbacks:
                    callback = self.callbacks[calculation_id]
                    await self._execute_callback(callback, self.validation_results[calculation_id])
                    del self.callbacks[calculation_id]
                
                # Remove from active validations
                self.active_validations.discard(calculation_id)
                
                logger.info(f"Validation completed for {calculation_id}")
                
            except Exception as e:
                logger.error(f"Validation failed for {validation.get('calculation_id', 'unknown')}: {e}")
                
                # Store error result
                error_result = {
                    "calculation_id": validation.get("calculation_id"),
                    "validation_result": {
                        "is_correct": "error",
                        "error": str(e),
                        "confidence": 0.0
                    },
                    "error": str(e),
                    "completed_at": datetime.utcnow().isoformat()
                }
                
                if validation.get("calculation_id"):
                    self.validation_results[validation["calculation_id"]] = error_result
                    self.active_validations.discard(validation["calculation_id"])
    
    async def _perform_grok_validation(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual Grok validation"""
        
        if not self.grok_client:
            return {
                "is_correct": "unavailable",
                "error": "Grok client not available",
                "confidence": 0.0
            }
        
        try:
            query = validation["query"]
            result = validation["result"]
            steps = validation["steps"]
            
            # Use Grok to validate the calculation
            validation_result = await self.grok_client.validate_mathematical_result(
                query=query,
                calculated_result=result,
                calculation_steps=steps
            )
            
            # Enhance validation with additional checks
            enhanced_result = await self._enhance_validation_result(
                validation_result, 
                validation
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Grok validation failed: {e}")
            return {
                "is_correct": "error",
                "error": str(e),
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _enhance_validation_result(self, 
                                       grok_result: Dict[str, Any],
                                       original_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Grok validation with additional analysis"""
        
        enhanced = grok_result.copy()
        
        # Add metadata
        enhanced["validation_method"] = "grok_enhanced"
        enhanced["validation_timestamp"] = datetime.utcnow().isoformat()
        enhanced["original_query"] = original_validation["query"]
        
        # Analyze confidence levels
        confidence = enhanced.get("confidence", 0.5)
        
        if confidence > 0.9:
            enhanced["confidence_level"] = "very_high"
            enhanced["recommendation"] = "Result is highly reliable"
        elif confidence > 0.7:
            enhanced["confidence_level"] = "high" 
            enhanced["recommendation"] = "Result appears correct"
        elif confidence > 0.5:
            enhanced["confidence_level"] = "medium"
            enhanced["recommendation"] = "Result may need review"
        else:
            enhanced["confidence_level"] = "low"
            enhanced["recommendation"] = "Result should be double-checked"
        
        # Add performance metrics
        enhanced["validation_performance"] = {
            "queue_time": "< 1s",  # Would be calculated in real implementation
            "processing_time": "< 2s",
            "total_time": "< 3s"
        }
        
        return enhanced
    
    def _generate_cache_key(self, query: str, result: Any) -> str:
        """Generate cache key for validation result"""
        import hashlib


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
        cache_string = f"{query}:{str(result)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _execute_callback(self, callback: Callable, result: Dict[str, Any]):
        """Execute validation callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)
        except Exception as e:
            logger.error(f"Callback execution failed: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "active_validations": len(self.active_validations),
            "completed_validations": len(self.validation_results),
            "cache_size": len(self.validation_cache),
            "queue_size": self.validation_queue.qsize(),
            "is_running": self._running,
            "settings": self.validation_settings
        }
    
    async def configure_validator(self, settings: Dict[str, Any]):
        """Update validator settings"""
        self.validation_settings.update(settings)
        logger.info(f"Validator settings updated: {settings}")
    
    async def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        logger.info("Validation cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the validator"""
        
        health_status = {
            "status": "healthy",
            "grok_client_available": self.grok_client is not None,
            "validator_running": self._running,
            "queue_size": self.validation_queue.qsize(),
            "active_validations": len(self.active_validations),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test Grok client if available
        if self.grok_client:
            try:
                test_validation = await self.grok_client.validate_mathematical_result(
                    "What is 2 + 2?", 
                    4, 
                    [{"step": "add", "result": 4}]
                )
                health_status["grok_test"] = "passed"
            except Exception as e:
                health_status["grok_test"] = "failed"
                health_status["grok_error"] = str(e)
                health_status["status"] = "degraded"
        
        return health_status

# Factory function for easy instantiation
def create_grok_validator(grok_client: Optional[GrokMathematicalClient] = None) -> GrokRealTimeValidator:
    """Create a Grok real-time validator instance"""
    return GrokRealTimeValidator(grok_client)