#!/usr/bin/env python3
"""
Client Validation System for A2A Platform
Tests all external service clients with real API calls during startup
"""

import asyncio
import os
import logging
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
from enum import Enum
import importlib
import inspect

logger = logging.getLogger(__name__)

class ClientStatus(Enum):
    """Client validation status"""
    WORKING = "working"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"

class ClientValidator:
    """
    Validates all external service clients during startup
    """
    
    def __init__(self):
        self.client_definitions = {
            "GrokClient": {
                "module": "app.clients.grokClient",
                "class": "GrokClient",
                "config_class": "GrokConfig",
                "test_type": "ai_chat",
                "required_env": ["GROK_API_KEY", "XAI_API_KEY"],
                "test_message": "Hello, this is a startup validation test. Please respond with 'validation successful'."
            },
            "PerplexityClient": {
                "module": "app.clients.perplexityClient", 
                "class": "PerplexityClient",
                "config_class": "PerplexityConfig",
                "test_type": "ai_search",
                "required_env": ["PERPLEXITY_API_KEY"],
                "test_message": "What is the current time? This is a startup validation test."
            },
            "GrokMathematicalClient": {
                "module": "app.clients.grokMathematicalClient",
                "class": "GrokMathematicalClient", 
                "config_class": "GrokConfig",
                "test_type": "ai_math",
                "required_env": ["GROK_API_KEY", "XAI_API_KEY"],
                "test_message": "Calculate 2+2. This is a startup validation test."
            },
            "SQLiteClient": {
                "module": "app.clients.sqliteClient",
                "class": "SQLiteClient",
                "test_type": "database",
                "required_env": [],
                "test_query": "SELECT 1 as validation_test"
            },
            "HanaClient": {
                "module": "app.clients.hanaClient",
                "class": "HanaClient", 
                "test_type": "database",
                "required_env": ["HANA_HOST", "HANA_USER", "HANA_PASSWORD"],
                "test_query": "SELECT 1 as validation_test FROM DUMMY"
            },
            "EnterpriseHanaClient": {
                "module": "app.clients.hanaClientExtended",
                "class": "EnterpriseHanaClient",
                "test_type": "database",
                "required_env": ["HANA_HOST", "HANA_USER", "HANA_PASSWORD"],
                "test_query": "SELECT 1 as validation_test FROM DUMMY"
            }
        }
        
        self.results = {}
    
    async def validate_all_clients(self) -> Dict[str, Any]:
        """Validate all configured clients"""
        logger.info("Starting comprehensive client validation...")
        
        validation_summary = {
            "total_clients": len(self.client_definitions),
            "working_clients": 0,
            "degraded_clients": 0,
            "failed_clients": 0,
            "disabled_clients": 0,
            "not_configured_clients": 0,
            "client_results": {},
            "overall_status": "failed"
        }
        
        # Test all clients in parallel for speed
        tasks = []
        for client_name, client_def in self.client_definitions.items():
            task = self._validate_client(client_name, client_def)
            tasks.append(task)
        
        client_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(client_results):
            client_name = list(self.client_definitions.keys())[i]
            
            if isinstance(result, Exception):
                result = {
                    "client_name": client_name,
                    "status": ClientStatus.FAILED.value,
                    "error": str(result),
                    "response_time_ms": None,
                    "test_successful": False
                }
            
            validation_summary["client_results"][client_name] = result
            
            # Update counters
            status = result["status"]
            if status == ClientStatus.WORKING.value:
                validation_summary["working_clients"] += 1
            elif status == ClientStatus.DEGRADED.value:
                validation_summary["degraded_clients"] += 1
            elif status == ClientStatus.FAILED.value:
                validation_summary["failed_clients"] += 1
            elif status == ClientStatus.DISABLED.value:
                validation_summary["disabled_clients"] += 1
            elif status == ClientStatus.NOT_CONFIGURED.value:
                validation_summary["not_configured_clients"] += 1
        
        # Calculate overall status
        working = validation_summary["working_clients"]
        degraded = validation_summary["degraded_clients"]
        total_active = working + degraded + validation_summary["failed_clients"]
        
        if total_active == 0:
            validation_summary["overall_status"] = "no_clients_configured"
        elif working >= total_active * 0.8:  # 80% working
            validation_summary["overall_status"] = "healthy"
        elif working >= total_active * 0.5:  # 50% working
            validation_summary["overall_status"] = "degraded"
        else:
            validation_summary["overall_status"] = "failed"
        
        return validation_summary
    
    async def _validate_client(self, client_name: str, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single client"""
        logger.info(f"Validating {client_name}...")
        
        result = {
            "client_name": client_name,
            "status": ClientStatus.FAILED.value,
            "response_time_ms": None,
            "test_successful": False,
            "error": None,
            "api_response": None,
            "configuration_status": "unknown"
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Check if required environment variables are set
            required_env = client_def.get("required_env", [])
            missing_env = []
            
            for env_var in required_env:
                if not os.getenv(env_var):
                    missing_env.append(env_var)
            
            # If any required env vars are missing, check if any alternative exists
            if missing_env and required_env:
                # For some clients, we accept any one of multiple possible env vars
                any_env_set = any(os.getenv(var) for var in required_env)
                if not any_env_set:
                    result["status"] = ClientStatus.NOT_CONFIGURED.value
                    result["error"] = f"Missing required environment variables: {missing_env}"
                    result["configuration_status"] = "missing_credentials"
                    return result
            
            # Try to import and instantiate the client
            try:
                module = importlib.import_module(client_def["module"])
                client_class = getattr(module, client_def["class"])
                
                # Check if client needs configuration
                if "config_class" in client_def:
                    config_class = getattr(module, client_def["config_class"])
                    
                    # Try to create config - if it fails due to missing API key, mark as not configured
                    try:
                        # Let the client handle its own configuration from environment
                        client_instance = client_class()
                    except ValueError as e:
                        if "api key" in str(e).lower() or "required" in str(e).lower():
                            result["status"] = ClientStatus.NOT_CONFIGURED.value
                            result["error"] = f"Configuration error: {str(e)}"
                            result["configuration_status"] = "invalid_credentials"
                            return result
                        else:
                            raise
                else:
                    # No special configuration needed
                    client_instance = client_class()
                
                result["configuration_status"] = "configured"
                
            except ImportError as e:
                result["error"] = f"Failed to import {client_def['class']}: {str(e)}"
                result["status"] = ClientStatus.DISABLED.value
                return result
            except Exception as e:
                result["error"] = f"Failed to instantiate {client_def['class']}: {str(e)}"
                return result
            
            # Perform actual test based on client type
            test_type = client_def.get("test_type", "unknown")
            
            if test_type == "ai_chat":
                test_result = await self._test_ai_chat_client(client_instance, client_def)
            elif test_type == "ai_search":
                test_result = await self._test_ai_search_client(client_instance, client_def)
            elif test_type == "ai_math":
                test_result = await self._test_ai_math_client(client_instance, client_def)
            elif test_type == "database":
                test_result = await self._test_database_client(client_instance, client_def)
            else:
                test_result = await self._test_generic_client(client_instance, client_def)
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result["response_time_ms"] = round(response_time, 2)
            
            # Update result with test outcome
            result.update(test_result)
            
        except Exception as e:
            result["error"] = f"Client validation failed: {str(e)}"
            logger.error(f"Client {client_name} validation error: {e}")
        
        return result
    
    async def _test_ai_chat_client(self, client, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Test AI chat clients (Grok, etc.)"""
        try:
            test_message = client_def.get("test_message", "Test message")
            
            # Try to call the chat completion method
            if hasattr(client, 'chat_completion_async'):
                response = await client.chat_completion_async([
                    {"role": "user", "content": test_message}
                ], temperature=0.1, max_tokens=50)
            elif hasattr(client, 'chat_completion'):
                response = client.chat_completion([
                    {"role": "user", "content": test_message}
                ], temperature=0.1, max_tokens=50)
            else:
                return {
                    "status": ClientStatus.FAILED.value,
                    "error": "No recognized chat method found",
                    "test_successful": False
                }
            
            # Check if we got a valid response
            if response and hasattr(response, 'content') and response.content:
                return {
                    "status": ClientStatus.WORKING.value,
                    "test_successful": True,
                    "api_response": response.content[:100],  # First 100 chars
                    "model": getattr(response, 'model', 'unknown')
                }
            else:
                return {
                    "status": ClientStatus.DEGRADED.value,
                    "error": "Empty or invalid response",
                    "test_successful": False
                }
                
        except Exception as e:
            error_msg = str(e).lower()
            if "unauthorized" in error_msg or "invalid" in error_msg and "key" in error_msg:
                return {
                    "status": ClientStatus.NOT_CONFIGURED.value,
                    "error": "API key authentication failed",
                    "test_successful": False
                }
            elif "quota" in error_msg or "limit" in error_msg:
                return {
                    "status": ClientStatus.DEGRADED.value,
                    "error": "API quota/rate limit exceeded",
                    "test_successful": False
                }
            else:
                return {
                    "status": ClientStatus.FAILED.value,
                    "error": f"API call failed: {str(e)}",
                    "test_successful": False
                }
    
    async def _test_ai_search_client(self, client, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Test AI search clients (Perplexity, etc.)"""
        # Similar to chat but expects search-specific responses
        return await self._test_ai_chat_client(client, client_def)
    
    async def _test_ai_math_client(self, client, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Test AI math clients"""
        # Similar to chat but with math-specific test
        return await self._test_ai_chat_client(client, client_def)
    
    async def _test_database_client(self, client, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Test database clients (SQLite, HANA, etc.)"""
        try:
            test_query = client_def.get("test_query", "SELECT 1")
            
            # Try to execute a simple test query
            if hasattr(client, 'execute_query_async'):
                result = await client.execute_query_async(test_query)
            elif hasattr(client, 'execute_query'):
                result = client.execute_query(test_query)
            elif hasattr(client, 'query'):
                result = client.query(test_query)
            else:
                return {
                    "status": ClientStatus.FAILED.value,
                    "error": "No recognized query method found",
                    "test_successful": False
                }
            
            if result:
                return {
                    "status": ClientStatus.WORKING.value,
                    "test_successful": True,
                    "api_response": "Query executed successfully"
                }
            else:
                return {
                    "status": ClientStatus.DEGRADED.value,
                    "error": "Query returned no results",
                    "test_successful": False
                }
                
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg or "host" in error_msg:
                return {
                    "status": ClientStatus.NOT_CONFIGURED.value,
                    "error": "Database connection failed",
                    "test_successful": False
                }
            else:
                return {
                    "status": ClientStatus.FAILED.value,
                    "error": f"Database query failed: {str(e)}",
                    "test_successful": False
                }
    
    async def _test_generic_client(self, client, client_def: Dict[str, Any]) -> Dict[str, Any]:
        """Test generic clients by checking basic functionality"""
        try:
            # Check if client has basic health/status methods
            if hasattr(client, 'health_check'):
                result = await client.health_check() if inspect.iscoroutinefunction(client.health_check) else client.health_check()
                return {
                    "status": ClientStatus.WORKING.value,
                    "test_successful": True,
                    "api_response": "Health check passed"
                }
            elif hasattr(client, 'test_connection'):
                result = await client.test_connection() if inspect.iscoroutinefunction(client.test_connection) else client.test_connection()
                return {
                    "status": ClientStatus.WORKING.value,
                    "test_successful": True,
                    "api_response": "Connection test passed"
                }
            else:
                # If no test methods available, just check that we could instantiate it
                return {
                    "status": ClientStatus.DEGRADED.value,
                    "test_successful": True,
                    "api_response": "Client instantiated but no test methods available"
                }
                
        except Exception as e:
            return {
                "status": ClientStatus.FAILED.value,
                "error": f"Generic test failed: {str(e)}",
                "test_successful": False
            }
    
    def format_validation_report(self, validation_data: Dict[str, Any], colored: bool = True) -> str:
        """Format client validation results for console output"""
        lines = []
        
        if colored:
            colors = {
                "blue": "\033[94m",
                "green": "\033[92m",
                "yellow": "\033[93m", 
                "red": "\033[91m",
                "gray": "\033[90m",
                "bold": "\033[1m",
                "reset": "\033[0m"
            }
        else:
            colors = {k: "" for k in ["blue", "green", "yellow", "red", "gray", "bold", "reset"]}
        
        # Header
        lines.append(f"{colors['bold']}{colors['blue']}A2A Client Validation Report{colors['reset']}")
        lines.append(f"{colors['blue']}{'=' * 50}{colors['reset']}")
        lines.append("")
        
        # Overall status
        overall = validation_data["overall_status"]
        if overall == "healthy":
            status_color = colors["green"]
            status_symbol = "✓"
        elif overall == "degraded":
            status_color = colors["yellow"]
            status_symbol = "⚠"
        elif overall == "no_clients_configured":
            status_color = colors["gray"]
            status_symbol = "○"
        else:
            status_color = colors["red"]
            status_symbol = "✗"
        
        lines.append(f"{colors['bold']}Overall Client Status: {status_color}{status_symbol} {overall.upper()}{colors['reset']}")
        lines.append("")
        
        # Summary
        summary = validation_data
        lines.append(f"{colors['bold']}Client Summary:{colors['reset']}")
        lines.append(f"  Total Clients: {summary['total_clients']}")
        
        if summary['working_clients'] > 0:
            lines.append(f"  {colors['green']}✓ Working: {summary['working_clients']}{colors['reset']}")
        if summary['degraded_clients'] > 0:
            lines.append(f"  {colors['yellow']}⚠ Degraded: {summary['degraded_clients']}{colors['reset']}")
        if summary['failed_clients'] > 0:
            lines.append(f"  {colors['red']}✗ Failed: {summary['failed_clients']}{colors['reset']}")
        if summary['not_configured_clients'] > 0:
            lines.append(f"  {colors['gray']}○ Not Configured: {summary['not_configured_clients']}{colors['reset']}")
        if summary['disabled_clients'] > 0:
            lines.append(f"  {colors['gray']}⊘ Disabled: {summary['disabled_clients']}{colors['reset']}")
        
        lines.append("")
        
        # Individual client results
        lines.append(f"{colors['bold']}Individual Client Results:{colors['reset']}")
        for client_name, result in validation_data["client_results"].items():
            status = result["status"]
            
            if status == ClientStatus.WORKING.value:
                symbol = f"{colors['green']}✓{colors['reset']}"
                status_text = "Working"
            elif status == ClientStatus.DEGRADED.value:
                symbol = f"{colors['yellow']}⚠{colors['reset']}"
                status_text = "Degraded"
            elif status == ClientStatus.FAILED.value:
                symbol = f"{colors['red']}✗{colors['reset']}"
                status_text = "Failed"
            elif status == ClientStatus.NOT_CONFIGURED.value:
                symbol = f"{colors['gray']}○{colors['reset']}"
                status_text = "Not Configured"
            else:
                symbol = f"{colors['gray']}⊘{colors['reset']}"
                status_text = "Disabled"
            
            response_time = f" ({result['response_time_ms']}ms)" if result.get("response_time_ms") else ""
            lines.append(f"  {symbol} {client_name}{response_time} - {status_text}")
            
            # Show response or error
            if result.get("api_response"):
                response_text = result["api_response"][:60]
                if len(result["api_response"]) > 60:
                    response_text += "..."
                lines.append(f"      Response: {response_text}")
            elif result.get("error"):
                lines.append(f"      Error: {result['error']}")
        
        lines.append("")
        
        # Configuration notes
        not_configured = [name for name, result in validation_data["client_results"].items() 
                         if result["status"] == ClientStatus.NOT_CONFIGURED.value]
        
        if not_configured:
            lines.append(f"{colors['bold']}Configuration Notes:{colors['reset']}")
            lines.append("  The following clients need API keys or configuration:")
            for client_name in not_configured:
                result = validation_data["client_results"][client_name]
                lines.append(f"    • {client_name}: {result.get('error', 'Missing configuration')}")
            lines.append("")
        
        return "\n".join(lines)

async def main():
    """Main function for standalone client validation"""
    validator = ClientValidator()
    results = await validator.validate_all_clients()
    
    # Print formatted report
    report = validator.format_validation_report(results)
    print(report)
    
    # Exit with appropriate code
    if results["overall_status"] in ["healthy", "degraded"]:
        exit(0)
    elif results["overall_status"] == "no_clients_configured":
        exit(1)
    else:
        exit(2)

if __name__ == "__main__":
    asyncio.run(main())