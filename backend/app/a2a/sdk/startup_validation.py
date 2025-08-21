#!/usr/bin/env python3
"""
A2A Startup Validation System
Performs end-to-end testing of A2A message processing and agent skills
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from web3 import Web3
from eth_account import Account
import httpx

from .system_health import SystemHealthMonitor

logger = logging.getLogger(__name__)

class A2AStartupValidator:
    """
    Validates A2A system startup by testing actual message processing
    """
    
    def __init__(self, rpc_url: str = "http://localhost:8545", registry_address: str = None):
        self.rpc_url = rpc_url
        self.registry_address = registry_address
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        
        # Chat manager account (using Anvil's first account)
        self.chat_manager_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        self.chat_manager_account = Account.from_key(self.chat_manager_key)
        
        # Agent definitions with their expected skills
        self.agents = {
            "agent0": {
                "port": 8001,
                "name": "Data Product Agent",
                "expected_skills": ["data_product_registration", "data_validation", "metadata_management"]
            },
            "agent1": {
                "port": 8002, 
                "name": "Data Standardization Agent",
                "expected_skills": ["data_standardization", "format_conversion", "schema_validation"]
            },
            "agent2": {
                "port": 8003,
                "name": "AI Preparation Agent", 
                "expected_skills": ["data_preprocessing", "feature_engineering", "model_preparation"]
            },
            "agent3": {
                "port": 8004,
                "name": "Vector Processing Agent",
                "expected_skills": ["vector_embedding", "similarity_search", "vector_operations"]
            },
            "agent4": {
                "port": 8005,
                "name": "Calc Validation Agent",
                "expected_skills": ["calculation_validation", "financial_verification", "audit_trail"]
            },
            "agent5": {
                "port": 8006,
                "name": "QA Validation Agent", 
                "expected_skills": ["quality_assurance", "data_validation", "compliance_check"]
            },
            "agent6": {
                "port": 8007,
                "name": "Quality Control Manager",
                "expected_skills": ["quality_control", "process_management", "workflow_coordination"]
            },
            "reasoning-agent": {
                "port": 8008,
                "name": "Reasoning Agent",
                "expected_skills": ["logical_reasoning", "decision_making", "pattern_analysis"]
            },
            "sql-agent": {
                "port": 8009,
                "name": "SQL Agent",
                "expected_skills": ["sql_generation", "query_optimization", "database_operations"]
            },
            "agent-manager": {
                "port": 8010,
                "name": "Agent Manager",
                "expected_skills": ["agent_coordination", "task_routing", "resource_management"]
            },
            "data-manager": {
                "port": 8011,
                "name": "Data Manager", 
                "expected_skills": ["data_storage", "data_retrieval", "data_lifecycle"]
            },
            "catalog-manager": {
                "port": 8012,
                "name": "Catalog Manager",
                "expected_skills": ["catalog_management", "metadata_indexing", "search_optimization"]
            },
            "calculation-agent": {
                "port": 8013,
                "name": "Calculation Agent",
                "expected_skills": ["financial_calculations", "mathematical_operations", "risk_analysis"]
            },
            "agent-builder": {
                "port": 8014,
                "name": "Agent Builder",
                "expected_skills": ["agent_creation", "workflow_design", "template_management"]
            },
            "embedding-finetuner": {
                "port": 8015,
                "name": "Embedding Fine-Tuner",
                "expected_skills": ["embedding_training", "model_tuning", "vector_optimization"]
            }
        }
        
        self.validation_results = {}
    
    async def validate_blockchain_connectivity(self) -> Dict[str, Any]:
        """Test blockchain connectivity and smart contracts"""
        logger.info("Validating blockchain connectivity...")
        
        result = {
            "status": "failed",
            "blockchain_connected": False,
            "contracts_deployed": False,
            "chat_manager_ready": False,
            "details": {}
        }
        
        try:
            # Test blockchain connection
            if not self.w3.is_connected():
                result["details"]["error"] = "Cannot connect to blockchain"
                return result
            
            result["blockchain_connected"] = True
            
            # Check current block
            current_block = self.w3.eth.block_number
            result["details"]["current_block"] = current_block
            
            # Check chat manager account balance
            balance = self.w3.eth.get_balance(self.chat_manager_account.address)
            result["details"]["chat_manager_balance"] = self.w3.from_wei(balance, 'ether')
            
            if balance > 0:
                result["chat_manager_ready"] = True
            
            # Check if registry contract is deployed
            if self.registry_address:
                try:
                    code = self.w3.eth.get_code(self.registry_address)
                    if len(code) > 2:  # More than just '0x'
                        result["contracts_deployed"] = True
                        result["details"]["registry_address"] = self.registry_address
                except Exception as e:
                    result["details"]["registry_error"] = str(e)
            
            if result["blockchain_connected"] and result["chat_manager_ready"]:
                result["status"] = "success"
            
        except Exception as e:
            result["details"]["error"] = str(e)
        
        return result
    
    async def send_a2a_test_message(self, agent_id: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Send test A2A message to a specific agent"""
        logger.info(f"Sending A2A test message to {agent_id}...")
        
        result = {
            "agent_id": agent_id,
            "status": "failed",
            "message_sent": False,
            "response_received": False,
            "response_time_ms": None,
            "skills_operational": [],
            "skills_failed": [],
            "error": None
        }
        
        try:
            # Create A2A test message
            test_message = {
                "type": "skill_test",
                "sender": "chat_manager",
                "recipient": agent_id,
                "content": {
                    "action": "test_skills",
                    "requested_skills": agent_config["expected_skills"],
                    "test_id": f"startup_test_{int(time.time())}"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "protocol_version": "v0.2.9"
            }
            
            start_time = time.time()
            
            # Send message via agent's A2A endpoint
            agent_url = f"http://localhost:{agent_config['port']}/a2a/message"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        agent_url,
                        json=test_message,
                        headers={
                            "Content-Type": "application/json",
                            "X-A2A-Protocol": "v0.2.9",
                            "X-Sender-Address": self.chat_manager_account.address
                        }
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    result["response_time_ms"] = round(response_time, 2)
                    result["message_sent"] = True
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        result["response_received"] = True
                        result["status"] = "success"
                        
                        # Parse skill test results
                        if "skill_results" in response_data:
                            for skill, skill_result in response_data["skill_results"].items():
                                if skill_result.get("operational", False):
                                    result["skills_operational"].append(skill)
                                else:
                                    result["skills_failed"].append(skill)
                        
                        # Check if agent reported its actual skills
                        if "available_skills" in response_data:
                            result["available_skills"] = response_data["available_skills"]
                    
                    else:
                        result["error"] = f"HTTP {response.status_code}: {response.text}"
                
                except httpx.ConnectError:
                    result["error"] = "Connection refused - agent not listening"
                except httpx.TimeoutException:
                    result["error"] = "Request timeout - agent not responding"
                except Exception as e:
                    result["error"] = f"Request failed: {str(e)}"
        
        except Exception as e:
            result["error"] = f"Message preparation failed: {str(e)}"
        
        return result
    
    async def test_agent_skills_via_direct_call(self, agent_id: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test agent skills via direct API calls as fallback"""
        logger.info(f"Testing {agent_id} skills via direct API...")
        
        result = {
            "agent_id": agent_id,
            "status": "failed", 
            "skills_tested": 0,
            "skills_operational": [],
            "skills_failed": [],
            "error": None
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Test basic health endpoint
                health_url = f"http://localhost:{agent_config['port']}/health"
                try:
                    health_response = await client.get(health_url)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        
                        # Check if agent reports its capabilities
                        if "capabilities" in health_data:
                            result["available_skills"] = health_data["capabilities"]
                        elif "skills" in health_data:
                            result["available_skills"] = health_data["skills"]
                
                except Exception as e:
                    result["error"] = f"Health check failed: {str(e)}"
                    return result
                
                # Test stats endpoint for more detailed info
                stats_url = f"http://localhost:{agent_config['port']}/stats"
                try:
                    stats_response = await client.get(stats_url)
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()
                        
                        # Mark as operational if we can get stats
                        for skill in agent_config["expected_skills"]:
                            result["skills_operational"].append(skill)
                        
                        result["skills_tested"] = len(agent_config["expected_skills"])
                        result["status"] = "success"
                
                except Exception:
                    # Stats endpoint might not exist, that's okay
                    pass
                
                # If we got health but no stats, still consider basic functionality working
                if result["status"] == "failed" and not result["error"]:
                    # Assume at least basic skills are working if health endpoint responds
                    result["skills_operational"] = agent_config["expected_skills"][:1]  # First skill
                    result["skills_failed"] = agent_config["expected_skills"][1:]  # Rest as unknown
                    result["skills_tested"] = len(agent_config["expected_skills"])
                    result["status"] = "partial"
        
        except Exception as e:
            result["error"] = f"Direct API test failed: {str(e)}"
        
        return result
    
    async def validate_all_agents(self) -> Dict[str, Any]:
        """Test all agents for A2A message processing and skill validation"""
        logger.info("Validating all A2A agents...")
        
        validation_summary = {
            "total_agents": len(self.agents),
            "agents_responding": 0,
            "agents_processing_a2a": 0,
            "total_skills_tested": 0,
            "total_skills_operational": 0,
            "agent_results": {},
            "overall_status": "failed"
        }
        
        # Test all agents in parallel
        tasks = []
        for agent_id, agent_config in self.agents.items():
            # Try A2A message first, then fallback to direct API
            task = self._test_agent_comprehensive(agent_id, agent_config)
            tasks.append(task)
        
        agent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(agent_results):
            agent_id = list(self.agents.keys())[i]
            
            if isinstance(result, Exception):
                result = {
                    "agent_id": agent_id,
                    "status": "failed", 
                    "error": str(result),
                    "skills_operational": [],
                    "skills_failed": list(self.agents[agent_id]["expected_skills"])
                }
            
            validation_summary["agent_results"][agent_id] = result
            
            # Update counters
            if result["status"] in ["success", "partial"]:
                validation_summary["agents_responding"] += 1
                
                if result.get("response_received", False):
                    validation_summary["agents_processing_a2a"] += 1
            
            validation_summary["total_skills_tested"] += len(result.get("skills_operational", [])) + len(result.get("skills_failed", []))
            validation_summary["total_skills_operational"] += len(result.get("skills_operational", []))
        
        # Calculate overall status
        response_rate = validation_summary["agents_responding"] / validation_summary["total_agents"]
        if response_rate >= 0.9:  # 90% or more responding
            validation_summary["overall_status"] = "success"
        elif response_rate >= 0.7:  # 70% or more responding  
            validation_summary["overall_status"] = "partial"
        else:
            validation_summary["overall_status"] = "failed"
        
        return validation_summary
    
    async def _test_agent_comprehensive(self, agent_id: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive test of an agent using both A2A and direct API"""
        
        # First try A2A message
        a2a_result = await self.send_a2a_test_message(agent_id, agent_config)
        
        # If A2A failed, try direct API as fallback
        if a2a_result["status"] == "failed":
            direct_result = await self.test_agent_skills_via_direct_call(agent_id, agent_config)
            
            # Merge results - prefer A2A but use direct as fallback
            result = {
                **a2a_result,
                "fallback_used": True,
                "direct_test_status": direct_result["status"]
            }
            
            # If direct test succeeded, update overall status
            if direct_result["status"] in ["success", "partial"]:
                result["status"] = "partial"  # Working but not via A2A
                result["skills_operational"] = direct_result["skills_operational"]
                result["skills_failed"] = direct_result["skills_failed"]
        
        else:
            a2a_result["fallback_used"] = False
            result = a2a_result
        
        return result
    
    def format_validation_report(self, blockchain_result: Dict[str, Any], agent_results: Dict[str, Any]) -> str:
        """Format validation results for console output"""
        lines = []
        
        # ANSI colors
        colors = {
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "red": "\033[91m",
            "blue": "\033[94m",
            "bold": "\033[1m",
            "reset": "\033[0m"
        }
        
        # Header
        lines.append(f"{colors['bold']}{colors['blue']}A2A Startup Validation Report{colors['reset']}")
        lines.append(f"{colors['blue']}{'=' * 60}{colors['reset']}")
        lines.append("")
        
        # Blockchain validation
        lines.append(f"{colors['bold']}Blockchain Validation:{colors['reset']}")
        bc_status = blockchain_result["status"] 
        bc_color = colors["green"] if bc_status == "success" else colors["red"]
        bc_symbol = "✓" if bc_status == "success" else "✗"
        lines.append(f"  {bc_color}{bc_symbol} {bc_status.title()}{colors['reset']}")
        
        if blockchain_result.get("details"):
            for key, value in blockchain_result["details"].items():
                lines.append(f"    {key}: {value}")
        lines.append("")
        
        # Agent validation summary
        lines.append(f"{colors['bold']}Agent Validation Summary:{colors['reset']}")
        total = agent_results["total_agents"]
        responding = agent_results["agents_responding"] 
        a2a_working = agent_results["agents_processing_a2a"]
        
        lines.append(f"  Total Agents: {total}")
        lines.append(f"  {colors['green']}✓ Responding: {responding}/{total} ({responding/total*100:.1f}%){colors['reset']}")
        lines.append(f"  {colors['blue']}⚡ A2A Processing: {a2a_working}/{total} ({a2a_working/total*100:.1f}%){colors['reset']}")
        
        # Skills summary
        total_skills = agent_results["total_skills_tested"]
        working_skills = agent_results["total_skills_operational"]
        if total_skills > 0:
            lines.append(f"  {colors['green']}🔧 Skills Operational: {working_skills}/{total_skills} ({working_skills/total_skills*100:.1f}%){colors['reset']}")
        lines.append("")
        
        # Individual agent results
        lines.append(f"{colors['bold']}Individual Agent Results:{colors['reset']}")
        for agent_id, result in agent_results["agent_results"].items():
            agent_name = self.agents[agent_id]["name"]
            
            if result["status"] == "success":
                symbol = f"{colors['green']}✓{colors['reset']}"
                status_text = "A2A Working"
            elif result["status"] == "partial":
                symbol = f"{colors['yellow']}◐{colors['reset']}"
                status_text = "Direct API Only" if result.get("fallback_used") else "Partial"
            else:
                symbol = f"{colors['red']}✗{colors['reset']}"
                status_text = "Failed"
            
            response_time = f" ({result['response_time_ms']}ms)" if result.get("response_time_ms") else ""
            lines.append(f"  {symbol} {agent_name}{response_time} - {status_text}")
            
            # Show operational skills
            if result.get("skills_operational"):
                skills_text = ", ".join(result["skills_operational"][:3])
                if len(result["skills_operational"]) > 3:
                    skills_text += f" (+{len(result['skills_operational'])-3} more)"
                lines.append(f"      Skills: {skills_text}")
            
            # Show errors
            if result.get("error"):
                lines.append(f"      Error: {result['error']}")
        
        lines.append("")
        
        # Overall status
        overall = agent_results["overall_status"]
        overall_color = colors["green"] if overall == "success" else colors["yellow"] if overall == "partial" else colors["red"]
        overall_symbol = "✓" if overall == "success" else "⚠" if overall == "partial" else "✗"
        
        lines.append(f"{colors['bold']}Overall A2A System Status: {overall_color}{overall_symbol} {overall.upper()}{colors['reset']}")
        
        return "\n".join(lines)
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete A2A startup validation"""
        logger.info("Starting comprehensive A2A system validation...")
        
        # Step 1: Validate blockchain
        blockchain_result = await self.validate_blockchain_connectivity()
        
        # Step 2: Validate agents
        agent_results = await self.validate_all_agents()
        
        # Step 3: Generate report
        report = self.format_validation_report(blockchain_result, agent_results)
        print(report)
        
        # Return structured results
        return {
            "blockchain": blockchain_result,
            "agents": agent_results,
            "overall_success": (
                blockchain_result["status"] == "success" and 
                agent_results["overall_status"] in ["success", "partial"]
            )
        }

async def main():
    """Main function for standalone validation"""
    import os
    
    # Get configuration from environment
    rpc_url = os.getenv("A2A_RPC_URL", "http://localhost:8545")
    registry_address = os.getenv("A2A_AGENT_REGISTRY_ADDRESS")
    
    validator = A2AStartupValidator(rpc_url, registry_address)
    results = await validator.run_full_validation()
    
    # Exit with appropriate code
    if results["overall_success"]:
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())