#!/usr/bin/env python3
"""
Comprehensive A2A System Health Check
Verifies all services are running and healthy
"""

import asyncio
import aiohttp
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

class A2AHealthChecker:
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {},
            "mcp_servers": {},
            "infrastructure": {},
            "blockchain": {},
            "summary": {}
        }
    
    async def check_service_health(self, url: str, service_name: str) -> Dict:
        """Check health of a single service"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "response_time": response.headers.get('response-time', 'N/A'),
                            "details": data
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": "unreachable",
                "error": str(e)
            }
    
    async def check_agents(self):
        """Check all A2A agents (ports 8000-8015)"""
        print("🤖 Checking A2A Agents...")
        
        agent_ports = {
            8000: "Registry Server",
            8001: "Data Agent", 
            8002: "Catalog Agent",
            8003: "Processing Agent",
            8004: "Validation Agent",
            8005: "QA Agent", 
            8006: "Quality Control",
            8007: "Processing Agent",
            8008: "Reasoning Agent",
            8009: "Analysis Agent",
            8010: "Agent Manager",
            8011: "Data Manager", 
            8012: "Catalog Manager",
            8013: "Calculation Agent",
            8014: "Agent Builder",
            8015: "Embedding Fine-Tuner"
        }
        
        tasks = []
        for port, name in agent_ports.items():
            tasks.append(self.check_service_health(f"http://localhost:{port}", name))
        
        results = await asyncio.gather(*tasks)
        
        healthy_count = 0
        for i, (port, name) in enumerate(agent_ports.items()):
            self.results["agents"][f"{name} (:{port})"] = results[i]
            if results[i]["status"] == "healthy":
                healthy_count += 1
                print(f"  ✅ {name} (:{port}) - {results[i]['status']}")
            else:
                print(f"  ❌ {name} (:{port}) - {results[i]['status']} - {results[i].get('error', '')}")
        
        print(f"📊 Agents: {healthy_count}/{len(agent_ports)} healthy")
        return healthy_count, len(agent_ports)
    
    async def check_mcp_servers(self):
        """Check all MCP servers (ports 8100-8109)"""
        print("\n🔧 Checking MCP Servers...")
        
        mcp_ports = {
            8100: "Enhanced Test Suite",
            8101: "Data Standardization",
            8102: "Vector Similarity", 
            8103: "Vector Ranking",
            8104: "Transport Layer",
            8105: "Reasoning Agent",
            8106: "Session Management",
            8107: "Resource Streaming",
            8108: "Confidence Calculator",
            8109: "Semantic Similarity"
        }
        
        tasks = []
        for port, name in mcp_ports.items():
            tasks.append(self.check_service_health(f"http://localhost:{port}", name))
        
        results = await asyncio.gather(*tasks)
        
        healthy_count = 0
        for i, (port, name) in enumerate(mcp_ports.items()):
            self.results["mcp_servers"][f"{name} (:{port})"] = results[i]
            if results[i]["status"] == "healthy":
                healthy_count += 1
                print(f"  ✅ {name} (:{port}) - {results[i]['status']}")
            else:
                print(f"  ❌ {name} (:{port}) - {results[i]['status']} - {results[i].get('error', '')}")
        
        print(f"📊 MCP Servers: {healthy_count}/{len(mcp_ports)} healthy")
        return healthy_count, len(mcp_ports)
    
    async def check_blockchain(self):
        """Check blockchain service"""
        print("\n⛓️  Checking Blockchain...")
        
        try:
            blockchain_health = await self.check_service_health("http://localhost:8545", "Blockchain")
            
            # Also check with JSON-RPC call
            async with aiohttp.ClientSession() as session:
                rpc_payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber", 
                    "params": [],
                    "id": 1
                }
                async with session.post("http://localhost:8545", json=rpc_payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        block_number = int(data.get("result", "0x0"), 16)
                        blockchain_health["details"] = {
                            "block_number": block_number,
                            "rpc_healthy": True
                        }
                        blockchain_health["status"] = "healthy"
                    else:
                        blockchain_health["details"] = {"rpc_healthy": False}
            
            self.results["blockchain"]["Anvil Blockchain"] = blockchain_health
            
            if blockchain_health["status"] == "healthy":
                print(f"  ✅ Anvil Blockchain - healthy (block #{blockchain_health['details']['block_number']})")
                return 1, 1
            else:
                print(f"  ❌ Anvil Blockchain - {blockchain_health['status']}")
                return 0, 1
                
        except Exception as e:
            print(f"  ❌ Blockchain check failed: {e}")
            self.results["blockchain"]["error"] = str(e)
            return 0, 1
    
    async def check_infrastructure(self):
        """Check infrastructure services"""
        print("\n🏗️  Checking Infrastructure...")
        
        infrastructure_services = {
            6379: "Redis Cache",
            3000: "Grafana",
            9090: "Prometheus"
        }
        
        healthy_count = 0
        total_count = len(infrastructure_services)
        
        for port, name in infrastructure_services.items():
            try:
                result = subprocess.run(
                    ["lsof", "-i", f":{port}"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0 and "LISTEN" in result.stdout:
                    self.results["infrastructure"][name] = {"status": "running", "port": port}
                    healthy_count += 1
                    print(f"  ✅ {name} (:{port}) - running")
                else:
                    self.results["infrastructure"][name] = {"status": "not_running", "port": port}
                    print(f"  ❌ {name} (:{port}) - not running")
            except Exception as e:
                self.results["infrastructure"][name] = {"status": "error", "error": str(e)}
                print(f"  ❌ {name} (:{port}) - error: {e}")
        
        print(f"📊 Infrastructure: {healthy_count}/{total_count} running")
        return healthy_count, total_count
    
    def generate_summary(self, agents_health, mcp_health, blockchain_health, infra_health):
        """Generate overall system health summary"""
        
        agent_healthy, agent_total = agents_health
        mcp_healthy, mcp_total = mcp_health  
        blockchain_healthy, blockchain_total = blockchain_health
        infra_healthy, infra_total = infra_health
        
        total_healthy = agent_healthy + mcp_healthy + blockchain_healthy + infra_healthy
        total_services = agent_total + mcp_total + blockchain_total + infra_total
        
        health_percentage = (total_healthy / total_services) * 100
        
        self.results["summary"] = {
            "total_services": total_services,
            "healthy_services": total_healthy,
            "health_percentage": round(health_percentage, 1),
            "agents": f"{agent_healthy}/{agent_total}",
            "mcp_servers": f"{mcp_healthy}/{mcp_total}",
            "blockchain": f"{blockchain_healthy}/{blockchain_total}",
            "infrastructure": f"{infra_healthy}/{infra_total}"
        }
        
        print(f"\n📊 SYSTEM HEALTH SUMMARY")
        print(f"{'='*50}")
        print(f"Overall Health: {health_percentage:.1f}% ({total_healthy}/{total_services} services)")
        print(f"🤖 A2A Agents: {agent_healthy}/{agent_total} healthy")
        print(f"🔧 MCP Servers: {mcp_healthy}/{mcp_total} healthy") 
        print(f"⛓️  Blockchain: {blockchain_healthy}/{blockchain_total} healthy")
        print(f"🏗️  Infrastructure: {infra_healthy}/{infra_total} running")
        
        if health_percentage >= 90:
            print("✅ SYSTEM STATUS: EXCELLENT")
        elif health_percentage >= 80:
            print("🟡 SYSTEM STATUS: GOOD") 
        elif health_percentage >= 70:
            print("🟠 SYSTEM STATUS: FAIR")
        else:
            print("🔴 SYSTEM STATUS: NEEDS ATTENTION")
        
        return health_percentage

async def main():
    print("🚀 A2A System Comprehensive Health Check")
    print("=" * 50)
    
    checker = A2AHealthChecker()
    
    # Run all health checks
    agents_health = await checker.check_agents()
    mcp_health = await checker.check_mcp_servers()
    blockchain_health = await checker.check_blockchain() 
    infra_health = await checker.check_infrastructure()
    
    # Generate summary
    health_percentage = checker.generate_summary(
        agents_health, mcp_health, blockchain_health, infra_health
    )
    
    # Save detailed results
    with open("a2a_health_report.json", "w") as f:
        json.dump(checker.results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: a2a_health_report.json")
    
    return health_percentage >= 80

if __name__ == "__main__":
    asyncio.run(main())