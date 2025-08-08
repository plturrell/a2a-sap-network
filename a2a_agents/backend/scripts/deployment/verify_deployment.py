#!/usr/bin/env python3
"""
A2A Deployment Verification Script
Verifies that all components are properly deployed and functional
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, List, Tuple
from datetime import datetime

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class DeploymentVerifier:
    def __init__(self):
        self.services = {
            "Data Manager": "http://localhost:8001",
            "Catalog Manager": "http://localhost:8002",
            "Agent 0 (Data Product)": "http://localhost:8003",
            "Agent 1 (Standardization)": "http://localhost:8004",
            "Agent 2 (AI Preparation)": "http://localhost:8005",
            "Agent 3 (Vector Processing)": "http://localhost:8008",
            "Agent 4 (Calc Validation)": "http://localhost:8006",
            "Agent 5 (QA Validation)": "http://localhost:8007",
        }
        
        self.blockchain_rpc = "http://localhost:8545"
        self.contracts = {
            "BusinessDataCloudA2A": "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
            "AgentRegistry": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "MessageRouter": "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
        }
        
        self.checks_passed = 0
        self.checks_failed = 0
        
    async def verify_all(self) -> bool:
        """Run all verification checks"""
        print(f"{BLUE}🔍 A2A Deployment Verification{RESET}")
        print(f"{BLUE}=============================={RESET}\n")
        
        # Check services
        await self.check_services()
        
        # Check blockchain
        await self.check_blockchain()
        
        # Check inter-service communication
        await self.check_communication()
        
        # Check trust system
        await self.check_trust_system()
        
        # Summary
        self.print_summary()
        
        return self.checks_failed == 0
    
    async def check_services(self):
        """Check if all services are running and healthy"""
        print(f"{BLUE}1️⃣ Checking Services{RESET}")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            for name, url in self.services.items():
                try:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("status") == "healthy":
                                print(f"{GREEN}✅ {name}: Healthy{RESET}")
                                self.checks_passed += 1
                            else:
                                print(f"{YELLOW}⚠️  {name}: Unhealthy - {data}{RESET}")
                                self.checks_failed += 1
                        else:
                            print(f"{RED}❌ {name}: HTTP {resp.status}{RESET}")
                            self.checks_failed += 1
                except Exception as e:
                    print(f"{RED}❌ {name}: Not reachable - {type(e).__name__}{RESET}")
                    self.checks_failed += 1
        print()
    
    async def check_blockchain(self):
        """Check blockchain connectivity and smart contracts"""
        print(f"{BLUE}2️⃣ Checking Blockchain{RESET}")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # Check RPC connection
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "eth_blockNumber",
                    "params": [],
                    "id": 1
                }
                async with session.post(self.blockchain_rpc, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        block_num = int(data["result"], 16)
                        print(f"{GREEN}✅ Blockchain RPC: Connected (Block #{block_num}){RESET}")
                        self.checks_passed += 1
                    else:
                        print(f"{RED}❌ Blockchain RPC: Connection failed{RESET}")
                        self.checks_failed += 1
            except Exception as e:
                print(f"{RED}❌ Blockchain RPC: {type(e).__name__}{RESET}")
                self.checks_failed += 1
            
            # Check smart contracts
            for name, address in self.contracts.items():
                try:
                    payload = {
                        "jsonrpc": "2.0",
                        "method": "eth_getCode",
                        "params": [address, "latest"],
                        "id": 1
                    }
                    async with session.post(self.blockchain_rpc, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            code = data.get("result", "0x")
                            if len(code) > 3:  # More than just "0x"
                                print(f"{GREEN}✅ {name}: Deployed at {address}{RESET}")
                                self.checks_passed += 1
                            else:
                                print(f"{RED}❌ {name}: No code at {address}{RESET}")
                                self.checks_failed += 1
                except Exception as e:
                    print(f"{RED}❌ {name}: Check failed - {type(e).__name__}{RESET}")
                    self.checks_failed += 1
        print()
    
    async def check_communication(self):
        """Check inter-service communication"""
        print(f"{BLUE}3️⃣ Checking Inter-Service Communication{RESET}")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # Test Data Manager -> Catalog Manager
            try:
                test_data = {
                    "test_id": "verify_001",
                    "data": {"test": "communication"}
                }
                
                # Store in Data Manager
                async with session.post(
                    f"{self.services['Data Manager']}/api/data",
                    json=test_data
                ) as resp:
                    if resp.status in [200, 201]:
                        print(f"{GREEN}✅ Data Manager: Can store data{RESET}")
                        self.checks_passed += 1
                    else:
                        print(f"{RED}❌ Data Manager: Storage failed{RESET}")
                        self.checks_failed += 1
                
                # Search in Catalog Manager
                search_payload = {"query": "verify_001"}
                async with session.post(
                    f"{self.services['Catalog Manager']}/api/search",
                    json=search_payload
                ) as resp:
                    if resp.status == 200:
                        print(f"{GREEN}✅ Catalog Manager: Search functional{RESET}")
                        self.checks_passed += 1
                    else:
                        print(f"{YELLOW}⚠️  Catalog Manager: Search returned {resp.status}{RESET}")
                        
            except Exception as e:
                print(f"{RED}❌ Communication test failed: {type(e).__name__}{RESET}")
                self.checks_failed += 1
        print()
    
    async def check_trust_system(self):
        """Check trust system between agents"""
        print(f"{BLUE}4️⃣ Checking Trust System{RESET}")
        print("-" * 40)
        
        async with aiohttp.ClientSession() as session:
            # Check Agent 0 public key endpoint
            try:
                async with session.get(
                    f"{self.services['Agent 0 (Data Product)']}/trust/public-key"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "public_key" in data:
                            print(f"{GREEN}✅ Agent 0: Trust system active{RESET}")
                            self.checks_passed += 1
                        else:
                            print(f"{YELLOW}⚠️  Agent 0: No public key{RESET}")
                            self.checks_failed += 1
                    else:
                        print(f"{RED}❌ Agent 0: Trust endpoint failed{RESET}")
                        self.checks_failed += 1
            except Exception as e:
                print(f"{RED}❌ Trust system check failed: {type(e).__name__}{RESET}")
                self.checks_failed += 1
        print()
    
    def print_summary(self):
        """Print verification summary"""
        total_checks = self.checks_passed + self.checks_failed
        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        
        print(f"{BLUE}📊 Verification Summary{RESET}")
        print("=" * 40)
        print(f"Total Checks: {total_checks}")
        print(f"{GREEN}Passed: {self.checks_passed}{RESET}")
        print(f"{RED}Failed: {self.checks_failed}{RESET}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if self.checks_failed == 0:
            print(f"{GREEN}✅ Deployment verification PASSED!{RESET}")
            print("All components are properly deployed and functional.")
        else:
            print(f"{RED}❌ Deployment verification FAILED!{RESET}")
            print("Please check the failed components above.")
        
        print()
        print(f"Timestamp: {datetime.now().isoformat()}")


async def main():
    verifier = DeploymentVerifier()
    success = await verifier.verify_all()
    
    if not success:
        print(f"\n{YELLOW}💡 Troubleshooting Tips:{RESET}")
        print("1. Check if all services are running: ps aux | grep python")
        print("2. Check logs: tail -f deployment_logs/*.log")
        print("3. Verify ports: lsof -i:8001-8008")
        print("4. Check blockchain: cast block-number --rpc-url http://localhost:8545")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)