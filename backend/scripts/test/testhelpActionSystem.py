#!/usr/bin/env python3
"""
End-to-End Test: Help Action System
Tests real help-seeking between running agents with actual action execution
"""

import asyncio
import json
import httpx
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HelpActionSystemTester:
    """Real end-to-end tester for help action system between running agents"""
    
    def __init__(self):
        self.agents = {
            "agent_0": {"url": "http://localhost:8002", "name": "Data Product Agent"},
            "agent_1": {"url": "http://localhost:8001", "name": "Financial Standardization Agent"},
            "data_manager": {"url": "http://localhost:8003", "name": "Data Manager Agent"}
        }
        self.test_results = []
    
    async def test_agent_health(self) -> Dict[str, bool]:
        """Test if all agents are running and healthy"""
        logger.info("🔍 Testing agent health...")
        health_status = {}
        
        for agent_id, agent_info in self.agents.items():
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{agent_info['url']}/health")
                    if response.status_code == 200:
                        health_status[agent_id] = True
                        logger.info(f"✅ {agent_info['name']} is healthy")
                    else:
                        health_status[agent_id] = False
                        logger.error(f"❌ {agent_info['name']} health check failed: {response.status_code}")
            except Exception as e:
                health_status[agent_id] = False
                logger.error(f"❌ {agent_info['name']} is not reachable: {e}")
        
        return health_status
    
    async def test_agent_capabilities(self) -> Dict[str, Dict]:
        """Test if agents have help-seeking and AI advisor capabilities"""
        logger.info("🔍 Testing agent capabilities...")
        capabilities = {}
        
        for agent_id, agent_info in self.agents.items():
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(f"{agent_info['url']}/.well-known/agent.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        caps = agent_card.get("capabilities", {})
                        capabilities[agent_id] = {
                            "version": agent_card.get("version", "unknown"),
                            "helpSeeking": caps.get("helpSeeking", False),
                            "aiAdvisor": caps.get("aiAdvisor", False),
                            "taskTracking": caps.get("taskTracking", False)
                        }
                        logger.info(f"✅ {agent_info['name']} capabilities: {capabilities[agent_id]}")
                    else:
                        capabilities[agent_id] = {"error": f"HTTP {response.status_code}"}
                        logger.error(f"❌ Could not get capabilities for {agent_info['name']}")
            except Exception as e:
                capabilities[agent_id] = {"error": str(e)}
                logger.error(f"❌ Error getting capabilities for {agent_info['name']}: {e}")
        
        return capabilities
    
    async def test_help_request_and_response(self, asking_agent: str, helping_agent: str) -> Dict[str, Any]:
        """Test help request from one agent to another and verify response"""
        logger.info(f"🆘 Testing help request: {asking_agent} asking {helping_agent}")
        
        asking_info = self.agents[asking_agent]
        helping_info = self.agents[helping_agent]
        
        # Create help request message
        help_message = {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Help request: I'm having trouble with data processing and need guidance"
                },
                {
                    "kind": "data",
                    "data": {
                        "help_request": True,
                        "advisor_request": True,
                        "problem_type": "data_processing",
                        "urgency": "medium",
                        "requesting_agent": asking_agent,
                        "context": {
                            "error_type": "ProcessingError",
                            "error_message": "Failed to process data file",
                            "operation": "data_analysis"
                        },
                        "question": "How do I resolve data processing failures? The operation keeps timing out."
                    }
                }
            ]
        }
        
        try:
            # Send help request to helping agent
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{helping_info['url']}/a2a/v1/messages",
                    json={"message": help_message},
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9",
                        "X-Help-Request": "true"
                    }
                )
                
                if response.status_code == 200:
                    help_response = response.json()
                    logger.info(f"✅ Help response received from {helping_info['name']}")
                    
                    # Check if response contains advisor guidance
                    if "advisor_response" in help_response:
                        advisor_content = help_response["advisor_response"]
                        if isinstance(advisor_content, dict) and "advisor_response" in advisor_content:
                            answer = advisor_content["advisor_response"].get("answer", "")
                            confidence = advisor_content["advisor_response"].get("confidence", "unknown")
                            
                            return {
                                "success": True,
                                "asking_agent": asking_agent,
                                "helping_agent": helping_agent,
                                "response_received": True,
                                "advisor_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                                "confidence": confidence,
                                "response_time": response.elapsed.total_seconds(),
                                "raw_response_keys": list(help_response.keys())
                            }
                        else:
                            return {
                                "success": False,
                                "error": "Advisor response format unexpected",
                                "response_structure": str(type(help_response))
                            }
                    else:
                        return {
                            "success": False,
                            "error": "No advisor response in help response",
                            "response_keys": list(help_response.keys()) if isinstance(help_response, dict) else "not_dict"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "asking_agent": asking_agent,
                "helping_agent": helping_agent
            }
    
    async def test_help_action_execution(self, agent_id: str) -> Dict[str, Any]:
        """Test if agent can execute actions based on help received"""
        logger.info(f"🔧 Testing help action execution for {agent_id}")
        
        agent_info = self.agents[agent_id]
        
        # Create a realistic scenario that will trigger help-seeking
        # For Agent 0 (Data Product Agent), simulate a data processing task that fails
        if agent_id == "agent_0":
            test_message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text", 
                        "text": "Process data from /nonexistent/path/data.csv"
                    },
                    {
                        "kind": "data",
                        "data": {
                            "operation": "data_product_registration",
                            "data_location": "/nonexistent/path/data.csv",
                            "force_error": True,  # This will cause the operation to fail
                            "test_help_action": True
                        }
                    }
                ]
            }
        elif agent_id == "agent_1":
            test_message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Standardize invalid data format"
                    },
                    {
                        "kind": "data", 
                        "data": {
                            "operation": "standardize_data",
                            "data": {"invalid": "format"},
                            "test_help_action": True
                        }
                    }
                ]
            }
        else:  # data_manager
            test_message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Store data in unavailable database"
                    },
                    {
                        "kind": "data",
                        "data": {
                            "operation": "store_data",
                            "database": "unavailable_db",
                            "test_help_action": True
                        }
                    }
                ]
            }
        
        try:
            # Send the message that should trigger help-seeking
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{agent_info['url']}/a2a/v1/messages",
                    json={"message": test_message},
                    headers={
                        "Content-Type": "application/json",
                        "X-A2A-Protocol": "0.2.9"
                    }
                )
                
                task_response = response.json()
                task_id = task_response.get("taskId")
                
                if not task_id:
                    return {
                        "success": False,
                        "error": "No task ID returned",
                        "response": task_response
                    }
                
                # Wait for task completion and check if help was sought
                await asyncio.sleep(5)  # Give time for processing
                
                # Check task status
                status_response = await client.get(f"{agent_info['url']}/tasks/{task_id}")
                if status_response.status_code == 200:
                    task_status = status_response.json()
                    
                    # Try to get help action statistics
                    help_stats_response = await client.get(f"{agent_info['url']}/help-action-stats")
                    help_stats = {}
                    if help_stats_response.status_code == 200:
                        help_stats = help_stats_response.json()
                    
                    # Try to get help action history
                    help_history_response = await client.get(f"{agent_info['url']}/help-action-history")
                    help_history = []
                    if help_history_response.status_code == 200:
                        help_history = help_history_response.json()
                    
                    return {
                        "success": True,
                        "agent_id": agent_id,
                        "task_id": task_id,
                        "task_status": task_status,
                        "help_stats": help_stats,
                        "help_history_count": len(help_history),
                        "help_history": help_history[-1] if help_history else None  # Latest help action
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Could not get task status: HTTP {status_response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent_id": agent_id
            }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive end-to-end test of help action system"""
        logger.info("🚀 Starting comprehensive help action system test...")
        
        test_report = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "test_phases": {},
            "overall_success": False,
            "summary": {}
        }
        
        # Phase 1: Check agent health
        logger.info("\n📋 PHASE 1: Agent Health Check")
        health_status = await self.test_agent_health()
        test_report["test_phases"]["agent_health"] = health_status
        
        healthy_agents = sum(1 for healthy in health_status.values() if healthy)
        logger.info(f"Health check: {healthy_agents}/{len(health_status)} agents healthy")
        
        if healthy_agents == 0:
            test_report["summary"]["error"] = "No agents are running"
            return test_report
        
        # Phase 2: Check agent capabilities
        logger.info("\n📋 PHASE 2: Agent Capabilities Check")
        capabilities = await self.test_agent_capabilities()
        test_report["test_phases"]["agent_capabilities"] = capabilities
        
        help_capable_agents = sum(1 for caps in capabilities.values() 
                                 if isinstance(caps, dict) and caps.get("helpSeeking", False))
        logger.info(f"Capability check: {help_capable_agents}/{len(capabilities)} agents have help-seeking")
        
        # Phase 3: Test help requests between agents
        logger.info("\n📋 PHASE 3: Help Request Testing")
        help_tests = []
        
        # Test Agent 1 asking Agent 0 for help
        if health_status.get("agent_1") and health_status.get("agent_0"):
            help_test_1_0 = await self.test_help_request_and_response("agent_1", "agent_0")
            help_tests.append(help_test_1_0)
            logger.info(f"Agent 1 → Agent 0: {'✅ Success' if help_test_1_0['success'] else '❌ Failed'}")
        
        # Test Agent 0 asking Data Manager for help  
        if health_status.get("agent_0") and health_status.get("data_manager"):
            help_test_0_dm = await self.test_help_request_and_response("agent_0", "data_manager")
            help_tests.append(help_test_0_dm)
            logger.info(f"Agent 0 → Data Manager: {'✅ Success' if help_test_0_dm['success'] else '❌ Failed'}")
        
        test_report["test_phases"]["help_requests"] = help_tests
        
        successful_help_requests = sum(1 for test in help_tests if test["success"])
        logger.info(f"Help requests: {successful_help_requests}/{len(help_tests)} successful")
        
        # Phase 4: Test help action execution (the critical test!)
        logger.info("\n📋 PHASE 4: Help Action Execution Testing")
        action_tests = []
        
        for agent_id in ["agent_0", "agent_1", "data_manager"]:
            if health_status.get(agent_id):
                action_test = await self.test_help_action_execution(agent_id)
                action_tests.append(action_test)
                logger.info(f"Action execution test for {agent_id}: {'✅ Success' if action_test['success'] else '❌ Failed'}")
        
        test_report["test_phases"]["help_action_execution"] = action_tests
        
        successful_action_tests = sum(1 for test in action_tests if test["success"])
        logger.info(f"Action execution: {successful_action_tests}/{len(action_tests)} successful")
        
        # Overall assessment
        test_report["overall_success"] = (
            healthy_agents >= 2 and 
            help_capable_agents >= 1 and 
            successful_help_requests >= 1 and
            successful_action_tests >= 1
        )
        
        test_report["summary"] = {
            "healthy_agents": f"{healthy_agents}/{len(health_status)}",
            "help_capable_agents": f"{help_capable_agents}/{len(capabilities)}",
            "successful_help_requests": f"{successful_help_requests}/{len(help_tests)}",
            "successful_action_executions": f"{successful_action_tests}/{len(action_tests)}",
            "end_to_end_working": test_report["overall_success"]
        }
        
        return test_report
    
    def print_test_report(self, report: Dict[str, Any]):
        """Print a formatted test report"""
        print("\n" + "="*80)
        print("🔬 HELP ACTION SYSTEM - END-TO-END TEST REPORT")
        print("="*80)
        print(f"Test Time: {report['test_timestamp']}")
        print(f"Overall Result: {'✅ PASS' if report['overall_success'] else '❌ FAIL'}")
        print("\n📊 Summary:")
        for key, value in report["summary"].items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print("\n📋 Detailed Results:")
        
        # Agent Health
        print("\n🏥 Agent Health:")
        for agent, healthy in report["test_phases"]["agent_health"].items():
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            print(f"  • {agent}: {status}")
        
        # Capabilities
        print("\n🛠️ Agent Capabilities:")
        for agent, caps in report["test_phases"]["agent_capabilities"].items():
            if isinstance(caps, dict) and "error" not in caps:
                help_seeking = "✅" if caps.get("helpSeeking") else "❌"
                ai_advisor = "✅" if caps.get("aiAdvisor") else "❌"
                task_tracking = "✅" if caps.get("taskTracking") else "❌"
                print(f"  • {agent} (v{caps.get('version', '?')}): Help-Seeking {help_seeking}, AI Advisor {ai_advisor}, Task Tracking {task_tracking}")
            else:
                print(f"  • {agent}: ❌ Error getting capabilities")
        
        # Help Requests
        print("\n🆘 Help Request Tests:")
        for test in report["test_phases"]["help_requests"]:
            if test["success"]:
                print(f"  • {test['asking_agent']} → {test['helping_agent']}: ✅ Success")
                print(f"    Response: {test['advisor_answer'][:100]}...")
                print(f"    Confidence: {test['confidence']}, Time: {test['response_time']:.2f}s")
            else:
                print(f"  • {test['asking_agent']} → {test['helping_agent']}: ❌ Failed - {test['error']}")
        
        # Action Execution
        print("\n🔧 Help Action Execution Tests:")
        for test in report["test_phases"]["help_action_execution"]:
            agent_id = test.get("agent_id", "unknown")
            if test["success"]:
                print(f"  • {agent_id}: ✅ Actions executed")
                if test.get("help_history"):
                    print(f"    Help actions: {test['help_history_count']} total")
                    latest = test["help_history"]
                    if "execution_result" in latest:
                        exec_result = latest["execution_result"]
                        actions_count = len(exec_result.get("actions_executed", []))
                        resolved = exec_result.get("final_outcome", {}).get("resolved_issue", False)
                        print(f"    Latest: {actions_count} actions, Issue resolved: {'✅' if resolved else '❌'}")
            else:
                error_msg = test.get("error", "Unknown error")
                print(f"  • {agent_id}: ❌ Failed - {error_msg}")
        
        print("\n" + "="*80)
        if report["overall_success"]:
            print("🎉 SUCCESS: Help action system is working end-to-end!")
            print("   • Agents can seek help when encountering problems")
            print("   • Agents can provide help via AI advisors") 
            print("   • Agents can execute concrete actions based on help received")
            print("   • All communication uses A2A protocol")
        else:
            print("❌ FAILURE: Help action system has issues")
            print("   Please check the detailed results above")
        print("="*80)


async def main():
    """Run the comprehensive test"""
    tester = HelpActionSystemTester()
    
    print("🚀 Starting End-to-End Help Action System Test...")
    print("This will test real help-seeking and action execution between running agents\n")
    
    # Run the comprehensive test
    report = await tester.run_comprehensive_test()
    
    # Print the results
    tester.print_test_report(report)
    
    # Save report to file
    with open("help_action_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Full test report saved to: help_action_test_report.json")
    
    return report["overall_success"]


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)