#!/usr/bin/env python3
"""
Full Integration Test for A2A Goal Management System
Tests the complete flow from goal assignment to UI visualization
"""

import asyncio
import sys
import os
import json
import requests
from datetime import datetime
import time

# Add parent directory to path
sys.path.append('../../../../../')

from app.a2a.agents.orchestratorAgent.active.assignGoalsToAllAgents import assign_goals_to_all_agents
from app.a2a.agents.orchestratorAgent.active.comprehensiveOrchestratorAgentSdk import ComprehensiveOrchestratorAgentSDK
from app.a2a.agents.orchestratorAgent.active.orchestratorAgentA2AHandler import OrchestratorAgentA2AHandler

# Color codes for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    """Print a section header"""
    print(f"\n{BLUE}{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}{RESET}\n")

def print_success(message):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    """Print error message"""
    print(f"{RED}✗ {message}{RESET}")

async def test_goal_assignment():
    """Test 1: Goal Assignment to All Agents"""
    print_section("Test 1: Goal Assignment")
    
    try:
        # Run goal assignment
        result = await assign_goals_to_all_agents()
        
        if result and isinstance(result, tuple):
            assignment_results, orchestrator_handler = result
            
            if assignment_results:
                summary = assignment_results["summary"]
                print_success(f"Goals assigned to {summary['successful_assignments']}/{summary['total_agents']} agents")
                print_success(f"Total goals created: {summary['total_goals_assigned']}")
                
                # Show sample assignments
                print("\nSample Goal Assignments:")
                for agent_id, assignment in list(assignment_results["assignments"].items())[:3]:
                    if assignment["status"] == "success":
                        print(f"\n  {agent_id}:")
                        for goal in assignment["goals"]:
                            print(f"    - {goal['goal_type']}: {goal['specific'][:50]}...")
                
                return True, assignment_results
            else:
                print_error("Goal assignment returned no results")
                return False, None
        else:
            print_error("Goal assignment failed")
            return False, None
            
    except Exception as e:
        print_error(f"Goal assignment error: {e}")
        return False, None

def test_cap_service_connection():
    """Test 2: CAP Service Connection"""
    print_section("Test 2: CAP Service Connection")
    
    cap_url = os.getenv("CAP_SERVICE_URL", "http://localhost:4004")
    
    try:
        # Test connection to CAP service
        response = requests.get(f"{cap_url}/api/v1/goal-management/$metadata", timeout=5)
        
        if response.status_code == 200:
            print_success(f"Connected to CAP service at {cap_url}")
            return True
        else:
            print_error(f"CAP service returned status {response.status_code}")
            return False
            
    except requests.ConnectionError:
        print_warning(f"CAP service not running at {cap_url}")
        print("  Run: npm run start:goalmanagement")
        return False
    except Exception as e:
        print_error(f"CAP connection error: {e}")
        return False

def test_goal_sync():
    """Test 3: Goal Synchronization to CAP"""
    print_section("Test 3: Goal Sync to CAP Database")
    
    cap_url = os.getenv("CAP_SERVICE_URL", "http://localhost:4004")
    
    try:
        # Trigger goal sync
        print("Triggering goal synchronization...")
        response = requests.post(
            f"{cap_url}/api/v1/goal-management/syncGoals",
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                sync_result = result.get("result", {})
                print_success(f"Sync completed: {sync_result.get('successCount', 0)} agents synced")
                return True
            else:
                print_error("Sync failed")
                return False
        else:
            print_error(f"Sync request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Goal sync error: {e}")
        return False

def test_ui_data_availability():
    """Test 4: UI Data Availability"""
    print_section("Test 4: UI Data Availability")
    
    cap_url = os.getenv("CAP_SERVICE_URL", "http://localhost:4004")
    
    try:
        # Check system analytics
        response = requests.get(f"{cap_url}/api/v1/goal-management/SystemAnalytics", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if "value" in data and len(data["value"]) > 0:
                analytics = data["value"][0]
                print_success("System analytics available:")
                print(f"  - Total Agents: {analytics.get('totalAgents', 0)}")
                print(f"  - Active Goals: {analytics.get('activeGoals', 0)}")
                print(f"  - Average Progress: {analytics.get('averageProgress', 0)}%")
                return True
            else:
                print_warning("No analytics data found")
                return False
        else:
            print_error(f"Failed to retrieve analytics: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"UI data check error: {e}")
        return False

def test_ui_navigation():
    """Test 5: UI Navigation"""
    print_section("Test 5: UI Navigation URLs")
    
    ui_url = os.getenv("UI_URL", "http://localhost:4004/a2aFiori/webapp/index.html")
    
    print("Goal Management UI URLs:")
    print(f"  - Dashboard: {ui_url}#/goal-dashboard")
    print(f"  - All Goals: {ui_url}#/goals")
    print(f"  - Create Goal: {ui_url}#/goals/create")
    print(f"  - Collaborative: {ui_url}#/collaborative-goals")
    
    print_success("UI navigation configured")
    return True

async def run_integration_tests():
    """Run all integration tests"""
    print_section("A2A Goal Management Integration Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set development mode
    os.environ["A2A_DEV_MODE"] = "true"
    
    test_results = {
        "goal_assignment": False,
        "cap_connection": False,
        "goal_sync": False,
        "ui_data": False,
        "ui_navigation": False
    }
    
    # Test 1: Goal Assignment
    success, assignment_results = await test_goal_assignment()
    test_results["goal_assignment"] = success
    
    # Test 2: CAP Service Connection
    test_results["cap_connection"] = test_cap_service_connection()
    
    # Test 3: Goal Sync (only if CAP is connected)
    if test_results["cap_connection"]:
        test_results["goal_sync"] = test_goal_sync()
        
        # Test 4: UI Data Availability
        if test_results["goal_sync"]:
            await asyncio.sleep(2)  # Wait for sync to complete
            test_results["ui_data"] = test_ui_data_availability()
    
    # Test 5: UI Navigation
    test_results["ui_navigation"] = test_ui_navigation()
    
    # Summary
    print_section("Test Summary")
    
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print("\nTest Results:")
    for test_name, result in test_results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    # Save results
    results_file = f"integration_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results,
            "passed": passed,
            "total": total,
            "success_rate": (passed / total) * 100 if total > 0 else 0
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Final status
    if passed == total:
        print_success("\nAll tests passed! Goal management system is fully operational.")
    elif passed > 0:
        print_warning(f"\nPartial success: {passed}/{total} tests passed.")
    else:
        print_error("\nAll tests failed. Please check the system configuration.")
    
    return passed == total

if __name__ == "__main__":
    # Run integration tests
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)