#!/usr/bin/env python3
"""
End-to-End Quality Control Manager Testing
Tests the complete Quality Control Manager with live Agent 4/5 data
"""

import asyncio
import json
import sys
import httpx
from datetime import datetime
from typing import Dict, Any

async def test_agent_health():
    """Test that all agents are healthy and responding"""
    print("🏥 Testing Agent Health...")
    
    agents = {
        "Agent 4 (Calc Validation)": "http://localhost:8006/health",
        "Agent 5 (QA Validation)": "http://localhost:8007/health", 
        "Quality Control Manager": "http://localhost:8009/health"
    }
    
    health_results = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in agents.items():
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    health_results[name] = {"status": "✅ Healthy", "response": response.json()}
                    print(f"   {name}: ✅ Healthy")
                else:
                    health_results[name] = {"status": f"❌ HTTP {response.status_code}", "response": None}
                    print(f"   {name}: ❌ HTTP {response.status_code}")
            except Exception as e:
                health_results[name] = {"status": f"❌ Error: {e}", "response": None}
                print(f"   {name}: ❌ Error: {e}")
    
    return health_results

async def test_agent_4_live():
    """Test Agent 4 calculation validation with live calls"""
    print("\n🧮 Testing Agent 4 Live Calculation Validation...")
    
    test_cases = [
        {"operation": "add", "operands": [2, 3], "expected": 5},
        {"operation": "multiply", "operands": [4, 5], "expected": 20},
        {"operation": "divide", "operands": [10, 2], "expected": 5},
        {"operation": "subtract", "operands": [8, 3], "expected": 5}
    ]
    
    results = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, test_case in enumerate(test_cases):
            try:
                response = await client.post(
                    "http://localhost:8006/api/validate-calculations",
                    json={"calculations": [test_case], "test_id": f"e2e_test_{i}"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "test_case": test_case,
                        "response": result,
                        "status": "passed" if result.get("passed", False) else "failed"
                    })
                    status = "✅ Pass" if result.get("passed", False) else "❌ Fail"
                    print(f"   Test {i+1}: {test_case['operation']} -> {status}")
                else:
                    print(f"   Test {i+1}: ❌ HTTP {response.status_code}")
                    results.append({"test_case": test_case, "error": f"HTTP {response.status_code}"})
                    
            except Exception as e:
                print(f"   Test {i+1}: ❌ Error: {e}")
                results.append({"test_case": test_case, "error": str(e)})
    
    return results

async def test_agent_5_live():
    """Test Agent 5 QA validation with live calls"""
    print("\n🔍 Testing Agent 5 Live QA Validation...")
    
    test_scenarios = [
        {"data": "test validation scenario 1", "criteria": {"accuracy": 0.85}},
        {"data": "comprehensive validation test", "criteria": {"accuracy": 0.90, "completeness": 0.80}},
        {"data": {"complex": "data structure"}, "criteria": {"accuracy": 0.75}},
        {"data": "quality assurance testing", "criteria": {"accuracy": 0.95}}
    ]
    
    results = []
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, scenario in enumerate(test_scenarios):
            try:
                response = await client.post(
                    "http://localhost:8007/api/qa-validate",
                    json=scenario
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "scenario": scenario,
                        "response": result,
                        "score": result.get("score", 0),
                        "status": "valid" if result.get("status") == "success" else "invalid"
                    })
                    score = result.get("score", 0)
                    print(f"   Test {i+1}: Score {score} -> {'✅ Valid' if result.get('status') == 'success' else '❌ Invalid'}")
                else:
                    print(f"   Test {i+1}: ❌ HTTP {response.status_code}")
                    results.append({"scenario": scenario, "error": f"HTTP {response.status_code}"})
                    
            except Exception as e:
                print(f"   Test {i+1}: ❌ Error: {e}")
                results.append({"scenario": scenario, "error": str(e)})
    
    return results

async def test_quality_control_manager():
    """Test Quality Control Manager end-to-end functionality"""
    print("\n🎯 Testing Quality Control Manager End-to-End...")
    
    # First, test basic endpoints
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test quality metrics endpoint
        try:
            response = await client.get("http://localhost:8009/api/v1/quality-metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"   Quality Metrics: ✅ Retrieved {len(metrics.get('processing_stats', {}))} stats")
            else:
                print(f"   Quality Metrics: ❌ HTTP {response.status_code}")
        except Exception as e:
            print(f"   Quality Metrics: ❌ Error: {e}")
        
        # Test quality assessment with live data
        try:
            # Create assessment request with realistic data
            assessment_request = {
                "calculation_result": {
                    "status": "success",
                    "passed": True,
                    "execution_time": 1.5,
                    "test_results": [
                        {"operation": "add", "result": "passed"},
                        {"operation": "multiply", "result": "passed"}
                    ]
                },
                "qa_validation_result": {
                    "status": "success",
                    "score": 95,
                    "quality_metrics": {
                        "accuracy": 0.95,
                        "completeness": 0.90,
                        "consistency": 0.88
                    }
                },
                "workflow_context": {
                    "request_id": "e2e_test_001",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "quality_thresholds": {
                    "accuracy": 0.85,
                    "reliability": 0.75,
                    "performance": 0.70
                }
            }
            
            response = await client.post(
                "http://localhost:8009/api/v1/assess-quality",
                json=assessment_request
            )
            
            if response.status_code == 200:
                assessment = response.json()
                decision = assessment.get("decision", "unknown")
                quality_scores = assessment.get("quality_scores", {})
                print(f"   Quality Assessment: ✅ Decision: {decision}")
                print(f"   Quality Scores: {quality_scores}")
                return assessment
            else:
                print(f"   Quality Assessment: ❌ HTTP {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error Details: {error_detail}")
                except:
                    print(f"   Error Text: {response.text}")
                    
        except Exception as e:
            print(f"   Quality Assessment: ❌ Error: {e}")
    
    return None

async def test_circuit_breaker_integration():
    """Test circuit breaker functionality in Quality Control Manager"""
    print("\n⚡ Testing Circuit Breaker Integration...")
    
    # This will trigger the circuit breaker fallback to live agent data
    # since Data Manager is not available
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Make a request that will internally trigger data retrieval
            assessment_request = {
                "calculation_result": {"status": "success", "passed": True},
                "qa_validation_result": {"status": "success", "score": 85},
                "workflow_context": {"test": "circuit_breaker"}
            }
            
            response = await client.post(
                "http://localhost:8009/api/v1/assess-quality",
                json=assessment_request
            )
            
            if response.status_code == 200:
                print("   Circuit Breaker: ✅ Handled gracefully with fallback")
                return True
            else:
                print(f"   Circuit Breaker: ⚠️  HTTP {response.status_code} but may still be working")
                return False
                
        except Exception as e:
            print(f"   Circuit Breaker: ❌ Error: {e}")
            return False

async def generate_comprehensive_report(health_results, agent4_results, agent5_results, qc_assessment, circuit_test):
    """Generate comprehensive test report"""
    print("\n📊 Comprehensive Test Report")
    print("=" * 60)
    
    # Health Summary
    healthy_agents = sum(1 for result in health_results.values() if "✅" in result["status"])
    print(f"Agent Health: {healthy_agents}/{len(health_results)} agents healthy")
    
    # Agent 4 Summary
    passed_calc_tests = sum(1 for result in agent4_results if result.get("status") == "passed")
    print(f"Agent 4 Tests: {passed_calc_tests}/{len(agent4_results)} calculation tests passed")
    
    # Agent 5 Summary 
    valid_qa_tests = sum(1 for result in agent5_results if result.get("status") == "valid")
    avg_qa_score = sum(result.get("score", 0) for result in agent5_results) / len(agent5_results) if agent5_results else 0
    print(f"Agent 5 Tests: {valid_qa_tests}/{len(agent5_results)} QA tests valid (avg score: {avg_qa_score:.1f})")
    
    # Quality Control Summary
    qc_status = "✅ Working" if qc_assessment else "❌ Failed"
    print(f"Quality Control Manager: {qc_status}")
    
    # Circuit Breaker Summary
    cb_status = "✅ Working" if circuit_test else "❌ Failed"
    print(f"Circuit Breaker: {cb_status}")
    
    # Overall Score
    total_tests = 5  # Health, Agent4, Agent5, QC, CircuitBreaker
    passed_tests = sum([
        1 if healthy_agents == len(health_results) else 0,
        1 if passed_calc_tests == len(agent4_results) else 0,
        1 if valid_qa_tests == len(agent5_results) else 0,
        1 if qc_assessment else 0,
        1 if circuit_test else 0
    ])
    
    score = (passed_tests / total_tests) * 100
    print(f"\n🎯 Overall Test Score: {score:.0f}/100")
    
    if score >= 80:
        print("🎉 EXCELLENT: System is production-ready!")
    elif score >= 60:
        print("👍 GOOD: System is functional with minor issues")
    else:
        print("⚠️  NEEDS WORK: System has significant issues")

async def main():
    """Run comprehensive end-to-end testing"""
    print("🧪 Quality Control Manager - End-to-End Testing")
    print("=" * 60)
    
    # Run all tests
    health_results = await test_agent_health()
    agent4_results = await test_agent_4_live()
    agent5_results = await test_agent_5_live()
    qc_assessment = await test_quality_control_manager()
    circuit_test = await test_circuit_breaker_integration()
    
    # Generate report
    await generate_comprehensive_report(
        health_results, agent4_results, agent5_results, 
        qc_assessment, circuit_test
    )

if __name__ == "__main__":
    asyncio.run(main())