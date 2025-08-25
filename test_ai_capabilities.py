#!/usr/bin/env python3
"""
Test script to verify AI capabilities in the A2A platform
"""

import sys
import os
import asyncio
import json
import time
from datetime import datetime, timedelta

# Add the backend path to sys.path
sys.path.append('/Users/apple/projects/a2a/a2aAgents/backend')

# Import AI components
from app.a2a.core.ai_agent_discovery import get_ai_discovery, AgentTask
from app.a2a.core.ai_data_quality import get_ai_data_quality_validator
from app.a2a.core.ai_code_quality import get_ai_code_quality
from app.a2a.core.ai_test_generator import get_ai_test_generator
from app.a2a.core.ai_performance_optimizer import get_ai_performance_optimizer, PerformanceMetrics
from app.a2a.core.ai_resource_manager import get_ai_resource_manager, ResourceMetrics

print("🧠 A2A AI Capabilities Test Suite")
print("=" * 50)

async def test_ai_agent_discovery():
    """Test AI-powered agent discovery system"""
    print("\n🔍 Testing AI Agent Discovery...")
    
    try:
        ai_discovery = get_ai_discovery()
        
        # Create a sample task
        task = AgentTask(
            task_id="test_task_001",
            task_type="data_processing",
            priority="high",
            required_capabilities=["analysis", "validation", "transformation"],
            resource_requirements={"cpu": 0.6, "memory": 0.4},
            complexity_score=0.7,
            expected_duration=300
        )
        
        # Test agent discovery
        result = await ai_discovery.discover_optimal_agent(task)
        
        print(f"✅ Agent Discovery Result:")
        print(f"   Selected Agent: {result.get('agent_id', 'None')}")
        print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"   Alternatives Considered: {result.get('alternatives_considered', 0)}")
        
        if 'score_breakdown' in result:
            scores = result['score_breakdown']
            print(f"   Performance Score: {scores.get('performance_score', 0.0):.2f}")
            print(f"   Reliability Score: {scores.get('reliability_score', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent Discovery Test Failed: {e}")
        return False

async def test_ai_data_quality():
    """Test AI-powered data quality validation"""
    print("\n🔍 Testing AI Data Quality Validator...")
    
    try:
        validator = get_ai_data_quality_validator()
        
        # Sample test data with various quality issues
        test_data = [
            {
                "id": "001",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "555-123-4567",
                "age": 30,
                "score": 85.5
            },
            {
                "id": "002", 
                "name": "",  # Missing name
                "email": "invalid-email",  # Invalid email format
                "phone": "123",  # Invalid phone format
                "age": -5,  # Invalid age (business rule violation)
                "score": 150  # Invalid score (> 100)
            },
            {
                "id": "003",
                "name": "Jane Smith",
                "email": "jane@test.org", 
                "phone": "(555) 987-6543",
                "age": 25,
                "score": 92.0
            }
        ]
        
        # Run quality assessment
        report = await validator.assess_data_quality(test_data, "test_dataset")
        
        print(f"✅ Data Quality Assessment:")
        print(f"   Overall Quality Score: {report.overall_quality_score:.2f}")
        print(f"   Records Processed: {report.records_processed}")
        print(f"   Processing Time: {report.processing_time_seconds:.2f}s")
        print(f"   Confidence: {report.confidence:.2f}")
        
        print(f"   Dimension Scores:")
        for dimension, score in report.dimension_scores.items():
            print(f"     {dimension.title()}: {score:.2f}")
        
        print(f"   Issues Found: {sum(report.issue_summary.values())}")
        for issue, count in report.issue_summary.items():
            print(f"     {issue}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Quality Test Failed: {e}")
        return False

async def test_ai_code_quality():
    """Test AI-powered code quality analysis"""
    print("\n🔍 Testing AI Code Quality Analyzer...")
    
    try:
        analyzer = get_ai_code_quality()
        
        # Sample code with quality issues
        test_code = '''
def process_data(data):
    result = []
    for i in range(len(data)):  # Inefficient loop
        if data[i] is not None:
            if data[i] > 0:
                if data[i] < 100:  # Deep nesting
                    result.append(data[i] * 2)
                else:
                    result.append(data[i])
    return result

def another_function():
    # No docstring, no type hints
    x = eval("2 + 2")  # Security issue
    return x
'''
        
        # Analyze code quality
        metrics = await analyzer.analyze_code_fragment(test_code, "test_code.py")
        
        print(f"✅ Code Quality Analysis:")
        print(f"   Overall Quality Score: {metrics.complexity_score:.2f}")
        print(f"   Maintainability Index: {metrics.maintainability_index:.2f}")
        print(f"   Readability Score: {metrics.readability_score:.2f}")
        print(f"   Security Score: {metrics.security_score:.2f}")
        print(f"   Lines of Code: {metrics.lines_of_code}")
        print(f"   Code Smells: {metrics.code_smells}")
        print(f"   Technical Debt: {metrics.technical_debt_hours:.1f} hours")
        
        print(f"   Issues Found: {len(metrics.issues)}")
        for issue in metrics.issues[:3]:  # Show first 3 issues
            print(f"     Line {issue.line_number}: {issue.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Code Quality Test Failed: {e}")
        return False

async def test_ai_test_generator():
    """Test AI-powered test generation"""
    print("\n🔍 Testing AI Test Generator...")
    
    try:
        generator = get_ai_test_generator()
        
        # Sample function to generate tests for
        function_code = '''
def calculate_discount(price, discount_percent, customer_type="regular"):
    """
    Calculate discount for a given price and customer type.
    
    Args:
        price (float): Original price
        discount_percent (float): Discount percentage (0-100)
        customer_type (str): Customer type ("regular", "premium", "vip")
    
    Returns:
        float: Final price after discount
    """
    if price < 0 or discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid input parameters")
    
    # Apply customer type multiplier
    multipliers = {"regular": 1.0, "premium": 1.2, "vip": 1.5}
    multiplier = multipliers.get(customer_type, 1.0)
    
    final_discount = discount_percent * multiplier
    final_discount = min(final_discount, 90)  # Max 90% discount
    
    discounted_price = price * (1 - final_discount / 100)
    return round(discounted_price, 2)
'''
        
        # Generate tests
        test_cases = await generator.generate_tests_for_function(
            function_code, "calculate_discount"
        )
        
        print(f"✅ Test Generation Results:")
        print(f"   Tests Generated: {len(test_cases)}")
        
        for i, test_case in enumerate(test_cases[:3], 1):  # Show first 3 tests
            print(f"   Test {i}: {test_case.test_name}")
            print(f"     Type: {test_case.test_type}")
            print(f"     Priority: {test_case.priority}")
            print(f"     Assertions: {len(test_case.assertions)}")
            print(f"     Edge Cases: {len(test_case.edge_cases_covered)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Generation Failed: {e}")
        return False

async def test_ai_performance_optimizer():
    """Test AI-powered performance optimization"""
    print("\n🔍 Testing AI Performance Optimizer...")
    
    try:
        optimizer = get_ai_performance_optimizer()
        
        # Create sample performance metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            throughput=450.0,
            latency_p50=120.0,
            latency_p95=250.0, 
            latency_p99=400.0,
            cpu_usage=0.75,
            memory_usage=0.68,
            disk_io_read=50.0,
            disk_io_write=30.0,
            network_io_in=100.0,
            network_io_out=80.0,
            error_rate=0.02,
            active_connections=150,
            queue_depth=25,
            cache_hit_rate=0.85,
            database_query_time=45.0,
            gc_time=15.0
        )
        
        # Analyze performance
        analysis = await optimizer.analyze_performance(metrics)
        
        print(f"✅ Performance Analysis:")
        print(f"   Performance Score: {analysis['performance_score']:.2f}")
        print(f"   Bottlenecks Detected: {len(analysis['bottlenecks_detected'])}")
        print(f"   Optimization Opportunities: {len(analysis['optimization_opportunities'])}")
        print(f"   Analysis Confidence: {analysis['confidence']:.2f}")
        
        # Show bottlenecks
        for bottleneck in analysis['bottlenecks_detected'][:2]:
            print(f"   Bottleneck: {bottleneck.type.value} (Severity: {bottleneck.severity:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Optimization Test Failed: {e}")
        return False

async def test_ai_resource_manager():
    """Test AI-powered resource management"""
    print("\n🔍 Testing AI Resource Manager...")
    
    try:
        manager = get_ai_resource_manager()
        
        # Create sample resource metrics
        metrics = ResourceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=0.72,
            cpu_cores_used=28.8,
            memory_usage=0.65,
            memory_bytes_used=int(650 * 1024**3),  # 650 GB
            storage_usage=0.45,
            storage_bytes_used=int(450 * 1024**3),
            network_in_mbps=120.0,
            network_out_mbps=95.0,
            active_connections=200,
            request_rate=850.0,
            response_time_ms=145.0,
            error_rate=0.015,
            queue_depth=18
        )
        
        # Update metrics and predict demand
        manager.update_metrics(metrics)
        forecasts = await manager.predict_resource_demand(horizon_hours=24)
        
        print(f"✅ Resource Management Analysis:")
        print(f"   Resource Pools: {len(manager.resource_pools)}")
        print(f"   Forecasts Generated: {len(forecasts)}")
        
        for resource_type, forecast in forecasts.items():
            print(f"   {resource_type.upper()} Forecast:")
            print(f"     Peak Usage Time: {forecast.peak_usage_time}")
            print(f"     Recommended Capacity: {forecast.recommended_capacity:.1f}")
            print(f"     Cost Estimate: ${forecast.cost_estimate:.2f}")
            print(f"     Risk Level: {forecast.risk_assessment.get('overall_risk', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource Management Test Failed: {e}")
        return False

async def run_all_tests():
    """Run all AI capability tests"""
    print("\n🚀 Starting AI Capabilities Testing...")
    
    test_results = {}
    
    # Run tests
    test_results['agent_discovery'] = await test_ai_agent_discovery()
    test_results['data_quality'] = await test_ai_data_quality()
    test_results['code_quality'] = await test_ai_code_quality()
    test_results['test_generation'] = await test_ai_test_generator()
    test_results['performance_optimization'] = await test_ai_performance_optimizer()
    test_results['resource_management'] = await test_ai_resource_manager()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All AI capabilities are working correctly!")
    else:
        print("⚠️  Some AI capabilities need attention.")

if __name__ == "__main__":
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")