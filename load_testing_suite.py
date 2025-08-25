#!/usr/bin/env python3
"""
A2A Platform Load Testing and Scalability Assessment Suite
Comprehensive testing for performance, throughput, and system stability
"""

import asyncio
import aiohttp
import time
import threading
import multiprocessing
import psutil
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import random
import string
import sqlite3
import requests

logger = logging.getLogger(__name__)

@dataclass
class LoadTestMetrics:
    """Load testing metrics collection"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    throughput_rps: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

@dataclass
class ScalabilityTestResult:
    """Scalability test results"""
    test_name: str
    concurrent_users: int
    metrics: LoadTestMetrics
    system_stability: str
    bottlenecks_identified: List[str]
    recommendations: List[str]

class A2ALoadTester:
    def __init__(self):
        self.project_root = Path("/Users/apple/projects/a2a")
        self.base_url = "http://localhost:8000"
        self.agent_endpoints = [
            "/api/agents/standardization",
            "/api/agents/analytics", 
            "/api/agents/coordinator",
            "/api/agents/processor",
            "/api/agents/manager"
        ]
        self.test_results = []
        
    async def simulate_agent_communication(self, session: aiohttp.ClientSession, 
                                         agent_id: str, duration: int) -> LoadTestMetrics:
        """Simulate agent-to-agent communication under load"""
        metrics = LoadTestMetrics()
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Simulate agent registration
                response_time_start = time.time()
                async with session.post(f"{self.base_url}/api/agents/register", 
                                      json={"agent_id": agent_id, "capabilities": ["process", "analyze"]}) as resp:
                    response_time = time.time() - response_time_start
                    metrics.response_times.append(response_time)
                    
                    if resp.status == 200:
                        metrics.successful_requests += 1
                    else:
                        metrics.failed_requests += 1
                        metrics.errors.append(f"Registration failed: {resp.status}")
                
                # Simulate data processing request
                response_time_start = time.time()
                test_data = {"data": "x" * random.randint(100, 1000), "timestamp": datetime.now().isoformat()}
                async with session.post(f"{self.base_url}/api/agents/process", 
                                      json=test_data) as resp:
                    response_time = time.time() - response_time_start
                    metrics.response_times.append(response_time)
                    
                    if resp.status == 200:
                        metrics.successful_requests += 1
                    else:
                        metrics.failed_requests += 1
                        metrics.errors.append(f"Processing failed: {resp.status}")
                
                # Simulate agent status check
                response_time_start = time.time()
                async with session.get(f"{self.base_url}/api/agents/status/{agent_id}") as resp:
                    response_time = time.time() - response_time_start
                    metrics.response_times.append(response_time)
                    
                    if resp.status == 200:
                        metrics.successful_requests += 1
                    else:
                        metrics.failed_requests += 1
                        metrics.errors.append(f"Status check failed: {resp.status}")
                
                metrics.total_requests += 3
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                metrics.failed_requests += 3
                metrics.errors.append(f"Connection error: {str(e)}")
                await asyncio.sleep(1)  # Backoff on error
        
        return metrics
    
    def monitor_system_resources(self, duration: int) -> Dict[str, List[float]]:
        """Monitor system resources during load test"""
        cpu_usage = []
        memory_usage = []
        disk_io = []
        network_io = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                cpu_usage.append(psutil.cpu_percent())
                memory_usage.append(psutil.virtual_memory().percent)
                
                # Disk I/O
                disk_stats = psutil.disk_io_counters()
                if disk_stats:
                    disk_io.append(disk_stats.read_bytes + disk_stats.write_bytes)
                
                # Network I/O
                net_stats = psutil.net_io_counters()
                if net_stats:
                    network_io.append(net_stats.bytes_sent + net_stats.bytes_recv)
                
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                continue
        
        return {
            "cpu_percent": cpu_usage,
            "memory_percent": memory_usage,
            "disk_io_bytes": disk_io,
            "network_io_bytes": network_io
        }
    
    async def run_concurrent_load_test(self, concurrent_users: int, duration: int) -> ScalabilityTestResult:
        """Run load test with specified concurrent users"""
        print(f"🔥 Starting load test: {concurrent_users} concurrent users for {duration}s")
        
        # Start system monitoring
        monitor_thread = threading.Thread(
            target=lambda: self.monitor_system_resources(duration + 5)
        )
        monitor_thread.start()
        
        # Create agent simulations
        tasks = []
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in range(concurrent_users):
                agent_id = f"load_test_agent_{i}"
                task = self.simulate_agent_communication(session, agent_id, duration)
                tasks.append(task)
            
            # Run all concurrent simulations
            start_time = datetime.now()
            individual_metrics = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = datetime.now()
        
        # Aggregate metrics
        combined_metrics = LoadTestMetrics()
        combined_metrics.start_time = start_time
        combined_metrics.end_time = end_time
        combined_metrics.duration_seconds = (end_time - start_time).total_seconds()
        
        for metrics in individual_metrics:
            if isinstance(metrics, LoadTestMetrics):
                combined_metrics.total_requests += metrics.total_requests
                combined_metrics.successful_requests += metrics.successful_requests
                combined_metrics.failed_requests += metrics.failed_requests
                combined_metrics.response_times.extend(metrics.response_times)
                combined_metrics.errors.extend(metrics.errors)
        
        # Calculate performance metrics
        if combined_metrics.response_times:
            combined_metrics.avg_response_time = statistics.mean(combined_metrics.response_times)
            combined_metrics.p95_response_time = statistics.quantiles(combined_metrics.response_times, n=20)[18]
            combined_metrics.p99_response_time = statistics.quantiles(combined_metrics.response_times, n=100)[98]
        
        combined_metrics.throughput_rps = combined_metrics.successful_requests / combined_metrics.duration_seconds
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Analyze results
        system_stability = self.analyze_system_stability(combined_metrics)
        bottlenecks = self.identify_bottlenecks(combined_metrics)
        recommendations = self.generate_recommendations(combined_metrics, concurrent_users)
        
        return ScalabilityTestResult(
            test_name=f"Load Test - {concurrent_users} Users",
            concurrent_users=concurrent_users,
            metrics=combined_metrics,
            system_stability=system_stability,
            bottlenecks_identified=bottlenecks,
            recommendations=recommendations
        )
    
    def test_database_performance(self) -> Dict[str, Any]:
        """Test database performance under load"""
        print("🗄️ Testing database performance...")
        
        db_path = self.project_root / "test_load.db"
        
        # Create test database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS load_test (
                id INTEGER PRIMARY KEY,
                agent_id TEXT,
                data TEXT,
                timestamp DATETIME,
                processed BOOLEAN
            )
        ''')
        conn.commit()
        
        # Performance testing
        start_time = time.time()
        insert_times = []
        select_times = []
        
        # Insert performance
        for i in range(1000):
            insert_start = time.time()
            cursor.execute('''
                INSERT INTO load_test (agent_id, data, timestamp, processed)
                VALUES (?, ?, ?, ?)
            ''', (f"agent_{i}", f"test_data_{i}" * 100, datetime.now(), False))
            insert_times.append(time.time() - insert_start)
        
        conn.commit()
        
        # Select performance
        for i in range(100):
            select_start = time.time()
            cursor.execute('SELECT * FROM load_test WHERE agent_id = ?', (f"agent_{i}",))
            cursor.fetchall()
            select_times.append(time.time() - select_start)
        
        total_time = time.time() - start_time
        
        conn.close()
        db_path.unlink()  # Clean up
        
        return {
            "total_duration": total_time,
            "avg_insert_time": statistics.mean(insert_times),
            "avg_select_time": statistics.mean(select_times),
            "insert_throughput": len(insert_times) / sum(insert_times),
            "select_throughput": len(select_times) / sum(select_times)
        }
    
    def stress_test_memory_limits(self) -> Dict[str, Any]:
        """Stress test memory usage limits"""
        print("🧠 Testing memory limits...")
        
        initial_memory = psutil.virtual_memory().percent
        peak_memory = initial_memory
        
        # Simulate memory-intensive operations
        large_data_sets = []
        
        try:
            for i in range(100):
                # Create large data structures
                data = ['x' * 10000 for _ in range(1000)]
                large_data_sets.append(data)
                
                current_memory = psutil.virtual_memory().percent
                if current_memory > peak_memory:
                    peak_memory = current_memory
                
                # Stop if memory usage gets too high
                if current_memory > 85:
                    break
                    
        except MemoryError:
            return {
                "status": "Memory limit reached",
                "initial_memory_percent": initial_memory,
                "peak_memory_percent": peak_memory,
                "memory_stress_successful": False
            }
        
        # Clean up
        large_data_sets.clear()
        
        return {
            "status": "Memory stress test completed",
            "initial_memory_percent": initial_memory,
            "peak_memory_percent": peak_memory,
            "memory_increase_percent": peak_memory - initial_memory,
            "memory_stress_successful": True
        }
    
    def analyze_system_stability(self, metrics: LoadTestMetrics) -> str:
        """Analyze system stability based on metrics"""
        error_rate = metrics.failed_requests / max(metrics.total_requests, 1)
        
        if error_rate < 0.01:  # Less than 1% error rate
            return "EXCELLENT"
        elif error_rate < 0.05:  # Less than 5% error rate
            return "GOOD"
        elif error_rate < 0.10:  # Less than 10% error rate
            return "FAIR"
        else:
            return "POOR"
    
    def identify_bottlenecks(self, metrics: LoadTestMetrics) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if metrics.avg_response_time > 2.0:
            bottlenecks.append("High average response time (>2s)")
        
        if metrics.p95_response_time > 5.0:
            bottlenecks.append("High P95 response time (>5s)")
        
        if metrics.throughput_rps < 10:
            bottlenecks.append("Low throughput (<10 RPS)")
        
        error_rate = metrics.failed_requests / max(metrics.total_requests, 1)
        if error_rate > 0.05:
            bottlenecks.append(f"High error rate ({error_rate:.2%})")
        
        return bottlenecks
    
    def generate_recommendations(self, metrics: LoadTestMetrics, concurrent_users: int) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if metrics.avg_response_time > 1.0:
            recommendations.append("Consider implementing response caching")
            recommendations.append("Optimize database queries and add indexing")
        
        if metrics.throughput_rps < concurrent_users * 0.5:
            recommendations.append("Increase server resources (CPU/Memory)")
            recommendations.append("Implement connection pooling")
        
        error_rate = metrics.failed_requests / max(metrics.total_requests, 1)
        if error_rate > 0.02:
            recommendations.append("Implement better error handling and retries")
            recommendations.append("Add circuit breaker patterns")
        
        if concurrent_users > 50 and metrics.p99_response_time > 10.0:
            recommendations.append("Consider implementing horizontal scaling")
            recommendations.append("Add load balancing")
        
        return recommendations
    
    def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load testing suite"""
        print("🚀 Starting comprehensive A2A load testing suite...")
        print("=" * 60)
        
        results = {
            "test_timestamp": datetime.now().isoformat(),
            "scalability_tests": [],
            "database_performance": {},
            "memory_stress_test": {},
            "overall_assessment": {}
        }
        
        # Test different load levels
        load_levels = [1, 5, 10, 25, 50]
        duration = 30  # 30 seconds per test
        
        for users in load_levels:
            try:
                result = asyncio.run(self.run_concurrent_load_test(users, duration))
                results["scalability_tests"].append({
                    "concurrent_users": users,
                    "total_requests": result.metrics.total_requests,
                    "successful_requests": result.metrics.successful_requests,
                    "failed_requests": result.metrics.failed_requests,
                    "throughput_rps": result.metrics.throughput_rps,
                    "avg_response_time": result.metrics.avg_response_time,
                    "p95_response_time": result.metrics.p95_response_time,
                    "p99_response_time": result.metrics.p99_response_time,
                    "system_stability": result.system_stability,
                    "bottlenecks": result.bottlenecks_identified,
                    "recommendations": result.recommendations
                })
                
                print(f"✅ Completed {users} user test - Throughput: {result.metrics.throughput_rps:.1f} RPS")
                
            except Exception as e:
                print(f"❌ Failed {users} user test: {e}")
                results["scalability_tests"].append({
                    "concurrent_users": users,
                    "error": str(e)
                })
        
        # Database performance test
        try:
            db_results = self.test_database_performance()
            results["database_performance"] = db_results
            print(f"✅ Database test - Insert: {db_results['insert_throughput']:.1f} ops/s")
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            results["database_performance"] = {"error": str(e)}
        
        # Memory stress test
        try:
            memory_results = self.stress_test_memory_limits()
            results["memory_stress_test"] = memory_results
            print(f"✅ Memory test - Peak usage: {memory_results['peak_memory_percent']:.1f}%")
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            results["memory_stress_test"] = {"error": str(e)}
        
        # Overall assessment
        successful_tests = [t for t in results["scalability_tests"] if "error" not in t]
        if successful_tests:
            max_stable_users = max(t["concurrent_users"] for t in successful_tests 
                                 if t.get("system_stability") in ["EXCELLENT", "GOOD"])
            max_throughput = max(t["throughput_rps"] for t in successful_tests)
            
            results["overall_assessment"] = {
                "max_stable_concurrent_users": max_stable_users,
                "peak_throughput_rps": max_throughput,
                "scalability_rating": "HIGH" if max_stable_users >= 25 else "MEDIUM" if max_stable_users >= 10 else "LOW",
                "performance_rating": "HIGH" if max_throughput >= 50 else "MEDIUM" if max_throughput >= 20 else "LOW"
            }
        
        print("\n" + "=" * 60)
        print("🎯 LOAD TESTING COMPLETE!")
        print("=" * 60)
        
        if "max_stable_concurrent_users" in results["overall_assessment"]:
            assessment = results["overall_assessment"]
            print(f"📊 Max Stable Users: {assessment['max_stable_concurrent_users']}")
            print(f"🚀 Peak Throughput: {assessment['peak_throughput_rps']:.1f} RPS")
            print(f"📈 Scalability: {assessment['scalability_rating']}")
            print(f"⚡ Performance: {assessment['performance_rating']}")
        
        # Save results
        with open(self.project_root / 'load_testing_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Full results saved to: load_testing_results.json")
        print("=" * 60)
        
        return results

def main():
    tester = A2ALoadTester()
    return tester.run_comprehensive_load_test()

if __name__ == "__main__":
    main()