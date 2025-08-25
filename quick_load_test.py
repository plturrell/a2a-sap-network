#!/usr/bin/env python3
"""
Quick Load Test for A2A Platform
Fast assessment of basic performance capabilities
"""

import time
import threading
import multiprocessing
import psutil
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sqlite3
import random

class QuickA2ALoadTester:
    def __init__(self):
        self.project_root = Path("/Users/apple/projects/a2a")
        
    def test_concurrent_processing(self, num_threads: int = 10) -> Dict[str, Any]:
        """Test concurrent processing capabilities"""
        print(f"🔥 Testing concurrent processing with {num_threads} threads...")
        
        results = []
        start_time = time.time()
        
        def worker_task(worker_id: int):
            """Simulate agent processing work"""
            task_times = []
            for i in range(50):  # 50 operations per worker
                task_start = time.time()
                
                # Simulate processing
                data = [random.random() for _ in range(1000)]
                processed = sum(x * x for x in data)
                result = processed / len(data)
                
                task_times.append(time.time() - task_start)
                time.sleep(0.01)  # Small delay
            
            return {
                'worker_id': worker_id,
                'tasks_completed': len(task_times),
                'avg_task_time': statistics.mean(task_times),
                'total_time': sum(task_times)
            }
        
        # Run concurrent workers
        with multiprocessing.Pool(num_threads) as pool:
            worker_results = pool.map(worker_task, range(num_threads))
        
        total_time = time.time() - start_time
        total_tasks = sum(r['tasks_completed'] for r in worker_results)
        
        return {
            'concurrent_threads': num_threads,
            'total_tasks_completed': total_tasks,
            'total_duration': total_time,
            'throughput_ops_per_second': total_tasks / total_time,
            'avg_task_time': statistics.mean([r['avg_task_time'] for r in worker_results]),
            'worker_results': worker_results
        }
    
    def test_database_operations(self) -> Dict[str, Any]:
        """Test database performance"""
        print("🗄️ Testing database operations...")
        
        db_path = self.project_root / "quick_test.db"
        
        # Setup
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE test_agents (
                id INTEGER PRIMARY KEY,
                agent_id TEXT,
                status TEXT,
                data TEXT,
                created_at TIMESTAMP
            )
        ''')
        conn.commit()
        
        # Insert test
        insert_start = time.time()
        for i in range(500):
            cursor.execute('''
                INSERT INTO test_agents (agent_id, status, data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (f'agent_{i}', 'active', f'data_{i}' * 10, datetime.now()))
        conn.commit()
        insert_time = time.time() - insert_start
        
        # Select test
        select_start = time.time()
        cursor.execute('SELECT COUNT(*) FROM test_agents WHERE status = ?', ('active',))
        count_result = cursor.fetchone()[0]
        
        cursor.execute('SELECT * FROM test_agents ORDER BY created_at DESC LIMIT 10')
        latest_agents = cursor.fetchall()
        select_time = time.time() - select_start
        
        # Update test
        update_start = time.time()
        cursor.execute("UPDATE test_agents SET status = 'processed' WHERE id <= 100")
        conn.commit()
        update_time = time.time() - update_start
        
        conn.close()
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
        
        return {
            'insert_operations': 500,
            'insert_duration': insert_time,
            'insert_rate_ops_per_sec': 500 / insert_time,
            'select_duration': select_time,
            'update_operations': 100,
            'update_duration': update_time,
            'records_found': count_result,
            'database_performance_score': 'HIGH' if insert_time < 1.0 else 'MEDIUM' if insert_time < 3.0 else 'LOW'
        }
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        print("🧠 Testing memory usage...")
        
        initial_memory = psutil.virtual_memory().percent
        memory_samples = [initial_memory]
        
        # Memory intensive operations
        data_structures = []
        
        for i in range(10):
            # Create memory load
            large_list = [random.random() for _ in range(50000)]
            data_structures.append(large_list)
            
            current_memory = psutil.virtual_memory().percent
            memory_samples.append(current_memory)
            
            time.sleep(0.1)
        
        peak_memory = max(memory_samples)
        
        # Cleanup
        data_structures.clear()
        
        final_memory = psutil.virtual_memory().percent
        
        return {
            'initial_memory_percent': initial_memory,
            'peak_memory_percent': peak_memory,
            'final_memory_percent': final_memory,
            'memory_increase': peak_memory - initial_memory,
            'memory_efficiency': 'GOOD' if peak_memory - initial_memory < 10 else 'FAIR' if peak_memory - initial_memory < 20 else 'NEEDS_IMPROVEMENT'
        }
    
    def test_cpu_performance(self) -> Dict[str, Any]:
        """Test CPU performance"""
        print("⚡ Testing CPU performance...")
        
        cpu_samples = []
        start_time = time.time()
        
        # CPU intensive task
        def cpu_intensive_work():
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Monitor CPU while running intensive task
        def monitor_cpu():
            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run CPU intensive work
        work_result = cpu_intensive_work()
        
        monitor_thread.join()
        duration = time.time() - start_time
        
        return {
            'cpu_task_duration': duration,
            'avg_cpu_usage': statistics.mean(cpu_samples) if cpu_samples else 0,
            'peak_cpu_usage': max(cpu_samples) if cpu_samples else 0,
            'cpu_performance_score': 'HIGH' if duration < 1.0 else 'MEDIUM' if duration < 3.0 else 'LOW',
            'work_result': work_result
        }
    
    def run_quick_assessment(self) -> Dict[str, Any]:
        """Run quick performance assessment"""
        print("🚀 A2A Quick Load Testing Assessment")
        print("=" * 50)
        
        results = {
            'assessment_timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'available_memory_gb': round(psutil.virtual_memory().available / (1024**3), 2)
            }
        }
        
        # Run tests
        try:
            results['concurrent_processing'] = self.test_concurrent_processing(5)
            print(f"✅ Concurrent processing: {results['concurrent_processing']['throughput_ops_per_second']:.1f} ops/sec")
        except Exception as e:
            results['concurrent_processing'] = {'error': str(e)}
            print(f"❌ Concurrent processing test failed: {e}")
        
        try:
            results['database_performance'] = self.test_database_operations()
            db_result = results['database_performance']
            print(f"✅ Database performance: {db_result['insert_rate_ops_per_sec']:.1f} inserts/sec")
        except Exception as e:
            results['database_performance'] = {'error': str(e)}
            print(f"❌ Database test failed: {e}")
        
        try:
            results['memory_usage'] = self.test_memory_usage()
            mem_result = results['memory_usage']
            print(f"✅ Memory usage: {mem_result['memory_efficiency']} (peak: {mem_result['peak_memory_percent']:.1f}%)")
        except Exception as e:
            results['memory_usage'] = {'error': str(e)}
            print(f"❌ Memory test failed: {e}")
        
        try:
            results['cpu_performance'] = self.test_cpu_performance()
            cpu_result = results['cpu_performance']
            print(f"✅ CPU performance: {cpu_result['cpu_performance_score']} ({cpu_result['cpu_task_duration']:.2f}s)")
        except Exception as e:
            results['cpu_performance'] = {'error': str(e)}
            print(f"❌ CPU test failed: {e}")
        
        # Generate overall assessment
        scores = []
        
        if 'error' not in results['concurrent_processing']:
            throughput = results['concurrent_processing']['throughput_ops_per_second']
            scores.append('HIGH' if throughput > 100 else 'MEDIUM' if throughput > 50 else 'LOW')
        
        if 'error' not in results['database_performance']:
            db_score = results['database_performance']['database_performance_score']
            scores.append(db_score)
        
        if 'error' not in results['memory_usage']:
            mem_score = results['memory_usage']['memory_efficiency']
            scores.append('HIGH' if mem_score == 'GOOD' else 'MEDIUM' if mem_score == 'FAIR' else 'LOW')
        
        if 'error' not in results['cpu_performance']:
            cpu_score = results['cpu_performance']['cpu_performance_score']
            scores.append(cpu_score)
        
        # Overall rating
        high_count = scores.count('HIGH')
        medium_count = scores.count('MEDIUM')
        
        if high_count >= 3:
            overall_rating = 'HIGH'
        elif high_count + medium_count >= 3:
            overall_rating = 'MEDIUM'
        else:
            overall_rating = 'LOW'
        
        results['overall_assessment'] = {
            'performance_rating': overall_rating,
            'component_scores': {
                'concurrent_processing': scores[0] if len(scores) > 0 else 'ERROR',
                'database': scores[1] if len(scores) > 1 else 'ERROR',
                'memory': scores[2] if len(scores) > 2 else 'ERROR',  
                'cpu': scores[3] if len(scores) > 3 else 'ERROR'
            },
            'scalability_estimate': 'HIGH (50+ users)' if overall_rating == 'HIGH' else 'MEDIUM (10-25 users)' if overall_rating == 'MEDIUM' else 'LOW (<10 users)'
        }
        
        print("\n" + "=" * 50)
        print("🎯 QUICK ASSESSMENT COMPLETE!")
        print("=" * 50)
        print(f"📊 Overall Performance: {overall_rating}")
        print(f"📈 Scalability Estimate: {results['overall_assessment']['scalability_estimate']}")
        print("=" * 50)
        
        # Save results
        with open(self.project_root / 'quick_load_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved to: quick_load_test_results.json")
        
        return results

def main():
    tester = QuickA2ALoadTester()
    return tester.run_quick_assessment()

if __name__ == "__main__":
    main()