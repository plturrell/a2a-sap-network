#!/usr/bin/env python3
"""
Test real complexity analysis implementation
"""
import asyncio
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

async def test_real_complexity():
    """Test the real complexity analysis implementation"""
    print("Testing Real Complexity Analysis Implementation")
    print("=" * 60)
    
    try:
        # Import the agent
        from app.a2a.agents.gleanAgent import GleanAgent
        print("✓ Successfully imported GleanAgent")
        
        # Create agent instance
        agent = GleanAgent()
        print(f"✓ Created agent: {agent.agent_id}")
        
        # Create a test directory with complex Python code
        test_dir = tempfile.mkdtemp(prefix="glean_complexity_test_")
        print(f"\n1. Created test directory: {test_dir}")
        
        # Create Python file with various complexity levels
        complex_py = Path(test_dir) / "complex_code.py"
        complex_py.write_text('''
# Complex Python code for complexity analysis
import os
import sys
from typing import Dict, List, Optional

def simple_function(a, b):
    """Simple function with low complexity"""
    return a + b

def moderate_complexity(x, y, z):
    """Function with moderate complexity"""
    if x > 0:
        if y > 0:
            result = x + y
            if z > 0:
                result += z
            return result
        else:
            return x
    else:
        return 0

def high_complexity_function(data: List[Dict], threshold: int = 10):
    """Function with high cyclomatic complexity"""
    results = []
    
    for item in data:
        if not item:
            continue
            
        if 'type' in item:
            if item['type'] == 'A':
                if item.get('value', 0) > threshold:
                    try:
                        processed = item['value'] * 2
                        if processed > 100:
                            results.append({'processed': processed, 'category': 'high'})
                        elif processed > 50:
                            results.append({'processed': processed, 'category': 'medium'})
                        else:
                            results.append({'processed': processed, 'category': 'low'})
                    except KeyError as e:
                        print(f"Error processing item: {e}")
                        continue
                else:
                    results.append({'processed': item['value'], 'category': 'unchanged'})
            elif item['type'] == 'B':
                if item.get('special', False):
                    for i in range(item.get('count', 1)):
                        if i % 2 == 0:
                            results.append({'processed': i, 'category': 'even'})
                        else:
                            results.append({'processed': i, 'category': 'odd'})
                else:
                    results.append({'processed': 0, 'category': 'default'})
            else:
                # Handle other types
                pass
        
        # Additional nested conditions
        if 'metadata' in item:
            meta = item['metadata']
            if isinstance(meta, dict):
                if 'priority' in meta:
                    if meta['priority'] == 'high':
                        results[-1]['priority'] = 'urgent'
                    elif meta['priority'] == 'medium':
                        results[-1]['priority'] = 'normal'
                    else:
                        results[-1]['priority'] = 'low'
    
    return results

class ComplexClass:
    """Class with multiple methods of varying complexity"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data = []
    
    def simple_method(self):
        """Simple method"""
        return len(self.data)
    
    def complex_method(self, items):
        """Method with high complexity"""
        for item in items:
            if item:
                if isinstance(item, dict):
                    if 'process' in item and item['process']:
                        try:
                            if item.get('type') == 'special':
                                self._handle_special(item)
                            elif item.get('type') == 'normal':
                                self._handle_normal(item)
                            else:
                                self._handle_default(item)
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
                    else:
                        self.data.append(item)
                else:
                    self.data.append({'value': item})
    
    def _handle_special(self, item):
        # More nested logic
        if 'value' in item:
            if item['value'] > 100:
                return item['value'] * 2
            elif item['value'] > 50:
                return item['value'] * 1.5
            else:
                return item['value']
        return 0
    
    def _handle_normal(self, item):
        return item.get('value', 0)
    
    def _handle_default(self, item):
        return 42

async def async_complex_function(data):
    """Async function with complexity"""
    results = []
    for item in data:
        if item:
            try:
                if isinstance(item, dict):
                    if 'async_process' in item:
                        # Simulate async processing
                        processed = await some_async_operation(item)
                        if processed:
                            results.append(processed)
                    else:
                        results.append(item)
                else:
                    results.append({'value': item})
            except Exception:
                continue
    return results

async def some_async_operation(item):
    return item.get('value', 0) * 2
''')
        
        print("\n2. Testing real complexity analysis:")
        
        # Test the real complexity analysis
        complexity_result = await agent.analyze_code_complexity(test_dir, ["*.py"])
        
        print(f"   Files analyzed: {complexity_result.get('files_analyzed', 0)}")
        print(f"   Functions analyzed: {complexity_result.get('functions_analyzed', 0)}")
        print(f"   Classes analyzed: {complexity_result.get('classes_analyzed', 0)}")
        print(f"   Average complexity: {complexity_result.get('average_complexity', 0):.2f}")
        print(f"   Max complexity: {complexity_result.get('max_complexity', 0)}")
        print(f"   Duration: {complexity_result.get('duration', 0):.2f}s")
        
        high_complexity = complexity_result.get('high_complexity_functions', [])
        if high_complexity:
            print(f"\n   High complexity functions ({len(high_complexity)}):")
            for func in high_complexity[:3]:  # Show top 3
                print(f"     - {func['name']}: complexity {func['complexity']} (line {func['line']})")
        
        if complexity_result.get('complexity_distribution'):
            print(f"\n   Complexity distribution:")
            for range_key, count in complexity_result['complexity_distribution'].items():
                print(f"     - {range_key}: {count} functions")
        
        recommendations = complexity_result.get('recommendations', [])
        if recommendations:
            print(f"\n   Recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                print(f"     • {rec}")
        
        print("\n✅ Real complexity analysis test completed!")
        
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the async test
    success = asyncio.run(test_real_complexity())
    sys.exit(0 if success else 1)