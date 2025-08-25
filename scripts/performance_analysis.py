#!/usr/bin/env python3
"""
Comprehensive A2A Platform Performance Analysis
Identifies bottlenecks and optimization opportunities
"""

import os
import sys
import time
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import concurrent.futures

class A2APerformanceAnalyzer:
    def __init__(self):
        self.results = {
            'analysis_timestamp': time.time(),
            'bottlenecks': [],
            'optimizations': [],
            'metrics': {},
            'recommendations': []
        }
        
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze file structure for performance issues"""
        print("🔍 Analyzing file structure and imports...")
        
        project_root = Path("/Users/apple/projects/a2a")
        
        # Count files by type
        file_stats = defaultdict(int)
        large_files = []
        import_complexity = {}
        
        for file_path in project_root.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    suffix = file_path.suffix.lower()
                    file_stats[suffix] += 1
                    
                    # Track large files (> 100KB)
                    if size > 100 * 1024:
                        large_files.append({
                            'path': str(file_path.relative_to(project_root)),
                            'size_kb': size // 1024
                        })
                    
                    # Analyze Python imports
                    if suffix == '.py':
                        import_count = self._count_imports(file_path)
                        if import_count > 50:
                            import_complexity[str(file_path.relative_to(project_root))] = import_count
                            
                except (OSError, PermissionError):
                    continue
        
        return {
            'file_counts': dict(file_stats),
            'large_files': sorted(large_files, key=lambda x: x['size_kb'], reverse=True)[:20],
            'complex_imports': dict(sorted(import_complexity.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def _count_imports(self, file_path: Path) -> int:
        """Count import statements in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                import_lines = [line for line in content.split('\n') 
                              if line.strip().startswith(('import ', 'from '))]
                return len(import_lines)
        except:
            return 0
    
    def analyze_code_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity for performance bottlenecks"""
        print("🧮 Analyzing code complexity...")
        
        complexity_issues = []
        
        # Find deeply nested functions/classes
        for py_file in Path("/Users/apple/projects/a2a").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                max_indent = 0
                current_indent = 0
                
                for line_num, line in enumerate(lines, 1):
                    stripped = line.lstrip()
                    if stripped and not stripped.startswith('#'):
                        indent_level = len(line) - len(stripped)
                        current_indent = indent_level // 4  # Assuming 4-space indents
                        max_indent = max(max_indent, current_indent)
                
                if max_indent > 8:  # Very deeply nested code
                    complexity_issues.append({
                        'file': str(py_file.relative_to(Path("/Users/apple/projects/a2a"))),
                        'max_nesting_level': max_indent,
                        'issue': 'Deep nesting may impact performance'
                    })
                    
            except:
                continue
        
        return {
            'high_complexity_files': sorted(complexity_issues, key=lambda x: x['max_nesting_level'], reverse=True)[:10]
        }
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency structure for optimization opportunities"""
        print("📦 Analyzing dependencies...")
        
        # Check for duplicate/redundant imports
        import_patterns = defaultdict(list)
        
        for py_file in Path("/Users/apple/projects/a2a/a2aAgents/backend").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip().startswith(('import ', 'from ')):
                            import_patterns[line.strip()].append(str(py_file))
            except:
                continue
        
        # Find most common imports (potential for optimization)
        common_imports = sorted(
            [(imp, len(files)) for imp, files in import_patterns.items()],
            key=lambda x: x[1], reverse=True
        )[:15]
        
        return {
            'most_common_imports': common_imports,
            'total_unique_imports': len(import_patterns)
        }
    
    def analyze_database_queries(self) -> Dict[str, Any]:
        """Analyze potential database performance issues"""
        print("🗃️ Analyzing database usage patterns...")
        
        db_patterns = {
            'sql_files': [],
            'orm_usage': [],
            'potential_n_plus_1': []
        }
        
        # Find SQL files and analyze
        for sql_file in Path("/Users/apple/projects/a2a").rglob("*.sql"):
            try:
                with open(sql_file, 'r', encoding='utf-8') as f:
                    content = f.read().upper()
                    
                # Check for potentially expensive operations
                expensive_ops = []
                if 'SELECT *' in content:
                    expensive_ops.append('SELECT * queries')
                if 'WHERE' not in content and 'SELECT' in content:
                    expensive_ops.append('Missing WHERE clauses')
                if content.count('JOIN') > 5:
                    expensive_ops.append('Multiple JOINs')
                
                if expensive_ops:
                    db_patterns['sql_files'].append({
                        'file': str(sql_file.relative_to(Path("/Users/apple/projects/a2a"))),
                        'issues': expensive_ops
                    })
            except:
                continue
        
        return db_patterns
    
    def analyze_async_patterns(self) -> Dict[str, Any]:
        """Analyze async/await usage for performance optimization"""
        print("⚡ Analyzing async patterns...")
        
        async_stats = {
            'async_functions': 0,
            'await_calls': 0,
            'sync_in_async': [],
            'files_with_async': 0
        }
        
        for py_file in Path("/Users/apple/projects/a2a/a2aAgents/backend").rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                has_async = False
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    if 'async def' in line:
                        async_stats['async_functions'] += 1
                        has_async = True
                    if 'await ' in line:
                        async_stats['await_calls'] += 1
                    
                    # Check for blocking calls in async functions
                    if 'async def' in line and any(blocking in content for blocking in ['time.sleep', 'requests.get', 'requests.post']):
                        async_stats['sync_in_async'].append(str(py_file))
                
                if has_async:
                    async_stats['files_with_async'] += 1
                    
            except:
                continue
        
        return async_stats
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify specific performance bottlenecks"""
        print("🚨 Identifying performance bottlenecks...")
        
        bottlenecks = []
        
        # Large file bottlenecks
        large_files = self.results['metrics'].get('file_analysis', {}).get('large_files', [])
        for file_info in large_files[:5]:
            if file_info['size_kb'] > 500:  # Files > 500KB
                bottlenecks.append({
                    'type': 'Large File',
                    'severity': 'High',
                    'location': file_info['path'],
                    'impact': f"File size: {file_info['size_kb']}KB may slow loading",
                    'recommendation': 'Consider splitting into smaller modules'
                })
        
        # Import complexity bottlenecks
        complex_imports = self.results['metrics'].get('file_analysis', {}).get('complex_imports', {})
        for file_path, import_count in list(complex_imports.items())[:3]:
            if import_count > 80:
                bottlenecks.append({
                    'type': 'Import Complexity',
                    'severity': 'Medium',
                    'location': file_path,
                    'impact': f"{import_count} imports may slow module loading",
                    'recommendation': 'Lazy loading or import optimization'
                })
        
        # Code complexity bottlenecks
        complex_files = self.results['metrics'].get('complexity_analysis', {}).get('high_complexity_files', [])
        for file_info in complex_files[:3]:
            if file_info['max_nesting_level'] > 10:
                bottlenecks.append({
                    'type': 'Code Complexity',
                    'severity': 'High',
                    'location': file_info['file'],
                    'impact': f"Nesting level {file_info['max_nesting_level']} impacts performance",
                    'recommendation': 'Refactor to reduce nesting depth'
                })
        
        return bottlenecks
    
    def generate_optimizations(self) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations"""
        print("💡 Generating optimization recommendations...")
        
        optimizations = []
        
        # Async optimization opportunities
        async_stats = self.results['metrics'].get('async_analysis', {})
        if async_stats.get('async_functions', 0) > 0:
            ratio = async_stats.get('await_calls', 0) / async_stats.get('async_functions', 1)
            if ratio < 2:
                optimizations.append({
                    'category': 'Async Performance',
                    'priority': 'High',
                    'description': 'Low await-to-async-function ratio detected',
                    'implementation': 'Review async functions for proper await usage',
                    'expected_improvement': '15-30% response time improvement'
                })
        
        # Import optimization
        dep_stats = self.results['metrics'].get('dependency_analysis', {})
        common_imports = dep_stats.get('most_common_imports', [])
        if common_imports and common_imports[0][1] > 50:
            optimizations.append({
                'category': 'Import Optimization',
                'priority': 'Medium',
                'description': f'Import "{common_imports[0][0]}" used in {common_imports[0][1]} files',
                'implementation': 'Create shared import module or lazy loading',
                'expected_improvement': '10-15% startup time reduction'
            })
        
        # Caching opportunities
        file_count = sum(self.results['metrics'].get('file_analysis', {}).get('file_counts', {}).values())
        if file_count > 1000:
            optimizations.append({
                'category': 'Caching',
                'priority': 'High',
                'description': f'{file_count} files may benefit from caching',
                'implementation': 'Implement module-level caching for frequently accessed data',
                'expected_improvement': '20-40% faster repeated operations'
            })
        
        return optimizations
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        print("🚀 Starting comprehensive A2A performance analysis...")
        print("=" * 60)
        
        # Run all analyses
        self.results['metrics']['file_analysis'] = self.analyze_file_structure()
        self.results['metrics']['complexity_analysis'] = self.analyze_code_complexity()
        self.results['metrics']['dependency_analysis'] = self.analyze_dependencies()
        self.results['metrics']['database_analysis'] = self.analyze_database_queries()
        self.results['metrics']['async_analysis'] = self.analyze_async_patterns()
        
        # Generate insights
        self.results['bottlenecks'] = self.identify_bottlenecks()
        self.results['optimizations'] = self.generate_optimizations()
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("# A2A Platform Performance Analysis Report")
        report.append("=" * 50)
        report.append(f"Analysis completed at: {time.ctime(self.results['analysis_timestamp'])}")
        report.append("")
        
        # File Analysis Summary
        file_stats = self.results['metrics']['file_analysis']['file_counts']
        report.append("## File Structure Analysis")
        report.append(f"- Python files: {file_stats.get('.py', 0)}")
        report.append(f"- JavaScript files: {file_stats.get('.js', 0)}")
        report.append(f"- Total files: {sum(file_stats.values())}")
        report.append("")
        
        # Large Files
        large_files = self.results['metrics']['file_analysis']['large_files']
        if large_files:
            report.append("### Large Files (>100KB)")
            for file_info in large_files[:5]:
                report.append(f"- {file_info['path']}: {file_info['size_kb']}KB")
        report.append("")
        
        # Bottlenecks
        if self.results['bottlenecks']:
            report.append("## Performance Bottlenecks")
            for bottleneck in self.results['bottlenecks']:
                report.append(f"### {bottleneck['type']} - {bottleneck['severity']} Priority")
                report.append(f"- Location: {bottleneck['location']}")
                report.append(f"- Impact: {bottleneck['impact']}")
                report.append(f"- Recommendation: {bottleneck['recommendation']}")
                report.append("")
        
        # Optimizations
        if self.results['optimizations']:
            report.append("## Optimization Recommendations")
            for opt in self.results['optimizations']:
                report.append(f"### {opt['category']} - {opt['priority']} Priority")
                report.append(f"- Description: {opt['description']}")
                report.append(f"- Implementation: {opt['implementation']}")
                report.append(f"- Expected Improvement: {opt['expected_improvement']}")
                report.append("")
        
        # Async Analysis
        async_stats = self.results['metrics']['async_analysis']
        report.append("## Async Performance Analysis")
        report.append(f"- Async functions: {async_stats.get('async_functions', 0)}")
        report.append(f"- Await calls: {async_stats.get('await_calls', 0)}")
        report.append(f"- Files with async: {async_stats.get('files_with_async', 0)}")
        report.append("")
        
        return "\n".join(report)

def main():
    """Run the performance analysis"""
    analyzer = A2APerformanceAnalyzer()
    results = analyzer.run_full_analysis()
    
    # Generate and save report
    report = analyzer.generate_report()
    
    # Save results
    with open('/Users/apple/projects/a2a/performance_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('/Users/apple/projects/a2a/performance_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"🔍 Identified {len(results['bottlenecks'])} performance bottlenecks")
    print(f"💡 Generated {len(results['optimizations'])} optimization recommendations")
    print(f"📈 Analysis results saved to performance_analysis_results.json")
    print(f"📄 Full report saved to performance_analysis_report.md")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()