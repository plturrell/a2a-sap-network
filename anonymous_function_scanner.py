#!/usr/bin/env python3
"""
Comprehensive Anonymous Function Scanner for A2A Codebase
Scans for all types of anonymous functions in JavaScript/TypeScript and Python files
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

class AnonymousFunctionScanner:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.findings = defaultdict(list)
        self.stats = defaultdict(int)
        
        # JavaScript/TypeScript patterns
        self.js_patterns = {
            'arrow_function': re.compile(r'(\w+\s*=\s*\([^)]*\)\s*=>\s*[{]|[^=]>\s*=>\s*[{]|\([^)]*\)\s*=>\s*[^{])', re.MULTILINE),
            'arrow_function_simple': re.compile(r'\w+\s*=>\s*', re.MULTILINE),
            'function_expression': re.compile(r'=\s*function\s*\([^)]*\)\s*{', re.MULTILINE),
            'anonymous_function': re.compile(r'(?:\.then|\.catch|\.finally|\.map|\.filter|\.reduce|\.forEach|\.find|\.some|\.every|setTimeout|setInterval|addEventListener|on|emit)\s*\(\s*function\s*\([^)]*\)', re.MULTILINE),
            'callback_arrow': re.compile(r'(?:\.then|\.catch|\.finally|\.map|\.filter|\.reduce|\.forEach|\.find|\.some|\.every|setTimeout|setInterval|addEventListener|on|emit)\s*\([^)]*=>', re.MULTILINE),
            'promise_constructor': re.compile(r'new\s+Promise\s*\(\s*(?:function|\([^)]*\)\s*=>)', re.MULTILINE),
            'event_handler': re.compile(r'(?:on\w+|addEventListener)\s*\([^,]+,\s*(?:function|\([^)]*\)\s*=>)', re.MULTILINE),
            'jquery_handler': re.compile(r'\$\([^)]+\)\.(?:\w+)\s*\(\s*(?:function|\([^)]*\)\s*=>)', re.MULTILINE),
            'express_route': re.compile(r'(?:app|router)\.(?:get|post|put|delete|use)\s*\([^,]+,\s*(?:function|\([^)]*\)\s*=>)', re.MULTILINE),
        }
        
        # Python patterns
        self.py_patterns = {
            'lambda_function': re.compile(r'lambda\s*[^:]*:', re.MULTILINE),
            'map_lambda': re.compile(r'map\s*\(\s*lambda', re.MULTILINE),
            'filter_lambda': re.compile(r'filter\s*\(\s*lambda', re.MULTILINE),
            'reduce_lambda': re.compile(r'reduce\s*\(\s*lambda', re.MULTILINE),
            'sorted_key': re.compile(r'sorted\s*\([^,]+,\s*key\s*=\s*lambda', re.MULTILINE),
            'defaultdict_factory': re.compile(r'defaultdict\s*\(\s*lambda', re.MULTILINE),
            'dataclass_factory': re.compile(r'default_factory\s*=\s*lambda', re.MULTILINE),
        }
        
    def scan_directory(self):
        """Scan the entire directory tree for anonymous functions"""
        for root, dirs, files in os.walk(self.root_path):
            # Skip node_modules and venv directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', 'venv', '.git', '__pycache__']]
            
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    self.scan_javascript_file(file_path)
                elif file.endswith('.py'):
                    self.scan_python_file(file_path)
                    
    def scan_javascript_file(self, file_path: str):
        """Scan a JavaScript/TypeScript file for anonymous functions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            relative_path = os.path.relpath(file_path, self.root_path)
            
            for pattern_name, pattern in self.js_patterns.items():
                matches = list(pattern.finditer(content))
                if matches:
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.findings[f'js_{pattern_name}'].append({
                            'file': relative_path,
                            'line': line_num,
                            'code': match.group(0).strip()[:100] + ('...' if len(match.group(0)) > 100 else '')
                        })
                        self.stats[f'js_{pattern_name}'] += 1
                        self.stats['total_js'] += 1
                        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
    def scan_python_file(self, file_path: str):
        """Scan a Python file for anonymous functions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            relative_path = os.path.relpath(file_path, self.root_path)
            
            for pattern_name, pattern in self.py_patterns.items():
                matches = list(pattern.finditer(content))
                if matches:
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        self.findings[f'py_{pattern_name}'].append({
                            'file': relative_path,
                            'line': line_num,
                            'code': match.group(0).strip()[:100] + ('...' if len(match.group(0)) > 100 else '')
                        })
                        self.stats[f'py_{pattern_name}'] += 1
                        self.stats['total_py'] += 1
                        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
    def generate_report(self):
        """Generate a comprehensive report of findings"""
        print("=" * 80)
        print("ANONYMOUS FUNCTION SCAN REPORT - A2A CODEBASE")
        print("=" * 80)
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS:")
        print(f"Total JavaScript/TypeScript anonymous functions: {self.stats.get('total_js', 0)}")
        print(f"Total Python anonymous functions: {self.stats.get('total_py', 0)}")
        print(f"GRAND TOTAL: {self.stats.get('total_js', 0) + self.stats.get('total_py', 0)}")
        
        # JavaScript/TypeScript breakdown
        print("\n🟨 JAVASCRIPT/TYPESCRIPT BREAKDOWN:")
        js_types = [(k.replace('js_', ''), v) for k, v in self.stats.items() if k.startswith('js_') and v > 0]
        js_types.sort(key=lambda x: x[1], reverse=True)
        for func_type, count in js_types:
            print(f"  {func_type}: {count}")
            
        # Python breakdown
        print("\n🐍 PYTHON BREAKDOWN:")
        py_types = [(k.replace('py_', ''), v) for k, v in self.stats.items() if k.startswith('py_') and v > 0]
        py_types.sort(key=lambda x: x[1], reverse=True)
        for func_type, count in py_types:
            print(f"  {func_type}: {count}")
            
        # Files with most anonymous functions
        file_counts = defaultdict(int)
        for findings_list in self.findings.values():
            for finding in findings_list:
                file_counts[finding['file']] += 1
                
        print("\n📁 FILES WITH MOST ANONYMOUS FUNCTIONS:")
        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, count in top_files:
            print(f"  {file_path}: {count} anonymous functions")
            
        # Critical findings (top 30)
        print("\n🚨 TOP 30 CRITICAL INSTANCES TO FIX:")
        all_findings = []
        for func_type, findings_list in self.findings.items():
            for finding in findings_list:
                all_findings.append({
                    'type': func_type,
                    'file': finding['file'],
                    'line': finding['line'],
                    'code': finding['code']
                })
                
        # Prioritize by type (callbacks, event handlers, etc.)
        priority_types = ['js_callback_arrow', 'js_anonymous_function', 'js_event_handler', 
                         'js_promise_constructor', 'py_lambda_function']
        
        critical_findings = []
        for ptype in priority_types:
            critical_findings.extend([f for f in all_findings if f['type'] == ptype][:10])
            
        # Add remaining findings
        remaining = [f for f in all_findings if f not in critical_findings]
        critical_findings.extend(remaining[:max(0, 30 - len(critical_findings))])
        
        for i, finding in enumerate(critical_findings[:30], 1):
            print(f"\n{i}. {finding['type'].replace('_', ' ').title()}")
            print(f"   File: {finding['file']}:{finding['line']}")
            print(f"   Code: {finding['code']}")
            
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("1. Replace arrow functions with named functions for better debugging")
        print("2. Convert anonymous callbacks to named functions for clarity")
        print("3. Replace lambda functions with def statements where appropriate")
        print("4. Use descriptive function names that explain the purpose")
        print("5. Consider using function references instead of inline definitions")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    scanner = AnonymousFunctionScanner("/Users/apple/projects/a2a")
    print("🔍 Scanning A2A codebase for anonymous functions...")
    scanner.scan_directory()
    scanner.generate_report()