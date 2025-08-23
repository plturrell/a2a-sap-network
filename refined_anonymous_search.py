#!/usr/bin/env python3
"""
Refined script to find actual anonymous functions in the codebase.
"""

import os
import re
from pathlib import Path
import json

def is_anonymous_function(line, pattern_match):
    """
    Check if the matched pattern is actually an anonymous function
    and not a false positive.
    """
    # Skip comments
    if '//' in line or '/*' in line or '*/' in line or line.strip().startswith('*'):
        return False
    
    # Skip string literals that might contain function-like patterns
    if line.count('"') % 2 == 1 or line.count("'") % 2 == 1:
        return False
    
    return True

def find_real_anonymous_functions():
    """Find actual anonymous functions in the codebase."""
    results = {
        'arrow_functions': [],
        'function_expressions': [],
        'lambda_functions': [],
        'callback_functions': []
    }
    
    # File extensions and their patterns
    js_ts_files = ['.js', '.ts', '.tsx']
    py_files = ['.py']
    
    # Directories to exclude
    exclude_dirs = {'.venv', 'node_modules', '.git', '__pycache__', 'quality_reports'}
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1]
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # JavaScript/TypeScript files
                    if file_ext in js_ts_files:
                        # Arrow functions
                        arrow_patterns = [
                            r'(\w+\s*=>\s*\{)',  # x => {
                            r'(\([^)]*\)\s*=>\s*\{)',  # (x, y) => {
                            r'(\([^)]*\)\s*=>\s*[^{])',  # (x) => expression
                        ]
                        
                        for pattern in arrow_patterns:
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                line_content = lines[line_num - 1].strip()
                                
                                if is_anonymous_function(line_content, match.group()):
                                    results['arrow_functions'].append({
                                        'file': file_path,
                                        'line': line_num,
                                        'content': line_content,
                                        'match': match.group(1)
                                    })
                        
                        # Function expressions
                        func_patterns = [
                            r'(function\s*\([^)]*\)\s*\{)',  # function() {
                            r'(\w+\s*:\s*function\s*\([^)]*\)\s*\{)',  # method: function() {
                        ]
                        
                        for pattern in func_patterns:
                            for match in re.finditer(pattern, content):
                                line_num = content[:match.start()].count('\n') + 1
                                line_content = lines[line_num - 1].strip()
                                
                                # Check if it's truly anonymous (not a named function)
                                if ('function ' in line_content and 
                                    not re.search(r'function\s+\w+', line_content) and
                                    is_anonymous_function(line_content, match.group())):
                                    results['function_expressions'].append({
                                        'file': file_path,
                                        'line': line_num,
                                        'content': line_content,
                                        'match': match.group(1)
                                    })
                    
                    # Python files
                    elif file_ext in py_files:
                        # Lambda functions
                        lambda_pattern = r'(lambda\s+[^:]*:)'
                        
                        for match in re.finditer(lambda_pattern, content):
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = lines[line_num - 1].strip()
                            
                            if is_anonymous_function(line_content, match.group()):
                                results['lambda_functions'].append({
                                    'file': file_path,
                                    'line': line_num,
                                    'content': line_content,
                                    'match': match.group(1)
                                })
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return results

def analyze_anonymous_functions(results):
    """Analyze the found anonymous functions for quality issues."""
    analysis = {
        'total_count': sum(len(funcs) for funcs in results.values()),
        'by_type': {k: len(v) for k, v in results.items()},
        'issues': [],
        'recommendations': []
    }
    
    # Check for common issues
    for category, functions in results.items():
        for func in functions:
            content = func['content']
            
            # Check for complex anonymous functions (long lines)
            if len(content) > 100:
                analysis['issues'].append({
                    'type': 'long_line',
                    'file': func['file'],
                    'line': func['line'],
                    'description': 'Anonymous function on very long line, consider extracting to named function',
                    'content': content[:100] + '...' if len(content) > 100 else content
                })
            
            # Check for nested anonymous functions (rough heuristic)
            if category in ['arrow_functions', 'function_expressions']:
                if '=>' in content and 'function' in content:
                    analysis['issues'].append({
                        'type': 'nested_anonymous',
                        'file': func['file'],
                        'line': func['line'],
                        'description': 'Potentially nested anonymous functions detected',
                        'content': content
                    })
    
    # Generate recommendations
    if analysis['by_type']['arrow_functions'] > analysis['by_type']['function_expressions']:
        analysis['recommendations'].append("Good: Consistent use of arrow functions over function expressions")
    
    if analysis['issues']:
        analysis['recommendations'].append(f"Consider refactoring {len(analysis['issues'])} complex anonymous functions to named functions for better readability")
    
    return analysis

def main():
    print("Searching for actual anonymous functions in the codebase...")
    results = find_real_anonymous_functions()
    
    # Analyze results
    analysis = analyze_anonymous_functions(results)
    
    # Print summary
    print(f"\n=== ANONYMOUS FUNCTIONS SUMMARY ===")
    print(f"Total anonymous functions found: {analysis['total_count']}")
    print(f"Arrow functions: {analysis['by_type']['arrow_functions']}")
    print(f"Function expressions: {analysis['by_type']['function_expressions']}")
    print(f"Lambda functions: {analysis['by_type']['lambda_functions']}")
    print(f"Callback functions: {analysis['by_type']['callback_functions']}")
    
    # Show some examples
    print(f"\n=== EXAMPLES ===")
    for category, functions in results.items():
        if functions:
            print(f"\n{category.replace('_', ' ').title()}:")
            for func in functions[:3]:  # Show first 3 examples
                print(f"  {func['file']}:{func['line']}")
                print(f"    {func['content']}")
    
    # Show issues
    if analysis['issues']:
        print(f"\n=== POTENTIAL ISSUES ===")
        for issue in analysis['issues'][:10]:  # Show first 10 issues
            print(f"  {issue['type']}: {issue['file']}:{issue['line']}")
            print(f"    {issue['description']}")
            print(f"    {issue['content']}")
    
    # Show recommendations
    if analysis['recommendations']:
        print(f"\n=== RECOMMENDATIONS ===")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
    
    # Save detailed results
    output = {
        'results': results,
        'analysis': analysis
    }
    
    with open('anonymous_functions_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed analysis saved to 'anonymous_functions_analysis.json'")

if __name__ == '__main__':
    main()
