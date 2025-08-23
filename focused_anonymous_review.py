#!/usr/bin/env python3
"""
Focused review of anonymous functions in main source code (excluding tests).
"""

import os
import re
from pathlib import Path

def analyze_file_for_anonymous_functions(file_path):
    """Analyze a single file for anonymous functions."""
    functions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            
        # JavaScript/TypeScript files
        if file_path.endswith(('.js', '.ts', '.tsx')):
            # Arrow functions
            arrow_patterns = [
                r'(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{',  # const fn = () => {}
                r'(?:const|let|var)\s+(\w+)\s*=\s*(\w+)\s*=>\s*\{',      # const fn = x => {}
                r'\.(\w+)\s*\(\s*\([^)]*\)\s*=>\s*\{',                   # .method(() => {})
                r'\.(\w+)\s*\(\s*(\w+)\s*=>\s*\{',                       # .method(x => {})
            ]
            
            for pattern in arrow_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    functions.append({
                        'type': 'arrow_function',
                        'line': line_num,
                        'content': line_content,
                        'pattern': pattern
                    })
            
            # Function expressions (anonymous)
            func_patterns = [
                r'function\s*\([^)]*\)\s*\{',  # Anonymous function() {}
                r'\w+\s*:\s*function\s*\([^)]*\)\s*\{',  # property: function() {}
            ]
            
            for pattern in func_patterns:
                for match in re.finditer(pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # Skip named functions
                    if not re.search(r'function\s+\w+', line_content):
                        functions.append({
                            'type': 'function_expression',
                            'line': line_num,
                            'content': line_content,
                            'pattern': pattern
                        })
        
        # Python files
        elif file_path.endswith('.py'):
            # Lambda functions
            lambda_pattern = r'lambda\s+[^:]*:'
            
            for match in re.finditer(lambda_pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1].strip()
                
                functions.append({
                    'type': 'lambda_function',
                    'line': line_num,
                    'content': line_content,
                    'pattern': lambda_pattern
                })
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return functions

def get_source_files():
    """Get main source files (excluding tests)."""
    source_files = []
    
    # Key source directories
    source_dirs = ['a2aAgents', 'a2aNetwork', 'common', 'src']
    exclude_patterns = ['test', 'Test', '.test.', '_test.', 'spec.', '.spec.']
    
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                # Skip test directories
                dirs[:] = [d for d in dirs if not any(pattern in d.lower() for pattern in exclude_patterns)]
                
                for file in files:
                    if file.endswith(('.js', '.ts', '.tsx', '.py')):
                        # Skip test files
                        if not any(pattern in file for pattern in exclude_patterns):
                            source_files.append(os.path.join(root, file))
    
    # Also check root level files
    for file in os.listdir('.'):
        if file.endswith(('.js', '.ts', '.tsx', '.py')) and os.path.isfile(file):
            if not any(pattern in file for pattern in exclude_patterns):
                source_files.append(file)
    
    return source_files

def categorize_and_review_functions(all_functions):
    """Categorize and review the found functions."""
    
    categories = {
        'event_handlers': [],
        'array_operations': [],
        'promises_async': [],
        'object_methods': [],
        'complex_logic': [],
        'simple_transforms': [],
        'other': []
    }
    
    for file_path, functions in all_functions.items():
        for func in functions:
            content = func['content'].lower()
            
            # Categorize based on content
            if any(keyword in content for keyword in ['onclick', 'onchange', 'onerror', 'addeventlistener']):
                categories['event_handlers'].append((file_path, func))
            elif any(keyword in content for keyword in ['map', 'filter', 'reduce', 'foreach', 'find', 'sort']):
                categories['array_operations'].append((file_path, func))
            elif any(keyword in content for keyword in ['then', 'catch', 'async', 'await', 'promise']):
                categories['promises_async'].append((file_path, func))
            elif ':' in content and func['type'] == 'function_expression':
                categories['object_methods'].append((file_path, func))
            elif len(content) > 80:
                categories['complex_logic'].append((file_path, func))
            elif func['type'] == 'lambda_function' and len(content) < 50:
                categories['simple_transforms'].append((file_path, func))
            else:
                categories['other'].append((file_path, func))
    
    return categories

def main():
    print("=== FOCUSED ANONYMOUS FUNCTION REVIEW ===")
    print("Analyzing main source code files (excluding tests)...\n")
    
    source_files = get_source_files()
    print(f"Found {len(source_files)} source files to analyze\n")
    
    all_functions = {}
    total_count = 0
    
    for file_path in source_files:
        functions = analyze_file_for_anonymous_functions(file_path)
        if functions:
            all_functions[file_path] = functions
            total_count += len(functions)
    
    print(f"Total anonymous functions in source code: {total_count}")
    print(f"Files with anonymous functions: {len(all_functions)}\n")
    
    # Categorize functions
    categories = categorize_and_review_functions(all_functions)
    
    print("=== CATEGORIZED REVIEW ===\n")
    
    for category, items in categories.items():
        if items:
            print(f"{category.replace('_', ' ').title()}: {len(items)} functions")
            print("-" * 50)
            
            for file_path, func in items[:5]:  # Show first 5 examples
                print(f"  {file_path}:{func['line']}")
                print(f"    {func['content']}")
                print()
            
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more\n")
    
    # Provide specific recommendations
    print("=== RECOMMENDATIONS ===\n")
    
    if categories['complex_logic']:
        print(f"• Consider extracting {len(categories['complex_logic'])} complex anonymous functions to named functions")
        print("  This will improve readability and make debugging easier.\n")
    
    if categories['event_handlers']:
        print(f"• Review {len(categories['event_handlers'])} event handler anonymous functions")
        print("  Consider using named functions for better error tracking.\n")
    
    if categories['array_operations']:
        print(f"• {len(categories['array_operations'])} array operation functions found")
        print("  These are generally acceptable as anonymous functions.\n")
    
    if categories['promises_async']:
        print(f"• {len(categories['promises_async'])} promise/async anonymous functions")
        print("  Consider using async/await syntax for better readability.\n")

if __name__ == '__main__':
    main()
