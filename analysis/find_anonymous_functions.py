#!/usr/bin/env python3
"""
Script to find all anonymous functions in the codebase.
This includes:
- JavaScript/TypeScript arrow functions
- JavaScript/TypeScript function expressions
- Python lambda functions
- Callback functions
"""

import os
import re
from pathlib import Path
import json

# Patterns for different types of anonymous functions
PATTERNS = {
    'arrow_functions': [
        r'(?:const|let|var)\s+\w+\s*=\s*\([^)]*\)\s*=>\s*\{',  # const fn = () => {}
        r'(?:const|let|var)\s+\w+\s*=\s*\w+\s*=>\s*\{',        # const fn = x => {}
        r'(?:const|let|var)\s+\w+\s*=\s*\([^)]*\)\s*=>\s*[^{]', # const fn = () => expr
        r'\.\w+\(\s*\([^)]*\)\s*=>\s*\{',                       # .map(() => {})
        r'\.\w+\(\s*\w+\s*=>\s*\{',                             # .map(x => {})
        r'\.\w+\(\s*\([^)]*\)\s*=>\s*[^{]',                     # .map(() => expr)
        r'=\s*\([^)]*\)\s*=>\s*\{',                             # = () => {}
        r'=\s*\w+\s*=>\s*\{',                                   # = x => {}
    ],
    'function_expressions': [
        r'(?:const|let|var)\s+\w+\s*=\s*function\s*\([^)]*\)\s*\{',  # const fn = function() {}
        r'\.\w+\(\s*function\s*\([^)]*\)\s*\{',                      # .method(function() {})
        r'=\s*function\s*\([^)]*\)\s*\{',                            # = function() {}
        r'function\s*\([^)]*\)\s*\{',                                # Anonymous function literal
    ],
    'lambda_functions': [
        r'lambda\s+[^:]*:',  # Python lambda functions
    ],
    'callback_functions': [
        r'(?:addEventListener|on\w+)\s*\(\s*["\'][^"\']*["\'],\s*function\s*\([^)]*\)\s*\{',
        r'(?:setTimeout|setInterval)\s*\(\s*function\s*\([^)]*\)\s*\{',
        r'(?:then|catch|finally)\s*\(\s*function\s*\([^)]*\)\s*\{',
        r'(?:then|catch|finally)\s*\(\s*\([^)]*\)\s*=>\s*\{',
    ]
}

def find_anonymous_functions():
    """Find all anonymous functions in the codebase."""
    results = {
        'arrow_functions': [],
        'function_expressions': [],
        'lambda_functions': [],
        'callback_functions': []
    }
    
    # File extensions to search
    extensions = ['.js', '.ts', '.tsx', '.py']
    
    # Directories to exclude
    exclude_dirs = {'.venv', 'node_modules', '.git', '__pycache__'}
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        # Search for each pattern type
                        for pattern_type, patterns in PATTERNS.items():
                            for pattern in patterns:
                                matches = re.finditer(pattern, content, re.MULTILINE)
                                for match in matches:
                                    # Find line number
                                    line_num = content[:match.start()].count('\n') + 1
                                    line_content = lines[line_num - 1].strip()
                                    
                                    results[pattern_type].append({
                                        'file': file_path,
                                        'line': line_num,
                                        'content': line_content,
                                        'match': match.group()
                                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return results

def main():
    print("Searching for anonymous functions in the codebase...")
    results = find_anonymous_functions()
    
    # Print results
    total_found = 0
    for category, functions in results.items():
        count = len(functions)
        total_found += count
        print(f"\n{category.replace('_', ' ').title()}: {count} found")
        
        if count > 0:
            print("-" * 50)
            for func in functions[:10]:  # Show first 10 for each category
                print(f"  {func['file']}:{func['line']}")
                print(f"    {func['content']}")
            if count > 10:
                print(f"    ... and {count - 10} more")
    
    print(f"\nTotal anonymous functions found: {total_found}")
    
    # Save detailed results to JSON file
    with open('anonymous_functions_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Detailed report saved to 'anonymous_functions_report.json'")

if __name__ == '__main__':
    main()
