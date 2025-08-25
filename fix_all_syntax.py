#!/usr/bin/env python3
"""Fix all syntax errors in the codebase"""

import os
import re
import ast

def fix_indentation_errors(content):
    """Fix common indentation errors"""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix lines with 16+ spaces at the start (usually wrong indentation)
        if line.startswith(' ' * 16) and not line.startswith(' ' * 20):
            # Replace with 8 spaces (2 indent levels)
            line = ' ' * 8 + line.lstrip()
        
        # Fix docstrings that are not indented properly after class definition
        if i > 0 and lines[i-1].strip().endswith(':') and line.strip().startswith('"""'):
            # Check the previous non-empty line for indentation
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            if line.startswith('"""'):  # No indent
                line = ' ' * (prev_indent + 4) + line
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_syntax_in_file(filepath):
    """Fix syntax errors in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Fix indentation errors
        content = fix_indentation_errors(content)
        
        # Fix common patterns
        # Fix self._init_security_features() etc at wrong indentation
        content = re.sub(r'^(\s*)self\._init_security_features\(\)\s*\n\s*self\._init_rate_limiting\(\)\s*\n\s*self\._init_input_validation\(\)\s*\n\s{16,}',
                        r'\1self._init_security_features()\n\1self._init_rate_limiting()\n\1self._init_input_validation()\n\1',
                        content, flags=re.MULTILINE)
        
        # Fix unescaped quotes in strings
        content = re.sub(r"r'([^']*)'([^']*)'([^']*)'", r"r'\1\'\2\'\3'", content)
        
        # Fix docstrings after class/function definitions
        content = re.sub(r'(class \w+[^:]*:)\s*\n\s*(#[^\n]*\n)*"""', r'\1\n    """', content)
        
        # Try to parse to check for syntax errors
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Still has syntax error in {filepath}: {e}")
            # Try additional fixes based on error
            if "expected an indented block" in str(e):
                # Add pass statements where needed
                content = re.sub(r'(def __init__\([^)]*\):\s*\n)(\s*)(\w)', r'\1\2pass\n\2\3', content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return False

# Find all Python files with errors
import subprocess
result = subprocess.run(['python3', '-m', 'compileall', '/Users/apple/projects/a2a/a2aAgents/backend/app/'], 
                       capture_output=True, text=True)

error_files = set()
for line in result.stderr.split('\n'):
    if 'Sorry:' in line and '.py' in line:
        # Extract filename
        match = re.search(r'\(([^,]+\.py),', line)
        if match:
            filename = match.group(1)
            # Find full path
            for root, dirs, files in os.walk('/Users/apple/projects/a2a/a2aAgents/backend/app/'):
                if filename in files:
                    error_files.add(os.path.join(root, filename))

print(f"Found {len(error_files)} files with syntax errors")

# Fix each file
fixed = 0
for filepath in sorted(error_files):
    if fix_syntax_in_file(filepath):
        fixed += 1

print(f"\nFixed {fixed} files")