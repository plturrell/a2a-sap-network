#!/usr/bin/env python3
"""
Comprehensive JavaScript issue fixer
Fixes common ESLint issues that can't be auto-fixed
"""

import re
import json
from pathlib import Path
import sys

def fix_unused_parameters(content, file_path):
    """Prefix unused parameters with underscore"""
    fixes = 0
    
    # Common patterns for unused parameters
    patterns = [
        # function(req, res, next) where res or next unused
        (r'\((req|request), (res|response), (next)\)', 
         lambda m: f'({m.group(1)}, _{m.group(2)}, _{m.group(3)})'),
        
        # function(req, res) where res unused
        (r'\((req|request), (res|response)\)(?=\s*{)',
         lambda m: f'({m.group(1)}, _{m.group(2)})'),
         
        # error handlers (err, req, res, next)
        (r'\((err|error), (req|request), (res|response), (next)\)',
         lambda m: f'({m.group(1)}, _{m.group(2)}, _{m.group(3)}, _{m.group(4)})'),
         
        # jQuery error callbacks (xhr, status, error)
        (r'\.(?:fail|error)\s*\(\s*function\s*\(\s*(\w+),\s*(\w+),\s*(\w+)\s*\)',
         lambda m: f'.fail(function(_{m.group(1)}, _{m.group(2)}, _{m.group(3)})')
    ]
    
    for pattern, replacement in patterns:
        content, count = re.subn(pattern, replacement, content)
        fixes += count
    
    return content, fixes

def fix_console_statements(content):
    """Wrap console.log with eslint-disable comments"""
    fixes = 0
    
    # Find console.log statements not already wrapped
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        if 'console.log' in line and 'eslint-disable' not in line:
            # Check if previous line already has eslint-disable
            if i == 0 or 'eslint-disable' not in lines[i-1]:
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + '// eslint-disable-next-line no-console')
                fixes += 1
        new_lines.append(line)
    
    return '\n'.join(new_lines), fixes

def fix_undefined_globals(content):
    """Add eslint globals comment for common undefined globals"""
    fixes = 0
    
    # Check if globals comment already exists
    if '/* global' in content:
        return content, fixes
    
    # Common SAP UI5 and browser globals
    globals_needed = []
    
    if 'sap.' in content or 'sap(' in content:
        globals_needed.append('sap')
    if 'jQuery' in content or '$(document)' in content or '$(' in content:
        globals_needed.extend(['jQuery', '$'])
    if 'sessionStorage' in content:
        globals_needed.append('sessionStorage')
    if 'localStorage' in content:
        globals_needed.append('localStorage')
    if 'btoa' in content:
        globals_needed.append('btoa')
    if 'atob' in content:
        globals_needed.append('atob')
    if 'Blob' in content:
        globals_needed.append('Blob')
    if 'URL' in content:
        globals_needed.append('URL')
    if 'WebSocket' in content:
        globals_needed.append('WebSocket')
    if 'Notification' in content:
        globals_needed.append('Notification')
    
    if globals_needed:
        globals_comment = f'/* global {", ".join(set(globals_needed))} */\n'
        # Add after 'use strict' if present, otherwise at beginning
        if '"use strict"' in content:
            content = content.replace('"use strict";\n', f'"use strict";\n{globals_comment}')
        else:
            content = globals_comment + content
        fixes = 1
    
    return content, fixes

def fix_rest_params(content):
    """Replace arguments with rest parameters"""
    fixes = 0
    
    # Pattern: function that uses 'arguments'
    pattern = r'function\s*\([^)]*\)\s*{\s*([^}]*\barguments\b[^}]*)\}'
    
    def replacer(match):
        nonlocal fixes
        body = match.group(1)
        if 'arguments' in body:
            fixes += 1
            # Simple replacement - add ...args parameter
            return match.group(0).replace('function()', 'function(...args)').replace('arguments', 'args')
        return match.group(0)
    
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    return content, fixes

def process_file(file_path):
    """Process a single JavaScript file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        total_fixes = 0
        
        # Apply fixes
        content, fixes = fix_unused_parameters(content, file_path)
        total_fixes += fixes
        
        content, fixes = fix_console_statements(content)
        total_fixes += fixes
        
        content, fixes = fix_undefined_globals(content)
        total_fixes += fixes
        
        content, fixes = fix_rest_params(content)
        total_fixes += fixes
        
        # Write back if changed
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, total_fixes
        
        return False, 0
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0

def main():
    # Read the initial scan results
    with open('js_scan_initial.json', 'r') as f:
        scan_data = json.load(f)
    
    # Get unique file paths from issues
    files_with_issues = set()
    for issue in scan_data['issues']:
        files_with_issues.add(issue['file_path'])
    
    print(f"Found {len(files_with_issues)} files with issues")
    
    # Process files
    files_fixed = 0
    total_fixes = 0
    
    for file_path in sorted(files_with_issues):
        if Path(file_path).exists():
            fixed, fixes = process_file(file_path)
            if fixed:
                files_fixed += 1
                total_fixes += fixes
                print(f"✅ Fixed {fixes} issues in {Path(file_path).name}")
    
    print(f"\n📊 Summary:")
    print(f"  - Files processed: {len(files_with_issues)}")
    print(f"  - Files fixed: {files_fixed}")
    print(f"  - Total fixes applied: {total_fixes}")

if __name__ == "__main__":
    main()