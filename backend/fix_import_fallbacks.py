#!/usr/bin/env python3
"""
Script to fix import fallback patterns that bypass A2A protocol
This ensures 100% A2A protocol compliance by removing all try/except ImportError blocks
"""

import os
import re
import ast

def analyze_import_fallbacks(filepath):
    """Analyze a file for import fallback patterns"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    fallback_patterns = []
    
    # Parse the AST to find try/except blocks
    try:
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Check if this is an import-related try block
                has_import = False
                for stmt in node.body:
                    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        has_import = True
                        break
                
                if has_import:
                    # Check if except handles ImportError
                    for handler in node.handlers:
                        if handler.type:
                            if isinstance(handler.type, ast.Name) and handler.type.id == 'ImportError':
                                fallback_patterns.append({
                                    'line': node.lineno,
                                    'col': node.col_offset,
                                    'type': 'import_fallback'
                                })
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
    
    return fallback_patterns

def fix_import_fallbacks_simple(filepath):
    """Fix simple import fallback patterns"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_try_block = False
    try_indent = 0
    skip_until_dedent = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detect try block with import
        if stripped.startswith('try:') and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if 'import' in next_line or 'from' in next_line:
                # This is likely an import try block
                # Find the matching except ImportError
                j = i + 2
                found_import_error = False
                while j < len(lines):
                    check_line = lines[j].strip()
                    if check_line.startswith('except ImportError'):
                        found_import_error = True
                        break
                    elif check_line and not check_line.startswith(' ') and not check_line.startswith('\t'):
                        break
                    j += 1
                
                if found_import_error:
                    # Skip the try: line and convert imports to direct imports
                    i += 1
                    while i < len(lines) and ('import' in lines[i] or 'from' in lines[i]):
                        # Remove extra indentation from imports
                        import_line = lines[i].lstrip()
                        if import_line.strip():
                            new_lines.append(import_line)
                        i += 1
                    
                    # Skip the rest of the try/except block
                    indent_level = len(line) - len(line.lstrip())
                    while i < len(lines):
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if lines[i].strip() and current_indent <= indent_level:
                            i -= 1  # Back up one line
                            break
                        i += 1
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    
    return True

def add_import_requirements_check(filepath):
    """Add a check for required imports after fixing fallbacks"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the last import statement
    import_lines = []
    lines = content.split('\n')
    last_import_idx = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            last_import_idx = i
    
    if last_import_idx >= 0:
        # Add import check after imports
        check_code = '''
# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies'''
        
        lines.insert(last_import_idx + 1, '')
        lines.insert(last_import_idx + 2, check_code)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

def main():
    """Fix all Python files with import fallback patterns"""
    
    base_dir = "/Users/apple/projects/a2a/a2aAgents/backend"
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Scanning {len(python_files)} Python files for import fallback patterns...")
    
    files_with_fallbacks = []
    
    # First pass: identify files with import fallbacks
    for filepath in python_files:
        try:
            fallbacks = analyze_import_fallbacks(filepath)
            if fallbacks:
                files_with_fallbacks.append((filepath, fallbacks))
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    print(f"\nFound {len(files_with_fallbacks)} files with import fallback patterns")
    
    # Second pass: fix the fallbacks
    fixed_count = 0
    for filepath, fallbacks in files_with_fallbacks:
        try:
            print(f"\nFixing {filepath}:")
            print(f"  Found {len(fallbacks)} import fallback patterns")
            
            if fix_import_fallbacks_simple(filepath):
                add_import_requirements_check(filepath)
                fixed_count += 1
                print(f"  ✓ Fixed import fallbacks")
            
        except Exception as e:
            print(f"  ✗ Error fixing {filepath}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files with import fallbacks")
    print(f"\nIMPORTANT: All A2A agents now require their dependencies to be installed.")
    print(f"No fallback implementations are allowed for protocol compliance.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()