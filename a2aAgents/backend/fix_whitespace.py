#!/usr/bin/env python3
"""
Fix common whitespace issues in Python files:
- W291: trailing whitespace
- W293: blank line contains whitespace
- W292: no newline at end of file
"""

import os
import sys
from pathlib import Path

def fix_whitespace_in_file(filepath):
    """Fix whitespace issues in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return False
        
        modified = False
        
        # Fix W291 and W293: Remove trailing whitespace
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if len(stripped) < len(line.rstrip('\n')):
                lines[i] = stripped + '\n' if line.endswith('\n') else stripped
                modified = True
        
        # Fix W292: Ensure newline at end of file
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
            modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
        return False
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Fix whitespace issues in all Python files under app/"""
    app_dir = Path("app")
    
    if not app_dir.exists():
        print("Error: 'app' directory not found!")
        sys.exit(1)
    
    fixed_count = 0
    total_count = 0
    
    for py_file in app_dir.rglob("*.py"):
        total_count += 1
        if fix_whitespace_in_file(py_file):
            fixed_count += 1
            print(f"Fixed: {py_file}")
    
    print(f"\nSummary: Fixed whitespace in {fixed_count} out of {total_count} files")

if __name__ == "__main__":
    main()