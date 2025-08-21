#!/usr/bin/env python3
"""
Fix logging format issues (W1203) - Convert f-strings to % formatting in logging calls
"""

import re
import sys
from pathlib import Path

def fix_logging_format(file_path: Path) -> int:
    """Fix logging format issues in a single file"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    fixes = 0
    
    # Pattern to match logger calls with f-strings
    # Matches: logger.info(f"..."), logger.error(f"..."), etc.
    pattern = r'(logger\.(debug|info|warning|error|critical))\(f["\']([^"\']*)\{([^}]+)\}([^"\']*)["\']'
    
    def replacer(match):
        nonlocal fixes
        fixes += 1
        
        logger_call = match.group(1)
        before_text = match.group(3)
        variable = match.group(4)
        after_text = match.group(5)
        
        # Build the format string with %s placeholder
        format_string = f'"{before_text}%s{after_text}"'
        
        # Return the fixed logging call
        return f'{logger_call}({format_string}, {variable}'
    
    # Apply the fix
    content = re.sub(pattern, replacer, content)
    
    # Handle multiple variables in f-strings (simplified approach)
    # This would need more sophisticated parsing for complex cases
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed {fixes} logging format issues in {file_path}")
    
    return fixes

def main():
    """Fix logging format issues in all Python files in the core directory"""
    
    core_dir = Path("/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/core")
    
    if not core_dir.exists():
        print(f"Directory not found: {core_dir}")
        sys.exit(1)
    
    total_fixes = 0
    files_fixed = 0
    
    # Find all Python files
    python_files = list(core_dir.rglob("*.py"))
    
    print(f"Scanning {len(python_files)} Python files for logging format issues...")
    
    for file_path in python_files:
        fixes = fix_logging_format(file_path)
        if fixes > 0:
            files_fixed += 1
            total_fixes += fixes
    
    print(f"\nSummary:")
    print(f"Files fixed: {files_fixed}")
    print(f"Total fixes: {total_fixes}")
    
    if total_fixes == 0:
        print("No logging format issues found or all issues were already fixed.")

if __name__ == "__main__":
    main()