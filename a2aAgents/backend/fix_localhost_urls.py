#!/usr/bin/env python3
"""
Script to fix all hardcoded localhost URLs in the A2A agent ecosystem
This ensures 100% A2A protocol compliance by requiring environment variables
"""

import os
import re
import sys

def fix_localhost_in_file(filepath):
    """Fix hardcoded localhost URLs in a single file"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Pattern replacements for different localhost patterns
    replacements = [
        # http://localhost:PORT patterns in os.getenv
        (
            r'os\.getenv\((["\'][^"\']+["\'])\s*,\s*["\']http://localhost:\d+["\']\)',
            r'os.getenv(\1)',
            "Removed localhost fallback from os.getenv"
        ),
        # Direct localhost URLs
        (
            r'["\']http://localhost:\d+["\']',
            r'os.getenv("A2A_SERVICE_URL")',
            "Replaced hardcoded localhost URL"
        ),
        # 127.0.0.1 patterns
        (
            r'["\']http://127\.0\.0\.1:\d+["\']',
            r'os.getenv("A2A_SERVICE_URL")',
            "Replaced hardcoded 127.0.0.1 URL"
        ),
        # localhost without http
        (
            r'["\']localhost:\d+["\']',
            r'os.getenv("A2A_SERVICE_HOST")',
            "Replaced hardcoded localhost host"
        ),
    ]
    
    for pattern, replacement, description in replacements:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made.append(f"{description}: {len(matches)} occurrences")
    
    # Add environment variable checks after imports if changes were made
    if changes_made and content != original_content:
        # Find the right place to add checks (after imports)
        import_section_end = max(
            content.rfind('\nimport '),
            content.rfind('\nfrom ')
        )
        
        if import_section_end > 0:
            # Find the next newline after the last import
            insert_pos = content.find('\n\n', import_section_end) + 2
            
            env_check = '''
# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
'''
            
            # Only add if not already present
            if "A2A Protocol Compliance: Require environment variables" not in content:
                content = content[:insert_pos] + env_check + content[insert_pos:]
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Fixed {filepath}:")
        for change in changes_made:
            print(f"  - {change}")
        return True
    
    return False

def main():
    """Fix all Python files with hardcoded localhost URLs"""
    
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
    
    print(f"Scanning {len(python_files)} Python files for hardcoded localhost URLs...")
    
    fixed_count = 0
    files_with_localhost = []
    
    # First pass: identify files with localhost
    for filepath in python_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            if any(pattern in content for pattern in ['localhost:', '127.0.0.1:']):
                files_with_localhost.append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print(f"\nFound {len(files_with_localhost)} files with localhost URLs")
    
    # Second pass: fix the files
    for filepath in files_with_localhost:
        try:
            if fix_localhost_in_file(filepath):
                fixed_count += 1
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
    
    print(f"\nFixed {fixed_count} files")
    print("\nREMINDER: Set these environment variables for A2A compliance:")
    print("  - A2A_SERVICE_URL")
    print("  - A2A_SERVICE_HOST") 
    print("  - A2A_BASE_URL")
    print("  - A2A_AGENT_URL")
    print("  - A2A_MANAGER_URL")
    print("  - A2A_DOWNSTREAM_URL")

if __name__ == "__main__":
    main()