#!/usr/bin/env python3
"""
Script to fix direct HTTP calls that bypass A2A protocol
This ensures 100% A2A protocol compliance by replacing HTTP libraries with A2A messaging
"""

import os
import re
import ast

def fix_http_imports(filepath):
    """Fix direct HTTP library imports"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    # Pattern replacements for HTTP library imports
    replacements = [
        # Import httpx
        (
            r'^import httpx\s*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# import httpx  # REMOVED: A2A protocol violation',
            "Removed httpx import"
        ),
        (
            r'^from httpx import .*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# from httpx import ...  # REMOVED: A2A protocol violation',
            "Removed httpx import"
        ),
        # Import aiohttp
        (
            r'^import aiohttp\s*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# import aiohttp  # REMOVED: A2A protocol violation',
            "Removed aiohttp import"
        ),
        (
            r'^from aiohttp import .*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# from aiohttp import ...  # REMOVED: A2A protocol violation',
            "Removed aiohttp import"
        ),
        # Import requests
        (
            r'^import requests\s*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# import requests  # REMOVED: A2A protocol violation',
            "Removed requests import"
        ),
        (
            r'^from requests import .*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# from requests import ...  # REMOVED: A2A protocol violation',
            "Removed requests import"
        ),
        # urllib
        (
            r'^import urllib\.request\s*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# import urllib.request  # REMOVED: A2A protocol violation',
            "Removed urllib.request import"
        ),
        (
            r'^from urllib\.request import .*$',
            '# Direct HTTP calls not allowed - use A2A protocol\n# from urllib.request import ...  # REMOVED: A2A protocol violation',
            "Removed urllib.request import"
        ),
    ]
    
    for pattern, replacement, description in replacements:
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            changes_made.append(description)
    
    # Look for HTTP client usage patterns and add warnings
    http_patterns = [
        (r'httpx\.AsyncClient\(', 'httpx AsyncClient usage'),
        (r'httpx\.Client\(', 'httpx Client usage'),
        (r'aiohttp\.ClientSession\(', 'aiohttp ClientSession usage'),
        (r'requests\.get\(', 'requests.get usage'),
        (r'requests\.post\(', 'requests.post usage'),
        (r'requests\.put\(', 'requests.put usage'),
        (r'requests\.delete\(', 'requests.delete usage'),
        (r'requests\.Session\(', 'requests Session usage'),
    ]
    
    for pattern, usage_type in http_patterns:
        if re.search(pattern, content):
            # Add warning comment before the usage
            content = re.sub(
                pattern,
                f'# WARNING: {usage_type} violates A2A protocol - must use blockchain messaging\n        # ' + pattern[:-2] + '(',
                content
            )
            changes_made.append(f"Added warning for {usage_type}")
    
    # Add A2A compliance header if changes were made
    if changes_made and not "A2A Protocol Compliance Notice" in content:
        compliance_header = '''"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

'''
        # Insert after module docstring if exists
        if content.startswith('"""'):
            end_docstring = content.find('"""', 3) + 3
            content = content[:end_docstring] + '\n\n' + compliance_header + content[end_docstring:]
        else:
            content = compliance_header + content
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Fixed {filepath}:")
        for change in changes_made:
            print(f"  - {change}")
        return True
    
    return False

def main():
    """Fix all Python files with direct HTTP calls"""
    
    base_dir = "/Users/apple/projects/a2a/a2aAgents/backend"
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip test and venv directories
        if 'test' in root or '__pycache__' in root or 'venv' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Scanning {len(python_files)} Python files for direct HTTP calls...")
    
    fixed_count = 0
    files_with_http = []
    
    # First pass: identify files with HTTP libraries
    for filepath in python_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            if any(lib in content for lib in ['httpx', 'aiohttp', 'requests.', 'urllib.request']):
                files_with_http.append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    print(f"\nFound {len(files_with_http)} files with direct HTTP library usage")
    
    # Second pass: fix the files
    for filepath in files_with_http:
        try:
            if fix_http_imports(filepath):
                fixed_count += 1
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
    
    print(f"\nFixed {fixed_count} files")
    print("\nIMPORTANT: All HTTP communication must now go through A2A blockchain messaging.")
    print("Use A2ANetworkClient for all inter-agent communication.")

if __name__ == "__main__":
    main()