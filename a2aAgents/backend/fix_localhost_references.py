#!/usr/bin/env python3
"""
Script to systematically fix localhost references in A2A agent files.
Replaces hardcoded localhost URLs with environment variable fallbacks.
"""

import os
import re
import glob
from typing import List, Dict, Tuple


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
def find_files_with_localhost(base_path: str, patterns: List[str]) -> List[str]:
    """Find all files containing localhost references."""
    files_with_localhost = []
    
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(base_path, pattern), recursive=True):
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'localhost' in content.lower():
                            files_with_localhost.append(file_path)
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    return list(set(files_with_localhost))  # Remove duplicates

def fix_localhost_references(file_path: str) -> Tuple[bool, List[str]]:
    """Fix localhost references in a single file."""
    changes_made = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Common patterns to replace with environment variables
        replacements = [
            # Blockchain RPC URLs
            (r'["\']http://localhost:8545["\']', 
             'os.getenv("A2A_RPC_URL", os.getenv("BLOCKCHAIN_RPC_URL"))'),
            
            # Agent service URLs
            (r'["\']http://localhost:8000["\']', 
             'os.getenv("A2A_BASE_URL")'),
            (r'["\']http://localhost:8001["\']', 
             'os.getenv("DATA_MANAGER_URL")'),
            (r'["\']http://localhost:8002["\']', 
             'os.getenv("CATALOG_MANAGER_URL")'),
            (r'["\']http://localhost:8003["\']', 
             'os.getenv("AGENT_MANAGER_URL")'),
            (r'["\']http://localhost:8010["\']', 
             'os.getenv("A2A_AGENT_MANAGER_URL")'),
            (r'["\']http://localhost:8080["\']', 
             'os.getenv("A2A_GATEWAY_URL")'),
            
            # Redis URLs
            (r'["\']redis://localhost:6379["\']', 
             'os.getenv("REDIS_URL", "redis://localhost:6379")'),
            
            # Frontend URLs
            (r'["\']http://localhost:3000["\']', 
             'os.getenv("A2A_FRONTEND_URL")'),
        ]
        
        for pattern, replacement in replacements:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, f'"{replacement}"', content)
                changes_made.append(f"Replaced {len(matches)} occurrences of {pattern}")
        
        # Special case: fix base_url assignments
        base_url_pattern = r'base_url\s*=\s*["\']http://localhost:(\d+)["\']'
        matches = re.findall(base_url_pattern, content)
        if matches:
            for port in matches:
                old_pattern = f'base_url = "http://localhost:{port}"'
                new_pattern = f'base_url = os.getenv("A2A_AGENT_BASE_URL", os.getenv("SERVICE_BASE_URL", "http://localhost:{port}"))'
                content = content.replace(old_pattern, new_pattern)
                changes_made.append(f"Replaced base_url assignment for port {port}")
        
        # Write back if changes were made
        if content != original_content:
            # Ensure os import is present
            if 'import os' not in content and 'from os import' not in content:
                lines = content.split('\n')
                # Find the best place to insert the import
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_index = i + 1
                    elif line.strip() and not line.startswith('#'):
                        break
                
                lines.insert(insert_index, 'import os')
                content = '\n'.join(lines)
                changes_made.append("Added 'import os' statement")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, changes_made
        
        return False, []
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, []

def main():
    """Main function to fix localhost references across the project."""
    base_path = '/Users/apple/projects/a2a/a2aAgents/backend'
    
    # File patterns to search
    patterns = [
        'app/a2a/agents/**/*.py',
        'services/**/*.py',
        'scripts/**/*.py',
        'config/*.py',
        'main.py'
    ]
    
    print("🔍 Finding files with localhost references...")
    files_with_localhost = find_files_with_localhost(base_path, patterns)
    
    print(f"📁 Found {len(files_with_localhost)} files with localhost references")
    
    # Process files
    total_files_changed = 0
    total_changes = 0
    
    for file_path in files_with_localhost:
        # Skip test files for now (lower priority)
        if '/test_' in file_path or file_path.endswith('_test.py'):
            continue
            
        print(f"\n🔧 Processing: {os.path.relpath(file_path, base_path)}")
        
        changed, changes = fix_localhost_references(file_path)
        
        if changed:
            total_files_changed += 1
            total_changes += len(changes)
            for change in changes:
                print(f"  ✅ {change}")
        else:
            print(f"  ℹ️  No changes needed")
    
    print(f"\n🎉 Summary:")
    print(f"   📁 Files processed: {len([f for f in files_with_localhost if '/test_' not in f])}")
    print(f"   ✅ Files changed: {total_files_changed}")
    print(f"   🔄 Total changes: {total_changes}")
    
    if total_files_changed > 0:
        print(f"\n📝 Next steps:")
        print(f"   1. Review changes in the modified files")
        print(f"   2. Update your .env file with the new environment variables")
        print(f"   3. Test the application to ensure everything works")
        print(f"   4. Consider running tests to validate the changes")

if __name__ == "__main__":
    main()