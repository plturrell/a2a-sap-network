#!/usr/bin/env python3
"""Fix common syntax errors in the codebase"""

import os
import re

def fix_file(filepath):
    """Fix common syntax issues in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        
        # Fix common indentation patterns
        # Fix lines that start with too many spaces (16+ spaces often indicate wrong indentation)
        content = re.sub(r'^(\s{16,})(\S)', lambda m: '        ' + m.group(2), content, flags=re.MULTILINE)
        
        # Fix empty __init__ methods
        content = re.sub(r'def __init__\(self[^)]*\):\s*\n\s*\n(\s+)(\S)', r'def __init__(self):\n\1pass\n\1\2', content)
        
        # Fix class definitions with missing pass
        content = re.sub(r'(class \w+[^:]*:)\s*\n(\s+)(#[^\n]*\n\s*)*\n(\w)', r'\1\n\2pass\n\4', content)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return False

# Fix specific files with known issues
files_to_fix = [
    "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent0DataProduct/active/pdfProcessingModule.py",
    "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent0DataProduct/active/perplexityApiModule.py",
    "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent1Standardization/active/comprehensiveDataStandardizationAgentSdk.py",
    "/Users/apple/projects/a2a/a2aAgents/backend/app/a2a/agents/agent1Standardization/active/enhancedDataStandardizationAgentMcp.py",
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        fix_file(filepath)

print("Done.")