#!/usr/bin/env python3
"""Fix enum syntax errors in schema.cds file"""

import re

def fix_enum_syntax(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern 1: Fix enums with semicolon on separate line
    # Match enum { ... \n; pattern and replace with enum { ... };
    pattern1 = r'(enum\s*\{[^}]+)\n\s*;'
    content = re.sub(pattern1, r'\1\n    };', content)
    
    # Pattern 2: Fix enums with default inside the enum definition
    # Match patterns like: enum { ... } default 'VALUE';
    pattern2 = r'(enum\s*\{[^}]+)\s+default\s+([^;]+);'
    content = re.sub(pattern2, r'\1\n    } default \2;', content)
    
    # Pattern 3: Ensure all enums have closing braces
    # This is more complex and needs careful handling
    lines = content.split('\n')
    fixed_lines = []
    in_enum = False
    enum_indent = 0
    
    for i, line in enumerate(lines):
        # Check if we're starting an enum
        if 'enum {' in line and not line.strip().endswith('}'):
            in_enum = True
            enum_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        elif in_enum:
            # Check if this line should close the enum
            current_indent = len(line) - len(line.lstrip())
            
            # If we hit a line with less or equal indentation that's not part of enum values
            if (current_indent <= enum_indent and line.strip() and 
                not line.strip().endswith(';') and 
                not any(keyword in line for keyword in ['DRAFT', 'PENDING', 'RUNNING', 'COMPLETED', 
                                                        'FAILED', 'CSV', 'JSON', 'XML', 'CLASSIFICATION',
                                                        'REGRESSION', 'CLUSTERING', 'TABULAR', 'TEXT'])):
                # Insert closing brace before this line
                fixed_lines.append(' ' * (enum_indent + 4) + '};')
                in_enum = False
                fixed_lines.append(line)
            elif line.strip() == ';':
                # Replace standalone semicolon with closing brace and semicolon
                fixed_lines.append(' ' * (enum_indent + 4) + '};')
                in_enum = False
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

if __name__ == '__main__':
    file_path = '/Users/apple/projects/a2a/a2aNetwork/db/schema.cds'
    
    # Create backup
    with open(file_path, 'r') as f:
        backup_content = f.read()
    
    with open(file_path + '.backup', 'w') as f:
        f.write(backup_content)
    
    # Fix the file
    fixed_content = fix_enum_syntax(file_path)
    
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print("Fixed enum syntax errors in schema.cds")
    print("Backup saved to schema.cds.backup")