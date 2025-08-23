#!/usr/bin/env python3
"""Fix extra closing braces in schema.cds"""

import re

def fix_extra_braces(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a standalone }; line
        if line.strip() == '};':
            # Look at the previous non-empty line
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            
            if j >= 0:
                prev_line = lines[j].strip()
                # If previous line ends with };, this is likely an extra one
                if prev_line.endswith('};'):
                    # Skip this line (don't add it)
                    i += 1
                    continue
                # If previous line is just a field definition, this might be extra
                elif (prev_line.endswith(';') and 
                      not prev_line.endswith('};') and 
                      ':' in prev_line and
                      not 'enum' in lines[j]):
                    # Look further back to see if we're inside an entity
                    k = j - 1
                    while k >= 0:
                        if 'entity' in lines[k] and ':' in lines[k]:
                            # We're inside an entity definition, this }; is extra
                            i += 1
                            continue
                        elif lines[k].strip() == '}':
                            # We already closed something, this is extra
                            i += 1
                            continue
                        k -= 1
        
        fixed_lines.append(line)
        i += 1
    
    return ''.join(fixed_lines)

if __name__ == '__main__':
    file_path = '/Users/apple/projects/a2a/a2aNetwork/db/schema.cds'
    
    fixed_content = fix_extra_braces(file_path)
    
    with open(file_path, 'w') as f:
        f.write(fixed_content)
    
    print("Fixed extra closing braces")