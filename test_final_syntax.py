#!/usr/bin/env python3
import sys
import os
import ast

sys.path.append('.')
sys.path.append('./a2aAgents/backend')

total_files = 0
importable_files = 0
syntax_errors = 0

for root, dirs, files in os.walk('./a2aAgents/backend/app/a2a/agents'):
    for file in files:
        if file.endswith('.py') and not file.startswith('test_') and file != '__init__.py':
            filepath = os.path.join(root, file)
            total_files += 1
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                ast.parse(content)
                importable_files += 1
            except SyntaxError:
                syntax_errors += 1
                print(f"❌ Syntax error in: {filepath}")
            except Exception:
                pass  # Other import errors are expected without full environment

print(f'📊 FINAL RESULTS:')
print(f'  Total agent files checked: {total_files}')
print(f'  Syntactically correct: {importable_files}')
print(f'  Syntax errors: {syntax_errors}')
print(f'  Success rate: {(importable_files/total_files*100):.1f}%')

if syntax_errors == 0:
    print('🎉 ALL FILES HAVE CORRECT SYNTAX!')
else:
    print(f'⚠️  {syntax_errors} files still have syntax errors')