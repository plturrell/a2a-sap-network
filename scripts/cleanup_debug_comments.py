#!/usr/bin/env python3
"""
A2A Platform Debug Comment Cleanup Script
Safely removes commented debug code (Phase 1 cleanup)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

class DebugCommentCleaner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.stats = {
            'files_processed': 0,
            'lines_removed': 0,
            'python_files': 0,
            'js_files': 0
        }
        
        # Safe patterns to remove (debug/temporary code only)
        self.python_safe_patterns = [
            r'^\s*#\s*print\s*\(',  # # print(...)
            r'^\s*#\s*console\.log',  # # console.log (in Python comments)
            r'^\s*#\s*logging\.debug\s*\(',  # # logging.debug(...)
            r'^\s*#\s*logger\.debug\s*\(',  # # logger.debug(...)
            r'^\s*#\s*sys\.stdout\.write',  # # sys.stdout.write
            r'^\s*#\s*pprint\s*\(',  # # pprint(...)
            r'^\s*#\s*time\.sleep\s*\(',  # # time.sleep(...) - debug timing
            r'^\s*#\s*import\s+pdb',  # # import pdb
            r'^\s*#\s*pdb\.set_trace',  # # pdb.set_trace()
            r'^\s*#\s*breakpoint\s*\(',  # # breakpoint()
        ]
        
        self.js_safe_patterns = [
            r'^\s*//\s*console\.log\s*\(',  # // console.log(...)
            r'^\s*//\s*console\.debug\s*\(',  # // console.debug(...)
            r'^\s*//\s*console\.info\s*\(',  # // console.info(...) - if clearly debug
            r'^\s*//\s*console\.warn\s*\(',  # // console.warn(...) - if clearly debug
            r'^\s*//\s*console\.error\s*\(',  # // console.error(...) - if clearly debug
            r'^\s*//\s*debugger\s*;',  # // debugger;
            r'^\s*//\s*alert\s*\(',  # // alert(...) - debug alerts
        ]

    def is_safe_to_remove_python_line(self, line: str) -> bool:
        """Check if a Python comment line is safe to remove"""
        line_stripped = line.strip()
        
        # Skip if it's documentation or important comments
        doc_keywords = ['TODO', 'FIXME', 'NOTE', 'WARNING', 'IMPORTANT', 'BUG', 'HACK']
        if any(keyword in line_stripped.upper() for keyword in doc_keywords):
            return False
            
        # Check against safe patterns
        for pattern in self.python_safe_patterns:
            if re.match(pattern, line):
                return True
                
        return False

    def is_safe_to_remove_js_line(self, line: str) -> bool:
        """Check if a JavaScript comment line is safe to remove"""
        line_stripped = line.strip()
        
        # Skip if it's documentation or important comments
        doc_keywords = ['TODO', 'FIXME', 'NOTE', 'WARNING', 'IMPORTANT', 'BUG', 'HACK']
        if any(keyword in line_stripped.upper() for keyword in doc_keywords):
            return False
            
        # Check against safe patterns
        for pattern in self.js_safe_patterns:
            if re.match(pattern, line):
                return True
                
        return False

    def clean_python_file(self, file_path: Path) -> int:
        """Clean a Python file and return number of lines removed"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            original_count = len(lines)
            cleaned_lines = []
            removed_count = 0
            
            for line in lines:
                if self.is_safe_to_remove_python_line(line):
                    removed_count += 1
                    print(f"  Removing: {line.strip()}")
                else:
                    cleaned_lines.append(line)
            
            if removed_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                print(f"  Cleaned {file_path}: {removed_count} lines removed")
            
            return removed_count
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return 0

    def clean_js_file(self, file_path: Path) -> int:
        """Clean a JavaScript file and return number of lines removed"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            original_count = len(lines)
            cleaned_lines = []
            removed_count = 0
            
            for line in lines:
                if self.is_safe_to_remove_js_line(line):
                    removed_count += 1
                    print(f"  Removing: {line.strip()}")
                else:
                    cleaned_lines.append(line)
            
            if removed_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)
                print(f"  Cleaned {file_path}: {removed_count} lines removed")
            
            return removed_count
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return 0

    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv', 'env'}
        skip_files = {'package-lock.json', '.gitignore'}
        
        # Skip if in excluded directories
        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            return True
            
        # Skip if excluded file
        if file_path.name in skip_files:
            return True
            
        return False

    def run_cleanup(self) -> Dict[str, int]:
        """Run the cleanup process"""
        print(f"Starting Phase 1 cleanup in: {self.root_dir}")
        print("Targeting commented debug statements only...")
        
        for file_path in self.root_dir.rglob('*'):
            if not file_path.is_file() or self.should_skip_file(file_path):
                continue
                
            self.stats['files_processed'] += 1
            
            if file_path.suffix == '.py':
                self.stats['python_files'] += 1
                removed = self.clean_python_file(file_path)
                self.stats['lines_removed'] += removed
                
            elif file_path.suffix == '.js':
                self.stats['js_files'] += 1
                removed = self.clean_js_file(file_path)
                self.stats['lines_removed'] += removed
        
        return self.stats

    def print_summary(self):
        """Print cleanup summary"""
        print("\n" + "="*50)
        print("PHASE 1 CLEANUP SUMMARY")
        print("="*50)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Python files: {self.stats['python_files']}")
        print(f"JavaScript files: {self.stats['js_files']}")
        print(f"Total debug lines removed: {self.stats['lines_removed']}")
        print("="*50)

if __name__ == '__main__':
    # Use current directory if no argument provided
    root_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/apple/projects/a2a'
    
    cleaner = DebugCommentCleaner(root_path)
    stats = cleaner.run_cleanup()
    cleaner.print_summary()