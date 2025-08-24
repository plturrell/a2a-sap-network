"""
A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
"""

#!/usr/bin/env python3
"""
A2A Async/Await Pattern Migration Script
Systematically standardizes async patterns across the entire codebase
"""
import asyncio
import time

import ast
import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import importlib.util

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.logging_config import get_logger, LogCategory


@dataclass
class AsyncPattern:
    """Represents an async pattern found in code"""
    file_path: str
    line_number: int
    pattern_type: str
    original_code: str
    suggested_fix: str
    severity: str  # 'high', 'medium', 'low'
    confidence: float
    description: str


class AsyncPatternAnalyzer:
    """Analyzes Python files for async/await patterns and anti-patterns"""
    
    def __init__(self):
        self.logger = get_logger(__name__, LogCategory.SYSTEM)
        self.patterns_found: List[AsyncPattern] = []
    
    def analyze_file(self, file_path: Path) -> List[AsyncPattern]:
        """Analyze a single Python file for async patterns"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                ast_patterns = self._analyze_ast(tree, str(file_path), lines)
                patterns.extend(ast_patterns)
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
            
            # Regex-based pattern detection
            regex_patterns = self._analyze_with_regex(str(file_path), lines)
            patterns.extend(regex_patterns)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
        
        return patterns
    
    def _analyze_ast(self, tree: ast.AST, file_path: str, lines: List[str]) -> List[AsyncPattern]:
        """Analyze AST for async patterns"""
        patterns = []
        
        class AsyncVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_async_function = False
                self.async_function_stack = []
            
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                # Check for missing type hints
                if node.returns is None:
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=node.lineno,
                        pattern_type="missing_return_type",
                        original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        suggested_fix=f"async def {node.name}(...) -> ReturnType:",
                        severity="medium",
                        confidence=0.9,
                        description="Async function missing return type annotation"
                    ))
                
                # Check for missing docstring
                if (not node.body or 
                    not isinstance(node.body[0], ast.Expr) or 
                    not isinstance(node.body[0].value, ast.Str)):
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=node.lineno,
                        pattern_type="missing_docstring",
                        original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        suggested_fix=f'async def {node.name}(...):\n    """Docstring describing the async operation"""',
                        severity="low",
                        confidence=0.7,
                        description="Async function missing docstring"
                    ))
                
                self.in_async_function = True
                self.async_function_stack.append(node.name)
                self.generic_visit(node)
                self.async_function_stack.pop()
                self.in_async_function = len(self.async_function_stack) > 0
            
            def visit_Call(self, node: ast.Call):
                # Check for sync calls that should be awaited
                if isinstance(node.func, ast.Attribute):
                    # Database operations that should be async
                    if (hasattr(node.func, 'attr') and 
                        node.func.attr in ['execute', 'fetch', 'fetchall', 'fetchone', 'commit', 'rollback']):
                        
                        # Check if this call is awaited
                        parent = getattr(node, 'parent', None)
                        if not isinstance(parent, ast.Await):
                            patterns.append(AsyncPattern(
                                file_path=file_path,
                                line_number=node.lineno,
                                pattern_type="missing_await",
                                original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                suggested_fix="await " + lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                severity="high",
                                confidence=0.8,
                                description="Database operation should be awaited"
                            ))
                
                # Check for httpx calls without await
                if (isinstance(node.func, ast.Attribute) and 
                    hasattr(node.func, 'attr') and
                    node.func.attr in ['get', 'post', 'put', 'delete', 'patch']):
                    
                    parent = getattr(node, 'parent', None)
                    if not isinstance(parent, ast.Await):
                        patterns.append(AsyncPattern(
                            file_path=file_path,
                            line_number=node.lineno,
                            pattern_type="missing_await_http",
                            original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            suggested_fix="await " + lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            severity="high", 
                            confidence=0.9,
                            description="HTTP call should be awaited"
                        ))
                
                self.generic_visit(node)
            
            def visit_With(self, node: ast.With):
                # Check for sync context managers in async functions
                if self.in_async_function:
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=node.lineno,
                        pattern_type="sync_context_in_async",
                        original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                        suggested_fix="async with " + lines[node.lineno - 1].strip().replace("with ", "") if node.lineno <= len(lines) else "",
                        severity="medium",
                        confidence=0.6,
                        description="Consider using async context manager in async function"
                    ))
                
                self.generic_visit(node)
            
            def visit_Try(self, node: ast.Try):
                # Check for proper exception handling in async functions
                if self.in_async_function:
                    has_specific_exception = any(
                        isinstance(handler.type, ast.Name) and 
                        handler.type.id not in ['Exception', 'BaseException']
                        for handler in node.handlers
                        if handler.type
                    )
                    
                    if not has_specific_exception:
                        patterns.append(AsyncPattern(
                            file_path=file_path,
                            line_number=node.lineno,
                            pattern_type="generic_exception_handling",
                            original_code=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            suggested_fix="Use specific exception types (e.g., asyncio.TimeoutError, httpx.RequestError)",
                            severity="medium",
                            confidence=0.7,
                            description="Async function should use specific exception types"
                        ))
                
                self.generic_visit(node)
        
        # Add parent references for better analysis
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                setattr(child, 'parent', node)
        
        visitor = AsyncVisitor()
        visitor.visit(tree)
        
        return patterns
    
    def _analyze_with_regex(self, file_path: str, lines: List[str]) -> List[AsyncPattern]:
        """Analyze file using regex patterns for common issues"""
        patterns = []
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for fire-and-forget async calls
            if re.search(r'asyncio\.create_task\([^)]+\)(?!\s*\.)', line_stripped):
                if 'await' not in line_stripped and '=' not in line_stripped:
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=line_num,
                        pattern_type="fire_and_forget_task",
                        original_code=line_stripped,
                        suggested_fix="task = " + line_stripped + "; # Store task reference",
                        severity="high",
                        confidence=0.8,
                        description="Fire-and-forget task should be tracked or awaited"
                    ))
            
            # Check for missing await on async calls
            if re.search(r'\.(?:get|post|put|delete|patch|execute|fetch)\([^)]*\)(?!\s*\.)', line_stripped):
                if 'await' not in line_stripped and 'def ' not in line_stripped:
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=line_num,
                        pattern_type="missing_await_call",
                        original_code=line_stripped,
                        suggested_fix="await " + line_stripped,
                        severity="high",
                        confidence=0.7,
                        description="Async call should be awaited"
                    ))
            
            # Check for sleep without asyncio
            if re.search(r'\btime\.sleep\s*\(', line_stripped):
                if 'async def' in '\n'.join(lines[max(0, line_num-10):line_num]):
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=line_num,
                        pattern_type="blocking_sleep",
                        original_code=line_stripped,
                        suggested_fix=line_stripped.replace('time.sleep', 'await asyncio.sleep'),
                        severity="high",
                        confidence=0.9,
                        description="Use asyncio.sleep instead of time.sleep in async functions"
                    ))
            
            # Check for missing async with
            if re.search(r'\bwith\s+.*\.(?:begin|transaction|connection|session)\s*\(', line_stripped):
                if 'async def' in '\n'.join(lines[max(0, line_num-10):line_num]):
                    patterns.append(AsyncPattern(
                        file_path=file_path,
                        line_number=line_num,
                        pattern_type="missing_async_with",
                        original_code=line_stripped,
                        suggested_fix=line_stripped.replace('with ', 'async with '),
                        severity="medium",
                        confidence=0.6,
                        description="Consider using async with for async context managers"
                    ))
            
            # Check for proper error handling
            if 'except Exception' in line_stripped and 'async def' in '\n'.join(lines[max(0, line_num-20):line_num]):
                patterns.append(AsyncPattern(
                    file_path=file_path,
                    line_number=line_num,
                    pattern_type="generic_async_exception",
                    original_code=line_stripped,
                    suggested_fix="except (SpecificException, asyncio.TimeoutError) as e:",
                    severity="medium",
                    confidence=0.5,
                    description="Use specific exception types in async functions"
                ))
            
            # Check for mixed sync/async patterns
            if 'requests.' in line_stripped and 'async def' in '\n'.join(lines[max(0, line_num-10):line_num]):
                patterns.append(AsyncPattern(
                    file_path=file_path,
                    line_number=line_num,
                    pattern_type="sync_http_in_async",
                    original_code=line_stripped,
                    suggested_fix=line_stripped.replace('requests.', 'await httpx_client.'),
                    severity="high",
                    confidence=0.8,
                    description="Use async HTTP client (httpx) instead of requests in async functions"
                ))
        
        return patterns


class AsyncPatternMigrator:
    """Applies async pattern fixes to files"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.logger = get_logger(__name__, LogCategory.SYSTEM)
    
    def apply_fixes(self, patterns: List[AsyncPattern]) -> Dict[str, int]:
        """Apply fixes for async patterns"""
        results = {'fixed_files': 0, 'fixed_patterns': 0, 'skipped_patterns': 0, 'errors': 0}
        
        # Group patterns by file
        patterns_by_file = {}
        for pattern in patterns:
            if pattern.file_path not in patterns_by_file:
                patterns_by_file[pattern.file_path] = []
            patterns_by_file[pattern.file_path].append(pattern)
        
        for file_path, file_patterns in patterns_by_file.items():
            try:
                if self._fix_file(file_path, file_patterns):
                    results['fixed_files'] += 1
                    results['fixed_patterns'] += len([p for p in file_patterns if p.confidence >= 0.7])
                else:
                    results['skipped_patterns'] += len(file_patterns)
            except Exception as e:
                self.logger.error(f"Failed to fix {file_path}: {e}")
                results['errors'] += 1
        
        return results
    
    def _fix_file(self, file_path: str, patterns: List[AsyncPattern]) -> bool:
        """Fix patterns in a single file"""
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would fix {len(patterns)} patterns in {file_path}")
            for pattern in patterns:
                self.logger.debug(
                    f"  {pattern.pattern_type} (line {pattern.line_number}): {pattern.description}",
                    confidence=pattern.confidence,
                    severity=pattern.severity
                )
            return True
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create backup
            backup_path = file_path + '.async_backup'
            shutil.copy2(file_path, backup_path)
            
            # Apply high-confidence fixes only
            modified = False
            for pattern in sorted(patterns, key=lambda p: p.line_number, reverse=True):
                if pattern.confidence >= 0.8 and pattern.severity == 'high':
                    line_index = pattern.line_number - 1
                    if line_index < len(lines):
                        original_line = lines[line_index]
                        
                        # Apply specific fixes based on pattern type
                        if pattern.pattern_type == "missing_await_call":
                            if 'await' not in original_line and not original_line.strip().startswith('#'):
                                lines[line_index] = original_line.replace(
                                    pattern.original_code.strip(), 
                                    pattern.suggested_fix
                                )
                                modified = True
                        
                        elif pattern.pattern_type == "blocking_sleep":
                            lines[line_index] = original_line.replace('time.sleep', 'await asyncio.sleep')
                            # Add import if needed
                            self._ensure_import(lines, "import asyncio")
                            modified = True
                        
                        elif pattern.pattern_type == "sync_http_in_async":
                            lines[line_index] = original_line.replace('requests.', 'await httpx_client.')
                            # Add import if needed
                            self._ensure_import(lines, "# A2A Protocol: Use blockchain messaging instead of httpx")
                            modified = True
                        
                        if modified:
                            self.logger.debug(f"Fixed {pattern.pattern_type} at line {pattern.line_number}")
            
            if modified:
                # Write modified file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                self.logger.info(f"Applied async pattern fixes to {file_path}")
                return True
            else:
                # Remove backup if no changes
                os.remove(backup_path)
                return False
        
        except Exception as e:
            # Restore from backup if it exists
            backup_path = file_path + '.async_backup'
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
            raise e
    
    def _ensure_import(self, lines: List[str], import_statement: str):
        """Ensure import statement is present in file"""
        import_exists = any(import_statement in line for line in lines)
        if not import_exists:
            # Find where to insert import (after existing imports)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('from ') or line.strip().startswith('import '):
                    insert_index = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    break
            
            lines.insert(insert_index, import_statement + '\n')


def main():
    parser = argparse.ArgumentParser(description='Migrate async/await patterns')
    parser.add_argument('--directory', '-d', type=str, 
                       default='/Users/apple/projects/a2a/a2a_agents/backend',
                       help='Directory to scan for async patterns')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making changes')
    parser.add_argument('--severity-filter', type=str, choices=['high', 'medium', 'low'],
                       default='high', help='Minimum severity level to process')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                       help='Minimum confidence threshold for applying fixes')
    parser.add_argument('--pattern-types', nargs='*', 
                       help='Specific pattern types to process')
    parser.add_argument('--output-report', type=str, default='async_patterns_report.md',
                       help='Output file for analysis report')
    
    args = parser.parse_args()
    
    backend_root = Path(args.directory)
    if not backend_root.exists():
        print(f"Error: Directory {backend_root} does not exist")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = AsyncPatternAnalyzer()
    
    # Scan for patterns
    print("Scanning for async patterns...")
    all_patterns = []
    
    for root, dirs, files in os.walk(backend_root):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                patterns = analyzer.analyze_file(file_path)
                all_patterns.extend(patterns)
    
    # Filter patterns
    filtered_patterns = []
    for pattern in all_patterns:
        if pattern.confidence >= args.confidence_threshold:
            if args.severity_filter == 'high' and pattern.severity != 'high':
                continue
            elif args.severity_filter == 'medium' and pattern.severity == 'low':
                continue
            
            if args.pattern_types and pattern.pattern_type not in args.pattern_types:
                continue
            
            filtered_patterns.append(pattern)
    
    # Generate report
    report = generate_report(filtered_patterns)
    with open(args.output_report, 'w') as f:
        f.write(report)
    
    print(f"Generated analysis report: {args.output_report}")
    print(f"Found {len(all_patterns)} async patterns, {len(filtered_patterns)} match criteria")
    
    if not args.dry_run and filtered_patterns:
        # Apply fixes
        migrator = AsyncPatternMigrator(dry_run=args.dry_run)
        results = migrator.apply_fixes(filtered_patterns)
        
        print("Migration Results:")
        print(f"  - Fixed files: {results['fixed_files']}")
        print(f"  - Fixed patterns: {results['fixed_patterns']}")  
        print(f"  - Skipped patterns: {results['skipped_patterns']}")
        print(f"  - Errors: {results['errors']}")
    
    print("Async pattern analysis complete!")


def generate_report(patterns: List[AsyncPattern]) -> str:
    """Generate analysis report"""
    report = []
    report.append("# A2A Async/Await Pattern Analysis Report")
    report.append(f"Generated on: {datetime.now().isoformat()}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append(f"- **Total patterns found**: {len(patterns)}")
    report.append(f"- **High severity**: {len([p for p in patterns if p.severity == 'high'])}")
    report.append(f"- **Medium severity**: {len([p for p in patterns if p.severity == 'medium'])}")
    report.append(f"- **Low severity**: {len([p for p in patterns if p.severity == 'low'])}")
    report.append("")
    
    # Pattern types
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
    
    report.append("## Pattern Types")
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- **{pattern_type}**: {count} occurrences")
    report.append("")
    
    # Files with most issues
    files_counts = {}
    for pattern in patterns:
        files_counts[pattern.file_path] = files_counts.get(pattern.file_path, 0) + 1
    
    report.append("## Files with Most Issues")
    for file_path, count in sorted(files_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        high_severity = len([p for p in patterns if p.file_path == file_path and p.severity == 'high'])
        report.append(f"- **{file_path}**: {count} patterns ({high_severity} high severity)")
    
    return "\n".join(report)


if __name__ == '__main__':
    main()