#!/usr/bin/env python3
"""
A2A Logging Migration Script
Systematically migrates all logging usage to standardized A2ALogger patterns
"""

import os
import re
import ast
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.logging_config import get_logger, LogCategory


@dataclass
class LoggingPattern:
    """Represents a logging pattern found in code"""
    file_path: str
    line_number: int
    original_code: str
    suggested_replacement: str
    pattern_type: str
    confidence: float


class LoggingMigrator:
    """Handles systematic migration of logging patterns"""
    
    def __init__(self, backend_root: Path, dry_run: bool = True):
        self.backend_root = backend_root
        self.dry_run = dry_run
        self.logger = get_logger(__name__, LogCategory.SYSTEM)
        
        # Migration patterns
        self.migration_patterns = {
            'standard_logging_import': {
                'pattern': r'^import logging$',
                'replacement': 'from app.core.logging_config import get_logger, LogCategory',
                'confidence': 0.9
            },
            'logger_creation': {
                'pattern': r'logger\s*=\s*logging\.getLogger\(__name__\)',
                'replacement': 'logger = get_logger(__name__, LogCategory.{category})',
                'confidence': 0.9
            },
            'print_startup': {
                'pattern': r'print\(f?["\'].*Starting.*["\'].*\)',
                'replacement': 'logger.info("Service starting", service_name="{service}", version="{version}")',
                'confidence': 0.8
            },
            'print_status': {
                'pattern': r'print\(f?["\'].*✅.*["\'].*\)',
                'replacement': 'logger.info("Operation completed", details="{details}")',
                'confidence': 0.7
            },
            'print_error': {
                'pattern': r'print\(f?["\'].*❌.*["\'].*\)',
                'replacement': 'logger.error("Operation failed", details="{details}")',
                'confidence': 0.8
            },
            'print_debug': {
                'pattern': r'print\(.*debug.*\)',
                'replacement': 'logger.debug("{message}")',
                'confidence': 0.6
            }
        }
        
        # Component-specific logger categories
        self.category_mapping = {
            'agent': LogCategory.AGENT,
            'skill': LogCategory.BUSINESS,
            'service': LogCategory.SYSTEM,
            'api': LogCategory.API,
            'database': LogCategory.DATABASE,
            'security': LogCategory.SECURITY,
            'performance': LogCategory.PERFORMANCE,
            'integration': LogCategory.INTEGRATION
        }
        
    def analyze_file(self, file_path: Path) -> List[LoggingPattern]:
        """Analyze a file for logging patterns"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_config in self.migration_patterns.items():
                    if re.search(pattern_config['pattern'], line, re.IGNORECASE):
                        suggested_replacement = self._generate_replacement(
                            line, pattern_name, pattern_config, file_path
                        )
                        
                        patterns.append(LoggingPattern(
                            file_path=str(file_path),
                            line_number=line_num,
                            original_code=line.strip(),
                            suggested_replacement=suggested_replacement,
                            pattern_type=pattern_name,
                            confidence=pattern_config['confidence']
                        ))
        
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}", exc_info=True)
        
        return patterns
    
    def _generate_replacement(self, line: str, pattern_name: str, pattern_config: dict, file_path: Path) -> str:
        """Generate appropriate replacement code"""
        
        if pattern_name == 'logger_creation':
            category = self._determine_category(file_path)
            return pattern_config['replacement'].format(category=category.value.upper())
        
        elif pattern_name == 'print_startup':
            # Extract service name and version if possible
            service_name = self._extract_service_name(line, file_path)
            return pattern_config['replacement'].format(
                service=service_name, 
                version="1.0"
            )
        
        elif pattern_name in ['print_status', 'print_error', 'print_debug']:
            # Extract the message content
            message = self._extract_message_from_print(line)
            return pattern_config['replacement'].format(
                message=message,
                details=message
            )
        
        return pattern_config['replacement']
    
    def _determine_category(self, file_path: Path) -> LogCategory:
        """Determine appropriate log category based on file path"""
        path_str = str(file_path).lower()
        
        if 'agent' in path_str:
            return LogCategory.AGENT
        elif 'skill' in path_str:
            return LogCategory.BUSINESS
        elif 'api' in path_str or 'endpoint' in path_str:
            return LogCategory.API
        elif 'database' in path_str or 'db' in path_str:
            return LogCategory.DATABASE
        elif 'security' in path_str or 'auth' in path_str:
            return LogCategory.SECURITY
        elif 'performance' in path_str or 'benchmark' in path_str:
            return LogCategory.PERFORMANCE
        elif 'integration' in path_str or 'sap' in path_str:
            return LogCategory.INTEGRATION
        else:
            return LogCategory.SYSTEM
    
    def _extract_service_name(self, line: str, file_path: Path) -> str:
        """Extract service name from file path or line content"""
        path_parts = file_path.parts
        
        # Look for service name in path
        for part in path_parts:
            if 'agent' in part.lower():
                return part
            elif 'manager' in part.lower():
                return part
        
        # Extract from line content
        if '{' in line and '}' in line:
            # Look for f-string variables
            match = re.search(r'\{([^}]*name[^}]*)\}', line)
            if match:
                return match.group(1)
        
        return "service"
    
    def _extract_message_from_print(self, line: str) -> str:
        """Extract message content from print statement"""
        # Remove print(), quotes, and f-string formatting
        line = re.sub(r'^[^(]*\(', '', line)  # Remove up to opening paren
        line = re.sub(r'\)[^)]*$', '', line)  # Remove closing paren to end
        line = re.sub(r'^[f]?["\']', '', line)  # Remove opening quote
        line = re.sub(r'["\']$', '', line)    # Remove closing quote
        
        # Simplify f-string variables
        line = re.sub(r'\{[^}]*\}', '{variable}', line)
        
        return line.strip()
    
    def scan_directory(self, directory: Path, extensions: Set[str] = {'.py'}) -> List[LoggingPattern]:
        """Scan directory for logging patterns"""
        all_patterns = []
        
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    patterns = self.analyze_file(file_path)
                    all_patterns.extend(patterns)
        
        return all_patterns
    
    def generate_migration_plan(self, patterns: List[LoggingPattern]) -> Dict[str, List[LoggingPattern]]:
        """Group patterns by file for migration planning"""
        migration_plan = {}
        
        for pattern in patterns:
            if pattern.file_path not in migration_plan:
                migration_plan[pattern.file_path] = []
            migration_plan[pattern.file_path].append(pattern)
        
        # Sort by confidence (high confidence first)
        for file_patterns in migration_plan.values():
            file_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return migration_plan
    
    def apply_migration(self, migration_plan: Dict[str, List[LoggingPattern]]) -> Dict[str, int]:
        """Apply migration to files"""
        results = {'migrated_files': 0, 'migrated_patterns': 0, 'skipped_files': 0, 'errors': 0}
        
        for file_path, patterns in migration_plan.items():
            try:
                if self._migrate_file(file_path, patterns):
                    results['migrated_files'] += 1
                    results['migrated_patterns'] += len(patterns)
                else:
                    results['skipped_files'] += 1
            
            except Exception as e:
                self.logger.error(f"Migration failed for {file_path}", exc_info=True)
                results['errors'] += 1
        
        return results
    
    def _migrate_file(self, file_path: str, patterns: List[LoggingPattern]) -> bool:
        """Migrate a single file"""
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would migrate {len(patterns)} patterns in {file_path}")
            for pattern in patterns:
                self.logger.debug(
                    f"Line {pattern.line_number}: {pattern.original_code} -> {pattern.suggested_replacement}",
                    pattern_type=pattern.pattern_type,
                    confidence=pattern.confidence
                )
            return True
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create backup
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)
            
            # Apply migrations (reverse order to preserve line numbers)
            modified = False
            for pattern in sorted(patterns, key=lambda p: p.line_number, reverse=True):
                if pattern.confidence >= 0.7:  # Only apply high-confidence changes
                    line_index = pattern.line_number - 1
                    if line_index < len(lines):
                        lines[line_index] = pattern.suggested_replacement + '\n'
                        modified = True
                        self.logger.debug(f"Migrated line {pattern.line_number} in {file_path}")
            
            if modified:
                # Add import if needed
                self._ensure_logging_import(lines, file_path)
                
                # Write modified file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                self.logger.info(f"Migrated {len(patterns)} patterns in {file_path}")
                return True
            
            else:
                # Remove backup if no changes made
                os.remove(backup_path)
                return False
        
        except Exception as e:
            # Restore from backup if it exists
            backup_path = file_path + '.backup'
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
            raise e
    
    def _ensure_logging_import(self, lines: List[str], file_path: str):
        """Ensure proper logging import is present"""
        has_a2a_import = any('from app.core.logging_config import' in line for line in lines)
        has_standard_import = any('import logging' in line for line in lines)
        
        if not has_a2a_import and has_standard_import:
            # Find the import logging line and replace it
            for i, line in enumerate(lines):
                if 'import logging' in line and not line.strip().startswith('#'):
                    lines[i] = 'from app.core.logging_config import get_logger, LogCategory\n'
                    break
        
        elif not has_a2a_import and not has_standard_import:
            # Add import at top after docstring
            insert_index = 0
            
            # Skip module docstring
            if lines and lines[0].strip().startswith('"""'):
                for i, line in enumerate(lines):
                    if line.strip().endswith('"""') and i > 0:
                        insert_index = i + 1
                        break
            
            lines.insert(insert_index, 'from app.core.logging_config import get_logger, LogCategory\n')
    
    def generate_report(self, patterns: List[LoggingPattern], migration_plan: Dict[str, List[LoggingPattern]]) -> str:
        """Generate migration report"""
        report = []
        report.append("# A2A Logging Migration Report")
        report.append(f"Generated on: {self.logger}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- **Total patterns found**: {len(patterns)}")
        report.append(f"- **Files affected**: {len(migration_plan)}")
        report.append(f"- **High confidence patterns**: {len([p for p in patterns if p.confidence >= 0.8])}")
        report.append(f"- **Medium confidence patterns**: {len([p for p in patterns if 0.6 <= p.confidence < 0.8])}")
        report.append(f"- **Low confidence patterns**: {len([p for p in patterns if p.confidence < 0.6])}")
        report.append("")
        
        # Pattern breakdown
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        report.append("## Pattern Types")
        for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- **{pattern_type}**: {count} occurrences")
        report.append("")
        
        # File details
        report.append("## Files to Migrate")
        for file_path, file_patterns in migration_plan.items():
            high_conf = len([p for p in file_patterns if p.confidence >= 0.8])
            report.append(f"- **{file_path}**: {len(file_patterns)} patterns ({high_conf} high confidence)")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Migrate A2A logging patterns')
    parser.add_argument('--directory', '-d', type=str, 
                       default='/Users/apple/projects/a2a/a2a_agents/backend',
                       help='Directory to scan for logging patterns')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be changed without making changes')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Minimum confidence threshold for applying changes')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report only, do not migrate')
    parser.add_argument('--output-report', type=str, default='logging_migration_report.md',
                       help='Output file for migration report')
    
    args = parser.parse_args()
    
    backend_root = Path(args.directory)
    if not backend_root.exists():
        print(f"Error: Directory {backend_root} does not exist")
        sys.exit(1)
    
    # Initialize migrator
    migrator = LoggingMigrator(backend_root, dry_run=args.dry_run or args.report_only)
    
    # Scan for patterns
    print("Scanning for logging patterns...")
    patterns = migrator.scan_directory(backend_root)
    
    # Generate migration plan
    migration_plan = migrator.generate_migration_plan(patterns)
    
    # Generate report
    report = migrator.generate_report(patterns, migration_plan)
    
    # Write report
    with open(args.output_report, 'w') as f:
        f.write(report)
    
    print(f"Generated migration report: {args.output_report}")
    print(f"Found {len(patterns)} logging patterns in {len(migration_plan)} files")
    
    if not args.report_only:
        # Apply migration
        print("Applying migration...")
        results = migrator.apply_migration(migration_plan)
        
        print("Migration Results:")
        print(f"  - Migrated files: {results['migrated_files']}")
        print(f"  - Migrated patterns: {results['migrated_patterns']}")
        print(f"  - Skipped files: {results['skipped_files']}")
        print(f"  - Errors: {results['errors']}")
    
    print("Migration analysis complete!")


if __name__ == '__main__':
    main()