#!/usr/bin/env python3
"""
A2A JSDoc Documentation Migration Script
Systematically adds JSDoc documentation to JavaScript and TypeScript files
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.logging_config import get_logger, LogCategory


@dataclass
class JSDocPattern:
    """Represents a code pattern that needs JSDoc documentation"""
    file_path: str
    line_number: int
    pattern_type: str  # 'class', 'method', 'function', 'module'
    name: str
    current_doc: Optional[str]
    suggested_doc: str
    priority: str  # 'high', 'medium', 'low'


class JSDocAnalyzer:
    """Analyzes JavaScript files for missing JSDoc documentation"""
    
    def __init__(self):
        self.logger = get_logger(__name__, LogCategory.SYSTEM)
        
        # Regex patterns for JavaScript/TypeScript constructs
        self.patterns = {
            'module': re.compile(r'^sap\.ui\.define\s*\(', re.MULTILINE),
            'controller_extend': re.compile(r'Controller\.extend\s*\(\s*["\']([^"\']+)["\']'),
            'method': re.compile(r'^\s*(\w+)\s*:\s*function\s*\(([^)]*)\)\s*\{', re.MULTILINE),
            'arrow_method': re.compile(r'^\s*(\w+)\s*:\s*\(([^)]*)\)\s*=>\s*\{', re.MULTILINE),
            'class_method': re.compile(r'^\s*(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE),
            'event_handler': re.compile(r'^\s*(on\w+)\s*:\s*function\s*\(([^)]*)\)', re.MULTILINE),
            'private_method': re.compile(r'^\s*(_\w+)\s*:\s*function\s*\(([^)]*)\)', re.MULTILINE),
            'export_function': re.compile(r'^export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE),
            'const_function': re.compile(r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()', re.MULTILINE)
        }
        
        # Common parameter patterns
        self.param_patterns = {
            'event': 'oEvent',
            'model': 'oModel', 
            'view': 'oView',
            'controller': 'oController',
            'data': 'oData',
            'context': 'oContext',
            'binding': 'oBinding'
        }
    
    def analyze_file(self, file_path: Path) -> List[JSDocPattern]:
        """Analyze a JavaScript file for missing JSDoc"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for module-level documentation
            if not self._has_file_header(content):
                patterns.append(self._create_file_header_pattern(file_path, content))
            
            # Check for class/controller documentation
            controller_match = self.patterns['controller_extend'].search(content)
            if controller_match and not self._has_class_jsdoc(content, controller_match):
                patterns.append(self._create_class_pattern(file_path, controller_match, lines))
            
            # Check methods
            for pattern_name, pattern in [('method', self.patterns['method']), 
                                         ('event_handler', self.patterns['event_handler']),
                                         ('private_method', self.patterns['private_method'])]:
                for match in pattern.finditer(content):
                    if not self._has_method_jsdoc(content, match):
                        patterns.append(self._create_method_pattern(
                            file_path, match, lines, pattern_name
                        ))
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
        
        return patterns
    
    def _has_file_header(self, content: str) -> bool:
        """Check if file has JSDoc header"""
        return content.strip().startswith('/**') and '@fileoverview' in content[:500]
    
    def _has_class_jsdoc(self, content: str, match) -> bool:
        """Check if class/controller has JSDoc"""
        pos = match.start()
        # Look backwards for JSDoc
        preceding = content[:pos].strip()
        return preceding.endswith('*/') and '/**' in preceding[-500:]
    
    def _has_method_jsdoc(self, content: str, match) -> bool:
        """Check if method has JSDoc"""
        pos = match.start()
        # Get preceding lines
        preceding_lines = content[:pos].split('\n')[-5:]
        preceding_text = '\n'.join(preceding_lines)
        return '*/' in preceding_text and '/**' in preceding_text
    
    def _create_file_header_pattern(self, file_path: Path, content: str) -> JSDocPattern:
        """Create pattern for missing file header"""
        module_name = self._extract_module_name(file_path)
        requires = self._extract_requires(content)
        
        suggested_doc = f'''/**
 * @fileoverview {self._generate_file_description(file_path)}
 * @module {module_name}'''
        
        for req in requires[:3]:  # First 3 dependencies
            suggested_doc += f'\n * @requires {req}'
        
        suggested_doc += '\n */'
        
        return JSDocPattern(
            file_path=str(file_path),
            line_number=1,
            pattern_type='module',
            name=module_name,
            current_doc=None,
            suggested_doc=suggested_doc,
            priority='high'
        )
    
    def _create_class_pattern(self, file_path: Path, match, lines: List[str]) -> JSDocPattern:
        """Create pattern for missing class documentation"""
        class_name = match.group(1)
        line_num = self._find_line_number(lines, match.group(0))
        
        suggested_doc = f'''/**
 * {self._generate_class_description(class_name)}
 * @class
 * @alias {class_name}
 * @extends sap.ui.core.mvc.Controller
 * @description {self._generate_detailed_description(class_name)}
 * @author A2A Development Team
 * @version 1.0.0
 * @public
 */'''
        
        return JSDocPattern(
            file_path=str(file_path),
            line_number=line_num,
            pattern_type='class',
            name=class_name,
            current_doc=None,
            suggested_doc=suggested_doc,
            priority='high'
        )
    
    def _create_method_pattern(
        self, 
        file_path: Path, 
        match, 
        lines: List[str], 
        pattern_type: str
    ) -> JSDocPattern:
        """Create pattern for missing method documentation"""
        method_name = match.group(1)
        params = match.group(2).strip()
        line_num = self._find_line_number(lines, match.group(0))
        
        # Determine method characteristics
        is_private = method_name.startswith('_')
        is_event_handler = method_name.startswith('on')
        is_async = 'async' in match.group(0)
        
        suggested_doc = f'''/**
 * {self._generate_method_description(method_name, is_event_handler)}
 * @memberof {self._extract_class_name(file_path)}
 * @function {method_name}
 * {"@private" if is_private else "@public"}'''
        
        # Add parameter documentation
        if params:
            for param in params.split(','):
                param_name = param.strip()
                if param_name:
                    param_type = self._infer_param_type(param_name)
                    param_desc = self._generate_param_description(param_name)
                    suggested_doc += f'\n * @param {{{param_type}}} {param_name} - {param_desc}'
        
        # Add return type
        return_type = 'Promise<void>' if is_async else 'void'
        suggested_doc += f'\n * @returns {{{return_type}}}'
        
        # Add common annotations
        if is_event_handler:
            suggested_doc += '\n * @listens sap.ui.base.Event'
        
        suggested_doc += f'\n * @description {self._generate_detailed_method_description(method_name)}'
        suggested_doc += '\n */'
        
        return JSDocPattern(
            file_path=str(file_path),
            line_number=line_num,
            pattern_type='method',
            name=method_name,
            current_doc=None,
            suggested_doc=suggested_doc,
            priority='medium' if is_private else 'high'
        )
    
    def _extract_module_name(self, file_path: Path) -> str:
        """Extract module name from file path"""
        # Convert file path to module format
        parts = file_path.parts
        if 'static' in parts:
            idx = parts.index('static')
            module_parts = parts[idx+1:]
            module_name = '/'.join(module_parts).replace('.js', '').replace('.ts', '')
            return f'a2a/portal/{module_name}'
        return 'unknown/module'
    
    def _extract_requires(self, content: str) -> List[str]:
        """Extract required modules from sap.ui.define"""
        define_match = re.search(r'sap\.ui\.define\s*\(\s*\[([^\]]+)\]', content, re.DOTALL)
        if define_match:
            requires = define_match.group(1)
            # Extract module names
            modules = re.findall(r'["\']([\w/\.]+)["\']', requires)
            return modules
        return []
    
    def _extract_class_name(self, file_path: Path) -> str:
        """Extract class name from file path"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        match = self.patterns['controller_extend'].search(content)
        if match:
            return match.group(1)
        
        # Fallback to filename
        return file_path.stem
    
    def _find_line_number(self, lines: List[str], text: str) -> int:
        """Find line number for given text"""
        for i, line in enumerate(lines, 1):
            if text[:50] in line:
                return i
        return 1
    
    def _generate_file_description(self, file_path: Path) -> str:
        """Generate file description based on path"""
        filename = file_path.stem
        
        descriptions = {
            'AgentBuilder': 'Agent Builder Controller for A2A Developer Portal',
            'ProjectDetail': 'Project Detail View Controller',
            'BPMNDesigner': 'BPMN Workflow Designer Controller',
            'Dashboard': 'Main Dashboard Controller',
            'NotificationService': 'Notification Management Service',
            'NavigationService': 'Application Navigation Service',
            'SecurityService': 'Security and Authentication Service'
        }
        
        return descriptions.get(filename, f'{filename} module')
    
    def _generate_class_description(self, class_name: str) -> str:
        """Generate class description"""
        # Extract meaningful name
        parts = class_name.split('.')
        name = parts[-1] if parts else class_name
        
        # Convert to readable format
        readable = re.sub(r'([A-Z])', r' \1', name).strip()
        return readable + ' Controller'
    
    def _generate_detailed_description(self, class_name: str) -> str:
        """Generate detailed class description"""
        descriptions = {
            'AgentBuilder': 'Manages the agent creation and configuration interface',
            'ProjectDetail': 'Handles project details display and management',
            'BPMNDesigner': 'Provides BPMN workflow design capabilities',
            'Dashboard': 'Controls the main application dashboard'
        }
        
        name = class_name.split('.')[-1]
        return descriptions.get(name, f'Manages {name} functionality')
    
    def _generate_method_description(self, method_name: str, is_event: bool) -> str:
        """Generate method description"""
        if is_event:
            # Remove 'on' prefix and convert
            action = method_name[2:]
            readable = re.sub(r'([A-Z])', r' \1', action).strip().lower()
            return f'Handles {readable} event'
        
        # Convert camelCase to readable
        readable = re.sub(r'([A-Z])', r' \1', method_name).strip()
        return readable.capitalize()
    
    def _infer_param_type(self, param_name: str) -> str:
        """Infer parameter type from name"""
        if param_name.startswith('o'):
            # SAP UI5 convention
            if 'Event' in param_name:
                return 'sap.ui.base.Event'
            elif 'Model' in param_name:
                return 'sap.ui.model.json.JSONModel'
            elif 'View' in param_name:
                return 'sap.ui.core.mvc.View'
            return 'Object'
        elif param_name.startswith('s'):
            return 'string'
        elif param_name.startswith('n'):
            return 'number'
        elif param_name.startswith('b'):
            return 'boolean'
        elif param_name.startswith('a'):
            return 'Array'
        return 'any'
    
    def _generate_param_description(self, param_name: str) -> str:
        """Generate parameter description"""
        common_params = {
            'oEvent': 'The event object',
            'oModel': 'The data model',
            'oView': 'The view instance',
            'oData': 'The data object',
            'sId': 'The identifier',
            'sValue': 'The value',
            'bEnabled': 'Whether enabled'
        }
        
        return common_params.get(param_name, f'The {param_name}')
    
    def _generate_detailed_method_description(self, method_name: str) -> str:
        """Generate detailed method description"""
        if method_name.startswith('_load'):
            return 'Loads data from the backend service'
        elif method_name.startswith('_save'):
            return 'Saves data to the backend service'
        elif method_name.startswith('on'):
            action = method_name[2:]
            return f'Responds to user {action} action and updates the UI accordingly'
        elif method_name.startswith('_'):
            return 'Internal helper method'
        else:
            return 'Performs the specified operation'


class JSDocMigrator:
    """Applies JSDoc documentation to JavaScript files"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.logger = get_logger(__name__, LogCategory.SYSTEM)
    
    def apply_documentation(self, patterns: List[JSDocPattern]) -> Dict[str, int]:
        """Apply JSDoc documentation to files"""
        results = {'documented_files': 0, 'documented_items': 0, 'skipped': 0, 'errors': 0}
        
        # Group patterns by file
        patterns_by_file = {}
        for pattern in patterns:
            if pattern.file_path not in patterns_by_file:
                patterns_by_file[pattern.file_path] = []
            patterns_by_file[pattern.file_path].append(pattern)
        
        for file_path, file_patterns in patterns_by_file.items():
            try:
                if self._document_file(file_path, file_patterns):
                    results['documented_files'] += 1
                    results['documented_items'] += len(file_patterns)
                else:
                    results['skipped'] += len(file_patterns)
            except Exception as e:
                self.logger.error(f"Failed to document {file_path}: {e}")
                results['errors'] += 1
        
        return results
    
    def _document_file(self, file_path: str, patterns: List[JSDocPattern]) -> bool:
        """Add JSDoc to a single file"""
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would add {len(patterns)} JSDoc blocks to {file_path}")
            for pattern in patterns:
                self.logger.debug(f"  {pattern.pattern_type}: {pattern.name} (line {pattern.line_number})")
            return True
        
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Create backup
            backup_path = file_path + '.jsdoc_backup'
            shutil.copy2(file_path, backup_path)
            
            # Sort patterns by line number (reverse order to preserve line numbers)
            sorted_patterns = sorted(patterns, key=lambda p: p.line_number, reverse=True)
            
            # Apply documentation
            modified = False
            for pattern in sorted_patterns:
                if pattern.pattern_type == 'module' and pattern.line_number == 1:
                    # Add file header at the beginning
                    lines.insert(0, pattern.suggested_doc + '\n\n')
                    modified = True
                else:
                    # Add before the target line
                    line_index = pattern.line_number - 1
                    if line_index >= 0 and line_index < len(lines):
                        # Calculate indentation
                        indent = self._get_indentation(lines[line_index])
                        
                        # Add JSDoc with proper indentation
                        jsdoc_lines = pattern.suggested_doc.split('\n')
                        indented_jsdoc = '\n'.join(indent + line for line in jsdoc_lines)
                        
                        lines.insert(line_index, indented_jsdoc + '\n')
                        modified = True
            
            if modified:
                # Write modified file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                self.logger.info(f"Added JSDoc documentation to {file_path}")
                return True
            else:
                # Remove backup if no changes
                os.remove(backup_path)
                return False
        
        except Exception as e:
            # Restore from backup if it exists
            backup_path = file_path + '.jsdoc_backup'
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
            raise e
    
    def _get_indentation(self, line: str) -> str:
        """Get indentation from a line"""
        stripped = line.lstrip()
        return line[:len(line) - len(stripped)]


def main():
    parser = argparse.ArgumentParser(description='Add JSDoc documentation to JavaScript files')
    parser.add_argument('--directory', '-d', type=str, 
                       default='/Users/apple/projects/a2a/a2a_agents/backend/app/a2a/developer_portal/static',
                       help='Directory to scan for JavaScript files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be documented without making changes')
    parser.add_argument('--priority-filter', type=str, choices=['high', 'medium', 'low'],
                       help='Only process patterns of specified priority or higher')
    parser.add_argument('--pattern-types', nargs='*', 
                       choices=['module', 'class', 'method', 'function'],
                       help='Specific pattern types to process')
    parser.add_argument('--output-report', type=str, default='jsdoc_migration_report.md',
                       help='Output file for analysis report')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = JSDocAnalyzer()
    
    # Scan for patterns
    print("Scanning for missing JSDoc documentation...")
    all_patterns = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', 'node_modules', 'dist', 'build', 'lib', 'thirdparty'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.js') or file.endswith('.ts'):
                file_path = Path(root) / file
                patterns = analyzer.analyze_file(file_path)
                all_patterns.extend(patterns)
    
    # Filter patterns
    filtered_patterns = []
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    
    for pattern in all_patterns:
        if args.priority_filter:
            if priority_order[pattern.priority] < priority_order[args.priority_filter]:
                continue
        
        if args.pattern_types and pattern.pattern_type not in args.pattern_types:
            continue
        
        filtered_patterns.append(pattern)
    
    # Generate report
    report = generate_report(filtered_patterns)
    with open(args.output_report, 'w') as f:
        f.write(report)
    
    print(f"Generated JSDoc analysis report: {args.output_report}")
    print(f"Found {len(all_patterns)} documentation opportunities, {len(filtered_patterns)} match criteria")
    
    if not args.dry_run and filtered_patterns:
        # Apply documentation
        migrator = JSDocMigrator(dry_run=args.dry_run)
        results = migrator.apply_documentation(filtered_patterns)
        
        print("Documentation Results:")
        print(f"  - Documented files: {results['documented_files']}")
        print(f"  - Documented items: {results['documented_items']}")
        print(f"  - Skipped items: {results['skipped']}")
        print(f"  - Errors: {results['errors']}")
    
    print("JSDoc migration complete!")


def generate_report(patterns: List[JSDocPattern]) -> str:
    """Generate analysis report"""
    report = []
    report.append("# JSDoc Documentation Migration Report")
    report.append(f"Generated on: {datetime.now().isoformat()}")
    report.append("")
    
    # Summary
    report.append("## Summary")
    report.append(f"- **Total documentation needed**: {len(patterns)}")
    report.append(f"- **High priority**: {len([p for p in patterns if p.priority == 'high'])}")
    report.append(f"- **Medium priority**: {len([p for p in patterns if p.priority == 'medium'])}")
    report.append(f"- **Low priority**: {len([p for p in patterns if p.priority == 'low'])}")
    report.append("")
    
    # Pattern types
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
    
    report.append("## Pattern Types")
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"- **{pattern_type}**: {count} occurrences")
    report.append("")
    
    # Files needing documentation
    files_counts = {}
    for pattern in patterns:
        files_counts[pattern.file_path] = files_counts.get(pattern.file_path, 0) + 1
    
    report.append("## Files Needing Documentation")
    for file_path, count in sorted(files_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        high_priority = len([p for p in patterns if p.file_path == file_path and p.priority == 'high'])
        file_name = os.path.basename(file_path)
        report.append(f"- **{file_name}**: {count} items ({high_priority} high priority)")
    
    return "\n".join(report)


if __name__ == '__main__':
    from datetime import datetime
    main()