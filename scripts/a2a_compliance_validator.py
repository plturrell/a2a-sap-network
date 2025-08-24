#!/usr/bin/env python3
"""
A2A Protocol Compliance Validator
Comprehensive scanner to ensure complete A2A protocol compliance
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
import json

class A2AComplianceValidator:
    def __init__(self):
        self.violations = []
        self.compliant_files = []
        self.exclusions = [
            'node_modules',
            '.git', 
            '__pycache__',
            'venv',
            '.venv',
            'dist',
            'build',
            '.backup'
        ]
        
        # Files that should use MCP protocol (not A2A)
        self.mcp_files = [
            'mcpTransportLayer.py',
            'mcpIntraAgentExtension.py', 
            'mcpServer.py',
            'mcpClient.py',
            'service_manager.py'  # MCP service manager
        ]
        
        # Critical violations that must be fixed
        self.critical_patterns = {
            'http_imports': [
                r'import requests(?!\s*#.*A2A)',
                r'from requests',
                r'import axios(?!\s*#.*A2A)',
                r'import aiohttp(?!\s*#.*A2A)', 
                r'import httpx(?!\s*#.*A2A)',
                r'from urllib.request'
            ],
            'http_calls': [
                r'requests\.(get|post|put|delete|patch)',
                r'axios\.(get|post|put|delete|patch)',
                r'aiohttp\.(get|post|put|delete|patch)',
                r'httpx\.(get|post|put|delete|patch)',
                r'(?<!blockchain)Client\(\)\.(?:get|post|put|delete|patch)',
                r'fetch\s*\(\s*[\'"]http'
            ],
            'websocket_usage': [
                r'new WebSocket\(',
                r'ws://.*(?!.*blockchain)',
                r'wss://.*(?!.*blockchain)',
                r'WebSocketServer'
            ]
        }
        
        # Compliance indicators
        self.compliance_indicators = [
            r'BlockchainClient',
            r'A2ANetworkClient', 
            r'blockchain messaging',
            r'A2A Protocol',
            r'@a2a_handler',
            r'SecureA2AAgent'
        ]
        
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for A2A compliance"""
        result = {
            'file': str(file_path),
            'compliant': True,
            'violations': [],
            'compliance_indicators': [],
            'size': 0,
            'is_mcp_file': False
        }
        
        # Check if this is an MCP file (should use MCP protocol, not A2A)
        if any(mcp_file in file_path.name for mcp_file in self.mcp_files):
            result['is_mcp_file'] = True
            result['compliant'] = True  # MCP files are exempt from A2A compliance
            return result
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result['size'] = len(content)
            
            # Check for violations
            for category, patterns in self.critical_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        result['violations'].append({
                            'category': category,
                            'pattern': pattern,
                            'line': line_num,
                            'text': match.group(0)
                        })
                        result['compliant'] = False
            
            # Check for compliance indicators
            for pattern in self.compliance_indicators:
                if re.search(pattern, content, re.IGNORECASE):
                    result['compliance_indicators'].append(pattern)
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Recursively scan directory for A2A compliance"""
        results = {
            'summary': {
                'total_files': 0,
                'compliant_files': 0,
                'violation_files': 0,
                'total_violations': 0,
                'total_size': 0
            },
            'violations_by_category': {},
            'violations_by_file': {},
            'compliant_files': [],
            'most_violations': []
        }
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclusions]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts')):
                    file_path = Path(root) / file
                    scan_result = self.scan_file(file_path)
                    
                    results['summary']['total_files'] += 1
                    results['summary']['total_size'] += scan_result.get('size', 0)
                    
                    if scan_result['compliant']:
                        results['summary']['compliant_files'] += 1
                        results['compliant_files'].append(str(file_path))
                    else:
                        results['summary']['violation_files'] += 1
                        results['summary']['total_violations'] += len(scan_result['violations'])
                        results['violations_by_file'][str(file_path)] = scan_result['violations']
                        
                        # Group by category
                        for violation in scan_result['violations']:
                            category = violation['category']
                            if category not in results['violations_by_category']:
                                results['violations_by_category'][category] = []
                            results['violations_by_category'][category].append({
                                'file': str(file_path),
                                'line': violation['line'],
                                'text': violation['text']
                            })
        
        # Find files with most violations
        file_violation_counts = [(f, len(v)) for f, v in results['violations_by_file'].items()]
        results['most_violations'] = sorted(file_violation_counts, key=lambda x: x[1], reverse=True)[:10]
        
        return results
    
    def generate_compliance_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable compliance report"""
        report = []
        report.append("=" * 80)
        report.append("A2A PROTOCOL COMPLIANCE VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        summary = results['summary']
        compliance_rate = (summary['compliant_files'] / summary['total_files'] * 100) if summary['total_files'] > 0 else 0
        
        report.append(f"📊 SUMMARY")
        report.append(f"   Total Files Scanned: {summary['total_files']:,}")
        report.append(f"   Compliant Files: {summary['compliant_files']:,}")
        report.append(f"   Files with Violations: {summary['violation_files']:,}")
        report.append(f"   Total Violations: {summary['total_violations']:,}")
        report.append(f"   Compliance Rate: {compliance_rate:.1f}%")
        report.append(f"   Total Code Size: {summary['total_size']:,} chars")
        report.append("")
        
        if summary['violation_files'] == 0:
            report.append("🎉 CONGRATULATIONS!")
            report.append("   Your codebase is 100% A2A Protocol compliant!")
            report.append("   All HTTP calls have been successfully migrated to blockchain messaging.")
        else:
            report.append("⚠️  VIOLATIONS FOUND")
            report.append("")
            
            # Violations by category
            if results['violations_by_category']:
                report.append("📋 VIOLATIONS BY CATEGORY:")
                for category, violations in results['violations_by_category'].items():
                    report.append(f"   {category.upper()}: {len(violations)} violations")
                report.append("")
            
            # Files with most violations
            if results['most_violations']:
                report.append("🔍 FILES NEEDING ATTENTION:")
                for file_path, violation_count in results['most_violations'][:5]:
                    rel_path = file_path.replace(str(Path.cwd()), '.')
                    report.append(f"   {rel_path}: {violation_count} violations")
                report.append("")
        
        report.append("=" * 80)
        return "\\n".join(report)
    
    def save_detailed_report(self, results: Dict[str, Any], output_file: str):
        """Save detailed JSON report"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    validator = A2AComplianceValidator()
    
    # Scan core directories
    core_dirs = [
        Path('a2aAgents/backend/app'),
        Path('a2aNetwork/srv'), 
        Path('shared')
    ]
    
    print("🔍 Starting A2A Protocol Compliance Validation...")
    
    all_results = {
        'summary': {
            'total_files': 0,
            'compliant_files': 0,
            'violation_files': 0,
            'total_violations': 0,
            'total_size': 0
        },
        'violations_by_category': {},
        'violations_by_file': {},
        'compliant_files': [],
        'most_violations': []
    }
    
    for directory in core_dirs:
        if directory.exists():
            print(f"📁 Scanning {directory}...")
            results = validator.scan_directory(directory)
            
            # Merge results
            all_results['summary']['total_files'] += results['summary']['total_files']
            all_results['summary']['compliant_files'] += results['summary']['compliant_files']
            all_results['summary']['violation_files'] += results['summary']['violation_files']
            all_results['summary']['total_violations'] += results['summary']['total_violations']
            all_results['summary']['total_size'] += results['summary']['total_size']
            
            # Merge violations
            for category, violations in results['violations_by_category'].items():
                if category not in all_results['violations_by_category']:
                    all_results['violations_by_category'][category] = []
                all_results['violations_by_category'][category].extend(violations)
            
            all_results['violations_by_file'].update(results['violations_by_file'])
            all_results['compliant_files'].extend(results['compliant_files'])
        else:
            print(f"⚠️  Directory not found: {directory}")
    
    # Recalculate most violations
    file_violation_counts = [(f, len(v)) for f, v in all_results['violations_by_file'].items()]
    all_results['most_violations'] = sorted(file_violation_counts, key=lambda x: x[1], reverse=True)[:10]
    
    # Generate and display report
    report = validator.generate_compliance_report(all_results)
    print(report)
    
    # Save detailed report
    validator.save_detailed_report(all_results, 'a2a_compliance_report.json')
    print(f"📄 Detailed report saved to: a2a_compliance_report.json")

if __name__ == "__main__":
    main()