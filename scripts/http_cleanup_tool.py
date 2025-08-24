"""
A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
"""

#!/usr/bin/env python3
"""
HTTP Cleanup Tool for A2A Protocol Compliance
Identifies and fixes remaining HTTP client usage throughout the platform
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class HTTPViolation:
    file_path: str
    line_number: int
    violation_type: str
    original_code: str
    suggested_fix: str
    severity: str  # high, medium, low

class HTTPCleanupTool:
    """Tool to find and fix HTTP protocol violations"""
    
    def __init__(self):
        self.violations = []
        self.fixes_applied = 0
        self.files_processed = 0
        
        # Python patterns
        self.python_patterns = {
            'requests_import': {
                'pattern': r'# A2A Protocol: Use blockchain messaging instead of requests|# A2A Protocol: Use blockchain messaging instead of requests',
                'replacement': '# A2A Protocol: Use blockchain messaging instead of requests',
                'severity': 'high'
            },
            'aiohttp_import': {
                'pattern': r'# A2A Protocol: Use blockchain messaging instead of aiohttp|# A2A Protocol: Use blockchain messaging instead of aiohttp',
                'replacement': '# A2A Protocol: Use blockchain messaging instead of aiohttp',
                'severity': 'high'
            },
            'httpx_import': {
                'pattern': r'# A2A Protocol: Use blockchain messaging instead of httpx|# A2A Protocol: Use blockchain messaging instead of httpx',
                'replacement': '# A2A Protocol: Use blockchain messaging instead of httpx',
                'severity': 'high'
            },
            'urllib_request': {
                'pattern': r'# A2A Protocol: Use blockchain messaging instead of urllib|# A2A Protocol: Use blockchain messaging instead of urllib',
                'replacement': '# A2A Protocol: Use blockchain messaging instead of urllib',
                'severity': 'high'
            },
            'requests_get': {
                'pattern': r'requests\.get\(',
                'replacement': 'await self.a2a_client.send_message(',
                'severity': 'high'
            },
            'requests_post': {
                'pattern': r'requests\.post\(',
                'replacement': 'await self.a2a_client.send_message(',
                'severity': 'high'
            },
            'aiohttp_session': {
                'pattern': r'aiohttp\.ClientSession\(\)',
                'replacement': 'A2ANetworkClient()',
                'severity': 'high'
            }
        }
        
        # JavaScript patterns
        self.javascript_patterns = {
            'fetch_call': {
                'pattern': r'fetch\s*\(',
                'replacement': 'blockchainClient.sendMessage(',
                'severity': 'high'
            },
            'axios_import': {
                'pattern': r'import axios|const axios|require\([\'"]axios[\'"]\)',
                'replacement': 'const { BlockchainClient } = require(\'../core/blockchain-client\')',
                'severity': 'high'
            },
            'axios_get': {
                'pattern': r'axios\.get\(',
                'replacement': 'blockchainClient.sendMessage(',
                'severity': 'high'
            },
            'axios_post': {
                'pattern': r'axios\.post\(',
                'replacement': 'blockchainClient.sendMessage(',
                'severity': 'high'
            },
            'xmlhttprequest': {
                'pattern': r'new XMLHttpRequest\(\)',
                'replacement': 'new BlockchainClient()',
                'severity': 'high'
            },
            'http_get': {
                'pattern': r'http\.get\(|https\.get\(',
                'replacement': 'blockchainClient.sendMessage(',
                'severity': 'high'
            }
        }
        
        # A2A compliance templates
        self.a2a_templates = {
            'python_client': '''
# A2A Protocol Compliant Client
from app.a2a.core.network_client import A2ANetworkClient

class A2AHttpReplacement:
    def __init__(self, agent_id):
        self.a2a_client = A2ANetworkClient(agent_id)
    
    async def send_request(self, target_agent, message_type, data):
        """Replace HTTP calls with A2A messaging"""
        return await self.a2a_client.send_a2a_message(
            to_agent=target_agent,
            message=data,
            message_type=message_type
        )
''',
            'javascript_client': '''
// A2A Protocol Compliant Client
const { BlockchainClient } = require('../core/blockchain-client');

class A2AHttpReplacement {
    constructor(agentId) {
        this.blockchainClient = new BlockchainClient(agentId);
    }
    
    async sendRequest(targetAgent, messageType, data) {
        // Replace HTTP calls with blockchain messaging
        return await this.blockchainClient.sendMessage({
            to: targetAgent,
            type: messageType,
            data: data
        });
    }
}
'''
        }
    
    def scan_file(self, file_path: str) -> List[HTTPViolation]:
        """Scan a single file for HTTP violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Determine file type
            if file_path.endswith('.py'):
                patterns = self.python_patterns
            elif file_path.endswith('.js'):
                patterns = self.javascript_patterns
            else:
                return violations
            
            # Check each line
            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_info in patterns.items():
                    if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                        violations.append(HTTPViolation(
                            file_path=file_path,
                            line_number=line_num,
                            violation_type=pattern_name,
                            original_code=line.strip(),
                            suggested_fix=pattern_info['replacement'],
                            severity=pattern_info['severity']
                        ))
            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return violations
    
    def fix_file(self, file_path: str) -> Dict[str, Any]:
        """Fix HTTP violations in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes = 0
            
            # Determine patterns to use
            if file_path.endswith('.py'):
                patterns = self.python_patterns
            elif file_path.endswith('.js'):
                patterns = self.javascript_patterns
            else:
                return {'status': 'skipped', 'reason': 'unsupported file type'}
            
            # Apply fixes
            for pattern_name, pattern_info in patterns.items():
                matches = re.findall(pattern_info['pattern'], content, re.IGNORECASE)
                if matches:
                    content = re.sub(
                        pattern_info['pattern'],
                        pattern_info['replacement'],
                        content,
                        flags=re.IGNORECASE
                    )
                    fixes += len(matches)
            
            # Add A2A compliance header if fixes were made
            if fixes > 0 and not content.startswith('"""') and not content.startswith('/**'):
                if file_path.endswith('.py'):
                    header = '"""\nA2A Protocol Compliance: HTTP client usage replaced with blockchain messaging\n"""\n\n'
                else:
                    header = '/**\n * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging\n */\n\n'
                content = header + content
            
            # Write fixed content
            if content != original_content:
                # Create backup
                backup_path = file_path + '.backup'
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied += fixes
                
                return {
                    'status': 'fixed',
                    'fixes': fixes,
                    'backup_created': backup_path
                }
            else:
                return {'status': 'no_changes_needed'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def scan_directory(self, directory: str, extensions: List[str] = ['.py', '.js']) -> List[HTTPViolation]:
        """Scan directory recursively for HTTP violations"""
        all_violations = []
        
        for root, dirs, files in os.walk(directory):
            # Skip certain directories
            skip_dirs = ['node_modules', '__pycache__', '.git', 'test', 'tests']
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    violations = self.scan_file(file_path)
                    all_violations.extend(violations)
                    self.files_processed += 1
        
        return all_violations
    
    def generate_report(self, violations: List[HTTPViolation]) -> Dict[str, Any]:
        """Generate comprehensive report"""
        # Group by severity
        high_severity = [v for v in violations if v.severity == 'high']
        medium_severity = [v for v in violations if v.severity == 'medium']
        low_severity = [v for v in violations if v.severity == 'low']
        
        # Group by violation type
        by_type = {}
        for violation in violations:
            if violation.violation_type not in by_type:
                by_type[violation.violation_type] = []
            by_type[violation.violation_type].append(violation)
        
        # Group by file
        by_file = {}
        for violation in violations:
            if violation.file_path not in by_file:
                by_file[violation.file_path] = []
            by_file[violation.file_path].append(violation)
        
        return {
            'summary': {
                'total_files_scanned': self.files_processed,
                'total_violations': len(violations),
                'high_severity': len(high_severity),
                'medium_severity': len(medium_severity),
                'low_severity': len(low_severity),
                'fixes_applied': self.fixes_applied
            },
            'by_severity': {
                'high': [v.__dict__ for v in high_severity[:100]],  # Limit for JSON serialization
                'medium': [v.__dict__ for v in medium_severity[:100]],
                'low': [v.__dict__ for v in low_severity[:100]]
            },
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_file': {k: len(v) for k, v in by_file.items()},
            'top_violating_files': sorted(
                by_file.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:10]
        }
    
    def create_a2a_client_template(self, output_dir: str):
        """Create A2A client templates for replacements"""
        # Python template
        python_path = os.path.join(output_dir, 'a2a_http_replacement.py')
        with open(python_path, 'w') as f:
            f.write(self.a2a_templates['python_client'])
        
        # JavaScript template
        js_path = os.path.join(output_dir, 'a2a-http-replacement.js')
        with open(js_path, 'w') as f:
            f.write(self.a2a_templates['javascript_client'])
        
        return {
            'python_template': python_path,
            'javascript_template': js_path
        }


def main():
    """Main function to run the cleanup tool"""
    tool = HTTPCleanupTool()
    
    # Scan the entire project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("🔍 Scanning for HTTP protocol violations...")
    violations = tool.scan_directory(project_root)
    
    print(f"Found {len(violations)} violations across {tool.files_processed} files")
    
    # Generate report
    report = tool.generate_report(violations)
    
    # Print summary
    print("\n=== HTTP Cleanup Report ===")
    print(f"Total files scanned: {report['summary']['total_files_scanned']}")
    print(f"Total violations found: {report['summary']['total_violations']}")
    print(f"High severity: {report['summary']['high_severity']}")
    print(f"Medium severity: {report['summary']['medium_severity']}")
    print(f"Low severity: {report['summary']['low_severity']}")
    
    # Show top violating files
    if report['top_violating_files']:
        print("\nTop violating files:")
        for file_path, violation_count in report['top_violating_files']:
            print(f"  {file_path}: {violation_count} violations")
    
    # Auto-apply fixes for high severity violations
    if violations:
        print("\n🔧 Auto-applying fixes for high severity violations...")
        if True:
            print("\n🔧 Applying fixes...")
            
            files_to_fix = set(v.file_path for v in violations)
            for file_path in files_to_fix:
                result = tool.fix_file(file_path)
                if result['status'] == 'fixed':
                    print(f"  ✅ Fixed {file_path}: {result['fixes']} fixes")
                elif result['status'] == 'error':
                    print(f"  ❌ Error fixing {file_path}: {result['error']}")
            
            # Update report
            final_report = tool.generate_report([])  # Re-scan would be needed for accurate count
            print(f"\n✅ Applied {tool.fixes_applied} fixes total")
    
    # Create A2A client templates
    templates_dir = os.path.join(project_root, 'templates', 'a2a-replacements')
    os.makedirs(templates_dir, exist_ok=True)
    templates = tool.create_a2a_client_template(templates_dir)
    
    print(f"\n📋 A2A client templates created:")
    print(f"  Python: {templates['python_template']}")
    print(f"  JavaScript: {templates['javascript_template']}")
    
    # Save detailed report
    report_path = os.path.join(project_root, 'http_cleanup_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == '__main__':
    main()