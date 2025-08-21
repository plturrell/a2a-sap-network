#!/usr/bin/env python3
"""
Security Scan and Fix Tool for A2A Agents
Identifies and fixes security vulnerabilities in the codebase
"""

import os
import re
import json
import logging
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security vulnerability finding"""
    file_path: str
    line_number: int
    finding_type: str
    severity: str
    description: str
    code_snippet: str
    recommendation: str
    fixed: bool = False


class SecurityScanner:
    """Comprehensive security scanner for A2A codebase"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.findings: List[SecurityFinding] = []
        self.fixed_files: Set[str] = set()
        
        # Patterns to detect security issues
        self.security_patterns = {
            'hardcoded_password': {
                'pattern': r'(?i)password\s*=\s*["\']([^"\']{3,})["\']',
                'severity': 'HIGH',
                'description': 'Hardcoded password detected'
            },
            'hardcoded_api_key': {
                'pattern': r'(?i)api_?key\s*=\s*["\']([A-Za-z0-9_-]{20,})["\']',
                'severity': 'CRITICAL',
                'description': 'Hardcoded API key detected'
            },
            'hardcoded_secret': {
                'pattern': r'(?i)secret\s*=\s*["\']([^"\']{10,})["\']',
                'severity': 'HIGH',
                'description': 'Hardcoded secret detected'
            },
            'hardcoded_token': {
                'pattern': r'(?i)token\s*=\s*["\']([A-Za-z0-9_.-]{20,})["\']',
                'severity': 'HIGH',
                'description': 'Hardcoded token detected'
            },
            'hardcoded_private_key': {
                'pattern': r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----',
                'severity': 'CRITICAL',
                'description': 'Hardcoded private key detected'
            },
            'sql_injection_risk': {
                'pattern': r'(?i)(SELECT|INSERT|UPDATE|DELETE).*%s|.*\+.*["\']',
                'severity': 'HIGH',
                'description': 'Potential SQL injection vulnerability'
            },
            'command_injection_risk': {
                'pattern': r'os\.system\(.*\+|subprocess\.(call|run|Popen)\(.*\+',
                'severity': 'HIGH',
                'description': 'Potential command injection vulnerability'
            },
            'weak_random': {
                'pattern': r'random\.(randint|choice|random)\(',
                'severity': 'MEDIUM',
                'description': 'Weak random number generation for security purposes'
            },
            'debug_enabled': {
                'pattern': r'(?i)debug\s*=\s*True',
                'severity': 'MEDIUM',
                'description': 'Debug mode enabled in production'
            },
            'insecure_http': {
                'pattern': r'http://(?!localhost|127\.0\.0\.1)',
                'severity': 'MEDIUM',
                'description': 'Insecure HTTP connection'
            },
            'yaml_load_unsafe': {
                'pattern': r'yaml\.load\([^,)]*\)',
                'severity': 'HIGH',
                'description': 'Unsafe YAML loading'
            },
            'eval_function': {
                'pattern': r'\beval\s*\(',
                'severity': 'CRITICAL',
                'description': 'Dangerous eval() function usage'
            },
            'exec_function': {
                'pattern': r'\bexec\s*\(',
                'severity': 'CRITICAL',
                'description': 'Dangerous exec() function usage'
            }
        }
        
        # Environment variable patterns for fixes
        self.env_var_suggestions = {
            'password': 'PASSWORD',
            'api_key': 'API_KEY',
            'secret': 'SECRET_KEY',
            'token': 'ACCESS_TOKEN',
            'private_key': 'PRIVATE_KEY'
        }
    
    async def scan_codebase(self) -> List[SecurityFinding]:
        """Perform comprehensive security scan"""
        logger.info(f"Starting security scan of {self.base_path}")
        
        # Scan Python files
        py_files = list(self.base_path.rglob("*.py"))
        # Exclude virtual environment and external packages
        py_files = [f for f in py_files if 'venv' not in str(f) and 'site-packages' not in str(f)]
        
        total_files = len(py_files)
        logger.info(f"Scanning {total_files} Python files for security vulnerabilities")
        
        for i, file_path in enumerate(py_files, 1):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total_files} files scanned")
            
            await self._scan_file(file_path)
        
        # Scan configuration files
        config_files = list(self.base_path.rglob("*.json")) + list(self.base_path.rglob("*.yaml")) + list(self.base_path.rglob("*.yml"))
        config_files = [f for f in config_files if 'venv' not in str(f)]
        
        for config_file in config_files:
            await self._scan_config_file(config_file)
        
        logger.info(f"Security scan complete. Found {len(self.findings)} potential issues")
        return self.findings
    
    async def _scan_file(self, file_path: Path):
        """Scan a single Python file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern_name, pattern_info in self.security_patterns.items():
                pattern = re.compile(pattern_info['pattern'])
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.findall(line)
                    if matches:
                        # Skip comments and test files for some patterns
                        if line.strip().startswith('#'):
                            continue
                        
                        if 'test' in str(file_path).lower() and pattern_name in ['hardcoded_password', 'hardcoded_api_key']:
                            continue
                        
                        finding = SecurityFinding(
                            file_path=str(file_path),
                            line_number=line_num,
                            finding_type=pattern_name,
                            severity=pattern_info['severity'],
                            description=pattern_info['description'],
                            code_snippet=line.strip(),
                            recommendation=self._get_recommendation(pattern_name, line)
                        )
                        
                        self.findings.append(finding)
        
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
    
    async def _scan_config_file(self, file_path: Path):
        """Scan configuration files for sensitive data"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Look for sensitive patterns in config files
            sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
            
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                for key in sensitive_keys:
                    if key in line_lower and ':' in line and not line.strip().startswith('#'):
                        # Check if value looks like a hardcoded secret
                        value_part = line.split(':', 1)[1].strip().strip('"\'')
                        if len(value_part) > 5 and not value_part.startswith('${') and 'localhost' not in value_part:
                            finding = SecurityFinding(
                                file_path=str(file_path),
                                line_number=line_num,
                                finding_type='hardcoded_config_secret',
                                severity='HIGH',
                                description=f'Potential hardcoded {key} in configuration',
                                code_snippet=line.strip(),
                                recommendation=f'Replace with environment variable ${{{key.upper()}}}'
                            )
                            self.findings.append(finding)
                            break
        
        except Exception as e:
            logger.error(f"Error scanning config file {file_path}: {e}")
    
    def _get_recommendation(self, pattern_name: str, line: str) -> str:
        """Generate recommendation for fixing the security issue"""
        recommendations = {
            'hardcoded_password': 'Replace with os.getenv("PASSWORD") or use a secure secrets manager',
            'hardcoded_api_key': 'Replace with os.getenv("API_KEY") or use a secure secrets manager',
            'hardcoded_secret': 'Replace with os.getenv("SECRET_KEY") or use a secure secrets manager',
            'hardcoded_token': 'Replace with os.getenv("ACCESS_TOKEN") or use a secure secrets manager',
            'hardcoded_private_key': 'Store private keys in secure key management system',
            'sql_injection_risk': 'Use parameterized queries or ORM with proper escaping',
            'command_injection_risk': 'Use subprocess with list arguments and validate inputs',
            'weak_random': 'Use secrets module for cryptographic random numbers',
            'debug_enabled': 'Set debug=False in production or use environment variables',
            'insecure_http': 'Use HTTPS instead of HTTP for external connections',
            'yaml_load_unsafe': 'Use yaml.safe_load() instead of yaml.safe_load()',
            'eval_function': 'Avoid eval(). Use safer alternatives like ast.literal_eval()',
            'exec_function': 'Avoid exec(). Refactor to use safer code patterns'
        }
        
        return recommendations.get(pattern_name, 'Review and fix security vulnerability')
    
    async def fix_findings(self, auto_fix: bool = False) -> Dict[str, int]:
        """Fix security findings automatically where possible"""
        logger.info(f"Attempting to fix {len(self.findings)} security findings")
        
        fix_stats = {
            'fixed': 0,
            'manual_review_required': 0,
            'failed': 0
        }
        
        # Group findings by file for efficient fixing
        files_to_fix = {}
        for finding in self.findings:
            if finding.file_path not in files_to_fix:
                files_to_fix[finding.file_path] = []
            files_to_fix[finding.file_path].append(finding)
        
        for file_path, file_findings in files_to_fix.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                    lines = original_content.split('\n')
                
                modified = False
                
                # Sort findings by line number (descending) to avoid line number shifts
                file_findings.sort(key=lambda x: x.line_number, reverse=True)
                
                for finding in file_findings:
                    if auto_fix:
                        fixed = await self._auto_fix_finding(lines, finding)
                        if fixed:
                            finding.fixed = True
                            modified = True
                            fix_stats['fixed'] += 1
                        else:
                            fix_stats['manual_review_required'] += 1
                    else:
                        fix_stats['manual_review_required'] += 1
                
                # Write back modified file
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    self.fixed_files.add(file_path)
                    logger.info(f"Fixed security issues in {file_path}")
            
            except Exception as e:
                logger.error(f"Error fixing file {file_path}: {e}")
                fix_stats['failed'] += len(file_findings)
        
        return fix_stats
    
    async def _auto_fix_finding(self, lines: List[str], finding: SecurityFinding) -> bool:
        """Attempt to automatically fix a security finding"""
        line_idx = finding.line_number - 1
        if line_idx >= len(lines):
            return False
        
        original_line = lines[line_idx]
        
        # Auto-fix patterns
        if finding.finding_type == 'hardcoded_password':
            if 'password' in original_line.lower():
                # Replace hardcoded password with environment variable
                new_line = re.sub(
                    r'password\s*=\s*["\']([^"\']+)["\']',
                    'password = os.getenv("PASSWORD", "")',
                    original_line,
                    flags=re.IGNORECASE
                )
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        elif finding.finding_type == 'hardcoded_api_key':
            if 'api_key' in original_line.lower():
                new_line = re.sub(
                    r'api_?key\s*=\s*["\']([^"\']+)["\']',
                    'api_key = os.getenv("API_KEY", "")',
                    original_line,
                    flags=re.IGNORECASE
                )
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        elif finding.finding_type == 'weak_random':
            if 'import random' in original_line:
                lines[line_idx] = original_line.replace('import random', 'import secrets')
                return True
            elif 'random.' in original_line:
                new_line = original_line.replace('random.randint', 'secrets.randbelow')
                new_line = new_line.replace('random.choice', 'secrets.choice')
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        elif finding.finding_type == 'yaml_load_unsafe':
            if 'yaml.load(' in original_line:
                new_line = original_line.replace('yaml.load(', 'yaml.safe_load(')
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        elif finding.finding_type == 'debug_enabled':
            if 'debug = os.getenv("DEBUG", "false").lower() == "true"' in original_line:
                new_line = re.sub(
                    r'debug\s*=\s*True',
                    'debug = os.getenv("DEBUG", "false").lower() == "true"',
                    original_line,
                    flags=re.IGNORECASE
                )
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        elif finding.finding_type == 'insecure_http':
            if 'http://' in original_line and 'localhost' not in original_line:
                new_line = original_line.replace('https://', 'https://')
                if new_line != original_line:
                    lines[line_idx] = new_line
                    return True
        
        return False
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate a comprehensive security report"""
        from datetime import datetime
        
        # Count by severity
        severity_counts = {}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for finding in self.findings:
            type_counts[finding.finding_type] = type_counts.get(finding.finding_type, 0) + 1
        
        # Count fixed vs unfixed
        fixed_count = sum(1 for f in self.findings if f.fixed)
        unfixed_count = len(self.findings) - fixed_count
        
        report = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'base_path': str(self.base_path),
            'total_findings': len(self.findings),
            'fixed_findings': fixed_count,
            'unfixed_findings': unfixed_count,
            'files_modified': len(self.fixed_files),
            'severity_breakdown': severity_counts,
            'finding_types': type_counts,
            'critical_issues': [
                {
                    'file': f.file_path,
                    'line': f.line_number,
                    'type': f.finding_type,
                    'description': f.description,
                    'code': f.code_snippet,
                    'fixed': f.fixed
                }
                for f in self.findings if f.severity == 'CRITICAL'
            ],
            'high_priority_issues': [
                {
                    'file': f.file_path,
                    'line': f.line_number,
                    'type': f.finding_type,
                    'description': f.description,
                    'recommendation': f.recommendation,
                    'fixed': f.fixed
                }
                for f in self.findings if f.severity == 'HIGH' and not f.fixed
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Security report saved to {output_file}")
        
        return report


async def main():
    """Main function to run security scan and fixes"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize scanner
    base_path = "/Users/apple/projects/a2a/a2aAgents/backend"
    scanner = SecurityScanner(base_path)
    
    # Run security scan
    findings = await scanner.scan_codebase()
    
    if findings:
        print(f"\n🔍 Security Scan Results")
        print(f"📋 Total Issues Found: {len(findings)}")
        
        # Group by severity
        by_severity = {}
        for finding in findings:
            if finding.severity not in by_severity:
                by_severity[finding.severity] = []
            by_severity[finding.severity].append(finding)
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                print(f"🚨 {severity}: {len(by_severity[severity])} issues")
        
        # Auto-fix what we can
        print(f"\n🔧 Attempting automatic fixes...")
        fix_stats = await scanner.fix_findings(auto_fix=True)
        
        print(f"✅ Fixed: {fix_stats['fixed']} issues")
        print(f"⚠️  Manual Review Required: {fix_stats['manual_review_required']} issues")
        if fix_stats['failed'] > 0:
            print(f"❌ Failed: {fix_stats['failed']} issues")
        
        # Generate report
        report = scanner.generate_report('security_scan_report.json')
        
        print(f"\n📄 Detailed report saved to: security_scan_report.json")
        
        # Show critical issues that need manual attention
        critical_unfixed = [f for f in findings if f.severity == 'CRITICAL' and not f.fixed]
        if critical_unfixed:
            print(f"\n🚨 CRITICAL ISSUES REQUIRING MANUAL ATTENTION:")
            for finding in critical_unfixed[:5]:  # Show first 5
                print(f"   📁 {finding.file_path}:{finding.line_number}")
                print(f"   🔍 {finding.description}")
                print(f"   💡 {finding.recommendation}")
                print()
    
    else:
        print("✅ No security vulnerabilities found!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())