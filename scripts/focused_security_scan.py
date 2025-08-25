#!/usr/bin/env python3
"""
Focused Security Scanner for A2A Platform Core Components
Scans critical areas for security vulnerabilities
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class FocusedSecurityScanner:
    def __init__(self):
        self.project_root = Path("/Users/apple/projects/a2a")
        self.vulnerabilities = []
        
        # Focus on critical directories
        self.critical_dirs = [
            "a2aAgents/backend/app/a2a/agents",
            "a2aAgents/backend/app/core", 
            "a2aAgents/backend/app/api",
            "a2aNetwork/srv",
            "a2aNetwork/app"
        ]

    def scan_hardcoded_secrets_focused(self) -> int:
        """Focused scan for hardcoded secrets in critical files"""
        print("🔍 Scanning critical areas for hardcoded secrets...")
        
        secret_patterns = [
            r'password\s*[:=]\s*["\'][^"\']{8,}["\']',
            r'api_key\s*[:=]\s*["\'][^"\']{20,}["\']',
            r'secret\s*[:=]\s*["\'][^"\']{16,}["\']',
            r'token\s*[:=]\s*["\'][^"\']{20,}["\']',
            r'private_key\s*[:=]\s*["\'][^"\']{50,}["\']',
        ]
        
        found_secrets = 0
        
        for critical_dir in self.critical_dirs:
            dir_path = self.project_root / critical_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in secret_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                # Filter out obvious placeholders
                                real_secrets = [m for m in matches if not any(placeholder in m.lower() 
                                    for placeholder in ['example', 'placeholder', 'your-', 'todo', 'xxx', 'test', 'demo'])]
                                
                                if real_secrets:
                                    found_secrets += len(real_secrets)
                                    print(f"  ⚠️ Found {len(real_secrets)} potential secrets in {py_file.relative_to(self.project_root)}")
                except Exception:
                    continue
                    
        return found_secrets

    def scan_sql_injection_patterns(self) -> int:
        """Scan for SQL injection vulnerabilities"""
        print("💉 Scanning for SQL injection patterns...")
        
        sql_patterns = [
            r'execute\s*\([^)]*%\s*[^)]*\)',
            r'query\s*\([^)]*\+[^)]*\)',
            r'SELECT.*\+.*FROM',
            r'cursor\.execute\s*\([^)]*%',
            r'\.format\s*\([^)]*SELECT'
        ]
        
        found_issues = 0
        
        for critical_dir in self.critical_dirs:
            dir_path = self.project_root / critical_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in sql_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                found_issues += 1
                                print(f"  🔍 Potential SQL injection in {py_file.relative_to(self.project_root)}")
                                break
                except Exception:
                    continue
                    
        return found_issues

    def scan_command_injection(self) -> int:
        """Scan for command injection vulnerabilities"""  
        print("🖥️ Scanning for command injection patterns...")
        
        cmd_patterns = [
            r'os\.system\s*\([^)]*\+',
            r'subprocess\.\w+\s*\([^)]*\+',
            r'exec\s*\([^)]*input',
            r'eval\s*\([^)]*input'
        ]
        
        found_issues = 0
        
        for critical_dir in self.critical_dirs:
            dir_path = self.project_root / critical_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in cmd_patterns:
                            if re.search(pattern, content):
                                found_issues += 1
                                print(f"  🚨 CRITICAL: Command injection risk in {py_file.relative_to(self.project_root)}")
                                break
                except Exception:
                    continue
                    
        return found_issues

    def scan_weak_crypto(self) -> int:
        """Scan for weak cryptography"""
        print("🔐 Scanning for weak cryptography...")
        
        crypto_patterns = [
            r'hashlib\.md5\s*\(',
            r'hashlib\.sha1\s*\(',
            r'random\.random\s*\(',
            r'DES\.|RC4\.'
        ]
        
        found_issues = 0
        
        for critical_dir in self.critical_dirs:
            dir_path = self.project_root / critical_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in crypto_patterns:
                            if re.search(pattern, content):
                                found_issues += 1
                                print(f"  🔓 Weak crypto in {py_file.relative_to(self.project_root)}")
                                break
                except Exception:
                    continue
                    
        return found_issues

    def scan_auth_bypass(self) -> int:
        """Scan for authentication bypass issues"""
        print("🔓 Scanning for authentication bypass patterns...")
        
        auth_patterns = [
            r'auth\s*=\s*False',
            r'verify\s*=\s*False',
            r'check_hostname\s*=\s*False',
            r'ssl_verify\s*=\s*False'
        ]
        
        found_issues = 0
        
        for critical_dir in self.critical_dirs:
            dir_path = self.project_root / critical_dir
            if not dir_path.exists():
                continue
                
            for py_file in dir_path.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in auth_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                found_issues += 1
                                print(f"  ⚡ Auth bypass risk in {py_file.relative_to(self.project_root)}")
                                break
                except Exception:
                    continue
                    
        return found_issues

    def check_file_security(self) -> Dict[str, int]:
        """Check critical file security"""
        print("📁 Checking critical file security...")
        
        issues = {'env_files': 0, 'config_files': 0, 'executable_configs': 0}
        
        # Check for exposed .env files
        for env_file in self.project_root.rglob("*.env*"):
            issues['env_files'] += 1
            print(f"  📄 Environment file found: {env_file.relative_to(self.project_root)}")
        
        # Check config files
        config_extensions = ['.conf', '.config', '.ini', '.properties']
        for ext in config_extensions:
            for config_file in self.project_root.rglob(f"*{ext}"):
                issues['config_files'] += 1
                
                # Check if executable
                try:
                    if config_file.stat().st_mode & 0o111:  # Has execute permission
                        issues['executable_configs'] += 1
                        print(f"  ⚠️ Executable config file: {config_file.relative_to(self.project_root)}")
                except OSError:
                    continue
                    
        return issues

    def run_focused_scan(self) -> Dict[str, Any]:
        """Run focused security scan on critical components"""
        print("🛡️ Starting focused A2A security scan...")
        print("=" * 50)
        
        results = {
            'scan_time': datetime.now().isoformat(),
            'hardcoded_secrets': self.scan_hardcoded_secrets_focused(),
            'sql_injection_risks': self.scan_sql_injection_patterns(),
            'command_injection_risks': self.scan_command_injection(),
            'weak_crypto_usage': self.scan_weak_crypto(),
            'auth_bypass_risks': self.scan_auth_bypass(),
            'file_security_issues': self.check_file_security()
        }
        
        # Calculate total issues
        total_issues = (results['hardcoded_secrets'] + results['sql_injection_risks'] + 
                       results['command_injection_risks'] + results['weak_crypto_usage'] +
                       results['auth_bypass_risks'] + sum(results['file_security_issues'].values()))
        
        results['total_security_issues'] = total_issues
        
        print("\n" + "=" * 50)
        print("🛡️ FOCUSED SECURITY SCAN COMPLETE!")
        print("=" * 50)
        print(f"📊 Total security issues found: {total_issues}")
        print(f"🔑 Hardcoded secrets: {results['hardcoded_secrets']}")
        print(f"💉 SQL injection risks: {results['sql_injection_risks']}")
        print(f"🖥️ Command injection risks: {results['command_injection_risks']}")
        print(f"🔐 Weak crypto usage: {results['weak_crypto_usage']}")
        print(f"🔓 Auth bypass risks: {results['auth_bypass_risks']}")
        print(f"📁 File security issues: {sum(results['file_security_issues'].values())}")
        
        # Security score (0-100)
        max_possible_issues = 100  # Baseline for scoring
        security_score = max(0, 100 - (total_issues * 2))  # 2 points per issue
        results['security_score'] = security_score
        
        if security_score >= 90:
            print(f"🎉 EXCELLENT security score: {security_score}/100")
        elif security_score >= 70:
            print(f"✅ GOOD security score: {security_score}/100")
        elif security_score >= 50:
            print(f"⚠️ FAIR security score: {security_score}/100 - needs improvement")
        else:
            print(f"🚨 POOR security score: {security_score}/100 - immediate attention required")
        
        print("=" * 50)
        
        # Save results
        with open(self.project_root / 'focused_security_scan_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    scanner = FocusedSecurityScanner()
    return scanner.run_focused_scan()

if __name__ == "__main__":
    main()