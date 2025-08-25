"""
Advanced Security Scanner for Glean Agent
Provides comprehensive security scanning with vulnerability detection and remediation suggestions
"""

import os
import re
import json
import hashlib
import subprocess
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict

from .base_agent import BaseAgent, A2AError, ErrorCode
from app.a2a.core.security_base import SecureA2AAgent


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    SQL_INJECTION = 'sql_injection'
    XSS = 'xss'
    CSRF = 'csrf'
    COMMAND_INJECTION = 'command_injection'
    PATH_TRAVERSAL = 'path_traversal'
    WEAK_CRYPTO = 'weak_crypto'
    HARDCODED_SECRET = 'hardcoded_secret'
    INSECURE_DESERIALIZATION = 'insecure_deserialization'
    UNSAFE_REFLECTION = 'unsafe_reflection'
    BUFFER_OVERFLOW = 'buffer_overflow'
    AUTHENTICATION_BYPASS = 'authentication_bypass'
    PRIVILEGE_ESCALATION = 'privilege_escalation'
    INFORMATION_DISCLOSURE = 'information_disclosure'
    DENIAL_OF_SERVICE = 'denial_of_service'
    INSECURE_CONFIGURATION = 'insecure_configuration'


class SeverityLevel(Enum):
    """Severity levels for vulnerabilities"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


@dataclass
class SecurityVulnerability:
    """Security vulnerability data structure"""
    id: str
    type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: Optional[List[str]] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['type'] = self.type.value
        result['severity'] = self.severity.value
        return result


@dataclass
class SecurityScanResult:
    """Security scan result"""
    file_path: str
    vulnerabilities: List[SecurityVulnerability]
    scan_duration: float
    file_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_path': self.file_path,
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'scan_duration': self.scan_duration,
            'file_hash': self.file_hash,
            'timestamp': self.timestamp,
            'vulnerability_count': len(self.vulnerabilities),
            'severity_breakdown': self._get_severity_breakdown()
        }

    def _get_severity_breakdown(self) -> Dict[str, int]:
        """Get breakdown of vulnerabilities by severity"""
        breakdown = {level.value: 0 for level in SeverityLevel}
        for vuln in self.vulnerabilities:
            breakdown[vuln.severity.value] += 1
        return breakdown


class SecurityScanner(SecureA2AAgent):
    """Advanced security scanner for code analysis"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('security-scanner', config)
        # Security features are initialized by SecureA2AAgent base class


        # Load vulnerability patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()

        # Initialize external tools
        self.external_tools = {
            'bandit': self._check_bandit_available(),
            'semgrep': self._check_semgrep_available(),
            'codeql': self._check_codeql_available()
        }

        # Scan statistics
        self.scan_stats = {
            'total_files_scanned': 0,
            'vulnerabilities_found': 0,
            'false_positives': 0,
            'scan_history': []
        }

        self.logger.info('Security Scanner initialized',
                        external_tools=self.external_tools)

    def _load_vulnerability_patterns(self) -> Dict[VulnerabilityType, List[Dict[str, Any]]]:
        """Load vulnerability detection patterns"""
        patterns = {
            VulnerabilityType.SQL_INJECTION: [
                {
                    'pattern': r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER).*?\+.*?(input|request|param)',
                    'description': 'Potential SQL injection through string concatenation',
                    'cwe': 'CWE-89',
                    'severity': SeverityLevel.HIGH
                },
                {
                    'pattern': r'(?i)execute\s*\(\s*["\'].*?\+',
                    'description': 'SQL query construction with string concatenation',
                    'cwe': 'CWE-89',
                    'severity': SeverityLevel.HIGH
                }
            ],

            VulnerabilityType.COMMAND_INJECTION: [
                {
                    'pattern': r'(?i)(subprocess\.|os\.system|os\.popen|commands\.|shell=True).*?(input|request|param)',
                    'description': 'Potential command injection through user input',
                    'cwe': 'CWE-78',
                    'severity': SeverityLevel.CRITICAL
                }
            ],

            VulnerabilityType.HARDCODED_SECRET: [
                {
                    'pattern': r'(?i)(password|passwd|pwd|secret|key|token)\s*[=:]\s*["\'][^"\']{8,}["\']',
                    'description': 'Hardcoded password or secret key',
                    'cwe': 'CWE-798',
                    'severity': SeverityLevel.HIGH
                },
                {
                    'pattern': r'(?i)(api_key|apikey|access_key|secret_key)\s*[=:]\s*["\'][A-Za-z0-9+/]{20,}["\']',
                    'description': 'Hardcoded API key or access token',
                    'cwe': 'CWE-798',
                    'severity': SeverityLevel.HIGH
                }
            ],

            VulnerabilityType.PATH_TRAVERSAL: [
                {
                    'pattern': r'(?i)(open|file|read).*?\.\./.*?\.\.',
                    'description': 'Potential path traversal vulnerability',
                    'cwe': 'CWE-22',
                    'severity': SeverityLevel.MEDIUM
                }
            ],

            VulnerabilityType.WEAK_CRYPTO: [
                {
                    'pattern': r'(?i)(md5|sha1|des|rc4)(?!\w)',
                    'description': 'Use of weak cryptographic algorithm',
                    'cwe': 'CWE-327',
                    'severity': SeverityLevel.MEDIUM
                }
            ],

            VulnerabilityType.XSS: [
                {
                    'pattern': r'(?i)(innerHTML|document\.write|eval).*?(input|request|param)',
                    'description': 'Potential XSS through dangerous DOM manipulation',
                    'cwe': 'CWE-79',
                    'severity': SeverityLevel.HIGH
                }
            ],

            VulnerabilityType.INSECURE_DESERIALIZATION: [
                {
                    'pattern': r'(?i)(pickle\.load|yaml\.load|marshal\.load)(?!\w)',
                    'description': 'Insecure deserialization function',
                    'cwe': 'CWE-502',
                    'severity': SeverityLevel.HIGH
                }
            ]
        }

        return patterns

    def _check_bandit_available(self) -> bool:
        """Check if Bandit is available"""
        try:
            subprocess.run(['bandit', '--version'],
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_semgrep_available(self) -> bool:
        """Check if Semgrep is available"""
        try:
            subprocess.run(['semgrep', '--version'],
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_codeql_available(self) -> bool:
        """Check if CodeQL is available"""
        try:
            subprocess.run(['codeql', '--version'],
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @BaseAgent.track_performance('security_scan_file')
    async def scan_file(self, file_path: str,
                       scan_options: Optional[Dict[str, Any]] = None) -> SecurityScanResult:
        """Scan a single file for security vulnerabilities"""
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise A2AError(
                    code=ErrorCode.DATA_NOT_FOUND,
                    message=f'File not found: {file_path}',
                    details={'file_path': file_path}
                )

            start_time = time.time()

            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Skip binary files
                self.logger.debug('Skipping binary file', file_path=file_path)
                return SecurityScanResult(
                    file_path=file_path,
                    vulnerabilities=[],
                    scan_duration=time.time() - start_time,
                    file_hash=file_hash,
                    timestamp=datetime.utcnow().isoformat()
                )

            vulnerabilities = []

            # Pattern-based scanning
            pattern_vulns = await self._scan_with_patterns(file_path, content)
            vulnerabilities.extend(pattern_vulns)

            # External tool scanning
            if scan_options and scan_options.get('use_external_tools', True):
                external_vulns = await self._scan_with_external_tools(file_path)
                vulnerabilities.extend(external_vulns)

            # Context-aware analysis
            context_vulns = await self._scan_with_context(file_path, content)
            vulnerabilities.extend(context_vulns)

            # Remove duplicates
            vulnerabilities = self._deduplicate_vulnerabilities(vulnerabilities)

            # Create result
            result = SecurityScanResult(
                file_path=file_path,
                vulnerabilities=vulnerabilities,
                scan_duration=time.time() - start_time,
                file_hash=file_hash,
                timestamp=datetime.utcnow().isoformat()
            )

            # Update statistics
            self.scan_stats['total_files_scanned'] += 1
            self.scan_stats['vulnerabilities_found'] += len(vulnerabilities)

            # Log scan result
            self.logger.info(
                'File security scan completed',
                file_path=file_path,
                vulnerabilities_found=len(vulnerabilities),
                scan_duration=result.scan_duration,
                severity_breakdown=result._get_severity_breakdown()
            )

            return result

        except A2AError:
            raise
        except Exception as e:
            raise self.handle_error(e, 'scan_file')

    async def _scan_with_patterns(self, file_path: str, content: str) -> List[SecurityVulnerability]:
        """Scan file using vulnerability patterns"""
        vulnerabilities = []
        lines = content.split('\n')

        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']

                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line)

                    for match in matches:
                        # Calculate confidence based on context
                        confidence = self._calculate_confidence(line, pattern_info, file_path)

                        if confidence > 0.3:  # Threshold to reduce false positives
                            vuln_id = hashlib.sha256(
                                f"{file_path}:{line_num}:{vuln_type.value}:{match.start()}".encode(),
                                usedforsecurity=False
                            ).hexdigest()[:8]

                            vulnerability = SecurityVulnerability(
                                id=vuln_id,
                                type=vuln_type,
                                severity=pattern_info['severity'],
                                title=pattern_info['description'],
                                description=self._generate_description(vuln_type, match.group()),
                                file_path=file_path,
                                line_number=line_num,
                                code_snippet=line.strip(),
                                cwe_id=pattern_info.get('cwe'),
                                confidence=confidence,
                                remediation=self._generate_remediation(vuln_type)
                            )

                            vulnerabilities.append(vulnerability)

        return vulnerabilities

    async def _scan_with_external_tools(self, file_path: str) -> List[SecurityVulnerability]:
        """Scan file using external security tools"""
        vulnerabilities = []

        # Bandit for Python files
        if self.external_tools['bandit'] and file_path.endswith('.py'):
            bandit_vulns = await self._run_bandit(file_path)
            vulnerabilities.extend(bandit_vulns)

        # Semgrep for various languages
        if self.external_tools['semgrep']:
            semgrep_vulns = await self._run_semgrep(file_path)
            vulnerabilities.extend(semgrep_vulns)

        return vulnerabilities

    async def _run_bandit(self, file_path: str) -> List[SecurityVulnerability]:
        """Run Bandit security scanner"""
        try:
            result = subprocess.run([
                'bandit', '-f', 'json', file_path
            ], capture_output=True, text=True, timeout=30)

            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = []

                for issue in data.get('results', []):
                    vuln_id = hashlib.sha256(
                        f"{file_path}:{issue['line_number']}:bandit:{issue['test_id']}".encode(),
                        usedforsecurity=False
                    ).hexdigest()[:8]

                    # Map Bandit severity to our severity levels
                    severity_map = {
                        'LOW': SeverityLevel.LOW,
                        'MEDIUM': SeverityLevel.MEDIUM,
                        'HIGH': SeverityLevel.HIGH
                    }

                    vulnerability = SecurityVulnerability(
                        id=vuln_id,
                        type=self._map_bandit_test_to_type(issue['test_id']),
                        severity=severity_map.get(issue['issue_severity'], SeverityLevel.MEDIUM),
                        title=issue['test_name'],
                        description=issue['issue_text'],
                        file_path=file_path,
                        line_number=issue['line_number'],
                        code_snippet=issue['code'],
                        cwe_id=issue.get('cwe', {}).get('id'),
                        confidence=issue['issue_confidence'] / 3.0,  # Convert HIGH/MEDIUM/LOW to 0-1 scale
                        remediation=self._get_bandit_remediation(issue['test_id'])
                    )

                    vulnerabilities.append(vulnerability)

                return vulnerabilities

        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            self.logger.warning(
                'Bandit scan failed',
                file_path=file_path,
                error=str(e)
            )

        return []

    async def _run_semgrep(self, file_path: str) -> List[SecurityVulnerability]:
        """Run Semgrep security scanner"""
        try:
            result = subprocess.run([
                'semgrep', '--json', '--config=auto', file_path
            ], capture_output=True, text=True, timeout=60)

            if result.stdout:
                data = json.loads(result.stdout)
                vulnerabilities = []

                for finding in data.get('results', []):
                    vuln_id = hashlib.sha256(
                        f"{file_path}:{finding['start']['line']}:semgrep:{finding['check_id']}".encode(),
                        usedforsecurity=False
                    ).hexdigest()[:8]

                    # Map Semgrep severity
                    severity_map = {
                        'INFO': SeverityLevel.LOW,
                        'WARNING': SeverityLevel.MEDIUM,
                        'ERROR': SeverityLevel.HIGH
                    }

                    vulnerability = SecurityVulnerability(
                        id=vuln_id,
                        type=self._map_semgrep_rule_to_type(finding['check_id']),
                        severity=severity_map.get(finding['extra']['severity'], SeverityLevel.MEDIUM),
                        title=finding['extra']['message'],
                        description=finding['extra'].get('metadata', {}).get('description', ''),
                        file_path=file_path,
                        line_number=finding['start']['line'],
                        code_snippet=finding['extra']['lines'],
                        references=finding['extra'].get('metadata', {}).get('references', []),
                        confidence=0.8  # Semgrep generally has high confidence
                    )

                    vulnerabilities.append(vulnerability)

                return vulnerabilities

        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            self.logger.warning(
                'Semgrep scan failed',
                file_path=file_path,
                error=str(e)
            )

        return []

    async def _scan_with_context(self, file_path: str, content: str) -> List[SecurityVulnerability]:
        """Perform context-aware vulnerability analysis"""
        vulnerabilities = []

        # Check for authentication bypass patterns
        if self._check_auth_bypass_patterns(content):
            vuln_id = hashlib.sha256(f"{file_path}:auth_bypass".encode(), usedforsecurity=False).hexdigest()[:8]

            vulnerability = SecurityVulnerability(
                id=vuln_id,
                type=VulnerabilityType.AUTHENTICATION_BYPASS,
                severity=SeverityLevel.CRITICAL,
                title='Potential authentication bypass',
                description='Code pattern suggests possible authentication bypass',
                file_path=file_path,
                line_number=1,  # Context-based, no specific line
                code_snippet='',
                confidence=0.6
            )

            vulnerabilities.append(vulnerability)

        return vulnerabilities

    def _check_auth_bypass_patterns(self, content: str) -> bool:
        """Check for authentication bypass patterns"""
        bypass_patterns = [
            r'(?i)if.*debug.*return\s+true',
            r'(?i)if.*test.*skip.*auth',
            r'(?i)authentication.*disabled'
        ]

        for pattern in bypass_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _calculate_confidence(self, line: str, pattern_info: Dict, file_path: str) -> float:
        """Calculate confidence score for vulnerability"""
        confidence = 0.5  # Base confidence

        # Increase confidence for certain file types
        if file_path.endswith(('.py', '.js', '.java', '.php')):
            confidence += 0.2

        # Decrease confidence for comments
        if line.strip().startswith(('#', '//', '/*', '*')):
            confidence -= 0.3

        # Increase confidence for certain keywords
        dangerous_keywords = ['input', 'request', 'param', 'user']
        for keyword in dangerous_keywords:
            if keyword in line.lower():
                confidence += 0.1
                break

        return max(0.0, min(1.0, confidence))

    def _deduplicate_vulnerabilities(self, vulnerabilities: List[SecurityVulnerability]) -> List[SecurityVulnerability]:
        """Remove duplicate vulnerabilities"""
        seen = set()
        deduplicated = []

        for vuln in vulnerabilities:
            key = (vuln.file_path, vuln.line_number, vuln.type, vuln.code_snippet[:50])
            if key not in seen:
                seen.add(key)
                deduplicated.append(vuln)

        return deduplicated

    def _generate_description(self, vuln_type: VulnerabilityType, match: str) -> str:
        """Generate detailed vulnerability description"""
        descriptions = {
            VulnerabilityType.SQL_INJECTION: f"SQL injection vulnerability detected in: {match}",
            VulnerabilityType.COMMAND_INJECTION: f"Command injection vulnerability detected in: {match}",
            VulnerabilityType.HARDCODED_SECRET: f"Hardcoded secret detected: {match[:20]}...",
            VulnerabilityType.XSS: f"Cross-site scripting vulnerability detected in: {match}",
        }

        return descriptions.get(vuln_type, f"Security vulnerability detected: {match}")

    def _generate_remediation(self, vuln_type: VulnerabilityType) -> str:
        """Generate remediation advice"""
        remediations = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries or prepared statements",
            VulnerabilityType.COMMAND_INJECTION: "Avoid executing user input as commands. Use allow-lists and input validation",
            VulnerabilityType.HARDCODED_SECRET: "Store secrets in environment variables or secure configuration",
            VulnerabilityType.XSS: "Sanitize and validate all user input before rendering",
            VulnerabilityType.WEAK_CRYPTO: "Use strong cryptographic algorithms (AES-256, SHA-256 or higher)",
            VulnerabilityType.PATH_TRAVERSAL: "Validate and sanitize file paths, use absolute paths"
        }

        return remediations.get(vuln_type, "Review and fix the security issue")

    def _map_bandit_test_to_type(self, test_id: str) -> VulnerabilityType:
        """Map Bandit test ID to vulnerability type"""
        mapping = {
            'B301': VulnerabilityType.INSECURE_DESERIALIZATION,
            'B501': VulnerabilityType.HARDCODED_SECRET,
            'B602': VulnerabilityType.COMMAND_INJECTION,
            'B608': VulnerabilityType.SQL_INJECTION
        }

        return mapping.get(test_id, VulnerabilityType.INSECURE_CONFIGURATION)

    def _map_semgrep_rule_to_type(self, check_id: str) -> VulnerabilityType:
        """Map Semgrep rule to vulnerability type"""
        if 'sql' in check_id.lower():
            return VulnerabilityType.SQL_INJECTION
        elif 'xss' in check_id.lower():
            return VulnerabilityType.XSS
        elif 'command' in check_id.lower():
            return VulnerabilityType.COMMAND_INJECTION
        elif 'crypto' in check_id.lower():
            return VulnerabilityType.WEAK_CRYPTO

        return VulnerabilityType.INSECURE_CONFIGURATION

    def _get_bandit_remediation(self, test_id: str) -> str:
        """Get specific remediation for Bandit test"""
        remediations = {
            'B301': 'Use safe alternatives to pickle (json, yaml.safe_load)',
            'B501': 'Remove hardcoded passwords, use environment variables',
            'B602': 'Avoid shell=True, use subprocess with list arguments',
            'B608': 'Use parameterized queries instead of string formatting'
        }

        return remediations.get(test_id, 'Review Bandit documentation for remediation')

    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get security scanner statistics"""
        return {
            **self.scan_stats,
            'external_tools_available': self.external_tools,
            'patterns_loaded': sum(len(patterns) for patterns in self.vulnerability_patterns.values())
        }
