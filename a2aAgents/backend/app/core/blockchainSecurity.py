"""
import time
Blockchain Security Audit and Protection System
Comprehensive security framework for A2A blockchain integration
"""
import subprocess

import os
import re
import json
import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
# Performance: Consider using asyncio.gather for concurrent operations

from .errorHandling import SecurityError
from .secrets import get_secrets_manager

logger = logging.getLogger(__name__)


class SecuritySeverity(str, Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityVulnerability:
    """Security vulnerability data class"""
    id: str
    severity: SecuritySeverity
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None
    cve_references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    fixed: bool = False


class BlockchainSecurityAuditor:
    """Comprehensive blockchain security auditor"""

    def __init__(self):
        self.vulnerabilities: List[SecurityVulnerability] = []
        self.audit_results: Dict[str, Any] = {}
        self.secrets_manager = get_secrets_manager()

        # Security patterns to detect
        self.security_patterns = {
            "hardcoded_private_keys": [
                r"private_key\s*=\s*['\"][0x][a-fA-F0-9]{64}['\"]",
                r"PRIVATE_KEY\s*=\s*['\"][0x][a-fA-F0-9]{64}['\"]",
                r"privateKey\s*:\s*['\"][0x][a-fA-F0-9]{64}['\"]"
            ],
            "weak_randomness": [
                r"os\.urandom\(\d+\)",
                r"random\.random\(\)",
                r"Math\.random\(\)"
            ],
            "insecure_defaults": [
                r"trust_score\s*=\s*1\.0",
                r"DEFAULT_TRUST\s*=\s*1\.0",
                r"verify\s*=\s*False"
            ],
            "command_injection": [
                r"os\.system\(",
                r"subprocess\.call\([^,]*shell\s*=\s*True",
                r"eval\(",
                r"exec\("
            ],
            "exposed_secrets": [
                r"['\"][0-9a-zA-Z]{32,}['\"]",  # Long strings that might be secrets
                r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{16,}['\"]"
            ]
        }

        # Blockchain-specific patterns
        self.blockchain_patterns = {
            "weak_signature_validation": [
                r"ecrecover\s*\(",
                r"\.recover\s*\(",
                r"signature_bytes\s*="
            ],
            "reentrancy_vulnerabilities": [
                r"\.call\s*\(",
                r"\.send\s*\(",
                r"\.transfer\s*\("
            ],
            "integer_overflow": [
                r"\+\+",
                r"\s*\+\s*",
                r"SafeMath\."
            ]
        }

        logger.info("Blockchain Security Auditor initialized")

    async def run_comprehensive_audit(self, target_paths: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive security audit on blockchain integration

        Args:
            target_paths: List of file/directory paths to audit

        Returns:
            Comprehensive audit report
        """
        logger.info("🔍 Starting comprehensive blockchain security audit")

        audit_start = datetime.utcnow()
        self.vulnerabilities = []

        try:
            # Phase 1: Static code analysis
            await self._static_code_analysis(target_paths)

            # Phase 2: Configuration security audit
            await self._configuration_audit(target_paths)

            # Phase 3: Cryptographic implementation review
            await self._cryptographic_audit(target_paths)

            # Phase 4: Access control validation
            await self._access_control_audit(target_paths)

            # Phase 5: Input validation analysis
            await self._input_validation_audit(target_paths)

            # Compile results
            self.audit_results = self._compile_audit_results(audit_start)

            logger.info(f"✅ Audit completed: {len(self.vulnerabilities)} vulnerabilities found")
            return self.audit_results

        except Exception as e:
            logger.error(f"❌ Audit failed: {e}")
            raise SecurityError(f"Security audit failed: {e}")

    async def _static_code_analysis(self, target_paths: List[str]):
        """Perform static code analysis for security vulnerabilities"""
        logger.info("🔎 Phase 1: Static code analysis")

        for path in target_paths:
            if os.path.isfile(path):
                await self._analyze_file(path)
            elif os.path.isdir(path):
                await self._analyze_directory(path)

    async def _analyze_file(self, file_path: str):
        """Analyze individual file for security issues"""
        try:
            # Skip binary files and common non-code files
            skip_extensions = {'.pyc', '.pyo', '.jpg', '.png', '.gif', '.pdf', '.zip'}
            if any(file_path.endswith(ext) for ext in skip_extensions):
                return

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            # Check all security patterns
            all_patterns = {**self.security_patterns, **self.blockchain_patterns}

            for category, patterns in all_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)

                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                        vulnerability = self._create_vulnerability(
                            category, file_path, line_num, line_content, match.group()
                        )

                        if vulnerability:
                            self.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")

    async def _analyze_directory(self, dir_path: str):
        """Recursively analyze directory"""
        try:
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden directories and common build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'build', 'dist'}]

                for file in files:
                    file_path = os.path.join(root, file)
                    await self._analyze_file(file_path)

        except Exception as e:
            logger.error(f"Error analyzing directory {dir_path}: {e}")

    def _create_vulnerability(self, category: str, file_path: str, line_num: int, line_content: str, match: str) -> Optional[SecurityVulnerability]:
        """Create vulnerability object from detected pattern"""

        vulnerability_configs = {
            "hardcoded_private_keys": {
                "severity": SecuritySeverity.CRITICAL,
                "title": "Hardcoded Private Key Detected",
                "description": "Private keys should never be hardcoded in source code. This poses extreme security risk.",
                "remediation": "Use environment variables or secure key management systems (HSM/KMS)"
            },
            "weak_randomness": {
                "severity": SecuritySeverity.HIGH,
                "title": "Weak Random Number Generation",
                "description": "Using non-cryptographically secure random number generation for security-sensitive operations.",
                "remediation": "Use secrets.token_bytes() or other cryptographically secure random generators"
            },
            "insecure_defaults": {
                "severity": SecuritySeverity.MEDIUM,
                "title": "Insecure Default Configuration",
                "description": "Default configuration values are too permissive and pose security risks.",
                "remediation": "Use secure defaults that require explicit configuration for elevated privileges"
            },
            "command_injection": {
                "severity": SecuritySeverity.HIGH,
                "title": "Potential Command Injection Vulnerability",
                "description": "Direct command execution may allow injection of malicious commands.",
                "remediation": "Use subprocess.run() with array arguments instead of shell=True"
            },
            "exposed_secrets": {
                "severity": SecuritySeverity.MEDIUM,
                "title": "Potential Exposed Secret",
                "description": "Long strings that may contain API keys or other secrets detected.",
                "remediation": "Move secrets to environment variables or secure storage"
            },
            "weak_signature_validation": {
                "severity": SecuritySeverity.HIGH,
                "title": "Weak Signature Validation",
                "description": "Blockchain signature validation may be vulnerable to attacks.",
                "remediation": "Implement proper signature validation with nonce checks and replay protection"
            },
            "reentrancy_vulnerabilities": {
                "severity": SecuritySeverity.CRITICAL,
                "title": "Potential Reentrancy Vulnerability",
                "description": "Smart contract calls may be vulnerable to reentrancy attacks.",
                "remediation": "Use reentrancy guards and follow checks-effects-interactions pattern"
            },
            "integer_overflow": {
                "severity": SecuritySeverity.MEDIUM,
                "title": "Potential Integer Overflow",
                "description": "Integer operations may overflow and cause unexpected behavior.",
                "remediation": "Use SafeMath library or built-in overflow checks"
            }
        }

        config = vulnerability_configs.get(category)
        if not config:
            return None

        # Generate unique vulnerability ID
        vuln_id = hashlib.sha256(f"{category}_{file_path}_{line_num}_{match}".encode()).hexdigest()[:12]

        return SecurityVulnerability(
            id=vuln_id,
            severity=config["severity"],
            title=config["title"],
            description=config["description"],
            file_path=file_path,
            line_number=line_num,
            code_snippet=line_content.strip(),
            remediation=config["remediation"]
        )

    async def _configuration_audit(self, target_paths: List[str]):
        """Audit configuration files for security issues"""
        logger.info("🔧 Phase 2: Configuration security audit")

        # Look for configuration files
        config_files = []
        for path in target_paths:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    config_files.extend([
                        os.path.join(root, f) for f in files
                        if any(f.endswith(ext) for ext in ['.toml', '.json', '.yaml', '.yml', '.env'])
                    ])
            elif any(path.endswith(ext) for ext in ['.toml', '.json', '.yaml', '.yml', '.env']):
                config_files.append(path)

        for config_file in config_files:
            await self._audit_config_file(config_file)

    async def _audit_config_file(self, config_file: str):
        """Audit individual configuration file"""
        try:
            with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Check for exposed secrets in config files
            secret_patterns = [
                r"api[_-]?key\s*[=:]\s*['\"][^'\"]{20,}['\"]",
                r"private[_-]?key\s*[=:]\s*['\"]0x[a-fA-F0-9]{64}['\"]",
                r"secret\s*[=:]\s*['\"][^'\"]{16,}['\"]",
                r"password\s*[=:]\s*['\"][^'\"]{8,}['\"]"
            ]

            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1

                    vulnerability = SecurityVulnerability(
                        id=hashlib.sha256(f"config_secret_{config_file}_{line_num}".encode()).hexdigest()[:12],
                        severity=SecuritySeverity.HIGH,
                        title="Secret Exposed in Configuration File",
                        description="Sensitive information found in configuration file",
                        file_path=config_file,
                        line_number=line_num,
                        remediation="Move secrets to environment variables or secure vault"
                    )

                    self.vulnerabilities.append(vulnerability)

        except Exception as e:
            logger.error(f"Error auditing config file {config_file}: {e}")

    async def _cryptographic_audit(self, target_paths: List[str]):
        """Audit cryptographic implementations"""
        logger.info("🔐 Phase 3: Cryptographic implementation review")

        crypto_issues = [
            (r"md5\(", SecuritySeverity.MEDIUM, "Weak Hash Algorithm (MD5)", "Use SHA-256 or stronger"),
            (r"sha1\(", SecuritySeverity.MEDIUM, "Weak Hash Algorithm (SHA-1)", "Use SHA-256 or stronger"),
            (r"key_size=1024", SecuritySeverity.HIGH, "Weak RSA Key Size", "Use at least 2048-bit keys"),
            (r"DES\.", SecuritySeverity.HIGH, "Weak Encryption (DES)", "Use AES or other strong encryption"),
            (r"ECB", SecuritySeverity.HIGH, "Insecure Block Cipher Mode", "Use CBC, GCM, or other secure modes")
        ]

        for path in target_paths:
            await self._check_crypto_patterns(path, crypto_issues)

    async def _check_crypto_patterns(self, path: str, patterns: List[Tuple]):
        """Check for cryptographic issues in files"""
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                files.extend([os.path.join(root, f) for f in filenames if f.endswith('.py')])

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern, severity, title, remediation in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1

                        vulnerability = SecurityVulnerability(
                            id=hashlib.sha256(f"crypto_{file_path}_{line_num}_{title}".encode()).hexdigest()[:12],
                            severity=severity,
                            title=title,
                            description=f"Weak cryptographic practice detected: {match.group()}",
                            file_path=file_path,
                            line_number=line_num,
                            remediation=remediation
                        )

                        self.vulnerabilities.append(vulnerability)

            except Exception as e:
                logger.error(f"Error checking crypto patterns in {file_path}: {e}")

    async def _access_control_audit(self, target_paths: List[str]):
        """Audit access control implementations"""
        logger.info("🔒 Phase 4: Access control validation")

        # Look for missing authorization checks
        access_patterns = [
            (r"def\s+\w+\(.*request.*\):", "Missing authorization check in endpoint"),
            (r"@app\.route\(", "Route without authentication decorator"),
            (r"trust_score\s*<\s*0\.[1-5]", "Very permissive trust requirements")
        ]

        for path in target_paths:
            await self._check_access_patterns(path, access_patterns)

    async def _check_access_patterns(self, path: str, patterns: List[Tuple]):
        """Check for access control issues"""
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                files.extend([os.path.join(root, f) for f in filenames if f.endswith('.py')])

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern, description in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1

                        vulnerability = SecurityVulnerability(
                            id=hashlib.sha256(f"access_{file_path}_{line_num}".encode()).hexdigest()[:12],
                            severity=SecuritySeverity.MEDIUM,
                            title="Access Control Issue",
                            description=description,
                            file_path=file_path,
                            line_number=line_num,
                            remediation="Implement proper authorization checks"
                        )

                        self.vulnerabilities.append(vulnerability)

            except Exception as e:
                logger.error(f"Error checking access patterns in {file_path}: {e}")

    async def _input_validation_audit(self, target_paths: List[str]):
        """Audit input validation implementations"""
        logger.info("✅ Phase 5: Input validation analysis")

        # Look for missing input validation
        input_patterns = [
            (r"request\.json\[", "Direct access to request data without validation"),
            (r"request\.args\[", "Direct access to query parameters without validation"),
            (r"request\.form\[", "Direct access to form data without validation"),
            (r"json\.loads\(.*request", "JSON parsing without validation")
        ]

        for path in target_paths:
            await self._check_input_patterns(path, input_patterns)

    async def _check_input_patterns(self, path: str, patterns: List[Tuple]):
        """Check for input validation issues"""
        if os.path.isfile(path):
            files = [path]
        else:
            files = []
            for root, dirs, filenames in os.walk(path):
                files.extend([os.path.join(root, f) for f in filenames if f.endswith('.py')])

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for pattern, description in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1

                        vulnerability = SecurityVulnerability(
                            id=hashlib.sha256(f"input_{file_path}_{line_num}".encode()).hexdigest()[:12],
                            severity=SecuritySeverity.MEDIUM,
                            title="Input Validation Issue",
                            description=description,
                            file_path=file_path,
                            line_number=line_num,
                            remediation="Implement proper input validation and sanitization"
                        )

                        self.vulnerabilities.append(vulnerability)

            except Exception as e:
                logger.error(f"Error checking input patterns in {file_path}: {e}")

    def _compile_audit_results(self, audit_start: datetime) -> Dict[str, Any]:
        """Compile comprehensive audit results"""
        audit_end = datetime.utcnow()

        # Count vulnerabilities by severity
        severity_counts = {
            SecuritySeverity.CRITICAL: 0,
            SecuritySeverity.HIGH: 0,
            SecuritySeverity.MEDIUM: 0,
            SecuritySeverity.LOW: 0,
            SecuritySeverity.INFO: 0
        }

        for vuln in self.vulnerabilities:
            severity_counts[vuln.severity] += 1

        # Calculate risk score
        risk_score = (
            severity_counts[SecuritySeverity.CRITICAL] * 10 +
            severity_counts[SecuritySeverity.HIGH] * 7 +
            severity_counts[SecuritySeverity.MEDIUM] * 4 +
            severity_counts[SecuritySeverity.LOW] * 1
        )

        # Determine overall security posture
        if risk_score == 0:
            security_posture = "EXCELLENT"
        elif risk_score <= 10:
            security_posture = "GOOD"
        elif risk_score <= 30:
            security_posture = "MODERATE"
        elif risk_score <= 60:
            security_posture = "POOR"
        else:
            security_posture = "CRITICAL"

        return {
            "audit_metadata": {
                "audit_id": secrets.token_hex(8),
                "started_at": audit_start.isoformat(),
                "completed_at": audit_end.isoformat(),
                "duration_seconds": (audit_end - audit_start).total_seconds(),
                "auditor_version": "1.0.0"
            },
            "summary": {
                "total_vulnerabilities": len(self.vulnerabilities),
                "severity_breakdown": {k.value: v for k, v in severity_counts.items()},
                "risk_score": risk_score,
                "security_posture": security_posture
            },
            "vulnerabilities": [
                {
                    "id": vuln.id,
                    "severity": vuln.severity.value,
                    "title": vuln.title,
                    "description": vuln.description,
                    "file_path": vuln.file_path,
                    "line_number": vuln.line_number,
                    "code_snippet": vuln.code_snippet,
                    "remediation": vuln.remediation,
                    "discovered_at": vuln.discovered_at.isoformat(),
                    "fixed": vuln.fixed
                }
                for vuln in self.vulnerabilities
            ],
            "recommendations": self._generate_recommendations(severity_counts, risk_score)
        }

    def _generate_recommendations(self, severity_counts: Dict, risk_score: int) -> List[str]:
        """Generate security recommendations based on audit results"""
        recommendations = []

        if severity_counts[SecuritySeverity.CRITICAL] > 0:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Fix all critical vulnerabilities before deployment")

        if severity_counts[SecuritySeverity.HIGH] > 0:
            recommendations.append("⚠️ HIGH PRIORITY: Address high-severity vulnerabilities within 24 hours")

        if risk_score > 30:
            recommendations.append("🔒 Implement comprehensive security review process")
            recommendations.append("🔐 Deploy additional security monitoring and alerting")

        if severity_counts[SecuritySeverity.MEDIUM] > 5:
            recommendations.append("📋 Create security remediation backlog for medium-priority issues")

        recommendations.extend([
            "🛡️ Implement automated security scanning in CI/CD pipeline",
            "📚 Provide security training for development team",
            "🔍 Schedule regular penetration testing",
            "📖 Establish secure coding guidelines and review process"
        ])

        return recommendations

    def export_report(self, output_path: str, format: str = "json") -> str:
        """Export audit report to file"""
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(self.audit_results, f, indent=2, default=str)

            logger.info(f"✅ Audit report exported to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ Failed to export report: {e}")
            raise SecurityError(f"Failed to export audit report: {e}")


# Global auditor instance
_security_auditor: Optional[BlockchainSecurityAuditor] = None

def get_security_auditor() -> BlockchainSecurityAuditor:
    """Get global security auditor instance"""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = BlockchainSecurityAuditor()
    return _security_auditor


async def run_blockchain_security_audit(target_paths: List[str]) -> Dict[str, Any]:
    """Run comprehensive blockchain security audit"""
    auditor = get_security_auditor()
    return await auditor.run_comprehensive_audit(target_paths)


# Export main classes and functions
__all__ = [
    'BlockchainSecurityAuditor',
    'SecurityVulnerability',
    'SecuritySeverity',
    'get_security_auditor',
    'run_blockchain_security_audit'
]
