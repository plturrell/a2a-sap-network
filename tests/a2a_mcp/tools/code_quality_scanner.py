#!/usr/bin/env python3
"""
Code Quality Scanner for A2A Project
Comprehensive code quality analysis with database storage and systematic fixing
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import sqlite3

logger = logging.getLogger(__name__)

class IssueType(Enum):
    """Code quality issue types."""
    SYNTAX_ERROR = "syntax_error"
    STYLE_VIOLATION = "style_violation" 
    TYPE_ERROR = "type_error"
    COMPLEXITY = "complexity"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    IMPORT_ERROR = "import_error"
    UNUSED_CODE = "unused_code"

class IssueSeverity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class CodeQualityIssue:
    """Represents a code quality issue."""
    id: str
    file_path: str
    line: int
    column: int
    tool: str
    issue_type: IssueType
    severity: IssueSeverity
    code: str
    message: str
    rule: Optional[str] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ScanResult:
    """Results of a code quality scan."""
    scan_id: str
    directory: str
    tools_used: List[str]
    total_files: int
    issues_found: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    scan_duration: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    issues: List[CodeQualityIssue] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

class CodeQualityDatabase:
    """Database manager for code quality results."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "code_quality.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Scan Results table
                CREATE TABLE IF NOT EXISTS scan_results (
                    id TEXT PRIMARY KEY,
                    directory TEXT NOT NULL,
                    tools_used TEXT NOT NULL,
                    total_files INTEGER NOT NULL,
                    issues_found INTEGER NOT NULL,
                    issues_by_severity TEXT NOT NULL,
                    issues_by_type TEXT NOT NULL,
                    scan_duration REAL NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Code Quality Issues table
                CREATE TABLE IF NOT EXISTS code_quality_issues (
                    id TEXT PRIMARY KEY,
                    scan_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    column_number INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    issue_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    rule TEXT,
                    suggestion TEXT,
                    auto_fixable BOOLEAN DEFAULT FALSE,
                    fixed BOOLEAN DEFAULT FALSE,
                    ignored BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scan_id) REFERENCES scan_results(id)
                );
                
                -- Issue Fix History table
                CREATE TABLE IF NOT EXISTS issue_fixes (
                    id TEXT PRIMARY KEY,
                    issue_id TEXT NOT NULL,
                    fix_type TEXT NOT NULL,
                    fix_description TEXT,
                    fixed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (issue_id) REFERENCES code_quality_issues(id)
                );
                
                -- Create indexes for performance
                CREATE INDEX IF NOT EXISTS idx_issues_scan_id ON code_quality_issues(scan_id);
                CREATE INDEX IF NOT EXISTS idx_issues_file_path ON code_quality_issues(file_path);
                CREATE INDEX IF NOT EXISTS idx_issues_severity ON code_quality_issues(severity);
                CREATE INDEX IF NOT EXISTS idx_issues_type ON code_quality_issues(issue_type);
                CREATE INDEX IF NOT EXISTS idx_issues_tool ON code_quality_issues(tool);
                CREATE INDEX IF NOT EXISTS idx_issues_fixed ON code_quality_issues(fixed);
            """)
        
        logger.info(f"Code quality database initialized at {self.db_path}")
    
    def store_scan_result(self, scan_result: ScanResult):
        """Store scan result and issues."""
        with sqlite3.connect(self.db_path) as conn:
            # Store scan result
            conn.execute("""
                INSERT INTO scan_results 
                (id, directory, tools_used, total_files, issues_found, 
                 issues_by_severity, issues_by_type, scan_duration, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_result.scan_id,
                scan_result.directory,
                json.dumps(scan_result.tools_used),
                scan_result.total_files,
                scan_result.issues_found,
                json.dumps(scan_result.issues_by_severity),
                json.dumps(scan_result.issues_by_type),
                scan_result.scan_duration,
                scan_result.started_at.isoformat(),
                scan_result.completed_at.isoformat() if scan_result.completed_at else None
            ))
            
            # Store issues
            for issue in scan_result.issues:
                conn.execute("""
                    INSERT INTO code_quality_issues
                    (id, scan_id, file_path, line_number, column_number, tool,
                     issue_type, severity, code, message, rule, suggestion, auto_fixable)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    issue.id,
                    scan_result.scan_id,
                    issue.file_path,
                    issue.line,
                    issue.column,
                    issue.tool,
                    issue.issue_type.value,
                    issue.severity.value,
                    issue.code,
                    issue.message,
                    issue.rule,
                    issue.suggestion,
                    issue.auto_fixable
                ))
        
        logger.info(f"Stored scan result {scan_result.scan_id} with {len(scan_result.issues)} issues")
    
    def get_scan_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get scan summary for the last N days."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total scans and issues
            cursor.execute("""
                SELECT COUNT(*), SUM(issues_found), AVG(scan_duration)
                FROM scan_results 
                WHERE started_at > datetime('now', '-{} days')
            """.format(days))
            total_scans, total_issues, avg_duration = cursor.fetchone()
            
            # Get issues by severity
            cursor.execute("""
                SELECT severity, COUNT(*) 
                FROM code_quality_issues cqi
                JOIN scan_results sr ON cqi.scan_id = sr.id
                WHERE sr.started_at > datetime('now', '-{} days')
                GROUP BY severity
            """.format(days))
            issues_by_severity = dict(cursor.fetchall())
            
            # Get issues by type
            cursor.execute("""
                SELECT issue_type, COUNT(*) 
                FROM code_quality_issues cqi
                JOIN scan_results sr ON cqi.scan_id = sr.id
                WHERE sr.started_at > datetime('now', '-{} days')
                GROUP BY issue_type
            """.format(days))
            issues_by_type = dict(cursor.fetchall())
            
            return {
                "total_scans": total_scans or 0,
                "total_issues": total_issues or 0,
                "avg_scan_duration": avg_duration or 0,
                "issues_by_severity": issues_by_severity,
                "issues_by_type": issues_by_type
            }
    
    def get_top_issues(self, limit: int = 50, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get top issues by frequency/severity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT file_path, issue_type, severity, rule, message, COUNT(*) as count
                FROM code_quality_issues
                WHERE fixed = FALSE AND ignored = FALSE
            """
            params = []
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
                
            query += """
                GROUP BY file_path, issue_type, rule, message
                ORDER BY 
                    CASE severity 
                        WHEN 'critical' THEN 1 
                        WHEN 'high' THEN 2 
                        WHEN 'medium' THEN 3 
                        WHEN 'low' THEN 4 
                        ELSE 5 
                    END,
                    count DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

class CodeQualityScanner:
    """Comprehensive code quality scanner."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db = CodeQualityDatabase()
        
        # Tool configurations
        self.python_tools = {
            "pylint": self._run_pylint,
            "flake8": self._run_flake8, 
            "mypy": self._run_mypy,
            "bandit": self._run_bandit,  # Security
            "vulture": self._run_vulture,  # Dead code
        }
        
        self.javascript_tools = {
            "eslint": self._run_eslint,
        }
        
        self.general_tools = {
            "semgrep": self._run_semgrep,  # Multi-language security
        }
    
    async def scan_directory(
        self, 
        directory: Path, 
        tools: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None
    ) -> ScanResult:
        """Scan a directory for code quality issues."""
        
        scan_id = str(uuid.uuid4())
        started_at = datetime.now()
        
        logger.info(f"Starting code quality scan {scan_id} for {directory}")
        
        # Default tools and extensions
        if tools is None:
            tools = ["pylint", "flake8", "eslint"]
        
        if file_extensions is None:
            file_extensions = [".py", ".js", ".ts", ".jsx", ".tsx"]
        
        # Find files to scan
        files_to_scan = self._find_files(directory, file_extensions)
        total_files = len(files_to_scan)
        
        logger.info(f"Found {total_files} files to scan")
        
        all_issues = []
        tools_used = []
        
        # Run Python tools
        for tool_name in tools:
            if tool_name in self.python_tools:
                try:
                    issues = await self.python_tools[tool_name](directory, files_to_scan)
                    all_issues.extend(issues)
                    tools_used.append(tool_name)
                    logger.info(f"{tool_name}: Found {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error running {tool_name}: {e}")
        
        # Run JavaScript tools
        for tool_name in tools:
            if tool_name in self.javascript_tools:
                try:
                    issues = await self.javascript_tools[tool_name](directory, files_to_scan)
                    all_issues.extend(issues)
                    tools_used.append(tool_name)
                    logger.info(f"{tool_name}: Found {len(issues)} issues")
                except Exception as e:
                    logger.error(f"Error running {tool_name}: {e}")
        
        # Calculate statistics
        completed_at = datetime.now()
        scan_duration = (completed_at - started_at).total_seconds()
        
        issues_by_severity = {}
        issues_by_type = {}
        
        for issue in all_issues:
            # Count by severity
            severity = issue.severity.value
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
            
            # Count by type
            issue_type = issue.issue_type.value
            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        
        scan_result = ScanResult(
            scan_id=scan_id,
            directory=str(directory),
            tools_used=tools_used,
            total_files=total_files,
            issues_found=len(all_issues),
            issues_by_severity=issues_by_severity,
            issues_by_type=issues_by_type,
            scan_duration=scan_duration,
            started_at=started_at,
            completed_at=completed_at,
            issues=all_issues
        )
        
        # Store results
        self.db.store_scan_result(scan_result)
        
        logger.info(f"Scan completed: {len(all_issues)} issues found in {scan_duration:.2f}s")
        
        return scan_result
    
    def _find_files(self, directory: Path, extensions: List[str]) -> List[Path]:
        """Find files to scan based on extensions."""
        files = []
        
        for ext in extensions:
            pattern = f"**/*{ext}"
            found_files = list(directory.rglob(pattern))
            
            # Filter out common exclusions
            excluded_patterns = [
                "node_modules",
                "__pycache__",
                ".git", 
                "venv",
                ".venv",
                "build",
                "dist",
                ".pytest_cache",
                "coverage",
                ".coverage"
            ]
            
            filtered_files = []
            for file_path in found_files:
                skip = False
                for excluded in excluded_patterns:
                    if excluded in str(file_path):
                        skip = True
                        break
                if not skip:
                    filtered_files.append(file_path)
            
            files.extend(filtered_files)
        
        return list(set(files))  # Remove duplicates
    
    async def _run_pylint(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run pylint on Python files."""
        python_files = [f for f in files if f.suffix == '.py']
        if not python_files:
            return []
        
        issues = []
        
        try:
            # Run pylint with JSON output
            cmd = [
                "/opt/homebrew/bin/python3.11", "-m", "pylint",
                "--output-format=json",
                "--disable=C0114,C0115,C0116",  # Disable missing docstring warnings initially
                *[str(f) for f in python_files[:10]]  # Limit to first 10 files for testing
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                try:
                    pylint_output = json.loads(stdout.decode())
                    
                    for item in pylint_output:
                        issue = CodeQualityIssue(
                            id=str(uuid.uuid4()),
                            file_path=item.get('path', ''),
                            line=item.get('line', 0),
                            column=item.get('column', 0),
                            tool="pylint",
                            issue_type=self._classify_pylint_issue(item.get('type', '')),
                            severity=self._map_pylint_severity(item.get('type', '')),
                            code=item.get('symbol', ''),
                            message=item.get('message', ''),
                            rule=item.get('message-id', ''),
                            auto_fixable=self._is_auto_fixable_pylint(item.get('symbol', ''))
                        )
                        issues.append(issue)
                        
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pylint JSON output")
            
        except Exception as e:
            logger.error(f"Error running pylint: {e}")
        
        return issues
    
    async def _run_flake8(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run flake8 on Python files."""
        python_files = [f for f in files if f.suffix == '.py']
        if not python_files:
            return []
        
        issues = []
        
        try:
            cmd = [
                "/opt/homebrew/bin/python3.11", "-m", "flake8",
                "--format=json",
                "--max-line-length=100",
                *[str(f) for f in python_files[:10]]  # Limit for testing
            ]
            
            # Try regular format if JSON fails
            cmd_fallback = [
                "/opt/homebrew/bin/python3.11", "-m", "flake8", 
                "--max-line-length=100",
                *[str(f) for f in python_files[:10]]
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd_fallback,
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                # Parse flake8 output (format: filename:line:col: code message)
                for line in stdout.decode().split('\n'):
                    if line.strip():
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            file_path, line_num, col_num, message = parts
                            
                            # Extract error code
                            message_parts = message.strip().split(' ', 1)
                            code = message_parts[0] if message_parts else ''
                            msg = message_parts[1] if len(message_parts) > 1 else message.strip()
                            
                            issue = CodeQualityIssue(
                                id=str(uuid.uuid4()),
                                file_path=file_path,
                                line=int(line_num),
                                column=int(col_num),
                                tool="flake8",
                                issue_type=self._classify_flake8_issue(code),
                                severity=self._map_flake8_severity(code),
                                code=code,
                                message=msg,
                                rule=code,
                                auto_fixable=self._is_auto_fixable_flake8(code)
                            )
                            issues.append(issue)
            
        except Exception as e:
            logger.error(f"Error running flake8: {e}")
        
        return issues
    
    async def _run_mypy(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run mypy on Python files.""" 
        python_files = [f for f in files if f.suffix == '.py']
        if not python_files:
            return []
        
        issues = []
        
        try:
            cmd = [
                "/opt/homebrew/bin/python3.11", "-m", "mypy",
                "--ignore-missing-imports",
                "--no-error-summary",
                *[str(f) for f in python_files[:5]]  # Limit for testing
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                # Parse mypy output
                for line in stdout.decode().split('\n'):
                    if ':' in line and 'error:' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path = parts[0]
                            line_num = parts[1]
                            rest = parts[2]
                            
                            if 'error:' in rest:
                                message = rest.split('error:', 1)[1].strip()
                                
                                try:
                                    line_num = int(line_num)
                                except ValueError:
                                    line_num = 0
                                
                                issue = CodeQualityIssue(
                                    id=str(uuid.uuid4()),
                                    file_path=file_path,
                                    line=line_num,
                                    column=0,
                                    tool="mypy",
                                    issue_type=IssueType.TYPE_ERROR,
                                    severity=IssueSeverity.MEDIUM,
                                    code="type-error",
                                    message=message,
                                    rule="mypy",
                                    auto_fixable=False
                                )
                                issues.append(issue)
            
        except Exception as e:
            logger.error(f"Error running mypy: {e}")
        
        return issues
    
    async def _run_bandit(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run bandit security scanner on Python files."""
        # Placeholder - would implement bandit scanning
        return []
    
    async def _run_vulture(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run vulture dead code detector on Python files."""
        # Placeholder - would implement vulture scanning
        return []
    
    async def _run_eslint(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run ESLint on JavaScript/TypeScript files."""
        js_files = [f for f in files if f.suffix in ['.js', '.ts', '.jsx', '.tsx']]
        if not js_files:
            return []
        
        issues = []
        
        try:
            cmd = [
                "npx", "eslint",
                "--format=json",
                "--no-eslintrc",  # Don't use project config
                "--config", '{"extends": ["eslint:recommended"], "parserOptions": {"ecmaVersion": 2020}}',
                *[str(f) for f in js_files[:5]]  # Limit for testing
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stdout:
                try:
                    eslint_output = json.loads(stdout.decode())
                    
                    for file_result in eslint_output:
                        file_path = file_result.get('filePath', '')
                        
                        for message in file_result.get('messages', []):
                            issue = CodeQualityIssue(
                                id=str(uuid.uuid4()),
                                file_path=file_path,
                                line=message.get('line', 0),
                                column=message.get('column', 0),
                                tool="eslint",
                                issue_type=self._classify_eslint_issue(message.get('ruleId', '')),
                                severity=self._map_eslint_severity(message.get('severity', 1)),
                                code=message.get('ruleId', ''),
                                message=message.get('message', ''),
                                rule=message.get('ruleId', ''),
                                auto_fixable=message.get('fix') is not None
                            )
                            issues.append(issue)
                            
                except json.JSONDecodeError:
                    logger.warning("Failed to parse ESLint JSON output")
        
        except Exception as e:
            logger.error(f"Error running ESLint: {e}")
        
        return issues
    
    async def _run_semgrep(self, directory: Path, files: List[Path]) -> List[CodeQualityIssue]:
        """Run Semgrep security scanner."""
        # Placeholder - would implement semgrep scanning  
        return []
    
    # Helper methods for classification and mapping
    def _classify_pylint_issue(self, pylint_type: str) -> IssueType:
        """Classify pylint issue type."""
        type_map = {
            'convention': IssueType.STYLE_VIOLATION,
            'refactor': IssueType.MAINTAINABILITY,
            'warning': IssueType.MAINTAINABILITY,
            'error': IssueType.SYNTAX_ERROR,
            'fatal': IssueType.SYNTAX_ERROR
        }
        return type_map.get(pylint_type.lower(), IssueType.STYLE_VIOLATION)
    
    def _map_pylint_severity(self, pylint_type: str) -> IssueSeverity:
        """Map pylint type to severity."""
        severity_map = {
            'fatal': IssueSeverity.CRITICAL,
            'error': IssueSeverity.HIGH, 
            'warning': IssueSeverity.MEDIUM,
            'refactor': IssueSeverity.MEDIUM,
            'convention': IssueSeverity.LOW
        }
        return severity_map.get(pylint_type.lower(), IssueSeverity.LOW)
    
    def _classify_flake8_issue(self, code: str) -> IssueType:
        """Classify flake8 issue type."""
        if code.startswith('E'):
            return IssueType.STYLE_VIOLATION
        elif code.startswith('W'):
            return IssueType.STYLE_VIOLATION
        elif code.startswith('F'):
            return IssueType.SYNTAX_ERROR
        elif code.startswith('C'):
            return IssueType.COMPLEXITY
        else:
            return IssueType.STYLE_VIOLATION
    
    def _map_flake8_severity(self, code: str) -> IssueSeverity:
        """Map flake8 code to severity."""
        if code.startswith('F'):  # PyFlakes errors
            return IssueSeverity.HIGH
        elif code.startswith('E9'):  # Runtime errors
            return IssueSeverity.HIGH
        elif code.startswith('W6'):  # Deprecated features
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _classify_eslint_issue(self, rule_id: str) -> IssueType:
        """Classify ESLint issue type."""
        if not rule_id:
            return IssueType.STYLE_VIOLATION
            
        security_rules = ['no-eval', 'no-implied-eval', 'no-new-func']
        if rule_id in security_rules:
            return IssueType.SECURITY
            
        return IssueType.STYLE_VIOLATION
    
    def _map_eslint_severity(self, severity: int) -> IssueSeverity:
        """Map ESLint severity to our severity."""
        if severity == 2:
            return IssueSeverity.HIGH
        elif severity == 1:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _is_auto_fixable_pylint(self, symbol: str) -> bool:
        """Check if pylint issue is auto-fixable."""
        auto_fixable = [
            'trailing-whitespace',
            'missing-final-newline',
            'unused-import',
            'wrong-import-order'
        ]
        return symbol in auto_fixable
    
    def _is_auto_fixable_flake8(self, code: str) -> bool:
        """Check if flake8 issue is auto-fixable."""
        auto_fixable = [
            'W291',  # trailing whitespace
            'W292',  # no newline at end of file
            'W293',  # blank line contains whitespace
            'E302',  # expected 2 blank lines
            'E303',  # too many blank lines
        ]
        return code in auto_fixable